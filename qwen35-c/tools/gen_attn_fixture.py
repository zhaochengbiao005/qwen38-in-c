#!/usr/bin/env python3
"""Ticket-18: NumPy reference fixture for the GQA full-attention layer.

Small synthetic config, same structural semantics as the real layer
(numspec section 2): q_heads=6, kv_heads=2, head_dim=64, rotary_dim=16
(full model: 24/4/256/64). All projections are FP8 e4m3 weights with BF16
128x128 block scales (q35_mm_fp8 layout, dequantized element-wise and used
in float32). Reference computed in float64 over the exact f32/fp8 values so
tolerances can be tight.

Writes tests/fixtures/attn/:
  manifest.json           config + file names + tolerances
  wq.w8 wq.sc ... wo.w8 wo.sc   FP8 weights + BF16 scales
  q_nw.bin k_nw.bin       zero-centered rmsnorm weights (f32)
  x.bin                   input tokens, f32 [nt, hidden]
  yref.bin                reference output, f64 [nt, hidden]

Run from repo root: python tools/gen_attn_fixture.py
"""
import json
import os

import ml_dtypes
import numpy as np

OUT = "tests/fixtures/attn"

CFG = dict(hidden=96, q_heads=6, kv_heads=2, head_dim=64,
           rotary_dim=16, theta=1e7, eps=1e-6, nt=5)

rng = np.random.default_rng(0xA77E)


def e4m3(a):
    return a.astype(ml_dtypes.float8_e4m3fn)


def bf16(a):
    return a.astype(ml_dtypes.bfloat16)


def make_fp8(rows, cols, sigma=0.5):
    """synthesize fp8 weight + bf16 128x128 block scales; return (bytes, sc_u16, sw_f32)"""
    w_f32 = rng.standard_normal((rows, cols), dtype=np.float32) * np.float32(sigma)
    w8 = e4m3(w_f32)
    sr, sc = (rows + 127) // 128, (cols + 127) // 128
    scale = rng.random((sr, sc), dtype=np.float32) * np.float32(0.1) + np.float32(0.05)
    sc_bf = bf16(scale)
    wd = w8.astype(np.float32).reshape(rows, cols)
    sc_f32 = sc_bf.astype(np.float32)
    sw = wd * sc_f32.repeat(128, 0)[:rows].repeat(128, 1)[:, :cols]
    return (w8.view(np.uint8).tobytes(),
            sc_bf.view(np.uint16).tobytes(), sw)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rmsnorm_zc(x, w, eps):
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * inv) * (1.0 + w)


def rope(x, rotary, theta):
    """x: [nt, heads, hd]; rotate_half on first `rotary` dims, positions 0..nt-1"""
    nt, heads, hd = x.shape
    out = x.copy()
    half = rotary // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary, 2, dtype=np.float64) / rotary))
    pos = np.arange(nt, dtype=np.float64)
    f = pos[:, None] * inv_freq[None, :]          # [nt, half]
    cos, sin = np.cos(f), np.sin(f)
    x0 = x[..., :half]
    x1 = x[..., half:rotary]
    out[..., :half] = x0 * cos[:, None, :] - x1 * sin[:, None, :]
    out[..., half:rotary] = x1 * cos[:, None, :] + x0 * sin[:, None, :]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    H = CFG["hidden"]
    qh, kh, hd = CFG["q_heads"], CFG["kv_heads"], CFG["head_dim"]
    nt = CFG["nt"]

    wq_b, wq_s, wq = make_fp8(qh * 2 * hd, H, 0.08)
    wk_b, wk_s, wk = make_fp8(kh * hd, H, 0.08)
    wv_b, wv_s, wv = make_fp8(kh * hd, H, 0.08)
    wo_b, wo_s, wo = make_fp8(H, qh * hd, 0.08)

    q_nw = (rng.standard_normal(hd, dtype=np.float32) * np.float32(0.3))
    k_nw = (rng.standard_normal(hd, dtype=np.float32) * np.float32(0.3))
    x = rng.standard_normal((nt, H), dtype=np.float32)

    # ---- reference, float64 over the exact stored values ----
    xw = x.astype(np.float64)
    wq64, wk64 = wq.astype(np.float64), wk.astype(np.float64)
    wv64, wo64 = wv.astype(np.float64), wo.astype(np.float64)

    # q_proj out is head-major [qh][2*hd]; per head split into [q|gate]

    qg = (xw @ wq64.T).reshape(nt, qh, 2, hd)
    q = qg[:, :, 0, :]
    gate = qg[:, :, 1, :]

    k = (xw @ wk64.T).reshape(nt, kh, hd)
    v = (xw @ wv64.T).reshape(nt, kh, hd)

    q = rmsnorm_zc(q, q_nw.astype(np.float64), CFG["eps"])
    k = rmsnorm_zc(k, k_nw.astype(np.float64), CFG["eps"])
    q = rope(q, CFG["rotary_dim"], CFG["theta"])
    k = rope(k, CFG["rotary_dim"], CFG["theta"])

    gsz = qh // kh
    k_rep = np.repeat(k, gsz, axis=1)          # GQA broadcast 2 -> 6
    v_rep = np.repeat(v, gsz, axis=1)

    scale = 1.0 / np.sqrt(hd)
    scores = np.einsum("qhd,thd->qht", q, k_rep) * scale
    # causal mask: query i sees keys <= i
    i_idx = np.arange(nt)[:, None]
    j_idx = np.arange(nt)[None, :]
    mask = j_idx <= i_idx
    scores = np.where(mask[:, None, :], scores, -np.inf)
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=-1, keepdims=True)   # softmax in f32-equivalent (here f64)
    attn = np.einsum("qht,thd->qhd", p, v_rep)  # [nt, qh, hd]

    attn = attn * sigmoid(gate)                 # sigmoid output gate
    y = attn.reshape(nt, qh * hd) @ wo64.T

    files = {
        "wq.w8": wq_b, "wq.sc": wq_s, "wk.w8": wk_b, "wk.sc": wk_s,
        "wv.w8": wv_b, "wv.sc": wv_s, "wo.w8": wo_b, "wo.sc": wo_s,
        "q_nw.bin": q_nw.astype(np.float32).tobytes(),
        "k_nw.bin": k_nw.astype(np.float32).tobytes(),
        "x.bin": x.astype(np.float32).tobytes(),
        "yref.bin": y.astype(np.float64).tobytes(),
    }
    for fn, data in files.items():
        with open(os.path.join(OUT, fn), "wb") as fh:
            fh.write(data)

    manifest = dict(CFG, tol_atol="1e-4", tol_rtol="1e-3",
                    files=sorted(files), format={
                        "w8": "fp8 e4m3fn row-major [rows, cols]",
                        "sc": "bf16 block scales [ceil(rows/128), ceil(cols/128)]",
                        "q_nw/k_nw": "f32 zero-centered rmsnorm weights [head_dim]",
                        "x": "f32 [nt, hidden]",
                        "yref": "f64 [nt, hidden]",
                    },
                    weight_shapes={"wq": [qh * 2 * hd, H], "wk": [kh * hd, H],
                                   "wv": [kh * hd, H], "wo": [H, qh * hd]})
    with open(os.path.join(OUT, "manifest.json"), "w", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
    print("attn fixtures ->", OUT, "| nt=%d hidden=%d q=%d kv=%d hd=%d" %
          (nt, H, qh, kh, hd))


if __name__ == "__main__":
    main()
