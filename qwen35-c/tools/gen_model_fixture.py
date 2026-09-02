#!/usr/bin/env python3
"""Ticket-19: whole-model tiny fixture (embed -> mixed layers -> norm -> lm_head).

Builds tests/fixtures/model/:
  model.safetensors   all weights (FP8 projections + BF16 scales/norms/embeds)
  config.json         mini Qwen3_5 config (layer_types drives DN/attn mix)
  tokens.json         input token ids
  logits_ref.bin      f64 [vocab] logits of last prefill token (numpy reference)

Numpy reference runs the exact stored (quantized) values in f64:
FP8 weights dequantized via block scales; BF16 norms/embeds expanded to f32.
"""
import json, os, struct
import numpy as np
import ml_dtypes

DIR = "tests/fixtures/model"
os.makedirs(DIR, exist_ok=True)

fp8  = lambda a: a.astype(ml_dtypes.float8_e4m3fn)
bf16 = lambda a: a.astype(ml_dtypes.bfloat16)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def silu(x): return x * sigmoid(x)
def softplus(x): return np.logaddexp(0.0, x)

# ---- dims (mini) ----
H, II, VOCAB, NLAY = 64, 128, 256, 3
Hk, Hv, dk, dv = 2, 6, 16, 16          # linear: K=32 V=96 QKV=160
QH, KVH, HD = 2, 1, 32                  # attn
ROT = 8                                  # partial_rotary_factor 0.25 * 32
THETA, EPS = 1e7, 1e-6
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention"]
TOKENS = [5, 42, 7, 99]

rng = np.random.default_rng(20250823)

def cvt_bytes(a, kind):
    if kind == "f8":  return fp8(a).view(np.uint8).tobytes()
    if kind == "bf16": return bf16(a).view(np.uint16).tobytes()
    raise ValueError(kind)

def dequant_f8(w8, sc, rows, cols):
    d = w8.astype(np.float64)
    out = np.empty((rows, cols), np.float64)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = d[r, c] * float(sc[r // 128, c // 128])
    return out

tensors = {}     # name -> (dtype_str, shape, bytes)
wd = {}          # name -> dequantized f64 array (reference weights)

def put_bf16(name, arr):
    a = np.asarray(arr, np.float32)
    tensors[name] = ("BF16", list(a.shape), cvt_bytes(a, "bf16"))
    wd[name] = bf16(a).astype(np.float64)

def put_f8(name, arr, std_scale=0.1):
    a = np.asarray(arr, np.float32)
    rows, cols = a.shape
    w8 = fp8(a)
    sr, sc_ = (rows + 127) // 128, (cols + 127) // 128
    scale = bf16(rng.random((sr, sc_)).astype(np.float32) * std_scale + 0.05)
    tensors[name] = ("F8_E4M3", [rows, cols], w8.view(np.uint8).tobytes())
    tensors[name + "_scale_inv"] = ("BF16", [sr, sc_], scale.view(np.uint16).tobytes())
    wd[name] = dequant_f8(w8.astype(np.float32), scale.astype(np.float64), rows, cols)

P = "model.language_model."
w = lambda suffix: P + suffix

# ---- synthesize ----
put_bf16(w("embed_tokens.weight"), rng.standard_normal((VOCAB, H)) * 0.05)
put_bf16("lm_head.weight",         rng.standard_normal((VOCAB, H)) * 0.05)
put_bf16(w("norm.weight"),         rng.standard_normal(H) * 0.2)

for i, lt in enumerate(LAYER_TYPES):
    L = w(f"layers.{i}.")
    put_bf16(L + "input_layernorm.weight",         rng.standard_normal(H) * 0.2)
    put_bf16(L + "post_attention_layernorm.weight", rng.standard_normal(H) * 0.2)
    if lt == "linear_attention":
        put_f8(L + "linear_attn.in_proj_qkv.weight", rng.standard_normal((2*Hk*dk + Hv*dv, H)) * 0.5)
        put_f8(L + "linear_attn.in_proj_z.weight",   rng.standard_normal((Hv*dv, H)) * 0.5)
        put_f8(L + "linear_attn.in_proj_a.weight",   rng.standard_normal((Hv, H)) * 0.5)
        put_f8(L + "linear_attn.in_proj_b.weight",   rng.standard_normal((Hv, H)) * 0.5)
        put_f8(L + "linear_attn.out_proj.weight",    rng.standard_normal((H, Hv*dv)) * 0.5)
        put_bf16(L + "linear_attn.conv1d.weight", rng.standard_normal((2*Hk*dk + Hv*dv, 1, 4)) * 0.3)
        put_bf16(L + "linear_attn.norm.weight",   1.0 + rng.standard_normal(dv) * 0.2)
        put_bf16(L + "linear_attn.A_log",  np.log(np.exp(rng.standard_normal(Hv) * 0.5 + 1.2)))
        put_bf16(L + "linear_attn.dt_bias", rng.standard_normal(Hv) * 0.5)
    else:
        put_f8(L + "self_attn.q_proj.weight", rng.standard_normal((QH*2*HD, H)) * 0.5)
        put_f8(L + "self_attn.k_proj.weight", rng.standard_normal((KVH*HD, H)) * 0.5)
        put_f8(L + "self_attn.v_proj.weight", rng.standard_normal((KVH*HD, H)) * 0.5)
        put_f8(L + "self_attn.o_proj.weight", rng.standard_normal((H, QH*HD)) * 0.5)
        put_bf16(L + "self_attn.q_norm.weight", rng.standard_normal(HD) * 0.3)
        put_bf16(L + "self_attn.k_norm.weight", rng.standard_normal(HD) * 0.3)
    put_f8(L + "mlp.gate_proj.weight", rng.standard_normal((II, H)) * 0.5)
    put_f8(L + "mlp.up_proj.weight",   rng.standard_normal((II, H)) * 0.5)
    put_f8(L + "mlp.down_proj.weight", rng.standard_normal((H, II)) * 0.5)

# ---- write safetensors (single shard) ----
names = sorted(tensors)
offs, cursor = {}, 0
for n in names:
    dt, shape, data = tensors[n]
    offs[n] = (cursor, cursor + len(data)); cursor += len(data)
header = json.dumps({n: {"dtype": tensors[n][0], "shape": tensors[n][1],
                         "data_offsets": list(offs[n])} for n in names})
pad = (8 - (len(header) % 8)) % 8
header += " " * pad
with open(f"{DIR}/model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(header)))
    f.write(header.encode())
    for n in names: f.write(tensors[n][2])

# ---- config ----
cfg = {
  "model_type": "qwen3_5_text",
  "num_hidden_layers": NLAY, "hidden_size": H, "intermediate_size": II,
  "num_attention_heads": QH, "num_key_value_heads": KVH, "head_dim": HD,
  "vocab_size": VOCAB,
  "linear_num_key_heads": Hk, "linear_key_head_dim": dk,
  "linear_num_value_heads": Hv, "linear_value_head_dim": dv,
  "linear_conv_kernel_dim": 4,
  "bos_token_id": 1, "eos_token_id": 2,
  "max_position_embeddings": 256,
  "rope_theta": THETA, "partial_rotary_factor": ROT / HD,
  "rms_norm_eps": EPS, "layer_types": LAYER_TYPES,
  "tie_word_embeddings": False,
}
with open(f"{DIR}/config.json", "w") as f:
    json.dump(cfg, f, indent=1)

# ---- numpy reference forward (f64 over exact stored weights) ----
# per-head normalization: mean over the LAST axis only (np.mean on a 2-D
# q/k silently averages across heads; fixed 2026-08-23, same fix as
# gen_oracle_fixture.py)
def rmsnorm_zc(x, wgt):
    ms = np.mean(x * x, axis=-1, keepdims=True) if np.asarray(x).ndim > 1 \
        else np.mean(x * x)
    return x / np.sqrt(ms + EPS) * (1.0 + wgt)
def rmsnorm_plain(x, wgt):
    return x / np.sqrt(np.mean(x * x) + EPS) * wgt
def l2n(x):
    return x / np.sqrt(np.sum(x * x) + 1e-6)

def rope_ref(x, positions):
    out = x.copy()
    half = ROT // 2
    inv = 1.0 / (THETA ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))
    f = np.asarray(positions, np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(f), np.sin(f)
    x0, x1 = x[..., :half], x[..., half:ROT]
    out[..., :half] = x0 * cos[:, None, :] - x1 * sin[:, None, :]
    out[..., half:ROT] = x1 * cos[:, None, :] + x0 * sin[:, None, :]
    return out


_dumps = []
def _dump(pos, layer, kind, vec):
    _dumps.append((pos, layer, kind, np.asarray(vec, np.float32).copy()))

L0, L1, L2 = (w(f"layers.{i}.") for i in range(NLAY))
S = [np.zeros((Hv, dk, dv)), None, np.zeros((Hv, dk, dv))]
conv = [np.zeros((2*Hk*dk + Hv*dv, 4)), None, np.zeros((2*Hk*dk + Hv*dv, 4))]
kvc = np.zeros((0, KVH, HD)); vvc = np.zeros((0, KVH, HD))

T = len(TOKENS)
logits = None
for t in range(T):
    tok = TOKENS[t]
    x = wd[w("embed_tokens.weight")][tok].copy()
    _dump(t, -1, 0, x)
    for i, lt in enumerate(LAYER_TYPES):
        L = w(f"layers.{i}.")
        xn = rmsnorm_zc(x, wd[L + "input_layernorm.weight"])
        if lt == "linear_attention":
            K, V = Hk*dk, Hv*dv
            qkv = wd[L+"linear_attn.in_proj_qkv.weight"] @ xn
            z   = wd[L+"linear_attn.in_proj_z.weight"] @ xn
            a   = wd[L+"linear_attn.in_proj_a.weight"] @ xn
            b   = wd[L+"linear_attn.in_proj_b.weight"] @ xn
            conv[i] = np.roll(conv[i], -1, axis=1); conv[i][:, 3] = qkv
            qkv = silu(np.sum(conv[i] * wd[L+"linear_attn.conv1d.weight"][:, 0, :], axis=1))
            q = qkv[:K].reshape(Hk, dk); kk = qkv[K:2*K].reshape(Hk, dk)
            v = qkv[2*K:].reshape(Hv, dv)
            for hh in range(Hk):
                q[hh] = l2n(q[hh]) / np.sqrt(dk); kk[hh] = l2n(kk[hh])
            o = np.zeros((Hv, dv))
            for hh in range(Hv):
                g = -np.exp(wd[L+"linear_attn.A_log"][hh]) * softplus(a[hh] + wd[L+"linear_attn.dt_bias"][hh])
                beta = sigmoid(b[hh]); kh = hh // (Hv // Hk)
                S[i][hh] *= np.exp(g)
                kv = S[i][hh].T @ kk[kh]
                S[i][hh] += np.outer(kk[kh], (v[hh] - kv) * beta)
                o[hh] = S[i][hh].T @ q[kh]
                o[hh] = rmsnorm_plain(o[hh], wd[L+"linear_attn.norm.weight"]) * \
                        silu(z.reshape(Hv, dv)[hh])
            y = wd[L+"linear_attn.out_proj.weight"] @ o.reshape(-1)
        else:
            qg = (wd[L+"self_attn.q_proj.weight"] @ xn).reshape(QH, 2, HD)
            q, gate = qg[:, 0, :], qg[:, 1, :]
            kk = (wd[L+"self_attn.k_proj.weight"] @ xn).reshape(KVH, HD)
            v  = (wd[L+"self_attn.v_proj.weight"] @ xn).reshape(KVH, HD)
            q = rmsnorm_zc(q, wd[L+"self_attn.q_norm.weight"])
            kk = rmsnorm_zc(kk, wd[L+"self_attn.k_norm.weight"])
            q = rope_ref(q[None], [t])[0]; kk = rope_ref(kk[None], [t])[0]
            kvc = np.concatenate([kvc, kk[None]]); vvc = np.concatenate([vvc, v[None]])
            gsz = QH // KVH
            kr, vr = np.repeat(kvc, gsz, axis=1), np.repeat(vvc, gsz, axis=1)
            sc = np.einsum("thd,hd->th", kr, q) / np.sqrt(HD)  # [len, QH]
            sc = sc.T  # [QH, len]
            e = np.exp(sc - sc.max(axis=-1, keepdims=True))
            p = e / e.sum(axis=-1, keepdims=True)
            at = (p @ vr.swapaxes(0, 1).reshape(QH, len(kvr) if False else len(kvc), HD)) if False else np.einsum("ht,thd->hd", p, vr)
            at = at * sigmoid(gate)
            y = wd[L+"self_attn.o_proj.weight"] @ at.reshape(-1)
        _dump(t, i, 1, y)
        x = x + y
        xn = rmsnorm_zc(x, wd[L + "post_attention_layernorm.weight"])
        _dump(t, i, 2, x)
        g8 = wd[L+"mlp.gate_proj.weight"] @ xn; u8 = wd[L+"mlp.up_proj.weight"] @ xn
        x = x + wd[L+"mlp.down_proj.weight"] @ (silu(g8) * u8)
        _dump(t, i, 3, x)
    xn = rmsnorm_zc(x, wd[w("norm.weight")])
    logits = wd["lm_head.weight"] @ xn

import struct as _st
with open(f"{DIR}/dump_ref.bin", "wb") as _f:
    for _p, _l, _k, _v in _dumps:
        _f.write(_st.pack("4i", _p, _l, _k, len(_v)))
        _f.write(_v.tobytes())

with open(f"{DIR}/logits_ref.bin", "wb") as f:
    f.write(np.asarray(logits, np.float64).tobytes())
with open(f"{DIR}/tokens.json", "w") as f:
    json.dump({"tokens": TOKENS, "atol": 0.03, "rtol": 0.08}, f)
print("mini model fixture ->", DIR, "| tensors:", len(tensors))
