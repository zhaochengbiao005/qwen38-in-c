#!/usr/bin/env python3
"""Ticket-20: tiny oracle fixture — engine vs numpy f64 reference, three paths.

Builds tests/fixtures/oracle/:
  model.safetensors  mini Qwen3_5 checkpoint (FP8 projections + BF16 scales,
                     BF16 norms/embeds), layer mix linear/full = 6:2 (interval 4)
  config.json        mini config (strict-parse compatible)
  oracle.json        manifest: prompt, reference greedy ids, declared tolerances
  tf_logits_ref.bin  f64 [T, vocab] teacher-forcing logits at every position
                     (T = len(prompt) + gen; row t = logits after feeding seq[t])

Reference runs in float64 over the exact stored (quantized) values: FP8 weights
dequantized via 128x128 block scales, BF16 expanded to f64 — the numerical
ground truth from research/qwen35-numspec.md, independent of the C engine.
"""
import json, os, struct
import numpy as np
import ml_dtypes

DIR = os.environ.get("ORACLE_DIR", "tests/fixtures/oracle")
os.makedirs(DIR, exist_ok=True)

# ---- dims: real-model ratios at tiny scale; FP8 grids carry partial edge
# blocks (e.g. qkv 320x160 -> scale [3,2] with edges 64/32) ----
H, II, VOCAB, NLAY = int(os.environ.get("ORACLE_H", 160)), 192, 256, \
    int(os.environ.get("ORACLE_NLAY", 8))
Hk, Hv, dk, dv = 2, 6, 32, 32          # K=64 V=192 QKV=320, Hv/Hk=3
QH = int(os.environ.get("ORACLE_QH", 6))
KVH = int(os.environ.get("ORACLE_KVH", 1))
HD = int(os.environ.get("ORACLE_HD", 64))
ROT = HD // 4                           # partial_rotary_factor 0.25
THETA = float(os.environ.get("ORACLE_THETA", 1e7))
EPS = 1e-6
ALL_LINEAR = os.environ.get("ORACLE_ALL_LINEAR", "0") == "1"
FULL_ONLY = os.environ.get("ORACLE_FULL_ONLY", "0") == "1"
ATTN_AT = [int(x) for x in os.environ.get("ORACLE_ATTN_AT", "").split(",") if x != ""]
if ATTN_AT:
    LAYER_TYPES = ["full_attention" if i in ATTN_AT else "linear_attention"
                   for i in range(NLAY)]
elif FULL_ONLY:
    LAYER_TYPES = ["full_attention"] * NLAY
elif ALL_LINEAR:
    LAYER_TYPES = ["linear_attention"] * NLAY
else:
    LAYER_TYPES = ((["linear_attention"] * 3 + ["full_attention"]) * 2)[:NLAY]
PROMPT = [5, 42, 7, 99, 13, 201]
GEN = 8
# Declared tolerance for engine(f32) vs this reference(f64). Measured on
# 2026-08-23 build: maxabs 6.3e-6 over all 14 TF positions (pure f32
# rounding); declared with >100x headroom, still tight enough to catch any
# real architecture misread (those land at 1e-1..1e0).
TF_ATOL = float(os.environ.get("ORACLE_ATOL", 1e-3))
TF_RTOL = float(os.environ.get("ORACLE_RTOL", 1e-2))

rng = np.random.default_rng(20260823)

fp8  = lambda a: a.astype(ml_dtypes.float8_e4m3fn)
bf16 = lambda a: a.astype(ml_dtypes.bfloat16)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
silu = lambda x: x * sigmoid(x)
softplus = lambda x: np.logaddexp(0.0, x)

tensors, wd = {}, {}

def put_bf16(name, arr):
    a = np.asarray(arr, np.float32)
    tensors[name] = ("BF16", list(a.shape), bf16(a).view(np.uint16).tobytes())
    wd[name] = bf16(a).astype(np.float64)

def put_f8(name, arr, std_scale=0.1):
    a = np.asarray(arr, np.float32)
    rows, cols = a.shape
    w8 = fp8(a)
    sr, sc = (rows + 127) // 128, (cols + 127) // 128
    scale = bf16(rng.random((sr, sc)).astype(np.float32) * std_scale + 0.05)
    tensors[name] = ("F8_E4M3", [rows, cols], w8.view(np.uint8).tobytes())
    tensors[name + "_scale_inv"] = ("BF16", [sr, sc], scale.view(np.uint16).tobytes())
    d = w8.astype(np.float64)
    sc64 = scale.astype(np.float64)
    out = np.empty((rows, cols), np.float64)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = d[r, c] * sc64[r // 128, c // 128]
    wd[name] = out

P = "model.language_model."
w = lambda sfx: P + sfx

put_bf16(w("embed_tokens.weight"), rng.standard_normal((VOCAB, H)) * 0.05)
put_bf16("lm_head.weight",         rng.standard_normal((VOCAB, H)) * 0.05)
put_bf16(w("norm.weight"),         rng.standard_normal(H) * 0.2)

for i, lt in enumerate(LAYER_TYPES):
    L = w(f"layers.{i}.")
    put_bf16(L + "input_layernorm.weight",          rng.standard_normal(H) * 0.2)
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

# ---- safetensors single shard ----
names = sorted(tensors)
offs, cursor = {}, 0
for n in names:
    dt, shape, data = tensors[n]
    offs[n] = (cursor, cursor + len(data)); cursor += len(data)
header = json.dumps({n: {"dtype": tensors[n][0], "shape": tensors[n][1],
                         "data_offsets": list(offs[n])} for n in names})
header += " " * ((8 - len(header) % 8) % 8)
with open(f"{DIR}/model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(header))); f.write(header.encode())
    for n in names: f.write(tensors[n][2])

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
with open(f"{DIR}/config.json", "w") as f: json.dump(cfg, f, indent=1)

# ---- numpy f64 reference forward (stateful token loop) ----
# per-head normalization: mean over the LAST axis only. np.mean on a 2-D
# q/k silently averages across all heads — the exact bug the 2026-08-23
# attention-internal dump comparison caught (engine was right all along).
def rms_zc(x, g):
    ms = np.mean(x * x, axis=-1, keepdims=True) if np.asarray(x).ndim > 1 \
        else np.mean(x * x)
    return x / np.sqrt(ms + EPS) * (1.0 + g)
rms_plain = lambda x, g: x / np.sqrt(np.mean(x*x) + EPS) * g
l2n = lambda x: x / np.sqrt(np.sum(x*x) + 1e-6)
half = ROT // 2
inv_freq = 1.0 / (THETA ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))

def rope(x, pos):  # rotate_half on first ROT dims of per-head vectors
    out = x.copy()
    f = np.float64(pos) * inv_freq
    cos, sin = np.cos(f), np.sin(f)
    x0, x1 = x[..., :half], x[..., half:ROT]
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:ROT] = x1 * cos + x0 * sin
    return out

S = [np.zeros((Hv, dk, dv)) if lt == "linear_attention" else None for lt in LAYER_TYPES]
conv = [np.zeros((2*Hk*dk + Hv*dv, 4)) if lt == "linear_attention" else None
        for lt in LAYER_TYPES]
# per-layer KV caches — a shared cache would let layer n attend over layer
# n-1's k/v (the exact bug the 2026-08-23 two-attn-layer probe caught)
kvc = [np.zeros((0, KVH, HD)) if lt == "full_attention" else None for lt in LAYER_TYPES]
vvc = [np.zeros((0, KVH, HD)) if lt == "full_attention" else None for lt in LAYER_TYPES]

def step(tok, dumps=None, t=0):
    x = wd[w("embed_tokens.weight")][tok].copy()
    if dumps is not None: dumps.append((t, -1, 0, x.astype(np.float32)))
    for i, lt in enumerate(LAYER_TYPES):
        L = w(f"layers.{i}.")
        xn = rms_zc(x, wd[L + "input_layernorm.weight"])
        if lt == "linear_attention":
            K, V = Hk*dk, Hv*dv
            qkv = wd[L+"linear_attn.in_proj_qkv.weight"] @ xn
            z = wd[L+"linear_attn.in_proj_z.weight"] @ xn
            a = wd[L+"linear_attn.in_proj_a.weight"] @ xn
            b = wd[L+"linear_attn.in_proj_b.weight"] @ xn
            conv[i] = np.roll(conv[i], -1, axis=1); conv[i][:, 3] = qkv
            qkv = silu(np.sum(conv[i] * wd[L+"linear_attn.conv1d.weight"][:, 0, :], axis=1))
            q = qkv[:K].reshape(Hk, dk).copy(); kk = qkv[K:2*K].reshape(Hk, dk).copy()
            v = qkv[2*K:].reshape(Hv, dv)
            for hh in range(Hk):
                q[hh] = l2n(q[hh]) / np.sqrt(dk); kk[hh] = l2n(kk[hh])
            zr = z.reshape(Hv, dv)
            o = np.zeros((Hv, dv))
            for hh in range(Hv):
                g = -np.exp(wd[L+"linear_attn.A_log"][hh]) * \
                    softplus(a[hh] + wd[L+"linear_attn.dt_bias"][hh])
                beta = sigmoid(b[hh]); kh = hh // (Hv // Hk)
                S[i][hh] *= np.exp(g)
                kv = S[i][hh].T @ kk[kh]
                S[i][hh] += np.outer(kk[kh], (v[hh] - kv) * beta)
                oh = S[i][hh].T @ q[kh]
                o[hh] = rms_plain(oh, wd[L+"linear_attn.norm.weight"]) * silu(zr[hh])
            y = wd[L+"linear_attn.out_proj.weight"] @ o.reshape(-1)
        else:
            pos = len(kvc[i])
            qg = (wd[L+"self_attn.q_proj.weight"] @ xn).reshape(QH, 2, HD)
            if dumps is not None:
                dumps.append((t, i, 26, qg.astype(np.float32)))
                dumps.append((t, i, 28, qg[:, 0, :].astype(np.float32)))
            q, gate = qg[:, 0, :].copy(), qg[:, 1, :]
            kk = (wd[L+"self_attn.k_proj.weight"] @ xn).reshape(KVH, HD).copy()
            v = (wd[L+"self_attn.v_proj.weight"] @ xn).reshape(KVH, HD)
            q = rms_zc(q, wd[L+"self_attn.q_norm.weight"])
            kk = rms_zc(kk, wd[L+"self_attn.k_norm.weight"])
            if dumps is not None:
                dumps.append((t, i, 29, q.astype(np.float32)))
            q = rope(q, pos); kk = rope(kk, pos)
            if dumps is not None:
                dumps.append((t, i, 20, q.astype(np.float32)))
                dumps.append((t, i, 21, kk.astype(np.float32)))
            kvc[i] = np.concatenate([kvc[i], kk[None]])
            vvc[i] = np.concatenate([vvc[i], v[None]])
            gsz = QH // KVH
            kr = np.repeat(kvc[i], gsz, axis=1); vr = np.repeat(vvc[i], gsz, axis=1)
            sc = np.einsum("thd,hd->th", kr, q) / np.sqrt(HD)   # [len, QH]
            sc = sc.T                                            # [QH, len]
            e = np.exp(sc - sc.max(axis=-1, keepdims=True))
            p = e / e.sum(axis=-1, keepdims=True)
            if dumps is not None:
                for h in range(QH):
                    dumps.append((t, i, 120 + h, p[h].astype(np.float32)))
            at = np.einsum("ht,thd->hd", p, vr)
            at = at * sigmoid(gate)
            if dumps is not None:
                dumps.append((t, i, 23, at.astype(np.float32)))
            y = wd[L+"self_attn.o_proj.weight"] @ at.reshape(-1)
        if dumps is not None: dumps.append((t, i, 1, y.astype(np.float32)))
        x = x + y
        if dumps is not None: dumps.append((t, i, 2, x.astype(np.float32)))
        xn = rms_zc(x, wd[L + "post_attention_layernorm.weight"])
        x = x + wd[L+"mlp.down_proj.weight"] @ \
            (silu(wd[L+"mlp.gate_proj.weight"] @ xn) * (wd[L+"mlp.up_proj.weight"] @ xn))
        if dumps is not None: dumps.append((t, i, 3, x.astype(np.float32)))
    xn = rms_zc(x, wd[w("norm.weight")])
    logits = wd["lm_head.weight"] @ xn
    if dumps is not None:
        dumps.append((t, -2, 8, xn.astype(np.float32)))
        dumps.append((t, -3, 9, logits.astype(np.float32)))
    return logits

def reset_ref():
    for i, lt in enumerate(LAYER_TYPES):
        if lt == "linear_attention":
            S[i][:] = 0.0; conv[i][:] = 0.0
        else:
            kvc[i] = np.zeros((0, KVH, HD)); vvc[i] = np.zeros((0, KVH, HD))

# teacher-forcing rows for the prompt, then greedy continuation
rows = []
for tok in PROMPT: rows.append(step(tok))
gen_ids = []
for _ in range(GEN):
    nxt = int(np.argmax(rows[-1]))
    gen_ids.append(nxt)
    rows.append(step(nxt))

# per-layer/per-position intermediate dump for the first NDUmp tokens,
# same record format as the engine's Q35_DUMP hook (pos,layer,kind,n,f32[])
if os.environ.get("ORACLE_DUMP"):
    reset_ref()
    dumps = []
    for t, tok in enumerate(PROMPT):
        step(tok, dumps, t)
    with open(os.environ["ORACLE_DUMP"], "wb") as f:
        for pos, layer, kind, vec in dumps:
            vec = np.asarray(vec, np.float32).reshape(-1)
            f.write(struct.pack("4i", pos, layer, kind, len(vec)))
            f.write(vec.tobytes())

with open(f"{DIR}/tf_logits_ref.bin", "wb") as f:
    f.write(np.asarray(rows, np.float64).tobytes())

manifest = {
  "prompt": PROMPT, "gen_ids": gen_ids,
  "tf_atol": TF_ATOL, "tf_rtol": TF_RTOL,
  "vocab": VOCAB, "layers": NLAY, "gen": GEN,
  "eos_ignored": True,
  "notes": "f64 reference over stored quantized values (research/qwen35-numspec.md); "
           "engine f32 must stay within declared tolerance and match ids exactly",
}
with open(f"{DIR}/oracle.json", "w") as f: json.dump(manifest, f, indent=1)
print(f"oracle fixture -> {DIR} | tensors={len(tensors)} seq={len(rows)} gen_ids={gen_ids}")
