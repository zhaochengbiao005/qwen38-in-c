"""Ticket-17: Gated DeltaNet fixtures.

Numpy reference per .wayfinder/research/qwen35-numspec.md sec 1, small dims
(Hv=6, Hk=2, dk=dv=16, H=64) with identical structural semantics:
group broadcast 3, conv kernel 4, f32 discretization + delta recurrence.

Files per case (tests/fixtures/deltanet/):
  {name}.x          f32  [L, H]        input
  {name}.w_qkv.w8/.sc  f8/u16 [2K+V, H] FP8 weights + bf16 128x128 block scales
  {name}.w_z.w8/.sc    f8/u16 [V, H]
  {name}.w_a.w8/.sc    f8/u16 [Hv, H]
  {name}.w_b.w8/.sc    f8/u16 [Hv, H]
  {name}.w_out.w8/.sc  f8/u16 [H, V]
  {name}.conv       f32  [2K+V, 4]
  {name}.normw      f32  [dv]          RMSNormGated weight (plain)
  {name}.alog       f32  [Hv]
  {name}.dtbias     f32  [Hv]
  {name}.s0         f32  [Hv, dk, dv]  initial state (usually zero)
  {name}.yref       f64  [L, H]        reference output
  {name}.sref       f64  [Hv, dk, dv]  reference final state
"""
import os, json
import numpy as np
import ml_dtypes

DIR = "tests/fixtures/deltanet"
os.makedirs(DIR, exist_ok=True)

fp8 = lambda a: a.astype(ml_dtypes.float8_e4m3fn)
bf16 = lambda a: a.astype(ml_dtypes.bfloat16)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def silu(x): return x * sigmoid(x)
def softplus(x): return np.logaddexp(0.0, x)

def dequant(w8, sc, rows, cols):
    d = w8.astype(np.float64)
    out = np.empty((rows, cols), np.float64)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = d[r, c] * float(sc[r // 128, c // 128])
    return out

def l2norm_head(x, eps):
    return x / np.sqrt(np.sum(x * x) + eps)

def make_weights(rng, rows, cols, std):
    wf = rng.standard_normal((rows, cols)).astype(np.float32) * std
    w8 = fp8(wf)
    sr, sc_ = (rows + 127) // 128, (cols + 127) // 128
    scale = bf16(rng.random((sr, sc_)).astype(np.float32) * 0.2 + 0.05)
    wd = dequant(w8.astype(np.float32), scale.astype(np.float64), rows, cols)
    return w8, scale, wd

def reference(Wqkv, Wz, Wa, Wb, Wout, conv_w, normw, alog, dtbias,
              x, S0, H, Hk, Hv, dk, dv):
    K, V = Hk * dk, Hv * dv
    group = Hv // Hk
    L = x.shape[0]
    S = S0.astype(np.float64).copy()
    conv_state = np.zeros((2 * K + V, 4), np.float64)
    ys = []
    for t in range(L):
        xt = x[t].astype(np.float64)
        qkv = Wqkv @ xt
        z = Wz @ xt
        a = Wa @ xt
        b = Wb @ xt
        conv_state = np.roll(conv_state, -1, axis=1)
        conv_state[:, 3] = qkv
        qkv = silu(np.sum(conv_state * conv_w.astype(np.float64), axis=1))
        q = qkv[:K].reshape(Hk, dk)
        k = qkv[K:2 * K].reshape(Hk, dk)
        v = qkv[2 * K:].reshape(Hv, dv)
        for hh in range(Hk):
            q[hh] = l2norm_head(q[hh], 1e-6) / np.sqrt(dk)
            k[hh] = l2norm_head(k[hh], 1e-6)
        o = np.zeros((Hv, dv), np.float64)
        for hh in range(Hv):
            g = -np.exp(alog[hh]) * softplus(a[hh] + dtbias[hh])
            alpha = np.exp(g)
            beta = sigmoid(b[hh])
            kh = hh // group
            S[hh] *= alpha
            kv = S[hh].T @ k[kh]
            S[hh] += np.outer(k[kh], (v[hh] - kv) * beta)
            o[hh] = S[hh].T @ q[kh]
            o[hh] = (o[hh] / np.sqrt(np.mean(o[hh] ** 2) + 1e-6)
                     * normw.astype(np.float64)
                     * silu(z.reshape(Hv, dv)[hh]))
        ys.append(Wout @ o.reshape(-1))
    return np.stack(ys), S

def add_case(rng, name, L, H, Hk, Hv, dk, dv):
    K, V = Hk * dk, Hv * dv
    w = {}
    wd = {}
    w["qkv"], w["qkv_sc"], wd["qkv"] = make_weights(rng, 2 * K + V, H, 0.05)
    w["z"],  w["z_sc"],  wd["z"]  = make_weights(rng, V, H, 0.05)
    w["a"],  w["a_sc"],  wd["a"]  = make_weights(rng, Hv, H, 0.05)
    w["b"],  w["b_sc"],  wd["b"]  = make_weights(rng, Hv, H, 0.05)
    w["out"], w["out_sc"], wd["out"] = make_weights(rng, H, V, 0.05)
    conv_w = rng.standard_normal((2 * K + V, 4)).astype(np.float32) * 0.3
    normw = (1.0 + rng.standard_normal(dv).astype(np.float32) * 0.2)
    alog = np.log(np.exp(rng.standard_normal(Hv).astype(np.float32) * 0.5 + 1.2))
    dtbias = rng.standard_normal(Hv).astype(np.float32) * 0.5

    x = rng.standard_normal((L, H)).astype(np.float32) * 0.5
    S0 = np.zeros((Hv, dk, dv), np.float32)
    yref, sref = reference(wd["qkv"], wd["z"], wd["a"], wd["b"], wd["out"],
                           conv_w, normw, alog, dtbias, x, S0,
                           H, Hk, Hv, dk, dv)

    def put(suffix, arr):
        with open(f"{DIR}/{name}.{suffix}", "wb") as f:
            f.write(arr.tobytes())

    put("x", x)
    for tag in ("qkv", "z", "a", "b", "out"):
        put(f"w_{tag}.w8", w[tag].view(np.uint8))
        put(f"w_{tag}.sc", w[tag + "_sc"].view(np.uint16))
    put("conv", conv_w)
    put("normw", normw)
    put("alog", alog)
    put("dtbias", dtbias)
    put("s0", S0)
    put("yref", yref.astype(np.float64))
    put("sref", sref.astype(np.float64))
    return dict(name=name, L=L, H=H, Hk=Hk, Hv=Hv, dk=dk, dv=dv,
                atol=1e-3, rtol=1e-3)

rng = np.random.default_rng(17)
cases = [
    add_case(rng, "decode1",  1, 64, 2, 6, 16, 16),
    add_case(rng, "prefill5", 5, 64, 2, 6, 16, 16),
]
with open(f"{DIR}/manifest.json", "w") as f:
    json.dump(cases, f, indent=1)
print("deltanet fixtures:", len(cases), "->", DIR)
