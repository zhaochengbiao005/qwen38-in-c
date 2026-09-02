#!/usr/bin/env python3
"""Ticket-16: generate NumPy reference fixtures for qwen35-c kernels.

Writes binary f32 little-endian files + manifest.txt under tests/fixtures/kern/.
Reference math mirrors HF modeling_qwen3_5.py in float32 (no bf16 rounding),
so manifest tolerances are tight ulp-level values.

Manifest line format (semicolon-separated key=value, whitespace-joined fields):
  <case> ; kernel=<k> ; n=<int> ; [heads=<int> head_dim=<int>] [rotary_dim=<int>
  pos=<int> theta=<f>] [eps=<f>] ; atol=<f> ; rtol=<f> ; crc32=<hex>
  ; in=<file,file,...> ; out=<file>
"""
import os
import struct
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "tests", "fixtures", "kern"))

rng = np.random.default_rng(0xC0FFEE)


def f32(a):
    return np.asarray(a, dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rmsnorm(x, w, eps=1e-6):
    # zero-centered weight, f32 throughout
    inv = 1.0 / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(eps))
    return (x * inv) * (1.0 + w)


def rmsnorm_fixed_sum(x):
    """mirror the C 8-accumulator reduction for bit-exact fixtures"""
    acc = np.zeros(8, dtype=np.float32)
    for i in range(x.shape[0]):
        acc[i % 8] = np.float32(acc[i % 8] + np.float32(x[i] * x[i]))
    s = acc[0]
    for j in range(1, 8):
        s = np.float32(s + acc[j])
    return s


def rmsnorm_bitexact(x, w, eps=1e-6):
    # f32 sequence identical to q35_kern.c
    s = rmsnorm_fixed_sum(x.astype(np.float32))
    mean = np.float32(s / np.float32(x.shape[0]))
    inv = np.float32(1.0 / np.float32(np.sqrt(np.float32(mean + np.float32(eps)))))
    t = x * inv
    return t * (1.0 + w)


def rope(x, rotary_dim, position, theta=1e7):
    # x: [heads, head_dim]; rotate_half on the first rotary_dim dims, f32
    heads, hd = x.shape
    out = x.copy()
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32)
                                / np.float32(rotary_dim)))
    f = np.float32(position) * inv_freq  # [half]
    cos = np.cos(f).astype(np.float32)
    sin = np.sin(f).astype(np.float32)
    x0 = x[:, :half]
    x1 = x[:, half:rotary_dim]
    out[:, :half] = x0 * cos[None, :] - x1 * sin[None, :]
    out[:, half:rotary_dim] = x1 * cos[None, :] + x0 * sin[None, :]
    return out


def softmax(x):
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


cases = []


def emit(name, kernel, inputs, out, atol, rtol, **params):
    files = []
    for i, a in enumerate(inputs):
        fn = f"{name}_in{i}.bin"
        f32(a).tofile(os.path.join(OUT, fn))
        files.append(fn)
    ofn = f"{name}_out.bin"
    ob = f32(out).tobytes()
    with open(os.path.join(OUT, ofn), "wb") as fh:
        fh.write(ob)
    crc = zlib.crc32(ob) & 0xFFFFFFFF
    fields = [name,
              f"kernel={kernel}",
              f"atol={atol}", f"rtol={rtol}",
              f"crc32={crc:08x}",
              f"in={','.join(files)}",
              f"out={ofn}"]
    for k, v in params.items():
        fields.append(f"{k}={v}")
    cases.append(" ".join(fields))


def main():
    os.makedirs(OUT, exist_ok=True)
    EPS = 1e-6

    # 1. rmsnorm, feature dim 5120, zero-centered weights ~ N(0, 0.5)
    x = f32(rng.standard_normal(5120) * 3.0)
    w = f32(rng.standard_normal(5120) * 0.5)
    emit("rmsnorm_5120", "rmsnorm", [x, w], rmsnorm_bitexact(x, w, EPS),
         "2e-6", "2e-6", n=5120, eps=EPS)

    # 2. silu
    x = f32(rng.standard_normal(17408) * 4.0)
    emit("silu_17408", "silu", [x], silu(x), "1e-6", "1e-5", n=17408)

    # 3. swiglu
    g = f32(rng.standard_normal(17408) * 2.0)
    u = f32(rng.standard_normal(17408) * 2.0)
    emit("swiglu_17408", "swiglu", [g, u], silu(g) * u, "1e-6", "1e-5", n=17408)

    # 4. per-head q/k rmsnorm: 24 heads x 256, shared weight vector [256]
    x = f32(rng.standard_normal((24, 256)) * 2.5)
    w = f32(rng.standard_normal(256) * 0.5)
    ref = np.stack([rmsnorm_bitexact(x[h], w, EPS) for h in range(24)])
    emit("qknorm_24x256", "qknorm", [x, w], ref, "2e-6", "2e-6",
         heads=24, head_dim=256, eps=EPS)

    # 5. partial rope: head_dim=256, rotary_dim=64, theta=1e7
    for pos in (7, 2025):
        x = f32(rng.standard_normal((24, 256)) * 2.0)
        emit(f"rope_pos{pos}", "rope", [x], rope(x, 64, pos, 1e7),
             "2e-6", "2e-5", heads=24, head_dim=256, rotary_dim=64,
             pos=pos, theta="1e7")

    # 6. attention output gate: 6144 wide
    a = f32(rng.standard_normal(6144) * 2.0)
    g = f32(rng.standard_normal(6144) * 2.0)
    emit("attn_gate_6144", "attn_gate", [a, g], a * sigmoid(g),
         "1e-6", "1e-5", n=6144)

    # 7. stable softmax
    for n in (1, 128, 1023):
        x = f32(rng.standard_normal(n) * 5.0)
        emit(f"softmax_{n}", "softmax", [x], softmax(x), "1e-6", "1e-5", n=n)

    with open(os.path.join(OUT, "manifest.txt"), "w", newline="\n") as fh:
        fh.write("# ticket-16 kernel fixtures; see gen_kern_fixtures.py\n")
        for line in cases:
            fh.write(line + "\n")
    print(f"wrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()