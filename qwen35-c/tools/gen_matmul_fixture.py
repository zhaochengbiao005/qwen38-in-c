"""FP8/BF16 matmul fixtures: w, scale, x in; y ref out (float64 ref, tolerance declared)."""
import os, struct, json
import numpy as np
import ml_dtypes

os.makedirs("tests/fixtures/mm", exist_ok=True)

def e4m3_roundtrip(vals):
    return vals.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)

rng = np.random.default_rng(7)

cases = []
def add_case(name, rows, cols, fp8=True):
    if fp8:
        w_f32 = rng.standard_normal((rows, cols)).astype(np.float32) * 0.5
        w8 = w_f32.astype(ml_dtypes.float8_e4m3fn)
        w_bytes = w8.view(np.uint8).tobytes() if hasattr(w8, 'view') else np.asarray(w8).view(np.uint8).tobytes()
        sr = (rows + 127) // 128; sc_ = (cols + 127) // 128
        scale = (rng.random((sr, sc_)).astype(np.float32) * 0.5 + 0.5)
        scale_bf16 = scale.astype(ml_dtypes.bfloat16).astype(np.float32)
        # per-block scale application in f32, then dot in f64 for reference
        wd = w8.astype(np.float32).reshape(rows, cols)
        sw = np.empty((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                sw[i, j] = wd[i, j] * scale_bf16[i // 128, j // 128]
        x = rng.standard_normal(cols).astype(np.float32)
        y_ref = sw.astype(np.float64) @ x.astype(np.float64)
        with open(f"tests/fixtures/mm/{name}.w8", "wb") as f: f.write(w_bytes)
        with open(f"tests/fixtures/mm/{name}.sc", "wb") as f: f.write(scale_bf16.astype(ml_dtypes.bfloat16).view(np.uint16).tobytes())
    else:
        w_f32 = rng.standard_normal((rows, cols)).astype(np.float32) * 0.25
        w_bf = w_f32.astype(ml_dtypes.bfloat16)
        w_bytes = w_bf.view(np.uint16).tobytes()
        x = rng.standard_normal(cols).astype(np.float32)
        y_ref = w_bf.astype(np.float64) @ x.astype(np.float64)
        with open(f"tests/fixtures/mm/{name}.w16", "wb") as f: f.write(w_bytes)
        sr = sc_ = 0
    with open(f"tests/fixtures/mm/{name}.x", "wb") as f: f.write(x.astype(np.float32).tobytes())
    with open(f"tests/fixtures/mm/{name}.yref", "wb") as f: f.write(y_ref.astype(np.float64).tobytes())
    cases.append(dict(name=name, rows=rows, cols=cols, fp8=fp8,
                      sr=sr, sc=sc_, tol=5e-4))

add_case("tiny_fp8", 3, 5, fp8=True)
add_case("edge_fp8", 129, 129, fp8=True)          # 跨 128 块
add_case("exact1_fp8", 128, 128, fp8=True)
add_case("wide_fp8", 16, 5120, fp8=True)          # hidden 宽度
add_case("deep_fp8", 5120, 16, fp8=True)
add_case("qkvlike_fp8", 10240, 5120, fp8=True)    # in_proj_qkv 形状
add_case("mid_bf16", 256, 128, fp8=False)
add_case("deep_bf16", 2048, 5120, fp8=False)

with open("tests/fixtures/mm/manifest.json", "w") as f:
    json.dump(cases, f, indent=1)
print("mm fixtures:", len(cases))
