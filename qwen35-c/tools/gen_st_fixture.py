"""Generate a synthetic safetensors shard + a mini config.json for tests."""
import json, struct, os
import numpy as np  # if unavailable, fall back to struct packing

os.makedirs("tests/fixtures/st", exist_ok=True)

def tensor_bytes(dtype, shape, seed):
    n = int(np.prod(shape)) if shape else 1
    rng = np.random.default_rng(seed)
    if dtype == "BF16":
        f = rng.standard_normal(n, dtype=np.float32)
        # ml_dtypes-free bf16: view f32 bits >> 16
        u = f.view(np.uint32)
        return ((u >> 16) & 0xFFFF).astype(np.uint16).tobytes()
    if dtype == "F8_E4M3":
        f = np.clip(rng.standard_normal(n), -3.0, 3.0)
        # encode via numpy: use ml_dtypes if present else integer approximation
        try:
            import ml_dtypes
            return f.astype(ml_dtypes.float8_e4m3fn).view(np.uint8).tobytes()
        except ImportError:
            return (np.clip(((f.astype(np.float32).view(np.uint32) >> 16)), 0, 255)).astype(np.uint8).tobytes()
    if dtype == "F32":
        return rng.standard_normal(n, dtype=np.float32).tobytes()
    raise SystemExit("bad dtype")

entries = [
    ("model.language_model.layers.0.mlp.gate_proj.weight", "F8_E4M3", [256, 128], 1),
    ("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv", "BF16", [2, 1], 2),
    ("model.language_model.layers.0.input_layernorm.weight", "BF16", [128], 3),
    ("model.language_model.norm.weight", "F32", [16], 4),
]
hdr = {"__metadata__": {"format": "pt"}}
off = 0
data = []
for name, dt, shape, seed in entries:
    b = tensor_bytes(dt, shape, seed)
    hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
    data.append(b)
    off += len(b)
hj = json.dumps(hdr).encode()
pad = (8 - (len(hj) + 8) % 8) % 8
hj += b" " * pad
with open("tests/fixtures/st/mini.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    for b in data:
        f.write(b)

cfg = {
    "text_config": {
        "num_hidden_layers": 4,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 512,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "bos_token_id": 11,
        "eos_token_id": 12,
        "max_position_embeddings": 8192,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"},
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    }
}
with open("tests/fixtures/st/config.json", "w", newline="\n") as f:
    json.dump({"model_type": "qwen3_5", **cfg}, f, indent=2)
print("wrote fixtures: mini.safetensors + config.json")

# per-tensor CRC manifest (regenerated deterministically from seeds)
import zlib
with open("tests/fixtures/st/manifest.txt", "w") as mf:
    for name, dt, shape, seed in entries:
        b = tensor_bytes(dt, shape, seed)
        mf.write("%s %s %s %08x\n" % (name, "x".join(map(str, shape)), dt, zlib.crc32(b)))
print("wrote manifest.txt")
