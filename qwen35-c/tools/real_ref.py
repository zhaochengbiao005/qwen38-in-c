#!/usr/bin/env python3
"""Ticket-22: HF transformers reference on the real FP8 checkpoint (manual load).

The from_pretrained fp8-quantizer path needs triton (Windows: none) and
segfaulted mid-load on this box, so we load manually:
  - construct the model on the meta device (no quantization config)
  - to_empty(cpu), re-init the non-persistent rotary buffers
  - per shard: dequantize fp8 x scale_inv per 128x128 block (the exact
    semantics of transformers/integrations/finegrained_fp8.py dequantize:
    w8.to(scale.dtype) * scale, block reshape) and assign into params

Forward: prompt + greedy continuation, DynamicCache, eager attention, bf16
per config dtype (fp32-critical nodes upcast inside the modeling code).
Dumps ref_logits.bin (f32 [rows, vocab]) + ref_ids.json.
"""
import json, sys, time
import numpy as np
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "../Qwen3.8-27B-FP8"
PROMPT = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [760, 6511, 314, 5440, 369]
GEN = 8

t0 = time.time()
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.cache_utils import DynamicCache
import safetensors.torch as st
print(f"[ref] imports {time.time()-t0:.1f}s", flush=True)

cfg = AutoConfig.from_pretrained(MODEL)
qc = cfg.quantization_config or {}
BM = (qc.get("weight_block_size") or [128, 128])[0]
cfg.quantization_config = None
if getattr(cfg, "text_config", None) is not None:
    cfg.text_config.quantization_config = None

t0 = time.time()
with torch.device("meta"):
    model = Qwen3_5ForConditionalGeneration(cfg)
model.to_empty(device="cpu")
for m in model.modules():
    n = type(m).__name__
    if n == "Qwen3_5TextRotaryEmbedding":
        type(m).__init__(m, m.config)
    elif n == "Qwen3_5VisionRotaryEmbedding":
        type(m).__init__(m, m.dim, m.theta)
print(f"[ref] meta construct + buffer re-init {time.time()-t0:.1f}s", flush=True)

params = dict(model.named_parameters())
sd = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
by_shard = {}
for name, fn in sd.items():
    by_shard.setdefault(fn, []).append(name)

t0 = time.time()
loaded = set()
with torch.no_grad():
    for fn, names in sorted(by_shard.items()):
        ts = time.time()
        tensors = st.load_file(f"{MODEL}/{fn}")
        for name in names:
            if name not in params:
                continue
            w = tensors[name]
            sc = tensors.get(name + "_scale_inv")
            if sc is not None and w.dtype == torch.float8_e4m3fn:
                rows, cols = w.shape
                assert rows % BM == 0 and cols % BM == 0, (name, rows, cols)
                w = (w.view(rows // BM, BM, cols // BM, BM).to(torch.bfloat16)
                     * sc.view(rows // BM, 1, cols // BM, 1).to(torch.bfloat16)
                     ).view(rows, cols)
            elif w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
            params[name].data = w
            loaded.add(name)
        del tensors
        print(f"[ref] {fn}: {len(names)} tensors {time.time()-ts:.1f}s", flush=True)
load_s = time.time() - t0

missing = [n for n in params if n not in loaded and "visual" not in n]
print(f"[ref] loaded in {load_s:.1f}s, missing={missing[:5]} ({len(missing)})", flush=True)
model.eval()

ids = list(PROMPT)
rows, times = [], []
past = [None]
def feed(batch):
    t = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([batch], dtype=torch.long),
                    past_key_values=past[0], use_cache=True)
    past[0] = out.past_key_values
    times.append(time.time() - t)
    return out.logits

lg = feed(ids)
rows.extend(lg[0].float().cpu().numpy())
nxt = int(lg[0, -1].float().argmax())
ids.append(nxt)
for _ in range(GEN - 1):
    lg = feed([nxt])
    rows.append(lg[0, -1].float().cpu().numpy())
    nxt = int(lg[0, -1].float().argmax())
    ids.append(nxt)

arr = np.stack(rows).astype(np.float32)
arr.tofile("ref_logits.bin")
gen_ids = ids[len(PROMPT):]
json.dump({"prompt": PROMPT, "ids": ids, "gen": gen_ids,
           "vocab": int(arr.shape[1]), "rows": int(arr.shape[0]),
           "load_s": load_s, "prefill_s": times[0], "decode_s": times[1:],
           "dtype": "bf16", "impl": "transformers-5.3.0-qwen3_5-manual-dequant-eager"},
          open("ref_ids.json", "w"), indent=1)
print(f"[ref] rows={arr.shape[0]} vocab={arr.shape[1]} gen={gen_ids}")
print(f"[ref] load {load_s:.1f}s prefill {times[0]:.1f}s "
      f"decode avg {sum(times[1:])/len(times[1:]):.2f}s/tok", flush=True)
