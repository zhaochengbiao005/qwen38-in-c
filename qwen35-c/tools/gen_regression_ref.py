#!/usr/bin/env python3
"""Ticket-regression: generate HF reference for 20 diverse prompts.

Each prompt: prefill + 8 greedy tokens. Stores per prompt:
  - the greedy token ids (to check exact match against engine)
  - the last-token logits (f32 [vocab]) for element-wise comparison

Output: tests/gates/real-22/regression/
  prompts.json     list of {prompt_text, ids, gen_ids, vocab}
  logits_<i>.bin   f32 [vocab] last-token logits for prompt i
"""
import json, sys, time
import numpy as np
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "../Qwen3.8-27B-FP8"

# 20 diverse prompts: QA, code, translation, math, reasoning, creative
PROMPTS_TEXT = [
    "The capital of China is",           # geography factual
    "def fibonacci(n):\n    ",           # python code
    "1 + 1 =",                          # arithmetic
    "Translate to French: Hello world",  # translation
    "The largest planet in our solar system is",  # science factual
    "def quicksort(arr):\n    ",         # python code 2
    "Once upon a time, there was a",     # creative story
    "The chemical formula for water is", # chemistry
    "2 * 3 + 4 =",                       # arithmetic 2
    "The capital of France is",          # geography 2
    "def binary_search(arr, target):\n    ",  # python code 3
    "Rome was not built in a",           # idiom
    "The speed of light is approximately",  # physics
    "def merge_sort(arr):\n    ",         # python code 4
    "The boiling point of water at sea level is",  # chemistry 2
    "A function that reverses a string in Python:\n    ",  # code instruction
    "The square root of 144 is",         # math 2
    "In machine learning, overfitting occurs when",  # ML knowledge
    "The primary language spoken in Brazil is",  # geography 3
    "def is_prime(n):\n    ",            # python code 5
]
GEN = 8

t0 = time.time()
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
import safetensors.torch as st
from transformers import AutoTokenizer
print(f"[ref] imports {time.time()-t0:.1f}s", flush=True)

cfg = AutoConfig.from_pretrained(MODEL)
BM = (cfg.quantization_config.get("weight_block_size") or [128,128])[0] if cfg.quantization_config else 128
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
print(f"[ref] meta construct {time.time()-t0:.1f}s", flush=True)

# tokenize prompts
tok = AutoTokenizer.from_pretrained(MODEL)
prompt_ids_list = [tok.encode(p, add_special_tokens=False) for p in PROMPTS_TEXT]

# load weights (reuse real_ref.py logic)
params = dict(model.named_parameters())
sd = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
by_shard = {}
for name, fn in sd.items():
    by_shard.setdefault(fn, []).append(name)
t0 = time.time()
with torch.no_grad():
    for fn, names in sorted(by_shard.items()):
        tensors = st.load_file(f"{MODEL}/{fn}")
        for name in names:
            if name not in params: continue
            w = tensors[name]
            sc = tensors.get(name + "_scale_inv")
            if sc is not None and w.dtype == torch.float8_e4m3fn:
                rows, cols = w.shape
                w = (w.view(rows//BM, BM, cols//BM, BM).to(torch.bfloat16)
                     * sc.view(rows//BM, 1, cols//BM, 1).to(torch.bfloat16)
                     ).view(rows, cols)
            elif w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
            params[name].data = w
        del tensors
print(f"[ref] loaded {time.time()-t0:.1f}s", flush=True)
model.eval()

import os
OUT = "tests/gates/real-22/regression"
os.makedirs(OUT, exist_ok=True)

results = []
past = [None]
def feed(batch):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([batch], dtype=torch.long),
                    past_key_values=past[0], use_cache=True)
    past[0] = out.past_key_values
    return out.logits

for pi, ids in enumerate(prompt_ids_list):
    past[0] = None
    lg = feed(ids)
    rows = [lg[0, -1].float().cpu().numpy()]
    nxt = int(lg[0, -1].float().argmax())
    gen = [nxt]
    for _ in range(GEN - 1):
        lg = feed([nxt])
        rows.append(lg[0, -1].float().cpu().numpy())
        nxt = int(lg[0, -1].float().argmax())
        gen.append(nxt)
    # store last-token logits
    rows[-1].astype(np.float32).tofile(f"{OUT}/logits_{pi}.bin")
    results.append({
        "idx": pi, "text": PROMPTS_TEXT[pi],
        "prompt_ids": ids, "gen_ids": gen,
    })
    print(f"[ref] prompt {pi}: gen={gen}  ({PROMPTS_TEXT[pi][:40]}...)", flush=True)

meta = {"vocab": int(rows[-1].shape[0]), "gen": GEN, "prompts": len(results)}
json.dump({**meta, "items": results},
          open(f"{OUT}/prompts.json", "w"), indent=1)
print(f"[ref] done: {len(results)} prompts, vocab={meta['vocab']}")
