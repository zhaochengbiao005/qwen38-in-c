#!/usr/bin/env python3
"""Ticket-22: element-wise engine-vs-HF logits gate on the real checkpoint.

Usage: python tools/cmp_real.py [ref_logits.bin] [eng_logits.bin]

Acceptance (research/qwen35-numspec.md sec 6, end-to-end row):
  - greedy token ids identical (checked from the *ids.json files)
  - per-row logits: top-1 identical, and maxabs <= atol(0.15) or
    cos-sim > 0.9999
"""
import json, sys
import numpy as np

ref_b = sys.argv[1] if len(sys.argv) > 1 else "ref_logits.bin"
eng_b = sys.argv[2] if len(sys.argv) > 2 else "eng_logits.bin"
ref = np.fromfile(ref_b, np.float32).reshape(-1, 248320)
eng = np.fromfile(eng_b, np.float32).reshape(-1, 248320)
ri = json.load(open("ref_ids.json"))
ei = json.load(open("eng_ids.json"))
assert ref.shape == eng.shape, (ref.shape, eng.shape)

print(f"rows={ref.shape[0]} vocab={ref.shape[1]}")
print(f"ref ids: {ri['ids']}")
print(f"eng ids: {ei['ids']}")
ids_ok = ri["ids"] == ei["ids"]
print(f"ids exact: {'YES' if ids_ok else 'NO'}  gen ref={ri['gen']} eng={ei['gen']}")

fails = 0
for r in range(ref.shape[0]):
    a, b = ref[r].astype(np.float64), eng[r].astype(np.float64)
    d = np.abs(a - b)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    t1r, t1e = int(a.argmax()), int(b.argmax())
    ok = (t1r == t1e) and (d.max() <= 0.15 or cos > 0.9999)
    tag = f"row{r:2d} top1 {t1r:6d}=={t1e:6d} maxabs={d.max():8.4f} mean={d.mean():7.4f} cos={cos:.6f} {'OK' if ok else 'FAIL'}"
    print(tag)
    if not ok: fails += 1
print("RESULT:", "PASS" if (fails == 0 and ids_ok) else "FAIL")
sys.exit(0 if (fails == 0 and ids_ok) else 1)
