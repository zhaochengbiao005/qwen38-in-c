#!/usr/bin/env python3
"""Ticket-20 debug: diff two Q35_DUMP-format intermediate dumps record by record.
Usage: python tools/cmp_dump.py <ref.bin> <engine.bin> [--all]
Stops at the first diverging record (or prints all with --all)."""
import sys, struct
import numpy as np

def load(path):
    recs = {}
    with open(path, "rb") as f:
        buf = f.read()
    off = 0
    while off + 16 <= len(buf):
        pos, layer, kind, n = struct.unpack_from("4i", buf, off); off += 16
        vec = np.frombuffer(buf, np.float32, n, off).astype(np.float64); off += 4 * n
        recs[(pos, layer, kind)] = vec
    return recs

a, b = load(sys.argv[1]), load(sys.argv[2])
show_all = "--all" in sys.argv
keys = sorted(set(a) | set(b))
worst = ("", 0.0)
for k in keys:
    if k not in a: print(f"{k}: only in engine"); continue
    if k not in b: print(f"{k}: only in ref"); continue
    d = np.abs(a[k] - b[k]).max()
    if d > worst[1]: worst = (str(k), d)
    if d > 1e-4 or show_all:
        i = int(np.abs(a[k] - b[k]).argmax())
        print(f"pos={k[0]:3d} layer={k[1]:3d} kind={k[2]} maxabs={d:.6g} "
              f"at[{i}]: ref={a[k][i]:.6g} eng={b[k][i]:.6g}")
        if not show_all and d > 1e-2:
            print("(stopping at first >1e-2 divergence)")
            break
print(f"worst record: {worst[0]} maxabs={worst[1]:.6g}")
