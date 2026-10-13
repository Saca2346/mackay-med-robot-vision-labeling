"""Remove box N (1-based over all groups) and the derived size labels that refer to it.

Derived size labels ({"of": M, ...}) of later boxes are renumbered. Plain-rect size labels are
kept as they are.
Usage: python tools/drop_box.py NAME N
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_spec, spec_groups, write_spec

name, n = sys.argv[1], int(sys.argv[2])
spec = load_spec(name)
groups = spec_groups(spec)
k = 0
for g in groups:
    boxes = g.get("box", [])
    if k + len(boxes) >= n:
        dropped = boxes.pop(n - k - 1)
        print(f"dropped box {n}: {dropped}")
        break
    k += len(boxes)
else:
    raise SystemExit(f"no box {n}")
for g in groups:
    out = []
    for e in g.get("size_label", []):
        if isinstance(e, dict):
            if e["of"] == n:
                print(f"dropped size_label {e}")
                continue
            if e["of"] > n:
                e = dict(e, of=e["of"] - 1)
        out.append(e)
    g["size_label"] = out
write_spec(name, {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")})
