"""Replace the size_labels of one box by height (or x) ranges read on the preview ruler.

Usage: python tools/set_sl.py NAME N y1-y2 [y1-y2 ...] [--x] [--v v1,v2]
  N        1-based box number (as numbered in the preview)
  y1-y2    start/end of each size_label along box N, in percent of the photo height (--x: of the width),
           i.e. where the box's centre-line reaches these heights; "0" or "-" removes all of box N's size_labels
  --v      width fraction across the box (default 0.05,0.95)
Only for specs with a single angle-0 group. Other boxes' size_labels are kept.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_spec, spec_groups, write_spec  # noqa: E402

args = sys.argv[1:]
name, n = args[0], int(args[1])
key = "x" if "--x" in args else "y"
v = [float(t) for t in args[args.index("--v") + 1].split(",")] if "--v" in args else None
ranges = [a for a in args[2:] if a not in ("--x", "--v") and (v is None or a != args[args.index("--v") + 1])]
spec = load_spec(name)
groups = spec_groups(spec)
if len(groups) != 1 or groups[0].get("angle", 0) != 0:
    raise SystemExit("spec must have a single angle-0 group")
g = groups[0]
if not 1 <= n <= len(g["box"]):
    raise SystemExit(f"no box {n}")
keep = [e for e in g.get("size_label", []) if not (isinstance(e, dict) and e.get("of") == n)]
new = []
for r in ranges:
    if r in ("0", "-"):
        continue
    a, b = (float(t) for t in r.split("-"))
    e = {"of": n, key: [a, b]}
    if v:
        e["v"] = v
    new.append(e)
g["size_label"] = sorted(keep + new, key=lambda e: (e["of"], min(e.get("y", e.get("x", e.get("t", [0]))))
                                                   if isinstance(e, dict) else 0))
write_spec(name, {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")})
print(f"box {n}: {len(new)} size_label -> {new}")
