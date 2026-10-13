"""Patch individual coordinates in a label spec.

Usage: python tools/patch_spec.py NAME [--group G] CLS IDX=x1,x2,y1,y2 [IDX=...]
  IDX is 1-based within the group (default group 0); use '_' to keep a value.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_spec, spec_groups, write_spec

args = sys.argv[1:]
gi = 0
if "--group" in args:
    i = args.index("--group")
    gi = int(args[i + 1])
    del args[i:i + 2]
name, cls = args[0], args[1]
spec = load_spec(name)
groups = spec_groups(spec)
spec = {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")}
for arg in args[2:]:
    idx, vals = arg.split("=")
    i = int(idx) - 1
    cur = groups[gi][cls][i]
    parts = vals.split(",")
    if isinstance(cur, dict):
        raise SystemExit(f"{cls}[{idx}] is a derived label {cur}; edit it in the json")
    if len(parts) > len(cur):
        raise SystemExit(f"{cls}[{idx}] has {len(cur)} values, got {len(parts)}")
    # fewer values than the entry has: patch the leading values and keep the rest
    new = [cur[k] if k >= len(parts) or parts[k] == "_" else float(parts[k]) for k in range(len(cur))]
    print(f"g{gi} {cls}[{idx}]: {cur} -> {new}")
    groups[gi][cls][i] = new
write_spec(name, spec)
