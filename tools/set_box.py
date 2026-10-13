"""Replace box N of a single-group (angle 0) spec with 8 numbers read in a rotated view.

Usage: python tools/set_box.py NAME N ANGLE x1 y1 x2 y2 x3 y3 x4 y4
  N      1-based box number
  ANGLE  the --angle of the grid.py view the corners were read in (0 = original photo)
  x/y    far-end corner 1, far-end corner 2, near-end corner 1, near-end corner 2, in percent of
         that view's canvas
Several boxes: repeat "N ANGLE x1 ... y4" groups after NAME.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import canvas_geom, canvas_to_orig, load_raw, load_spec, spec_groups, write_spec
from make_labels import resolve_obbs
from specops import rebase_size_labels

args = sys.argv[1:]
name = args[0]
vals = args[1:]
spec = load_spec(name)
groups = spec_groups(spec)
if len(groups) != 1 or groups[0].get("angle", 0) != 0:
    raise SystemExit("spec must have a single angle-0 group")
_, (W, H) = load_raw(name, 64)
spec = {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")}
_, old_sls, _ = resolve_obbs(name, spec, W, H)  # keep size labels on the same text
while vals:
    n, a = int(vals[0]), float(vals[1])
    q = [float(v) for v in vals[2:10]]
    vals = vals[10:]
    Wc, Hc, _, _ = canvas_geom(W, H, a)
    out = []
    for k in (0, 2, 4, 6):
        x, y = canvas_to_orig(q[k] / 100 * Wc, q[k + 1] / 100 * Hc, W, H, a)
        out += [round(x / W * 100, 2), round(y / H * 100, 2)]
    print(f"box {n}: {groups[0]['box'][n - 1]} -> {out}")
    groups[0]["box"][n - 1] = out
rebase_size_labels(old_sls, spec, W, H)
write_spec(name, spec)
