"""Replace the far or near end (two corners) of 8-number boxes, read in a rotated view.

Usage: python tools/set_end.py NAME  N far|near ANGLE x1 y1 x2 y2  [N far|near ANGLE x1 y1 x2 y2 ...]
  N      1-based box number over all groups
  ANGLE  the --angle of the grid.py view the corners were read in (0 = original photo)
  x/y    corner 1 then corner 2 of that end, in percent of that view's canvas
The other end is kept. Derived size labels stay on the same physical text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import canvas_geom, canvas_to_orig, load_raw, load_spec, orig_to_canvas, spec_groups, write_spec
from make_labels import resolve_obbs
from specops import rebase_size_labels

args = sys.argv[1:]
name, vals = args[0], args[1:]
spec = load_spec(name)
groups = spec_groups(spec)
spec = {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")}
_, (W, H) = load_raw(name, 64)
_, old_sls, _ = resolve_obbs(name, spec, W, H)

index = []
for gi, g in enumerate(groups):
    for bi in range(len(g.get("box", []))):
        index.append((gi, bi))

while vals:
    n, end, a = int(vals[0]), vals[1], float(vals[2])
    x1, y1, x2, y2 = (float(v) for v in vals[3:7])
    vals = vals[7:]
    gi, bi = index[n - 1]
    g = groups[gi]
    e = g["box"][bi]
    ga = g.get("angle", 0)
    if len(e) != 8:
        raise SystemExit(f"box {n} is not an 8-number box")
    Wc, Hc, _, _ = canvas_geom(W, H, a)
    Wg, Hg, _, _ = canvas_geom(W, H, ga)
    new_pts = []
    for (x, y) in ((x1, y1), (x2, y2)):
        ox, oy = canvas_to_orig(x / 100 * Wc, y / 100 * Hc, W, H, a)   # view -> photo
        gx, gy = orig_to_canvas(ox, oy, W, H, ga)                        # photo -> group frame
        new_pts += [round(gx / Wg * 100, 2), round(gy / Hg * 100, 2)]
    k = 0 if end == "far" else 4
    old = list(e)
    e[k:k + 4] = new_pts
    print(f"box {n} {end}: {old[k:k + 4]} -> {new_pts}")

rebase_size_labels(old_sls, spec, W, H)
write_spec(name, spec)
