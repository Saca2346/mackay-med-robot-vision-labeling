"""Move the ends of 8-number boxes along their own axis to heights (or x positions) read on the grid ruler.

Usage: python tools/set_ends.py NAME N far=Y near=Y [N far=Y ...] [--x]
  N         1-based box number (preview numbering), single angle-0 group specs only
  far/near  percent of the photo height where that end's midpoint must lie (--x: percent of the width, for
            lying boxes); either may be left out. Both corners of the end move by the same vector along the
            box axis, so the sides stay where they are. Derived size labels stay on the same physical text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_raw, load_spec, spec_groups, write_spec  # noqa: E402
from make_labels import resolve_obbs  # noqa: E402
from specops import rebase_size_labels  # noqa: E402

args = sys.argv[1:]
name = args[0]
use_x = "--x" in args
spec = load_spec(name)
groups = spec_groups(spec)
if len(groups) != 1 or groups[0].get("angle", 0) != 0:
    raise SystemExit("spec must have a single angle-0 group")
spec = {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")}
_, (W, H) = load_raw(name, 64)
_, old_sls, _ = resolve_obbs(name, spec, W, H)
boxes = groups[0]["box"]
i, n = 1, None
while i < len(args):
    a = args[i]
    i += 1
    if a == "--x":
        continue
    if "=" not in a:
        n = int(a)
        continue
    end, val = a.split("=")
    e = boxes[n - 1]
    px = [(e[k] / 100 * W, e[k + 1] / 100 * H) for k in range(0, 8, 2)]
    A = ((px[0][0] + px[1][0]) / 2, (px[0][1] + px[1][1]) / 2)
    B = ((px[2][0] + px[3][0]) / 2, (px[2][1] + px[3][1]) / 2)
    u = (B[0] - A[0], B[1] - A[1])
    P = A if end == "far" else B
    target = float(val) / 100 * (W if use_x else H)
    comp = 0 if use_x else 1
    if abs(u[comp]) < 1e-6:
        raise SystemExit(f"box {n} runs across that axis; use {'no --x' if use_x else '--x'}")
    s = (target - P[comp]) / u[comp]
    d = (u[0] * s, u[1] * s)
    idx = (0, 1) if end == "far" else (2, 3)
    for k in idx:
        px[k] = (px[k][0] + d[0], px[k][1] + d[1])
    boxes[n - 1] = [round(v, 2) for x, y in px for v in (x / W * 100, y / H * 100)]
    print(f"box {n} {end}: {'x' if use_x else 'y'} {P[comp] / (W if use_x else H) * 100:.2f} -> {float(val):.2f}")
rebase_size_labels(old_sls, spec, W, H)
write_spec(name, spec)
