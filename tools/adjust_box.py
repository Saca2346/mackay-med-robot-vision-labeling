"""Move the ends / sides of an 8-number box along its own axis, in percent of the photo.

Usage: python tools/adjust_box.py NAME BOX KEY=DELTA [KEY=DELTA ...]
  BOX    1-based box number over all groups (as numbered in previews / endcheck)
  KEY    far | near            both corners of that end, along the spine axis
         far1 | far2 | near1 | near2   a single corner (numbering as in the spec entry)
         side1 | side2         the corner-1 / corner-2 long side, sideways (both ends)
         far1_side | far2_side | near1_side | near2_side   one corner, sideways
  DELTA  percent, positive = outward (longer / wider), negative = inward. Along a mostly
         vertical axis a percent is of the photo height, otherwise of its width - the same
         unit as the endcheck ruler.
Example: python tools/adjust_box.py IMG_x 3 near=+0.6 far1=-0.25
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import canvas_geom, canvas_to_orig, load_raw, load_spec, orig_to_canvas, spec_groups, write_spec
from make_labels import resolve_obbs
from specops import rebase_size_labels


def locate(spec, n):
    k = 0
    for gi, g in enumerate(spec_groups(spec)):
        for bi, e in enumerate(g.get("box", [])):
            k += 1
            if k == n:
                return gi, bi
    raise SystemExit(f"no box {n}")


def main(name, n, changes):
    spec = load_spec(name)
    groups = spec_groups(spec)
    spec = {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")}
    _, (W0, H0) = load_raw(name, 64)
    _, old_sls, _ = resolve_obbs(name, spec, W0, H0)  # keep size labels on the same text
    gi, bi = locate(spec, n)
    g = groups[gi]
    e = g["box"][bi]
    if len(e) != 8:
        raise SystemExit(f"box {n} is not an 8-number box: {e}")
    a = g.get("angle", 0)
    _, (W, H) = load_raw(name, 64)
    Wc, Hc, _, _ = canvas_geom(W, H, a)
    pts = [canvas_to_orig(e[k] / 100 * Wc, e[k + 1] / 100 * Hc, W, H, a) for k in (0, 2, 4, 6)]
    A = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    B = ((pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2)
    L = math.hypot(B[0] - A[0], B[1] - A[1])
    u = ((B[0] - A[0]) / L, (B[1] - A[1]) / L)
    unit = (H if abs(u[1]) >= abs(u[0]) else W) / 100
    # lateral direction pointing from corner 2 towards corner 1 (averaged over both ends)
    lx, ly = (pts[0][0] - pts[1][0] + pts[2][0] - pts[3][0]), (pts[0][1] - pts[1][1] + pts[2][1] - pts[3][1])
    ll = math.hypot(lx, ly) or 1
    lat = (lx / ll, ly / ll)
    moves = {"far": [(0, -1), (1, -1)], "near": [(2, 1), (3, 1)], "far1": [(0, -1)], "far2": [(1, -1)],
             "near1": [(2, 1)], "near2": [(3, 1)]}
    for key, val in changes:
        d = val * unit
        if key in moves:
            for idx, sgn in moves[key]:
                pts[idx] = (pts[idx][0] + sgn * u[0] * d, pts[idx][1] + sgn * u[1] * d)
        elif key in ("side1", "side2"):
            sgn = 1 if key == "side1" else -1
            for idx in ((0, 2) if key == "side1" else (1, 3)):
                pts[idx] = (pts[idx][0] + sgn * lat[0] * d, pts[idx][1] + sgn * lat[1] * d)
        elif key in ("far1_side", "far2_side", "near1_side", "near2_side"):
            idx = {"far1_side": 0, "far2_side": 1, "near1_side": 2, "near2_side": 3}[key]
            sgn = 1 if idx in (0, 2) else -1
            pts[idx] = (pts[idx][0] + sgn * lat[0] * d, pts[idx][1] + sgn * lat[1] * d)
        else:
            raise SystemExit(f"unknown key {key}")
    new = []
    for x, y in pts:
        X, Y = orig_to_canvas(x, y, W, H, a)
        new += [round(X / Wc * 100, 2), round(Y / Hc * 100, 2)]
    print(f"box {n}: {e} -> {new}")
    g["box"][bi] = new
    rebase_size_labels(old_sls, spec, W0, H0)
    write_spec(name, spec)


if __name__ == "__main__":
    args = sys.argv[1:]
    ch = []
    for s in args[2:]:
        k, v = s.split("=")
        ch.append((k, float(v)))
    main(args[0], int(args[1]), ch)
