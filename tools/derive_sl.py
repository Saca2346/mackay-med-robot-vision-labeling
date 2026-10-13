"""Turn old axis-aligned size_labels into slices of the new rotated boxes.

For every old size_label rect (percent of the original photo, from an old spec), find the box
whose centre-line passes closest to the rect centre and take the part of that centre-line
inside the rect as the slice {"of": N, "t": [t1, t2]}. The result replaces the size_label lists
of the current spec (all put in group 0, since "t" does not depend on the frame).

Usage: python tools/derive_sl.py NAME OLD_SPEC_JSON [--dry]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_raw, load_spec, spec_groups, write_spec
from make_labels import resolve_obbs


def seg_rect_t(A, B, rect):
    """Parameter range of segment A->B inside axis-aligned rect (x1, x2, y1, y2) in pixels."""
    t0, t1 = 0.0, 1.0
    dx, dy = B[0] - A[0], B[1] - A[1]
    for p, q in ((-dx, A[0] - rect[0]), (dx, rect[1] - A[0]), (-dy, A[1] - rect[2]), (dy, rect[3] - A[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    return (t0, t1) if t1 > t0 else None


def main(name, old_path, dry=False):
    spec = load_spec(name)
    _, (W, H) = load_raw(name, 64)
    boxes, _, _ = resolve_obbs(name, {"groups": [dict(g, size_label=[]) for g in spec_groups(spec)]}, W, H)
    old = json.load(open(old_path, encoding="utf-8"))
    old_sls = [sl for g in spec_groups(old) for sl in g.get("size_label", [])]
    out = []
    for i, sl in enumerate(old_sls, 1):
        rect = (sl[0] / 100 * W, sl[1] / 100 * W, sl[2] / 100 * H, sl[3] / 100 * H)
        c = ((rect[0] + rect[1]) / 2, (rect[2] + rect[3]) / 2)
        best = None
        for n, (A, B, hw) in enumerate(boxes, 1):
            dx, dy = B[0] - A[0], B[1] - A[1]
            L2 = dx * dx + dy * dy
            t = ((c[0] - A[0]) * dx + (c[1] - A[1]) * dy) / L2
            tc = min(max(t, 0), 1)
            px, py = A[0] + dx * tc, A[1] + dy * tc
            d = math.hypot(c[0] - px, c[1] - py) / max(hw, 1)
            rng = seg_rect_t(A, B, rect)
            if rng is None:
                continue
            if best is None or d < best[0]:
                best = (d, n, rng)
        if best is None:
            print(f"  old size_label[{i}] {sl}: no box crosses it -> dropped")
            continue
        d, n, (t1, t2) = best
        flag = "" if d < 1.0 else "   <-- centre outside box, check"
        print(f"  old size_label[{i}] -> box {n} t=[{t1:.3f}, {t2:.3f}] (centre {d:.2f} half-widths off axis){flag}")
        out.append({"of": n, "t": [round(t1, 3), round(t2, 3)]})
    if dry:
        return
    groups = spec_groups(spec)
    for g in groups:
        g["size_label"] = []
    groups[0]["size_label"] = out
    write_spec(name, {"groups": groups, "status": spec.get("status", "selesai"), "note": spec.get("note", "")})


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], "--dry" in a)
