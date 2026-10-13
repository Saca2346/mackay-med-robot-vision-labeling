"""Sanity checks over label_specs, on the final geometry (original photo pixels).

Every size_label must lie inside some box polygon: each of its corners may stick out of the
best-matching box by at most TOL_PX pixels. Also reports boxes that had to be shortened.

Exit code is non-zero if any spec has a problem that isn't recorded in ACCEPTED (see accept_warning()
below / --accept), or if a spec crashes the checker; a pipeline should treat this as a hard stop, not
a warning to log and continue past.

Usage: python tools/check_specs.py [NAME ...]   (default: all specs)
       python tools/check_specs.py --accept NAME "exact problem text" "reason for accepting it"
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SPECS, load_expiry, load_raw, load_spec, obb_polygon
from make_labels import resolve_obbs

TOL_PX = 6.0  # ~0.15% of the 4080 px side
ACCEPTED = os.path.join(ROOT, "check_specs_accepted.json")


def load_accepted():
    """{name: {problem_text: reason}} of warnings a human has reviewed and approved, e.g. a package
    genuinely cut by the photo border. Matched on the exact problem text, so if the geometry changes
    (spec edited) the old acceptance no longer applies and the gate fails again until re-reviewed."""
    return json.load(open(ACCEPTED)) if os.path.exists(ACCEPTED) else {}


def accept_warning(name, problem, reason):
    accepted = load_accepted()
    accepted.setdefault(name, {})[problem.strip()] = reason
    json.dump(accepted, open(ACCEPTED, "w"), indent=1, ensure_ascii=False)


def outside_dist(pt, poly):
    """How far `pt` lies outside convex polygon `poly` (0 if inside)."""
    worst = 0.0
    n = len(poly)
    # orientation of the polygon
    area = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    sgn = 1 if area > 0 else -1
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        L = math.hypot(ex, ey) or 1.0
        cross = (ex * (pt[1] - y1) - ey * (pt[0] - x1)) / L * sgn
        if cross < 0:
            worst = max(worst, -cross)
    return worst


def clip(subject, clipper):
    """Sutherland-Hodgman intersection of two convex polygons."""
    def area2(p):
        return sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))

    if area2(clipper) < 0:
        clipper = clipper[::-1]
    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def inter(p, q):
            x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
            x3, y3, x4, y4 = p[0], p[1], q[0], q[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-12:
                return q
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out.append(inter(p, q))
                out.append(q)
            elif inside(p):
                out.append(inter(p, q))
    return out


def poly_area(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                   for i in range(len(p)))) / 2 if len(p) >= 3 else 0.0


OVERLAP = 0.12  # report box pairs sharing more than this fraction of the smaller box


def check(name):
    spec = load_spec(name)
    _, (W, H) = load_raw(name, 64)
    boxes, sls, warnings = resolve_obbs(name, spec, W, H)
    polys = [obb_polygon(b) for b in boxes]
    # boxes cut at the photo border are expected; only a large cut hints at a wrong corner
    problems = ["  " + w for w in warnings if float(w.split("shortened ")[1].split("%")[0]) > 10]
    rotated = "groups" in spec and any(len(e) == 8 for g in spec["groups"] for e in g.get("box", []))
    if rotated:
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                inter = poly_area(clip(polys[i], polys[j]))
                small = min(poly_area(polys[i]), poly_area(polys[j]))
                if small and inter / small > OVERLAP:
                    problems.append(f"  box[{i + 1}] and box[{j + 1}] overlap {inter / small * 100:.0f}% "
                                    f"of the smaller one")
    for i, sl in enumerate(sls, 1):
        if not polys:
            problems.append(f"  size_label[{i}] but no boxes")
            continue
        best = min((max(outside_dist(p, bp) for p in sl), j) for j, bp in enumerate(polys, 1))
        if best[0] > TOL_PX:
            problems.append(f"  size_label[{i}] sticks out of box[{best[1]}] by {best[0]:.0f}px "
                            f"({best[0] / W * 100:.2f}% of width)")
    # expiry_label (proposals too): same rule, and must sit inside the box it was read from
    ex = load_expiry(name) or {}
    for i, e in enumerate(ex.get("expiry_label", []), 1):
        pts = [(e["poly"][k] / 100 * W, e["poly"][k + 1] / 100 * H) for k in range(0, 8, 2)]
        if not polys:
            problems.append(f"  expiry_label[{i}] but no boxes")
            continue
        best = min((max(outside_dist(p, bp) for p in pts), j) for j, bp in enumerate(polys, 1))
        if best[0] > TOL_PX:
            problems.append(f"  expiry_label[{i}] sticks out of box[{best[1]}] by {best[0]:.0f}px")
        elif e.get("box") and best[1] != e["box"]:
            problems.append(f"  expiry_label[{i}] lies in box[{best[1]}], file says box {e['box']}")
    return problems


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--accept"]:
        accept_warning(*args[1:4])
        print(f"diterima: {args[1]}: {args[2].strip()}  ({args[3]})")
        sys.exit(0)

    names = args or sorted(f[:-5] for f in os.listdir(SPECS) if f.endswith(".json"))
    accepted = load_accepted()
    bad, crashed, accepted_count = 0, 0, 0
    for n in names:
        try:
            problems = check(n)
        except Exception as e:
            crashed += 1
            print(f"{n}: CRASH saat check_specs - {e!r}")
            continue
        new = [p for p in problems if p.strip() not in accepted.get(n, {})]
        shown_accepted = [p for p in problems if p.strip() in accepted.get(n, {})]
        if new or shown_accepted:
            print(n)
            print("\n".join(new))
            for p in shown_accepted:
                print(p + f"   [diterima: {accepted[n][p.strip()]}]")
        if new:
            bad += 1
        accepted_count += len(shown_accepted)
    print(f"checked {len(names)} specs, {bad} with new/unaccepted problems, "
          f"{accepted_count} accepted warnings shown, {crashed} crashed")
    if bad or crashed:
        sys.exit(1)
