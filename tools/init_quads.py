"""Initial 8-number (oriented) boxes from an old axis-aligned spec.

Every old box [x1, x2, y1, y2] hugged its spine's extreme points, so for a spine leaning by a
known angle the rectangle inside it is determined up to its width: the AABB width minus the
drift of the spine over its length. The lean of each box is measured from the photo (dominant
edge direction inside the box, nearest to the box's long axis) unless given with --lean.
The result is only a starting point; check it with tools/endcheck.py and fix with
tools/adjust_box.py. Size labels are then re-derived with tools/derive_sl.py.

Usage: python tools/init_quads.py NAME OLD_SPEC_JSON [--lean "1-4=99.5,5=92"] [--dry]
  --lean  spine direction per box (degrees, as printed by est_angle: 90 = vertical,
          0/180 = horizontal), overriding the measured one
"""
import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_raw, load_spec, spec_groups, write_spec


def edge_dir(gray, W, H, rect, tall, window=25):
    x1, x2, y1, y2 = rect
    crop = gray.crop((W * x1 / 100, H * y1 / 100, W * x2 / 100, H * y2 / 100))
    s = 700 / max(crop.size)
    if s < 1:
        crop = crop.resize((max(8, round(crop.size[0] * s)), max(8, round(crop.size[1] * s))))
    a = np.asarray(crop, dtype=np.float32)
    gy, gx = np.gradient(a)
    mag = np.hypot(gx, gy)
    m = mag > np.percentile(mag, 88)
    ang = (np.degrees(np.arctan2(gy[m], gx[m])) + 90.0) % 180.0
    w = mag[m] ** 2
    target = 90.0 if tall else 0.0
    d = (ang - target + 90) % 180 - 90  # signed distance to the target direction
    keep = np.abs(d) <= window
    if keep.sum() < 20:
        return target
    hist, edges = np.histogram(d[keep], bins=4 * window, range=(-window, window), weights=w[keep])
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    hs = np.convolve(hist, k / k.sum(), mode="same")
    i = int(np.argmax(hs))
    lo, hi = max(0, i - 2), min(len(hs), i + 3)
    centers = (edges[:-1] + edges[1:]) / 2
    best = float((centers[lo:hi] * hs[lo:hi]).sum() / max(hs[lo:hi].sum(), 1e-9))
    return (target + best) % 180


def parse_lean(s):
    out = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=")
        if "-" in k:
            a, b = map(int, k.split("-"))
            for n in range(a, b + 1):
                out[n] = float(v)
        else:
            out[int(k)] = float(v)
    return out


def quad_from_aabb(rect, theta, W, H):
    x1, x2, y1, y2 = (rect[0] / 100 * W, rect[1] / 100 * W, rect[2] / 100 * H, rect[3] / 100 * H)
    ux, uy = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    tall = (y2 - y1) >= (x2 - x1)
    if tall:
        if uy < 0:
            ux, uy = -ux, -uy
        drift = (y2 - y1) * ux / uy
        w = (x2 - x1) - abs(drift)
        w = max(w, 0.35 * (x2 - x1))
        if drift >= 0:  # going down moves right: top end on the left
            far = [(x1, y1), (x1 + w, y1)]
            near = [(x2 - w, y2), (x2, y2)]
        else:
            far = [(x2 - w, y1), (x2, y1)]
            near = [(x1, y2), (x1 + w, y2)]
    else:
        if ux < 0:
            ux, uy = -ux, -uy
        drift = (x2 - x1) * uy / ux
        w = (y2 - y1) - abs(drift)
        w = max(w, 0.35 * (y2 - y1))
        if drift >= 0:  # going right moves down: left end on top
            far = [(x1, y1), (x1, y1 + w)]
            near = [(x2, y2 - w), (x2, y2)]
        else:
            far = [(x1, y2 - w), (x1, y2)]
            near = [(x2, y1), (x2, y1 + w)]
    out = []
    for x, y in far + near:
        out += [round(x / W * 100, 2), round(y / H * 100, 2)]
    return out, tall


def assign_sls(boxes, sls):
    """Old size labels were written in box order: give each to the next box that contains it."""
    owner, k = [], 0
    for sl in sls:
        cx, cy = (sl[0] + sl[1]) / 2, (sl[2] + sl[3]) / 2
        j = k
        while j < len(boxes) and not (boxes[j][0] - 0.3 <= cx <= boxes[j][1] + 0.3
                                      and boxes[j][2] - 0.3 <= cy <= boxes[j][3] + 0.3):
            j += 1
        if j == len(boxes):  # not in order: fall back to any containing box
            j = min(range(len(boxes)), key=lambda i: abs((boxes[i][0] + boxes[i][1]) / 2 - cx))
        owner.append(j)
        k = j
    return owner


def quad_from_axis(rect, theta, pt, w, W, H):
    """Rectangle-ish quad along direction theta through point pt (px), ends at the AABB extremes."""
    x1, x2, y1, y2 = (rect[0] / 100 * W, rect[1] / 100 * W, rect[2] / 100 * H, rect[3] / 100 * H)
    ux, uy = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    tall = (y2 - y1) >= (x2 - x1)
    if tall:
        if uy < 0:
            ux, uy = -ux, -uy
        xa = pt[0] + (y1 - pt[1]) * ux / uy
        xb = pt[0] + (y2 - pt[1]) * ux / uy
        hwx = w / 2 / max(uy, 1e-6)  # horizontal half-width of a spine of width w
        pts = [(xa - hwx, y1), (xa + hwx, y1), (xb - hwx, y2), (xb + hwx, y2)]
    else:
        if ux < 0:
            ux, uy = -ux, -uy
        ya = pt[1] + (x1 - pt[0]) * uy / ux
        yb = pt[1] + (x2 - pt[0]) * uy / ux
        hwy = w / 2 / max(ux, 1e-6)
        pts = [(x1, ya - hwy), (x1, ya + hwy), (x2, yb - hwy), (x2, yb + hwy)]
    out = []
    for x, y in pts:
        out += [round(x / W * 100, 2), round(y / H * 100, 2)]
    return out


def main(name, old_path, lean_s="", dry=False, use_sl=True):
    old = json.load(open(old_path, encoding="utf-8"))
    boxes = [b for g in spec_groups(old) for b in g.get("box", [])]
    sls = [s for g in spec_groups(old) for s in g.get("size_label", []) if isinstance(s, list)]
    if not use_sl:
        sls = []
    owner = assign_sls(boxes, sls)
    img, (W, H) = load_raw(name)
    gray = img.convert("L")
    lean = parse_lean(lean_s)
    info = []
    for n, b in enumerate(boxes, 1):
        tall = (b[3] - b[2]) * H >= (b[1] - b[0]) * W
        theta = lean.get(n)
        if theta is None:
            theta = edge_dir(gray, W, H, b, tall)
        mine = [sls[i] for i in range(len(sls)) if owner[i] == n - 1]
        if mine:  # axis through the (area-weighted) centre of this box's size labels
            wts = [(s[1] - s[0]) * (s[3] - s[2]) for s in mine]
            cx = sum((s[0] + s[1]) / 2 * wt for s, wt in zip(mine, wts)) / sum(wts) / 100 * W
            cy = sum((s[2] + s[3]) / 2 * wt for s, wt in zip(mine, wts)) / sum(wts) / 100 * H
            pt = (cx, cy)
        else:
            pt = None
        q_aabb, _ = quad_from_aabb(b, theta, W, H)
        # spine width from the AABB (width minus drift), in px across the spine
        ux, uy = abs(math.cos(math.radians(theta))), abs(math.sin(math.radians(theta)))
        if tall:
            w = max(((b[1] - b[0]) / 100 * W - (b[3] - b[2]) / 100 * H * ux / max(uy, 1e-6)) * uy, 0.0)
        else:
            w = max(((b[3] - b[2]) / 100 * H - (b[1] - b[0]) / 100 * W * uy / max(ux, 1e-6)) * ux, 0.0)
        info.append([b, theta, pt, w, tall, q_aabb])
    # cap each width by the spacing to the neighbouring axes (spines of a stack touch)
    mids = []
    for b, theta, pt, w, tall, q in info:
        if pt is None:
            mids.append(None)
            continue
        mids.append(pt)
    new = []
    for i, (b, theta, pt, w, tall, q_aabb) in enumerate(info):
        if pt is None:
            q = q_aabb
            how = "aabb"
        else:
            gaps = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(info) and info[j][2] is not None:
                    pj = info[j][2]
                    ux, uy = math.cos(math.radians(theta)), math.sin(math.radians(theta))
                    gaps.append(abs((pj[0] - pt[0]) * -uy + (pj[1] - pt[1]) * ux))
            cap = 0.95 * min(gaps) if gaps else w
            ww = min(w, cap) if w > 0 else cap
            ww = max(ww, 0.5 * cap)
            q = quad_from_axis(b, theta, pt, ww, W, H)
            how = f"axis w={ww:.0f}px"
        print(f"box {i + 1:2d} {'tall' if tall else 'wide'} theta={theta:6.2f} {how} -> {q}")
        new.append(q)
    if dry:
        return
    spec = load_spec(name)
    status, note = spec.get("status", "selesai"), spec.get("note", "")
    write_spec(name, {"groups": [{"angle": 0, "box": new, "size_label": []}], "status": status, "note": note})


if __name__ == "__main__":
    a = sys.argv[1:]
    lean_s = ""
    if "--lean" in a:
        i = a.index("--lean")
        lean_s = a[i + 1]
        del a[i:i + 2]
    dry = "--dry" in a
    use_sl = "--aabb" not in a
    a = [x for x in a if x not in ("--dry", "--aabb")]
    main(a[0], a[1], lean_s, dry, use_sl)
