"""Snap boxes to the spine edges visible in the photo.

Each box (the rectangle made from its 8-number spec entry) is refined in two steps:
  1. sides: each long side is moved to the strongest long straight edge near it. Candidate lines
     parallel to the side are scored by the median (over the middle of the side) of the colour
     gradient perpendicular to the line - a spine boundary runs the whole length, printed text
     does not. Among the profile's peaks the one nearest the current side is taken (peaks weaker
     than REL x the strongest are ignored), then both ends of the line are tuned independently.
  2. ends: the same along the axis for each end (skipped where the box is cut by the photo edge).
Every refined line is finally pulled INSET px towards the box interior, so boxes stay inside
their package. Size labels (derived slices) keep covering the same physical text.

Usage: python tools/refine.py NAME [--boxes 1,2,5] [--no-ends] [--no-sides] [--dry]
                              [--side-range 0.45] [--end-range 60]
Always check the result with tools/endcheck.py.
"""
import math
import os
import sys
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fit_obb_inside, load_raw, load_spec, obb_from_quad, spec_groups, write_spec
from make_labels import resolve_obbs

F = 2          # gradients are computed on the photo downscaled by F
INSET = 3.0    # full-res px, final pull towards the interior
REL = 0.45     # peaks weaker than REL * strongest are ignored
BORDER = 6.0   # full-res px: an end this close to the photo border is a cut end


class Grad:
    def __init__(self, img):
        small = img.resize((img.size[0] // F, img.size[1] // F), Image.LANCZOS)
        small = small.filter(ImageFilter.GaussianBlur(1.0))
        a = np.asarray(small, dtype=np.float32)
        self.gy, self.gx = np.gradient(a, axis=(0, 1))
        self.h, self.w = a.shape[:2]

    def perp(self, xs, ys, m):
        """|gradient . m| (max over colour channels) at full-res points xs, ys."""
        x = xs / F - 0.5
        y = ys / F - 0.5
        ok = (x >= 0) & (x <= self.w - 1.001) & (y >= 0) & (y <= self.h - 1.001)
        out = np.full(xs.shape, np.nan, dtype=np.float32)
        if not ok.any():
            return out
        x, y = x[ok], y[ok]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]

        def bl(arr):
            return (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x0 + 1] * fx * (1 - fy)
                    + arr[y0 + 1, x0] * (1 - fx) * fy + arr[y0 + 1, x0 + 1] * fx * fy)

        v = np.abs(bl(self.gx) * m[0] + bl(self.gy) * m[1]).max(axis=1)
        out[ok] = v
        return out


def scores(G, P0s, P1s, t0=0.08, t1=0.92):
    """Median perpendicular gradient along each of k segments P0s[k] -> P1s[k] (arrays k x 2)."""
    d = P1s - P0s
    L = np.hypot(d[:, 0], d[:, 1])
    n = int(max(20, min(600, L.max() / 3)))
    ts = np.linspace(t0, t1, n)
    xs = P0s[:, 0:1] + d[:, 0:1] * ts[None, :]
    ys = P0s[:, 1:2] + d[:, 1:2] * ts[None, :]
    m = np.stack([-d[:, 1] / L, d[:, 0] / L], axis=1)
    out = np.zeros(len(P0s))
    for k in range(len(P0s)):  # m differs per segment only slightly; keep exact
        v = G.perp(xs[k], ys[k], m[k])
        good = v[~np.isnan(v)]
        out[k] = np.median(good) if good.size >= 0.5 * n else 0.0
    return out


def pick_peak(offsets, prof, w=3):
    best = prof.max()
    if best <= 0:
        return None
    cand = []
    for i in range(len(prof)):
        lo, hi = max(0, i - w), min(len(prof), i + w + 1)
        if prof[i] >= prof[lo:hi].max() and prof[i] >= REL * best:
            cand.append(i)
    if not cand:
        return None
    i = min(cand, key=lambda j: abs(offsets[j]))
    return offsets[i]


def snap_line(G, P0, P1, out_dir, rng, step=1.0, tune=8):
    """Move segment P0-P1 along out_dir (unit, pointing outwards); return new (P0, P1)."""
    P0, P1, o = np.asarray(P0, float), np.asarray(P1, float), np.asarray(out_dir, float)
    offs = np.arange(-rng, rng + step / 2, step)
    prof = scores(G, P0[None, :] + offs[:, None] * o, P1[None, :] + offs[:, None] * o)
    c = pick_peak(offs, prof)
    if c is None:
        return P0, P1, False
    grid = np.arange(-tune, tune + 0.5, 1.0)
    a, b = np.meshgrid(c + grid, c + grid, indexing="ij")
    a, b = a.ravel(), b.ravel()
    s = scores(G, P0[None, :] + a[:, None] * o, P1[None, :] + b[:, None] * o)
    k = int(np.argmax(s))
    return P0 + (a[k] - INSET) * o, P1 + (b[k] - INSET) * o, True


def rect_corners(A, B, hw):
    ux, uy = (B[0] - A[0]), (B[1] - A[1])
    L = math.hypot(ux, uy)
    u = np.array([ux / L, uy / L])
    n = np.array([-u[1], u[0]])
    A, B = np.asarray(A, float), np.asarray(B, float)
    return u, n, [A + n * hw, A - n * hw, B + n * hw, B - n * hw]


def to_pct(pts, W, H):
    out = []
    for p in pts:
        out += [round(float(p[0]) / W * 100, 2), round(float(p[1]) / H * 100, 2)]
    return out


def near_border(p, W, H):
    return p[0] < BORDER or p[1] < BORDER or p[0] > W - BORDER or p[1] > H - BORDER


def main(name, only=None, do_sides=True, do_ends=True, dry=False, side_frac=0.45, end_rng=60.0):
    spec = load_spec(name)
    img, (W, H) = load_raw(name)
    boxes, sls, _ = resolve_obbs(name, spec, W, H)
    G = Grad(img)
    new_boxes = []
    for i, (A, B, hw) in enumerate(boxes, 1):
        u, n, (fp, fm, np_, nm) = rect_corners(A, B, hw)
        if only and i not in only:
            new_boxes.append(to_pct([fp, fm, np_, nm], W, H))
            continue
        msg = []
        if do_sides:
            rng = side_frac * 2 * hw + 4
            fp, np_, ok1 = snap_line(G, fp, np_, n, rng)
            fm, nm, ok2 = snap_line(G, fm, nm, -n, rng)
            msg.append(f"sides {'ok' if ok1 and ok2 else 'KEPT'}")
            A2, B2, hw2 = obb_from_quad(to_pct([fp, fm, np_, nm], W, H), W, H)
            u, n, (fp, fm, np_, nm) = rect_corners(A2, B2, hw2)
        if do_ends:
            for end in ("far", "near"):
                c1, c2 = (fp, fm) if end == "far" else (np_, nm)
                if near_border(c1, W, H) or near_border(c2, W, H):
                    msg.append(f"{end} cut")
                    continue
                o = -u if end == "far" else u
                c1n, c2n, ok = snap_line(G, c1, c2, o, end_rng, tune=10)
                if end == "far":
                    fp, fm = c1n, c2n
                else:
                    np_, nm = c1n, c2n
                msg.append(f"{end} {'ok' if ok else 'KEPT'}")
        q = to_pct([fp, fm, np_, nm], W, H)
        new_boxes.append(q)
        print(f"box {i:2d}: {', '.join(msg)}")

    # keep size labels on the same physical text: express them as fractions of the new boxes
    new_sl = []
    k = 0
    for g in spec_groups(spec):
        for e in g.get("size_label", []):
            poly = sls[k]
            k += 1
            if not isinstance(e, dict):
                new_sl.append(e)
                continue
            (A_new, B_new, _), _ = fit_obb_inside(obb_from_quad(new_boxes[e["of"] - 1], W, H), W, H)
            du = np.array(B_new) - np.array(A_new)
            L2 = float(du @ du)
            ts = [float((np.array(p) - np.array(A_new)) @ du / L2) for p in poly]
            ent = {"of": e["of"], "t": [round(max(0.0, min(ts)), 3), round(min(1.0, max(ts)), 3)]}
            if "v" in e:
                ent["v"] = e["v"]
            new_sl.append(ent)
    if dry:
        return
    write_spec(name, {"groups": [{"angle": 0, "box": new_boxes, "size_label": new_sl}],
                      "status": spec.get("status", "selesai"), "note": spec.get("note", "")})


if __name__ == "__main__":
    a = sys.argv[1:]
    only = None
    kw = {}
    if "--boxes" in a:
        i = a.index("--boxes")
        only = {int(v) for v in a[i + 1].split(",")}
        del a[i:i + 2]
    if "--side-range" in a:
        i = a.index("--side-range")
        kw["side_frac"] = float(a[i + 1])
        del a[i:i + 2]
    if "--end-range" in a:
        i = a.index("--end-range")
        kw["end_rng"] = float(a[i + 1])
        del a[i:i + 2]
    main(a[0], only, "--no-sides" not in a, "--no-ends" not in a, "--dry" in a, **kw)
