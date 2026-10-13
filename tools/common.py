"""Shared helpers for the box/size_label labeling tools.

Rotated labeling: a spec is a list of groups, each with an `angle` (degrees, counter-clockwise
as seen on screen). Boxes of a group are axis-aligned rectangles given in percent of the
*rotated canvas* (the photo rotated by `angle` with expand=True), so packages lying diagonally
can be labeled as tight rectangles in a view where they stand straight. They are mapped back to
the original photo as 4-corner oriented boxes (OBB).
"""
import json
import math
import os
import re
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
PRETRAIN = os.path.join(ROOT, "labels_pretrain")
LABELS_FINAL = os.path.join(ROOT, "labels_final")
LABELS_OBB = os.path.join(ROOT, "labels_final_obb")
PREVIEW_FINAL = os.path.join(ROOT, "preview_final")
SPECS = os.path.join(ROOT, "label_specs")
PROGRESS = os.path.join(ROOT, "labeling_progress.csv")
EXPIRY = os.path.join(ROOT, "label_specs_expiry")   # expiry_label per photo, see tools/auto_expiry.py

CLASSES = ["box", "size_label", "expiry_label"]     # expiry_label added 2026-10-08 (printed expiry date)
GREEN = (0, 220, 60)
ORANGE = (255, 140, 0)


def raw_path(name):
    return os.path.join(RAW, name + ".jpg")


def load_raw(name, long_side=None):
    im = Image.open(raw_path(name)).convert("RGB")
    full = im.size
    if long_side:
        s = long_side / max(full)
        im = im.resize((round(full[0] * s), round(full[1] * s)), Image.LANCZOS)
    return im, full


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_ticks(im, step=10):
    """Percent ruler along the top and left edges (unobtrusive)."""
    d = ImageDraw.Draw(im)
    W, H = im.size
    f = font(max(11, W // 90))
    for p in range(step, 100, step):
        x = W * p / 100
        d.line([(x, 0), (x, 10)], fill=(255, 255, 255), width=2)
        d.text((x + 2, 10), str(p), fill=(255, 255, 0), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
        y = H * p / 100
        d.line([(0, y), (10, y)], fill=(255, 255, 255), width=2)
        d.text((12, y - 6), str(p), fill=(255, 255, 0), font=f, stroke_width=2, stroke_fill=(0, 0, 0))


# ---------------------------------------------------------------- rotation geometry

def canvas_geom(W, H, angle):
    r = math.radians(angle)
    c, s = math.cos(r), math.sin(r)
    return abs(W * c) + abs(H * s), abs(W * s) + abs(H * c), c, s


def canvas_to_orig(X, Y, W, H, angle):
    """Rotated-canvas pixel -> original pixel."""
    Wc, Hc, c, s = canvas_geom(W, H, angle)
    dx, dy = X - Wc / 2, Y - Hc / 2
    return W / 2 + dx * c - dy * s, H / 2 + dx * s + dy * c


def orig_to_canvas(x, y, W, H, angle):
    """Original pixel -> rotated-canvas pixel."""
    Wc, Hc, c, s = canvas_geom(W, H, angle)
    dx, dy = x - W / 2, y - H / 2
    return Wc / 2 + dx * c + dy * s, Hc / 2 - dx * s + dy * c


def render_view(img, angle, region=(0, 100, 0, 100), long_side=1400):
    """Render `region` (percent of the rotated canvas) of `img` rotated by `angle`.

    Returns (view, k, X1, Y1, Wc, Hc): view pixel u = (Xcanvas - X1) * k.
    """
    W, H = img.size
    Wc, Hc, c, s = canvas_geom(W, H, angle)
    X1, X2 = region[0] / 100 * Wc, region[1] / 100 * Wc
    Y1, Y2 = region[2] / 100 * Hc, region[3] / 100 * Hc
    k = long_side / max(X2 - X1, Y2 - Y1)
    ow, oh = max(1, round((X2 - X1) * k)), max(1, round((Y2 - Y1) * k))
    data = (c / k, -s / k, W / 2 + (X1 - Wc / 2) * c - (Y1 - Hc / 2) * s,
            s / k, c / k, H / 2 + (X1 - Wc / 2) * s + (Y1 - Hc / 2) * c)
    view = img.transform((ow, oh), Image.AFFINE, data, resample=Image.BICUBIC, fillcolor=(40, 40, 40))
    return view, k, X1, Y1, Wc, Hc


# ---------------------------------------------------------------- specs

def spec_groups(spec):
    if "groups" in spec:
        return spec["groups"]
    return [{"angle": 0, "box": spec.get("box", []), "size_label": spec.get("size_label", [])}]


def load_spec(name):
    return json.load(open(os.path.join(SPECS, name + ".json"), encoding="utf-8"))


def load_expiry(name):
    """label_specs_expiry/NAME.json -> dict, or None when the photo was never checked for expiry dates."""
    p = os.path.join(EXPIRY, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def expiry_polygons(name, W, H):
    """Reviewed (status "selesai") expiry_label polygons in original pixels; [] when not reviewed yet."""
    e = load_expiry(name)
    if not e or e.get("status") != "selesai":
        return []
    return [[(p[i] / 100 * W, p[i + 1] / 100 * H) for i in range(0, 8, 2)]
            for p in (x["poly"] for x in e.get("expiry_label", []))]


def write_spec(name, spec):
    def rows(lst):
        return ",\n".join("    " + json.dumps(r, separators=(", ", ": ")) for r in lst)

    groups = spec_groups(spec)
    parts = []
    for g in groups:
        parts.append('  {\n   "angle": ' + json.dumps(g.get("angle", 0))
                     + ',\n   "box": [\n' + rows(g.get("box", [])) + '\n   ],\n   "size_label": [\n'
                     + rows(g.get("size_label", [])) + '\n   ]\n  }')
    text = ('{\n "groups": [\n' + ",\n".join(parts) + '\n ],\n "status": '
            + json.dumps(spec.get("status", "selesai"), ensure_ascii=False)
            + ',\n "note": ' + json.dumps(spec.get("note", ""), ensure_ascii=False) + '\n}\n')
    with open(os.path.join(SPECS, name + ".json"), "w", encoding="utf-8") as fh:
        fh.write(text)


def rect_polygon(rect_pct, W, H, angle):
    """Canvas-percent rect [x1, x2, y1, y2] -> 4 original-pixel corners (clockwise in canvas)."""
    Wc, Hc, _, _ = canvas_geom(W, H, angle)
    x1, x2, y1, y2 = (rect_pct[0] / 100 * Wc, rect_pct[1] / 100 * Wc,
                      rect_pct[2] / 100 * Hc, rect_pct[3] / 100 * Hc)
    return [canvas_to_orig(X, Y, W, H, angle) for X, Y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]


def canvas_quad_to_orig_pct(e, W, H, a):
    """8 numbers in percent of the canvas rotated by `a` -> 8 numbers in percent of the photo."""
    if not a:
        return list(e)
    Wc, Hc, _, _ = canvas_geom(W, H, a)
    out = []
    for k in (0, 2, 4, 6):
        x, y = canvas_to_orig(e[k] / 100 * Wc, e[k + 1] / 100 * Hc, W, H, a)
        out += [x / W * 100, y / H * 100]
    return out


# ---------------------------------------------------------------- oriented boxes
# An OBB is (A, B, hw): centre-line from point A to point B (original pixels), half-width hw.

def _mid(p, q):
    return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)


def _unit_normal(A, B):
    dx, dy = B[0] - A[0], B[1] - A[1]
    L = math.hypot(dx, dy) or 1.0
    return (dx / L, dy / L), (-dy / L, dx / L), L


def obb_from_rect(rect_pct, W, H, angle):
    p = rect_polygon(rect_pct, W, H, angle)
    Wc, Hc, _, _ = canvas_geom(W, H, angle)
    wpx = (rect_pct[1] - rect_pct[0]) / 100 * Wc
    hpx = (rect_pct[3] - rect_pct[2]) / 100 * Hc
    if wpx >= hpx:
        return _mid(p[0], p[3]), _mid(p[1], p[2]), hpx / 2
    return _mid(p[0], p[1]), _mid(p[3], p[2]), wpx / 2


# Perspective makes a spine a trapezoid (near end wider) whose ends are skewed. The rectangle is
# kept inside the spine: its length stops at the inner corner of each end, and its width is that
# of the narrow end (blend 0); a blend > 0 would widen it towards the wide end and spill over the
# narrow end onto the neighbouring package.
QUAD_WIDTH_BLEND = 0.0


def obb_from_quad(q, W, H, blend=None):
    """[x,y of far-end corner 1, far-end corner 2, near-end corner 1, near-end corner 2] in percent."""
    blend = QUAD_WIDTH_BLEND if blend is None else blend
    p1, p2 = (q[0] / 100 * W, q[1] / 100 * H), (q[2] / 100 * W, q[3] / 100 * H)
    q1, q2 = (q[4] / 100 * W, q[5] / 100 * H), (q[6] / 100 * W, q[7] / 100 * H)
    A, B = _mid(p1, p2), _mid(q1, q2)
    u, n, _ = _unit_normal(A, B)
    along = lambda p: (p[0] - A[0]) * u[0] + (p[1] - A[1]) * u[1]
    across = lambda p, o: abs((p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1])
    hw_a = (across(p1, A) + across(p2, A)) / 2
    hw_b = (across(q1, B) + across(q2, B)) / 2
    lo, hi = min(hw_a, hw_b), max(hw_a, hw_b)
    s0, s1 = max(along(p1), along(p2)), min(along(q1), along(q2))
    A2 = (A[0] + u[0] * s0, A[1] + u[1] * s0)
    B2 = (A[0] + u[0] * s1, A[1] + u[1] * s1)
    return A2, B2, lo + blend * (hi - lo)


def obb_polygon(obb, lo=None, hi=None):
    A, B, hw = obb
    _, n, _ = _unit_normal(A, B)
    lo = -hw if lo is None else lo
    hi = hw if hi is None else hi
    return [(A[0] + n[0] * hi, A[1] + n[1] * hi), (B[0] + n[0] * hi, B[1] + n[1] * hi),
            (B[0] + n[0] * lo, B[1] + n[1] * lo), (A[0] + n[0] * lo, A[1] + n[1] * lo)]


def project_t(obb, pt):
    A, B, _ = obb
    u, _, L = _unit_normal(A, B)
    return ((pt[0] - A[0]) * u[0] + (pt[1] - A[1]) * u[1]) / L


def sub_obb(obb, t1, t2, v1=0.0, v2=1.0):
    """Part of `obb` between fractions t1..t2 along its axis and v1..v2 across it (0..1)."""
    A, B, hw = obb
    t1, t2 = max(0.0, min(t1, t2)), min(1.0, max(t1, t2))
    A2 = (A[0] + (B[0] - A[0]) * t1, A[1] + (B[1] - A[1]) * t1)
    B2 = (A[0] + (B[0] - A[0]) * t2, A[1] + (B[1] - A[1]) * t2)
    return (A2, B2, hw), -hw + 2 * hw * v1, -hw + 2 * hw * v2


def fit_obb_inside(obb, W, H, tol=1.0):
    """Move the ends of an OBB inward until its corners lie inside the photo.

    Returns (obb, cut_fraction_of_length)."""
    A, B, hw = obb
    _, n, L = _unit_normal(A, B)

    def inside(p):
        return -tol <= p[0] <= W + tol and -tol <= p[1] <= H + tol

    def ok(P):
        return inside((P[0] + n[0] * hw, P[1] + n[1] * hw)) and inside((P[0] - n[0] * hw, P[1] - n[1] * hw))

    t1, t2, step = 0.0, 1.0, 0.0005
    at = lambda t: (A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t)
    while t1 < t2 and not ok(at(t1)):
        t1 += step
    while t2 > t1 and not ok(at(t2)):
        t2 -= step
    return (at(t1), at(t2), hw), (t1 + 1 - t2)


def shot_time(name):
    """Seconds since epoch from IMG_YYYYMMDD_HHMMSS_mmm (date included: equal clock times on other days differ)."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})$", name)
    if not m:
        return None
    y, mo, d, h, mi, s, ms = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, s).timestamp() + ms / 1000


def group_bursts(names, burst_s=5.0):
    """Group names into bursts: consecutive shots (by filename timestamp) less than burst_s apart are
    near-duplicates of the same moment and belong together. Names without a parseable timestamp each
    form their own singleton burst. Returns a list of bursts, each a list of names in time order."""
    timed = sorted(names, key=lambda n: (shot_time(n) is None, shot_time(n) or 0, n))
    bursts, last = [], None
    for n in timed:
        t = shot_time(n)
        if bursts and t is not None and last is not None and t - last < burst_s:
            bursts[-1].append(n)
        else:
            bursts.append([n])
        last = t
    return bursts
