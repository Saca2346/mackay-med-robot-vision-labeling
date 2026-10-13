"""Contact sheet of both ends of every box, for checking that each rectangle hugs its spine.

Each tile is centred on one end of a box, rotated so the spine runs vertically: for the first
(far) end the box extends downwards from the tile centre, for the second (near) end upwards.
The box is drawn green, its size_labels orange, other boxes thin grey. Tile title: box number
and end. A tick marks the box's end line so over/under-shoot is easy to see.

Usage: python tools/endcheck.py OUT_DIR NAME [--boxes 1,2,5] [--tile 300] [--span 7]
  --span  tile width in box widths (default 7)
Writes OUT_DIR/NAME_ends.jpg (and _ends2.jpg ... when many boxes).
"""
import math
import os
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GREEN, ORANGE, canvas_geom, canvas_to_orig, font, load_raw, load_spec, obb_polygon
from make_labels import resolve_obbs


def tile(img, P, u, s, T):
    n = (u[1], -u[0])  # right-handed frame: x along n, y along u
    c = (P[0] - T / 2 * s * (n[0] + u[0]), P[1] - T / 2 * s * (n[1] + u[1]))
    data = (s * n[0], s * u[0], c[0], s * n[1], s * u[1], c[1])
    t = img.transform((T, T), Image.AFFINE, data, resample=Image.BICUBIC, fillcolor=(30, 30, 30))

    def to_t(Q):
        dx, dy = Q[0] - P[0], Q[1] - P[1]
        return (T / 2 + (dx * n[0] + dy * n[1]) / s, T / 2 + (dx * u[0] + dy * u[1]) / s)

    return t, to_t


def main(out_dir, name, only=None, T=300, span=7.0):
    spec = load_spec(name)
    img, (W, H) = load_raw(name)
    boxes, sls, _ = resolve_obbs(name, spec, W, H)
    polys = [obb_polygon(b) for b in boxes]
    # which size_labels belong to which box: centre inside box polygon
    owner = []
    for p in sls:
        cx, cy = sum(q[0] for q in p) / 4, sum(q[1] for q in p) / 4
        best = None
        for k, (A, B, hw) in enumerate(boxes):
            L = math.hypot(B[0] - A[0], B[1] - A[1]) or 1
            u = ((B[0] - A[0]) / L, (B[1] - A[1]) / L)
            t = ((cx - A[0]) * u[0] + (cy - A[1]) * u[1]) / L
            d = abs((cx - A[0]) * -u[1] + (cy - A[1]) * u[0])
            if -0.05 <= t <= 1.05 and (best is None or d < best[0]):
                best = (d, k)
        owner.append(best[1] if best else -1)

    # measured corners of 8-number boxes (original pixels), to show how the end was read
    raw = []
    for g in spec.get("groups", []):
        a = g.get("angle", 0)
        Wc, Hc, _, _ = canvas_geom(W, H, a)
        for e in g.get("box", []):
            if len(e) == 8:
                raw.append([canvas_to_orig(e[k] / 100 * Wc, e[k + 1] / 100 * Hc, W, H, a) for k in (0, 2, 4, 6)])
            else:
                raw.append(None)

    idx = [i for i in range(len(boxes)) if only is None or (i + 1) in only]
    tiles = []
    f = font(16)
    for i in idx:
        A, B, hw = boxes[i]
        L = math.hypot(B[0] - A[0], B[1] - A[1])
        u = ((B[0] - A[0]) / L, (B[1] - A[1]) / L)
        s = max(span * 2 * hw, 0.018 * W) / T
        for end, P in (("far", A), ("near", B)):
            t, to_t = tile(img, P, u, s, T)
            d = ImageDraw.Draw(t)
            for j, p in enumerate(polys):
                if j != i:
                    d.polygon([to_t(q) for q in p], outline=(150, 150, 150), width=1)
            for j, p in enumerate(sls):
                if owner[j] == i:
                    d.polygon([to_t(q) for q in p], outline=ORANGE, width=2)
            d.polygon([to_t(q) for q in polys[i]], outline=GREEN, width=2)
            if i < len(raw) and raw[i]:
                corners = raw[i][:2] if end == "far" else raw[i][2:]
                for label, q in zip("12", corners):
                    x, y = to_t(q)
                    d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 0, 255), width=2)
                    d.text((x + 5, y - 7), label, fill=(255, 0, 255), font=font(13), stroke_width=2,
                           stroke_fill=(0, 0, 0))
            # ruler along the spine axis: a tick every 0.25% of the photo side the axis runs along
            unit = 0.0025 * (H if abs(u[1]) >= abs(u[0]) else W) / s
            for k in range(-8, 9):
                y = T / 2 + k * unit
                if 0 <= y <= T:
                    w = 14 if k % 4 == 0 else 7
                    d.line([(T - w, y), (T, y)], fill=(255, 255, 255), width=2 if k == 0 else 1)
                    if k % 4 == 0 and k:
                        d.text((T - 42, y - 8), f"{k * 0.25:+g}", fill=(255, 255, 255), font=font(12),
                               stroke_width=2, stroke_fill=(0, 0, 0))
            d.text((4, 2), f"{i + 1} {end}", fill=(255, 255, 0), font=f, stroke_width=3, stroke_fill=(0, 0, 0))
            tiles.append(t)

    cols = min(6, len(tiles))
    per_sheet = cols * 4
    os.makedirs(out_dir, exist_ok=True)
    for sheet in range(0, len(tiles), per_sheet):
        chunk = tiles[sheet:sheet + per_sheet]
        rows = math.ceil(len(chunk) / cols)
        out = Image.new("RGB", (cols * (T + 4), rows * (T + 4)), (0, 0, 0))
        for k, t in enumerate(chunk):
            out.paste(t, ((k % cols) * (T + 4), (k // cols) * (T + 4)))
        suffix = "" if sheet == 0 else str(sheet // per_sheet + 1)
        p = os.path.join(out_dir, f"{name}_ends{suffix}.jpg")
        out.save(p, quality=90)
        print(p, out.size)


if __name__ == "__main__":
    a = sys.argv[1:]
    only, T, span = None, 300, 7.0
    if "--boxes" in a:
        i = a.index("--boxes")
        only = {int(v) for v in a[i + 1].split(",")}
        del a[i:i + 2]
    if "--tile" in a:
        i = a.index("--tile")
        T = int(a[i + 1])
        del a[i:i + 2]
    if "--span" in a:
        i = a.index("--span")
        span = float(a[i + 1])
        del a[i:i + 2]
    main(a[0], a[1], only, T, span)
