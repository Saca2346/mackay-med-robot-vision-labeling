"""Working view: photo rotated by --angle, with a percent grid of the rotated canvas.

Usage: python tools/grid.py OUT_DIR NAME [NAME ...] [--angle A] [--crop x1,x2,y1,y2] [--final]
  --angle  rotate the photo counter-clockwise by A degrees (canvas expands); grid is in percent
           of that rotated canvas, i.e. the coordinates used by a spec group with the same angle
  --crop   render only this region of the rotated canvas (percent), with a 1% grid
  --final  overlay labels_final_obb boxes (green=box with number, orange=size_label)
"""
import os
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GREEN, ORANGE, LABELS_OBB, font, load_raw, orig_to_canvas, render_view


def load_obb(name, W, H):
    p = os.path.join(LABELS_OBB, name + ".txt")
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p).read().splitlines():
        parts = line.split()
        if len(parts) != 9:
            continue
        v = list(map(float, parts[1:]))
        out.append((int(parts[0]), [(v[i] * W, v[i + 1] * H) for i in range(0, 8, 2)]))
    return out


def render(name, out_dir, angle=0.0, crop=None, final=False, long_side=1400):
    img, _ = load_raw(name)
    W, H = img.size
    region = crop or (0, 100, 0, 100)
    view, k, X1, Y1, Wc, Hc = render_view(img, angle, region, long_side)
    VW, VH = view.size
    x0, x1, y0, y1 = region

    def to_view_pct(px, py):
        return (px - x0) / (x1 - x0) * VW, (py - y0) / (y1 - y0) * VH

    ov = view.copy()
    d = ImageDraw.Draw(ov)
    f = font(max(12, VW // 80))
    step = 1 if crop else 5
    major = 5 if crop else 10
    p = 0
    while p <= 100:
        is_major = p % major == 0
        col = (255, 255, 0) if is_major else (0, 255, 255)
        if x0 <= p <= x1:
            x, _ = to_view_pct(p, y0)
            d.line([(x, 0), (x, VH)], fill=col, width=1)
            if is_major:
                d.text((x + 2, 2), f"{p:g}", fill=col, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
        if y0 <= p <= y1:
            _, y = to_view_pct(x0, p)
            d.line([(0, y), (VW, y)], fill=col, width=1)
            if is_major:
                d.text((2, y + 2), f"{p:g}", fill=col, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
        p += step
    view = Image.blend(view, ov, 0.55)

    if final:
        d = ImageDraw.Draw(view)
        n_box = 0
        for c, poly in load_obb(name, W, H):
            pts = []
            for (x, y) in poly:
                X, Y = orig_to_canvas(x, y, W, H, angle)
                pts.append(((X - X1) * k, (Y - Y1) * k))
            d.polygon(pts, outline=GREEN if c == 0 else ORANGE, width=3)
            if c == 0:
                n_box += 1
                tx, ty = min(pts, key=lambda q: (q[1], q[0]))
                d.text((tx + 4, ty + 3), str(n_box), fill=(0, 0, 0), font=f, stroke_width=3, stroke_fill=GREEN)

    tag = f"_a{angle:g}" if angle else ""
    suffix = ("_crop" if crop else "_grid") + tag + ("_final" if final else "")
    out = os.path.join(out_dir, name + suffix + ".jpg")
    view.save(out, quality=88)
    print(out, view.size)


if __name__ == "__main__":
    args = sys.argv[1:]
    angle, crop = 0.0, None
    if "--angle" in args:
        i = args.index("--angle")
        angle = float(args[i + 1])
        del args[i:i + 2]
    if "--crop" in args:
        i = args.index("--crop")
        crop = tuple(float(v) for v in args[i + 1].split(","))
        del args[i:i + 2]
    ocrop = None
    if "--ocrop" in args:  # region in ORIGINAL photo percent; converted to the rotated canvas
        i = args.index("--ocrop")
        ocrop = tuple(float(v) for v in args[i + 1].split(","))
        del args[i:i + 2]
    final = "--final" in args
    args = [a for a in args if a not in ("--final", "--no-pre")]
    out_dir = args[0]
    os.makedirs(out_dir, exist_ok=True)
    for n in args[1:]:
        if ocrop:
            from common import canvas_geom
            _, (W, H) = load_raw(n, 64)
            Wc, Hc, _, _ = canvas_geom(W, H, angle)
            pts = [orig_to_canvas(x / 100 * W, y / 100 * H, W, H, angle)
                   for x in ocrop[:2] for y in ocrop[2:]]
            crop = (min(p[0] for p in pts) / Wc * 100, max(p[0] for p in pts) / Wc * 100,
                    min(p[1] for p in pts) / Hc * 100, max(p[1] for p in pts) / Hc * 100)
            crop = tuple(round(v, 1) for v in crop)
            print("canvas crop", crop)
        render(n, out_dir, angle, crop, final)
