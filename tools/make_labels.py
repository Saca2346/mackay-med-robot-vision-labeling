"""Convert label specs to oriented + axis-aligned YOLO labels, overlay previews and progress log.

label_specs/<NAME>.json:
{
  "groups": [
    {"angle": A, "box": [...], "size_label": [...]},
    ...
  ],
  "status": "selesai" | "perlu dicek manual",
  "note": "free text"
}
box entries:
  [x1, x2, y1, y2]            axis-aligned rect in percent of the photo rotated counter-clockwise
                              by the group's angle A (see tools/grid.py --angle)
  [ax1,ay1, ax2,ay2, bx1,by1, bx2,by2]
                              the two corners of one end and the two corners of the other end of
                              a package spine, in percent of the group's rotated canvas (the
                              original photo when angle is 0); becomes the rectangle along the
                              centre-line (width leaning to the narrow end, see QUAD_WIDTH_BLEND)
size_label entries:
  [x1, x2, y1, y2]            rect in the group's rotated canvas (like a box)
  {"of": N, "from": [x, y], "to": [x, y], "v": [v1, v2]}
                              the part of box N (1-based over all groups, in order) between the
                              projections of two original-percent points on its axis, spanning
                              v1..v2 of its width (default 0.05..0.95); always inside box N
  {"of": N, "t": [t1, t2], "v": [...]}   same with fractions along the axis
  {"of": N, "y": [y1, y2]} / {"of": N, "x": [x1, x2]}
                              same, cut where box N's centre-line reaches these heights / x
                              positions (percent of this group's rotated canvas)
Old specs with top-level "box"/"size_label" are read as one group with angle 0.

Outputs per photo:
  labels_final_obb/NAME.txt  YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized corners)
  labels_final/NAME.txt      YOLO detect: class cx cy w h (box enclosing each oriented box)
  preview_final/NAME_prev.jpg

Usage: python tools/make_labels.py NAME [NAME ...]   (or --all)
"""
import csv
import os
import sys
from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CLASSES, GREEN, ORANGE, LABELS_FINAL, LABELS_OBB, PREVIEW_FINAL, PROGRESS, SPECS,
                    canvas_geom, canvas_to_orig, draw_ticks, expiry_polygons, fit_obb_inside, font, load_raw, load_spec,
                    obb_from_quad, obb_from_rect, obb_polygon, orig_to_canvas, project_t, spec_groups,
                    sub_obb)

FIELDS = ["nama_file", "jumlah_box", "jumlah_size_label", "status", "catatan"]
MAGENTA = (255, 0, 255)         # expiry_label (class 2), only from reviewed label_specs_expiry/NAME.json
DEFAULT_V = [0.05, 0.95]


def canvas_quad_to_orig(e, W, H, a):
    """8 numbers in percent of the canvas rotated by `a` -> 8 numbers in percent of the photo."""
    if not a:
        return list(e)
    Wc, Hc, _, _ = canvas_geom(W, H, a)
    out = []
    for k in (0, 2, 4, 6):
        x, y = canvas_to_orig(e[k] / 100 * Wc, e[k + 1] / 100 * Hc, W, H, a)
        out += [x / W * 100, y / H * 100]
    return out


def resolve_obbs(name, spec, W, H):
    """Return (box OBBs, size_label polygons, warnings) in original pixels."""
    boxes, pending, warnings = [], [], []
    for gi, g in enumerate(spec_groups(spec)):
        a = g.get("angle", 0)
        for i, e in enumerate(g.get("box", []), 1):
            if len(e) == 8:
                obb = obb_from_quad(canvas_quad_to_orig(e, W, H, a), W, H)
            else:
                if e[1] <= e[0] or e[3] <= e[2]:
                    raise ValueError(f"{name} g{gi} box[{i}] degenerate {e}")
                obb = obb_from_rect(e, W, H, a)
            fitted, cut = fit_obb_inside(obb, W, H)
            if cut > 0.01:
                warnings.append(f"g{gi} box[{i}] shortened {cut * 100:.1f}% of its length to stay inside photo")
            boxes.append(fitted)
        for i, e in enumerate(g.get("size_label", []), 1):
            pending.append((gi, i, a, e))
    sls = []
    for gi, i, a, e in pending:
        if isinstance(e, dict):
            n = e["of"]
            if not 1 <= n <= len(boxes):
                raise ValueError(f"{name} g{gi} size_label[{i}] refers to missing box {n}")
            ob = boxes[n - 1]
            # "y"/"x" are read in the frame of the group holding this size_label
            Wc, Hc, _, _ = canvas_geom(W, H, a)
            (Ax, Ay), (Bx, By) = (orig_to_canvas(p[0], p[1], W, H, a) for p in ob[:2])
            if "t" in e:
                t1, t2 = e["t"]
            elif "y" in e:  # heights (percent) where the text cluster starts/ends along the spine
                t1, t2 = ((v / 100 * Hc - Ay) / (By - Ay) for v in e["y"])
            elif "x" in e:
                t1, t2 = ((v / 100 * Wc - Ax) / (Bx - Ax) for v in e["x"])
            else:
                pt = lambda p: (p[0] / 100 * W, p[1] / 100 * H)
                t1, t2 = project_t(ob, pt(e["from"])), project_t(ob, pt(e["to"]))
            v1, v2 = e.get("v", DEFAULT_V)
            sub, lo, hi = sub_obb(ob, t1, t2, v1, v2)
            sls.append(obb_polygon(sub, lo, hi))
        else:
            if e[1] <= e[0] or e[3] <= e[2]:
                raise ValueError(f"{name} g{gi} size_label[{i}] degenerate {e}")
            fitted, cut = fit_obb_inside(obb_from_rect(e, W, H, a), W, H)
            sls.append(obb_polygon(fitted))
    return boxes, sls, warnings


def build(name, spec):
    im, (W, H) = load_raw(name, 1400)
    boxes, sls, warnings = resolve_obbs(name, spec, W, H)
    rows = [(0, obb_polygon(b)) for b in boxes] + [(1, p) for p in sls] \
        + [(2, p) for p in expiry_polygons(name, W, H)]

    with open(os.path.join(LABELS_OBB, name + ".txt"), "w", newline="\n") as fh:
        for cls_id, poly in rows:
            coords = " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}" for x, y in poly)
            fh.write(f"{cls_id} {coords}\n")

    with open(os.path.join(LABELS_FINAL, name + ".txt"), "w", newline="\n") as fh:
        for cls_id, poly in rows:
            xs = [min(max(x, 0), W) for x, _ in poly]
            ys = [min(max(y, 0), H) for _, y in poly]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            fh.write(f"{cls_id} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")

    VW, VH = im.size
    s = VW / W
    draw_ticks(im)
    d = ImageDraw.Draw(im)
    lw = max(2, VW // 450)
    f = font(max(13, VW // 70))
    n_box = 0
    for cls_id, poly in rows:
        pts = [(x * s, y * s) for x, y in poly]
        if cls_id == 0:
            n_box += 1
            d.polygon(pts, outline=GREEN, width=lw)
            tx, ty = min(pts, key=lambda q: (q[1], q[0]))
            tx, ty = min(max(tx, 0), VW - 30), min(max(ty, 0), VH - 20)
            d.text((tx + 2, ty + 1), str(n_box), fill=(0, 0, 0), font=f, stroke_width=3, stroke_fill=GREEN)
        elif cls_id == 1:
            d.polygon(pts, outline=ORANGE, width=lw)
        else:
            d.polygon(pts, outline=MAGENTA, width=lw)
    im.save(os.path.join(PREVIEW_FINAL, name + "_prev.jpg"), quality=88)
    n_sl = sum(1 for c, _ in rows if c == 1)
    return n_box, n_sl, warnings


def update_progress(results):
    existing = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                existing[r["nama_file"]] = r
    existing.update(results)
    with open(PROGRESS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])


def main(names):
    for dname in (LABELS_FINAL, LABELS_OBB, PREVIEW_FINAL):
        os.makedirs(dname, exist_ok=True)
    for dname in (LABELS_FINAL, LABELS_OBB):
        with open(os.path.join(dname, "classes.txt"), "w", newline="\n") as fh:
            fh.write("\n".join(CLASSES) + "\n")
    if names == ["--all"]:
        names = sorted(f[:-5] for f in os.listdir(SPECS) if f.endswith(".json"))
    results = {}
    for n in names:
        s = load_spec(n)
        nb, ns, warns = build(n, s)
        results[n] = {"nama_file": n + ".jpg", "jumlah_box": nb, "jumlah_size_label": ns,
                      "status": s.get("status", "selesai"), "catatan": s.get("note", "")}
        angles = sorted({g.get("angle", 0) for g in spec_groups(s)})
        print(f"{n}: box={nb} size_label={ns} angle={angles} [{results[n]['status']}]")
        for w in warns:
            print("   ! " + w)
    update_progress({v["nama_file"]: v for v in results.values()})


if __name__ == "__main__":
    main(sys.argv[1:])
