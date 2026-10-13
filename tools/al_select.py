"""Active-learning round: rank the unlabeled pool by model uncertainty and write pre-labels for
CVAT / X-AnyLabeling / Label Studio (model-assisted labeling).

Predictions come from either
  --model best.pt   run Ultralytics YOLO-OBB here (pip install ultralytics), or
  --pred-dir DIR    YOLO-OBB txt files written on the training machine by
                    yolo obb predict model=best.pt source=export/yolo_obb/pool.txt conf=0.05 \
                         imgsz=1280 save_txt=True save_conf=True
                    (lines: cls x1 y1 x2 y2 x3 y3 x4 y4 conf, normalized)
The pool is export/yolo_obb/pool.txt (written by tools/export_datasets.py).

Uncertainty per photo (every component is written to ranking.csv):
  amb      mean of 1-|2c-1| over detections with conf >= LOW (1 = coin flip, 0 = certain)
  unsure   share of those detections with conf < KEEP
  orphan   size_labels (conf >= KEEP) whose centre lies in no box (conf >= KEEP)
  overlap  pairs of boxes (conf >= KEEP) overlapping more than 30% of the smaller one
  score    amb + unsure + 0.1 * orphan + 0.1 * overlap   (orphan and overlap capped at 10)
  conf_mean mean confidence of the kept detections (conf >= KEEP); 0 when nothing is detected
Selection (--order conf, default): lowest conf_mean first; (--order al): highest score first.
Photos shot within --min-gap seconds of an already selected photo are skipped first (burst shots
are near-duplicates), then the list is filled up with the next ones.

Outputs (--out, default export/al_round):
  ranking.csv, selected.txt
  pre_labels/yolo_obb/NAME.txt + classes.txt   detections with conf >= KEEP, every pool photo
in --root (default: the project folder), regenerated on every run:
  labels_auto/NAME.txt       draft YOLO-OBB labels (conf >= KEEP) for every pool photo
  preview_auto/NAME_prev.jpg draft overlay (green box, orange size_label, confidence per shape)
  review_priority.csv        lowest mean confidence first, with PERIKSA / OK, cek cepat notes
and the work folder batches/<--batch, default al_round>/ for the selected photos (tools/batch.py):
  images/, labels/ (X-AnyLabeling), cvat_annotations.xml, label_studio_tasks.json (predictions)

With --calib FILE (written by tools/calibrate.py) the draft threshold, NMS IoU, the confidence of
the detections the structure checks look at, and the PERIKSA / OK limits come from the model's
behaviour on photos whose labels are known; an explicit --keep still wins.

Usage: python tools/al_select.py (--model best.pt | --pred-dir DIR) [--calib FILE] [--k 10]
                                 [--keep 0.35] [--imgsz 1280] [--device 0] [--min-gap 4] [--out DIR]
                                 [--batch NAME] [--order conf|al] [--root DIR] [--ls-prefix URL]
"""
import csv
import json
import os
import re
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch import make_batch
from check_specs import clip, poly_area
from common import CLASSES, GREEN, ORANGE, ROOT, font, load_raw
from export_datasets import EXPORT

LOW = 0.15
NO_SL_SHARE = 0.2     # verified photos: share of boxes without size_label p90 0.18, max 0.35
OK_CONF = 0.6         # mean confidence needed for "OK, cek cepat"
RATIO = (0.67, 2.0)   # verified photos: size_label per box always within this range


def opt(args, key, default, cast=str):
    return cast(args[args.index(key) + 1]) if key in args else default


def shot_seconds(name):
    m = re.search(r"_(\d{2})(\d{2})(\d{2})_(\d{3})$", name)
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def preds_from_dir(d, name, W, H):
    rows = []
    p = os.path.join(d, name + ".txt")
    if os.path.exists(p):
        for line in open(p).read().splitlines():
            v = line.split()
            if len(v) == 10:
                xy = [float(t) for t in v[1:9]]
                rows.append((int(float(v[0])), [(xy[k] * W, xy[k + 1] * H) for k in range(0, 8, 2)], float(v[9])))
    return rows


def preds_from_model(model, path, imgsz, iou=0.7, device=None):
    kw = {"device": device} if device not in (None, "") else {}
    r = model.predict(path, imgsz=imgsz, conf=0.05, iou=iou, verbose=False, **kw)[0]
    rows = []
    if r.obb is not None and len(r.obb):
        polys = r.obb.xyxyxyxy.cpu().numpy()
        for poly, c, conf in zip(polys, r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
            rows.append((int(c), [(float(x), float(y)) for x, y in poly], float(conf)))
    return rows


def inside(pt, poly):
    sign = 0
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        cr = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
        if cr:
            if sign == 0:
                sign = 1 if cr > 0 else -1
            elif (cr > 0) != (sign > 0):
                return False
    return True


def centre(p):
    return sum(x for x, _ in p) / 4, sum(y for _, y in p) / 4


def metrics(rows, keep, flag_conf=None):
    """Per-photo statistics. Drafts are the detections with conf >= keep; the structure checks
    only look at confident detections (conf >= flag_conf, default keep), so a stray low-confidence
    shape does not flag a photo: a confident size_label whose centre lies in no draft box
    (orphan), a confident box with no draft size_label inside (box_no_sl), and pairs of confident
    boxes overlapping more than 30% of the smaller one (overlap)."""
    flag_conf = keep if flag_conf is None else max(flag_conf, keep)
    cand = [r for r in rows if r[2] >= LOW]
    amb = sum(1 - abs(2 * c - 1) for _, _, c in cand) / len(cand) if cand else 0.0
    unsure = sum(1 for r in cand if r[2] < keep) / len(cand) if cand else 0.0
    boxes = [p for c, p, s in rows if c == 0 and s >= keep]
    sls = [p for c, p, s in rows if c == 1 and s >= keep]
    boxes_c = [p for c, p, s in rows if c == 0 and s >= flag_conf]
    sls_c = [p for c, p, s in rows if c == 1 and s >= flag_conf]
    orphan = sum(1 for p in sls_c if not any(inside(centre(p), b) for b in boxes))
    overlap = 0
    for i in range(len(boxes_c)):
        for j in range(i + 1, len(boxes_c)):
            inter = poly_area(clip(boxes_c[i], boxes_c[j]))
            if inter > 0.3 * min(poly_area(boxes_c[i]), poly_area(boxes_c[j])):
                overlap += 1
    cents = [centre(p) for p in sls]
    box_no_sl = sum(1 for b in boxes_c if not any(inside(c, b) for c in cents))
    kept = [s for _, _, s in rows if s >= keep]
    return {"n_box": len(boxes), "n_size_label": len(sls), "n_candidates": len(cand),
            "n_box_conf": len(boxes_c), "n_size_label_conf": len(sls_c),
            "amb": round(amb, 4), "unsure": round(unsure, 4), "orphan": orphan, "overlap": overlap,
            "box_no_sl": box_no_sl, "conf_mean": round(sum(kept) / len(kept), 4) if kept else 0.0,
            "score": round(amb + unsure + 0.1 * min(orphan, 10) + 0.1 * min(overlap, 10), 4)}


DEFAULT_TH = {"orphan_max": 0, "overlap_max": 0, "no_sl_share_max": NO_SL_SHARE, "ok_conf": OK_CONF}


def review_note(m, th=None):
    """Flags for review_priority.csv. Without calibration the thresholds come from the verified
    labels (share of boxes without a size_label p90 0.18, size_label/box ratio 0.67..2.0); with
    --calib they are the normal range the model shows on photos whose labels are known."""
    th = {**DEFAULT_TH, **(th or {})}
    notes = []
    if m["n_box"] == 0:
        notes.append("PERIKSA: 0 box terdeteksi")
    elif m["box_no_sl"]:
        n_ref = m.get("n_box_conf") or m["n_box"]
        share = m["box_no_sl"] / n_ref
        if share > max(th["no_sl_share_max"], NO_SL_SHARE):
            notes.append(f"PERIKSA: {m['box_no_sl']} dari {n_ref} box yakin tanpa size_label")
        else:
            notes.append(f"info: {m['box_no_sl']} box tanpa size_label (bisa wajar: terpotong/buram)")
    if m["orphan"] > th["orphan_max"]:
        notes.append(f"PERIKSA: {m['orphan']} size_label yakin di luar box")
    elif m["orphan"]:
        notes.append(f"info: {m['orphan']} size_label di luar box (dalam batas normal model)")
    if m["overlap"] > th["overlap_max"]:
        notes.append(f"PERIKSA: {m['overlap']} pasang box yakin tumpang tindih")
    elif m["overlap"]:
        notes.append(f"info: {m['overlap']} pasang box tumpang tindih (dalam batas normal model)")
    if not any(n.startswith("PERIKSA") for n in notes):
        ratio = m["n_size_label"] / m["n_box"]
        if m["conf_mean"] >= th["ok_conf"] and RATIO[0] <= ratio <= RATIO[1]:
            notes.insert(0, "OK, cek cepat")
        elif not RATIO[0] <= ratio <= RATIO[1]:
            notes.insert(0, f"cek normal: rasio size_label/box {ratio:.2f} tidak wajar")
        else:
            notes.insert(0, "cek normal: confidence sedang")
    return "; ".join(notes)


def write_review(table, preds, keep, root):
    """labels_auto/, preview_auto/ and review_priority.csv (lowest confidence first)."""
    la, pa = os.path.join(root, "labels_auto"), os.path.join(root, "preview_auto")
    for d, suffix in ((la, ".txt"), (pa, "_prev.jpg")):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(suffix):
                os.remove(os.path.join(d, f))
    with open(os.path.join(la, "classes.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(CLASSES) + "\n")
    f = font(22)
    for r in table:
        name = r["nama_file"][:-4]
        W, H, rows = preds[name]
        kept = [x for x in rows if x[2] >= keep]
        with open(os.path.join(la, name + ".txt"), "w", newline="\n") as fh:
            for c, p, _ in kept:
                fh.write(f"{c} " + " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}"
                                            for x, y in p) + "\n")
        im, _ = load_raw(name, 1400)
        s = im.size[0] / W
        d = ImageDraw.Draw(im)
        for c, p, conf in sorted(kept, key=lambda x: x[0]):
            pts = [(x * s, y * s) for x, y in p]
            d.polygon(pts, outline=GREEN if c == 0 else ORANGE, width=3)
            tx, ty = min(pts, key=lambda q: (q[1], q[0]))
            d.text((tx + 2, ty + 1), f"{conf:.2f}", fill=(255, 255, 255), font=font(14),
                   stroke_width=2, stroke_fill=(0, 0, 0))
        d.text((8, 6), f"{r['n_box']} box, {r['n_size_label']} size_label, conf rata2 {r['conf_mean']:.2f}"
                       f" | {r['catatan']}", fill=(255, 255, 0), font=f, stroke_width=3, stroke_fill=(0, 0, 0))
        im.save(os.path.join(pa, name + "_prev.jpg"), quality=88)
    order = sorted(table, key=lambda r: (r["conf_mean"], r["nama_file"]))
    with open(os.path.join(root, "review_priority.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["urutan", "nama_file", "jumlah_box", "jumlah_size_label", "confidence_rata2", "catatan",
                    "masuk_batch"])
        for i, r in enumerate(order, 1):
            w.writerow([i, r["nama_file"], r["n_box"], r["n_size_label"], f"{r['conf_mean']:.3f}", r["catatan"],
                        "ya" if r["selected"] else ""])
    return order


def main(args):
    model_path, pred_dir = opt(args, "--model", None), opt(args, "--pred-dir", None)
    if not (model_path or pred_dir):
        raise SystemExit(__doc__)
    calib = {}
    if "--calib" in args:
        with open(opt(args, "--calib", None)) as fh:
            calib = json.load(fh)
    k = opt(args, "--k", 10, int)
    keep = opt(args, "--keep", calib.get("keep", 0.35), float)        # an explicit --keep wins
    flag_conf = calib.get("flag_conf", keep)
    nms_iou, device = calib.get("nms_iou", 0.7), opt(args, "--device", None)
    th = {key: calib[key] for key in ("orphan_max", "overlap_max", "no_sl_share_max", "ok_conf") if key in calib}
    imgsz, gap = opt(args, "--imgsz", 1280, int), opt(args, "--min-gap", 4.0, float)
    out = opt(args, "--out", os.path.join(EXPORT, "al_round"))
    ls_prefix = opt(args, "--ls-prefix", "/data/local-files/?d=raw/")
    batch = opt(args, "--batch", "al_round")
    order = opt(args, "--order", "conf")
    root = opt(args, "--root", ROOT)
    if order not in ("conf", "al"):
        raise SystemExit("--order must be conf or al")

    with open(os.path.join(EXPORT, "yolo_obb", "pool.txt")) as fh:
        pool = [ln.strip() for ln in fh if ln.strip()]
    if "--list" in args:                              # hanya foto pool yang ada di daftar ini (satu nama per baris)
        with open(opt(args, "--list", None)) as fh:
            wanted = {os.path.splitext(ln.strip())[0] for ln in fh if ln.strip()}
        pool = [p for p in pool if os.path.splitext(os.path.basename(p))[0] in wanted]
        print(f"--list: {len(pool)} foto pool dipakai")
    model = None
    if model_path:
        from ultralytics import YOLO  # only needed for --model
        model = YOLO(model_path)

    pre = os.path.join(out, "pre_labels", "yolo_obb")
    os.makedirs(pre, exist_ok=True)
    with open(os.path.join(pre, "classes.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(CLASSES) + "\n")

    table, preds = [], {}
    for path in pool:
        name = os.path.splitext(os.path.basename(path))[0]
        W, H = Image.open(path).size
        rows = (preds_from_model(model, path, imgsz, nms_iou, device) if model
                else preds_from_dir(pred_dir, name, W, H))
        preds[name] = (W, H, rows)
        table.append({"nama_file": name + ".jpg", **metrics(rows, keep, flag_conf)})
        with open(os.path.join(pre, name + ".txt"), "w", newline="\n") as fh:
            for c, p, s in rows:
                if s >= keep:
                    fh.write(f"{c} " + " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}"
                                                for x, y in p) + "\n")

    for r in table:
        r["catatan"] = review_note(r, th)
    if calib:
        print(f"kalibrasi: keep {keep}, flag_conf {flag_conf}, NMS IoU {nms_iou}, batas {th}")
    if order == "al":
        table.sort(key=lambda r: -r["score"])
    else:                                             # lowest mean confidence first (0 detections = 0)
        table.sort(key=lambda r: (r["conf_mean"], r["nama_file"]))
    chosen = []
    for r in table:                                   # diverse first pass
        t = shot_seconds(r["nama_file"][:-4])
        if len(chosen) < k and all(t is None or c[1] is None or abs(t - c[1]) > gap for c in chosen):
            chosen.append((r["nama_file"], t))
    for r in table:                                   # fill up
        if len(chosen) >= k:
            break
        if r["nama_file"] not in {c[0] for c in chosen}:
            chosen.append((r["nama_file"], None))
    picked = {c[0] for c in chosen}
    for rank, r in enumerate(table, 1):
        r["rank"] = rank
        r["selected"] = int(r["nama_file"] in picked)

    with open(os.path.join(out, "ranking.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "selected", "nama_file", "conf_mean", "score", "amb", "unsure",
                                           "orphan", "overlap", "box_no_sl", "n_box", "n_size_label",
                                           "n_box_conf", "n_size_label_conf", "n_candidates", "catatan"])
        w.writeheader()
        w.writerows(table)
    sel = [r["nama_file"][:-4] for r in table if r["selected"]]
    with open(os.path.join(out, "selected.txt"), "w", newline="\n") as fh:
        fh.write("".join(n + ".jpg\n" for n in sel))

    kept = {n: [r for r in preds[n][2] if r[2] >= keep] for n in sel}
    bdir, _ = make_batch(batch, sel, predictions=kept, ls_prefix=ls_prefix,
                         model_version=os.path.basename(model_path or pred_dir))

    write_review(table, preds, keep, root)
    n_ok = sum(1 for r in table if r["catatan"].startswith("OK"))
    n_chk = sum(1 for r in table if "PERIKSA" in r["catatan"])
    print(f"pool {len(pool)} photos ranked ({order}); selected {len(sel)} for the next labeling round -> {out}")
    print(f"review_priority.csv: {n_chk} PERIKSA, {n_ok} OK cek cepat, {len(table) - n_chk - n_ok} cek normal")
    print(f"drafts: {os.path.join(root, 'labels_auto')}  previews: {os.path.join(root, 'preview_auto')}")
    print(f"work folder for CVAT / X-AnyLabeling / Label Studio: {bdir}")
    for r in table[:k + 5]:
        print(f"  {r['rank']:3d} {'*' if r['selected'] else ' '} {r['nama_file']}  conf={r['conf_mean']:.2f} "
              f"score={r['score']:.3f} box={r['n_box']} sl={r['n_size_label']} | {r['catatan']}")


if __name__ == "__main__":
    main(sys.argv[1:])
