"""Calibrate the draft and flag thresholds of a bootstrap model on photos whose labels are known.

Runs the model on the seed photos of --split (default val, never trained on), matches predictions to
the verified labels (same class, rotated-box IoU >= 0.5, greedy by confidence) and picks
  nms_iou    NMS IoU with the best F1: 0.7 (Ultralytics default) or 0.5 (fewer duplicate boxes)
  flag_conf  confidence with the best F1 over both classes; detections at or above it are "confident"
             and are the only ones the PERIKSA checks look at
  keep       draft threshold: the highest confidence (<= flag_conf) whose recall is still
             >= --draft-recall (default 0.95), so drafts rather have a box too many than one too few;
             0.15 (or flag_conf if lower) when that recall is never reached
and, from the structure of the confident detections on these well-labeled photos, the normal range
used by tools/al_select.py --calib:
  orphan_max, overlap_max, no_sl_share_max   largest value seen (PERIKSA above it)
  ok_conf                                    median mean confidence (OK needs at least this)
Writes the JSON given by --out (default logs/calibration.json) and prints the P/R/F1 table.

Usage: python tools/calibrate.py --model best.pt [--imgsz 1280] [--split val] [--device 0]
                                 [--draft-recall 0.95] [--out FILE]
"""
import json
import os
import statistics
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al_select import metrics, opt, preds_from_model
from check_specs import clip, poly_area
from common import CLASSES, ROOT
from export_datasets import EXPORT

IOU_MATCH = 0.5
SWEEP = [round(0.05 * i, 2) for i in range(1, 19)]      # 0.05 .. 0.90


def iou(a, b):
    inter = poly_area(clip(a, b))
    union = poly_area(a) + poly_area(b) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path, W, H):
    rows = []
    for line in open(path).read().splitlines():
        v = line.split()
        if len(v) == 9:
            xy = [float(t) for t in v[1:]]
            rows.append((int(v[0]), [(xy[k] * W, xy[k + 1] * H) for k in range(0, 8, 2)]))
    return rows


def match(preds, gt):
    """[(cls, conf, is_tp)] for every prediction; greedy by confidence, one GT per prediction."""
    used, out = set(), []
    for c, poly, conf in sorted(preds, key=lambda r: -r[2]):
        best, bi = 0.0, None
        for i, (gc, gpoly) in enumerate(gt):
            if gc == c and i not in used:
                v = iou(poly, gpoly)
                if v > best:
                    best, bi = v, i
        tp = best >= IOU_MATCH
        if tp:
            used.add(bi)
        out.append((c, conf, tp))
    return out


def prf(matches, n_gt, t, cls=None):
    tp = sum(1 for c, s, ok in matches if s >= t and ok and (cls is None or c == cls))
    fp = sum(1 for c, s, ok in matches if s >= t and not ok and (cls is None or c == cls))
    n = n_gt if cls is None else n_gt[cls] if isinstance(n_gt, dict) else n_gt
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n if n else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main(args):
    model_path = opt(args, "--model", None)
    if not model_path:
        raise SystemExit(__doc__)
    imgsz, split = opt(args, "--imgsz", 1280, int), opt(args, "--split", "val")
    device, target = opt(args, "--device", None), opt(args, "--draft-recall", 0.95, float)
    out = opt(args, "--out", os.path.join(ROOT, "logs", "calibration.json"))
    from ultralytics import YOLO
    model = YOLO(model_path)
    img_dir = os.path.join(EXPORT, "yolo_obb", "images", split)
    lab_dir = os.path.join(EXPORT, "yolo_obb", "labels", split)
    names = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg"))
    if not names:
        raise SystemExit(f"tidak ada foto di {img_dir} - jalankan tools/export_datasets.py dulu")

    gts, results = {}, {}
    for n in names:
        W, H = Image.open(os.path.join(img_dir, n + ".jpg")).size
        gts[n] = load_gt(os.path.join(lab_dir, n + ".txt"), W, H)
    n_gt = {c: sum(1 for n in names for g, _ in gts[n] if g == c) for c in range(len(CLASSES))}
    n_all = sum(n_gt.values())

    best = None
    for nms in (0.7, 0.5):
        preds = {n: preds_from_model(model, os.path.join(img_dir, n + ".jpg"), imgsz, nms, device) for n in names}
        allm = [m for n in names for m in match(preds[n], gts[n])]
        table = {t: prf(allm, n_all, t) for t in SWEEP}
        t_f1 = max(SWEEP, key=lambda t: (round(table[t][2], 4), t))
        results[nms] = (preds, allm, table, t_f1)
        if best is None or table[t_f1][2] > results[best][2][results[best][3]][2] + 1e-9:
            best = nms
    preds, allm, table, flag_conf = results[best]
    ok_t = [t for t in SWEEP if t <= flag_conf and table[t][1] >= target]
    keep = max(ok_t) if ok_t else min(0.15, flag_conf)

    stats = [metrics(preds[n], keep, flag_conf) for n in names]
    share = [s["box_no_sl"] / s["n_box"] for s in stats if s["n_box"]]
    calib = {
        "model": os.path.abspath(model_path), "split": split, "photos": len(names),
        "objects": {CLASSES[c]: v for c, v in n_gt.items()},
        "nms_iou": best, "flag_conf": flag_conf, "keep": keep, "draft_recall_target": target,
        "orphan_max": max(s["orphan"] for s in stats), "overlap_max": max(s["overlap"] for s in stats),
        "no_sl_share_max": round(max(share), 3) if share else 0.2,
        "ok_conf": round(statistics.median(s["conf_mean"] for s in stats), 4),
        "at_keep": dict(zip(("P", "R", "F1"), (round(v, 3) for v in table[keep]))),
        "at_flag_conf": dict(zip(("P", "R", "F1"), (round(v, 3) for v in table[flag_conf]))),
        "per_class_at_keep": {CLASSES[c]: dict(zip(("P", "R", "F1"), (round(v, 3) for v in prf(allm, n_gt, keep, c))))
                              for c in range(len(CLASSES))},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(calib, open(out, "w"), indent=1)

    print(f"kalibrasi pada {len(names)} foto {split} ({n_all} objek), IoU cocok >= {IOU_MATCH}")
    for nms in (0.7, 0.5):
        _, _, tb, tf = results[nms]
        print(f"  NMS IoU {nms}: F1 terbaik {tb[tf][2]:.3f} pada conf {tf}" + ("  <- dipakai" if nms == best else ""))
    print("  conf   P      R      F1")
    for t in SWEEP:
        mark = " <- draft (keep)" if t == keep else " <- confident (flag_conf)" if t == flag_conf else ""
        print(f"  {t:4.2f}  {table[t][0]:.3f}  {table[t][1]:.3f}  {table[t][2]:.3f}{mark}")
    print(f"batas penanda: orphan>{calib['orphan_max']}, tumpang tindih>{calib['overlap_max']}, "
          f"porsi box tanpa size_label>{calib['no_sl_share_max']}, OK bila conf rata2>={calib['ok_conf']}")
    print("ditulis:", out)


if __name__ == "__main__":
    main(sys.argv[1:])
