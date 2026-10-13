"""Train the best detector the labeled photos allow: choose the configuration by cross-validation, then train
the final model on every labeled photo and calibrate its per-class confidence thresholds.

1. Every seed photo (verified labels, see export_datasets.seed_state) is put in one of K folds by burst: photos
   taken less than BURST_S seconds apart are near-duplicates and always land in the same fold, so no fold is
   validated on a near-copy of its training photos. A burst longer than MAX_BURST (a whole session shot
   continuously) is cut into runs of consecutive shots, so one session cannot fill a fold on its own.
2. Each candidate "MODEL@DEGREES" (default yolo11s-obb with rotation augmentation +-180 and +-10 degrees) is
   trained K times, one fold held out, with "bad camera" augmentations (BadCamera: low resolution, blur, JPEG,
   noise, lighting) and evaluated on the held-out fold twice: as photographed, and degraded to a 640 px wide,
   blurred, JPEG-60 webcam-like image. Full rotation matters: in the 90-photo CV (2026-09-27) the fold with
   lying boxes scored 0.51 against 0.64-0.80 for the others, with only +-10 degrees.
3. The candidate with the best mean mAP50-95 over (clean + degraded) wins; candidates within --tie (0.01) of the
   best are a draw and the first listed one wins (yolo11s vs yolo11m differed by 0.001 while yolo11m is 2.4x
   slower on the laptop). Its out-of-fold predictions give the per-class confidence threshold with the best F1.
4. The winner is trained on ALL seed photos -> runs/best_final/weights/best.pt, summary in logs/best_model.json.
Finished runs are skipped when the script is started again (same dataset); a changed dataset starts a new CV.

Usage (Ubuntu GPU server, venv with ultralytics):
  python tools/train_best.py                          # 5 folds, 200 epochs CV, 300 epochs final
  python tools/train_best.py --quick                  # 3 folds, 120 epochs CV, 200 final (about 3x faster)
  python tools/train_best.py --models yolo11s-obb.pt@180,yolo11s-obb.pt@10,yolo11m-obb.pt@180
  options: --folds K --epochs E --final-epochs E --imgsz 1280 --batch -1 --device 0 --workers 8 --tie 0.01
           --no-final
"""
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al_select import opt, preds_from_model  # noqa: E402
from calibrate import load_gt, match, prf  # noqa: E402
from common import CLASSES, RAW, ROOT, group_bursts, shot_time  # noqa: E402
from export_datasets import drop_duplicates, label_dir, link_or_copy, seed_state  # noqa: E402

BURST_S = 5.0
MAX_BURST = 8                  # longest run of consecutive shots kept in one fold
CUT_SEARCH = 2                 # when cutting a long burst, look this many slots either side of the equal-size
                                # cut point for the locally largest time gap, so the cut lands at the weakest link
SPLIT = f"burst{BURST_S:g}s-max{MAX_BURST}-search{CUT_SEARCH}"
CV_DIR = os.path.join(ROOT, "export", "cv")
LOGS = os.path.join(ROOT, "logs")
RESULTS = os.path.join(LOGS, "cv_results.json")
SWEEP = [round(0.05 * i, 2) for i in range(1, 19)]


class BadCamera:
    """Webcam-like degradation of a training image, pixels only (boxes unchanged), each part independently:
    low resolution (downscale 0.25-0.6x and back), Gaussian / motion blur, JPEG 35-90, sensor noise, lighting.
    Written without albumentations on purpose: installing it pulls opencv-python-headless, which can replace the
    GUI OpenCV the webcam window needs in the same venv."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, labels):
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        if random.random() < 0.5:
            s = random.uniform(0.25, 0.6)
            small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if random.random() < 0.3:
            k = random.choice((3, 5, 7))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.3:
            k = random.choice((3, 5, 7, 9))
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1.0 / k
            m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1.0)
            img = cv2.filter2D(img, -1, cv2.warpAffine(kern, m, (k, k)))
        if random.random() < 0.4:
            a, b = random.uniform(0.75, 1.25), random.uniform(-25, 25)
            img = cv2.convertScaleAbs(img, alpha=a, beta=b)
        if random.random() < 0.3:
            img = np.clip(img + np.random.normal(0, random.uniform(2, 12), img.shape), 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 90)])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        labels["img"] = np.ascontiguousarray(img)
        return labels


def bad_camera_trainer():
    """OBB trainer whose training dataset applies BadCamera just before the tensors are made."""
    from ultralytics.models.yolo.obb import OBBTrainer

    class BadCameraTrainer(OBBTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                t = ds.transforms.transforms
                i = next((j for j, x in enumerate(t) if type(x).__name__ == "Format"), len(t))
                t.insert(i, BadCamera())
            return ds

    return BadCameraTrainer


def split_candidate(cand):
    """'yolo11s-obb.pt@180' -> ('yolo11s-obb.pt', 180); no '@' -> rotation augmentation of 10 degrees."""
    model, _, deg = cand.partition("@")
    return model, float(deg) if deg else 10.0


def short(cand):
    """'yolo11s-obb.pt' / '/path/yolo11s-obb.pt@180' -> 'yolo11s-obb' / 'yolo11s-obb@180' (run and result key)."""
    model, _, deg = cand.partition("@")
    return os.path.splitext(os.path.basename(model))[0] + (f"@{deg}" if deg else "")


def seed_photos():
    names = sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".jpg"))
    names, _ = drop_duplicates(names)
    return [n for n in names if seed_state(n)[0] == "seed"]


def fingerprint(names):
    """Identity of the labeled dataset: changes if a name, a label file's bytes, or a photo's bytes change
    (so re-exporting the same labels over a replaced/re-rotated photo starts a fresh CV, not a stale one)."""
    h = hashlib.sha1(SPLIT.encode())                       # another fold rule = another CV (no reuse of old runs)
    for n in names:
        h.update(n.encode())
        h.update(open(os.path.join(label_dir(n), n + ".txt"), "rb").read())
        h.update(open(os.path.join(RAW, n + ".jpg"), "rb").read())
    return h.hexdigest()[:16]


def split_burst(b, parts):
    """Cut a time-ordered run of consecutive shots into `parts` pieces. Each cut is placed near its
    equal-size position but nudged within +-CUT_SEARCH slots to the locally largest time gap, so a cut
    rarely falls between two near-duplicate shots instead of at the weakest link in the chain."""
    if parts <= 1:
        return [b]
    times = [shot_time(n) for n in b]
    n = len(b)
    cuts = []
    for i in range(1, parts):
        ideal = round(n * i / parts)
        lo = max(1, ideal - CUT_SEARCH, (cuts[-1] + 1) if cuts else 1)
        hi = min(n - 1, ideal + CUT_SEARCH)
        if lo > hi:
            lo = hi
        cuts.append(max(range(lo, hi + 1), key=lambda j: times[j] - times[j - 1]))
    runs, start = [], 0
    for c in cuts:
        runs.append(b[start:c])
        start = c
    runs.append(b[start:])
    return runs


def make_folds(names, k):
    """Bursts (shots < BURST_S apart) -> folds, largest bursts first into the fold with fewest photos.
    A continuous session shot every 1-3 s chains into one huge burst (2026-09-28: 53 photos, all in one fold), so
    bursts longer than MAX_BURST are cut into runs of consecutive shots near equal size, each cut nudged to the
    locally largest time gap (see split_burst) so a cut rarely separates two near-duplicate shots."""
    bursts = group_bursts(names, BURST_S)
    runs = []
    for b in bursts:
        parts = -(-len(b) // MAX_BURST)
        runs += split_burst(b, parts)
    folds = [[] for _ in range(k)]
    for b in sorted(runs, key=lambda b: (-len(b), b[0])):
        min(folds, key=len).extend(b)
    return [sorted(f) for f in folds], len(runs)


def degrade(src, dst):
    """Webcam-like copy: 640 px wide, slight blur, JPEG quality 60 (labels are normalized, so unchanged)."""
    im = cv2.imread(src)
    s = 640 / im.shape[1]
    im = cv2.resize(im, (640, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    im = cv2.GaussianBlur(im, (3, 3), 0.8)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 60])


def write_dataset(root, train, val, deg=True):
    for sub in ("images/train", "images/val", "images/val_deg", "labels/train", "labels/val", "labels/val_deg"):
        shutil.rmtree(os.path.join(root, sub), ignore_errors=True)
        os.makedirs(os.path.join(root, sub))
    for split, names in (("train", train), ("val", val)):
        for n in names:
            link_or_copy(os.path.join(RAW, n + ".jpg"), os.path.join(root, "images", split, n + ".jpg"), False)
            shutil.copy2(os.path.join(label_dir(n), n + ".txt"), os.path.join(root, "labels", split, n + ".txt"))
    if deg:
        for n in val:
            degrade(os.path.join(RAW, n + ".jpg"), os.path.join(root, "images", "val_deg", n + ".jpg"))
            shutil.copy2(os.path.join(label_dir(n), n + ".txt"), os.path.join(root, "labels", "val_deg", n + ".txt"))
    names = "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    base = root.replace(os.sep, "/")
    for fname, val_dir in (("data.yaml", "images/val"), ("data_deg.yaml", "images/val_deg")):
        with open(os.path.join(root, fname), "w", newline="\n") as fh:
            fh.write(f"path: {base}\ntrain: images/train\nval: {val_dir}\n{names}")


def metrics_of(r):
    out = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4), "per_class": {}}
    for j, c in enumerate(r.ap_class_index):
        p, rc, ap50, ap = r.class_result(j)
        out["per_class"][r.names[int(c)]] = {"P": round(float(p), 3), "R": round(float(rc), 3),
                                             "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    return out


def train(cand, data, run_dir, name, epochs, a, tag=None):
    """A finished run (weights + "done") is reused. With `tag` (candidate + dataset), only when its "done" file holds
    the same tag: runs/best_final keeps its name, so a new winner or new photos must retrain it."""
    from ultralytics import YOLO
    weights = os.path.join(run_dir, name, "weights", "best.pt")
    done = os.path.join(run_dir, name, "done")
    if os.path.exists(weights) and os.path.exists(done) and (tag is None or open(done).read().strip() == tag):
        return weights
    model_name, degrees = split_candidate(cand)
    YOLO(model_name).train(trainer=bad_camera_trainer(), data=data, imgsz=a["imgsz"], epochs=epochs,
                           batch=a["batch"], device=a["device"], workers=a["workers"], degrees=degrees, fliplr=0.0,
                           patience=max(50, epochs // 2), project=run_dir, name=name, exist_ok=True, plots=False,
                           seed=0, verbose=False)
    with open(done, "w") as fh:
        fh.write(tag or "")
    return weights


def oof_predictions(weights, val, a):
    """Predictions (conf >= 0.05, NMS 0.5) on held-out photos, as calibrate.match input."""
    from ultralytics import YOLO
    model = YOLO(weights)
    out = {}
    for n in val:
        img = cv2.imread(os.path.join(RAW, n + ".jpg"))
        H, W = img.shape[:2]
        preds = preds_from_model(model, os.path.join(RAW, n + ".jpg"), a["imgsz"], 0.5, a["device"])
        gt = load_gt(os.path.join(label_dir(n), n + ".txt"), W, H)
        out[n] = [(c, s, ok) for c, s, ok in match(preds, gt)], {c: sum(1 for g, _ in gt if g == c) for c in range(len(CLASSES))}
    return out


def best_thresholds(oof):
    """Per-class confidence with the best F1 over all out-of-fold photos."""
    res = {}
    for c, cname in enumerate(CLASSES):
        m = [(cc, s, ok) for n in oof for cc, s, ok in oof[n][0] if cc == c]
        n_gt = sum(oof[n][1][c] for n in oof)
        table = {t: prf(m, n_gt, t) for t in SWEEP}
        t = max(SWEEP, key=lambda t: (round(table[t][2], 4), t))
        if table[t][2] == 0:                      # nothing detected correctly (e.g. a test run): keep a default
            t = 0.25
        res[cname] = {"conf": t, "P": round(table[t][0], 3), "R": round(table[t][1], 3), "F1": round(table[t][2], 3)}
    return res


def main(args):
    quick = "--quick" in args
    a = {"imgsz": opt(args, "--imgsz", 1280, int), "batch": opt(args, "--batch", -1, int),
         "device": opt(args, "--device", "0"), "workers": opt(args, "--workers", 8, int)}
    k = opt(args, "--folds", 3 if quick else 5, int)
    epochs = opt(args, "--epochs", 120 if quick else 200, int)
    final_epochs = opt(args, "--final-epochs", 200 if quick else 300, int)
    models = opt(args, "--models", "yolo11s-obb.pt@180,yolo11s-obb.pt@10").split(",")
    tie = opt(args, "--tie", 0.01, float)
    os.makedirs(LOGS, exist_ok=True)

    names = seed_photos()
    fp = fingerprint(names)
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    if results.get("fingerprint") not in (None, fp) or results.get("folds") not in (None, k):
        old = RESULTS.replace(".json", time.strftime("_%Y%m%d_%H%M%S.json"))
        os.replace(RESULTS, old)
        print(f"dataset / jumlah fold berubah -> CV baru (hasil lama: {old})")
        shutil.rmtree(os.path.join(ROOT, "runs", "cv"), ignore_errors=True)
        results = {}
    results.update({"fingerprint": fp, "folds": k, "photos": len(names), "imgsz": a["imgsz"], "epochs": epochs})
    results.setdefault("runs", {})
    folds, n_bursts = make_folds(names, k)
    print(f"{len(names)} foto berlabel, {n_bursts} burst -> {k} fold: {[len(f) for f in folds]} foto")

    run_tag = f"i{a['imgsz']}_e{epochs}"  # imgsz/epochs change the result, so they're part of the cache key
    run_dir = os.path.join(ROOT, "runs", "cv")
    t_start = time.time()
    for fi, val in enumerate(folds):
        train_names = [n for f in folds if f is not val for n in f]
        root = os.path.join(CV_DIR, f"fold{fi}")
        if not os.path.exists(os.path.join(root, "data_deg.yaml")) or \
                sorted(os.listdir(os.path.join(root, "labels", "val"))) != sorted(n + ".txt" for n in val):
            write_dataset(root, train_names, val)
        for m in models:
            key = f"{short(m)}/fold{fi}/{run_tag}"
            if key in results["runs"]:
                continue
            t0 = time.time()
            print(f"\n=== {key}: train {len(train_names)} foto, val {len(val)} foto, {epochs} epoch")
            w = train(m, os.path.join(root, "data.yaml"), run_dir, key.replace("/", "_"), epochs, a)
            from ultralytics import YOLO
            vkw = dict(imgsz=a["imgsz"], split="val", device=a["device"], plots=False, verbose=False,
                       project=os.path.join(ROOT, "runs", "cv_val"), exist_ok=True)
            run_name = key.replace("/", "_")
            clean = metrics_of(YOLO(w).val(data=os.path.join(root, "data.yaml"), name=run_name, **vkw))
            deg = metrics_of(YOLO(w).val(data=os.path.join(root, "data_deg.yaml"), name=run_name + "_deg", **vkw))
            oof = oof_predictions(w, val, a)
            results["runs"][key] = {"clean": clean, "degraded": deg, "minutes": round((time.time() - t0) / 60, 1),
                                    "oof": {n: [[c, round(s, 4), ok] for c, s, ok in v[0]] for n, v in oof.items()},
                                    "oof_gt": {n: v[1] for n, v in oof.items()}}
            json.dump(results, open(RESULTS, "w"), indent=1)
            print(f"    mAP50-95 asli {clean['mAP50_95']} | rusak (mirip webcam) {deg['mAP50_95']} | "
                  f"{results['runs'][key]['minutes']} menit")

    print("\nHasil cross-validation (rata-rata +- simpangan antar fold)")
    summary = {}
    for m in models:
        mk = short(m)
        runs = [results["runs"][f"{mk}/fold{i}/{run_tag}"] for i in range(k)]
        vals = {part: [r[part]["mAP50_95"] for r in runs] for part in ("clean", "degraded")}
        sl_r = [r["clean"]["per_class"].get("size_label", {}).get("R", 0) for r in runs]
        score = statistics.mean(vals["clean"] + vals["degraded"])
        summary[mk] = {"score": round(score, 4), **{p: [round(statistics.mean(v), 4), round(statistics.pstdev(v), 4)]
                                                    for p, v in vals.items()},
                       "size_label_R": round(statistics.mean(sl_r), 3)}
        print(f"  {mk:14s} mAP50-95 asli {summary[mk]['clean'][0]:.3f}+-{summary[mk]['clean'][1]:.3f} | "
              f"rusak {summary[mk]['degraded'][0]:.3f}+-{summary[mk]['degraded'][1]:.3f} | "
              f"recall size_label {summary[mk]['size_label_R']:.3f} | skor {score:.3f}")
    order = [short(m) for m in models]
    top = max(s["score"] for s in summary.values())
    # candidates within `tie` of the best score are a draw (fold spread is ~0.1): the first listed one wins,
    # so list the faster / more robust configuration first
    best = next(mk for mk in order if summary[mk]["score"] >= top - tie)
    oof = {}
    for i in range(k):
        r = results["runs"][f"{best}/fold{i}/{run_tag}"]
        for n, rows in r["oof"].items():
            oof[n] = ([(c, s, bool(ok)) for c, s, ok in rows], {int(c): v for c, v in r["oof_gt"][n].items()})
    thr = best_thresholds(oof)
    print(f"terpilih: {best} | threshold (out-of-fold, F1 terbaik): "
          + ", ".join(f"{c} conf>={v['conf']} (P {v['P']} R {v['R']})" for c, v in thr.items()))

    out = {"model": best, "cv": summary, "thresholds": thr, "photos": len(names), "fingerprint": fp,
           "imgsz": a["imgsz"], "final_weights": None}
    if "--no-final" not in args:
        root = os.path.join(CV_DIR, "all")
        write_dataset(root, names, names, deg=False)
        print(f"\n=== model final {best}: {len(names)} foto, {final_epochs} epoch")
        tag = f"{best}|{fp}|{final_epochs}|{a['imgsz']}"
        final_dir = os.path.join(ROOT, "runs", "best_final")
        done = os.path.join(final_dir, "done")
        if os.path.isdir(final_dir) and not (os.path.exists(done) and open(done).read().strip() == tag):
            old = final_dir + time.strftime("_%Y%m%d_%H%M%S")          # model lain / foto lain: simpan, latih ulang
            while os.path.exists(old):
                old += "_"
            os.rename(final_dir, old)
            print(f"model final sebelumnya dipindah ke {old}")
        w = train(models[order.index(best)], os.path.join(root, "data.yaml"), os.path.join(ROOT, "runs"), "best_final",
                  final_epochs, a, tag=tag)
        out["final_weights"] = w
        shutil.copy2(w, os.path.join(ROOT, "runs", "best_final", "box_best.pt"))
    json.dump(out, open(os.path.join(LOGS, "best_model.json"), "w"), indent=1)
    print(f"\nselesai dalam {(time.time() - t_start) / 60:.0f} menit -> logs/best_model.json")
    if out["final_weights"]:
        print(f"model final: {os.path.join(ROOT, 'runs', 'best_final', 'box_best.pt')}")
        print("pakai di pipeline: conf box = {box}, size_label = {sl}".format(
            box=thr["box"]["conf"], sl=thr["size_label"]["conf"]))


if __name__ == "__main__":
    main(sys.argv[1:])
