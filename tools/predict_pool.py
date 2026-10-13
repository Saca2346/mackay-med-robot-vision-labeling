"""Predictions of a trained model for every pool photo (or all photos), with confidence, for labeling drafts.

Writes pred_pool/NAME.txt: "cls x1 y1 x2 y2 x3 y3 x4 y4 conf" (normalized corners), low threshold (conf >= 0.10)
so a reviewer sees weak detections too. tools/pred_to_spec.py turns one file into an editable spec draft.

Usage: python tools/predict_pool.py --model runs/best_final/weights/best.pt [--imgsz 1280] [--device 0] [--all]
                                     [--out pred_pool]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al_select import opt  # noqa: E402
from common import RAW, ROOT  # noqa: E402
from export_datasets import drop_duplicates, seed_state  # noqa: E402


def main(args):
    model_path = opt(args, "--model", None)
    if not model_path:
        raise SystemExit(__doc__)
    imgsz, device = opt(args, "--imgsz", 1280, int), opt(args, "--device", None)
    out = opt(args, "--out", os.path.join(ROOT, "pred_pool"))
    os.makedirs(out, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(model_path)
    names, _ = drop_duplicates(sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".jpg")))
    if "--all" not in args:
        names = [n for n in names if seed_state(n)[0] != "seed"]
    kw = {"device": device} if device not in (None, "") else {}
    for i, n in enumerate(names, 1):
        r = model.predict(os.path.join(RAW, n + ".jpg"), imgsz=imgsz, conf=0.10, iou=0.5, verbose=False, **kw)[0]
        H, W = r.orig_shape
        with open(os.path.join(out, n + ".txt"), "w", newline="\n") as fh:
            if r.obb is not None:
                for poly, c, s in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
                    fh.write(f"{int(c)} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly) + f" {s:.4f}\n")
        print(f"{i}/{len(names)} {n}: {0 if r.obb is None else len(r.obb)} deteksi", flush=True)
    print("ditulis ke", out)


if __name__ == "__main__":
    main(sys.argv[1:])
