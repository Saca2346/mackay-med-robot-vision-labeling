"""Copy a verified spec to a neighbouring photo of the same scene (burst shot) by image registration.

SIFT features inside the source boxes are matched to the target photo, a RANSAC homography maps every box
corner, and size labels are carried over as fractions along their box ({"of": N, "t": [...]}), which the small
camera motion within a burst leaves unchanged. Boxes whose mapped corners fall outside the target photo are
cut to it (fit happens in make_labels). The note starts with "SEDANG DIKERJAKAN" so the photo stays in the pool
until the result has been checked (tools/endcheck.py) and marked done (tools/set_note.py).
Prints the number of RANSAC inliers and the median reprojection error; below MIN_INLIERS nothing is written.

Usage: python tools/transfer_spec.py SRC_NAME DST_NAME [--dry]
"""
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SPECS, load_raw, load_spec, spec_groups, write_spec  # noqa: E402
from make_labels import canvas_quad_to_orig, resolve_obbs  # noqa: E402

MIN_INLIERS = 60
SCALE = 0.35          # features are computed on photos downscaled by this factor


def gray(name):
    img = cv2.imread(os.path.join(ROOT, "raw", name + ".jpg"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def main(src, dst, dry=False):
    spec = load_spec(src)
    groups = spec_groups(spec)
    _, (W, H) = load_raw(src, 64)
    boxes, sls, _ = resolve_obbs(src, spec, W, H)
    entries = [canvas_quad_to_orig(e, W, H, g.get("angle", 0)) for g in groups for e in g.get("box", [])]
    if any(len(e) != 8 for e in entries):
        raise SystemExit("source spec must use 8-number boxes")

    a, b = gray(src), gray(dst)
    mask = np.zeros_like(a)
    for A, B, hw in boxes:
        ux, uy = B[0] - A[0], B[1] - A[1]
        L = math.hypot(ux, uy) or 1.0
        nx, ny = -uy / L * hw, ux / L * hw
        poly = np.array([(A[0] + nx, A[1] + ny), (B[0] + nx, B[1] + ny), (B[0] - nx, B[1] - ny),
                         (A[0] - nx, A[1] - ny)]) * SCALE
        cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(a, mask)
    kb, db = sift.detectAndCompute(b, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good]) / SCALE
    pb = np.float32([kb[m.trainIdx].pt for m in good]) / SCALE
    Hm, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 6.0)
    n_in = int(inl.sum()) if inl is not None else 0
    err = np.linalg.norm(cv2.perspectiveTransform(pa[inl.ravel() == 1].reshape(-1, 1, 2), Hm).reshape(-1, 2)
                         - pb[inl.ravel() == 1], axis=1) if n_in else np.array([np.inf])
    print(f"{src} -> {dst}: {len(good)} pasangan, {n_in} inlier, galat median {np.median(err):.1f} px")
    if n_in < MIN_INLIERS:
        raise SystemExit(f"terlalu sedikit inlier (< {MIN_INLIERS}): foto tidak cukup mirip, label manual")

    new_boxes = []
    for e in entries:
        pts = np.float32([(e[k] / 100 * W, e[k + 1] / 100 * H) for k in range(0, 8, 2)]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
        new_boxes.append([round(float(v), 2) for x, y in q for v in (x / W * 100, y / H * 100)])
    out_sl, k = [], 0
    for g in groups:
        for e in g.get("size_label", []):
            poly = sls[k]
            k += 1
            if not isinstance(e, dict):
                print(f"   size_label persegi lama dilewati: {e}")
                continue
            A, B, _ = boxes[e["of"] - 1]
            du = np.array(B) - np.array(A)
            L2 = float(du @ du)
            ts = [float((np.array(p) - np.array(A)) @ du / L2) for p in poly]
            ent = {"of": e["of"], "t": [round(max(0.0, min(ts)), 3), round(min(1.0, max(ts)), 3)]}
            if "v" in e:
                ent["v"] = e["v"]
            out_sl.append(ent)
    new = {"groups": [{"angle": 0, "box": new_boxes, "size_label": out_sl}], "status": "perlu dicek manual",
           "note": f"SEDANG DIKERJAKAN: dipindahkan dari {src} (homografi, {n_in} inlier); belum diperiksa"}
    print(f"   {len(new_boxes)} box, {len(out_sl)} size_label")
    if dry:
        return
    path = os.path.join(SPECS, dst + ".json")
    if os.path.exists(path):
        bdir = os.path.join(ROOT, "label_specs_backup")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(path, os.path.join(bdir, f"{dst}_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    write_spec(dst, new)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], "--dry" in sys.argv)
