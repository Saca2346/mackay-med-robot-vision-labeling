#!/usr/bin/env python3
"""Lembar review dari label tanggal TERKINI (label_specs_expiry/NAME.json), bukan dari hasil OCR awal.

Tiap strip = satu kotak (ujung jauh kiri, ujung dekat kanan). Warna:
  magenta = label lama / sudah direview      cyan = usulan baru (cek_ketat: OCR, model_v7: posisi dari model)
  bingkai merah = kotak 'periksa' yang belum berlabel dan belum 'kosong' (wajib dilihat)
  python tools/lembar_tanggal.py NAME [NAME ...]   -> preview_expiry/NAME_cek.jpg
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_expiry import PAD_W  # noqa: E402
from cek_tanggal_terlewat import strip_M  # noqa: E402
from common import ROOT, load_expiry, load_spec, raw_path  # noqa: E402
from make_labels import resolve_obbs  # noqa: E402

OUT = os.path.join(ROOT, "preview_expiry")
WIDTH = 1500


def render(name, out=OUT, quality=85):
    img = cv2.imread(raw_path(name))
    H, W = img.shape[:2]
    boxes, _, _ = resolve_obbs(name, load_spec(name), W, H)
    e = load_expiry(name) or {}
    labels = e.get("expiry_label", [])
    done = {x.get("box") for x in labels} | set(e.get("kosong", []))
    rows = []
    for n, ob in enumerate(boxes, 1):
        s, M = strip_M(img, ob, PAD_W)
        k = WIDTH / s.shape[1]
        s = cv2.resize(s, (WIDTH, max(24, int(s.shape[0] * k))), interpolation=cv2.INTER_AREA)
        tags = []
        for x in labels:
            if x.get("box") != n:
                continue
            p = np.float32(x["poly"]).reshape(4, 2) * [W / 100, H / 100]
            q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), M).reshape(4, 2) * k
            new = x.get("sumber") in ("cek_ketat", "model_v7", "model_v7+ocr")
            col = (255, 255, 0) if new else (255, 0, 255)
            cv2.polylines(s, [q.astype(np.int32)], True, col, 3)
            tags.append(f"{x.get('date') or '?'}{' BARU' if new else ''}")
        if n in e.get("kosong", []):
            tags.append("kosong (sudah dilihat)")
        open_ = n in e.get("periksa", []) and n not in done
        if open_:
            cv2.rectangle(s, (0, 0), (s.shape[1] - 1, s.shape[0] - 1), (0, 0, 255), 6)
            tags.append("PERIKSA")
        bar = np.full((28, WIDTH, 3), 255, np.uint8)
        cv2.putText(bar, f"box {n}: " + (", ".join(tags) or "-"), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 220) if open_ else (40, 40, 40), 2)
        rows += [s, bar]
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"{name}_cek.jpg")
    cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return p


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        print(render(nm))
