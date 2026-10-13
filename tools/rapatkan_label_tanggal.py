#!/usr/bin/env python3
"""Rapatkan label tanggal 'posisi dari model' (sumber model_v7) menjadi TEPAT baris tanggal kedaluwarsa.

Aturan cetak (iVascular, dikonfirmasi pengguna 12-10): urutan baris = LOT (atas), KEDALUWARSA (tengah), PRODUKSI (bawah).
Label harus hanya melingkupi baris kedaluwarsa. Posisi dari model sering menutupi blok dua baris; di sini daerah label
dibaca ulang dengan OCR dan, bila ada tanggal, poligonnya diganti dengan baris TANGGAL TERBARU (= kedaluwarsa).
Label yang tidak terbaca (buram) dibiarkan. Foto berstatus 'selesai' (sudah direview manusia) tidak disentuh.

  python tools/rapatkan_label_tanggal.py            hitung saja, tulis tanggal_rapat.csv
  python tools/rapatkan_label_tanggal.py --terapkan --dari-csv   pakai hasil hitung terakhir (tanpa OCR ulang)
  python tools/rapatkan_label_tanggal.py --terapkan  cadangkan label_specs_expiry lalu terapkan
"""
import csv
import datetime as dt
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_expiry import PAD_T, PAD_V, PAD_W, label_poly  # noqa: E402
from cek_tanggal_terlewat import read_region, strip_M  # noqa: E402
from common import EXPIRY, ROOT, load_expiry, load_spec, raw_path  # noqa: E402
from make_labels import resolve_obbs  # noqa: E402

OUT = os.path.join(ROOT, "tanggal_rapat.csv")
F = 1 + 2 * PAD_W


def work(name):
    e = load_expiry(name)
    todo = [x for x in e["expiry_label"] if x.get("sumber") == "model_v7"]
    img = cv2.imread(raw_path(name))
    H, W = img.shape[:2]
    boxes, _, _ = resolve_obbs(name, load_spec(name), W, H)
    m = re.search(r"(20\d\d)(\d\d)(\d\d)", name)
    shot = dt.date(*map(int, m.groups())) if m else None
    out = []
    for x in todo:
        ob = boxes[x["box"] - 1]
        s, M = strip_M(img, ob, PAD_W)
        Ls, Ws = s.shape[1], s.shape[0]
        qm = cv2.perspectiveTransform((np.float32(x["poly"]).reshape(4, 2) * [W / 100, H / 100]).reshape(-1, 1, 2), M).reshape(4, 2)
        old_h = (qm[:, 1].max() - qm[:, 1].min()) / Ws * F
        mx, my = 0.6 * (qm[:, 0].max() - qm[:, 0].min()), 0.8 * (qm[:, 1].max() - qm[:, 1].min())
        b = read_region(s, int(qm[:, 0].min() - mx), int(qm[:, 0].max() + mx), int(qm[:, 1].min() - my), int(qm[:, 1].max() + my), shot)
        rec = {"foto": name, "box": x["box"], "ok": False, "tinggi_lama": round(old_h, 2)}
        if b is not None:
            day, q, text, score = b
            t1, t2 = max(0.0, q[:, 0].min() / Ls - PAD_T), min(1.0, q[:, 0].max() / Ls + PAD_T)
            v1 = min(1.0, max(0.0, (q[:, 1].min() / Ws) * F - PAD_W - PAD_V))
            v2 = min(1.0, max(0.0, (q[:, 1].max() / Ws) * F - PAD_W + PAD_V))
            if v2 - v1 >= 0.15:
                rec.update(ok=True, tanggal=day.isoformat(), teks=text, skor=round(float(score), 3), tinggi_baru=round(v2 - v1, 2),
                           poly=json.dumps(label_poly(ob, t1, t2, v1, v2, W, H)))
        out.append(rec)
    return out


def main(args):
    names = []
    for f in sorted(os.listdir(EXPIRY)):
        if not f.endswith(".json"):
            continue
        e = json.load(open(os.path.join(EXPIRY, f), encoding="utf-8"))
        if e.get("status") != "selesai" and any(x.get("sumber") == "model_v7" for x in e["expiry_label"]):
            names.append(f[:-5])
    print(f"{len(names)} foto (belum selesai) punya label posisi-model")
    if "--dari-csv" in args:                       # pakai hasil hitung sebelumnya (tanpa OCR ulang)
        recs = [{**r, "ok": r["ok"] == "True"} for r in csv.DictReader(open(OUT, encoding="utf-8"))]
    else:
        with mp.get_context("spawn").Pool(6) as pool:
            recs = [r for part in pool.map(work, names, chunksize=2) for r in part]
    ok = [r for r in recs if r["ok"]]
    if "--dari-csv" not in args:
      with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["foto", "box", "ok", "tanggal", "teks", "skor", "tinggi_lama", "tinggi_baru", "poly"])
        w.writeheader()
        w.writerows(recs)
    print(f"label posisi-model {len(recs)}: terbaca & dirapatkan {len(ok)}, tidak terbaca (dibiarkan) {len(recs) - len(ok)}")
    if ok:
        print(f"tinggi rata-rata (fraksi lebar kotak): {np.mean([float(r['tinggi_lama']) for r in ok]):.2f} -> {np.mean([float(r['tinggi_baru']) for r in ok]):.2f}")
    if "--terapkan" not in args:
        print("Belum diubah. Terapkan dengan --terapkan")
        return
    bak = os.path.expanduser(f"~/Downloads/cadangan_label_specs_expiry_{time.strftime('%Y%m%d_%H%M')}")
    shutil.copytree(EXPIRY, bak)
    n = skip = longgar = tak_dikenal = 0
    known = {}
    for f in os.listdir(EXPIRY):
        for x in json.load(open(os.path.join(EXPIRY, f), encoding="utf-8")).get("expiry_label", []):
            if x.get("date") and x.get("sumber") not in ("model_v7", "model_v7+ocr", "cek_ketat"):
                known[x["date"]] = known.get(x["date"], 0) + 1
    for r in ok:
        if float(r["tinggi_baru"]) > float(r["tinggi_lama"]) + 0.03:   # perapatan tidak boleh melebarkan label
            longgar += 1
            continue
        p = os.path.join(EXPIRY, r["foto"] + ".json")
        e = json.load(open(p, encoding="utf-8"))            # baca ulang: pengguna mungkin sedang mereview
        if e.get("status") == "selesai":
            skip += 1
            continue
        for x in e["expiry_label"]:
            if x["box"] == int(r["box"]) and x.get("sumber") == "model_v7":
                dikenal = known.get(r["tanggal"], 0) >= 2               # tanggal yang hanya terbaca sekali dianggap salah baca
                tak_dikenal += not dikenal
                x["poly"], x["date"] = json.loads(r["poly"]), (r["tanggal"] if dikenal else "")
                x["text"], x["score"], x["sumber"] = f"OCR baris kedaluwarsa: {r['teks']}", float(r["skor"]), "model_v7+ocr"
                n += 1
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(e, fh, ensure_ascii=False, indent=1)
    print(f"Diterapkan: {n} label dirapatkan; dilewati: {skip} foto sudah selesai, {longgar} karena akan melebar; "
          f"{tak_dikenal} tanggal tidak dikenal dikosongkan (posisi tetap). Cadangan: {bak}")


if __name__ == "__main__":
    main(sys.argv[1:])
