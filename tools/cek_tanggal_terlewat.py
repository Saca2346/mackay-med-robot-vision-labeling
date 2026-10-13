#!/usr/bin/env python3
"""Cari tanggal kedaluwarsa yang TERLEWAT: kotak yang cukup tajam tetapi belum punya expiry_label.

Tiga sumber, per kotak tanpa label:
  1. ketajaman strip kotak (Laplacian; ambang = persentil 20 dari kotak yang tanggalnya terbaca),
  2. model 3 kelas (runs/best_final, kelas 2) yang melihat tanggal di dalam kotak itu,
  3. OCR ulang yang lebih ketat: daerah prediksi model diperbesar 3x, atau strip dibaca dalam 5 potong x3, dua arah.
Kategori:  A = OCR ketat menemukan tanggal (usulan label baru)  |  B = model melihat tanggal, OCR gagal (cek mata)
           C = tajam, tanpa prediksi/tanggal (cek mata)          |  kotak buram tanpa prediksi: tidak dipersoalkan

  python tools/cek_tanggal_terlewat.py              hitung saja -> tanggal_terlewat.csv + ringkasan (tidak mengubah label)
  python tools/cek_tanggal_terlewat.py --terapkan   + cadangkan label_specs_expiry, tambahkan usulan A, simpan daftar
                                                   "periksa" (B, C) per foto, dan buka lagi (status auto) foto "selesai"
                                                   yang ternyata punya kotak A/B/C
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
from auto_expiry import PAD_T, PAD_V, PAD_W, _lines, _perp, find_dates, label_poly  # noqa: E402
from common import EXPIRY, ROOT, SPECS, load_expiry, load_spec, obb_polygon, raw_path  # noqa: E402
from make_labels import resolve_obbs  # noqa: E402

MODEL = os.path.join(ROOT, "runs", "best_final", "weights", "best.pt")
CACHE = os.path.join(ROOT, "logs", "cek_tanggal_prediksi.json")
OUT = os.path.join(ROOT, "tanggal_terlewat.csv")
CONF = 0.25
MIN_SCORE = 0.5
MAX_YEAR = 2035
TAJAM_B = 85          # B (model melihat tanggal): di bawah ini tanggal tidak terbaca mata (dicek visual 10-11)


def strip_M(img, obb, pad):
    A, B, hw = obb
    (px, py), L = _perp(A, B)
    k = min(1.0, 2400 / L)
    hp = hw * (1 + 2 * pad)
    Ls, Ws = max(8, int(L * k)), max(8, int(2 * hp * k))
    src = np.float32([(A[0] - px * hp, A[1] - py * hp), (B[0] - px * hp, B[1] - py * hp),
                      (B[0] + px * hp, B[1] + py * hp), (A[0] + px * hp, A[1] + py * hp)])
    M = cv2.getPerspectiveTransform(src, np.float32([(0, 0), (Ls, 0), (Ls, Ws), (0, Ws)]))
    return cv2.warpPerspective(img, M, (Ls, Ws), flags=cv2.INTER_CUBIC), M


def sharpness(img, obb):
    s, _ = strip_M(img, obb, 0.0)
    g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (max(8, int(g.shape[1] * 100 / g.shape[0])), 100), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def predictions(names):
    if os.path.exists(CACHE):
        c = json.load(open(CACHE))
        if c.get("model_mtime") == os.path.getmtime(MODEL) and set(names) <= set(c["pred"]):
            return c["pred"]
    from ultralytics import YOLO
    m = YOLO(MODEL)
    pred = {}
    for i, n in enumerate(names, 1):
        r = m.predict(raw_path(n), imgsz=1280, conf=CONF, device="0", verbose=False)[0]
        pred[n] = [[float(s), q.reshape(-1).tolist()] for q, c, s in
                   zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()) if int(c) == 2]
        if i % 20 == 0:
            print(f"  prediksi {i}/{len(names)}", flush=True)
    json.dump({"model_mtime": os.path.getmtime(MODEL), "pred": pred}, open(CACHE, "w"))
    return pred


_OCR = None


def ocr_engine():
    global _OCR
    if _OCR is None:
        import logging
        from rapidocr import RapidOCR
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        _OCR = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 2400,
                                "EngineConfig.onnxruntime.intra_op_num_threads": 4})
    return _OCR


def best_date(lines, shot):
    best = None
    for q, text, score in lines:
        ds = [d for d in find_dates(text) if (shot is None or d > shot) and d.year <= MAX_YEAR]
        if ds and score >= MIN_SCORE and (best is None or max(ds) > best[0]):
            best = (max(ds), q, text, score)
    return best


def read_region(s, x0, x1, y0, y1, shot, scales=(1.0, 2.0, 3.0)):
    """OCR of strip region at increasing scale (enlarging blurry print can hurt, so 1x first), upright and 180 deg."""
    x0, x1, y0, y1 = max(0, x0), min(s.shape[1], x1), max(0, y0), min(s.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    for sc in scales:
        b = _read_region(s, x0, x1, y0, y1, shot, sc)
        if b:
            return b
    return None


def _read_region(s, x0, x1, y0, y1, shot, scale):
    crop = cv2.resize(s[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = crop.shape[:2]
    lines = [(q / scale + [x0, y0], t, sc) for q, t, sc in _lines(ocr_engine(), crop)]
    lines += [((np.array([w, h]) - q) / scale + [x0, y0], t, sc) for q, t, sc in _lines(ocr_engine(), cv2.rotate(crop, cv2.ROTATE_180))]
    return best_date(lines, shot)


def check_photo(args):
    name, pred, labelled = args
    exp_labels = {}
    for x in (load_expiry(name) or {}).get("expiry_label", []):
        exp_labels.setdefault(x.get("box"), []).append(x)
    img = cv2.imread(raw_path(name))
    H, W = img.shape[:2]
    boxes, _, _ = resolve_obbs(name, load_spec(name), W, H)
    m = re.search(r"(20\d\d)(\d\d)(\d\d)", name)
    shot = dt.date(*map(int, m.groups())) if m else None
    f = 1 + 2 * PAD_W
    rows = []
    for n, ob in enumerate(boxes, 1):
        poly = np.float32(obb_polygon(ob))
        mine = [(sc, np.float32(q).reshape(4, 2)) for sc, q in pred
                if cv2.pointPolygonTest(poly, tuple(map(float, np.float32(q).reshape(4, 2).mean(0))), False) >= 0]
        row = {"foto": name, "box": n, "berlabel": n in labelled, "tajam": round(sharpness(img, ob), 1),
               "prediksi_model": round(max((sc for sc, _ in mine), default=0.0), 3), "tanggal": "", "kategori": "", "poly": ""}
        if not row["berlabel"] and (mine or True):
            s, M = strip_M(img, ob, PAD_W)
            Ls, Ws = s.shape[1], s.shape[0]
            best = None
            for sc, q in sorted(mine, key=lambda x: -x[0]):
                qs = cv2.perspectiveTransform(q.reshape(-1, 1, 2), M).reshape(4, 2)
                mx, my = 0.6 * (qs[:, 0].max() - qs[:, 0].min()), 0.8 * (qs[:, 1].max() - qs[:, 1].min())
                best = read_region(s, int(qs[:, 0].min() - mx), int(qs[:, 0].max() + mx),
                                   int(qs[:, 1].min() - my), int(qs[:, 1].max() + my), shot)
                if best:
                    break
            row["_tajam_dulu"] = True
            if best is None and not mine:
                row["_strip"] = (Ls, Ws)
            if best is not None:
                day, q, text, score = best
                t1, t2 = max(0.0, q[:, 0].min() / Ls - PAD_T), min(1.0, q[:, 0].max() / Ls + PAD_T)
                v1 = min(1.0, max(0.0, (q[:, 1].min() / Ws) * f - PAD_W - PAD_V))
                v2 = min(1.0, max(0.0, (q[:, 1].max() / Ws) * f - PAD_W + PAD_V))
                if v2 - v1 >= 0.15:
                    row.update(tanggal=day.isoformat(), teks=text, skor=round(score, 3),
                               poly=json.dumps(label_poly(ob, t1, t2, v1, v2, W, H)))
            row["_ob"] = ob
        if row["berlabel"]:
            for x in exp_labels.get(n, []):
                if x.get("sumber") == "model_v7" and not x.get("date"):
                    s, M = strip_M(img, ob, PAD_W)
                    q = np.float32(x["poly"]).reshape(4, 2) * [W / 100, H / 100]
                    qs = cv2.perspectiveTransform(q.reshape(-1, 1, 2), M).reshape(4, 2)
                    mx, my = 0.6 * (qs[:, 0].max() - qs[:, 0].min()), 0.8 * (qs[:, 1].max() - qs[:, 1].min())
                    b = read_region(s, int(qs[:, 0].min() - mx), int(qs[:, 0].max() + mx),
                                    int(qs[:, 1].min() - my), int(qs[:, 1].max() + my), shot)
                    if b:
                        row["isi_tanggal"] = b[0].isoformat()
                        row["teks"] = b[2]
        rows.append(row)
    return name, rows


def tiled_pass(args):
    """Second pass for sharp boxes without model prediction: whole strip in 5 tiles x3, both directions."""
    name, boxes_n = args
    img = cv2.imread(raw_path(name))
    H, W = img.shape[:2]
    boxes, _, _ = resolve_obbs(name, load_spec(name), W, H)
    m = re.search(r"(20\d\d)(\d\d)(\d\d)", name)
    shot = dt.date(*map(int, m.groups())) if m else None
    f = 1 + 2 * PAD_W
    out = {}
    for n in boxes_n:
        ob = boxes[n - 1]
        s, _ = strip_M(img, ob, PAD_W)
        Ls, Ws = s.shape[1], s.shape[0]
        step = Ls / (5 - 4 * 0.25)
        best = None
        for i in range(5):
            x0 = int(i * step * 0.75)
            b = read_region(s, x0, int(x0 + step), 0, Ws, shot)
            if b and (best is None or b[0] > best[0]):
                best = b
        if best:
            day, q, text, score = best
            t1, t2 = max(0.0, q[:, 0].min() / Ls - PAD_T), min(1.0, q[:, 0].max() / Ls + PAD_T)
            v1 = min(1.0, max(0.0, (q[:, 1].min() / Ws) * f - PAD_W - PAD_V))
            v2 = min(1.0, max(0.0, (q[:, 1].max() / Ws) * f - PAD_W + PAD_V))
            if v2 - v1 >= 0.15:
                out[n] = dict(tanggal=day.isoformat(), teks=text, skor=round(score, 3),
                              poly=json.dumps(label_poly(ob, t1, t2, v1, v2, W, H)))
    return name, out


def main(args):
    t0 = time.time()
    names = sorted(f[:-5] for f in os.listdir(SPECS) if f.endswith(".json"))
    print(f"1/3 prediksi model 3 kelas ({os.path.basename(os.path.dirname(os.path.dirname(MODEL)))}) pada {len(names)} foto")
    pred = predictions(names)
    exp = {n: (load_expiry(n) or {}) for n in names}
    jobs = [(n, pred.get(n, []), {x["box"] for x in exp[n].get("expiry_label", [])}) for n in names]
    print("2/3 ketajaman + OCR ketat di daerah prediksi model")
    with mp.get_context("spawn").Pool(6) as pool:
        res = dict(pool.map(check_photo, jobs, chunksize=2))
    rows = [r for n in names for r in res[n]]
    lab = sorted(r["tajam"] for r in rows if r["berlabel"])
    ambang = lab[len(lab) // 5]
    need = {}
    for r in rows:
        if not r["berlabel"] and not r["tanggal"] and not r["prediksi_model"] and r["tajam"] >= ambang:
            need.setdefault(r["foto"], []).append(r["box"])
    print(f"3/3 OCR strip penuh untuk {sum(map(len, need.values()))} kotak tajam tanpa prediksi (ambang tajam {ambang:.0f})")
    with mp.get_context("spawn").Pool(6) as pool:
        extra = dict(pool.map(tiled_pass, list(need.items()), chunksize=1))
    for r in rows:
        for k in ("_ob", "_strip", "_tajam_dulu"):
            r.pop(k, None)
        e = extra.get(r["foto"], {}).get(r["box"])
        if e:
            r.update(e)
        if r["berlabel"]:
            continue
        if r["tanggal"]:
            r["kategori"] = "A"
        elif r["prediksi_model"]:
            r["kategori"] = "B"
        elif r["tajam"] >= ambang:
            r["kategori"] = "C"
    for r in rows:
        if r.get("isi_tanggal") and not r["kategori"]:
            r["kategori"] = "D"
    flagged = [r for r in rows if r["kategori"]]
    fields = ["foto", "box", "kategori", "isi_tanggal", "berlabel", "tajam", "prediksi_model", "tanggal", "teks", "skor", "poly"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(flagged)
    st = {n: exp[n].get("status") for n in names}
    for k, label in (("A", "OCR ketat menemukan tanggal"), ("B", "model melihat tanggal, OCR gagal"),
                     ("C", "tajam tanpa tanggal/prediksi")):
        rs = [r for r in flagged if r["kategori"] == k]
        print(f"  {k}: {len(rs):4d} kotak di {len({r['foto'] for r in rs}):3d} foto ({label}); "
              f"di foto 'selesai': {sum(1 for r in rs if st[r['foto']] == 'selesai')}")
    print(f"kotak berlabel {sum(r['berlabel'] for r in rows)}, tanpa label {sum(not r['berlabel'] for r in rows)}, "
          f"buram & tanpa prediksi (dibiarkan) {sum(not r['berlabel'] and not r['kategori'] for r in rows)}")
    print(f"-> {OUT}  ({time.time() - t0:.0f} s)")

    json.dump({"ambang_tajam": ambang}, open(os.path.join(ROOT, "logs", "cek_tanggal_ambang.json"), "w"))
    if "--terapkan" in args:
        terapkan()
    else:
        print("Belum ada label yang diubah. Terapkan: python tools/cek_tanggal_terlewat.py --hanya-terapkan")


def terapkan():
    """Pakai tanggal_terlewat.csv: A -> label OCR, B yang tajam -> posisi dari model (tanggal kosong; training hanya
    butuh posisi), A/B-tajam/C -> daftar 'periksa'. Foto 'selesai' yang terkena dibuka lagi (status auto)."""
    ambang = json.load(open(os.path.join(ROOT, "logs", "cek_tanggal_ambang.json")))["ambang_tajam"]
    pred = json.load(open(CACHE))["pred"]
    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    bak = os.path.expanduser(f"~/Downloads/cadangan_label_specs_expiry_{time.strftime('%Y%m%d_%H%M')}")
    shutil.copytree(EXPIRY, bak)
    per = {}
    for r in rows:
        if r["kategori"] == "B" and float(r["tajam"]) < TAJAM_B:
            continue                      # buram: tanggal tidak terbaca mata -> tidak dilabeli
        per.setdefault(r["foto"], []).append(r)
    dibuka = tambah_a = tambah_b = diisi = 0
    for n, mine in sorted(per.items()):
        e = load_expiry(n) or {"expiry_label": [], "status": "auto", "note": ""}
        have = {x["box"] for x in e["expiry_label"]}
        img = cv2.imread(raw_path(n))
        H, W = img.shape[:2]
        boxes, _, _ = resolve_obbs(n, load_spec(n), W, H)
        for r in mine:
            b = int(r["box"])
            if r["kategori"] == "D":              # tanggal untuk label posisi-model yang masih kosong
                for x in e["expiry_label"]:
                    if x.get("box") == b and x.get("sumber") == "model_v7" and not x.get("date"):
                        x["date"], x["text"] = r["isi_tanggal"], f"OCR di posisi model: {r.get('teks', '')}"
                        diisi += 1
                continue
            if b in have:
                continue
            if r["kategori"] == "A":
                e["expiry_label"].append({"poly": json.loads(r["poly"]), "box": b, "date": r["tanggal"],
                                          "text": r.get("teks", ""), "score": float(r.get("skor") or 0), "sumber": "cek_ketat"})
                tambah_a += 1
            elif r["kategori"] == "B":
                poly = np.float32(obb_polygon(boxes[b - 1]))
                cand = [(sc, np.float32(q).reshape(4, 2)) for sc, q in pred.get(n, [])
                        if cv2.pointPolygonTest(poly, tuple(map(float, np.float32(q).reshape(4, 2).mean(0))), False) >= 0]
                if cand:
                    sc, q = max(cand, key=lambda x: x[0])
                    st, M = strip_M(img, boxes[b - 1], PAD_W)       # posisi model -> dijepit di dalam kotak (check_specs)
                    qs = cv2.perspectiveTransform(q.reshape(-1, 1, 2), M).reshape(4, 2)
                    ff = 1 + 2 * PAD_W
                    t1, t2 = np.clip([qs[:, 0].min() / st.shape[1], qs[:, 0].max() / st.shape[1]], 0, 1)
                    v1, v2 = np.clip([(qs[:, 1].min() / st.shape[0]) * ff - PAD_W, (qs[:, 1].max() / st.shape[0]) * ff - PAD_W], 0, 1)
                    e["expiry_label"].append({"poly": label_poly(boxes[b - 1], float(t1), float(t2), float(v1), float(v2), W, H),
                                              "box": b, "date": "", "text": "posisi dari model", "score": round(sc, 3),
                                              "sumber": "model_v7"})
                    tambah_b += 1
        e["periksa"] = sorted(set(e.get("periksa", [])) | {int(r["box"]) for r in mine if r["kategori"] != "D"})
        if e.get("status") == "selesai" and any(r["kategori"] != "D" for r in mine):
            e["status"] = "auto"
            e["note"] = (e.get("note", "") + f" | dibuka lagi {time.strftime('%Y-%m-%d')}: tanggal tajam terlewat").strip(" |")
            dibuka += 1
        with open(os.path.join(EXPIRY, n + ".json"), "w", encoding="utf-8") as fh:
            json.dump(e, fh, ensure_ascii=False, indent=1)
    print(f"Diterapkan pada {len(per)} foto: +{tambah_a} label OCR (A), +{tambah_b} posisi model (B tajam); "
          f"tanggal diisi ke posisi-model: {diisi}; foto 'selesai' dibuka lagi: {dibuka}. Cadangan: {bak}")


if __name__ == "__main__":
    terapkan() if "--hanya-terapkan" in sys.argv else main(sys.argv[1:])
