#!/usr/bin/env python3
"""Satu folder hasil label tanggal untuk dicek BANYAK foto sekaligus (seperti preview_final), + galeri HTML.

  python tools/preview_tanggal.py            -> hasil_tanggal/  (index.html, 1_menunggu_review/, 2_selesai/, detail_strip/)
  python tools/preview_tanggal.py --zip      juga buat ~/Downloads/hasil_tanggal_YYYYMMDD.zip

Per foto: foto utuh + kotak bernomor + label tanggal.
  magenta = label tanggal (OCR awal / sudah direview)     cyan = usulan baru (OCR ketat / posisi dari model)
  kotak MERAH tebal = PERIKSA (tajam tetapi belum berlabel & belum 'kosong')    hijau = sudah berlabel / dicatat kosong
  abu-abu = tanpa label (tidak diwajibkan: biasanya buram)
Foto menunggu review diberi nomor urut prioritas (paling banyak masalah di atas). detail_strip/ berisi lembar per kotak
(ujung jauh kiri) untuk membaca tanggalnya; hanya foto yang menunggu review."""
import csv
import html
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_spec, obb_polygon, raw_path  # noqa: E402
from lembar_tanggal import render as render_strip  # noqa: E402
from make_labels import resolve_obbs  # noqa: E402

OUT = os.path.join(ROOT, "hasil_tanggal")
MAGENTA, CYAN, RED, GREEN, GRAY = (255, 0, 255), (255, 255, 0), (0, 0, 255), (0, 190, 0), (150, 150, 150)
BARU = ("cek_ketat", "model_v7", "model_v7+ocr")


def state(name):
    e = json.load(open(os.path.join(ROOT, "label_specs_expiry", name + ".json"), encoding="utf-8"))
    L = e["expiry_label"]
    done = {x["box"] for x in L} | set(e.get("kosong", []))
    return e, L, done, [b for b in e.get("periksa", []) if b not in done]


def overview(name, e, L, done, open_):
    img = cv2.imread(raw_path(name))
    H, W = img.shape[:2]
    k = 1600 / max(H, W)
    img = cv2.resize(img, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    boxes, _, _ = resolve_obbs(name, load_spec(name), W, H)
    sc = lambda pts: (np.float32(pts) * [w / W, h / H]).astype(np.int32)
    for n, ob in enumerate(boxes, 1):
        p = sc(obb_polygon(ob))
        col, th = (RED, 3) if n in open_ else (GREEN, 1) if n in done else (GRAY, 1)
        cv2.polylines(img, [p], True, col, th, cv2.LINE_AA)
        org = tuple(p[:2].mean(0).astype(int) - [8, 4])
        cv2.putText(img, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if n in open_ else (255, 255, 255), 1, cv2.LINE_AA)
    for x in L:
        q = sc((np.float32(x["poly"]).reshape(4, 2) * [W / 100, H / 100]))
        cv2.polylines(img, [q], True, CYAN if x.get("sumber") in BARU else MAGENTA, 2, cv2.LINE_AA)
    nb = len(boxes)
    head = f"{name}   {'SELESAI' if e['status'] == 'selesai' else 'MENUNGGU REVIEW'}   label {len({x['box'] for x in L})}/{nb} kotak"
    if open_:
        head += f"   PERIKSA: {', '.join(map(str, open_))}"
    bar = np.full((34, w, 3), 30, np.uint8)
    cv2.putText(bar, head, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255) if open_ else (255, 255, 255), 1, cv2.LINE_AA)
    leg = np.full((26, w, 3), 30, np.uint8)
    cv2.putText(leg, "magenta=label tanggal  cyan=usulan baru  merah=PERIKSA  hijau=sudah  abu=tanpa label (boleh buram)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, img, leg])


def main(args):
    names = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "label_specs_expiry")) if f.endswith(".json"))
    info = {}
    for n in names:
        e, L, done, open_ = state(n)
        nb = sum(1 for l in open(os.path.join(ROOT, "labels_final_obb", n + ".txt")) if l.split()[:1] == ["0"])
        info[n] = dict(e=e, L=L, done=done, open=open_, nb=nb, tanpa=nb - len({x["box"] for x in L} | set(e.get("kosong", []))))
    pend = sorted([n for n in names if info[n]["e"]["status"] != "selesai"], key=lambda n: (-len(info[n]["open"]), -info[n]["tanpa"], n))
    fin = [n for n in names if info[n]["e"]["status"] == "selesai"]
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for d in ("1_menunggu_review", "2_selesai", "detail_strip"):
        os.makedirs(os.path.join(OUT, d))
    rows, files = [], {}
    for rank, n in enumerate(pend, 1):
        i = info[n]
        fn = f"1_menunggu_review/{rank:03d}_{n}.jpg"
        cv2.imwrite(os.path.join(OUT, fn), overview(n, i["e"], i["L"], i["done"], i["open"]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        render_strip(n, out=os.path.join(OUT, "detail_strip"), quality=78)
        files[n] = (fn, f"detail_strip/{n}_cek.jpg")
    for n in fin:
        i = info[n]
        fn = f"2_selesai/{n}.jpg"
        cv2.imwrite(os.path.join(OUT, fn), overview(n, i["e"], i["L"], i["done"], i["open"]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        files[n] = (fn, None)
    for n in pend + fin:
        i = info[n]
        rows.append(dict(foto=n, status=i["e"]["status"], kotak=i["nb"], berlabel=len({x["box"] for x in i["L"]}), kosong=len(i["e"].get("kosong", [])),
                         periksa_terbuka=",".join(map(str, i["open"])), tanpa_label=i["tanpa"], file=files[n][0]))
    with open(os.path.join(OUT, "ringkasan.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = ["<!doctype html><meta charset=utf-8><title>Hasil label tanggal</title><style>body{background:#16181c;color:#ddd;font:14px sans-serif;margin:0;padding:16px}"
           "h2{margin:24px 0 4px}.b{padding:2px 8px;border-radius:9px;font-size:12px}.p{background:#8a6d00}.s{background:#1e6b2e}img{max-width:100%;border:1px solid #333}"
           "a{color:#7ab7ff}.d{color:#aaa;margin:2px 0 6px}</style>",
           f"<h1>Hasil label tanggal ({time.strftime('%d-%m-%Y %H:%M')})</h1><p>Menunggu review: <b>{len(pend)}</b> foto &nbsp; Selesai: <b>{len(fin)}</b> foto. "
           "magenta = label tanggal, cyan = usulan baru, merah = PERIKSA (wajib: beri label atau <code>kosong N</code>), hijau = sudah, abu = tanpa label (boleh buram).</p>"]
    for n in pend + fin:
        i = info[n]
        ds = ", ".join(f"{x['box']}: {x.get('date') or '?'}" for x in sorted(i["L"], key=lambda x: x["box"]))
        out.append(f"<h2>{html.escape(n)} <span class='b {'p' if i['e']['status'] != 'selesai' else 's'}'>{'MENUNGGU REVIEW' if i['e']['status'] != 'selesai' else 'SELESAI'}</span></h2>"
                   f"<div class=d>label {len({x['box'] for x in i['L']})}/{i['nb']} kotak"
                   + (f" &middot; <b style='color:#f66'>PERIKSA box {', '.join(map(str, i['open']))}</b>" if i["open"] else "")
                   + (f" &middot; <a href='{files[n][1]}'>lembar per kotak</a>" if files[n][1] else "")
                   + f"</div><div class=d>tanggal: {html.escape(ds) or '-'}</div><img loading=lazy src='{files[n][0]}'>")
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write("\n".join(out))
    print(f"{len(pend)} foto menunggu review, {len(fin)} selesai -> {OUT}")
    if "--zip" in args:
        z = os.path.expanduser(f"~/Downloads/hasil_tanggal_{time.strftime('%Y%m%d')}")
        shutil.make_archive(z, "zip", os.path.dirname(OUT), os.path.basename(OUT))
        print(f"zip: {z}.zip ({os.path.getsize(z + '.zip') / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
