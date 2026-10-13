"""Review / correct expiry_label proposals (label_specs_expiry/NAME.json, made by tools/auto_expiry.py).

  python tools/set_expiry.py NAME --show               list the labels (box, date, OCR text, score)
  python tools/set_expiry.py NAME --drop N             remove the label on box N (wrong date / not an expiry date)
  python tools/set_expiry.py NAME --add N T1 T2 [V1 V2] [--date YYYY-MM-DD]
                                                       label box N between fractions T1..T2 along it (0 = far end,
                                                       as in tools/grid.py strips) and V1..V2 across (default 0..1)
  python tools/set_expiry.py NAME --date N YYYY-MM-DD   fix only the date text of box N (position unchanged)
  python tools/set_expiry.py NAME --kosong N           box N checked by eye: no readable expiry date (clears a 'periksa' flag)
  python tools/set_expiry.py NAME --ok                 review done: labels become training labels (class 2)
  python tools/set_expiry.py NAME --redo               back to "auto" (not used for training)

After --ok run tools/make_labels.py NAME so labels_final_obb/ gets the class-2 rows.
"""
import json
import os
import re
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXPIRY, load_expiry, load_spec, raw_path  # noqa: E402
from make_labels import resolve_obbs  # noqa: E402
from auto_expiry import label_poly  # noqa: E402


def save(name, e):
    os.makedirs(EXPIRY, exist_ok=True)
    with open(os.path.join(EXPIRY, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(e, fh, ensure_ascii=False, indent=1)


def main(a):
    name, op = a[0], a[1]
    if not os.path.exists(raw_path(name)) or not os.path.exists(os.path.join(os.path.dirname(EXPIRY), "label_specs", name + ".json")):
        sys.exit(f"'{name}' bukan nama foto (contoh: IMG_20260923_132244_872). Tidak ada yang diubah.")
    for i, x in enumerate(a[2:], 2):
        if x in ("N", "T1", "T2", "YYYY-MM-DD") or (op in ("--drop", "--date", "--add", "--kosong") and i == 2 and not x.isdigit()):
            sys.exit(f"'{x}' harus diganti angka/tanggal sungguhan (nomor box, posisi 0..1, YYYY-MM-DD). Tidak ada yang diubah.")
    e = load_expiry(name) or {"expiry_label": [], "status": "auto", "note": ""}
    labels = e["expiry_label"]
    if op == "--show":
        print(f"{name}: status {e.get('status')}, {len(labels)} expiry_label")
        for x in sorted(labels, key=lambda x: x.get("box", 0)):
            print(f"  box {x.get('box')}: {x.get('date', '?')}  ({x.get('text', '')!r}, {x.get('score', '-')})")
        return
    if op == "--drop":
        n = int(a[2])
        e["expiry_label"] = [x for x in labels if x.get("box") != n]
        print(f"{name}: label box {n} dihapus ({len(labels) - len(e['expiry_label'])})")
    elif op == "--add":
        n, t1, t2 = int(a[2]), float(a[3]), float(a[4])
        rest = [x for x in a[5:] if not x.startswith("--")]
        v1, v2 = (float(rest[0]), float(rest[1])) if len(rest) >= 2 and "-" not in rest[0] else (0.0, 1.0)
        date = a[a.index("--date") + 1] if "--date" in a else ""
        img = cv2.imread(raw_path(name))
        H, W = img.shape[:2]
        boxes, _, _ = resolve_obbs(name, load_spec(name), W, H)
        poly = label_poly(boxes[n - 1], t1, t2, v1, v2, W, H)
        e["expiry_label"] = [x for x in labels if x.get("box") != n] + [
            {"poly": poly, "box": n, "date": date, "text": "manual", "score": 1.0}]
        print(f"{name}: label box {n} t {t1}-{t2} v {v1}-{v2} ditambahkan")
    elif op == "--date":
        n, d = int(a[2]), a[3]
        hit = [x for x in labels if x.get("box") == n]
        if not hit or not re.fullmatch(r"20\d\d-\d\d-\d\d", d):
            sys.exit(f"{name}: tidak ada label di box {n}, atau tanggal bukan YYYY-MM-DD. Tidak ada yang diubah.")
        for x in hit:
            x["date"], x["text"] = d, f"dikoreksi dari {x.get('date')}"
        print(f"{name}: tanggal box {n} -> {d} (posisi label tetap)")
    elif op == "--kosong":
        n = int(a[2])
        e["kosong"] = sorted(set(e.get("kosong", [])) | {n})
        print(f"{name}: box {n} dicatat: sudah dilihat, TIDAK ada tanggal yang terbaca")
    elif op == "--ok":
        done = {x.get("box") for x in labels} | set(e.get("kosong", []))
        open_ = [b for b in e.get("periksa", []) if b not in done]
        if open_:
            sys.exit(f"{name}: box {open_} ditandai tajam tanpa label (tools/cek_tanggal_terlewat.py). Beri label "
                     f"(--add / dari usulan) atau --kosong N bila memang tidak ada tanggal terlihat. Status tidak diubah.")
        e["status"] = "selesai"
        print(f"{name}: expiry_label selesai direview ({len(labels)} label) -> jalankan tools/make_labels.py {name}")
    elif op == "--redo":
        e["status"] = "auto"
    else:
        sys.exit(__doc__)
    save(name, e)


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 2 else sys.exit(__doc__)
