"""Bring labels corrected in CVAT / X-AnyLabeling / Label Studio back into labels_reviewed_obb/.

SOURCE can be
  - a CVAT export zip "Ultralytics YOLO Oriented Bounding Boxes 1.0" (or its unpacked folder),
  - an X-AnyLabeling "Export YOLO Annotations" folder (rotation / OBB task) with classes.txt,
  - a Label Studio export in "JSON" format (.json).
Class ids are remapped by NAME (data.yaml / classes.txt), so a different label order in the tool is
harmless. Every file is checked before anything is written:
  - lines must be OBB (class + 4 corners); axis-aligned exports (class + 4 numbers) are rejected,
  - class names must be box / size_label, coordinates inside the photo, shapes not degenerate,
  - the photo must exist in raw/.
  - a photo must not be identical to its model draft (labels_auto/NAME.txt): an uncorrected draft is
    not a reviewed label. --allow-unchanged accepts such photos when the draft really was right.
Warnings (written anyway): size_labels whose centre lies in no box, a photo without any box.
Per photo it prints how many objects are unchanged / changed or new / removed against the draft.
Photos that already have a verified rotated spec (label_specs/NAME.json, status "selesai") are skipped, since
reviewed labels win over specs; --replace-spec imports them anyway.
Existing reviewed files are backed up to labels_reviewed_obb/_backup/<time>/ before overwriting.

Usage: python tools/import_reviewed.py SOURCE [--dry] [--allow-unchanged] [--replace-spec] [--draft DIR]
Then run: python tools/export_datasets.py
"""
import io
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, RAW, ROOT
from export_datasets import REVIEWED

TOL = 0.01  # normalized coordinates may overshoot the photo border by this much (then clamped)
SAME_PX = 3.0  # corners this close (pixels) to a draft shape count as the unchanged draft shape


def parse_names(text):
    """Class names from data.yaml text or classes.txt text."""
    if "names" in text:
        try:
            import yaml
            names = yaml.safe_load(text)["names"]
        except Exception:
            names = dict(re.findall(r"^\s*(\d+)\s*:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", text, re.M))
        if isinstance(names, dict):
            return {int(k): str(v).strip() for k, v in names.items()}
        return {i: str(v).strip() for i, v in enumerate(names)}
    return {i: ln.strip() for i, ln in enumerate(l for l in text.splitlines() if l.strip())}


def collect_txt(files, read):
    """files: relative paths; read(path) -> text. Returns ({stem: text}, names)."""
    names = None
    for f in files:
        base = os.path.basename(f).lower()
        if base.endswith((".yaml", ".yml")) and names is None:
            names = parse_names(read(f))
    if names is None:
        for f in files:
            if os.path.basename(f).lower() in ("classes.txt", "obj.names"):
                names = parse_names(read(f))
                break
    labels = {}
    for f in files:
        base = os.path.basename(f)
        if base.lower().endswith(".txt") and base.lower() not in ("classes.txt", "train.txt", "val.txt",
                                                                     "test.txt", "obj.names"):
            labels[os.path.splitext(base)[0]] = read(f)
    return labels, names or {i: c for i, c in enumerate(CLASSES)}


def from_yolo_text(text, names, stem, errors):
    rows = []
    for k, line in enumerate(text.splitlines(), 1):
        v = line.split()
        if not v:
            continue
        if len(v) == 5:
            errors.append(f"{stem}: baris {k} format kotak tegak (5 angka) - ekspor ulang sebagai OBB/rotasi")
            continue
        if len(v) not in (9, 10):
            errors.append(f"{stem}: baris {k} jumlah angka {len(v)}, harus 9 (kelas + 4 sudut)")
            continue
        cid = int(float(v[0]))
        if cid not in names:
            errors.append(f"{stem}: baris {k} id kelas {cid} tidak ada di daftar kelas")
            continue
        rows.append((names[cid], [float(t) for t in v[1:9]]))
    return rows


def from_label_studio(path, errors):
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for task in data:
        url = str(task.get("data", {}).get("image", ""))
        stem = os.path.splitext(os.path.basename(url.split("?d=")[-1].split("?")[0]))[0]
        if not os.path.exists(os.path.join(RAW, stem + ".jpg")):
            m = re.match(r"^[0-9a-f]{8}-(.+)$", stem)       # files uploaded to Label Studio get a prefix
            if m:
                stem = m.group(1)
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        ann = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        rows = []
        for r in ann.get("result", []):
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            W, H = r.get("original_width"), r.get("original_height")
            if not (W and H):
                W, H = Image.open(os.path.join(RAW, stem + ".jpg")).size
            x0, y0 = v["x"] / 100 * W, v["y"] / 100 * H
            w, h = v["width"] / 100 * W, v["height"] / 100 * H
            t = math.radians(v.get("rotation", 0))
            u, n = (math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t))
            pts = [(x0, y0), (x0 + w * u[0], y0 + w * u[1]),
                   (x0 + w * u[0] + h * n[0], y0 + w * u[1] + h * n[1]), (x0 + h * n[0], y0 + h * n[1])]
            for lab in v.get("rectanglelabels", []):
                rows.append((lab, [c for x, y in pts for c in (x / W, y / H)]))
        out[stem] = rows
    return out


def inside(pt, poly):
    sign = 0
    for i in range(4):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
        cr = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
        if cr:
            if sign == 0:
                sign = 1 if cr > 0 else -1
            elif (cr > 0) != (sign > 0):
                return False
    return True


def validate(stem, rows, errors, warnings):
    if not os.path.exists(os.path.join(RAW, stem + ".jpg")):
        errors.append(f"{stem}: foto tidak ada di raw/")
        return None
    W, H = Image.open(os.path.join(RAW, stem + ".jpg")).size
    good = []
    for name, v in rows:
        if name not in CLASSES:
            errors.append(f"{stem}: kelas '{name}' bukan box/size_label")
            continue
        if len(v) != 8 or not all(math.isfinite(c) for c in v):
            errors.append(f"{stem}: koordinat {name} tidak valid (harus 8 angka, bukan NaN/Inf)")
            continue
        if any(c < -TOL or c > 1 + TOL for c in v):
            errors.append(f"{stem}: koordinat {name} keluar foto (>1%)")
            continue
        v = [min(max(c, 0.0), 1.0) for c in v]
        pts = [(v[k] * W, v[k + 1] * H) for k in range(0, 8, 2)]
        area = abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))) / 2
        if area < 25:
            cx, cy = sum(x for x, _ in pts) / 4 / W * 100, sum(y for _, y in pts) / 4 / H * 100
            errors.append(f"{stem}: {name} berukuran nol/terlalu kecil ({area:.0f} px2, sekitar x {cx:.0f}% "
                          f"y {cy:.0f}% dari kiri atas foto) - hapus atau gambar ulang")
            continue
        good.append((CLASSES.index(name), v, pts))
    boxes = [p for c, _, p in good if c == 0]
    if not boxes:
        warnings.append(f"{stem}: tidak ada box sama sekali")
    orphans = sum(1 for c, _, p in good if c == 1 and not any(
        inside((sum(x for x, _ in p) / 4, sum(y for _, y in p) / 4), b) for b in boxes))
    if orphans:
        warnings.append(f"{stem}: {orphans} size_label tidak berada di dalam box mana pun")
    return good


def same_shape(a, b):
    return all(min(math.dist(p, q) for q in b) <= SAME_PX for p in a) and \
        all(min(math.dist(p, q) for q in a) <= SAME_PX for p in b)


def draft_diff(stem, good, draft_dir):
    """(unchanged, changed_or_new, removed) against the model draft DRAFT_DIR/STEM.txt, None without draft."""
    path = os.path.join(draft_dir, stem + ".txt")
    if not os.path.exists(path):
        return None
    W, H = Image.open(os.path.join(RAW, stem + ".jpg")).size
    left = []
    for line in open(path).read().splitlines():
        v = line.split()
        if len(v) >= 9:
            xy = [float(t) for t in v[1:9]]
            left.append((int(float(v[0])), [(xy[k] * W, xy[k + 1] * H) for k in range(0, 8, 2)]))
    same = 0
    for c, _, pts in good:
        hit = next((d for d in left if d[0] == c and same_shape(d[1], pts)), None)
        if hit:
            left.remove(hit)
            same += 1
    return same, len(good) - same, len(left)


def finished_spec(stem):
    """Status of a verified rotated spec for `stem` (label_specs/, status 'selesai', not work in progress), or ''.
    Reviewed labels win over specs in export_datasets, so importing over such a photo replaces checked labels."""
    path = os.path.join(ROOT, "label_specs", stem + ".json")
    if not os.path.exists(path):
        return ""
    spec = json.load(open(path, encoding="utf-8"))
    groups = spec.get("groups")
    if not groups or any(len(e) != 8 for g in groups for e in g.get("box", [])):
        return ""
    if spec.get("note", "").startswith("SEDANG DIKERJAKAN") or spec.get("status") != "selesai":
        return ""
    return spec.get("status", "")


def main(args):
    if not args:
        raise SystemExit(__doc__)
    src, dry, allow_unchanged = args[0], "--dry" in args, "--allow-unchanged" in args
    draft_dir = args[args.index("--draft") + 1] if "--draft" in args else os.path.join(ROOT, "labels_auto")
    errors, warnings = [], []
    if src.lower().endswith(".json"):
        per_photo = from_label_studio(src, errors)
    else:
        if src.lower().endswith(".zip"):
            z = zipfile.ZipFile(src)
            files = [f for f in z.namelist() if not f.endswith("/")]
            labels, names = collect_txt(files, lambda f: io.TextIOWrapper(z.open(f), encoding="utf-8").read())
        else:
            files = [os.path.relpath(os.path.join(d, f), src) for d, _, fs in os.walk(src) for f in fs]
            labels, names = collect_txt(files, lambda f: open(os.path.join(src, f), encoding="utf-8").read())
        print("kelas di sumber:", names)
        per_photo = {stem: from_yolo_text(text, names, stem, errors) for stem, text in labels.items()}

    replace_spec = "--replace-spec" in args
    for stem in sorted(per_photo):
        spec_done = finished_spec(stem)
        if spec_done and not replace_spec:
            print(f"  DILEWATI {stem}: sudah punya spec terverifikasi ({spec_done}); label impor akan menimpanya "
                  f"(pakai --replace-spec kalau memang itu maksudnya)")
            del per_photo[stem]
    checked = {stem: validate(stem, rows, errors, warnings) for stem, rows in per_photo.items()}
    print(f"per foto, dibanding draf model di {draft_dir}:")
    for stem in sorted(checked):
        good = checked[stem]
        if good is None:
            continue
        nb = sum(1 for c, _, _ in good if c == 0)
        line = f"  {stem}: box {nb}, size_label {len(good) - nb}"
        diff = draft_diff(stem, good, draft_dir)
        if diff is None:
            line += " | tidak ada draf model untuk dibandingkan"
        else:
            same, new, gone = diff
            line += f" | sama dengan draf {same}, diubah/baru {new}, dihapus dari draf {gone}"
            if same and not new and not gone:
                line += "  <- BELUM DIKOREKSI"
                if not allow_unchanged:
                    errors.append(f"{stem}: sama persis dengan draf model - koreksi dulu (atau --allow-unchanged "
                                  f"bila draf ini memang sudah benar)")
        print(line)
    for w in warnings:
        print("  PERINGATAN", w)
    if errors:
        print(f"{len(errors)} KESALAHAN - tidak ada file yang ditulis:")
        for e in errors:
            print("  ", e)
        raise SystemExit(1)
    if dry:
        print(f"cek selesai (--dry): {len(checked)} foto valid, tidak ada yang ditulis")
        return
    os.makedirs(REVIEWED, exist_ok=True)
    backup = os.path.join(REVIEWED, "_backup", time.strftime("%Y%m%d_%H%M%S"))
    for stem, good in checked.items():
        dst = os.path.join(REVIEWED, stem + ".txt")
        if os.path.exists(dst):
            os.makedirs(backup, exist_ok=True)
            shutil.copy2(dst, os.path.join(backup, stem + ".txt"))
        with open(dst, "w", newline="\n") as fh:
            for c, v, _ in good:
                fh.write(f"{c} " + " ".join(f"{t:.6f}" for t in v) + "\n")
        nb = sum(1 for c, _, _ in good if c == 0)
        print(f"  {stem}.jpg  box={nb} size_label={len(good) - nb}")
    print(f"{len(checked)} foto ditulis ke {REVIEWED}; lanjutkan: python tools/export_datasets.py")


if __name__ == "__main__":
    main(sys.argv[1:])
