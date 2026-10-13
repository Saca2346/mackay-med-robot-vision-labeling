"""Export the verified labels for model-assisted labeling (CVAT / X-AnyLabeling / Label Studio)
and for training the seed model used in the active-learning loop.

Seed set  = photos with a reviewed label file in labels_reviewed_obb/ (corrections made in CVAT /
            X-AnyLabeling / Label Studio, exported as YOLO-OBB; these win), plus photos whose spec uses
            the rotated format (8-number boxes) and is not work-in-progress.
Pool      = every other photo (unlabeled, old axis-aligned specs, or work-in-progress); these are the
            photos the seed model pre-labels and active learning ranks.
Everything under export/ is regenerated on each run; keep reviewed labels in labels_reviewed_obb/.

Outputs (export/):
  manifest.csv                       per photo: category, split/pool, status, box/size_label counts
  yolo_obb/images|labels/{train,val} Ultralytics YOLO-OBB dataset of the seed set + data.yaml
  yolo_obb/pool.txt                  absolute paths of the pool images (input for tools/al_select.py)
  cvat/annotations.xml               CVAT for images 1.1, rotated boxes (seed set)
  cvat/ultralytics_obb_labels.zip    "Ultralytics YOLO Oriented Bounding Boxes 1.0" import zip (seed set)
  label_studio/label_config.xml      RectangleLabels with rotation
  label_studio/tasks_seed.json       seed photos with their labels as annotations
  xanylabeling/labels/*.txt + classes.txt   YOLO-OBB labels for X-AnyLabeling "Upload YOLO-OBB"

Usage: python tools/export_datasets.py [--ls-prefix "/data/local-files/?d=raw/"] [--copy]
  --copy       copy images into yolo_obb/ instead of hard-linking them
  --ls-prefix  URL prefix Label Studio uses to reach raw/NAME.jpg
"""
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile
from xml.sax.saxutils import quoteattr

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, LABELS_OBB, RAW, ROOT, SPECS, group_bursts, load_expiry, load_spec, spec_groups

EXPORT = os.path.join(ROOT, "export")
REVIEWED = os.path.join(ROOT, "labels_reviewed_obb")
# photo categories by 1-based position in the sorted raw/ list (see labeling notes)
CLOSE_UP = range(32, 72)       # 13:25:12 .. 13:38:43, text and sizes large
SEMI_CLOSE_UP = range(28, 32)  # one package pulled out, whole row visible
VAL_EVERY = 6                  # ~1 in 6 seed bursts goes to val (fixed per burst)
BURST_S = 5.0                  # shots less than this many seconds apart are near-duplicates (see train_best.py)


def category(idx):
    if idx in CLOSE_UP:
        return "close_up"
    if idx in SEMI_CLOSE_UP:
        return "semi_close_up"
    return "wide"


def label_dir(name):
    """Folder holding the YOLO-OBB labels used for `name`: reviewed labels win over generated ones."""
    return REVIEWED if os.path.exists(os.path.join(REVIEWED, name + ".txt")) else LABELS_OBB


def seed_state(name):
    """('seed', status) for reviewed labels or a verified rotated spec, ('pool', reason) otherwise."""
    if label_dir(name) == REVIEWED:
        # same expiry rule as below: an unlabelled printed date would be learned as "not an expiry_label" (14 Oct)
        return _expiry_state(name, "direview di tool anotasi")
    p = os.path.join(SPECS, name + ".json")
    if not os.path.exists(p):
        return "pool", "belum dilabel"
    s = load_spec(name)
    if "groups" not in s or any(len(e) != 8 for g in s["groups"] for e in g.get("box", [])):
        return "pool", "format lama (kotak tegak), belum dikonversi"
    if s.get("note", "").startswith("SEDANG DIKERJAKAN"):
        return "pool", "sedang dikerjakan"
    status = s.get("status", "selesai")
    if status != "selesai":
        return "pool", status
    return _expiry_state(name, status)


def _expiry_state(name, status):
    # every printed date must be labelled before a photo trains the 3-class model: an unlabelled date would be
    # learned as "not an expiry_label" (see tools/auto_expiry.py)
    e = load_expiry(name)
    if e is None:
        return "pool", "expiry_label belum dicek (tools/auto_expiry.py)"
    if e.get("status") != "selesai":
        return "pool", "expiry_label belum direview"
    return "seed", status


def read_obb(name):
    rows = []
    with open(os.path.join(label_dir(name), name + ".txt")) as fh:
        for line in fh.read().splitlines():
            parts = line.split()
            if len(parts) == 9:
                rows.append((int(parts[0]), [float(v) for v in parts[1:]]))
    return rows


def rect_params(poly):
    """Pixel polygon (4 corners of a rectangle) -> (p0, w, h, theta_deg).

    The rectangle is p0 + s*w*u + t*h*v (s, t in 0..1) with u at theta (clockwise on screen) and
    v = u turned +90 deg on screen. The direction comes from the mean of the two long sides, so a
    corner clamped to the photo border (up to ~1 px) does not tilt the result."""
    p = poly
    sides = [(p[(k + 1) % 4][0] - p[k][0], p[(k + 1) % 4][1] - p[k][1]) for k in range(4)]
    lens = [math.hypot(*s) for s in sides]
    k = 0 if lens[0] + lens[2] >= lens[1] + lens[3] else 1
    ax, ay = sides[k][0] - sides[k + 2][0], sides[k][1] - sides[k + 2][1]   # the two long sides run opposite
    n = math.hypot(ax, ay)
    ax, ay = ax / n, ay / n
    bx, by = -ay, ax
    L, S = (lens[k] + lens[k + 2]) / 2, (lens[k + 1] + lens[(k + 3) % 4]) / 2
    cx, cy = sum(q[0] for q in p) / 4, sum(q[1] for q in p) / 4
    p0 = (cx - L / 2 * ax - S / 2 * bx, cy - L / 2 * ay - S / 2 * by)
    return p0, L, S, math.degrees(math.atan2(ay, ax)) % 360


def write_cvat_xml(path, items, source="manual"):
    """items: [(name, W, H, [(cls, poly_px, conf|None), ...]), ...] -> CVAT for images 1.1 XML."""
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
           "  <meta><task><labels>"]
    out += [f"    <label><name>{c}</name><type>rectangle</type><attributes></attributes></label>" for c in CLASSES]
    out += ["  </labels></task></meta>"]
    for i, (n, W, H, rows) in enumerate(items):
        out.append(f'  <image id="{i}" name={quoteattr(n + ".jpg")} width="{W}" height="{H}">')
        for c, poly, _ in rows:
            _, w, h, th = rect_params(poly)
            cx = sum(x for x, _ in poly) / 4
            cy = sum(y for _, y in poly) / 4
            out.append(f'    <box label="{CLASSES[c]}" source="{source}" occluded="0" '
                       f'xtl="{cx - w / 2:.2f}" ytl="{cy - h / 2:.2f}" xbr="{cx + w / 2:.2f}" '
                       f'ybr="{cy + h / 2:.2f}" rotation="{th:.2f}" z_order="0"></box>')
        out.append("  </image>")
    out.append("</annotations>")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")


def ls_result(rows, W, H):
    """[(cls, poly_px, conf|None), ...] -> Label Studio rectanglelabels results (rotated)."""
    result = []
    for k, (c, poly, conf) in enumerate(rows):
        p0, w, h, th = rect_params(poly)
        r = {"id": f"r{k}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
             "original_width": W, "original_height": H, "image_rotation": 0,
             "value": {"x": p0[0] / W * 100, "y": p0[1] / H * 100, "width": w / W * 100,
                       "height": h / H * 100, "rotation": th, "rectanglelabels": [CLASSES[c]]}}
        if conf is not None:
            r["score"] = conf
        result.append(r)
    return result


def drop_duplicates(names):
    """Byte-identical photos (e.g. 'X (1).jpg' copies made while moving files) count once.

    Per group of identical files the kept name is the labeled one, else the plainest name.
    Returns (unique names in sorted order, [(dropped, kept), ...])."""
    groups = {}
    for n in names:
        with open(os.path.join(RAW, n + ".jpg"), "rb") as fh:
            groups.setdefault(hashlib.sha1(fh.read()).hexdigest(), []).append(n)
    keep, dups = set(), []
    for members in groups.values():
        best = min(members, key=lambda n: (seed_state(n)[0] != "seed", "(" in n, len(n), n))
        keep.add(best)
        dups += [(n, best) for n in members if n != best]
    return [n for n in names if n in keep], sorted(dups)


def val_split(names):
    """Assign each name to 'train' or 'val', whole bursts (shots < BURST_S apart) together so a
    near-duplicate of a validation photo never leaks into train. ~1 in VAL_EVERY bursts goes to val,
    picked by hashing the burst's first name (stable across runs, independent of overall photo order)."""
    assignment = {}
    for b in group_bursts(names, BURST_S):
        split = "val" if int(hashlib.md5(b[0].encode()).hexdigest(), 16) % VAL_EVERY == 0 else "train"
        for n in b:
            assignment[n] = split
    return assignment


def link_or_copy(src, dst, copy):
    if os.path.exists(dst):
        os.remove(dst)
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main(args):
    copy = "--copy" in args
    ls_prefix = "/data/local-files/?d=raw/"
    if "--ls-prefix" in args:
        ls_prefix = args[args.index("--ls-prefix") + 1]

    names = sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".jpg"))
    yolo = os.path.join(EXPORT, "yolo_obb")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(yolo, sub)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for sub in ("cvat", "label_studio", "xanylabeling/labels"):
        os.makedirs(os.path.join(EXPORT, sub), exist_ok=True)
    for f in os.listdir(os.path.join(EXPORT, "xanylabeling/labels")):
        os.remove(os.path.join(EXPORT, "xanylabeling/labels", f))

    names, dups = drop_duplicates(names)
    for d, keep in dups:
        print(f"duplikat diabaikan: {d}.jpg (isi sama dengan {keep}.jpg)")

    seed_names = [n for n in names if seed_state(n)[0] == "seed"]
    val_for = val_split(seed_names)

    manifest, seed, pool = [], [], []
    for idx, n in enumerate(names, 1):
        state, info = seed_state(n)
        rec = {"no": idx, "nama_file": n + ".jpg", "kategori": category(idx), "status": info,
               "jumlah_box": "", "jumlah_size_label": "", "split": "pool"}
        if state == "seed":
            rows = read_obb(n)
            rec["jumlah_box"] = sum(1 for c, _ in rows if c == 0)
            rec["jumlah_size_label"] = sum(1 for c, _ in rows if c == 1)
            rec["split"] = val_for[n]
            seed.append((n, rec, rows))
        else:
            pool.append(n)
        manifest.append(rec)

    with open(os.path.join(EXPORT, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    # --- Ultralytics YOLO-OBB dataset (seed) --------------------------------------------------
    for n, rec, _ in seed:
        sp = rec["split"]
        link_or_copy(os.path.join(RAW, n + ".jpg"), os.path.join(yolo, "images", sp, n + ".jpg"), copy)
        shutil.copy2(os.path.join(label_dir(n), n + ".txt"), os.path.join(yolo, "labels", sp, n + ".txt"))
        shutil.copy2(os.path.join(label_dir(n), n + ".txt"), os.path.join(EXPORT, "xanylabeling/labels", n + ".txt"))
    with open(os.path.join(yolo, "data.yaml"), "w", newline="\n") as fh:
        fh.write(f"path: {yolo.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\n"
                 "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    for d in (yolo, os.path.join(EXPORT, "xanylabeling/labels")):
        with open(os.path.join(d, "classes.txt"), "w", newline="\n") as fh:
            fh.write("\n".join(CLASSES) + "\n")
    with open(os.path.join(yolo, "pool.txt"), "w", newline="\n") as fh:
        fh.write("".join(os.path.join(RAW, n + ".jpg").replace(os.sep, "/") + "\n" for n in pool))

    # --- CVAT -------------------------------------------------------------------------------------
    sizes = {n: Image.open(os.path.join(RAW, n + ".jpg")).size for n, _, _ in seed}
    px_rows = {}
    for n, _, rows in seed:
        W, H = sizes[n]
        px_rows[n] = [(c, [(v[k] * W, v[k + 1] * H) for k in range(0, 8, 2)], None) for c, v in rows]
    write_cvat_xml(os.path.join(EXPORT, "cvat", "annotations.xml"),
                   [(n, *sizes[n], px_rows[n]) for n, _, _ in seed])

    zpath = os.path.join(EXPORT, "cvat", "ultralytics_obb_labels.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.yaml", "path: .\ntrain: train.txt\nnames:\n"
                   + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
        z.writestr("train.txt", "".join(f"images/train/{n}.jpg\n" for n, _, _ in seed))
        for n, _, _ in seed:
            z.write(os.path.join(label_dir(n), n + ".txt"), f"labels/train/{n}.txt")

    # --- Label Studio -----------------------------------------------------------------------------
    with open(os.path.join(EXPORT, "label_studio", "label_config.xml"), "w", newline="\n") as fh:
        fh.write('<View>\n  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
                 '  <RectangleLabels name="label" toName="image" canRotate="true">\n'
                 + "".join(f'    <Label value="{c}"/>\n' for c in CLASSES)
                 + "  </RectangleLabels>\n</View>\n")
    tasks = []
    for n, rec, _ in seed:
        tasks.append({"data": {"image": ls_prefix + n + ".jpg", "kategori": rec["kategori"],
                               "status": rec["status"]},
                      "annotations": [{"result": ls_result(px_rows[n], *sizes[n]),
                                       "ground_truth": rec["status"] == "selesai"}]})
    with open(os.path.join(EXPORT, "label_studio", "tasks_seed.json"), "w", encoding="utf-8") as fh:
        json.dump(tasks, fh, ensure_ascii=False, indent=1)

    n_tr = sum(1 for _, r, _ in seed if r["split"] == "train")
    print(f"seed: {len(seed)} photos (train {n_tr}, val {len(seed) - n_tr}), pool: {len(pool)} photos")
    print(f"boxes: {sum(r['jumlah_box'] for _, r, _ in seed)}, size_labels: "
          f"{sum(r['jumlah_size_label'] for _, r, _ in seed)}")
    print("written to", EXPORT)


if __name__ == "__main__":
    main(sys.argv[1:])
