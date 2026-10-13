"""Build a ready-to-import work folder (a "batch") for CVAT, X-AnyLabeling and Label Studio.

A batch holds only the photos to review, so every tool gets exactly matching images and labels:
  batches/<NAME>/images/NAME.jpg          photos (hard links to raw/, or copies)
  batches/<NAME>/labels/NAME.txt          YOLO-OBB labels + classes.txt   (X-AnyLabeling import)
  batches/<NAME>/cvat_annotations.xml     CVAT for images 1.1, rotated boxes   (CVAT upload)
  batches/<NAME>/label_studio_tasks.json  tasks; verified labels as annotations, model output as
                                          predictions                        (Label Studio import)
  batches/<NAME>/list.txt                 the photo names
Labels per photo: labels_reviewed_obb/ > labels_final_obb/ (verified rotated spec) > none.
tools/al_select.py builds its batch with this module too (model pre-labels as predictions).

Usage: python tools/batch.py BATCH_NAME (--review | --list FILE | NAME [NAME ...]) [--copy]
                             [--ls-prefix "/data/local-files/?d=raw/"]
  --review   photos whose spec status is "perlu dicek manual" (not in the seed set; draft labels shown as predictions)
  --list     text file with one photo name per line (with or without .jpg)
Afterwards bring the corrections back with tools/import_reviewed.py.
"""
import json
import os
import shutil
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, RAW, ROOT, SPECS, load_spec
from export_datasets import label_dir, link_or_copy, ls_result, read_obb, seed_state, write_cvat_xml

BATCHES = os.path.join(ROOT, "batches")
LS_PREFIX = "/data/local-files/?d=raw/"


def verified_rows(name, W, H):
    """[(cls, poly_px, None)] from reviewed/generated labels, or None when the photo has none."""
    state, _ = seed_state(name)
    if state != "seed":
        return None
    return [(c, [(v[k] * W, v[k + 1] * H) for k in range(0, 8, 2)], None) for c, v in read_obb(name)]


def make_batch(batch, names, predictions=None, copy=False, ls_prefix=LS_PREFIX, model_version="model"):
    """predictions: {name: [(cls, poly_px, conf)]} used for photos without verified labels."""
    out = os.path.join(BATCHES, batch)
    for n in names:                            # validate every input before touching any output
        if not os.path.exists(os.path.join(RAW, n + ".jpg")):
            raise SystemExit(f"foto tidak ada di raw/: {n}")
    for sub in ("images", "labels"):
        d = os.path.join(out, sub)
        if os.path.isdir(d):
            # reused batch name with a different photo list: without this, members of the old list
            # that aren't in the new one would linger in images/labels alongside list.txt/annotations
            backup = os.path.join(out, "_backup", time.strftime("%Y%m%d_%H%M%S"), sub)
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            shutil.move(d, backup)
        os.makedirs(d)
    with open(os.path.join(out, "labels", "classes.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(CLASSES) + "\n")
    items, tasks, summary = [], [], []
    for n in names:
        src = os.path.join(RAW, n + ".jpg")
        W, H = Image.open(src).size
        link_or_copy(src, os.path.join(out, "images", n + ".jpg"), copy)
        rows, kind = verified_rows(n, W, H), "verified"
        if rows is None:
            rows, kind = (predictions or {}).get(n, []), "prediction"
        with open(os.path.join(out, "labels", n + ".txt"), "w", newline="\n") as fh:
            for c, poly, _ in rows:
                fh.write(f"{c} " + " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}"
                                            for x, y in poly) + "\n")
        items.append((n, W, H, rows))
        task = {"data": {"image": ls_prefix + n + ".jpg"}}
        if rows and kind == "verified":
            task["annotations"] = [{"result": ls_result(rows, W, H)}]
        elif rows:
            confs = [r[2] for r in rows if r[2] is not None]
            task["predictions"] = [{"model_version": model_version,
                                    "score": sum(confs) / len(confs) if confs else 0.0,
                                    "result": ls_result(rows, W, H)}]
        tasks.append(task)
        summary.append((n, kind if rows else "kosong", len(rows)))
    write_cvat_xml(os.path.join(out, "cvat_annotations.xml"), items,
                   source="manual" if all(k == "verified" for _, k, _ in summary) else "auto")
    with open(os.path.join(out, "label_studio_tasks.json"), "w", encoding="utf-8") as fh:
        json.dump(tasks, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "list.txt"), "w", newline="\n") as fh:
        fh.write("".join(n + ".jpg\n" for n in names))
    return out, summary


def main(args):
    if not args:
        raise SystemExit(__doc__)
    copy = "--copy" in args
    args = [a for a in args if a != "--copy"]
    ls_prefix = LS_PREFIX
    if "--ls-prefix" in args:
        i = args.index("--ls-prefix")
        ls_prefix = args[i + 1]
        del args[i:i + 2]
    batch, rest = args[0], args[1:]
    predictions = None
    if rest == ["--review"]:
        # "perlu dicek manual" photos are kept out of the seed set (see export_datasets.seed_state), so they are
        # selected here by spec status; their current draft labels go into the batch as editable predictions.
        names, predictions = [], {}
        for f in sorted(os.listdir(RAW)):
            n = f[:-4]
            if f.endswith(".jpg") and os.path.exists(os.path.join(SPECS, n + ".json")) \
                    and label_dir(n).endswith("labels_final_obb") and load_spec(n).get("status") == "perlu dicek manual":
                names.append(n)
                W, H = Image.open(os.path.join(RAW, f)).size
                predictions[n] = [(c, [(v[k] * W, v[k + 1] * H) for k in range(0, 8, 2)], None) for c, v in read_obb(n)]
    elif rest[:1] == ["--list"]:
        names = [ln.strip() for ln in open(rest[1], encoding="utf-8") if ln.strip()]
    else:
        names = rest
    names = [n[:-4] if n.lower().endswith(".jpg") else n for n in names]
    out, summary = make_batch(batch, names, predictions=predictions, copy=copy, ls_prefix=ls_prefix)
    print(f"batch {batch}: {len(names)} foto -> {out}")
    for n, kind, k in summary:
        print(f"  {n}.jpg  {kind}  {k} objek")


if __name__ == "__main__":
    main(sys.argv[1:])
