"""Turn labels corrected in CVAT / X-AnyLabeling / Label Studio (labels_reviewed_obb/NAME.txt) back into a label spec.

Why: the expiry-date tools (auto_expiry, review_tanggal, set_expiry, cek_tanggal_terlewat) number the boxes from the
spec, and make_labels (which the 3-class training runs) writes class 2 only into labels_final_obb. A photo whose boxes
were corrected in a tool and imported with import_reviewed.py would keep the old draft boxes for the dates, and
tools/mulai_training_3kelas.sh stops on it ("cek kelas 2 manual").

Per photo:
  1. the spec is rebuilt from the reviewed box / size_label rows (tools/pred_to_spec.py; the old spec is backed up to
     label_specs_backup/), status "selesai", the note says where it came from;
  2. the reviewed file moves to labels_reviewed_obb/_converted/ (the spec is the source from now on);
  3. labels_final_obb / labels_final / preview_final are rewritten (make_labels) and the box and size_label counts are
     checked against the reviewed file;
  4. the expiry proposals were made on the draft boxes, so they are proposed again on the corrected boxes
     (tools/auto_expiry.py --ulang --sheet; status "auto"). Review them next with: python tools/review_tanggal.py
A photo whose expiry labels are already "selesai" is refused: its dates refer to the old box numbers.

Usage: python tools/reviewed_to_spec.py NAME [NAME ...]        (or --batch BATCH_NAME: every photo of that batch
                                                               that has a reviewed file)   [--dry] [--no-dates]
"""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXPIRY, LABELS_OBB, ROOT, load_spec, spec_groups, write_spec  # noqa: E402
from export_datasets import REVIEWED  # noqa: E402
import make_labels  # noqa: E402
import pred_to_spec  # noqa: E402

CONVERTED = os.path.join(REVIEWED, "_converted")


def rows(path):
    """{class id: count} of a YOLO-OBB label file."""
    out = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        v = line.split()
        if len(v) >= 9:
            out[int(float(v[0]))] = out.get(int(float(v[0])), 0) + 1
    return out


def check(name):
    """Reason why `name` cannot be converted, or None."""
    if not os.path.exists(os.path.join(REVIEWED, name + ".txt")):
        return "no labels_reviewed_obb file (import it first: tools/import_reviewed.py)"
    p = os.path.join(EXPIRY, name + ".json")
    if os.path.exists(p) and json.load(open(p, encoding="utf-8")).get("status") == "selesai":
        return "expiry labels already 'selesai' on the old box numbers (set_expiry.py NAME --redo first)"
    return None


def convert(name, dry):
    src = os.path.join(REVIEWED, name + ".txt")
    want = rows(src)
    if want.get(2):
        print(f"{name}: {want[2]} expiry_label row(s) in the reviewed file ignored (dates are done with review_tanggal.py)")
    pred_to_spec.main([name, src, "--box-conf", "0", "--sl-conf", "0"] + (["--dry"] if dry else []))
    if dry:
        return True
    spec = load_spec(name)
    write_spec(name, {"groups": spec_groups(spec), "status": "selesai",
                      "note": f"direview di tool anotasi, diubah ke spec {dt.date.today().isoformat()} "
                              f"(tools/reviewed_to_spec.py; asli: labels_reviewed_obb/_converted/{name}.txt)"})
    os.makedirs(CONVERTED, exist_ok=True)
    shutil.move(src, os.path.join(CONVERTED, name + ".txt"))
    make_labels.main([name])
    have = rows(os.path.join(LABELS_OBB, name + ".txt"))
    ok = have.get(0, 0) == want.get(0, 0) and have.get(1, 0) == want.get(1, 0)
    print(f"{name}: box {have.get(0, 0)}/{want.get(0, 0)}, size_label {have.get(1, 0)}/{want.get(1, 0)} "
          f"{'OK' if ok else '-> CHECK: a size_label lies outside every box (see preview_final)'}")
    return ok


def main(args):
    dry, dates = "--dry" in args, "--no-dates" not in args
    args = [a for a in args if a not in ("--dry", "--no-dates")]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--batch":
        lst = os.path.join(ROOT, "batches", args[1], "list.txt")
        names = [n.strip()[:-4] for n in open(lst, encoding="utf-8") if n.strip()]
        names = [n for n in names if os.path.exists(os.path.join(REVIEWED, n + ".txt"))]
        print(f"batch {args[1]}: {len(names)} photo(s) with a reviewed file")
    else:
        names = [n[:-4] if n.lower().endswith(".jpg") else n for n in args]
    bad = [(n, check(n)) for n in names if check(n)]
    for n, why in bad:
        print(f"{n}: NOT converted - {why}")
    names = [n for n in names if n not in dict(bad)]
    if not names:
        raise SystemExit("nothing to convert")
    ok = [convert(n, dry) for n in names]
    if dry:
        print(f"--dry: {len(names)} photo(s) would be converted; nothing written")
        return
    if dates:
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "auto_expiry.py"), "--ulang", "--sheet", *names],
                       check=True)
    print(f"\n{len(names)} converted, {ok.count(False)} to check. Next: python tools/review_tanggal.py")


if __name__ == "__main__":
    main(sys.argv[1:])
