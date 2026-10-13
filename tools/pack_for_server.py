"""Pack labels + tools for a training run on the GPU server (photos are NOT packed: the server already has raw/).

  python tools/pack_for_server.py            -> ../paket_3kelas_YYYYMMDD.zip  (next to this folder)
  python tools/pack_for_server.py --labeling -> ../paket_label_tanggal_YYYYMMDD.zip: tools + labels so far, for
                                               running tools/auto_expiry.py on the server (no training gate)

Contents: tools/, label_specs/, label_specs_expiry/, labels_final_obb/, labels_final/, classes.txt,
check_specs_accepted.json, PAKET_SHA256.txt (every packed file) and RAW_SHA256.txt (every photo in raw/, so the
server can prove its copy of the photos is the same: `sha256sum -c --quiet RAW_SHA256.txt`).
Refuses to pack when the export gate would leave photos out of training (unreviewed expiry_label, specs not done),
so the server never trains on a half-labelled set by accident.
"""
import hashlib
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, RAW, ROOT  # noqa: E402
from export_datasets import drop_duplicates, seed_state  # noqa: E402

DIRS = ["tools", "label_specs", "label_specs_expiry", "labels_final_obb", "labels_final"]
FILES = ["classes.txt", "check_specs_accepted.json", "PANDUAN_TRAINING_3KELAS.md"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    names = sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".jpg"))
    uniq, _ = drop_duplicates(names)
    states = {n: seed_state(n) for n in uniq}
    pool = {n: s[1] for n, s in states.items() if s[0] != "seed"}
    labeling = "--labeling" in sys.argv         # step before training: auto_expiry runs on the server
    if pool and "--force" not in sys.argv and not labeling:
        reasons = {}
        for r in pool.values():
            reasons[r] = reasons.get(r, 0) + 1
        sys.exit(f"BERHENTI: {len(pool)} dari {len(uniq)} foto belum boleh ikut training: {reasons}")
    with open(os.path.join(ROOT, "classes.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(CLASSES) + "\n")
    out = os.path.join(os.path.dirname(ROOT), time.strftime(
        "paket_label_tanggal_%Y%m%d.zip" if labeling else "paket_3kelas_%Y%m%d.zip"))
    files = [f for f in FILES if os.path.exists(os.path.join(ROOT, f))]
    for d in DIRS:
        for base, _, fs in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in base:
                continue
            files += [os.path.relpath(os.path.join(base, f), ROOT).replace("\\", "/") for f in fs]
    lines = [f"{sha(os.path.join(ROOT, f))}  {f}" for f in sorted(files)]
    raw = [f"{sha(os.path.join(RAW, n + '.jpg'))}  raw/{n}.jpg" for n in names]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(os.path.join(ROOT, f), f)
        z.writestr("PAKET_SHA256.txt", "\n".join(lines) + "\n")
        z.writestr("RAW_SHA256.txt", "\n".join(raw) + "\n")
    print(f"{out}: {len(files)} file, {len(uniq) - len(pool)} foto siap training, kelas {CLASSES}, "
          f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
