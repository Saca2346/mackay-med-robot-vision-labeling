"""Mark expiry_label proposals as reviewed for every photo whose coverage (expiry labels / boxes) is high enough.

  python tools/ok_expiry_batch.py --min-coverage 0.8 [--dry]

Why a coverage rule: for training only the label POSITION counts, but a date that is clearly visible and NOT labelled
teaches the model that dates are background. Photos with most boxes labelled are safe; photos below the threshold
(2026-10-11 samples: 6 of 6 photos >= 80 % correct; the 50-79 % band had clearly legible dates missing, e.g. a Sled Bag
date) stay "auto" and out of the 3-class training set until they are completed with tools/set_expiry.py.
The note records the rule and the sample, so the review is not mistaken for a photo-by-photo check.
Undo one photo: tools/set_expiry.py NAME --redo
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXPIRY, load_spec, spec_groups  # noqa: E402

NOTE = ("accepted by coverage rule >= {r:.0%} ({k}/{nb} boxes) on 2026-10-11; sample check: 6 random photos of this band "
        "all correct; NOT a photo-by-photo review")


def main(args):
    r = float(args[args.index("--min-coverage") + 1]) if "--min-coverage" in args else 0.8
    dry = "--dry" in args
    done = skipped = 0
    for f in sorted(glob.glob(os.path.join(EXPIRY, "*.json"))):
        n = os.path.basename(f)[:-5]
        d = json.load(open(f, encoding="utf-8"))
        if d.get("status") == "selesai":
            continue
        nb = sum(len(g.get("box", [])) for g in spec_groups(load_spec(n)))
        k = len(d["expiry_label"])
        if nb and k / nb >= r:
            if not dry:
                d["status"], d["note"] = "selesai", NOTE.format(r=r, k=k, nb=nb)
                json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            done += 1
        else:
            skipped += 1
    print(f"{'would accept' if dry else 'accepted'} {done} photos; {skipped} stay 'auto' (coverage < {r:.0%})")


if __name__ == "__main__":
    main(sys.argv[1:])
