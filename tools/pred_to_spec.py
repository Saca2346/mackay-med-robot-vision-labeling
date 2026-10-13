"""Turn model predictions (or any YOLO-OBB label file) into an editable label spec draft.

Boxes become 8-number entries [far c1, far c2, near c1, near c2] in percent of the photo (far = upper end, or
the left end of a lying box); size_labels become slices {"of": N, "t": [t1, t2]} of the box that contains their
centre (default width 0.05..0.95, like the hand-made specs); size_labels outside every box are dropped.
The spec note starts with "SEDANG DIKERJAKAN" so export_datasets keeps the photo in the pool (out of training)
until it has been checked and the note is replaced (tools/set_note.py).
An existing spec is backed up to label_specs_backup/NAME_<time>.json first.

Usage: python tools/pred_to_spec.py NAME PRED.txt [--box-conf 0.25] [--sl-conf 0.25] [--dry]
  PRED.txt lines: cls x1 y1 x2 y2 x3 y3 x4 y4 [conf]  (normalized)
"""
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al_select import opt  # noqa: E402
from common import ROOT, SPECS, load_raw, write_spec  # noqa: E402
from import_reviewed import inside  # noqa: E402


def ends(poly):
    """Rectangle corners -> ((far c1, far c2), (near c1, near c2)) in pixels."""
    d = lambda a, b: math.dist(poly[a], poly[b])
    if d(0, 1) + d(2, 3) <= d(1, 2) + d(3, 0):
        e1, e2 = (poly[0], poly[1]), (poly[2], poly[3])
    else:
        e1, e2 = (poly[1], poly[2]), (poly[3], poly[0])
    mid = lambda e: ((e[0][0] + e[1][0]) / 2, (e[0][1] + e[1][1]) / 2)
    (ax, ay), (bx, by) = mid(e1), mid(e2)
    vertical = abs(by - ay) >= abs(bx - ax)
    if (ay > by) if vertical else (ax > bx):
        e1, e2 = e2, e1
    k = 0 if vertical else 1                      # order corners of each end left->right (or top->bottom)
    return tuple(sorted(e1, key=lambda p: p[k])), tuple(sorted(e2, key=lambda p: p[k]))


def main(args):
    name, pred = args[0], args[1]
    box_conf, sl_conf = opt(args, "--box-conf", 0.25, float), opt(args, "--sl-conf", 0.25, float)
    _, (W, H) = load_raw(name, 64)
    boxes, sls = [], []
    for line in open(pred).read().splitlines():
        v = line.split()
        if len(v) < 9:
            continue
        c, xy = int(float(v[0])), [float(t) for t in v[1:9]]
        conf = float(v[9]) if len(v) > 9 else 1.0
        poly = [(xy[k] * W, xy[k + 1] * H) for k in range(0, 8, 2)]
        if c == 0 and conf >= box_conf:
            boxes.append((poly, conf))
        elif c == 1 and conf >= sl_conf:
            sls.append((poly, conf))
    boxes.sort(key=lambda b: (sum(p[0] for p in b[0]) / 4))            # left to right, like the previews
    entries = []
    for poly, _ in boxes:
        (f1, f2), (n1, n2) = ends(poly)
        entries.append([round(v, 2) for p in (f1, f2, n1, n2) for v in (p[0] / W * 100, p[1] / H * 100)])
    out_sl, dropped = [], 0
    for poly, _ in sls:
        cx, cy = sum(p[0] for p in poly) / 4, sum(p[1] for p in poly) / 4
        owners = [i for i, (b, _) in enumerate(boxes) if inside((cx, cy), b)]
        if not owners:
            dropped += 1
            continue
        i = min(owners, key=lambda i: abs(math.dist(boxes[i][0][0], boxes[i][0][1]) * math.dist(boxes[i][0][1], boxes[i][0][2])))
        (f1, f2), (n1, n2) = ends(boxes[i][0])
        A = ((f1[0] + f2[0]) / 2, (f1[1] + f2[1]) / 2)
        B = ((n1[0] + n2[0]) / 2, (n1[1] + n2[1]) / 2)
        ux, uy = B[0] - A[0], B[1] - A[1]
        L2 = ux * ux + uy * uy
        ts = [((p[0] - A[0]) * ux + (p[1] - A[1]) * uy) / L2 for p in poly]
        out_sl.append({"of": i + 1, "t": [round(max(0.0, min(ts)), 3), round(min(1.0, max(ts)), 3)]})
    out_sl.sort(key=lambda e: (e["of"], e["t"][0]))
    spec = {"groups": [{"angle": 0, "box": entries, "size_label": out_sl}], "status": "perlu dicek manual",
            "note": f"SEDANG DIKERJAKAN: draf dari prediksi model ({os.path.basename(pred)}, box conf>={box_conf}, "
                    f"size_label conf>={sl_conf}); belum diverifikasi"}
    print(f"{name}: {len(entries)} box, {len(out_sl)} size_label ({dropped} size_label di luar box dibuang)")
    if "--dry" in args:
        return
    path = os.path.join(SPECS, name + ".json")
    if os.path.exists(path):
        bdir = os.path.join(ROOT, "label_specs_backup")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(path, os.path.join(bdir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    write_spec(name, spec)


if __name__ == "__main__":
    main(sys.argv[1:])
