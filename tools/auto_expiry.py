"""Propose expiry_label (class 2) for every box from OCR of the full-resolution photo.

Each box of a spec is straightened into a strip (far end left, near end right) and read with RapidOCR. A text line
holding a printed date YYYY-MM-DD becomes an expiry_label covering that line. When a box shows two dates (iVascular
prints the manufacturing date and the expiry date), the LATEST one is the expiry date.

Expiry labels live in a sidecar file per photo, label_specs_expiry/NAME.json, in photo-percent coordinates, so they
survive the many tools that rewrite label_specs (drop_box, refine, transfer_spec renumber boxes):
  {"expiry_label": [{"poly": [x1,y1,...,x4,y4], "box": N, "date": "2029-03-10", "text": "...", "score": 0.98}],
   "status": "auto" | "selesai", "note": ""}
An empty list means: photo checked, no printed date found. A missing file means: not checked yet.
"status": "auto" files are proposals; they become training labels only after review (status "selesai").

Usage (from this folder, with the medicine_shelf_vision venv that has rapidocr + opencv):
  python tools/auto_expiry.py NAME [NAME ...]      (or --all; existing "selesai" files are never overwritten)
  python tools/auto_expiry.py --all --sheet         also write review sheets to preview_expiry/
"""
import datetime as dt
import json
import logging
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SPECS, _unit_normal, load_spec, obb_polygon, raw_path  # noqa: E402
from make_labels import resolve_obbs  # noqa: E402

EXPIRY = os.path.join(ROOT, "label_specs_expiry")
PREVIEW = os.path.join(ROOT, "preview_expiry")
MIN_SCORE = 0.60                      # the date pattern itself is strict (valid YYYY-MM-DD), so a lower OCR score is ok
PAD_T, PAD_V = 0.008, 0.06            # margin around the text line, as a fraction of box length / width
VERSI = 2                             # proposal rules version; photos already done with this version are skipped
PAD_W = 0.25                         # strip is read 25 % wider on each side (dates printed at the spine's edge)

# same rule as medicine_shelf_vision/src/box_expiry.py (kept separate: the two projects do not import each other)
_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "Z": "2", "S": "5",
                        "B": "8"})
_D = r"[0-9OoDQIl|ZSB]"
# one leading digit allowed: the hourglass icon before the date is often read as "8" (82029-09-21, 10-11)
_DATE = re.compile(rf"(?<![0-9])[0-9]?(2{_D}{_D}{_D})\s*[-./\s]\s*({_D}{_D})\s*[-./\s]\s*({_D}{_D})(?![0-9])")


def find_dates(text):
    out = []
    for m in _DATE.finditer(text or ""):
        y, mo, d = (int(g.translate(_DIGIT)) for g in m.groups())
        if 2015 <= y <= 2045:
            try:
                out.append(dt.date(y, mo, d))
            except ValueError:
                pass
    return out


def engine():
    from rapidocr import RapidOCR
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    return RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 2400,
                            "EngineConfig.onnxruntime.intra_op_num_threads": 2})


def _perp(A, B):
    """Unit vector across the box that makes (along, across) a ROTATION of the image axes, not a mirror:
    image x -> strip x along A->B, image y -> strip y along this vector."""
    L = float(np.hypot(B[0] - A[0], B[1] - A[1]))
    ux, uy = (B[0] - A[0]) / L, (B[1] - A[1]) / L
    return (-uy, ux), L


def strip_of(img, obb, max_len=2400, pad=0.0):
    """Box straightened (not mirrored): x = 0 at the far end (A) -> L at the near end (B), y across the width.
    pad: extra width on EACH side as a fraction of the box width (dates are often printed right at the edge, and
    the labelled box can be a few pixels narrower than the real spine)."""
    A, B, hw = obb
    (px, py), L = _perp(A, B)
    k = min(1.0, max_len / L)
    hp = hw * (1 + 2 * pad)
    Ls, Ws = max(8, int(L * k)), max(8, int(2 * hp * k))
    src = np.float32([(A[0] - px * hp, A[1] - py * hp), (B[0] - px * hp, B[1] - py * hp),
                      (B[0] + px * hp, B[1] + py * hp), (A[0] + px * hp, A[1] + py * hp)])
    dst = np.float32([(0, 0), (Ls, 0), (Ls, Ws), (0, Ws)])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (Ls, Ws), flags=cv2.INTER_CUBIC)


def label_poly(ob, t1, t2, v1, v2, W, H):
    """Part of box `ob` between t1..t2 along it and v1..v2 across it, in the strip's frame (strip_of: v = 0 at the
    strip's top edge) -> 8 numbers in percent of the photo."""
    A, B, hw = ob
    A2 = (A[0] + (B[0] - A[0]) * t1, A[1] + (B[1] - A[1]) * t1)
    B2 = (A[0] + (B[0] - A[0]) * t2, A[1] + (B[1] - A[1]) * t2)
    # strip y = v * width runs along _perp from -hw to +hw; obb_polygon offsets run along its own normal
    (px, py), _ = _perp(A, B)
    _, nrm, _ = _unit_normal(A, B)
    sgn = 1.0 if px * nrm[0] + py * nrm[1] > 0 else -1.0
    a1, a2 = sgn * (-hw + 2 * hw * v1), sgn * (-hw + 2 * hw * v2)
    poly = obb_polygon((A2, B2, hw), lo=min(a1, a2), hi=max(a1, a2))
    return [round(c, 3) for x, y in poly for c in (x / W * 100, y / H * 100)]


def propose(name, ocr, img=None):
    spec = load_spec(name)
    img = cv2.imread(raw_path(name)) if img is None else img
    H, W = img.shape[:2]
    boxes, _, _ = resolve_obbs(name, spec, W, H)
    out, strips = [], []
    f = 1 + 2 * PAD_W
    m = re.search(r"(20\d\d)(\d\d)(\d\d)", name)
    shot = dt.date(*map(int, m.groups())) if m else None
    for n, ob in enumerate(boxes, 1):
        s = strip_of(img, ob, pad=PAD_W)
        Ls, Ws = s.shape[1], s.shape[0]
        best = _best_date(_lines(ocr, s), shot)
        if best is None:                                  # small print: read again in 3 tiles, enlarged 2x
            best = _best_date(_tiled_lines(ocr, s), shot)
        # box edges inside the padded strip (for the review sheet)
        e1, e2 = int(PAD_W / f * Ws), int((1 + PAD_W) / f * Ws)
        cv2.line(s, (0, e1), (Ls, e1), (0, 220, 60), 1)
        cv2.line(s, (0, e2), (Ls, e2), (0, 220, 60), 1)
        if best is not None:
            day, q, text, score = best
            t1, t2 = max(0.0, q[:, 0].min() / Ls - PAD_T), min(1.0, q[:, 0].max() / Ls + PAD_T)
            # padded-strip fraction -> fraction of the box width, kept inside the box
            v1 = min(1.0, max(0.0, (q[:, 1].min() / Ws) * f - PAD_W - PAD_V))
            v2 = min(1.0, max(0.0, (q[:, 1].max() / Ws) * f - PAD_W + PAD_V))
            if v2 - v1 < 0.15:                           # date almost entirely outside the box: not labelled
                best = None
            else:
                out.append({"poly": label_poly(ob, t1, t2, v1, v2, W, H), "box": n,
                            "date": day.isoformat(), "text": text, "score": round(score, 3)})
                y1, y2 = int((v1 + PAD_W) / f * Ws), int((v2 + PAD_W) / f * Ws)
                cv2.rectangle(s, (int(t1 * Ls), y1), (int(t2 * Ls), y2), (255, 0, 255), 3)
        strips.append((n, s, best))
    return out, strips


def _lines(ocr, img):
    from rapidocr.utils.process_img import ResizeImgError
    try:
        res = ocr(img)
    except ResizeImgError:            # extremely thin tile (e.g. a box draft over a bottle cap, 14 Oct): no text
        return []
    return [] if res.txts is None else [(np.asarray(q, dtype=float), t, float(s))
                                        for q, t, s in zip(res.boxes, res.txts, res.scores)]


def _tiled_lines(ocr, s, parts=3, overlap=0.25, scale=2.0):
    """OCR in overlapping tiles along the strip, each enlarged; coordinates mapped back to the strip."""
    Ls = s.shape[1]
    step = Ls / (parts - (parts - 1) * overlap)
    out = []
    for i in range(parts):
        x0 = int(i * step * (1 - overlap))
        x1 = min(Ls, int(x0 + step))
        tile = cv2.resize(s[:, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        out += [(q / scale + [x0, 0], t, sc) for q, t, sc in _lines(ocr, tile)]
    return out


def _best_date(lines, after=None):
    """Latest valid date among the OCR lines (expiry is printed after manufacture); None if no date.
    after: photo date - an expiry date is always later (08-10: K10 got its manufacture date 2025-09-22 when only
    that line was read)."""
    best = None
    for q, text, score in lines:
        ds = [d for d in find_dates(text) if after is None or d > after]
        if not ds or score < MIN_SCORE:
            continue
        day = max(ds)
        if best is None or day > best[0]:
            best = (day, q, text, score)
    return best


def sheet(name, strips):
    if not strips:                    # photo without boxes: nothing to show (np.vstack of [] raised, 14 Oct)
        return "no sheet (0 box)"
    os.makedirs(PREVIEW, exist_ok=True)
    rows = []
    for n, s, best in strips:
        r = cv2.resize(s, (1500, max(40, int(s.shape[0] * 1500 / s.shape[1]))))
        tag = f"{name[-10:]} box {n}: " + (f"{best[0]} ({best[3]:.2f})" if best else "-")
        cv2.putText(r, tag, (6, r.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        rows.append(r)
    path = os.path.join(PREVIEW, name + "_expiry.jpg")
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def keep_awake():
    """Windows: keep the PC from sleeping while this process runs (released automatically when it exits).
    08-10: the laptop slept for 7 hours in the middle of a run."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS|SYSTEM_REQUIRED


def main(args):
    keep_awake()
    want_sheet = "--sheet" in args
    names = [a for a in args if not a.startswith("--")]
    if "--all" in args:
        names = sorted(f[:-5] for f in os.listdir(SPECS) if f.endswith(".json"))
    os.makedirs(EXPIRY, exist_ok=True)
    ocr = engine()
    total = 0
    for name in names:
        path = os.path.join(EXPIRY, name + ".json")
        old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        if old.get("status") == "selesai":
            print(f"{name}: sudah selesai direview, dilewati")
            continue
        if old.get("versi") == VERSI and "--ulang" not in args:   # resumable: a stopped run continues here
            continue
        labels, strips = propose(name, ocr)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"expiry_label": labels, "status": "auto", "note": "", "versi": VERSI}, fh,
                      ensure_ascii=False, indent=1)
        total += len(labels)
        extra = f" -> {sheet(name, strips)}" if want_sheet else ""
        print(f"{name}: {len(labels)} expiry_label dari {len(strips)} box "
              f"({', '.join(f'#{e['box']} {e['date']}' for e in labels)}){extra}", flush=True)
    print(f"{len(names)} foto, {total} expiry_label diusulkan")


if __name__ == "__main__":
    main(sys.argv[1:])
