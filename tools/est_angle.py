"""Estimate the dominant edge direction inside a region of a photo.

Usage: python tools/est_angle.py NAME x1,x2,y1,y2 [x1,x2,y1,y2 ...]   (percent of the original photo)
Prints the strongest line directions (degrees clockwise from horizontal, 0-180) and the
--angle that makes each direction vertical ("rot_v") or horizontal ("rot_h").
Spine edges and the text printed along a spine share one direction, so the strongest peak
is usually the spine axis or the perpendicular box-end edges; confirm visually with grid.py.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_raw


def peaks(name, region, n=3):
    im, (W, H) = load_raw(name)
    x1, x2, y1, y2 = region
    crop = im.crop((W * x1 / 100, H * y1 / 100, W * x2 / 100, H * y2 / 100)).convert("L")
    s = 900 / max(crop.size)
    if s < 1:
        crop = crop.resize((round(crop.size[0] * s), round(crop.size[1] * s)))
    a = np.asarray(crop, dtype=np.float32)
    gy, gx = np.gradient(a)
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 90)
    m = mag > thr
    ang = (np.degrees(np.arctan2(gy[m], gx[m])) + 90.0) % 180.0  # edge line direction
    w = mag[m] ** 2
    hist, edges = np.histogram(ang, bins=360, range=(0, 180), weights=w)
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    hs = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), k / k.sum(), mode="valid")
    out = []
    used = np.zeros_like(hs, dtype=bool)
    for _ in range(n):
        hs2 = np.where(used, -1, hs)
        i = int(np.argmax(hs2))
        # refine with weighted mean of neighbours
        idx = [(i + j) % 360 for j in range(-3, 4)]
        centers = (edges[:-1] + 0.25)
        vals = hs[idx]
        cs = np.array([centers[j] for j in idx])
        if i < 3 or i > 356:
            cs = np.where(cs > 90, cs - 180, cs) if i < 3 else np.where(cs < 90, cs + 180, cs)
        theta = float((cs * vals).sum() / vals.sum()) % 180
        out.append((theta, float(hs[i] / hs.max())))
        for j in range(-12, 13):
            used[(i + j) % 360] = True
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    for r in sys.argv[2:]:
        region = tuple(float(v) for v in r.split(","))
        res = peaks(name, region)
        txt = "  ".join(f"theta={t:6.2f} (rot_v={t - 90:+6.2f}, rot_h={t if t <= 90 else t - 180:+6.2f}) w={w:.2f}"
                        for t, w in res)
        print(f"{r}: {txt}")
