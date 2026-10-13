#!/usr/bin/env python3
"""Bandingkan hasil training terbaru (logs/cv_results.json) dengan v6 (~/Downloads/cadangan_v6/cv_results.json).

  python tools/bandingkan_model.py [BARU.json] [PEMBANDING.json]

Aturan: model baru BOLEH menggantikan v6 hanya bila mAP50-95 dan recall untuk box dan size_label tidak turun lebih
dari TOLERANSI (0,02). Lolos di sini belum cukup: lanjutkan dengan uji keselamatan (0 salah MATCH)."""
import json
import os
import statistics as st
import sys

TOLERANSI = 0.02
baru = sys.argv[1] if len(sys.argv) > 1 else "logs/cv_results.json"
lama = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/Downloads/cadangan_v6/cv_results.json")


def per_kelas(path):
    r = json.load(open(path))
    acc = {}
    for v in r["runs"].values():
        for c, m in v["clean"]["per_class"].items():
            acc.setdefault(c, []).append((m["mAP50_95"], m["R"]))
    return r.get("photos"), {c: (st.mean(x[0] for x in l), st.mean(x[1] for x in l)) for c, l in acc.items()}


nb, b = per_kelas(baru)
nl, l = per_kelas(lama)
print(f"{'kelas':13} {'pembanding (' + str(nl) + ' foto)':>24} {'baru (' + str(nb) + ' foto)':>24}  hasil")
lolos = True
for c in ["box", "size_label", "expiry_label"]:
    if c not in b:
        continue
    o = l.get(c)
    s = f"mAP {b[c][0]:.3f}  R {b[c][1]:.3f}"
    if o is None:
        print(f"{c:13} {'-':>24} {s:>24}  kelas baru")
        continue
    ok = b[c][0] >= o[0] - TOLERANSI and b[c][1] >= o[1] - TOLERANSI
    lolos &= ok
    print(f"{c:13} {f'mAP {o[0]:.3f}  R {o[1]:.3f}':>24} {s:>24}  {'OK' if ok else 'TURUN'}")
print()
if lolos:
    print("LOLOS tahap 1 (box & size_label tidak lebih buruk). Berikutnya: minta Claude menjalankan uji keselamatan")
    print("(eval_verify_detector, harus 0 salah MATCH) sebelum model dipakai.")
else:
    print("TIDAK LOLOS: jangan gantikan v4/v6. Biasanya karena foto yang di-review (--ok) masih sedikit.")
sys.exit(0 if lolos else 1)
