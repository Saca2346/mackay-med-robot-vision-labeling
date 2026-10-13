#!/usr/bin/env bash
# What changed on the server copy of the project (another Claude session works there). Read-only.
cd ~/Downloads/prelabel_size_label_full || exit 1
echo "--- file diubah sejak 11 Okt 16:40 (tanpa raw/ runs/ __pycache__):"
find . -newermt "2026-10-11 16:40" -type f ! -path "./raw/*" ! -path "./runs/*" ! -path "*/__pycache__/*" ! -path "./logs/*" ! -path "./preview_expiry/*" | sort | head -60
echo "--- jumlah per folder:"; find . -newermt "2026-10-11 16:40" -type f ! -path "./raw/*" ! -path "./runs/*" ! -path "*/__pycache__/*" | cut -d/ -f2 | sort | uniq -c
echo "--- status label tanggal di server:"
python3 - <<'PY'
import json, glob, collections
c = collections.Counter(json.load(open(f)).get("status") for f in glob.glob("label_specs_expiry/*.json"))
print(dict(c))
PY
echo "--- classes.txt:"; cat classes.txt
echo "--- folder hasil_tanggal:"; ls hasil_tanggal 2>/dev/null; for d in hasil_tanggal/*/; do echo "$d $(ls "$d" | wc -l)"; done 2>/dev/null
echo "--- file teks baru (rencana / catatan) di folder utama:"; ls -la --time-style=+%d-%m_%H:%M *.md *.txt 2>/dev/null | awk "{print \$6, \$7}" | sort | tail -8
