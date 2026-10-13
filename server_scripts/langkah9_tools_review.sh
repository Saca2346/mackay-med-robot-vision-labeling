#!/bin/bash
# Install the review tools of 14 Oct on the server (CHANGES the server; a backup is made first):
#   tools/reviewed_to_spec.py  (new) CVAT / X-AnyLabeling result -> label spec, then new date proposals
#   tools/review_tanggal.py    skips photos whose boxes are still a model draft; prints the sheet path; counts all photos
# Needs ~/Downloads/tools_review_20261014.tgz (uploaded by Claude with scp).
# Undo: tar xzf ~/Downloads/cadangan_sebelum_tools_review_20261014.tgz && rm tools/reviewed_to_spec.py
set -e
cd ~/Downloads/prelabel_size_label_full || { echo "folder not found"; exit 1; }
P=~/Downloads/tools_review_20261014.tgz
[ -e "$P" ] || { echo "missing $P"; exit 1; }
tar czf ~/Downloads/cadangan_sebelum_tools_review_20261014.tgz tools/review_tanggal.py
tar xzf "$P"
source ~/trse_venv/bin/activate 2>/dev/null || true
python3 -c "import sys; sys.path.insert(0, 'tools'); import reviewed_to_spec, review_tanggal" 2>&1 | head -3 || true
python3 tools/reviewed_to_spec.py 2>&1 | head -1
printf 'keluar\n' | python3 tools/review_tanggal.py 2>/dev/null | grep "=====" | head -1
echo done
