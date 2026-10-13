#!/bin/bash
# Apply the laptop package of 14 Oct on the server (CHANGES the server; a backup is made first).
# Needs in ~/Downloads (uploaded by Claude with scp): sync_laptop_20261014.tgz, sync_list.txt, new33.txt.
# Contents: laptop labeling_progress.csv (7 photos re-checked 8 Oct), the 33 New image photos of 28 Sep 16h
# (drafts, expiry proposals, previews), fixed tools (auto_expiry, export_datasets), compare_server.py,
# server_scripts, README and docs. Undo: tar xzf ~/Downloads/cadangan_sebelum_sync_laptop_20261014.tgz
# (and remove the 33 photos from raw/).
set -e
cd ~/Downloads/prelabel_size_label_full || { echo "folder not found"; exit 1; }
D=~/Downloads
for f in sync_laptop_20261014.tgz sync_list.txt new33.txt; do [ -e "$D/$f" ] || { echo "missing $D/$f"; exit 1; }; done
sed -i 's/\r$//' "$D/sync_list.txt" "$D/new33.txt"
if [ -e "$D/cadangan_sebelum_sync_laptop_20261014.tgz" ]; then echo "already applied (backup exists); nothing done"; exit 1; fi

# 1. backup of every file that will be overwritten
while read -r p; do [ -n "$p" ] && [ -e "$p" ] && echo "$p"; done < "$D/sync_list.txt" > "$D/sync_existing.txt" || true
tar czf "$D/cadangan_sebelum_sync_laptop_20261014.tgz" -T "$D/sync_existing.txt"
echo "backup: $(wc -l < "$D/sync_existing.txt") files -> $D/cadangan_sebelum_sync_laptop_20261014.tgz"

# 2. the 33 photos: from New image/ into raw/ (never overwrite)
n=0
while read -r f; do
  [ -n "$f" ] || continue
  if [ -e "raw/$f.jpg" ]; then echo "raw exists, kept: $f"; continue; fi
  cp "New image/$f.jpg" raw/
  n=$((n + 1))
done < "$D/new33.txt"
echo "photos copied to raw/: $n (raw/ now $(ls raw | wc -l) files)"

# 3. the laptop files
tar xzf "$D/sync_laptop_20261014.tgz"
echo "extracted $(tar tzf "$D/sync_laptop_20261014.tgz" | wc -l) files"

# 4. rebuild the review batch here (images are links to raw/)
source ~/trse_venv/bin/activate 2>/dev/null || true
python3 tools/batch.py baru_20260928_16h --review | head -3
echo "expiry status: $(grep -ho '"status": "[a-z]*"' label_specs_expiry/*.json | sort | uniq -c | tr '\n' ' ')"
echo done
