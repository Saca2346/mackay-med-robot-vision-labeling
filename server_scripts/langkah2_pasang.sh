#!/usr/bin/env bash
# Step 2: back up v6, install the 3-class package, verify it. Output is read by Claude from the laptop.
set -u
cd ~/Downloads/prelabel_size_label_full || exit 1
mkdir -p ~/Downloads/cadangan_v6
cp -r runs/best_final logs/best_model.json ~/Downloads/cadangan_v6/ 2>&1 || echo "(v6 backup: nothing to copy, continuing)"
unzip -oq ~/Downloads/paket_3kelas_20261011.zip && echo UNZIP OK
sha256sum -c --quiet PAKET_SHA256.txt && echo PAKET OK || echo PAKET GAGAL
sha256sum -c --quiet RAW_SHA256.txt && echo FOTO OK || echo FOTO GAGAL
echo "--- classes.txt:"; cat classes.txt
echo "--- ruang disk:"; df -h ~ | tail -1
echo "--- GPU:"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
