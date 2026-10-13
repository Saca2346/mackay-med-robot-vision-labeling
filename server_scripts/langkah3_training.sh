#!/usr/bin/env bash
# Step 3: check the labels, then start the 3-class training in the background (survives a closed desktop / ssh).
set -u
cd ~/Downloads/prelabel_size_label_full || exit 1
source ~/trse_venv/bin/activate
echo "--- check_specs:"; python tools/check_specs.py | tail -1
if pgrep -af train_best >/dev/null; then echo "TRAINING SUDAH BERJALAN:"; pgrep -af train_best; exit 0; fi
mkdir -p logs
nohup python tools/train_best.py --quick --models yolo11s-obb.pt@180 > logs/train_3kelas.log 2>&1 &
echo "training dimulai, PID $!"
sleep 90
echo "--- log:"; tail -6 logs/train_3kelas.log
