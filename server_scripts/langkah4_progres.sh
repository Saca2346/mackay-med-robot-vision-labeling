#!/usr/bin/env bash
# Progress of the 3-class training (safe to run any time). Progress bars are filtered out.
cd ~/Downloads/prelabel_size_label_full || exit 1
pgrep -af train_best >/dev/null && echo "TRAINING BERJALAN" || echo "TIDAK BERJALAN"
echo "--- foto / fold:"; grep -a -m1 "foto berlabel" logs/train_3kelas.log
echo "--- fold dan model yang sudah dimulai:"; tr "\r" "\n" < logs/train_3kelas.log | grep -a -E "^=== |model final" | tail -6
echo "--- epoch terakhir:"; tr "\r" "\n" < logs/train_3kelas.log | grep -a -o -E "^ +[0-9]+/[0-9]+ +[0-9.]+G" | tail -1
echo "--- validasi terakhir (all = rata-rata semua kelas):"; tr "\r" "\n" < logs/train_3kelas.log | grep -a -E "^ +(all|box|size_label|expiry_label) +[0-9]+ +[0-9]+ " | tail -4
echo "--- hasil per fold (mAP50-95 asli / rusak):"; grep -a "mAP50-95 asli" logs/train_3kelas.log | tail -3 || true
echo "--- hasil cross-validation:"; grep -a -A6 "Hasil cross-validation" logs/train_3kelas.log || echo "(belum selesai)"
echo "--- GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ls -la runs/best_final/box_best.pt 2>/dev/null || echo "(bobot final belum ada)"
