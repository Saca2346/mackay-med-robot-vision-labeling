#!/usr/bin/env bash
# Mulai training model 3 kelas (box, size_label, expiry_label) di server, HANYA bila semua syarat terpenuhi.
# Setiap pemeriksaan yang gagal menghentikan skrip dengan pesan apa yang harus dibereskan; tidak ada yang dihapus.
#
#   bash tools/mulai_training_3kelas.sh          # periksa, cadangkan hasil lama, lalu mulai training di latar
#   CEK=1 bash tools/mulai_training_3kelas.sh    # hanya periksa (tidak cadangkan, tidak training)
#   MIN_SEED=60 BATCH=8 bash tools/mulai_training_3kelas.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MIN_SEED=${MIN_SEED:-60}     # foto berlabel lengkap minimal (v4 = 144, v6 = 156)
BATCH=${BATCH:-8}            # tetap, bukan -1: GPU dipakai bersama, autobatch bisa mengambil semua VRAM
MIN_VRAM_MB=${MIN_VRAM_MB:-9000}
LOG=logs/train_3kelas.log
source "${VENV:-$HOME/trse_venv}/bin/activate"
mkdir -p logs
gagal() { echo; echo "BERHENTI: $*"; exit 1; }

echo "== 1. tidak ada training lain dari proyek ini =="
ps -C python,python3 -o args= | grep -q "tools/train_best.py" && gagal "train_best.py masih berjalan (lihat: pgrep -af train_best)"
echo "OK"

echo "== 2. kelas =="
[ "$(tr -d '\r' < classes.txt | paste -sd,)" = "box,size_label,expiry_label" ] || gagal "classes.txt bukan box,size_label,expiry_label"
python - <<'EOF' || gagal "tools/common.py CLASSES tidak 3 kelas"
import sys; sys.path.insert(0, "tools")
from common import CLASSES; assert CLASSES == ["box", "size_label", "expiry_label"], CLASSES
EOF
echo "OK"

echo "== 3. geometri label (check_specs) =="
python tools/check_specs.py > logs/check_specs_3kelas.log 2>&1 || { tail -5 logs/check_specs_3kelas.log; gagal "check_specs menemukan masalah baru / crash (detail: logs/check_specs_3kelas.log)"; }
tail -1 logs/check_specs_3kelas.log

echo "== 4. label tanggal: yang sudah direview + label kelas 2 sudah ditulis =="
python - "$MIN_SEED" <<'EOF' || gagal "lihat pesan di atas"
import collections, os, subprocess, sys
sys.path.insert(0, "tools")
from common import RAW, load_expiry
from export_datasets import drop_duplicates, label_dir, seed_state
names, _ = drop_duplicates(sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".jpg")))
why = collections.Counter(); seed = []
for n in names:
    s, info = seed_state(n)
    why[info if s == "pool" else "SEED"] += 1
    if s == "seed":
        seed.append(n)
for k, v in why.most_common():
    print(f"  {v:4d}  {k}")
# class-2 rows must be present in the label file the trainer reads, otherwise dates are learned as background
stale = []
for n in seed:
    want = len(load_expiry(n).get("expiry_label", []))
    have = sum(1 for l in open(os.path.join(label_dir(n), n + ".txt")) if l.split()[:1] == ["2"])
    if have != want:
        stale.append(n)
unres = [n for n in seed if [b for b in load_expiry(n).get("periksa", [])
         if b not in {x.get("box") for x in load_expiry(n).get("expiry_label", [])} | set(load_expiry(n).get("kosong", []))]]
if unres:
    print(f"  {len(unres)} foto 'selesai' masih punya kotak 'periksa' terbuka (tools/cek_tanggal_terlewat.py):", unres[:5])
    sys.exit(1)
if stale:
    print(f"  {len(stale)} foto: label kelas 2 belum ditulis -> make_labels.py")
    subprocess.run([sys.executable, "tools/make_labels.py", *stale], check=True, stdout=subprocess.DEVNULL)
    still = [n for n in stale if label_dir(n).endswith("labels_reviewed_obb")]
    if still:
        print("  foto dengan labels_reviewed_obb (tidak ditulis ulang make_labels), cek kelas 2 manual:", still)
        sys.exit(1)
need = int(sys.argv[1])
print(f"  seed {len(seed)} foto (minimal {need})")
if len(seed) < need:
    print(f"  Review label tanggal dulu: python tools/set_expiry.py NAMA --show / --drop / --add / --ok")
    sys.exit(1)
# --quick = 3 fold by burst; foto dari satu sesi saja -> fold kosong -> ultralytics crash "No images found"
from train_best import make_folds
folds, _ = make_folds(seed, 3)
print(f"  3 fold: {[len(f) for f in folds]} foto")
if min(len(f) for f in folds) < 5:
    print("  Ada fold dengan < 5 foto: review foto dari sesi lain juga (bukan satu sesi saja).")
    sys.exit(1)
EOF

echo "== 5. GPU =="
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "VRAM bebas ${free} MB (minimal ${MIN_VRAM_MB})"
[ "$free" -ge "$MIN_VRAM_MB" ] || gagal "VRAM kurang (dipakai pengguna lain). Coba lagi nanti, atau BATCH=4 MIN_VRAM_MB=6000"
[ -f yolo11s-obb.pt ] || gagal "yolo11s-obb.pt (bobot dasar) tidak ada di folder proyek"

if [ "${CEK:-0}" = 1 ]; then echo; echo "SEMUA SYARAT TERPENUHI (CEK=1: tidak training)"; exit 0; fi

echo "== 6. cadangkan hasil training sebelumnya (train_best menghapus runs/cv, menimpa runs/best_final) =="
B=~/Downloads/cadangan_sebelum_3kelas_$(date +%Y%m%d_%H%M)
mkdir -p "$B"
[ -d runs/best_final ] && cp -a runs/best_final "$B/"
[ -d runs/cv ] && cp -a runs/cv "$B/"
cp -p logs/best_model.json logs/cv_results.json "$B/" 2>/dev/null || true
echo "-> $B ($(du -sh "$B" | cut -f1))"

echo "== 7. mulai training (latar, aman bila jendela ditutup) =="
setsid nohup python tools/train_best.py --quick --models yolo11s-obb.pt@180 --batch "$BATCH" --workers 4 \
  > "$LOG" 2>&1 < /dev/null &
sleep 45
grep -a "foto berlabel" "$LOG" || { tail -20 "$LOG"; gagal "training tidak mulai, lihat $LOG"; }
echo
echo "Training berjalan. Cek:   tail -3 $LOG     |   selesai bila ada baris 'model final'"
echo "Perkiraan 1-1,5 jam (3 fold x 120 epoch + final 200 epoch)."
