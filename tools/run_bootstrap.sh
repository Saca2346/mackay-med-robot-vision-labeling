#!/usr/bin/env bash
# One bootstrap round on the Ubuntu GPU machine, start to finish:
#   0 verify the copy   1 venv + CUDA   2 labels from specs   3 spec check   4 dataset
#   5 GPU training      6 per-class val metrics   6b calibrate draft/flag thresholds on val
#   7 drafts + review priority   8 summary
#
# Usage:  bash tools/run_bootstrap.sh [ROUND]      ROUND 1 -> runs/bootstrap_v1, batches/al_round1
# Environment (optional):
#   EPOCHS=80 IMGSZ=1280 MODEL=yolo11s-obb.pt BATCH=-1 DEVICE=0 K=10 VENV=~/trse_venv
#   KEEP=0.2    fixed draft threshold instead of the calibrated one
#   RETRAIN=1   train again even if runs/bootstrap_vN/weights/best.pt exists
#   TEST_MODE=1 PRED_DIR=dir   skip venv/CUDA/training/val and rank with prediction txt files (tests)
# Everything is logged to logs/bootstrap_vN_<time>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
ROUND=${1:-1}
EPOCHS=${EPOCHS:-80}; IMGSZ=${IMGSZ:-1280}; MODEL=${MODEL:-yolo11s-obb.pt}; BATCH=${BATCH:--1}
DEVICE=${DEVICE:-0}; KEEP=${KEEP:-}; K=${K:-10}; VENV=${VENV:-$HOME/trse_venv}; TEST_MODE=${TEST_MODE:-0}
RUN=bootstrap_v$ROUND
BEST=$ROOT/runs/$RUN/weights/best.pt
mkdir -p logs
LOG=logs/${RUN}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
stop() { echo; echo "BERHENTI: $*"; echo "log: $LOG"; exit 1; }
T0=$(date +%s)

step "0. cek kelengkapan salinan"
if [ "$ROUND" = 1 ] && [ -f sync_check.sha256 ]; then
  sha256sum -c sync_check.sha256 --quiet || stop "ada file yang hilang/berbeda (daftar di atas) - salin ulang dari laptop"
  echo "salinan lengkap ($(wc -l < sync_check.sha256) file)"
else
  echo "dilewati (hanya putaran 1)"
fi

step "1. python venv + CUDA"
if [ "$TEST_MODE" = 1 ]; then
  echo "TEST_MODE: dilewati"
else
  if grep -qi microsoft /proc/version 2>/dev/null; then
    stop "ini WSL di laptop, bukan server GPU. Jalankan perintah ini di terminal Ubuntu server (lewat Remmina)."
  fi
  if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    if [ -f "$VENV/pyvenv.cfg" ]; then rm -r -f -- "$VENV"; fi         # broken venv from an earlier attempt
    if ! python3 -m venv "$VENV" >/dev/null 2>&1; then
      # python3-venv/ensurepip missing: venv without pip + get-pip.py (no sudo needed)
      echo "python3-venv tidak ada -> venv tanpa pip + get-pip.py (tanpa sudo)"
      if [ -f "$VENV/pyvenv.cfg" ]; then rm -r -f -- "$VENV"; fi
      python3 -m venv --without-pip "$VENV" || stop "python3 -m venv gagal - pasang: sudo apt install python3-venv"
      GETPIP=$(mktemp /tmp/get-pip.XXXXXX.py)
      curl -sSf https://bootstrap.pypa.io/get-pip.py -o "$GETPIP" || wget -q -O "$GETPIP" https://bootstrap.pypa.io/get-pip.py \
        || stop "tidak bisa mengunduh get-pip.py (cek internet)"
      "$VENV/bin/python" "$GETPIP" -q || stop "pemasangan pip gagal"
      rm -f -- "$GETPIP"
    fi
  fi
  echo "venv: $VENV ($("$VENV/bin/python" --version))"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -c "import ultralytics, PIL, numpy, yaml" 2>/dev/null || pip install -q ultralytics pillow numpy pyyaml
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
  python - <<'EOF' || stop "torch tidak melihat GPU. Pasang torch versi CUDA dari https://pytorch.org/get-started/locally/"
import sys, torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA:", ok, "|", torch.cuda.get_device_name(0) if ok else "-")
sys.exit(0 if ok else 1)
EOF
fi

step "2. label dari label_specs (make_labels --all)"
python tools/make_labels.py --all > "logs/make_labels_$RUN.log" 2>&1 || { tail -20 "logs/make_labels_$RUN.log"; stop "make_labels gagal"; }
echo "$(grep -c 'box=' "logs/make_labels_$RUN.log") foto dibuat ulang (detail: logs/make_labels_$RUN.log)"

step "3. cek spec (gate: berhenti jika ada masalah baru yang belum diterima, lihat check_specs_accepted.json)"
python tools/check_specs.py $(ls label_specs | sed 's/\.json$//') > "logs/check_specs_$RUN.log" 2>&1 \
  || { cat "logs/check_specs_$RUN.log"; stop "check_specs menemukan masalah baru (atau crash) - review lalu tools/check_specs.py --accept NAME \"teks masalah\" \"alasan\", atau perbaiki spec-nya"; }
grep -E '^IMG' "logs/check_specs_$RUN.log" | sed 's/^/  masalah: /' || true
tail -1 "logs/check_specs_$RUN.log"

step "4. dataset seed/pool"
python tools/export_datasets.py | tee "logs/export_$RUN.log"
SEED=$(grep -oE 'seed: [0-9]+' "logs/export_$RUN.log" | grep -oE '[0-9]+')
[ "${SEED:-0}" -ge 15 ] || stop "seed hanya ${SEED:-0} foto (< 15) - terlalu sedikit untuk bootstrap"

step "5. training bootstrap di GPU ($MODEL, imgsz $IMGSZ, $EPOCHS epoch)"
# identity of this experiment: dataset bytes (images+labels actually used) + the config that affects the result
FP_FILE="$ROOT/runs/$RUN/weights/fingerprint.txt"
FP="$(find "$ROOT/export/yolo_obb/images" "$ROOT/export/yolo_obb/labels" -type f | sort | xargs sha1sum | sha1sum | awk '{print $1}')|$MODEL|$IMGSZ|$EPOCHS|$BATCH|$DEVICE"
if [ "$TEST_MODE" = 1 ]; then
  echo "TEST_MODE: dilewati"
elif [ -f "$BEST" ] && [ -f "$FP_FILE" ] && [ "$(cat "$FP_FILE")" = "$FP" ] && [ "${RETRAIN:-0}" != 1 ]; then
  echo "sudah ada $BEST dengan dataset+config yang sama - dilewati (RETRAIN=1 untuk melatih ulang)"
else
  T1=$(date +%s)
  yolo obb train model="$MODEL" data="$ROOT/export/yolo_obb/data.yaml" imgsz="$IMGSZ" epochs="$EPOCHS" \
       batch="$BATCH" device="$DEVICE" degrees=10 fliplr=0 project="$ROOT/runs" name="$RUN" exist_ok=True
  echo "waktu training: $(( ($(date +%s) - T1) / 60 )) menit"
  [ -f "$BEST" ] || stop "training selesai tanpa $BEST"
  echo "$FP" > "$FP_FILE"
fi

step "6. metrik val per kelas"
if [ "$TEST_MODE" = 1 ]; then
  echo "TEST_MODE: dilewati"
else
  BEST="$BEST" IMGSZ="$IMGSZ" DEVICE="$DEVICE" OUT="logs/metrics_$RUN.json" python - <<'EOF'
import json, os
from ultralytics import YOLO
r = YOLO(os.environ["BEST"]).val(data="export/yolo_obb/data.yaml", imgsz=int(os.environ["IMGSZ"]),
                                  device=os.environ["DEVICE"], split="val", plots=False, verbose=False)
out = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4), "per_kelas": {}}
for j, c in enumerate(r.ap_class_index):
    p, rc, ap50, ap = r.class_result(j)
    out["per_kelas"][r.names[int(c)]] = {"P": round(float(p), 3), "R": round(float(rc), 3),
                                         "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
json.dump(out, open(os.environ["OUT"], "w"), indent=1)
print(json.dumps(out, indent=1))
EOF
fi

CALIB=logs/calibration_$RUN.json
step "6b. kalibrasi batas draft & penanda pada foto val yang labelnya diketahui"
if [ "$TEST_MODE" = 1 ]; then
  echo "TEST_MODE: dilewati"
else
  python tools/calibrate.py --model "$BEST" --imgsz "$IMGSZ" --device "$DEVICE" --out "$CALIB" \
    || stop "kalibrasi gagal"
fi

step "7. draft label + prioritas review (al_select)"
AL_ARGS=(--k "$K" --batch "al_round$ROUND")
if [ -n "$KEEP" ]; then AL_ARGS+=(--keep "$KEEP"); fi
if [ "$TEST_MODE" = 1 ]; then
  python tools/al_select.py --pred-dir "${PRED_DIR:?PRED_DIR wajib di TEST_MODE}" "${AL_ARGS[@]}"
else
  python tools/al_select.py --model "$BEST" --imgsz "$IMGSZ" --device "$DEVICE" --calib "$CALIB" "${AL_ARGS[@]}"
fi

step "8. ringkasan"
RUN="$RUN" ROUND="$ROUND" python - <<'EOF'
import csv, json, os
rows = list(csv.DictReader(open("review_priority.csv", encoding="utf-8")))
per = sum(1 for r in rows if "PERIKSA" in r["catatan"])
ok = sum(1 for r in rows if r["catatan"].startswith("OK"))
print(f"foto pool ter-cover : {len(rows)} (labels_auto/, preview_auto/)")
print(f"PERIKSA             : {per}")
print(f"OK, cek cepat       : {ok}")
print(f"cek normal          : {len(rows) - per - ok}")
m = f"logs/metrics_{os.environ['RUN']}.json"
if os.path.exists(m):
    d = json.load(open(m))
    print(f"val mAP50 / mAP50-95: {d['mAP50']} / {d['mAP50_95']}")
    for c, v in d["per_kelas"].items():
        print(f"   {c:<10s} mAP50 {v['mAP50']}  mAP50-95 {v['mAP50_95']}  P {v['P']}  R {v['R']}")
c = f"logs/calibration_{os.environ['RUN']}.json"
if os.path.exists(c):
    k = json.load(open(c))
    print(f"kalibrasi (val)     : draft conf>={k['keep']} (P {k['at_keep']['P']} R {k['at_keep']['R']}), "
          f"yakin conf>={k['flag_conf']} (F1 {k['at_flag_conf']['F1']}), NMS IoU {k['nms_iou']}")
b = f"batches/al_round{os.environ['ROUND']}"
print(f"kerjakan dulu       : {b}/ ({len(os.listdir(os.path.join(b, 'images')))} foto), lalu dari atas review_priority.csv")
print("catatan             : draft untuk foto lebar (14-27) dan sesi 17:31 (72-104) diperkirakan paling lemah;")
print("                      seed didominasi close-up. Draft BUKAN label final - wajib dikoreksi.")
EOF
echo "total waktu: $(( ($(date +%s) - T0) / 60 )) menit | log: $LOG"
