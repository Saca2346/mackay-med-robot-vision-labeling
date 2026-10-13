# Training the 3-class model (box, size_label, expiry_label) on the server

English version of `PANDUAN_TRAINING_3KELAS.md` (the Indonesian file stays in the project root because
`tools/pack_for_server.py` packs it). Status: the training of 11-10 is done; results are in the code repo,
`docs/FIXES_LOG_20261008-11.md` section 7. Before the next training read `server_results/COMPARE_SERVER_LAPTOP.md`: the server
and laptop label copies differ (14-10).

The new class `expiry_label` = the expiry date printed on the box spine (next to the hourglass icon). Labels are made
automatically with OCR (`tools/auto_expiry.py`), reviewed, then packed on the laptop.

Every block says where it runs. Copy it exactly. With an SSH key (code repo, `docs/REMOTE_SERVER_CLAUDE_CODEX.md`) the server
steps can be run from the laptop: `RUN_ON_SERVER.bat NAME` runs `server_scripts/NAME.sh` and saves the output in `server_results/`.

## 0. Date labels on the server (the server CPU is much faster than the laptop)

**Laptop, cmd** - send the labelling package (tools + the 29 photos already done):

```bat
cd C:\Users\sacab\Downloads
scp paket_label_tanggal_20261009.zip Zen@140.113.149.94:~/Downloads/
```

**Server, terminal** - install and run (photos already done are skipped automatically):

```bash
cd ~/Downloads/prelabel_size_label_full
unzip -o ~/Downloads/paket_label_tanggal_20261009.zip
sha256sum -c --quiet PAKET_SHA256.txt && echo PACKAGE OK
sha256sum -c --quiet RAW_SHA256.txt && echo PHOTOS OK
source ~/trse_venv/bin/activate
pip install -q rapidocr onnxruntime opencv-python-headless
mkdir -p logs
nohup python tools/auto_expiry.py --all --sheet > logs/auto_expiry.log 2>&1 &
sleep 60; grep -c "expiry_label dari" logs/auto_expiry.log
```

The last number = photos processed so far. Check again any time with the same `grep`; it is finished when the last line of
`logs/auto_expiry.log` says `163 foto` (or `... foto, ... expiry_label diusulkan`).

**Server, terminal** - when finished, pack the result:

```bash
cd ~/Downloads/prelabel_size_label_full
zip -qr ~/Downloads/hasil_label_tanggal.zip label_specs_expiry preview_expiry logs/auto_expiry.log && echo ZIP OK
```

**Laptop, cmd** - fetch the result, then tell Claude (review + training package):

```bat
cd C:\Users\sacab\Downloads
scp Zen@140.113.149.94:~/Downloads/hasil_label_tanggal.zip .
```

## Status 11-10 (before step 1)

- Date labels from the server complete: 163 photos, 1,215 labels. A sample of 12 photos was checked by eye: the >= 80 % coverage
  group was right (6 of 6), the 50-79 % group missed clearly readable dates (not safe).
- **76 photos** (coverage >= 80 %) were marked done with `tools/ok_expiry_batch.py`; its note says this is a coverage rule +
  a sample, not a photo-by-photo review. **87 photos stay in the pool** (not in training) until completed.
- So the 3-class training used 76 photos, not 163. Compare `box` and `size_label` with v4 (163 photos) before using it: the
  3-class model is only usable if the two old classes are not worse.
- The package was built with `python tools/pack_for_server.py --force` (the 87-photo pool is intentional; the export gate on the
  server uses the same rule).

## 1. Laptop, cmd - send the package to the server

```bat
cd C:\Users\sacab\Downloads
scp paket_3kelas_20261011.zip Zen@140.113.149.94:~/Downloads/
```

## 2. Server, terminal - back up v6, install the package, check the photos

`train_best.py` deletes `runs/cv` and overwrites `runs/best_final` when the dataset changes, so back up the v6 results first.

```bash
cd ~/Downloads/prelabel_size_label_full
mkdir -p ~/Downloads/cadangan_v6 && cp -r runs/best_final logs/best_model.json ~/Downloads/cadangan_v6/
unzip -o ~/Downloads/paket_3kelas_20261011.zip
sha256sum -c --quiet PAKET_SHA256.txt && echo PACKAGE OK
sha256sum -c --quiet RAW_SHA256.txt && echo PHOTOS OK
cat classes.txt
```

It must print `PACKAGE OK`, `PHOTOS OK`, then three lines `box`, `size_label`, `expiry_label`.
If `PHOTOS OK` does not appear, stop and send the screen to Claude (the photos on the server differ from the laptop).

## 3. Server, terminal - check the labels, then start training

```bash
cd ~/Downloads/prelabel_size_label_full
source ~/trse_venv/bin/activate
python tools/check_specs.py | tail -1
mkdir -p logs
nohup python tools/train_best.py --quick --models yolo11s-obb.pt@180 > logs/train_3kelas.log 2>&1 &
sleep 60; tail -5 logs/train_3kelas.log
```

`check_specs` must say `0 with new/unaccepted problems, ... 0 crashed`. The `tail` lines must show
`... foto berlabel, ... burst -> 3 fold`. Training runs in the background; the window may be closed (better: run it inside
`tmux`, see the remote guide).

`--quick --models yolo11s-obb.pt@180` = one configuration (the winner of v4-v6), 3 folds x 120 epochs + a final model of
200 epochs. About 1-1.5 hours (the full v6 = 2 configurations x 5 folds = 4 hours).

## 4. Server, terminal - check progress at any time

```bash
cd ~/Downloads/prelabel_size_label_full
tail -3 logs/train_3kelas.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

## 5. Server, terminal - when finished

Finished when the log contains `model final` and the weight file exists:

```bash
cd ~/Downloads/prelabel_size_label_full
grep -A6 "Hasil cross-validation" logs/train_3kelas.log
ls -la runs/best_final/
cat logs/best_model.json | head -40
```

Send this screen to Claude. What is checked: mAP50-95 per class (above all `expiry_label`), compared with v4 / v6 for `box` and
`size_label`: the new model must not be worse for the two old classes.

## 6. Laptop, cmd - fetch the model

```bat
cd C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision
scp Zen@140.113.149.94:~/Downloads/prelabel_size_label_full/runs/best_final/box_best.pt models\box_v7s_3kelas.pt
```

This model is not used for the demo. The demo stays on v4 until a new model is proven not to be worse.

## Check of the date-label outputs on the server

```bash
cd ~/Downloads/prelabel_size_label_full
ls label_specs | wc -l
ls label_specs_expiry | sed 's/.json$//' | sort > /tmp/a.txt
ls label_specs | sed 's/.json$//' | sort > /tmp/b.txt
echo "--- extra in label_specs_expiry (not in label_specs):"; comm -23 /tmp/a.txt /tmp/b.txt
echo "--- no result yet:"; comm -13 /tmp/a.txt /tmp/b.txt
echo "--- process:"; pgrep -af auto_expiry || echo "not running"
```
