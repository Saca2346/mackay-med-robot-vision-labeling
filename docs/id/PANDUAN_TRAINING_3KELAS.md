# Training model 3 kelas (box, size_label, expiry_label) di server

Kelas baru `expiry_label` = tanggal kedaluwarsa yang tercetak di punggung kotak (di samping ikon jam pasir).
Label dibuat otomatis dengan OCR (`tools/auto_expiry.py`), direview, lalu dipaket di laptop.

Setiap blok di bawah ditandai tempat menjalankannya. Salin persis.

## 0. Label tanggal di server (CPU server jauh lebih cepat dari laptop)

**Laptop, cmd** — kirim paket pelabelan (tools + 29 foto yang sudah selesai):

```bat
cd C:\Users\sacab\Downloads
scp paket_label_tanggal_20261009.zip Zen@140.113.149.94:~/Downloads/
```

**Server, terminal (Remote Desktop)** — pasang dan jalankan (foto yang sudah selesai dilewati otomatis):

```bash
cd ~/Downloads/prelabel_size_label_full
unzip -o ~/Downloads/paket_label_tanggal_20261009.zip
sha256sum -c --quiet PAKET_SHA256.txt && echo PAKET OK
sha256sum -c --quiet RAW_SHA256.txt && echo FOTO OK
source ~/trse_venv/bin/activate
pip install -q rapidocr onnxruntime opencv-python-headless
mkdir -p logs
nohup python tools/auto_expiry.py --all --sheet > logs/auto_expiry.log 2>&1 &
sleep 60; grep -c "expiry_label dari" logs/auto_expiry.log
```

Angka terakhir = jumlah foto yang sudah diproses. Cek lagi kapan saja dengan perintah `grep` yang sama; selesai kalau
baris terakhir `logs/auto_expiry.log` berisi `163 foto` (atau `... foto, ... expiry_label diusulkan`).

**Server, terminal** — setelah selesai, bungkus hasilnya:

```bash
cd ~/Downloads/prelabel_size_label_full
zip -qr ~/Downloads/hasil_label_tanggal.zip label_specs_expiry preview_expiry logs/auto_expiry.log && echo ZIP OK
```

**Laptop, cmd** — ambil hasilnya, lalu beri tahu Claude (review + paket training):

```bat
cd C:\Users\sacab\Downloads
scp Zen@140.113.149.94:~/Downloads/hasil_label_tanggal.zip .
```

## Status 11-10 (sebelum langkah 1)

- Label tanggal dari server lengkap: 163 foto, 1.215 label. Sampel 12 foto diperiksa mata: kelompok >= 80 % liputan benar (6 dari 6),
  kelompok 50-79 % ada tanggal jelas yang terlewat (tidak aman).
- **76 foto** (liputan >= 80 %) ditandai selesai dengan `tools/ok_expiry_batch.py`; catatannya menyebut ini aturan liputan + sampel,
  bukan review foto per foto. **87 foto tetap di kolam** (tidak ikut training) sampai dilengkapi.
- Akibatnya training 3 kelas memakai 76 foto, bukan 163. Bandingkan hasil `box` dan `size_label` dengan v4 (163 foto) sebelum
  memakainya: model 3 kelas baru layak dipakai kalau dua kelas lama tidak lebih buruk.
- Paket dibuat dengan `python tools/pack_for_server.py --force` (kolam 87 foto disengaja; gerbang export di server memakai aturan yang sama).

## 1. Laptop, cmd — kirim paket ke server

```bat
cd C:\Users\sacab\Downloads
scp paket_3kelas_20261011.zip Zen@140.113.149.94:~/Downloads/
```

## 2. Server, terminal (Remote Desktop) — cadangkan v6, pasang paket, cek foto

`train_best.py` menghapus `runs/cv` dan menimpa `runs/best_final` kalau dataset berubah, jadi hasil v6 dicadangkan dulu.

```bash
cd ~/Downloads/prelabel_size_label_full
mkdir -p ~/Downloads/cadangan_v6 && cp -r runs/best_final logs/best_model.json ~/Downloads/cadangan_v6/
unzip -o ~/Downloads/paket_3kelas_20261011.zip
sha256sum -c --quiet PAKET_SHA256.txt && echo PAKET OK
sha256sum -c --quiet RAW_SHA256.txt && echo FOTO OK
cat classes.txt
```

Harus tercetak `PAKET OK`, `FOTO OK`, lalu tiga baris `box`, `size_label`, `expiry_label`.
Kalau `FOTO OK` tidak muncul, berhenti dan kirim layar ke Claude (foto di server berbeda dengan laptop).

## 3. Server, terminal — cek label lalu mulai training

```bash
cd ~/Downloads/prelabel_size_label_full
source ~/trse_venv/bin/activate
python tools/check_specs.py | tail -1
mkdir -p logs
nohup python tools/train_best.py --quick --models yolo11s-obb.pt@180 > logs/train_3kelas.log 2>&1 &
sleep 60; tail -5 logs/train_3kelas.log
```

`check_specs` harus `0 with new/unaccepted problems, ... 0 crashed`. Baris `tail` harus menunjukkan
`... foto berlabel, ... burst -> 3 fold`. Training berjalan di latar; jendela boleh ditutup.

`--quick --models yolo11s-obb.pt@180` = satu konfigurasi (yang sudah menang di v4–v6), 3 fold x 120 epoch +
model final 200 epoch. Perkiraan sekitar 1–1,5 jam (v6 penuh = 2 konfigurasi x 5 fold = 4 jam).

## 4. Server, terminal — cek progres kapan saja

```bash
cd ~/Downloads/prelabel_size_label_full
tail -3 logs/train_3kelas.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

## 5. Server, terminal — setelah selesai

Selesai kalau log berisi `model final` dan file bobot ada:

```bash
cd ~/Downloads/prelabel_size_label_full
grep -A6 "Hasil cross-validation" logs/train_3kelas.log
ls -la runs/best_final/
cat logs/best_model.json | head -40
```

Kirim layar hasil ini ke Claude. Yang dicek: mAP50-95 per kelas (terutama `expiry_label`), dibandingkan v4/v6
untuk `box` dan `size_label` — model baru tidak boleh lebih buruk untuk dua kelas lama.

## 6. Laptop, cmd — ambil model

```bat
cd C:\Users\sacab\Downloads\medicine_shelf_vision\medicine_shelf_vision
scp Zen@140.113.149.94:~/Downloads/prelabel_size_label_full/runs/best_final/box_best.pt models\box_v7s_3kelas.pt
```

Model ini belum dipakai demo. Demo Senin tetap v4 sampai model baru terbukti tidak lebih buruk.

cd ~/Downloads/prelabel_size_label_full
ls label_specs | wc -l
ls label_specs_expiry | sed 's/.json$//' | sort > /tmp/a.txt
ls label_specs | sed 's/.json$//' | sort > /tmp/b.txt
echo "--- ekstra di label_specs_expiry (tidak ada di label_specs):"; comm -23 /tmp/a.txt /tmp/b.txt
echo "--- belum ada hasilnya:"; comm -13 /tmp/a.txt /tmp/b.txt
echo "--- proses:"; pgrep -af auto_expiry || echo "tidak berjalan"
