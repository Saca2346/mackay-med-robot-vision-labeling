# Lanjutan training 3 kelas: review label tanggal di SERVER, lalu satu perintah training

Menggantikan langkah 1–4 di PANDUAN_TRAINING_3KELAS.md. Tidak perlu lagi kirim zip bolak-balik ke laptop: review
dikerjakan langsung di server (terminal desktop Remote Desktop/Remmina). Model demo (v4) tidak terpengaruh.

## Yang sudah dikerjakan (10-10)
- Label tanggal otomatis: 163 foto, 1.215 usulan, semua status `auto` (belum direview).
- Lembar review untuk semua foto: `preview_expiry/NAMA_expiry.jpg`.
- Daftar prioritas: `review_tanggal_prioritas.csv` (foto paling bermasalah di atas).
- Geometri label (`check_specs`) lolos; hasil v6 dicadangkan di `~/Downloads/cadangan_v6`.
- Cadangan usulan tanggal sebelum review: `~/Downloads/cadangan_label_specs_expiry_20261010`.
- Skrip pengaman training: `tools/mulai_training_3kelas.sh` (berhenti dengan pesan jelas bila ada syarat belum terpenuhi).

## Diperketat 11/12-10 (tanggal tajam yang terlewat)
- Bug: ikon jam pasir terbaca "8" (`82029-09-21`) sehingga tanggal yang sudah terbaca dibuang. Diperbaiki.
- `tools/cek_tanggal_terlewat.py` (model 3 kelas + OCR ulang 1x/2x/3x + ketajaman): +64 label OCR, +165 posisi dari
  model, 22 tanggal diisi. Kotak buram (ketajaman < 85, dicek visual) sengaja tidak dilabeli.
- 94 kotak tajam tanpa tanggal/prediksi ditandai **PERIKSA** (bingkai merah). `--ok` DITOLAK sampai tiap kotak
  PERIKSA diberi label atau `kosong N` (memang tidak ada tanggal terlihat). Skrip training juga memeriksanya.
- 26 foto yang tadinya `selesai` dibuka lagi karena punya tanggal terlewat. Sekarang: 50 selesai, 113 perlu review.
- Cara review termudah: `python tools/review_tanggal.py` (lembar terkini: magenta = lama, cyan = usulan baru).
- Cadangan tiap langkah: `~/Downloads/cadangan_label_specs_expiry_*`.

## Aturan label iVascular (dikonfirmasi 12-10)
Urutan baris di punggung iVascular: LOT (atas) -> **KEDALUWARSA (tengah)** -> produksi (bawah). Label HANYA melingkupi baris
tengah. Label OCR sudah benar; label 'posisi dari model' (cyan, tanpa tanggal) lebih longgar. `tools/rapatkan_label_tanggal.py`
merapatkan yang bisa dibaca (16 label); sisanya (buram) dibiarkan. Kotak buram: ketik `buang N` (hapus label + catat kosong).

## Langkah A. Review label tanggal (Anda, di server)

Selalu dari folder proyek:
```bash
cd ~/Downloads/prelabel_size_label_full && source ~/trse_venv/bin/activate
```

Per foto (urut dari atas `review_tanggal_prioritas.csv`):
1. Buka lembarnya: `xdg-open preview_expiry/NAMA_expiry.jpg`
   Tiap strip = satu kotak, ujung jauh di KIRI, ujung dekat di KANAN. Kotak magenta = usulan label tanggal.
2. Lihat daftar label: `python tools/set_expiry.py NAMA --show`
3. Perbaiki:
   | Kasus | Perintah |
   | --- | --- |
   | Posisi benar, angka salah baca (mis. 2038 → 2028) | `python tools/set_expiry.py NAMA --date N 2028-01-31` |
   | Label menempel pada yang bukan tanggal kedaluwarsa | `python tools/set_expiry.py NAMA --drop N` |
   | Tanggal tercetak tetapi TIDAK ada kotak magenta | `python tools/set_expiry.py NAMA --add N T1 T2 --date YYYY-MM-DD` |
   `N` = nomor box (tertulis di bawah strip). `T1 T2` = posisi tanggal sepanjang strip, 0 = kiri, 1 = kanan;
   contoh tanggal di sekitar 45%–60% lebar strip: `--add 5 0.45 0.60 --date 2028-09-30`.
   Tanggal yang benar-benar tidak terbaca (buram/tertutup): biarkan tanpa label.
4. Setelah semua tanggal yang terlihat berlabel: `python tools/set_expiry.py NAMA --ok`
   (label kelas 2 untuk training ditulis otomatis oleh skrip training; tidak perlu `make_labels` manual.)

Penting: SETIAP tanggal yang terlihat harus berlabel sebelum `--ok`. Tanggal yang terlihat tetapi tidak berlabel
akan dipelajari model sebagai "bukan tanggal". Bila ragu, jangan `--ok` foto itu dulu.

Beban kerja (hasil hitung 10-10): 61 foto cukup dilirik lalu `--ok` (paling banyak 1 kotak tanpa label, tanpa
tanggal janggal); 102 foto perlu penambahan label (total ±600 kotak tanpa label, terutama frame webcam `_000`).
Mulai dari 4 tanggal janggal: 2038-01-31, 2037-06-30, 2036-09-30 (kemungkinan 2028/2027/2028 salah baca) dan
2026-09-30 (sudah lewat; cek di lembar).

Salah langkah? `python tools/set_expiry.py NAMA --redo` mengembalikan status `auto`; file asli ada di cadangan.

## Langkah B. Cek kesiapan (kapan saja, aman)
```bash
CEK=1 bash tools/mulai_training_3kelas.sh
```
Memeriksa: tidak ada training lain, 3 kelas, geometri label, jumlah foto yang sudah `--ok` (minimal 60),
3 fold tidak kosong (foto harus dari beberapa sesi, bukan satu sesi), VRAM GPU bebas. Tidak mengubah apa pun.

## Langkah C. Mulai training
```bash
bash tools/mulai_training_3kelas.sh
```
Mencadangkan hasil sebelumnya, lalu `train_best.py --quick` (3 fold x 120 epoch + final 200, batch 8) di latar.
Cek progres: `tail -3 logs/train_3kelas.log`. Selesai bila ada baris `model final` (±1–1,5 jam).
Lalu kirim ke Claude: `grep -A6 "Hasil cross-validation" logs/train_3kelas.log` dan `cat logs/best_model.json`.

## Keputusan yang perlu Anda ambil sebelum review besar
Syarat training 3 kelas: foto hanya ikut training bila label tanggalnya sudah direview. Bila hanya 61 foto mudah
yang di-`--ok`, model baru belajar `box` dan `size_label` dari 61 foto saja (v6: 156 foto) dan hampir pasti LEBIH
BURUK untuk dua kelas lama, sehingga tidak boleh menggantikan v4/v6. Agar adil, praktis semua 156 foto perlu
direview. Rangkuman meeting 05-10 menandai "OCR tanggal kedaluwarsa" sebagai klaim AI, bukan keputusan meeting:
pastikan fitur ini memang dibutuhkan sebelum mengerjakan ±600 label tambahan.

## Jangan
- `pip install opencv-python-headless` di `trse_venv`: menghapus jendela video (uji kamera Jalur A rusak).
  Sudah dipulihkan 10-10 ke `opencv-python==5.0.0.93`.
- Mengubah status `auto` menjadi `selesai` tanpa melihat lembarnya.
- Menjalankan `export_datasets.py` / `run_bootstrap.sh` sebelum review: seed 0 foto mengosongkan `export/yolo_obb`.
