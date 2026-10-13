# Panduan koreksi label di Ubuntu (hucenrotia-ai): CVAT, X-AnyLabeling, Label Studio

Semua pekerjaan kini dilakukan di **Ubuntu (hucenrotia-ai)**, dan folder proyek di sana menjadi **master**.

Dalam panduan ini:
- **PROYEK** = `/home/Zen/Downloads/prelabel_size_label_full`, atau `~/Downloads/prelabel_size_label_full`.
- Semua perintah `python ...` dijalankan dari folder PROYEK dengan venv aktif:

```
cd ~/Downloads/prelabel_size_label_full
source ~/trse_venv/bin/activate
```

## Tentang Remmina

**Pengaturan Remmina.** Untuk melabel lewat Remmina, pilih **Quality: Best** dan **Color depth: 32 bpp**. Dengan kualitas lebih rendah, gambar dikompres dan tepi kemasan jadi kabur.

**CVAT dan Label Studio berbasis web.** Anda bisa:
- membuka Firefox/Chrome **di desktop Ubuntu** lewat Remmina, atau
- memakai **browser laptop** lewat SSH tunnel (lebih tajam dan responsif), kalau laptop bisa SSH ke Ubuntu:
  ```
  ssh -L 8080:localhost:8080 -L 8081:localhost:8081 Zen@hucenrotia-ai
  ```

Aturan penting: **unggah foto dan unduh hasil ekspor selalu dari browser di Ubuntu.** Dengan begitu file langsung berada di Ubuntu (master). Pelabelan sendiri boleh dari browser mana saja, karena datanya tersimpan di server.

## Gambaran alur

Siklus lengkapnya:

1. Siapkan folder **batch**:
   - `batches/cek_manual`: 7 foto seed yang perlu dicek, dibuat dengan `python tools/batch.py cek_manual --review`.
   - `batches/al_round1`: dibuat oleh `al_select.py`.
2. Buka batch di **satu** tool dan koreksi labelnya.
3. Ekspor hasil koreksi dari tool.
4. `python tools/import_reviewed.py <hasil ekspor>` memindahkan hasil ke `labels_reviewed_obb/`.
5. `python tools/export_datasets.py` membuat dataset seed baru.
6. Train ulang di GPU, lalu `python tools/al_select.py ...` memilih batch berikutnya. Kembali ke langkah 2.

Setiap folder batch berisi:

| File / folder | Dipakai oleh |
|---|---|
| `images/` | Foto-foto batch (semua tool) |
| `labels/` + `classes.txt` | X-AnyLabeling |
| `cvat_annotations.xml` | CVAT |
| `label_studio_tasks.json` | Label Studio |
| `list.txt` | Daftar nama foto |

Batch **tidak ikut disalin** dari laptop. Buat ulang di Ubuntu:

```
python tools/batch.py cek_manual --review
```

Batch ini berisi foto 3, 12, 13, 29, 30, 31, 48. Yang paling perlu diperiksa adalah kelompok kiri yang buram di foto 29–31.

**Aturan penting:**
- Satu batch dikerjakan di **satu tool saja**.
- Nama kelas harus persis `box` dan `size_label`. Urutannya di tool boleh berbeda, karena `import_reviewed.py` mencocokkan kelas berdasarkan **nama**.

---

## A. CVAT (Docker)

### A1. Instalasi (sekali saja)

1. Pasang Docker Engine + plugin compose (kalau belum ada), lalu **logout dan login lagi**:
   ```
   sudo apt install docker.io docker-compose-v2
   sudo usermod -aG docker $USER
   ```
2. Jalankan CVAT:
   ```
   git clone https://github.com/cvat-ai/cvat ~/cvat
   cd ~/cvat
   git checkout v2.76.0
   docker compose up -d
   ```
   Download pertama beberapa GB. Tunggu sampai `docker compose ps` menunjukkan semua container `running` atau `healthy`.
3. Buat akun admin:
   ```
   docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
   ```
4. Buka `http://localhost:8080`.

**Mematikan dan menyalakan** (dari `~/cvat`): `docker compose stop` / `docker compose start`. **Jangan** jalankan `docker compose down -v`, karena `-v` menghapus semua data CVAT.

### A2. Buat project (sekali saja)

1. Klik **Projects**, lalu **+**, lalu **Create a new project**.
2. Name: `TRSE box size_label`.
3. Tambahkan dua label dengan **Add label**, tipe **Rectangle**:
   - `box`
   - `size_label`
4. Klik **Submit & Open**.

### A3. Buat task untuk satu batch (dari browser di Ubuntu)

1. Di dalam project, klik **+**, lalu **Create a new task**. Name = nama batch.
2. **Select files → My computer**, lalu pilih **semua** foto dari `PROYEK/batches/<batch>/images`.
3. Buka **Advanced configuration** dan set **Image quality** ke `95` atau lebih.
4. Klik **Submit & Open**.

### A4. Masukkan label awal

1. Di halaman task, klik **Actions**, lalu **Upload annotations**.
2. Import format: **CVAT 1.1**.
3. Pilih file `PROYEK/batches/<batch>/cvat_annotations.xml`, lalu klik **OK**.

### A5. Mengoreksi

**Kontrol dasar:**
- Pindah foto: **F** (berikutnya) dan **D** (sebelumnya).
- Simpan: **Ctrl+S** (sering-sering).
- Undo: **Ctrl+Z**.

**Mengubah kotak:**
- Klik kotaknya, lalu tarik sudut atau sisinya.
- Untuk **memutar**, tarik **pegangan rotasi** (lingkaran kecil di atas kotak).

**Menambah kotak:**
1. Pilih alat **Draw new rectangle**.
2. Pilih label, lalu gambar kotaknya.
3. Putar dengan pegangan rotasi sampai sejajar punggung kemasan.

Tekan **N** untuk mengulang alat gambar terakhir.

**Menghapus kotak:** pilih kotak, lalu tekan **Delete**.

### A6. Ekspor (dari browser di Ubuntu)

1. **Actions → Export task dataset**.
2. Format: **Ultralytics YOLO Oriented Bounding Boxes 1.0**.
3. Matikan **Save images**, lalu klik **OK**.
4. Unduh zip-nya dari menu **Requests**. File tersimpan di `~/Downloads/`.
5. Kembalikan hasilnya ke PROYEK:
   ```
   python tools/import_reviewed.py ~/Downloads/<nama-file>.zip --dry
   python tools/import_reviewed.py ~/Downloads/<nama-file>.zip
   python tools/export_datasets.py
   ```

Jangan pakai format "YOLO 1.1" atau "Ultralytics YOLO Detection", karena keduanya membuang rotasi. `import_reviewed.py` akan menolak hasil ekspor kotak tegak.

---

## B. X-AnyLabeling (aplikasi desktop, lewat Remmina)

### B1. Instalasi (sekali saja)

X-AnyLabeling butuh **Python 3.11–3.13**. Cek dulu dengan `python3 --version`:
- Ubuntu 24.04 memakai Python 3.12, jadi bisa langsung.
- Ubuntu 22.04 memakai Python 3.10. Pakai versi Linux dari **Releases** (`https://github.com/CVHub520/X-AnyLabeling/releases`), atau pasang Python 3.12 lewat `uv` / conda.

Pakai venv terpisah dari `trse_venv`. Jalankan di terminal **di dalam desktop Remmina**, bukan lewat SSH, karena aplikasinya butuh layar:

```
python3 -m venv ~/xal-env
source ~/xal-env/bin/activate
pip install -U pip
pip install "x-anylabeling-cvhub[cpu]"
xanylabeling
```

Versi `[cpu]` cukup untuk koreksi, karena draf sudah dibuat model kita. `[gpu]` hanya perlu untuk B5 (auto-label di dalam X-AnyLabeling).

### B2. Buka batch dan label awal

1. **File → Open Dir** (**Ctrl+U**), lalu pilih `PROYEK/batches/<batch>/images`. Judul jendela harus menunjukkan **[1/N]** dengan N = jumlah foto batch.
2. Pastikan **File → Auto Save** aktif.
3. **Upload → Upload YOLO-Obb Annotations** (jangan YOLO-Hbb, itu kotak tegak):
   - File kelas: `PROYEK/batches/<batch>/labels/classes.txt`.
   - Folder label: `PROYEK/batches/<batch>/labels`.
4. Satu `.json` per foto muncul di `images/`; itu file kerja X-AnyLabeling dan normal.

### B3. Mengoreksi

- Buat kotak miring baru: tekan **O**.
- Memutar: pilih kotak (mode edit), lalu tekan **Z / X / C / V**.
- Pindah foto: **D** (berikutnya) dan **A** (sebelumnya).
- Hapus: pilih kotak, lalu tekan **Delete**.
- Tekan **Ctrl+S** di setiap foto. Jam `.json` foto itu harus berubah (`ls -l images/*.json`).
- **Jangan** pakai **Tools → Save Visualization Image** selama batch dibuka: X-AnyLabeling ikut membaca subfolder, sehingga folder `images_visualizations` membuat gambar visualisasi terhitung sebagai foto (misalnya [20/20]). Kalau sudah terlanjur, pindahkan folder itu keluar dari `images/`.

### B4. Ekspor

1. **Export → Export YOLO-Obb Annotations**, file kelas yang sama.
2. Hasil ekspor masuk ke `PROYEK/batches/<batch>/labels` (menimpa draf; drafnya tetap ada di `.json` dan `labels_auto/`).
3. Kembalikan hasilnya ke PROYEK:
   ```
   python tools/import_reviewed.py batches/<batch>/labels --dry
   python tools/import_reviewed.py batches/<batch>/labels
   python tools/export_datasets.py
   ```
   `import_reviewed` menolak foto yang **sama persis dengan draf model** (belum dikoreksi), dan per foto menampilkan berapa objek sama / diubah / dihapus dibanding draf.

### B5. (Opsional) Auto-label langsung dengan model bootstrap di GPU

1. Ubah model ke ONNX:
   ```
   yolo export model=runs/bootstrap_v1/weights/best.pt format=onnx imgsz=1280
   ```
2. Buat file `trse_obb.yaml`:
   ```yaml
   type: yolo11_obb          # yolov8_obb bila modelnya YOLOv8-OBB
   name: trse_obb
   display_name: TRSE box/size_label
   model_path: /home/Zen/Downloads/prelabel_size_label_full/runs/bootstrap_v1/weights/best.onnx
   conf_threshold: 0.35
   iou_threshold: 0.7
   classes:
     - box
     - size_label
   ```
3. Di X-AnyLabeling, buka dropdown model, pilih **...Load Custom Model**, lalu pilih `trse_obb.yaml`.
4. Tekan **Ctrl+A** (auto-label semua foto), lalu koreksi seperti biasa.

---

## C. Label Studio

### C1. Instalasi (sekali saja)

Pakai venv terpisah:

```
python3 -m venv ~/ls-env
source ~/ls-env/bin/activate
pip install label-studio
```

### C2. Menjalankan Label Studio

Jalankan setiap kali:

```
source ~/ls-env/bin/activate
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$HOME/Downloads/prelabel_size_label_full
label-studio start --port 8081
```

Port 8081 dipakai supaya tidak bentrok dengan CVAT di 8080. Buka `http://localhost:8081`, lalu buat akun lokal.

### C3. Buat project (sekali saja)

1. **Create Project**, beri nama `TRSE box size_label`.
2. **Labeling Setup → Custom template → Code**, lalu tempel isi `PROYEK/export/label_studio/label_config.xml`. Klik **Save**.
3. Hubungkan folder foto:
   1. **Settings → Cloud Storage → Add Source Storage**.
   2. Storage type: **Local files**.
   3. Absolute local path: `/home/Zen/Downloads/prelabel_size_label_full/raw`.
   4. Klik **Add Storage**.
   5. **Jangan klik "Sync Storage"**, karena itu akan membuat 104 task kosong.

### C4. Masukkan batch

1. Di project, klik **Import → Upload Files**, lalu pilih `PROYEK/batches/<batch>/label_studio_tasks.json`.
2. Hasilnya berbeda tergantung jenis batch:
   - **`cek_manual`**: label masuk sebagai *annotation*. Perbaiki, lalu klik **Update**.
   - **`al_round`**: label model masuk sebagai *prediction*. Aktifkan **Settings → Annotation → Use predictions to prelabel tasks** (di versi lama: **Settings → Machine Learning**). Setelah itu koreksi, lalu klik **Submit**.

### C5. Mengoreksi

- Hotkey **1** = `box`, **2** = `size_label`.
- Tarik sudut untuk mengubah ukuran, pegangan rotasi untuk memutar.
- **Backspace** atau **Delete** untuk menghapus.

### C6. Ekspor (dari browser di Ubuntu)

1. **Export**, pilih format **JSON** (bukan JSON-MIN, bukan YOLO), lalu klik **Export**.
2. Kembalikan hasilnya ke PROYEK:
   ```
   python tools/import_reviewed.py ~/Downloads/project-1-at-....json --dry
   python tools/import_reviewed.py ~/Downloads/project-1-at-....json
   python tools/export_datasets.py
   ```

Ekspor "YOLOv8 OBB" di Label Studio punya laporan masalah. JSON menyimpan rotasi secara lengkap dan dikonversi dengan benar oleh `import_reviewed.py` (selisih kurang dari 0,5 px).

---

## D. Setelah koreksi

1. **Selalu jalankan `--dry` dulu.** Hasilnya salah satu dari tiga:
   - **KESALAHAN:** tidak ada yang ditulis. Perbaiki di tool, lalu ekspor ulang.
   - **PERINGATAN**, misalnya `size_label` di luar box atau foto tanpa box: cek lagi foto itu.
   - **Valid:** jalankan tanpa `--dry`. File lama otomatis dicadangkan di `labels_reviewed_obb/_backup/`.
2. `python tools/export_datasets.py`. Foto yang sudah direview pindah dari pool ke seed.
3. Putaran berikutnya: `bash tools/run_bootstrap.sh 2`. Perintah ini melatih `runs/bootstrap_v2` dengan seed yang sudah bertambah dan membuat `batches/al_round2`.

---

## Aturan label (sama di semua tool)

**`box`**
- Hanya punggung kemasan, berupa kotak miring yang sejajar punggung.
- Muka ujung, muka samping, celah, tangan dan jari tidak termasuk.
- Kotak tidak boleh keluar dari kemasan yang ditarget.
- Kemasan yang tertutup sebagian: label hanya bagian yang terlihat.
- Sliver tipis (kurang dari ±1% lebar foto) tidak dilabel.

**`size_label`**
- Klaster nama produk + ukuran (diameter/panjang), selalu di dalam `box`-nya.
- Terumo (Nagomi, Accuforce): 2 `size_label`, yaitu nama+ukuran dan blok logo TERUMO.
- iVascular: 1 `size_label`, yaitu panel berwarna.
- Sled Bag: logo Boston Scientific + teks "Permanent Sled Bag".
- REF, LOT, tanggal atau barcode saja **tidak** diberi `size_label`.

**Jangan pernah:**
- Membuat kelas baru.
- Menulis teks atau nama produk sebagai nama kelas.
