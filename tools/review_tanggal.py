#!/usr/bin/env python3
"""Review label tanggal kedaluwarsa satu per satu, tanpa mengetik nama foto.

  python tools/review_tanggal.py

Lembar dibuat dari label TERKINI (tools/lembar_tanggal.py): magenta = label lama, cyan = usulan baru,
bingkai merah = kotak PERIKSA (tajam tapi belum berlabel). Setiap perubahan langsung digambar ulang.
Membuka lembar review foto berikutnya yang statusnya masih "auto" (urut review_tanggal_prioritas.csv), menampilkan
labelnya, lalu menunggu perintah:
  ok                  semua tanggal yang terlihat sudah benar & berlabel -> selesai, lanjut foto berikutnya
  tgl 14 2028-01-31   ganti tanggal label box 14 (posisi tetap)
  hapus 14            hapus label box 14 (bukan tanggal kedaluwarsa)
  tambah 5 0.45 0.60 2028-09-30   tambah label di box 5, dari 45% sampai 60% panjang strip (0 = kiri)
  tambah 5 0.45 0.60 0.3 0.6 2028-09-30   sama, tapi dibatasi lebarnya 30%-60% dari lebar kotak (label rapat, lebih baik)
  buang 7             hapus label di box 7 DAN catat kosong (tanggalnya buram / tidak terbaca): satu perintah
  kosong 7            box 7 sudah dilihat: memang tidak ada tanggal terbaca (wajib untuk kotak yang ditandai 'periksa')
  lewati              lewati foto ini (tetap "auto"), lanjut berikutnya
  buka                buka ulang gambarnya
  keluar              berhenti (aman; lanjutkan kapan saja, foto yang sudah ok tidak diulang)
"""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable


def todo():
    order = [r["foto"] for r in csv.DictReader(open("review_tanggal_prioritas.csv", encoding="utf-8"))]
    names = sorted(f[:-5] for f in os.listdir("label_specs_expiry") if f.endswith(".json"))
    order += [n for n in names if n not in order]
    return [n for n in order if os.path.exists(f"label_specs_expiry/{n}.json")
            and json.load(open(f"label_specs_expiry/{n}.json")).get("status") != "selesai"]


def se(*args):
    return subprocess.run([PY, "tools/set_expiry.py", *args], text=True, capture_output=True)


def show(name):
    r = se(name, "--show")
    print(r.stdout or r.stderr, end="")
    e = json.load(open(f"label_specs_expiry/{name}.json"))
    done = {x.get("box") for x in e.get("expiry_label", [])} | set(e.get("kosong", []))
    sisa = [b for b in e.get("periksa", []) if b not in done]
    if sisa:
        print(f"  !! PERIKSA box {sisa}: tajam tapi belum berlabel -> tambah label, atau 'kosong N' bila memang tidak ada tanggal")


def open_sheet(name):
    r = subprocess.run([PY, "tools/lembar_tanggal.py", name], text=True, capture_output=True)
    p = r.stdout.strip() or f"preview_expiry/{name}_expiry.jpg"
    subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    lewati = set()
    while True:
        rest = [n for n in todo() if n not in lewati]
        selesai = 163 - len(todo())
        if not rest:
            print(f"\nTidak ada foto tersisa (selesai {selesai}/163, dilewati {len(lewati)}).")
            break
        name = rest[0]
        print(f"\n===== {name}   (selesai {selesai}/163, sisa {len(rest)}) =====")
        open_sheet(name)
        show(name)
        while True:
            try:
                cmd = input("perintah [ok / tgl N TGL / hapus N / tambah N T1 T2 TGL / kosong N / buang N / lewati / buka / keluar]: ").split()
            except EOFError:
                return
            if not cmd:
                continue
            c = cmd[0].lower()
            if c == "keluar":
                return
            if c == "buka":
                open_sheet(name)
                continue
            if c == "lewati":
                lewati.add(name)
                break
            if c == "ok":
                r = se(name, "--ok")
                print(r.stdout or r.stderr, end="")
                if r.returncode == 0:
                    break
                continue
            if c == "tgl" and len(cmd) == 3:
                r = se(name, "--date", cmd[1], cmd[2])
            elif c == "buang" and len(cmd) == 2:           # hapus label + catat 'kosong' (tanggal buram / tidak terbaca)
                se(name, "--drop", cmd[1])
                r = se(name, "--kosong", cmd[1])
            elif c == "kosong" and len(cmd) == 2:
                r = se(name, "--kosong", cmd[1])
            elif c == "hapus" and len(cmd) == 2:
                r = se(name, "--drop", cmd[1])
            elif c == "tambah" and len(cmd) == 5:
                r = se(name, "--add", cmd[1], cmd[2], cmd[3], "--date", cmd[4])
            elif c == "tambah" and len(cmd) == 7:      # tambah N T1 T2 V1 V2 TANGGAL (batas lebar: label rapat)
                r = se(name, "--add", cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], "--date", cmd[6])
            else:
                print("  format salah. contoh: tgl 14 2028-01-31 | hapus 14 | tambah 5 0.45 0.60 0.3 0.6 2028-09-30")
                continue
            print((r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")
            show(name)
            open_sheet(name)


if __name__ == "__main__":
    main()
