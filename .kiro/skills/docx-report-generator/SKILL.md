---
name: docx-report-generator
description: Membuat atau memperbarui laporan .docx proyek Senpro memakai python-docx, mengikuti pola generator yang sudah ada (generate_dataset_report.py, generate_pure_dataset_review.py). Gunakan saat user minta laporan dataset/progress/review baru, atau update angka pada laporan existing. Prosa Bahasa Indonesia, script snake_case di root repo, angka harus konsisten dengan manifest.
---

# DOCX Report Generator

Panduan membuat/memelihara skrip generator laporan `.docx` untuk proyek klasifikasi sampah SigLIP.
Laporan dibuat lewat skrip Python `python-docx` yang di-commit, BUKAN mengedit `.docx` langsung,
supaya reproducible.

## Kapan dipakai

- User minta laporan baru (review dataset, progress, evaluasi model, dsb).
- User minta memperbarui angka pada laporan yang sudah ada setelah dataset/metrik berubah.

## Konvensi wajib (steering proyek)

- **Bahasa Indonesia** untuk seluruh prosa laporan (match dokumen existing).
- Skrip baru diletakkan **di root repo** bersebelahan dengan generator yang ada, nama **snake_case**:
  `generate_<sesuatu>_report.py`.
- Perintah menjalankan (contoh existing):
  ```powershell
  python generate_dataset_report.py        # -> Laporan_Review_Dataset_Senpro.docx
  python generate_pure_dataset_review.py    # -> Review_Murni_Dataset.docx
  ```
- Angka dataset HARUS konsisten dengan sumber kebenaran. Jangan menebak. Ambil dari:
  - `preprocessing_output/train.csv|val.csv|test.csv` (jumlah & distribusi split),
  - `preprocessing_output/class_weights.json`,
  - `preprocessing_output/dedup_log.csv` (jumlah duplikat dipindah),
  - `documentation.md` dan metrik di steering `product.md` (hasil test/val).

  Lebih baik skrip MEMBACA CSV/JSON dan menghitung angka saat runtime daripada menuliskan angka
  secara hardcode, agar laporan selalu sinkron dengan dataset terbaru.

## Angka kunci saat ini (verifikasi ulang sebelum dipakai)

- Total valid: 30.154 (Organik 13.744 / Recylable 12.944 / B3 3.466 agregat; per-split lihat CSV).
- Split: Train 21.107 / Val 4.523 / Test 4.524.
- Duplikat identik dipindah: 515 (ke `Dataset/duplicates_removed/`, tidak dihapus).
- Kelas minoritas: B3 (prioritas metrik: Recall B3).

Catatan: nilai di `product.md` menyebut 4524 test (Recylable 1942, Organik 2062, B3 520) — selaras
dengan output manifest. Jika ada selisih antar sumber, hitung ulang dari CSV dan laporkan ke user.

## Pola python-docx proyek

Ikuti helper yang sudah dipakai `generate_dataset_report.py`:

```python
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_heading(doc, text, level):
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = None  # paksa warna default (hindari biru tema)

def add_bullet(doc, title, content):
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(title).bold = True
    p.add_run(f": {content}")

doc = Document()
title = doc.add_heading('Judul Laporan', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
# ... isi ...
doc.save('Nama_Laporan.docx')
```

## Checklist sebelih selesai

1. Skrip membaca angka dari sumber kebenaran (CSV/JSON) bila memungkinkan.
2. Jalankan skrip, pastikan `.docx` ter-generate tanpa error.
3. Konfirmasi `python-docx` terpasang (`pip install python-docx` bila belum).
4. Prosa Bahasa Indonesia, heading pakai `add_heading` (warna default).
5. Jangan commit `Dataset/`/`Archive/`; `.docx` dan skrip boleh di-commit.
