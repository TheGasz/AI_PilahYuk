---
name: methodology-fault-fixer
description: Audit integritas metodologi eksperimen SigLIP lalu perbaiki otomatis fault yang aman/reversibel. Gunakan sebelum/sesudah menjalankan pipeline untuk menangkap kesalahan yang membatalkan kesahihan hasil - data leakage antar split, path backslash di manifest, RUN_NAME yang menimpa hasil lama, CV tanpa group orig_idx, standardisasi bocor, seleksi model via test. Auto-fix hanya hal objektif & reversibel (path separator, nama run duplikat, backup); fault data/hyperparameter hanya DILAPORKAN, tidak diubah otomatis.
---

# Methodology Fault Fixer

Pemeriksa + perbaik fault **metodologis** pada pipeline (bukan fault model/data). Fault metodologis
adalah pelanggaran aturan eksperimen yang membuat hasil **tidak sahih** meski angka terlihat bagus.

Filosofi keamanan: **deteksi semuanya, perbaiki otomatis hanya yang objektif dan reversibel.**
Yang menyentuh data gambar atau hyperparameter TIDAK pernah diubah otomatis — hanya dilaporkan,
karena itu keputusan yang butuh verifikasi manusia (lihat skill `embedding-data-diagnostics` dan
`model-performance-advisor` untuk jalur tersebut).

## Kapan dipakai

- Sebelum upload `preprocessing_output/` atau sebelum sesi training Colab.
- Setelah mengubah konfig/split, untuk memastikan tidak ada kebocoran yang membatalkan metrik.
- Saat ragu apakah hasil eksperimen sah untuk dilaporkan.

## Fault yang dideteksi

Dari aturan proyek (steering `product.md` / `tech.md` / `structure.md`):

| Fault | Kenapa fatal | Auto-fix? |
|---|---|---|
| Path pakai backslash `\` di manifest | tidak portabel ke Linux/Drive | YA (ubah ke `/`, backup `.bak`) |
| Path absolut di manifest | tidak portabel | lapor (butuh keputusan path dasar) |
| Leakage: rel_path sama di >1 split | metrik tidak jujur | lapor (perbaiki di preprocessing) |
| File di `dedup_log` muncul di manifest | duplikat ikut terlatih | lapor |
| Label di luar {Recylable,Organik,B3} | label corrupt | lapor |
| `RUN_NAME` menimpa `head_runs/<RUN>` yang sudah ada | hasil lama tertimpa | YA (sarankan suffix unik; `--apply` menulis config) |
| CV/resampling tanpa group `orig_idx` | view augmentasi bocor antar fold | lapor (tunjukkan baris kode) |
| `INPUT_NORM` != 'standardize' tanpa alasan | stats bisa bocor dari val/test bila salah fit | lapor |
| Indikasi seleksi model via metrik test | melanggar aturan "test sekali" | lapor |

Auto-fix selalu: (1) buat backup sebelum menulis, (2) idempoten, (3) hanya jalan bila `--apply`
diberikan. Tanpa `--apply` script hanya melakukan dry-run (lapor rencana perubahan).

## Cara pakai

```powershell
# Audit saja (dry-run, tidak mengubah apa pun)
python .kiro/skills/methodology-fault-fixer/fix_methodology.py

# Terapkan auto-fix yang aman (backup .bak dibuat lebih dulu)
python .kiro/skills/methodology-fault-fixer/fix_methodology.py --apply
```

Opsi:
- `--manifest-dir` : default `preprocessing_output`.
- `--notebook` : default `siglip_embedding_probe_colab.ipynb` (dipindai untuk pola berisiko).
- `--apply` : izinkan menulis perbaikan aman; tanpa ini hanya dry-run.

Exit code: 0 bila tidak ada fault yang belum tertangani; 1 bila ada fault yang butuh tindakan manual.

## Output

- Daftar fault per kategori: `[FIXED]` (sudah diperbaiki, ada path backup), `[WOULD-FIX]` (dry-run,
  jalankan `--apply` untuk menerapkan), `[MANUAL]` (butuh keputusan manusia, dengan saran langkah).
- Ringkasan: berapa auto-fixed, berapa butuh manual.

## Batasan (sengaja)

- TIDAK menghapus/relabel/memindah gambar. Perbaikan data dilakukan di `preprocessing_lokal.ipynb`
  setelah ditinjau; duplikat dipindah ke `Dataset/duplicates_removed/`, tidak dihapus.
- TIDAK mengubah hyperparameter untuk "memperbaiki" overfitting/underfitting — itu tugas advisor +
  keputusan manusia, dan tiap perubahan wajib ganti `RUN_NAME`.
- Deteksi pola kode CV di notebook bersifat heuristik (string match); verifikasi manual tetap perlu.
