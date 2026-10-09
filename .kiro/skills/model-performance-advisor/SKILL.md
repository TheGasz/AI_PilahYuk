---
name: model-performance-advisor
description: Analisis performa head classifier SigLIP lalu rekomendasikan aksi konkret untuk menaikkannya. Gunakan saat ingin tahu KENAPA metrik mentok dan APA langkah berdampak terbesar - membaca history training (overfitting/underfitting, plateau, early stop) dan tabel perbandingan head (headroom, trade-off Recall vs Precision B3), lalu menghasilkan rekomendasi terprioritas. Fokus Recall B3 dengan B3 Precision sebagai cek false alarm. Tidak memilih model dari metrik test.
---

# Model Performance Advisor

Penasihat performa untuk head classifier di atas embedding SigLIP. Berbeda dari
`embedding-data-diagnostics` (yang mencari data bermasalah), skill ini membaca **hasil pelatihan**
untuk menjawab tiga hal:

1. **Analisis penyebab** — kenapa error terjadi? (overfitting, underfitting, kelas minoritas kurang
   terwakili, plateau, head salah pilih).
2. **Analisis potensi** — berapa headroom tersisa dan aksi mana yang paling berdampak per effort.
3. **Rekomendasi terprioritas** — daftar langkah konkret terurut, fokus menaikkan Recall B3 tanpa
   meledakkan false alarm (B3 Precision).

## Kapan dipakai

- Metrik val mentok dan ingin tahu arah perbaikan yang paling berdampak.
- Setelah menjalankan beberapa head dan ada `perbandingan_head_<RUN>.csv`.
- Ingin membaca `history_<run>.json` untuk mendeteksi overfitting/underfitting/plateau.

## Prinsip diagnosa (sebab -> aksi)

Aturan yang dipakai advisor saat membaca history/registry:

- **Overfitting** (val_loss naik sementara train_loss turun; gap train_acc >> val_acc):
  -> naikkan dropout, perkuat weight decay, kurangi ukuran head, tambah augmentasi (terutama B3),
  pertimbangkan head lebih sederhana (linear/mlp vs resmlp).
- **Underfitting** (train_loss & val_loss sama-sama tinggi, acc rendah): -> head lebih kuat
  (mlp2/resmlp), latih lebih lama (early stop terlalu cepat), kurangi regularisasi, cek input norm.
- **Early stop terlalu dini** (best_epoch sangat awal, patience habis cepat): -> turunkan LR / pakai
  scheduler lebih landai, atau naikkan patience; cek apakah val_loss masih noisy.
- **B3 recall rendah / tertinggal** (val_recall_B3 << recall kelas lain, atau plateau): -> tambah
  augmentasi B3, naikkan bobot kelas B3 di focal loss, oversample view B3, atau threshold-tuning
  (turunkan ambang B3) — tapi pantau B3 Precision agar false alarm tak meledak.
- **B3 recall tinggi tapi B3 precision anjlok**: model over-trigger B3. -> kurangi bobot/threshold B3,
  bersihkan near-duplicate/label noise yang membuat batas kelas kabur.
- **Semua head ~99% dan rapat**: headroom dari arsitektur kecil. -> alihkan effort ke DATA
  (jalankan `embedding-data-diagnostics`) dan ke ensembling/threshold-tuning, bukan head lebih besar.
- **Satu pasangan kelas mendominasi error** (mis. Recylable<->Organik): -> fokus perbaikan data di
  pasangan itu (relabel ambigu, pisahkan sub-kategori), bukan tuning global.

## Metrik yang dipakai (sesuai konvensi proyek)

- Seleksi berdasarkan **val**: macro F1 utama; per-kelas Recall/Precision/F1; **Recall B3** prioritas
  dengan **B3 Precision** sebagai cek false alarm.
- **Test disentuh sekali** untuk model terpilih. Advisor TIDAK pernah merekomendasikan seleksi
  berdasarkan metrik test (lihat Rules proyek).
- Label index `{Recylable:0, Organik:1, B3:2}`.

## Input yang dibaca

- `history_<run>.json` (per epoch): `train_loss`, `train_acc`, `val_loss`, `val_acc`,
  `val_recall_B3`, `val_precision_B3`, `val_f1_B3`, `val_recall_Recylable`, `val_recall_Organik`, ...
- `perbandingan_head_<RUN>.csv`: tabel head diurut val macro F1 (kolom metrik val per head).

## Cara pakai

Jalankan terhadap folder `head_runs/<RUN>` (bisa offline, hanya butuh file hasil):

```powershell
# Analisis satu history training
python .kiro/skills/model-performance-advisor/advise.py --history <path>/history_<run>.json

# Analisis tabel perbandingan head
python .kiro/skills/model-performance-advisor/advise.py --comparison <path>/perbandingan_head_<RUN>.csv

# Keduanya sekaligus (disarankan)
python .kiro/skills/model-performance-advisor/advise.py --run-dir <path>/head_runs/<RUN>
```

Opsi:
- `--run-dir` : folder berisi `history_*.json` dan `perbandingan_head_*.csv` (auto-discover).
- `--b3-recall-target` : target Recall B3 untuk hitung gap (default 0.99).
- `--out` : tulis rekomendasi ke file markdown (opsional).

## Output

- **Diagnosa per history**: status (overfit/underfit/sehat), best epoch, gap train-val, tren
  val_recall_B3 (naik/plateau/turun), apakah early stop wajar.
- **Analisis potensi**: head terbaik vs terburuk per val macro F1, selisih, trade-off B3 recall vs
  precision antar head, perkiraan headroom.
- **Rekomendasi terprioritas (P1/P2/P3)**: tiap item punya sebab, aksi, dan di mana menerapkannya
  (Cell/konfig di `siglip_embedding_probe_colab.ipynb`, atau loop data lewat
  `embedding-data-diagnostics` + `preprocessing_lokal.ipynb`).

## Catatan penting

- Advisor memberi arah, bukan jaminan. Setiap perubahan konfig = **ganti `RUN_NAME`** agar hasil lama
  tak tertimpa, lalu bandingkan val sebelum/sesudah.
- Bila rekomendasi mengarah ke perbaikan data, lanjutkan dengan skill `embedding-data-diagnostics`.
- Jangan pernah mengubah pilihan model berdasarkan test; test hanya untuk laporan final satu kali.
