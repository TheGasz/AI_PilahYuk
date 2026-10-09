---
name: embedding-data-diagnostics
description: Analisis data-centric di atas embedding SigLIP (.npz) untuk menemukan data yang menaikkan atau menurunkan performa model klasifikasi sampah 3 kelas. Gunakan saat ingin meningkatkan metrik (fokus Recall B3) lewat perbaikan data, bukan head yang lebih besar - mendeteksi label noise, near-duplicate lintas split, sampel ambigu/outlier, dan pasangan kelas yang paling sering tertukar. Output: daftar kandidat terurut + ringkasan untuk diumpan balik ke preprocessing_lokal.ipynb.
---

# Embedding Data Diagnostics (data-centric)

Toolkit "data scientist" untuk proyek SigLIP waste-classification. Premisnya: semua head sudah ~99%,
jadi perbaikan selanjutnya datang dari **data**, bukan arsitektur. Skill ini membedah ruang embedding
dan prediksi model untuk menjawab: **data mana yang membantu, dan data mana yang menahan/merusak
performa** — dengan prioritas pada **Recall B3** (B3 Precision sebagai cek false alarm).

## Kapan dipakai

- Metrik mentok di ~99% dan ingin naik lewat pembersihan data.
- Mau tahu penyebab error residual (mis. Recylable <-> Organik) secara kuantitatif.
- Sebelum/selama siklus: diagnosa -> perbaiki di `preprocessing_lokal.ipynb` -> re-ekstrak embedding.

## Prinsip data-centric (apa yang menaikkan vs menurunkan model)

Hal-hal yang biasanya **menurunkan** performa dan harus dicari:
1. **Label noise** — gambar yang label aslinya kemungkinan salah. Sinyal: label berbeda dari mayoritas
   tetangga terdekatnya di embedding space (kNN disagreement tinggi) DAN diprediksi salah dengan
   confidence tinggi. Ini racun terbesar untuk kelas minoritas B3.
2. **Near-duplicate lintas split** — gambar train yang nyaris identik (cosine sangat tinggi) dengan
   gambar val/test. Ini membocorkan metrik: skor terlihat bagus tapi tidak jujur. (Dedup lama hanya
   pixel-exact; embedding menangkap duplikat "mirip tapi tak identik".)
3. **Outlier / off-distribution** — gambar yang jauh dari centroid kelasnya (mis. meme, teks, watermark,
   objek campur). Menambah noise gradient.
4. **Sampel ambigu** — berada di perbatasan dua kelas (margin probabilitas kecil). Sering = foto yang
   memang mengandung dua jenis sampah; kandidat untuk relabel/aturan atau dibuang.

Hal yang **menaikkan** performa: menambah contoh B3 yang bersih & beragam, membuang/relabel noise di
atas, dan memutus kebocoran near-duplicate. Augmentasi ekstra B3 membantu recall TAPI augmentasi tak
bisa memperbaiki label yang salah — bersihkan dulu, augmentasi kemudian.

## Kontrak data (dari notebook Colab)

`.npz` per split di `EMBEDDING_DIR` berisi key:
`embeddings [M,1152] float32` (pooler_output), `labels [M] int`, `label_names [M] str`,
`paths [M] str`, `orig_idx [M] int` (view gambar sama -> orig_idx sama; kunci anti-leakage),
`view_idx [M] int8` (0 = asli/bersih, >=1 augmentasi).

Label index: `{Recylable:0, Organik:1, B3:2}`.

Aturan analisis:
- Untuk analisis per-gambar gunakan **view 0 saja** (`view_idx == 0`) agar augmentasi tidak
  menggandakan satu gambar.
- Normalisasi/standardisasi apa pun yang butuh statistik harus dihitung **dari train saja**.
- Jangan pernah memilih model dari metrik test; skill ini untuk diagnosa data, bukan seleksi model.

## Cara pakai

Jalankan terhadap `.npz` yang sudah diunduh dari Drive (bisa offline, tanpa GPU/SigLIP):

```powershell
python .kiro/skills/embedding-data-diagnostics/diagnose_embeddings.py --embedding-dir <folder_npz>
```

Opsi penting:
- `--embedding-dir` : folder berisi `embeddings_train.npz|val.npz|test.npz` (wajib).
- `--k` : tetangga untuk kNN label-noise/ambiguity (default 15, samakan dgn konvensi kNN proyek).
- `--dup-threshold` : ambang cosine near-duplicate lintas split (default 0.97).
- `--top` : berapa kandidat teratas ditampilkan/ditulis per kategori (default 50).
- `--out-dir` : folder output CSV ringkasan (default `Output_data_diagnostics`).

Output (CSV, path pakai label+path asli supaya langsung bisa ditelusuri):
- `label_noise_candidates.csv` — gambar paling mungkin salah label, terurut skor.
- `cross_split_duplicates.csv` — pasangan train<->(val/test) yang terlalu mirip (potensi leakage).
- `ambiguous_samples.csv` — sampel margin-kecil / perbatasan kelas.
- `class_centroid_outliers.csv` — gambar terjauh dari centroid kelasnya.
- `confusion_summary.csv` — pasangan kelas yang paling sering tertukar (via kNN leave-one-out).
- Ringkasan dicetak ke stdout, termasuk fokus B3.

## Menindaklanjuti (loop data-centric)

1. Tinjau manual kandidat teratas (jangan auto-apply; keputusan label butuh mata manusia).
2. Terapkan keputusan di `preprocessing_lokal.ipynb` (relabel/buang/pindah), bukan edit CSV manual.
   Duplikat dipindah ke `Dataset/duplicates_removed/`, tidak dihapus.
3. Re-run preprocessing -> re-ekstrak embedding ke `EMBEDDING_DIR` **baru** (ganti nama) -> update
   `documentation.md`.
4. Bandingkan metrik val sebelum/sesudah untuk konfirmasi perbaikan (fokus Recall B3 + B3 Precision).
