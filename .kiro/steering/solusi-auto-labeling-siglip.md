# Solusi Auto-Labeling Sub-Kelas (Hybrid: Zero-Shot SigLIP + Clustering + Active Learning)

> Status: **terimplementasi (taksonomi v2)**. Pendekatan: **B (hybrid)**.
> Pipeline berjalan sebagai Tahap S0-S6 di dalam `siglip_embedding_probe_colab.ipynb`
> (bukan notebook terpisah). Output ada di `sublabel_v2/` (sublabels.csv, taxonomy.json, dst.)
> dan sub-head di `sublabel_head_v2_v3clean_r1/`. **`sublabel_v2/taxonomy.json` adalah sumber
> kebenaran taksonomi**; dokumen ini menjelaskan prinsip & metode (draf awal taksonomi di §3
> sudah digantikan oleh v2 — lihat §3).

## 1. Masalah

- Dataset bersih di `Dataset/` dengan 3 label kasar (`0_Recylable`, `1_Organik`, `2_B3`).
  (Angka komposisi di bawah adalah snapshot awal ~29.134 gambar; setelah data cleaning v3clean
  jumlah berubah — lihat `product.md`/manifest. `sublabels.csv` v2 mencakup 28.965 gambar.)
- Fitur panduan daur ulang butuh label yang lebih spesifik (material / jenis objek).
- Relabel manual dari nol tidak efisien.

### Komposisi sumber (dari prefix nama file)

| Kelas | Sumber utama | Catatan |
|---|---|---|
| Recylable | `R_` 9.384, `clothes` 2.556, `Image` 206, `cardboard` 75, `O_` 29, beberapa foto minyak jelantah | `R_` (Kaggle O/R) tidak punya sub-label; ~21% kelas ini tekstil |
| Organik | `O_` 12.369, `biological` 933, `Image` 181, `R_` 16 | `O_` tidak punya sub-label |
| B3 | `battery`+`baterai` 1.292, `trash` 829, `tabletcapsule` 497, `Image` 190 | |

Implikasi:
- ~5.300 gambar sudah punya sub-label implisit dari nama file -> **gold set gratis** untuk evaluasi.
- Pekerjaan berat ada di `R_` (Recylable) dan sebagian B3 (`trash`, `Image`).
- File `O_` di Recylable dan `R_` di Organik menandakan noise label kategori; jangan diasumsikan label kategori selalu benar.

## 2. Prinsip

1. Sub-kelas dibedakan berdasarkan **instruksi daur ulang yang berbeda**, bukan per objek.
2. **Reuse embedding yang sudah ada.** `.npz` di `ROOT/embeddings/<model>_naug_2_2_5/` berisi
   `pooler_output` mentah `SiglipVisionModel` `google/siglip-so400m-patch14-384`, identik dengan
   `get_image_features`. Zero-shot cukup memakai **text tower dari checkpoint yang sama**;
   gambar tidak perlu diproses ulang. Jangan pakai checkpoint lain (mis. `siglip-base-patch16-224`),
   ruang embedding-nya berbeda.
3. Label hanya ditentukan di **view 0** per `orig_idx`, lalu disebarkan ke view augmentasi.
4. Pseudo-label **tidak pernah** dipakai sebagai ground truth evaluasi. Val/test sub-head hanya dari label terverifikasi manusia.
5. Split train/val/test tetap mengikuti manifest yang ada; tidak membuat split baru.

## 3. Taksonomi (implementasi v2)

**Sumber kebenaran: `sublabel_v2/taxonomy.json`.** Draf v1 (plastik_botol/kaca/logam dst.)
sudah digantikan oleh v2 di bawah; setiap kategori punya `lainnya` sebagai fallback open-set.
Nama sub-label Bahasa Indonesia, prompt Bahasa Inggris, dan tiap sub-label memikul teks verbatim
`tindakan_konsumen_pra_buang` + `rute_penyaluran_pengolahan` (wajib diisi user, bukan dikarang).

### `0_Recylable` (k_cluster 40)
| Sub-label | Identifikasi | Kebijakan kelas kecil |
|---|---|---|
| `Tekstil` | gold `^clothes` | latih |
| `Kertas_Karton` | gold `^cardboard` | latih |
| `Minyak_Jelantah` | gold (nama file minyak) | rule_only |
| `Plastik_Kemasan` | prior `^R_` (Kaggle Recyclable) | latih |
| `Plastik_Campuran` | prior `^(Image_|images)` | merge:Plastik_Kemasan |
| `Anorganik_Lainnya` | generik (penampung sisa) | merge:Plastik_Kemasan |
| `lainnya` | fallback open-set | — |

### `1_Organik` (k_cluster 30)
| Sub-label | Identifikasi | Kebijakan kelas kecil |
|---|---|---|
| `Limbah_Biologis` | gold `^biological` | latih |
| `Sisa_Makanan` | prior `^(O_|Image_)` (Kaggle Organic) | latih |
| `Sampah_Kebun` | serasah/kebun (penampung sisa) | merge:Sisa_Makanan |
| `lainnya` | fallback open-set | — |

### `2_B3` (k_cluster 30)
| Sub-label | Identifikasi | Kebijakan kelas kecil |
|---|---|---|
| `Baterai` | gold `^(battery|baterai)` | latih |
| `Medis_Farmasi` | gold `^tabletcapsule` | latih |
| `Aerosol_Elektronik` | prior `^(trash|Image_)` | latih |
| `B3_Rumah_Tangga` | B3 spesifik/residu (penampung sisa) | merge:Aerosol_Elektronik |
| `lainnya` | fallback open-set | — |

Kebijakan **Kelas_Kecil**: sub-label dengan sampel < ambang (`MIN_TRAIN_SUBHEAD`) tidak dilatih
sendiri — `merge:<target>` digabung ke sub-label lain, `rule_only` hanya diberi dari aturan nama
file (tak masuk head). Status kalibrasi `τ` per kategori ada di `taxonomy.json` (`tau_constrained`).

## 4. Struktur konfigurasi

Konfigurasi hidup di sel **S0a** notebook (`TAXONOMY_SPEC`, `SL_PARAMS`, `TAXONOMY_VERSION`,
`SL_DIR`, `SUB_RUN_NAME`) lalu ditulis ke `taxonomy.json` setelah lolos `validate_taxonomy`.
`build_taxonomy()` **sengaja gagal** selama placeholder `<<ISI DARI USER>>` masih ada di field
tindakan/rute. Setiap perubahan sub-label / prompt / pola nama file / teks menaikkan
`TAXONOMY_VERSION` agar output versi lama tetap utuh. Prompt Bahasa Inggris (SigLIP dilatih dengan
caption Inggris); minimal 2 prompt non-kosong per sub-label.

## 5. Pipeline

### Tahap 1: Clustering untuk finalisasi taksonomi
- Per kelas, k-means (k ~30-50) pada embedding view 0 train yang L2-normalized.
- Simpan grid gambar per cluster (~16 sampel terdekat ke centroid).
- Manusia memberi nama tiap cluster -> revisi taksonomi + memberi seed label awal (label cluster untuk sampel yang dekat centroid).
- Output: taksonomi final + prompt yang direvisi.

### Tahap 2: Zero-shot dengan text tower so400m
Teks di-encode sekali, lalu skor = perkalian matriks dengan embedding yang sudah ada.

```python
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device).eval()
proc  = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

@torch.no_grad()
def encode_text(prompts):
    # SigLIP dilatih dengan padding="max_length"; padding lain menurunkan kualitas.
    inp = proc(text=prompts, padding="max_length", return_tensors="pt").to(device)
    return F.normalize(model.get_text_features(**inp).float(), dim=-1)

def build_prototypes(prompts_by_sub):
    # Prompt ensembling: rata-rata embedding teks ternormalisasi per sub-label (lebih stabil dari max).
    names = list(prompts_by_sub)
    protos = torch.stack([F.normalize(encode_text(prompts_by_sub[n]).mean(0), dim=-1) for n in names])
    return names, protos

scale, bias = model.logit_scale.exp(), model.logit_bias
def logits(img_emb, protos):            # img_emb: [N,1152] mentah dari .npz
    return scale * F.normalize(img_emb, dim=-1) @ protos.T + bias
```

Aturan keputusan:
- **Constrained:** kandidat hanya sub-label kategori gambar itu. Pakai softmax antar kandidat dan `margin = p_top1 - p_top2`.
  Jangan pakai threshold sigmoid absolut (mis. 0.3); skor sigmoid zero-shot SigLIP umumnya kecil.
- **Fallback:** `margin < τ` -> `lainnya` + masuk antrian review. `τ` dikalibrasi di gold set, bukan ditebak.
- **Global (cek label kategori):** argmax di seluruh sub-label semua kategori. Jika kategorinya berbeda dari label
  kategori dengan margin besar -> flag `cek_kategori`. Ini sekaligus menjadi input untuk pembersihan dataset (goal #2).
- Kategori tanpa sub-kelas (Organik) langsung mendapat `DEFAULT_SUBLABEL`, tetapi tetap ikut cek global.

### Tahap 3: Evaluasi di gold set dari nama file
| Prefix | Sub-label |
|---|---|
| `clothes` | `tekstil` |
| `cardboard` | `kertas_karton` |
| `battery`, `baterai` | `baterai` |
| `tabletcapsule` | `obat_medis` |

- Hitung precision/recall per sub-label dan confusion matrix zero-shot; kalibrasi `τ`.
- Mapping prefix tetap bisa salah. Cek acak sebagian sebelum dianggap gold.
- Gold set hanya mencakup sebagian sub-label. Sub-label lain (mis. `plastik_botol`, `kaca`) dievaluasi dari label manual Tahap 4.

### Tahap 4: Active learning (iterasi 2-3 putaran)
1. Data berlabel = gold set + seed cluster + koreksi manual.
2. Latih probe ringan di embedding view 0 (`LogisticRegression(class_weight='balanced')` per kategori, atau kNN cosine).
   Fitur zero-shot boleh dipakai sebagai prior/inisialisasi.
3. Prediksi seluruh data, lalu pilih untuk dilabeli manusia:
   - sampel dengan margin terendah per sub-label (paling informatif);
   - **sampel acak per sub-label** (estimasi precision yang tidak bias);
   - sebagian isi `lainnya` (mengecek sub-label yang terlewat).
4. Ulangi sampai tiap sub-label mencapai precision >= 0.90 di sampel acak, dan jumlah `lainnya` stabil.

Probe dilatih dengan label manusia, bukan hanya pseudo-label, untuk menghindari circularity
(head yang dilatih dari pseudo-label embedding yang sama hanya meniru batas zero-shot).

### Tahap 5: Sub-head final
- Train sub-head (reuse pipeline head di `siglip_embedding_probe_colab.ipynb`) pada label final.
- View augmentasi mewarisi label view 0 dengan `orig_idx` yang sama. Group CV/resampling wajib per `orig_idx`.
- Seleksi model memakai **val terverifikasi manusia**; test disentuh sekali saja untuk model terpilih.
- `RUN_NAME` baru untuk setiap eksperimen sub-head.

## 6. Output

Artefak taksonomi v2 ada di **`sublabel_v2/`** (`SL_DIR = OUTPUT_DIR/sublabel_<TAXONOMY_VERSION>`;
artefak generated, jangan diedit manual). File utama:

| File | Isi |
|---|---|
| `sublabels.csv` | label final per gambar (path, orig_idx, split, kategori, sub_label, sumber, confidence, margin, flag) |
| `taxonomy.json` | sumber kebenaran taksonomi: kategori, sub_labels (prompt, pola nama file, teks tindakan/rute), `tau_*`, params, versi, content_hash |
| `zeroshot.csv` / `zeroshot_scores.npz` | skor & keputusan zero-shot per gambar |
| `cluster_*.csv`, `clusters/` | assignment k-means, review cluster, seed label, grid gambar |
| `al_round_*.csv`, `probe_pred_round_*.csv`, `al_status.json` | iterasi active learning |
| `gold_status.csv`, `spotcheck_*`, `kalibrasi_tau.csv` | gold set, spot check, kalibrasi τ |
| `cek_kategori.csv` | gambar terindikasi salah label kategori (umpan balik ke data cleaning) |
| `audit_log.csv` | riwayat koreksi manual per putaran |
| `output_manifest.json` | ringkasan jumlah baris & daftar kelas kecil |

Sub-head final (checkpoint + `perbandingan_subhead_*.csv`) di
`sublabel_head_<TAXONOMY_VERSION>_<DATA_VERSION>_r1/` (per kategori + `flat`, head `linear/mlp/mlp2`).
`path` memakai `/` relatif terhadap `Dataset/`, sama seperti manifest. Perubahan pipeline dicatat di
`documentation.md`.

## 7. Status

- [x] Tahap S0-S1: config/taksonomi v2 + label awal nama file + gold set + spot check
- [x] Tahap S2: clustering k-means per kategori
- [x] Tahap S3: zero-shot constrained + global, kalibrasi `τ` (sebagian kategori masih `perlu_kalibrasi_ulang` di `taxonomy.json`)
- [x] Tahap S4: active learning (3 putaran, `al_round_1..3.csv`)
- [x] Tahap S5: `sublabels.csv` + kebijakan kelas kecil
- [x] Tahap S6: sub-head v2/v3clean (`sublabel_head_v2_v3clean_r1/`)
- [ ] Lanjutan: finalisasi kalibrasi `τ` Organik/B3, isi/konfirmasi teks tindakan-rute dari user, review `lainnya` (1561) & `cek_kategori`
