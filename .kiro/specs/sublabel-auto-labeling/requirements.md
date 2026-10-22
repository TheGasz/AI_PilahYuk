# Requirements Document

## Introduction

Fitur ini menambahkan **label sub-kelas** (13 sub-label) di bawah 3 kategori kasar (`Recylable`, `Organik`, `B3`) supaya fitur panduan daur ulang aplikasi dapat menampilkan *tindakan konsumen pra-buang* dan *rute penyaluran/pengolahan* yang spesifik. Pendekatan yang dipilih adalah **hybrid (B)**: label awal dari heuristik nama file, clustering untuk meninjau taksonomi, zero-shot SigLIP (text tower `google/siglip-so400m-patch14-384`) di atas embedding `.npz` yang sudah ada, kalibrasi di gold set dari nama file, active learning dengan verifikasi manusia, lalu training sub-head hierarkis. Seluruh langkah komputasi ditambahkan sebagai **bagian baru di notebook `siglip_embedding_probe_colab.ipynb`**, memakai ulang loader, fungsi training head, dan utilitas perbandingan yang sudah ada.

### Taksonomi final (dari user, menggantikan draft v1 di steering)

| Kategori | Sub_Label | Jumlah (heuristik) | Metode identifikasi utama |
|---|---|---|---|
| B3 | Aerosol_Elektronik | 1.103 | Eksplisit/Indikasi (`trash*` / `Image*`) |
| B3 | B3_Rumah_Tangga | 288 | B3 spesifik / residu |
| B3 | Baterai | 1.292 | Eksplisit (`battery*` / `baterai*`) |
| B3 | Medis_Farmasi | 497 | Eksplisit (`tabletcapsule*`) |
| Organik | Limbah_Biologis | 933 | Eksplisit (`biological*`) |
| Organik | Sampah_Kebun | 17 | Organik serasah/kebun |
| Organik | Sisa_Makanan | 12.550 | Kaggle Organic standard |
| Recylable | Anorganik_Lainnya | 34 | Anorganik generic |
| Recylable | Kertas_Karton | 75 | Eksplisit (`cardboard*`) |
| Recylable | Minyak_Jelantah | 14 | Eksplisit (nama file minyak goreng) |
| Recylable | Plastik_Campuran | 222 | Dataset Image anorganik |
| Recylable | Plastik_Kemasan | 9.384 | Kaggle Recyclable standard |
| Recylable | Tekstil | 2.556 | Eksplisit (`clothes*`) |

Teks `tindakan_konsumen_pra_buang` dan `rute_penyaluran_pengolahan` per sub-label disimpan apa adanya (verbatim dari user) di Taxonomy_Config.

### Temuan analisis yang menjadi dasar requirement

Hasil pemeriksaan workspace (manifest `preprocessing_output/*.csv` dan notebook):

1. **Cakupan data = manifest, bukan isi folder.** Manifest v3clean berisi 28.965 gambar (train 20.275 / val 4.345 / test 4.345; Recylable 12.285, Organik 13.500, B3 3.180). Jumlah taksonomi user cocok persis dengan manifest (B3 = 3.180). Folder `Dataset/2_B3/` berisi 3.349 file; **169 file B3 (nama hash/angka) tidak ada di manifest** dan karenanya tidak punya embedding. File ini dilaporkan, bukan dilabeli.
2. **Heuristik nama file yang mereproduksi jumlah user** (prefix di manifest): B3 `battery*`+`baterai*` 1.292, `tabletcapsule*` 497, `trash*` 829 + `Image_*` 274 = 1.103, sisa (hash + 2 `O_`) 288. Organik `biological*` 933, `O_*` 12.369 + `Image_*` 181 = 12.550, sisa (`R_` 16 + `clothes` 1) = 17. Recylable `clothes*` 2.556, `cardboard*` 75, `R_*` 9.384, `Image_*`+`images*` 222, nama file minyak (`*cooking-oil*`, `Dau_an*`, `*oil*`) 14, sisa (`O_` 29 + `istockphoto*` 5) 34.
3. **Label heuristik adalah prior, bukan kebenaran.** `Plastik_Kemasan` = seluruh `R_` (Kaggle O/R), yang juga berisi kaca, logam, dan kertas (teks rute user pun menyebut plastik, kaca, kaleng). `Sisa_Makanan` = seluruh `O_` + `Image_`. `Sampah_Kebun` (17) dan `Anorganik_Lainnya` (34) adalah file sisa hasil relabel di `cleaning_log.csv`, bukan identifikasi isi gambar.
4. **Kelas sangat kecil**: Minyak_Jelantah 14, Sampah_Kebun 17, Anorganik_Lainnya 34; metrik val/test per kelas tidak reliabel.
5. **Kebocoran sumber Roboflow.** File `*.rf.<hash>.jpg` adalah salinan augmentasi dari satu gambar sumber. Contoh: `2020-sti-oil-black...` muncul di train dan 2× di val; `istockphoto-1479603703...` di train dan test; `disadvantages-of-reusing-cooking-oil...` di train dan test. Selain itu `2020-sti-oil-black...` tampaknya **oli mesin bekas**, bukan minyak jelantah, sehingga perlu dicek kategorinya.
6. **Noise kategori**: `O_` di Recylable dan `R_` di Organik sebagian berasal dari relabel di `cleaning_log.csv` (sudah ditinjau), sebagian belum.
7. **Notebook**: `EMBEDDING_DIR` aktif adalah `embeddings/siglip-so400m-patch14-384_naug_2_2_5_v3clean/` (`DATA_VERSION='v3clean'`), bukan `_naug_2_2_5` seperti di steering. `.npz` menyimpan `paths` absolut (path saat ekstraksi) dan `orig_idx` = indeks baris manifest split tersebut. Hanya `SiglipVisionModel` yang dimuat; text tower belum ada. Fungsi `metrics_from_preds`, `evaluate_head`, `train_head`, `register_result`, `knn_probs`, `save_head`, `comparison_table`, `report_test` terikat ke 3 kelas lewat global (`NUM_CLASSES`, `IDX_TO_LABEL`, `X_tr/y_tr/X_va/y_va`, `cw_tensor`, key history B3). Utilitas yang bisa dipakai ulang langsung: `load_embeddings`, `fit_input_norm`/`apply_input_norm`, `build_head`, `train_one_epoch_head`, `predict_proba`, `std_l2`, `top_neighbors`, `safe_write_csv`, `_path_key`, `mark_test_use`, pola `StratifiedGroupKFold` group `orig_idx` di Cell 12b.

## Glossary

- **Notebook**: `siglip_embedding_probe_colab.ipynb` di root repo.
- **Bagian_Sublabel**: kumpulan cell baru di Notebook untuk seluruh pipeline sub-label (Tahap 0–6).
- **Kategori**: label kasar `Recylable`, `Organik`, `B3` (indeks `{'Recylable': 0, 'Organik': 1, 'B3': 2}`; ejaan `0_Recylable` disengaja).
- **Sub_Label**: salah satu dari 13 label spesifik di tabel taksonomi final, masing-masing milik tepat satu Kategori.
- **Fallback_Sublabel**: nilai `lainnya` per Kategori untuk gambar yang belum dapat ditetapkan; berbeda dari Sub_Label `Anorganik_Lainnya`.
- **Taxonomy_Config**: file `taxonomy.json` sebagai satu-satunya sumber kebenaran taksonomi (versi, Sub_Label, teks tindakan & rute, prompt, aturan nama file, ambang, kebijakan kelas kecil).
- **TAXONOMY_VERSION**: string versi taksonomi (mis. `v2`); setiap perubahan Sub_Label, prompt, atau aturan menaikkan versi.
- **Output_Dir**: `ROOT_DIR/preprocessing_output/sublabel_<TAXONOMY_VERSION>/`.
- **Manifest**: `train.csv`, `val.csv`, `test.csv` (kolom `rel_path`, `label`) di `preprocessing_output/`.
- **Embedding_Store**: `embeddings_{train,val,test}.npz` di `EMBEDDING_DIR` aktif Notebook.
- **View_0**: baris `.npz` dengan `view_idx == 0` (gambar asli tanpa augmentasi). Val/test hanya berisi View_0.
- **orig_idx**: indeks gambar asli dalam Manifest split; semua view augmentasi satu gambar berbagi `orig_idx` yang sama.
- **Rel_Path**: path relatif terhadap `Dataset/` dengan separator `/`, sama dengan kolom `rel_path` Manifest.
- **Filename_Heuristic**: aturan pola nama file (regex) per Sub_Label di Taxonomy_Config yang memberi label awal.
- **Gold_Set**: gambar dengan Sub_Label eksplisit dari nama file (`clothes*`→Tekstil, `cardboard*`→Kertas_Karton, `battery*`/`baterai*`→Baterai, `tabletcapsule*`→Medis_Farmasi, `biological*`→Limbah_Biologis, nama file minyak goreng→Minyak_Jelantah) yang lolos Spot_Check.
- **Spot_Check**: verifikasi manusia atas sampel acak per Sub_Label Gold_Set.
- **Source_Group**: kunci gambar sumber Roboflow (bagian nama file sebelum `.rf.`; file tanpa `.rf.` = Source_Group sendiri).
- **Reviewer**: manusia yang mengisi kolom `keputusan` di Review_Queue.
- **Review_Queue**: CSV berisi kolom `keputusan` yang diisi Reviewer, ditampilkan bersama grid gambar di Notebook.
- **Text_Prototype**: rata-rata embedding teks ternormalisasi L2 dari semua prompt satu Sub_Label, lalu dinormalisasi ulang.
- **Zero_Shot_Scorer**: komponen Bagian_Sublabel yang menghitung skor gambar-teks dari Embedding_Store dan Text_Prototype.
- **Margin**: selisih probabilitas softmax peringkat 1 dan peringkat 2 di antara kandidat, bernilai di [0, 1].
- **Tau_Constrained (τ)**: ambang Margin per Kategori untuk keputusan constrained, dikalibrasi di Gold_Set.
- **Tau_Global**: ambang Margin untuk flag `cek_kategori` pada keputusan global.
- **Probe_AL**: classifier ringan (`LogisticRegression(class_weight='balanced')` per Kategori atau kNN cosine) di embedding View_0 untuk active learning.
- **AL_Round**: satu putaran active learning (latih Probe_AL → pilih sampel → review → catat).
- **Verified_Label**: Sub_Label yang ditetapkan atau dikonfirmasi Reviewer, atau anggota Gold_Set yang lolos Spot_Check.
- **Pseudo_Label**: Sub_Label dari Filename_Heuristic non-gold, cluster, Zero_Shot_Scorer, atau Probe_AL yang belum diverifikasi.
- **Sublabel_Table**: `sublabels.csv` di Output_Dir, satu baris per gambar Manifest.
- **Cek_Kategori_Table**: `cek_kategori.csv` di Output_Dir.
- **Audit_Log**: `audit_log.csv` di Output_Dir, riwayat semua keputusan manual.
- **Kelas_Kecil**: Sub_Label yang jumlah Verified_Label train-nya di bawah `MIN_TRAIN_SUBHEAD` atau jumlah Verified_Label val-nya di bawah `MIN_EVAL_SUBLABEL`.
- **Sub_Head**: head classifier per Kategori yang memprediksi Sub_Label dari embedding.
- **Hierarchical_Classifier**: head 3-kelas yang sudah dipilih + Sub_Head milik Kategori hasil prediksi.
- **Flat_Head**: satu head yang langsung memprediksi seluruh Sub_Label (pembanding).
- **Eval_Set_Terverifikasi**: baris val/test yang berlabel Verified_Label.
- **Sub_Registry**: registry hasil sub-label yang terpisah dari `results` 3-kelas.
- **Tim_Proyek**: anggota tim Senpro yang memelihara dokumentasi dan steering.

## Requirements

### Requirement 1: Taxonomy_Config sebagai satu sumber kebenaran

**User Story:** Sebagai pengembang aplikasi, saya ingin taksonomi, teks tindakan, dan rute disimpan dalam satu file JSON, supaya fitur panduan daur ulang dapat membaca tindakan dan rute per Sub_Label tanpa menyalin teks ke kode.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL membangun Taxonomy_Config dari satu cell konfigurasi yang berisi, untuk setiap Sub_Label: nama Kategori, nama Sub_Label, `metode_identifikasi_utama`, `tindakan_konsumen_pra_buang`, `rute_penyaluran_pengolahan`, daftar prompt bahasa Inggris, dan daftar pola Filename_Heuristic.
2. THE Taxonomy_Config SHALL menyimpan teks `tindakan_konsumen_pra_buang` dan `rute_penyaluran_pengolahan` identik karakter-per-karakter dengan tabel taksonomi final dari user.
3. THE Taxonomy_Config SHALL berisi, per Kategori, entri Fallback_Sublabel `lainnya` dengan teks tindakan dan rute tingkat Kategori.
4. THE Taxonomy_Config SHALL berisi `TAXONOMY_VERSION`, `MODEL_CHECKPOINT`, nama `EMBEDDING_DIR` sumber, Tau_Constrained per Kategori, Tau_Global, `MIN_TRAIN_SUBHEAD`, `MIN_EVAL_SUBLABEL`, dan kebijakan Kelas_Kecil per Sub_Label.
5. WHEN Taxonomy_Config ditulis lalu dibaca kembali, THE Bagian_Sublabel SHALL menghasilkan struktur yang sama dengan struktur sebelum ditulis (round-trip).
6. IF dua Sub_Label memiliki nama sama, sebuah Sub_Label tidak memiliki Kategori yang valid, sebuah Sub_Label tidak memiliki minimal 2 prompt, atau teks tindakan/rute kosong, THEN THE Bagian_Sublabel SHALL menghentikan eksekusi dengan pesan yang menyebut Sub_Label dan field yang bermasalah.
7. WHERE sebuah Sub_Label baru ditambahkan ke cell konfigurasi, THE Bagian_Sublabel SHALL memproses Sub_Label tersebut di seluruh tahap tanpa perubahan kode di luar cell konfigurasi.
8. WHEN isi Sub_Label, prompt, atau Filename_Heuristic berubah, THE Bagian_Sublabel SHALL menulis hasil ke Output_Dir dengan TAXONOMY_VERSION baru dan membiarkan Output_Dir versi lama tetap utuh.

### Requirement 2: Cakupan data dan pemetaan ke Manifest

**User Story:** Sebagai peneliti, saya ingin setiap sub-label terkait tepat satu gambar Manifest dan split yang sudah ada, supaya tidak terjadi kebocoran atau split baru.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL memetakan setiap baris View_0 Embedding_Store ke Rel_Path Manifest split yang sama melalui `orig_idx`.
2. WHEN pemetaan `orig_idx` ke Rel_Path selesai, THE Bagian_Sublabel SHALL memverifikasi bahwa kunci `<folder_kelas>/<nama_file>` dari `paths` `.npz` sama dengan kunci dari Rel_Path untuk setiap baris, dan menghentikan eksekusi bila ada satu pun yang berbeda.
3. THE Bagian_Sublabel SHALL memakai split train/val/test dari Manifest apa adanya.
4. THE Sublabel_Table SHALL berisi tepat satu baris per gambar Manifest (28.965 baris untuk manifest v3clean) dengan kombinasi (`split`, `orig_idx`) yang unik.
5. WHEN Bagian_Sublabel dijalankan, THE Bagian_Sublabel SHALL menulis `di_luar_manifest.csv` di Output_Dir berisi file gambar di folder Kategori `Dataset/` yang tidak tercantum di Manifest (saat ini 169 file di `2_B3/`), tanpa memberi Sub_Label pada file tersebut.
6. THE Bagian_Sublabel SHALL membaca file di `Dataset/` dan `Archive/` hanya untuk ditampilkan, tanpa memindah, mengubah, atau menghapus file.
7. THE Sublabel_Table SHALL menyimpan kolom `path` sebagai Rel_Path dengan separator `/`.

### Requirement 3: Label awal dari Filename_Heuristic

**User Story:** Sebagai peneliti, saya ingin label awal dari nama file yang mereproduksi jumlah di taksonomi user, supaya ada titik awal yang bisa diaudit dan diperbaiki.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL memberi setiap gambar Manifest tepat satu Sub_Label awal milik Kategori gambar tersebut, dengan mencocokkan pola Filename_Heuristic sesuai urutan prioritas di Taxonomy_Config.
2. IF nama file tidak cocok dengan pola Sub_Label mana pun dan Kategori tersebut memiliki Sub_Label penampung sisa di Taxonomy_Config, THEN THE Bagian_Sublabel SHALL memberi Sub_Label penampung tersebut dengan `sumber='filename_sisa'`.
3. IF nama file tidak cocok dengan pola mana pun dan Kategori tersebut tidak memiliki Sub_Label penampung sisa, THEN THE Bagian_Sublabel SHALL memberi Fallback_Sublabel dan flag `review`.
4. WHEN label awal selesai, THE Bagian_Sublabel SHALL menampilkan tabel jumlah per Kategori × Sub_Label beserta selisihnya terhadap kolom `jumlah_sampel` taksonomi user.
5. IF jumlah label awal suatu Sub_Label berbeda dari `jumlah_sampel` taksonomi user, THEN THE Bagian_Sublabel SHALL mencetak peringatan yang menyebut Sub_Label, jumlah hasil, dan jumlah user, lalu melanjutkan eksekusi.
6. THE Bagian_Sublabel SHALL menandai label awal dari pola eksplisit Gold_Set dengan `sumber='filename_gold'` dan label awal lainnya dengan `sumber='filename_prior'`.
7. THE Bagian_Sublabel SHALL memperlakukan seluruh label `filename_prior` dan `filename_sisa` sebagai Pseudo_Label yang boleh diubah oleh tahap Zero_Shot_Scorer, Probe_AL, dan Reviewer.
8. FOR ALL gambar Manifest, menjalankan Filename_Heuristic dua kali dengan Taxonomy_Config yang sama SHALL menghasilkan Sub_Label awal yang identik (determinisme).

### Requirement 4: Gold_Set, Spot_Check, dan Source_Group

**User Story:** Sebagai peneliti, saya ingin gold set dari nama file diverifikasi sebelum dipakai, supaya kalibrasi dan evaluasi tidak bertumpu pada label yang salah.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL membentuk kandidat Gold_Set dari gambar berlabel `sumber='filename_gold'`.
2. WHEN kandidat Gold_Set terbentuk, THE Bagian_Sublabel SHALL menulis Review_Queue `spotcheck_gold.csv` berisi sampel acak per Sub_Label sebanyak `min(SPOTCHECK_N, jumlah)` (default `SPOTCHECK_N=50`) dengan seed tetap, dan menampilkan grid gambarnya.
3. WHEN Reviewer mengisi `keputusan` di `spotcheck_gold.csv`, THE Bagian_Sublabel SHALL menghitung precision Spot_Check per Sub_Label dan mencatat setiap keputusan ke Audit_Log.
4. IF precision Spot_Check suatu Sub_Label di bawah `GOLD_MIN_PRECISION` (default 0,95), THEN THE Bagian_Sublabel SHALL mengeluarkan Sub_Label tersebut dari Gold_Set dan menurunkan labelnya menjadi `filename_prior`.
5. IF Reviewer menolak gambar tertentu di Spot_Check, THEN THE Bagian_Sublabel SHALL mengeluarkan gambar tersebut dari Gold_Set dan memasukkannya ke antrian review berikutnya.
6. THE Bagian_Sublabel SHALL menghitung Source_Group untuk setiap gambar dan menulis `lintas_split_sumber.csv` berisi Source_Group yang anggotanya berada di lebih dari satu split.
7. WHILE sebuah gambar val/test memiliki Source_Group yang juga ada di train, THE Bagian_Sublabel SHALL memberi flag `lintas_split_sumber` pada gambar tersebut dan mengecualikannya dari kolom metrik `_nodup`.
8. IF `spotcheck_gold.csv` sudah berisi keputusan Reviewer, THEN THE Bagian_Sublabel SHALL menulis hasil baru ke file berakhiran `_baru.csv` dan mempertahankan file lama.

### Requirement 5: Clustering untuk meninjau taksonomi

**User Story:** Sebagai peneliti, saya ingin melihat kelompok visual per Kategori, supaya saya dapat memastikan taksonomi mencakup isi data dan mendapat seed label awal.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL menjalankan k-means per Kategori pada embedding View_0 train yang sudah distandardisasi (statistik train) dan dinormalisasi L2, dengan `k` per Kategori dari konfigurasi (default 30–50, dibatasi tidak melebihi jumlah gambar / 20) dan seed tetap.
2. WHEN clustering selesai, THE Bagian_Sublabel SHALL menyimpan satu grid PNG per cluster berisi hingga 16 gambar terdekat ke centroid di Output_Dir `/clusters/`.
3. WHEN clustering selesai, THE Bagian_Sublabel SHALL menulis Review_Queue `cluster_review.csv` berisi per cluster: Kategori, id cluster, ukuran, distribusi Sub_Label awal, dan kolom `keputusan` untuk diisi Reviewer dengan nama Sub_Label, `campuran`, atau `usul:<nama_baru>`.
4. WHEN Reviewer menetapkan Sub_Label pada sebuah cluster, THE Bagian_Sublabel SHALL memberi `sumber='cluster'` hanya pada `SEED_TOP_N` gambar train terdekat ke centroid cluster tersebut (default 16, yaitu gambar yang tampil di grid).
5. IF Reviewer mengisi `usul:<nama_baru>` pada sebuah cluster, THEN THE Bagian_Sublabel SHALL mencetak daftar usulan Sub_Label baru beserta jumlah cluster pendukung, tanpa mengubah Taxonomy_Config secara otomatis.
6. THE Bagian_Sublabel SHALL menjalankan clustering hanya pada split train.

### Requirement 6: Zero_Shot_Scorer dengan text tower so400m

**User Story:** Sebagai peneliti, saya ingin skor zero-shot dihitung langsung dari embedding yang sudah ada, supaya gambar tidak perlu diproses ulang oleh SigLIP.

#### Acceptance Criteria

1. THE Zero_Shot_Scorer SHALL memakai text tower dan parameter `logit_scale` serta `logit_bias` dari checkpoint yang sama dengan Embedding_Store (`google/siglip-so400m-patch14-384`).
2. IF `MODEL_CHECKPOINT` di Taxonomy_Config berbeda dari checkpoint yang dipakai membuat Embedding_Store, THEN THE Zero_Shot_Scorer SHALL menghentikan eksekusi dengan pesan yang menyebut kedua checkpoint.
3. THE Zero_Shot_Scorer SHALL men-tokenisasi prompt dengan `padding="max_length"`.
4. THE Zero_Shot_Scorer SHALL membentuk satu Text_Prototype per Sub_Label dari rata-rata embedding teks ternormalisasi seluruh prompt Sub_Label tersebut.
5. WHEN Text_Prototype selesai dihitung, THE Zero_Shot_Scorer SHALL menyimpan Text_Prototype ke cache di Output_Dir dengan kunci hash isi prompt, dan WHEN hash prompt tidak berubah, THE Zero_Shot_Scorer SHALL memakai cache tanpa memuat text tower.
6. THE Zero_Shot_Scorer SHALL menghitung skor sebagai `logit_scale × cos(embedding_gambar_L2, Text_Prototype) + logit_bias` dari embedding View_0 mentah `.npz`.
7. FOR ALL gambar, skor Zero_Shot_Scorer dari embedding `.npz` SHALL sama dengan skor dari embedding yang di-encode ulang oleh vision tower yang sama dalam toleransi 1e-2, diuji pada sampel 32 gambar.
8. WHEN text tower tidak lagi diperlukan, THE Zero_Shot_Scorer SHALL membebaskan memori GPU text tower sebelum tahap berikutnya.

### Requirement 7: Keputusan constrained dan cek kategori global

**User Story:** Sebagai peneliti, saya ingin sub-label dipilih hanya dari kandidat Kategori gambar dan sekaligus mendeteksi gambar yang tampak salah kategori, supaya hasil sub-label konsisten dan dataset dapat dibersihkan.

#### Acceptance Criteria

1. THE Zero_Shot_Scorer SHALL menghitung softmax hanya di antara Sub_Label milik Kategori gambar (constrained) dan mencatat Sub_Label peringkat 1, `confidence` (probabilitas peringkat 1), dan Margin.
2. WHEN Margin constrained sebuah gambar kurang dari Tau_Constrained Kategori tersebut, THE Zero_Shot_Scorer SHALL memberi Fallback_Sublabel dan flag `review`.
3. THE Zero_Shot_Scorer SHALL memakai ambang Margin softmax sebagai dasar keputusan, bukan ambang probabilitas sigmoid absolut.
4. FOR ALL nilai τ1 ≤ τ2, jumlah gambar yang menerima Fallback_Sublabel dengan τ2 SHALL lebih besar atau sama dengan jumlah dengan τ1 (monotonik).
5. FOR ALL gambar, Sub_Label hasil constrained SHALL merupakan Sub_Label milik Kategori gambar tersebut atau Fallback_Sublabel.
6. THE Zero_Shot_Scorer SHALL menghitung argmax global di seluruh Sub_Label semua Kategori untuk setiap gambar.
7. WHEN Kategori dari Sub_Label global berbeda dari Kategori gambar dan Margin global antara Sub_Label teratas dan Sub_Label teratas milik Kategori gambar lebih besar atau sama dengan Tau_Global, THE Zero_Shot_Scorer SHALL memberi flag `cek_kategori` dan menulis gambar tersebut ke Cek_Kategori_Table.
8. THE Cek_Kategori_Table SHALL berisi kolom `path`, `split`, `kategori`, `kategori_global`, `sub_label_global`, `margin_global`, `sudah_di_cleaning_log`, dan `keputusan`, dengan `sudah_di_cleaning_log=True` untuk gambar yang tercantum di `preprocessing_output/cleaning_log.csv`.
9. THE Zero_Shot_Scorer SHALL mengurutkan Cek_Kategori_Table dengan baris yang melibatkan Kategori B3 (asli atau global) di atas, lalu menurut `margin_global` menurun.
10. THE Bagian_Sublabel SHALL hanya melaporkan Cek_Kategori_Table; perubahan Kategori diterapkan di `preprocessing_lokal.ipynb` diikuti re-ekstraksi embedding ke `EMBEDDING_DIR` baru.

### Requirement 8: Kalibrasi τ dan evaluasi zero-shot di Gold_Set

**User Story:** Sebagai peneliti, saya ingin ambang τ dipilih dari data berlabel, supaya keputusan fallback tidak ditebak.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL mengkalibrasi Tau_Constrained per Kategori dengan menyapu kandidat τ di grid [0, 0,5] langkah 0,01 pada Gold_Set split train.
2. THE Bagian_Sublabel SHALL memilih τ terkecil yang membuat precision Sub_Label non-fallback pada Gold_Set train mencapai `TARGET_PRECISION` (default 0,90), dan mencatat coverage (fraksi gambar non-fallback) pada τ tersebut.
3. IF tidak ada τ di grid yang mencapai `TARGET_PRECISION` untuk suatu Kategori, THEN THE Bagian_Sublabel SHALL memilih τ dengan precision tertinggi, memberi peringatan, dan menyimpan alasannya di Taxonomy_Config.
4. IF Gold_Set suatu Kategori hanya mencakup satu Sub_Label, THEN THE Bagian_Sublabel SHALL mencetak peringatan bahwa kalibrasi Kategori tersebut bias ke Sub_Label itu dan menandai τ tersebut `perlu_kalibrasi_ulang` sampai Verified_Label dari Sub_Label lain tersedia.
5. WHEN kalibrasi selesai, THE Bagian_Sublabel SHALL menampilkan precision, recall, F1 per Sub_Label dan confusion matrix zero-shot pada Gold_Set val.
6. THE Bagian_Sublabel SHALL menulis τ terpilih, grid sapuan, dan metrik kalibrasi ke Taxonomy_Config dan `kalibrasi_tau.csv`.
7. THE Bagian_Sublabel SHALL memakai Gold_Set split test hanya pada evaluasi akhir model terpilih (Requirement 12).

### Requirement 9: Active learning dan Audit_Log

**User Story:** Sebagai peneliti, saya ingin memverifikasi sampel yang paling informatif dan sampel acak per sub-label, supaya label akhir mencapai precision target dengan usaha review yang terbatas.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL melatih Probe_AL per Kategori hanya dari Verified_Label split train, dengan fitur embedding View_0 dan opsi skor zero-shot sebagai fitur tambahan.
2. WHEN Probe_AL selesai dilatih, THE Bagian_Sublabel SHALL memprediksi seluruh gambar train dan menulis Review_Queue `al_round_<n>.csv` berisi, per Sub_Label: `AL_N_UNCERTAIN` gambar Margin terendah (default 20), `AL_N_RANDOM` gambar acak (default 20, seed tetap), dan `AL_N_FALLBACK` gambar dari Fallback_Sublabel per Kategori (default 20).
3. THE Review_Queue `al_round_<n>.csv` SHALL menandai asal pilihan setiap baris (`uncertain`, `random`, `fallback`) dan menampilkan grid gambar beserta Sub_Label prediksi dan 5 tetangga terdekat Verified_Label.
4. WHEN Reviewer mengisi `keputusan` (`ok`, `<Sub_Label>`, `lainnya`, atau `cek_kategori`), THE Bagian_Sublabel SHALL memperbarui Verified_Label dan menulis setiap keputusan ke Audit_Log dengan kolom `path`, `split`, `round`, `sub_label_sebelum`, `sub_label_sesudah`, `asal_pilihan`, `waktu`.
5. THE Bagian_Sublabel SHALL menghitung precision per Sub_Label hanya dari baris `asal_pilihan='random'`.
6. WHEN setiap Sub_Label non-Kelas_Kecil mencapai precision random ≥ 0,90 dengan minimal 20 sampel random dan jumlah Fallback_Sublabel berubah kurang dari 5% dibanding AL_Round sebelumnya, THE Bagian_Sublabel SHALL menyatakan active learning selesai.
7. WHILE active learning belum selesai dan jumlah AL_Round kurang dari `AL_MAX_ROUNDS` (default 3), THE Bagian_Sublabel SHALL mengizinkan AL_Round berikutnya.
8. IF `AL_MAX_ROUNDS` tercapai sebelum kriteria selesai terpenuhi, THEN THE Bagian_Sublabel SHALL melaporkan Sub_Label yang belum memenuhi kriteria beserta precision terakhirnya.
9. THE Bagian_Sublabel SHALL melatih Probe_AL hanya dari Verified_Label, sehingga Pseudo_Label tidak menjadi target training Probe_AL.
10. WHEN Bagian_Sublabel dijalankan ulang, THE Bagian_Sublabel SHALL membangun ulang Verified_Label dari Audit_Log sehingga hasilnya sama dengan sebelum dijalankan ulang (idempoten).
11. THE Bagian_Sublabel SHALL menambah baris Audit_Log tanpa menghapus atau menimpa baris lama.

### Requirement 10: Verifikasi Eval_Set_Terverifikasi untuk val dan test

**User Story:** Sebagai peneliti, saya ingin metrik val/test sub-head hanya dihitung dari label yang diverifikasi manusia, supaya metrik tidak sirkular terhadap pseudo-label.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL membentuk Review_Queue `eval_val.csv` dan `eval_test.csv` berisi sampel acak bertingkat per Sub_Label hasil terakhir sebanyak `min(EVAL_N_PER_SUBLABEL, jumlah)` (default 50), ditambah seluruh anggota Kelas_Kecil di split tersebut.
2. THE Bagian_Sublabel SHALL memakai Gold_Set val/test yang lolos Spot_Check sebagai bagian dari Eval_Set_Terverifikasi tanpa review ulang.
3. THE Bagian_Sublabel SHALL menghitung semua metrik val dan test sub-label hanya dari Eval_Set_Terverifikasi.
4. THE Bagian_Sublabel SHALL memakai label val dan test hanya untuk evaluasi, tidak untuk training Probe_AL atau Sub_Head.
5. WHEN metrik per Sub_Label dilaporkan, THE Bagian_Sublabel SHALL menampilkan jumlah sampel evaluasi (`n_eval`) di samping setiap metrik.
6. IF `n_eval` suatu Sub_Label kurang dari `MIN_EVAL_SUBLABEL` (default 20), THEN THE Bagian_Sublabel SHALL menandai metrik Sub_Label tersebut `tidak_reliabel` dan mengecualikannya dari macro F1 utama.

### Requirement 11: Propagasi label dan file output

**User Story:** Sebagai peneliti, saya ingin label final tersimpan dalam format tetap dan diturunkan konsisten ke view augmentasi, supaya training sub-head dan aplikasi memakai data yang sama.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL menetapkan Sub_Label hanya pada View_0 per (`split`, `orig_idx`).
2. WHEN data training sub-head disusun, THE Bagian_Sublabel SHALL menurunkan Sub_Label View_0 ke seluruh view augmentasi dengan `orig_idx` yang sama.
3. FOR ALL `orig_idx` di train, seluruh view SHALL memiliki Sub_Label yang sama dengan View_0 (invarian propagasi).
4. THE Bagian_Sublabel SHALL menentukan Sub_Label final per gambar dengan prioritas `manual` > `filename_gold` > `cluster` > `probe` > `zeroshot` > `filename_prior` > `filename_sisa`.
5. THE Sublabel_Table SHALL berisi kolom `path`, `orig_idx`, `split`, `kategori`, `sub_label`, `sumber`, `confidence`, `margin`, `flag`, `source_group`, dan `terverifikasi`.
6. THE Sublabel_Table SHALL hanya berisi nilai `sub_label` yang terdaftar di Taxonomy_Config (Sub_Label atau Fallback_Sublabel) dengan Kategori yang sesuai (integritas referensial).
7. WHEN Sublabel_Table ditulis lalu dibaca kembali, THE Bagian_Sublabel SHALL menghasilkan isi yang sama, termasuk tipe kolom `orig_idx`, `confidence`, `margin`, dan `terverifikasi` (round-trip).
8. THE Bagian_Sublabel SHALL menulis Sublabel_Table, Taxonomy_Config, Cek_Kategori_Table, Audit_Log, `di_luar_manifest.csv`, `lintas_split_sumber.csv`, dan `kalibrasi_tau.csv` ke Output_Dir.
9. WHEN Sublabel_Table ditulis, THE Bagian_Sublabel SHALL menampilkan ringkasan jumlah per Kategori × Sub_Label × split × `sumber`, jumlah Fallback_Sublabel, dan jumlah flag.

### Requirement 12: Kebijakan Kelas_Kecil

**User Story:** Sebagai peneliti, saya ingin sub-label yang sampelnya terlalu sedikit ditangani secara eksplisit, supaya sub-head tidak melaporkan metrik yang menyesatkan.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL menandai Sub_Label sebagai Kelas_Kecil bila jumlah Verified_Label train kurang dari `MIN_TRAIN_SUBHEAD` (default 30) atau jumlah Verified_Label val kurang dari `MIN_EVAL_SUBLABEL` (default 20).
2. WHERE kebijakan Kelas_Kecil suatu Sub_Label adalah `rule_only`, THE Bagian_Sublabel SHALL mengecualikan Sub_Label tersebut dari kelas output Sub_Head dan menetapkannya hanya lewat Filename_Heuristic atau Verified_Label.
3. WHERE kebijakan Kelas_Kecil suatu Sub_Label adalah `merge:<Sub_Label_target>`, THE Bagian_Sublabel SHALL menggabungkan Sub_Label tersebut ke Sub_Label target dalam Kategori yang sama untuk training Sub_Head dan tetap menyimpan Sub_Label asli di Sublabel_Table.
4. WHEN Kelas_Kecil terdeteksi, THE Bagian_Sublabel SHALL memberi flag `kelas_kecil` pada baris terkait dan mencetak Sub_Label, jumlah train, jumlah val, dan kebijakan yang diterapkan.
5. IF kebijakan `merge` menunjuk Sub_Label di Kategori lain atau Sub_Label yang tidak ada, THEN THE Bagian_Sublabel SHALL menghentikan eksekusi dengan pesan yang menyebut Sub_Label asal dan target.

### Requirement 13: Training Sub_Head dengan memakai ulang pipeline Notebook

**User Story:** Sebagai peneliti, saya ingin melatih sub-head dengan fungsi training yang sama seperti head 3-kelas, supaya hasil sebanding dan kode tidak diduplikasi.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL melatih Sub_Head per Kategori yang memiliki minimal 2 kelas output, memakai `build_head`, `train_one_epoch_head`, normalisasi input `standardize` dengan statistik train Kategori tersebut, early stopping pada val loss dengan `EARLY_STOP_PATIENCE`, dan history JSON per epoch.
2. WHERE sebuah Kategori hanya memiliki satu kelas output Sub_Head, THE Bagian_Sublabel SHALL menetapkan Sub_Label tersebut langsung tanpa melatih Sub_Head.
3. THE Bagian_Sublabel SHALL melatih Sub_Head hanya dari baris train dengan Sub_Label final non-fallback, dengan Verified_Label dan Pseudo_Label ber-Margin ≥ Tau_Constrained sebagai target, dan mencatat proporsi Verified_Label vs Pseudo_Label per kelas di `run_config`.
4. THE Bagian_Sublabel SHALL menghitung class weight Sub_Head dengan rumus `N / (C × N_c)` dari jumlah baris train (termasuk view augmentasi) per kelas output.
5. WHILE Sub_Head dilatih, THE Bagian_Sublabel SHALL mencatat precision, recall, dan F1 per Sub_Label setiap epoch pada Eval_Set_Terverifikasi val.
6. WHEN fungsi Notebook yang terikat 3 kelas dipakai untuk Sub_Head, THE Bagian_Sublabel SHALL memanggilnya dengan parameter jumlah kelas, nama kelas, dan tensor data eksplisit, dan pemanggilan dengan nilai default SHALL menghasilkan perilaku 3-kelas yang sama seperti sebelum perubahan.
7. THE Bagian_Sublabel SHALL menyimpan history, checkpoint, dan `run_config` Sub_Head ke `ROOT_DIR/head_runs/<SUB_RUN_NAME>/` dengan `SUB_RUN_NAME` yang memuat TAXONOMY_VERSION.
8. IF folder `head_runs/<SUB_RUN_NAME>/` sudah berisi history, THEN THE Bagian_Sublabel SHALL mencetak peringatan agar `SUB_RUN_NAME` diganti sebelum hasil lama tertimpa.
9. WHERE resampling atau cross-validation dipakai pada Sub_Head, THE Bagian_Sublabel SHALL memakai `StratifiedGroupKFold` dengan group `orig_idx` dan memverifikasi tidak ada `orig_idx` yang muncul di dua fold.
10. THE Bagian_Sublabel SHALL mencatat hasil Sub_Head di Sub_Registry yang terpisah dari registry `results` 3-kelas.
11. THE Bagian_Sublabel SHALL melatih Flat_Head untuk seluruh kelas output Sub_Label sebagai pembanding dengan konfigurasi training yang sama.
12. THE Bagian_Sublabel SHALL membandingkan minimal head `linear`, `mlp`, dan kNN cosine untuk setiap Sub_Head.

### Requirement 14: Evaluasi, seleksi model, dan prioritas B3

**User Story:** Sebagai peneliti, saya ingin memilih sistem sub-label dari metrik val dan menyentuh test sekali, supaya hasil akhir sahih dan recall B3 tetap terjaga.

#### Acceptance Criteria

1. THE Bagian_Sublabel SHALL mengevaluasi Hierarchical_Classifier dalam dua mode di val: Sub_Head dengan Kategori asli (oracle) dan end-to-end dengan Kategori dari head 3-kelas terpilih.
2. THE Bagian_Sublabel SHALL menampilkan tabel perbandingan val berisi macro F1 Sub_Label, precision/recall/F1 per Sub_Label dengan `n_eval`, recall dan precision setiap Sub_Label B3, recall Kategori B3, dan kolom `_nodup` yang mengecualikan gambar ber-flag `lintas_split_sumber` atau near-duplicate.
3. THE Bagian_Sublabel SHALL mengurutkan tabel perbandingan menurut macro F1 Sub_Label val, dengan recall minimum Sub_Label B3 sebagai pengurut kedua.
4. IF recall Kategori B3 dari Flat_Head (Kategori diturunkan dari Sub_Label prediksi) lebih rendah dari recall B3 head 3-kelas terpilih di val, THEN THE Bagian_Sublabel SHALL memberi tanda `B3_turun` pada baris Flat_Head tersebut.
5. THE Bagian_Sublabel SHALL memilih model sub-label final berdasarkan metrik val saja.
6. WHEN model sub-label final dievaluasi di test, THE Bagian_Sublabel SHALL memanggil `mark_test_use` dan menampilkan peringatan bila test sudah dipakai untuk model sub-label lain.
7. THE Bagian_Sublabel SHALL menyimpan tabel perbandingan sebagai `perbandingan_subhead_<SUB_RUN_NAME>.csv` di folder run.
8. WHEN evaluasi test selesai, THE Bagian_Sublabel SHALL menampilkan confusion matrix Sub_Label per Kategori dan metrik per Sub_Label dengan `n_eval`.

### Requirement 15: Penempatan dan eksekusi di Notebook

**User Story:** Sebagai peneliti, saya ingin pipeline sub-label menjadi bagian notebook training yang bisa dijalankan bertahap, supaya alur Colab yang sudah ada tetap dipakai.

#### Acceptance Criteria

1. THE Notebook SHALL berisi Bagian_Sublabel sebagai cell-cell baru dengan judul markdown berbahasa Indonesia, ditempatkan setelah Cell 21 sebagai bagian terpisah.
2. THE Bagian_Sublabel SHALL dapat dijalankan setelah Cell 1–12 tanpa menjalankan Cell 12b–21.
3. THE Bagian_Sublabel SHALL membagi pipeline ke tahap-tahap terpisah (konfigurasi, label awal + Gold_Set, clustering, zero-shot, kalibrasi, active learning, eval set, output, Sub_Head, evaluasi) sehingga setiap tahap dapat dijalankan ulang tanpa menjalankan ulang tahap sebelumnya, selama artefak tahap sebelumnya ada di Output_Dir.
4. IF artefak yang dibutuhkan suatu tahap tidak ada di Output_Dir, THEN THE Bagian_Sublabel SHALL menghentikan tahap tersebut dengan pesan yang menyebut file yang hilang dan tahap yang harus dijalankan.
5. WHILE sebuah Review_Queue belum memiliki keputusan Reviewer, THE Bagian_Sublabel SHALL melewati tahap yang bergantung padanya dan mencetak instruksi pengisian Review_Queue.
6. THE Bagian_Sublabel SHALL menjalankan cell validasi berbasis `assert` untuk invarian Requirement 2.4, 3.8, 7.4, 7.5, 11.3, 11.6, dan 11.7 setelah tahap terkait.
7. THE Bagian_Sublabel SHALL berjalan tanpa memuat ulang vision tower SigLIP, kecuali untuk uji kesetaraan pada Requirement 6.7.
8. THE Bagian_Sublabel SHALL menghasilkan seluruh output tahap 0–4 dari Embedding_Store dalam waktu kurang dari 10 menit di runtime GPU Colab, di luar waktu review manual dan pengunduhan model.

### Requirement 16: Dokumentasi

**User Story:** Sebagai anggota tim, saya ingin perubahan pipeline dan taksonomi tercatat, supaya anggota lain dan laporan memakai definisi yang sama.

#### Acceptance Criteria

1. WHEN Bagian_Sublabel diimplementasikan, THE Tim_Proyek SHALL memperbarui `documentation.md` dengan alur tahap sub-label, lokasi Output_Dir, arti kolom Sublabel_Table, dan hasil evaluasi terakhir.
2. WHEN Bagian_Sublabel diimplementasikan, THE Tim_Proyek SHALL memperbarui steering `solusi-auto-labeling-siglip.md` dengan taksonomi final 13 Sub_Label, `EMBEDDING_DIR` v3clean, dan status implementasi.
3. WHEN TAXONOMY_VERSION berubah, THE Tim_Proyek SHALL mencatat versi baru, alasan perubahan, dan perbedaan Sub_Label di `documentation.md`.
4. THE Notebook SHALL menjelaskan setiap tahap Bagian_Sublabel dalam cell markdown berbahasa Indonesia yang menyebut input, output, dan keputusan manual yang dibutuhkan.
