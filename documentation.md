# Dokumentasi Proyek: Klasifikasi Sampah dengan SigLIP

Dokumen ini berisi rekam jejak pekerjaan, alur pemrosesan data, serta spesifikasi pelatihan model yang telah dilakukan di dalam *workspace* ini. Fokus utama dari tahap pengerjaan ini adalah melakukan pra-pemrosesan data secara lokal (menghindari proses *upload* data kotor yang berat ke *cloud*) dan menyusun *pipeline* pelatihan model visi komputer berkinerja tinggi menggunakan *backbone* SigLIP di Google Colab.

---

## 1. Inisialisasi Proyek & *Version Control*
- Menginisialisasi repositori Git lokal di dalam *workspace* proyek (`d:\UGM\Semester 5\Senpro\Project\Senpro`).
- Membuat dan menggunakan *branch* `Data_Preprocessing` untuk memisahkan pekerjaan pembersihan data dan pembuatan *script* dari struktur utama (*main*).
- Menjalankan inspeksi awal pada folder `Dataset` yang memiliki tiga kelas: `0_Recylable`, `1_Organik`, dan `2_B3`.
- Menganalisis berkas hasil Exploratory Data Analysis (EDA) sebelumnya yang berada di folder `EDA/` (`eda_kualitas_gambar.csv`, `eda_properti_teknis.csv`, `eda_duplikat_identik.csv`, dll.) untuk memahami karakteristik dataset. Pada tahap EDA ini, ditemukan ~1000 file duplikat identik, variasi format warna, resolusi gambar yang sangat bervariasi, serta masalah ketidakseimbangan (imbalance) antarkelas (terutama pada kelas B3).

---

## 2. Pra-pemrosesan Data Lokal (`preprocessing_lokal.ipynb`)
Mengingat dataset berada di mesin lokal dengan ukuran yang cukup besar, tahap pembersihan (*cleaning*) dan persiapan (*preparation*) dilakukan sepenuhnya di lokal melalui *notebook* `preprocessing_lokal.ipynb`. 

Tahapan yang dijalankan di dalam *script* ini meliputi:
1. **Deduplikasi Gambar**: 
   - Melakukan deteksi gambar ganda menggunakan dua metode *hashing*: MD5 untuk byte mentah file dan SHA256 untuk pixel RGB.
   - Menyimpan gambar dengan resolusi tertinggi sebagai perwakilan (representatif) grup duplikat.
   - Memindahkan secara fisik 515 file duplikat (hasil run lokal) ke dalam folder `Dataset/duplicates_removed` agar data mentah tidak hilang namun tidak ikut dilatih dan menghindari *data leakage*.
   - Menyimpan log duplikasi lengkap di `dedup_log.csv`.
2. **Filter Kualitas & Standardisasi Format**:
   - Membuang gambar yang rusak (*corrupt*) dan yang resolusinya terlalu kecil (di bawah 50x50 piksel).
   - Mengubah paksa (*convert in-place*) gambar dengan format non-RGB (seperti RGBA, Grayscale, atau CMYK) menjadi format RGB standar.
   - Mengekstraksi kecerahan rata-rata (*mean brightness*) untuk memberikan *flag* pada gambar yang terindikasi terlalu gelap atau terlalu terang (disimpan di `metadata_kualitas.csv`).
3. **Penanganan Ketidakseimbangan Kelas (*Class Imbalance*)**:
   - Menghitung bobot kelas (*class weights*) dengan rumus berbanding terbalik terhadap frekuensi kelas untuk digunakan di fungsi *loss* nantinya.
   - Menyimpan bobot tersebut ke dalam `class_weights.json`.
4. **Data Splitting (Train/Val/Test)**:
   - Melakukan pembagian dataset secara *stratified* (mempertahankan proporsi kelas) menjadi 70% Pelatihan (*Train*), 15% Validasi (*Val*), dan 15% Pengujian (*Test*).
   - Memastikan tidak ada data yang *overlap* antarfase.
   - Menghasilkan *file manifest* (`train.csv`, `val.csv`, `test.csv`) yang menggunakan *path* relatif (memakai *forward slash* `/`) sehingga portabel ketika diunggah ke *platform* Linux/Google Drive.
5. **Konfigurasi Augmentasi**: 
   - Mendefinisikan resep augmentasi (terutama augmentasi yang lebih agresif untuk kelas minoritas `B3` guna menghindari *overfitting*). Resep ini disimpan di `augmentasi_config.json` sebagai rujukan untuk DataLoader. Konsep utamanya adalah **augmentasi acak HANYA diterapkan pada data Train**, sedangkan **Val dan Test murni (tanpa augmentasi)** untuk mencegah *data leakage* dan menjaga keabsahan evaluasi metrik.

**Status**: Tahap ini *telah berhasil dieksekusi* oleh pengguna. Sebanyak 515 gambar duplikat telah dipindahkan, dan folder keluaran `preprocessing_output` telah terbuat beserta isi-isinya.

---

## 3. Pelatihan dengan Strategi *Embedding Probe* di Colab (`siglip_embedding_probe_colab.ipynb`)
Sebagai tahap lanjutan, *notebook* telah diperbarui ke versi `siglip_embedding_probe_colab.ipynb` yang menggunakan strategi **Embedding Probe (2 Tahap)**. Strategi ini jauh lebih cepat dan sangat hemat memori karena meniadakan pemanggilan *backbone* SigLIP secara berulang di setiap *epoch*. *Notebook* ini disiapkan dengan anggapan bahwa data bersih hasil pra-pemrosesan telah diunggah ke Google Drive.

Alur eksekusi pada *pipeline* ini dibagi menjadi 2 tahap utama:

**Tahap 1: Ekstraksi Embedding (Dijalankan Sekali)**
- Menggunakan *checkpoint* `google/siglip-so400m-patch14-384`.
- **Dataset Multi-View**: Tiap gambar dibaca sekali, kemudian diubah menjadi `1 + N_AUG` *view* (1 versi asli/bersih, ditambah sejumlah *view* teraugmentasi secara acak khusus untuk set *train*). Kelas `B3` mendapat porsi augmentasi yang lebih agresif.
- **Ekstraksi Fitur Lanjutan**: Melakukan inferensi seluruh gambar (dan versi augmentasinya) melalui *frozen vision encoder* SigLIP untuk menghasilkan vektor `1152-dimensi`.
- **Penyimpanan Lokal (.npz)**: Kumpulan vektor (*embedding*) ini langsung disimpan ke Google Drive sebagai *file* berformat `.npz` (`embeddings_train.npz`, `embeddings_val.npz`, `embeddings_test.npz`). Jika sesi Colab terputus, ekstraksi ini tidak perlu diulang dari awal.
- **Sanity Check**: Disediakan modul untuk mem-plot perbandingan gambar sebelum dan sesudah augmentasi secara visual sebelum ekstraksi berjalan, guna memastikan wajarnya transformasi.

**Tahap 2: Pelatihan *Classifier Head***
- **Bebas Komputasi Berat**: Tahap ini murni melatih jaringan *head* di atas sekumpulan vektor (tanpa perlu memanggil SigLIP maupun membaca *file* gambar secara manual lagi). Ini membuat waktu *training* dalam satu *epoch* selesai dalam hitungan detik.
- **Arsitektur *Head***: Dapat berupa MLP (Linear → GELU → Dropout → Linear) yang ringan.
- **Mekanisme Pelatihan Andal**: Tetap didukung oleh bobot kelas (*Class Weights*) untuk komputasi *loss*, *Early Stopping* berbasis *validation loss*, serta penyimpanan *checkpoint* model terbaik (berikut sinkronisasi riwayat *training* berformat JSON) ke Google Drive.
- **Pemantauan Kualitas Terperinci**: Melacak *Recall*, *Precision*, dan *F1-score* untuk masing-masing kelas secara berkesinambungan per *epoch* (dengan fokus prioritas pada *Recall B3*).
- **Fleksibilitas (*Swap Classifier*)**: Karena *embedding* sudah terekstrak di memori, di sesi akhir *notebook* diberikan implementasi instan bagaimana *head* berbasis *Neural Network* bisa ditukar/dibandingkan dengan metode *Machine Learning* klasik, misalnya menggunakan `LogisticRegression` berbobot seimbang (`class_weight='balanced'`) dari Scikit-Learn.

**Pembaruan v2 (`RUN_NAME = mlp_head_v2_ce_ls_mixup`)** – hasil audit skill `methodology-fault-fixer`, `model-performance-advisor`, dan `embedding-data-diagnostics`:
- **Class weight efektif** (`CLASS_WEIGHT_MODE='effective'`): rumus `N/(C*N_c)` dihitung dari jumlah *baris* train `.npz`. Sebelumnya bobot B3 (2.90, dari jumlah gambar) ditumpuk dengan oversampling `N_AUG` B3=5, sehingga total kontribusi loss B3 ~2x kelas lain. Kini B3 ≈ 1.6, Recylable ≈ 0.87, Organik ≈ 0.82 (perkiraan dari 70.599 baris).
- **Loss**: default CrossEntropy + *label smoothing* 0.05 (Focal opsional). Bug Focal Loss diperbaiki: `pt` kini dari CE tanpa bobot.
- **Mixup di ruang embedding** (`MIXUP_ALPHA=0.2`) untuk generalisasi.
- **Diagnostik data (Cell 12b/12c)**: prediksi OOF `StratifiedGroupKFold` (group `orig_idx`, standardisasi per fold), kandidat label noise (confident learning + kNN), near-duplicate lintas split, sampel ambigu, outlier centroid. CSV ditulis ke `ROOT/data_diagnostics/<embedding>/`; kolom `keputusan` diisi manual, lalu `TRAIN_EXCLUDE_CSV` dipakai untuk eksperimen cepat (hanya train). Perbaikan permanen tetap lewat `preprocessing_lokal.ipynb` + re-ekstrak embedding.
- **Evaluasi jujur**: kolom `macro_F1_nodup`/`recall_B3_nodup` (val tanpa near-duplicate train) dan macro F1 OOF sebagai estimasi generalisasi kedua.
- **Test sekali**: `mark_test_use()` memberi peringatan bila test dilihat untuk lebih dari satu model; error analysis dipindah dari test ke **val**.
- **SVM**: `C` dipilih dari `SVM_C_GRID` via val; ensemble baru `ens_lintas_model` (MLP multi-seed + kNN + SVM).
- Konfigurasi tiap run disimpan ke `run_config_<RUN_NAME>.json`.

**Analisis hasil v2 (dijalankan di data asli) & keputusan model final.**
- `model-performance-advisor` pada `history_*.json` v2: semua head sehat, early stopping bekerja, head rapat (<0.01) → headroom arsitektur habis, alihkan ke data + ensemble.
- Tabel `perbandingan_head_mlp_head_v2_ce_ls_mixup.csv`: pemenang macro F1 val `ens_seed_resmlp` (0.9915, recall B3 0.9942, prec B3 0.9885). Selisih `macro_F1` vs `macro_F1_nodup` hanya ~0.0001–0.0002 di semua model → near-duplicate val **tidak** menggelembungkan skor (metodologi jujur). Bobot B3 yang diturunkan (2.90→1.62) tidak menurunkan recall B3.
- Confusion OOF train & test sama-sama menunjukkan B3 nyaris sempurna (test: 516/520, 0 false alarm masuk B3); sisa error terkonsentrasi di **Recylable↔Organik**.
- **Keputusan model final: `ens_lintas_model`** (MLP multi-seed + kNN + SVM). Macro F1 val setara pemenang, tetapi precision B3 lebih tinggi (0.9923) dengan recall B3 sama (0.9942) → false alarm B3 lebih sedikit, dan ensembling lintas keluarga lebih stabil di data baru. Cell 19 notebook probe kini mengevaluasi test (sekali) untuk `ens_lintas_model` (`FINAL_MODEL`), dan error analysis (Cell 20) mengikuti model final ini.
- **Keputusan arah perbaikan: data cleaning**, bukan TTA. TTA ditolak karena embedding sudah di-fix (butuh re-ekstrak SigLIP, mahal) dan gain-nya kecil (error sisa = ambiguitas kelas, bukan noise inferensi). Jalur berdampak: bersihkan 9 kandidat label noise (2 di antaranya label asli B3) + rapikan pasangan Recylable↔Organik.

**Tahap 1.5 – Data Cleaning di `preprocessing_lokal.ipynb` (baru).** Disisipkan antara dedup (Tahap 1) dan filter kualitas (Tahap 2). Membaca CSV review manual (`CLEANING_CSVS`; kolom `path` + `keputusan`: `buang` / `relabel:<Kelas>` / `ok`) dan menerapkannya permanen ke disk: `buang` → dipindah ke `Dataset/duplicates_removed/` (tidak dihapus), `relabel:<Kelas>` → dipindah fisik ke folder kelas tujuan sehingga `collect_all_images` berikutnya memungutnya dengan label benar. Idempoten dan aman (CSV kosong/tidak ada → dilewati; file yang sudah hilang dicatat, bukan error). Log ke `cleaning_log.csv`. Setelah cleaning: re-run preprocessing (split baru dari data bersih) → re-ekstrak embedding ke `EMBEDDING_DIR` baru → bandingkan val (`macro_F1`, `macro_F1_nodup`, recall/prec B3) terhadap run v2.

**Perbaikan kebersihan metodologi (bukan penaik metrik).** Dari CSV diagnostik run v2 yang diunduh: `cross_split_near_duplicates.csv` berisi 119 pasang (konflik label 0; cosine sampai 1.0000 → ada duplikat identik yang lolos dedup pixel-hash, kemungkinan beda format `.png`/`.jpg`).
- **Tahap 1 dedup + perceptual hash (dHash).** Ditambahkan `compute_dhash` (difference hash 8x8) dan `hamming_hex`; `find_duplicates` kini punya jalur ketiga: kelompokkan dHash **dalam kelas yang sama** dan gabung yang berjarak Hamming ≤ `DHASH_MAX_HAMMING` (default 5). Menangkap duplikat beda-format yang lolos MD5/SHA256. Representatif tetap resolusi tertinggi; alasan log `dhash_match`.
- **Tahap 1.5 + split: pindah near-duplicate lintas split ke TRAIN.** `CROSS_SPLIT_DUP_CSV` dibaca di Cell split; setiap gambar sisi val/test yang kembar dengan train (plus kedua sisi pasangan val↔test) dipindah ke train, sehingga val/test bebas bayangan train tanpa membuang data. Dilewati bila CSV kosong/tidak ada.
- Catatan: 119/21k gambar hanya memperbaiki macro F1 secara marginal; tujuannya membuat skor jujur, bukan menaikkan metrik. Penaik metrik yang berarti dibahas sebagai rekomendasi lanjutan (lihat bagian berikut).

**P2 – Kalibrasi bias per kelas dari val (Cell 21 notebook probe, baru).** Menambahkan vektor bias `b=(b_R,b_O,b_B3)` ke log-probabilitas model final lalu argmax ulang; `b` dicari grid search murni di val (objektif macro F1 val, rambu recall B3 tak boleh turun), `b=0` selalu kandidat (hasil val tak mungkin lebih buruk). B3 dikunci 0 secara default (`TUNE_B3=False`) karena B3 sudah sangat baik. Test dievaluasi sekali dengan bias terpilih; bila `b*=0` test tak disentuh. Menyasar error simetris Recylable↔Organik tanpa re-ekstrak. Catatan: cell melaporkan base vs P2 di val DAN test, karena perbaikan val belum tentu terbawa ke test (risiko overfit ke val kecil pada val ~4.5k, tapi tetap harus dicek).

**Flow eksperimen (dua jalur terpisah).**
- Jalur A (P2, cepat, tanpa upload): pakai embedding v2 yang sudah di Drive. Colab: jalankan Cell 1–4 (mount, config, GPU, import) → Cell 10–12 (load npz, head, data) → Cell 16–19 (bandingkan head + pilih final) → Cell 21 (P2). Tidak menyentuh dataset.
- Jalur B (data cleaning, siklus penuh): review kandidat → isi `keputusan` → set `CLEANING_CSVS`, `CROSS_SPLIT_DUP_CSV`, `DHASH_MAX_HAMMING` di `preprocessing_lokal.ipynb` → run notebook preprocessing → upload ulang `Dataset/` + `preprocessing_output/` → ekstrak embedding ke folder BARU + `RUN_NAME` baru → Cell 16–19 → bandingkan val vs v2.

**Hasil P2 (dijalankan di run v2) & keputusan.** Bias terpilih (R,O,B3)=(−0.3, 0.05, 0): VAL macro F1 0.9913→0.9922 (+0.0009, dalam batas noise) tetapi **TEST turun** 0.9936→0.9928 (−0.0007) dengan precision B3 ikut turun (0.9961→0.9923). Confusion test menunjukkan P2 hanya *memindahkan* error R↔O (recall Organik naik, recall Recylable turun), bukan menguranginya → model sudah terkalibrasi, tidak ada perbaikan gratis dari threshold/bias. **Keputusan: pakai baseline `ens_lintas_model` tanpa bias; Cell 21 dinonaktifkan (`P2_ENABLED=False`) + diberi catatan hasil.** Keputusan diambil dari val + alasan struktural, bukan dari skor test.

**P1 – daftar review terarah R↔O (`build_ro_review_list.py`, skrip permanen di root).** Membaca `oof_predictions_train.csv` dan menulis dua CSV ke folder embedding: `review_RO_label_noise.csv` (142 kandidat: 60 Recylable→Organik + 82 Organik→Recylable, urut `skor_salah = p_pred − p_label` menurun; max 0.918) dan `review_RO_ambiguous.csv` (46 gambar perbatasan, margin p(R) vs p(O) ≤ 0.15). Kolom `keputusan` kosong untuk diisi manual (`buang`/`relabel:<Kelas>`/`ok`; kolom `saran_relabel` memberi usulan dari OOF). `match_key` sudah selaras dengan matcher Tahap 1.5. Jumlah 142 cocok dengan confusion OOF R↔O. Ini jalur penaik macro F1 yang tersisa setelah arsitektur (habis) dan kalibrasi (P2 gagal) buntu.

**Eksekusi data cleaning v3 (dijalankan di `Dataset/` lokal).** `apply_cleaning` diperluas mendukung keputusan `setuju` (memakai usulan kolom `saran_relabel`) selain `relabel:<Kelas>`/`buang`/`ok`. `CLEANING_CSVS` di-set ke `review_RO_label_noise_cleaned.csv` (142 baris; 66 keputusan aktif: 46 setuju + 17 buang + 3 relabel:B3, sisanya ok). `CROSS_SPLIT_DUP_CSV=None` (near-dup lintas split sengaja dilewati untuk mengisolasi efek cleaning). Hasil run:
- Dedup (MD5+SHA256+**dHash**): 1174 duplikat dipindah ke `duplicates_removed/` (vs 515 run lama; selisih = duplikat beda-format yang kini tertangkap dHash). 29.151 dipertahankan.
- Tahap 1.5: 49 relabel (29→Recylable, 17→Organik, 3→B3) + 17 buang; semua 66 aksi status `ok` (`cleaning_log.csv`).
- Filter kualitas: 28.965 ok, 0 corrupt.
- Split baru 70/15/15: Train 20.275 (B3 2226, Organik 9450, Recylable 8599), Val/Test masing-masing 4.345 (B3 477). `class_weights.json` baru: Recylable 0.7859, Organik 0.7152, B3 3.0362.

Manifest `train/val/test.csv` + `class_weights.json` + `cleaning_log.csv` siap. Langkah berikut: upload `Dataset/` (tanpa `duplicates_removed/`) + `preprocessing_output/` ke Drive, ekstrak embedding ke `EMBEDDING_DIR` baru (`RUN_NAME`/dir v3), lalu Cell 16–19 bandingkan vs v2. Cell 21 (P2) tetap dimatikan.

**Notebook probe kini bisa run LOKAL atau COLAB (satu variabel `ENV`).** Karena GPU Colab diakses dari lokal lewat tunnel ngrok, workspace pindah ke repo lokal (tak perlu upload/download Drive). Perubahan di `siglip_embedding_probe_colab.ipynb`:
- Cell 2 (mount): mount Drive hanya bila terdeteksi Colab (`import google.colab`); di lokal dilewati.
- Cell 2-config: variabel `ENV` (`'auto'|'local'|'colab'`). `auto` → Colab pakai `ROOT_COLAB` (Drive), selain itu pakai `ROOT_LOCAL` (path repo Windows). Cukup ganti `ENV` (atau `ROOT_LOCAL`) untuk beralih; sisa path (`DATASET_DIR`, `OUTPUT_DIR`, `EMBEDDING_DIR`, manifest) ikut otomatis.
- Cell 3 (pip): install hanya di Colab; di lokal diasumsikan environment sudah lengkap.
Diverifikasi di mode local: `ENV` auto → `local`, semua path resolve ke repo, manifest v3 terbaca (20275/4345/4345), file gambar ada di disk. Ekstraksi embedding v3 selanjutnya dijalankan langsung di repo lokal (GPU via ngrok), output `.npz` ke `embeddings/<model>_naug_2_2_5/` lokal.

**Run All aman untuk ekstraksi v3.** Dua pengaman ditambahkan di `siglip_embedding_probe_colab.ipynb` agar bisa Run All tanpa menimpa/mencampur hasil v2 dan tanpa menyentuh test dua kali:
- `DATA_VERSION='v3clean'` di Cell 2-config → masuk ke `EMBEDDING_DIR` (`..._naug_2_2_5_v3clean`) dan `RUN_NAME` (`mlp_head_v3clean_ce_ls_mixup`). Karena `N_AUG` tak berubah, tanpa suffix ini folder embedding akan sama dengan v2 dan logika `[SKIP] sudah ada` keliru melewati ekstraksi (melatih di data lama). Ganti `DATA_VERSION` tiap versi dataset baru; `''` = v2 lama.
- CELL 15 (eval test head tunggal) dibungkus guard `RUN_SINGLE_HEAD_TEST=False` → saat Run All dilewati, sehingga test hanya dievaluasi sekali di Cell 19 (model final). Set True hanya bila head tunggal memang final.
Diverifikasi di mode local: `EMBEDDING_DIR`/`RUN_NAME` resolve ke `v3clean` (tidak bentrok v2), CELL 15 no-op tanpa menyentuh test. Dengan ini Run All = ekstraksi v3 ke folder baru → latih head → bandingkan val, aman.

**Fix `ImportError: cannot import name '_Ink' from 'PIL._typing'`.** Colab terbaru (Nov 2025) mem-ship Pillow 12 yang menghapus symbol internal `_Ink` yang masih dipakai torchvision -> import gagal. Penyebabnya cell install lama memakai `pip install --upgrade pillow` (menarik Pillow 12). Perbaikan di Cell install `siglip_embedding_probe_colab.ipynb`: pin `pillow>=10,<12`, dan bila Pillow 12 terlanjur aktif di memori cell otomatis me-restart runtime sekali (lalu jalankan ulang dari Cell 1). Di lokal cell hanya memperingatkan bila Pillow>=12 terdeteksi. Perbaikan manual cepat bila perlu: `pip install "pillow>=10,<12"` lalu restart kernel.

---

## 4. Penutup & Status Pengerjaan Terkini
Kedua file implementasi kode di atas (`preprocessing_lokal.ipynb` dan `siglip_embedding_probe_colab.ipynb`) telah disiapkan untuk menjalankan seluruh siklus pemrosesan data dan pemodelan secara menyeluruh.

- **Total File Notebook Dihasilkan**: 2
- **Posisi Branch**: `Data_Preprocessing`
- **Tugas Selanjutnya**: 
  1. *Upload* hasil dari `preprocessing_output` ke *cloud storage* (Google Drive).
  2. Mulai sesi eksekusi di Google Colab.
