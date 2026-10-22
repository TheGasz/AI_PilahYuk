# Implementation Plan: Sub-Label Auto-Labeling (`sublabel-auto-labeling`)

## Overview

Rencana ini menambahkan **Bagian C (Bagian_Sublabel)** ke `siglip_embedding_probe_colab.ipynb`, ditempatkan **setelah Cell 21** (cell terakhir saat ini, indeks 50 di JSON notebook). Semua kode ditulis dalam **Python** (sesuai design), berupa cell baru yang dipanggil lewat *entry point* per tahap (`run_s1a()`, `run_s3c()`, dst.) dan hanya membaca file di `Output_Dir` (`preprocessing_output/sublabel_<TAXONOMY_VERSION>/`).

Urutan tahap mengikuti design: S0 (konfigurasi + helper) → S1 (label awal, Gold_Set) → S2 (clustering) → S3 (zero-shot, kalibrasi τ) → S4 (active learning, eval set) → S5 (output + Kelas_Kecil) → S6 (Sub_Head, Flat_Head, evaluasi) → dokumentasi.

### Aturan mekanika edit notebook (berlaku untuk semua task bertanda "cell")

- `.ipynb` adalah JSON besar. **Tambahkan cell di akhir** (setelah indeks 50) dengan edit terarah (mis. skrip `nbformat` sekali pakai yang memuat notebook, `nb.cells.append(...)`, lalu menyimpan). **Jangan menulis ulang seluruh file** dan jangan mengubah urutan atau isi Cell 1-21, kecuali task 2.3 yang memang mengubah signature fungsi di Cell 11-12 (dan satu fungsi di Cell 16) secara *backward-compatible* (argumen opsional, default = perilaku lama).
- Buat cadangan `siglip_embedding_probe_colab.ipynb.bak` sebelum edit pertama (jangan di-commit; hapus setelah semua task selesai bila sudah tidak diperlukan).
- Setiap tahap diawali satu **cell markdown Bahasa Indonesia** yang menyebut *input*, *output*, dan *keputusan manual yang dibutuhkan* (Req 16.4), lalu satu atau lebih cell kode kecil yang hanya mendefinisikan fungsi dan memanggil *entry point*.
- Fungsi murni `sl_*` khusus tahap diletakkan di cell tahap terkait (bukan semuanya di S0b) agar tidak ada kode menggantung; nama dan signature tetap sama dengan tabel helper di design.
- Penulisan file hanya lewat `sl_atomic_write_csv` / `sl_write_*` yang memanggil `sl_guard_write`. **Dilarang menulis ke `Dataset/` atau `Archive/`** (Req 2.6).
- **Jangan menjalankan training atau pipeline panjang dari shell.** Cell dijalankan manusia secara interaktif di Colab/Jupyter. Verifikasi oleh agent terbatas pada: `ast.parse` semua source cell baru, dan menjalankan fungsi murni dengan data sintetis kecil di mesin lokal bila memungkinkan (tanpa GPU, tanpa model SigLIP).
- Task bertanda **[MANUSIA]** dikerjakan user. Agent tidak boleh mengisi kolom `keputusan` Review_Queue, tidak boleh mengarang teks `tindakan_konsumen_pra_buang` / `rute_penyaluran_pengolahan`, dan tidak boleh menjalankan sel yang butuh GPU.

## Tasks

- [ ] 1. [MANUSIA] Checkpoint persiapan sebelum implementasi
  - Tempel teks **verbatim** `tindakan_konsumen_pra_buang` dan `rute_penyaluran_pengolahan` untuk 13 Sub_Label (plus teks `lainnya` per Kategori bila ada) dari tabel taksonomi final ke satu file/pesan yang bisa dipakai agent untuk mengisi `TAXONOMY_SPEC`. Bila belum tersedia, agent tetap membuat `TAXONOMY_SPEC` dengan placeholder `<<ISI DARI USER>>`; `build_taxonomy()` akan **sengaja gagal validasi** sampai teks diisi (Req 1.2, 1.6).
  - Konfirmasi nilai awal `TAU_DEFAULT=0.10` dan `TAU_GLOBAL_DEFAULT=0.50`, serta keputusan D4 (hasil zero-shot `lainnya` boleh menimpa `filename_prior`, sesuai Req 3.7 dan 11.4).
  - Pastikan Cell 1-12 notebook dapat dijalankan di Colab (Drive ter-mount, `EMBEDDING_DIR` v3clean ada) dan buat cadangan notebook.
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 1.2, 1.6, 3.7, 11.4, 15.2_

- [ ] 2. S0: konfigurasi taksonomi, helper bersama, parameterisasi fungsi 3-kelas
  - [x] 2.1 Tambah cell markdown judul Bagian C + cell **S0a** (taksonomi dan parameter)
    - Cell markdown judul "Bagian C: Auto-Labeling Sub-Label" (setelah Cell 21, Bahasa Indonesia) berisi ringkasan alur S0-S6, prasyarat (Cell 1-12 saja, bukan 12b-21), dan lokasi `Output_Dir`.
    - Cell S0a: definisikan `TAXONOMY_VERSION='v2'`, `SL_DIR`, `ARCHIVE_DIR`, `SUB_RUN_NAME`, `SUB_HEAD_DIR`, dict `SL_PARAMS` (semua parameter di design §A), `SUMBER_PRIORITAS`, `FLAG_TOKENS`, dan dict `TAXONOMY_SPEC` berisi 13 Sub_Label + 3 entri `lainnya` (kategori, `metode_identifikasi_utama`, `jumlah_sampel_user`, `penampung_sisa`, prompt draf v2, `filename_patterns` dari tabel Filename_Heuristic, `kelas_kecil_policy`: `Minyak_Jelantah=rule_only`, `Sampah_Kebun=merge:Sisa_Makanan`, `Anorganik_Lainnya=merge:Plastik_Kemasan`, `urutan_heuristik` per Kategori).
    - Implementasikan `validate_taxonomy(cfg)`, `taxonomy_hash(cfg)`, `prompt_hash(cfg)`, `sl_write_taxonomy`, `sl_read_taxonomy` (JSON `ensure_ascii=False`), `class SublabelError(RuntimeError)`, dan `build_taxonomy()`: panggil validasi, hitung `content_hash`, tulis `taxonomy.json`; bila file ada dengan hash berbeda dan versi sama, berhenti dengan pesan "naikkan `TAXONOMY_VERSION`" tanpa menimpa; bila hash sama pertahankan field τ yang sudah ada.
    - Teks tindakan/rute **hanya** diisi dari input user pada task 1; jangan dikarang. Verifikasi: `ast.parse`; cell SV kecil yang memanggil `build_taxonomy()` dan menampilkan jumlah Sub_Label (13) per Kategori.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 12.5_

  - [x] 2.2 Tambah cell **S0b**: helper I/O, penjaga tulis, gerbang, audit, shim
    - Definisikan: `sl_guard_write` (izinkan hanya di bawah `SL_DIR`/`SUB_HEAD_DIR`; tolak `DATASET_DIR`/`ARCHIVE_DIR`, normalisasi `..`), `sl_atomic_write_csv` (tulis `*.tmp` lalu `os.replace`), `sl_write_csv_reviewsafe` (bila file lama punya `keputusan` terisi, tulis ke `*_baru.csv`), `sl_require(files, hint)`, `sl_gate_queue(path, nama)`, `sl_read_csv(path, schema)` (`dtype` eksplisit, `keep_default_na=False`), `sl_path_key`, `sl_source_group`, `write_sublabel_table` / `read_sublabel_table`, `sl_audit_append`, `sl_sample_queue` (urut `sorted(path)` sebelum sampling, seed tetap), `sl_grid` (thumbnail dari `Dataset/`, hanya baca; sel gagal diganti kotak "N/A").
    - Definisikan `sl_std_l2` dan `sl_train_stats()` (cache `cache/train_stats_view0.npz`), `sl_load_view0(split)`, serta shim ber-guard `if 'top_neighbors' not in globals():` (salinan identik dari Cell 12b) dan `knn_probs` ber-guard dengan argumen `num_classes=None`. Jangan memodifikasi Cell 12b/16 untuk ini.
    - Verifikasi: `ast.parse`; panggil `sl_guard_write` pada contoh path di dalam dan di luar `SL_DIR` (assert).
    - _Requirements: 2.6, 2.7, 4.2, 4.6, 4.8, 5.1, 9.9, 9.11, 11.7, 15.4, 15.5_

  - [x] 2.3 Parameterisasi fungsi 3-kelas di Cell 11-12 (dan `knn_probs` di Cell 16) secara backward-compatible + cell **SV-regresi**
    - Edit terarah (bukan tulis ulang) pada source cell terkait: `make_criterion(weight=None)`, `metrics_from_preds(..., num_classes=None, idx_to_label=None)`, `evaluate_head(..., num_classes=None, idx_to_label=None)`, `head_paths(run_name, head_dir=None)`, `save_head(model, path, head_type, ctx=None)`, `train_head(head_type, run_name, seed=None, verbose=True, ctx=None)` dengan `_default_ctx()` dibangun saat dipanggil, `register_result` / `register_neural` (`registry=None, y_val=None, num_classes=None, idx_to_label=None, X_val=None, X_test=None`), `mark_test_use(name, tracker=None)`, dan `knn_probs(..., num_classes=None)`.
    - Nilai default `None` harus menghasilkan perilaku 3-kelas yang sama persis; `history` untuk ctx default harus superset key lama (`val_recall_B3`, `val_precision_B3`, `val_f1_B3`, dst.). Siapkan `SUB_REGISTRY = {}` dan `_SUB_TEST_REPORTED = []` di S0b sebagai registry/tracker terpisah.
    - Cell **SV-regresi**: jalankan `train_head('mlp', RUN_NAME+'__regresi', verbose=False)` dengan `HEAD_EPOCHS` sementara 3 dan seed tetap, bandingkan `train_loss`/`val_loss` 3 epoch pertama dengan `history_<RUN_NAME>.json` lama (toleransi 1e-3), dan assert `metrics_from_preds(y, p) == metrics_from_preds(y, p, 3, IDX_TO_LABEL)`. Cell ini dijalankan manusia; agent hanya memastikan sintaks.
    - Pastikan Cell 13-21 tidak perlu diubah.
    - _Requirements: 13.6, 13.10, 15.2, 15.6_

  - [x] 2.4 Tambah cell scaffold **SV-PBT** (property-based test di dalam notebook)
    - Cell berisi `RUN_PBT = False` (flag), `%pip install -q hypothesis` yang hanya jalan bila `RUN_PBT`, `from hypothesis import given, settings, strategies as st`, profil `@settings(max_examples=100, deadline=None)`, dan konvensi komentar tag `# Feature: sublabel-auto-labeling, Property <n>: <judul>`.
    - Cell-cell test berikutnya (task bertanda `*`) ditambahkan di bawah scaffold ini; tiap test dibungkus `if RUN_PBT:` agar menjalankan notebook tanpa hypothesis tetap aman. `hypothesis` tidak ditambahkan ke manifest paket (repo tidak punya).
    - _Requirements: 15.6_

  - [x] 2.5 Tulis property test untuk round-trip Taxonomy_Config
    - **Property 1: Round-trip Taxonomy_Config**
    - Generator: taksonomi valid acak (nama unik `[A-Za-z_]+`, prompt unicode, teks berbaris-baru, τ `null`). Uji `sl_write_taxonomy` lalu `sl_read_taxonomy` identik.
    - **Validates: Requirements 1.5, 1.2**

  - [x] 2.6 Tulis property test untuk validasi taksonomi
    - **Property 2: Validasi menolak taksonomi tidak valid**
    - Mutasi tepat satu cara (nama duplikat, Kategori tak dikenal, < 2 prompt, teks kosong/placeholder, target `merge` salah); assert `SublabelError` dan pesan memuat nama Sub_Label dan field.
    - **Validates: Requirements 1.6, 12.5**

  - [x] 2.7 Tulis property test untuk hash kanonik
    - **Property 3: Hash kanonik**
    - `prompt_hash`/`taxonomy_hash` tidak berubah saat urutan kunci diacak dan berubah saat satu prompt/pola/nama diubah.
    - **Validates: Requirements 1.8, 6.5**

  - [x] 2.8 Tulis property test untuk normalisasi kunci path
    - **Property 4: Normalisasi kunci path**
    - Root Windows (`\`) dan Linux (`/`), separator campuran; `sl_path_key` menghasilkan `folder/file` tanpa `\`.
    - **Validates: Requirements 2.2, 2.7**

  - [x] 2.9 Tulis property test untuk penjaga penulisan
    - **Property 5: Penjaga penulisan**
    - `sl_guard_write` mengizinkan jika dan hanya jika path (setelah normalisasi `..`) di bawah `SL_DIR`/`SUB_HEAD_DIR`; `DATASET_DIR`/`ARCHIVE_DIR` selalu ditolak. Gunakan direktori sementara.
    - **Validates: Requirements 2.6**

  - [x] 2.10 Tulis property test untuk penulisan Review_Queue yang aman
    - **Property 11: Penulisan Review_Queue tidak menimpa keputusan**
    - File lama dengan minimal satu `keputusan` terisi: hasil baru ke `*_baru.csv`, byte file lama tidak berubah.
    - **Validates: Requirements 4.8**

  - [ ] 2.11 Tulis property test untuk Audit_Log append-only
    - **Property 20: Audit_Log append-only**
    - Log lama menjadi awalan log baru; keputusan baru yang berbeda menambah tepat satu baris dengan kolom wajib terisi; keputusan identik tidak menambah baris.
    - **Validates: Requirements 9.4, 9.11**

  - [ ] 2.12 Tulis property test untuk gerbang prasyarat dan antrian
    - **Property 37: Gerbang prasyarat dan antrian**
    - `sl_require` melempar `SublabelError` (pesan memuat file hilang dan tahap) jika dan hanya jika ada file hilang; `sl_gate_queue` mengembalikan `None` tanpa menulis berkas jika ada `keputusan` kosong.
    - **Validates: Requirements 15.4, 15.5**

  - [ ] 2.13 Tulis property test untuk kompatibilitas mundur metrik dan isolasi registry
    - **Property 32: Kompatibilitas mundur dan isolasi registry**
    - `metrics_from_preds(y, p)` sama dengan versi eksplisit `num_classes=3` dan dengan referensi `precision_recall_fscore_support`; `C` kelas generik: `macro_f1` = rata-rata F1 per kelas; `register_result(..., registry=SUB_REGISTRY)` tidak mengubah `results`.
    - **Validates: Requirements 13.6, 13.10**

- [ ] 3. Checkpoint S0
  - **[MANUSIA]** Jalankan Cell 1-12, lalu cell Bagian C (judul, S0a, S0b, SV-regresi). Pastikan `taxonomy.json` terbentuk (butuh teks verbatim dari task 1), SV-regresi lulus, dan (opsional) set `RUN_PBT=True` untuk menjalankan test S0.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. S1: pemetaan manifest-npz, label awal, Source_Group, Spot_Check, Gold_Set
  - [ ] 4.1 Tambah cell markdown S1 + cell **S1a** (`run_s1a`) dan cell **SV-S1a**
    - `run_s1a()`: untuk tiap split, `load_manifest(...)` (Cell 5), `assert DEBUG_MAX_PER_CLASS is None`, `load_embeddings(split)`, ambil baris `view_idx==0`, assert `orig_idx` unik dan sama dengan `range(len(df))`, assert `sl_path_key(npz_path) == sl_path_key(rel_path)` untuk semua baris (berhenti dengan `SublabelError` memuat 5 selisih pertama), pastikan `kategori` sama dengan `label_names`.
    - Tulis `index_view0.csv` (`split, orig_idx, npz_row, path, kategori, source_group`) dan `di_luar_manifest.csv` (scan `os.scandir` folder Kategori `Dataset/`, lewati `duplicates_removed/`, hanya baca; diharapkan 169 baris di `2_B3`). Path memakai `/`.
    - Cell SV-S1a: assert 28.965 baris dan (`split`,`orig_idx`) unik.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 15.3, 15.6_

  - [ ] 4.2 Tulis property test untuk `di_luar_manifest`
    - **Property 6: `di_luar_manifest` adalah selisih himpunan**
    - Dua himpunan Rel_Path acak: hasil = isi folder dikurangi Manifest; tidak ada elemen hasil di Sublabel_Table.
    - **Validates: Requirements 2.5**

  - [ ] 4.3 Tambah cell **S1b** (`sl_classify_filename`, `run_s1b`) dan cell **SV-S1b**
    - `sl_classify_filename(kategori, filename, tax)`: `re.search` pada nama file mengikuti `urutan_heuristik`, pencocokan pertama menang; kembalikan `(sub_label, sumber, rule_id)` dengan `sumber` ∈ `filename_gold` / `filename_prior` / `filename_sisa`; tanpa penampung → `lainnya` + flag `review`. Pola dibaca dari `taxonomy.json` (tidak ada nama Sub_Label hard-coded di kode).
    - `run_s1b()`: tulis `label_awal.csv`, `ringkasan_label_awal.csv` (bandingkan dengan `jumlah_sampel_user`, cetak peringatan bila selisih ≠ 0, tidak menghentikan), `lintas_split_sumber.csv`, flag `lintas_split_sumber` untuk val/test yang grupnya ada di train, dan flag `near_dup` (pasangan val→train, test→train, test→val via `top_neighbors(k=1)` pada `sl_std_l2` View_0, cos ≥ `NEAR_DUP_COS`). Simpan `cache/train_stats_view0.npz`.
    - Cell SV-S1b: assert menjalankan klasifikasi dua kali identik (determinisme) dan selisih jumlah 13 Sub_Label terhadap user = 0.
    - _Requirements: 1.7, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.6, 4.7, 15.6_

  - [ ] 4.4 Tulis property test untuk klasifikasi nama file
    - **Property 7: Klasifikasi nama file (total, deterministik, konsisten sumber)**
    - Taksonomi acak (1-8 Sub_Label/Kategori, 0-1 penampung) dan nama file acak; hasil selalu satu Sub_Label milik Kategori, identik dua kali, `sumber` sesuai jenis pola, tanpa penampung → `lainnya` + `review`.
    - **Validates: Requirements 1.7, 3.1, 3.2, 3.3, 3.6, 3.8**

  - [ ] 4.5 Tulis property test untuk Source_Group dan flag lintas split
    - **Property 8: Source_Group dan flag lintas split**
    - Varian `<dasar>.rf.<hash>.<ext>` bergrup sama; nama tanpa `.rf.` grup sendiri; grup tercatat lintas split jika dan hanya jika anggotanya di > 1 split; flag val/test jika dan hanya jika grup juga ada di train.
    - **Validates: Requirements 4.6, 4.7**

  - [ ] 4.6 Tambah cell **S1c** (`run_s1c`): antrian Spot_Check
    - Untuk tiap Sub_Label bersumber `filename_gold`, `sl_sample_queue(pool, min(SPOTCHECK_N, n), rng=default_rng(SEED))` dari semua split; tulis `spotcheck_gold.csv` (`no, path, split, orig_idx, kategori, sub_label_gold, source_group, keputusan`) lewat `sl_write_csv_reviewsafe`, dan grid bernomor `spotcheck_grid/<sub_label>.png` (`sl_grid`).
    - Markdown tahap menjelaskan nilai `keputusan` yang diizinkan: `ok`, `salah`, `lainnya`, `cek_kategori`, atau nama Sub_Label lain di Kategori yang sama.
    - _Requirements: 4.1, 4.2, 4.8_

  - [ ] 4.7 Tambah cell **S1d** (`sl_gold_status`, `run_s1d`)
    - `run_s1d()` memakai `sl_gate_queue('spotcheck_gold.csv')` (lewati dengan instruksi bila ada `keputusan` kosong); validasi nilai `keputusan` (di luar daftar → `SublabelError` menyebut baris dan nilai).
    - `sl_gold_status`: precision Spot_Check per Sub_Label = `n(ok)/n`; `< GOLD_MIN_PRECISION` → seluruh anggota `sublabel_diturunkan` (kembali `filename_prior`); keputusan ≠ `ok` → `ditolak` + flag `tolak_gold`; keputusan Sub_Label lain/`lainnya` menjadi keputusan manual.
    - Tulis `gold_status.csv`, `spotcheck_precision.csv`, dan tambahkan ke `audit_log.csv` (`tahap='spotcheck'`) via `sl_audit_append`.
    - _Requirements: 4.3, 4.4, 4.5, 9.4, 9.11, 15.5_

  - [ ] 4.8 Tulis property test untuk status Gold_Set
    - **Property 10: Status Gold_Set dari hasil Spot_Check**
    - Precision = `n(ok)/n(keputusan)`; `sublabel_diturunkan` jika dan hanya jika precision `< GOLD_MIN_PRECISION`; `ditolak` jika dan hanya jika keputusan bukan `ok`; gambar `ditolak` muncul di antrian AL berikutnya dengan `asal_pilihan='tolak_gold'`; setiap keputusan menambah satu baris Audit_Log.
    - **Validates: Requirements 4.3, 4.4, 4.5**

- [ ] 5. [MANUSIA] Checkpoint S1: Spot_Check Gold_Set
  - Jalankan S1a, S1b, S1c. Buka `spotcheck_gold.csv` dan grid, isi kolom `keputusan` untuk **semua** baris (agent tidak boleh mengisi). Perhatikan `Minyak_Jelantah` (3 file `2020-sti-oil-black...` kemungkinan oli mesin) dan kemungkinan salah kategori.
  - Jalankan S1d; periksa `spotcheck_precision.csv` dan peringatan selisih jumlah label awal.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. S2: clustering untuk meninjau taksonomi
  - [ ] 6.1 Tambah cell markdown S2 + cell **S2a** (`run_s2a`)
    - Per Kategori: View_0 train, `Z = sl_std_l2(E, mu, sd)`, `k = min(K_CLUSTER[kat], n // 20)`, `KMeans(n_clusters=k, n_init=3, random_state=SEED)`; hitung `dist_rank` per cluster; simpan grid 4x4 `clusters/<kat>_c<id>.png` (16 terdekat centroid).
    - Tulis `cluster_assign.csv` (`path, orig_idx, kategori, cluster_id, dist_rank`) dan `cluster_review.csv` (`kategori, cluster_id, ukuran, dist_sub_label_awal` sebagai teks `Nama:jumlah;...`, `grid_png`, `keputusan`) lewat `sl_write_csv_reviewsafe`. Hanya split train.
    - Markdown menjelaskan nilai `keputusan`: nama Sub_Label Kategori itu, `campuran`, atau `usul:<nama_baru>`.
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 15.3_

  - [ ] 6.2 Tambah cell **S2b** (`sl_select_seeds`, `run_s2b`)
    - `run_s2b()` memakai `sl_gate_queue('cluster_review.csv')`; validasi nilai `keputusan`. `sl_select_seeds(dist, cluster_decisions, top_n)`: hanya cluster ber-Sub_Label (bukan `campuran`/`usul:*`), hanya `dist_rank <= SEED_TOP_N`, hanya train.
    - Tulis `cluster_seeds.csv` (`path, orig_idx, kategori, sub_label, cluster_id, dist_rank`) dan `cluster_usulan.csv` (cetak daftar usulan + jumlah cluster pendukung; **tidak** mengubah `taxonomy.json`).
    - _Requirements: 5.4, 5.5, 5.6_

  - [ ] 6.3 Tulis property test untuk seed cluster
    - **Property 12: Seed cluster**
    - Titik, centroid, dan keputusan acak; seed hanya dari train dan cluster ber-nama Sub_Label, ≤ `SEED_TOP_N` per cluster, dan jarak seed ≤ jarak non-seed. Uji pada fungsi seleksi murni (jarak diberikan), bukan `KMeans`.
    - **Validates: Requirements 5.4, 5.6**

- [ ] 7. [MANUSIA] Checkpoint S2: tinjau cluster
  - Jalankan S2a (±1-2 menit per Kategori di CPU). Buka grid `clusters/*.png` dan `cluster_review.csv`, isi `keputusan` per cluster. Jalankan S2b; tinjau `cluster_usulan.csv` dan putuskan apakah taksonomi/prompt perlu direvisi (bila ya, naikkan `TAXONOMY_VERSION` dan ulang dari S0).
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. S3: zero-shot, uji kesetaraan, kalibrasi τ, cek kategori
  - [ ] 8.1 Tambah cell markdown S3 + cell **S3a** (`sl_proto`, `sl_scores`, `run_s3a`)
    - Cek checkpoint: `taxonomy.model_checkpoint == MODEL_CHECKPOINT` dan `os.path.basename(EMBEDDING_DIR)` diawali `MODEL_CHECKPOINT.split('/')[-1]`, selain itu `SublabelError` menyebut kedua nama.
    - Cache prototype `cache/text_prototypes_<prompt_hash>.npz` (`protos`, `sub_labels`, `logit_scale` sebagai nilai `exp`, `logit_bias`). Bila cache ada, **jangan** memuat `AutoModel`/`AutoProcessor`. Bila tidak: `AutoModel.from_pretrained(MODEL_CHECKPOINT, torch_dtype=torch.float16)`, tokenisasi `padding='max_length'`, prototype = `normalize(mean(normalize(text_emb)))`, simpan cache, lalu `del model; gc.collect(); torch.cuda.empty_cache()`.
    - Skor semua View_0: `a * normalize(E) @ protos.T + b`; simpan `zeroshot_scores.npz` (`scores float32 [N,13]`, `sub_labels`, `split`, `orig_idx`, `path`, `a`, `b`, `prompt_hash`) berurutan sama dengan `index_view0.csv`.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.8, 15.7_

  - [ ] 8.2 Tulis property test untuk prototype dan skor
    - **Property 13: Prototype teks dan skor gambar-teks**
    - `sl_proto` bernorma 1, invarian terhadap urutan prompt dan skala positif; `sl_scores` = `a * normalize(E) @ T.T + b` dan invarian terhadap skala positif `E`.
    - **Validates: Requirements 6.4, 6.6**

  - [ ] 8.3 Tambah cell **S3b** (`run_s3b`): uji kesetaraan 32 gambar
    - Ambil 32 View_0 train acak (seed tetap), encode ulang dengan `load_vision_encoder()` + `load_image` + `build_aug('val')` + `to_tensor_norm` (satu-satunya pemakaian vision tower di Bagian C), bandingkan dengan `zeroshot_scores.npz`; `max |selisih| <= 1e-2` atau `SublabelError` dengan hint fp16 vs fp32 (`USE_FP16=False`). Tulis `zeroshot_equiv_check.json`; bebaskan vision tower sesudahnya.
    - _Requirements: 6.7, 15.7_

  - [ ] 8.4 Tambah cell **S3c** (`sl_softmax_masked`, `sl_decide_constrained`, `sl_calibrate_tau`, `run_s3c`)
    - `sl_softmax_masked(scores, mask)` dan `sl_decide_constrained(scores, cat_cols, tau)`: softmax hanya atas kolom Kategori, `confidence=p1`, `margin=p1-p2`, `lainnya` bila `margin < tau`.
    - `sl_calibrate_tau`: himpunan kalibrasi = gambar train `status=='aktif'` per Kategori; `prec(tau)` = rata-rata makro per Sub_Label gold; grid 0.00-0.50 langkah 0.01; pilih τ terkecil dengan `prec >= TARGET_PRECISION`; bila tidak ada, τ dengan precision tertinggi + status `presisi_tidak_tercapai`; Gold_Set satu Sub_Label (Organik) → status `perlu_kalibrasi_ulang` + peringatan bias; catat `cakupan_gold`.
    - `run_s3c()`: bila `gold_status.csv` belum ada, τ tetap `TAU_DEFAULT` (`status='default'`, catatan "belum dikalibrasi"). Tulis `kalibrasi_tau.csv`, `zeroshot_gold_val.csv` (precision/recall/F1 per Sub_Label + confusion matrix Gold_Set **val**, `lainnya` sebagai prediksi tersendiri), dan perbarui field τ di `taxonomy.json` secara atomik. Gold_Set test tidak disentuh.
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ] 8.5 Tambah cell **S3d** (`sl_decide_global`, `run_s3d`) dan cell **SV-S3**
    - `sl_decide_global`: softmax 13 kolom, `k*` argmax, `margin_global = P[k*] - max_{k di Kategori gambar} P[k]`, flag `cek_kategori` bila Kategori(`k*`) ≠ Kategori gambar dan `margin_global >= tau_global`.
    - `run_s3d()`: tulis `zeroshot.csv` (`path, split, orig_idx, kategori, zs_sub_label, zs_confidence, zs_margin, zs_flag_review, glob_sub_label, glob_kategori, glob_margin, flag_cek_kategori`) dan `cek_kategori.csv` (urut: baris yang melibatkan B3 dulu, lalu `margin_global` menurun; kolom `sudah_di_cleaning_log` dari `cleaning_log.csv` pada `match_key` atau `tujuan`; `keputusan` valid: `tetap`, `pindah:<Kategori>`, `buang`). Notebook ini **tidak** mengubah Kategori; perbaikan dilakukan di `preprocessing_lokal.ipynb` lalu re-ekstraksi.
    - SV-S3: assert fallback tidak turun untuk 3 nilai τ naik, dan semua hasil constrained milik Kategorinya atau `lainnya`.
    - _Requirements: 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 15.6_

  - [ ] 8.6 Tulis property test untuk keputusan constrained
    - **Property 14: Keputusan constrained**
    - Probabilitas non-negatif, jumlah 1 di kolom Kategori dan 0 di luarnya; `margin` ∈ [0,1] = `p1-p2`; `lainnya` jika dan hanya jika `margin < tau`; invarian terhadap konstanta aditif pada skor.
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.5**

  - [ ] 8.7 Tulis property test untuk monotonik τ
    - **Property 15: Monotonik terhadap τ**
    - Untuk `tau1 <= tau2`, himpunan `lainnya` pada `tau1` ⊆ pada `tau2`.
    - **Validates: Requirements 7.4**

  - [ ] 8.8 Tulis property test untuk flag cek kategori global
    - **Property 16: Flag cek kategori global**
    - Flag jika dan hanya jika Kategori argmax global berbeda dan `margin_global >= tau_global`; `margin_global >= 0` bila Kategori argmax berbeda.
    - **Validates: Requirements 7.6, 7.7**

  - [ ] 8.9 Tulis property test untuk isi dan urutan Cek_Kategori_Table
    - **Property 17: Isi dan urutan Cek_Kategori_Table**
    - `sudah_di_cleaning_log` True jika dan hanya jika path ada di `match_key`/`tujuan`; semua baris yang melibatkan B3 di depan; di dalam blok `margin_global` menurun.
    - **Validates: Requirements 7.8, 7.9**

  - [ ] 8.10 Tulis property test untuk kalibrasi τ
    - **Property 18: Kalibrasi τ**
    - Hanya baris train yang dipakai; τ terkecil yang mencapai target; bila tidak ada → precision tertinggi (imbang → τ terkecil) + `presisi_tidak_tercapai`; Gold_Set satu Sub_Label → `perlu_kalibrasi_ulang`.
    - **Validates: Requirements 8.2, 8.3, 8.4, 8.7**

- [ ] 9. [MANUSIA] Checkpoint S3: jalankan zero-shot di Colab
  - Jalankan S3a (unduh model sekali, ±2 GB VRAM), S3b (harus lulus, selisih ≤ 1e-2), S3c, S3d. Periksa `kalibrasi_tau.csv`, `zeroshot_gold_val.csv`, status τ per Kategori, dan baris teratas `cek_kategori.csv`.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. S4a-b: komposisi label, probe, putaran active learning
  - [ ] 10.1 Tambah cell markdown S4 + cell **S4-core** (`sl_resolve`, `sl_propagate`, `sl_rebuild_verified`, `sl_compose`)
    - `sl_resolve(cands)`: pilih sumber paling awal di `SUMBER_PRIORITAS`. `sl_propagate(row_orig_idx, label_by_orig)`: label tiap baris view = label View_0 dengan `orig_idx` sama.
    - `sl_rebuild_verified(audit_df, gold_status, cluster_seeds)`: Verified_Label deterministik dari log (keputusan terakhir per `path`+`tahap`+`round` menang; `cek_kategori` mengeluarkan gambar dari semua himpunan Verified).
    - `sl_compose(round=None)`: baca `label_awal`, `gold_status`, `cluster_seeds`, `zeroshot`, `audit_log`, dan `probe_pred_round_<n>.csv` (probe dipakai hanya bila `probe_margin >= tau`); hasilkan Sublabel_Table 11 kolom dengan flag turunan (`review`, `beda_prior`, `cek_kategori`, `tolak_gold`, `kelas_kecil`, `lintas_split_sumber`, `near_dup`; urut alfabetis, dipisah `;`) dan `terverifikasi` True jika sumber ∈ {`manual`, `filename_gold`, `cluster`}.
    - _Requirements: 3.7, 9.10, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [ ] 10.2 Tulis property test untuk prioritas sumber
    - **Property 26: Prioritas sumber**
    - `sl_resolve` mengembalikan kandidat paling awal di `SUMBER_PRIORITAS`; menghapus kandidat berprioritas lebih rendah tidak mengubah hasil; `terverifikasi` jika dan hanya jika sumber ∈ {`manual`, `filename_gold`, `cluster`}.
    - **Validates: Requirements 3.7, 11.4**

  - [ ] 10.3 Tulis property test untuk propagasi label ke view
    - **Property 27: Invarian propagasi label ke view**
    - Semua baris dengan `orig_idx` sama berlabel identik dengan View_0.
    - **Validates: Requirements 11.2, 11.3**

  - [ ] 10.4 Tulis property test untuk Verified_Label idempoten
    - **Property 21: Verified_Label idempoten**
    - `sl_rebuild_verified` dua kali menghasilkan himpunan sama; menerapkan ulang Review_Queue yang sama tidak mengubah apa pun; keputusan terakhir menentukan; `cek_kategori` mengeluarkan dari Verified.
    - **Validates: Requirements 9.10, 9.4**

  - [ ] 10.5 Tulis property test untuk integritas Sublabel_Table
    - **Property 28: Integritas Sublabel_Table**
    - Tepat satu baris per (`split`,`orig_idx`); `sub_label` terdaftar dan sesuai `kategori` (atau `lainnya`); jumlah ringkasan Kategori × Sub_Label × split × `sumber` = jumlah baris.
    - **Validates: Requirements 2.4, 11.1, 11.6, 11.9**

  - [ ] 10.6 Tambah cell **S4a** (`sl_select_al`, `run_s4a(n)`)
    - Probe per Kategori hanya dari Verified train View_0 non-fallback; Kategori dengan < 2 kelas terverifikasi dilewati (dicetak). `PROBE_TYPE='logreg'` (`LogisticRegression(class_weight='balanced', max_iter=1000)` pada `sl_std_l2`) atau `'knn'` (`knn_probs(..., num_classes=C)`); opsi fitur skor zero-shot. Pseudo_Label tidak pernah jadi target training probe.
    - Prediksi seluruh View_0 → `probe_pred_round_<n>.csv` (`probe_sub_label, probe_conf, probe_margin`). Panggil `sl_compose(round=n)` untuk label sementara dan `fallback_count_n`.
    - `sl_select_al(pool, params, rng)`: pool = train belum Verified dan bukan `cek_kategori`; per (Kategori, Sub_Label sementara) `AL_N_UNCERTAIN` margin terendah (NaN paling akhir) + `AL_N_RANDOM` acak + per Kategori `AL_N_FALLBACK` dari `lainnya` + semua `tolak_gold`; tiap gambar hanya sekali (prioritas `tolak_gold` > `uncertain` > `random` > `fallback`).
    - Tulis `al_round_<n>.csv` (`no, path, split, orig_idx, kategori, sub_label_prediksi, sumber_prediksi, confidence, margin, asal_pilihan, tetangga_label, tetangga_path, keputusan`) dan grid `al_grid/r<n>_<kat>_<sub>_p<k>.png` ([gambar | 5 tetangga Verified terdekat]); gunakan `sl_write_csv_reviewsafe`.
    - _Requirements: 9.1, 9.2, 9.3, 9.9, 15.3_

  - [ ] 10.7 Tulis property test untuk sampling antrian
    - **Property 9: Sampling antrian**
    - `sl_sample_queue` mengembalikan `min(n, len(pool))` elemen unik dari pool, deterministik untuk seed sama meski urutan pool berbeda; bucket AL tidak melebihi ukuran, tidak tumpang tindih, hanya train belum Verified, dan `uncertain` berisi margin terendah.
    - **Validates: Requirements 4.2, 9.2, 10.1**

  - [ ] 10.8 Tambah cell **S4b** (`sl_al_precision`, `sl_al_done`, `run_s4b(n)`)
    - Gerbang `sl_gate_queue('al_round_<n>.csv')`; validasi `keputusan` ∈ {`ok`, nama Sub_Label Kategori itu, `lainnya`, `cek_kategori`}; tambahkan ke `audit_log.csv` (`tahap='al_round'`, `round=n`, kolom wajib Req 9.4).
    - Hitung `precision_random[k]` hanya dari baris `asal_pilihan=='random'` dengan `sub_label_sebelum==k`; tulis `al_status.json` (per putaran: `fallback_count`, `precision_random`, `n_random`, `selesai`, `belum_memenuhi`). `sl_al_done`: selesai bila ≥ 2 putaran, semua Sub_Label non-Kelas_Kecil `precision_random >= 0.90` dengan `n_random >= 20`, dan perubahan relatif `fallback_count < 5%`; bila `n == AL_MAX_ROUNDS` dan belum selesai, laporkan Sub_Label gagal + precision terakhir.
    - _Requirements: 9.4, 9.5, 9.6, 9.7, 9.8, 9.10, 9.11, 15.5_

  - [ ] 10.9 Tulis property test untuk precision AL
    - **Property 22: Precision AL hanya dari sampel random**
    - Menambah baris `uncertain`/`fallback`/`tolak_gold` tidak mengubah `precision_random[k]`.
    - **Validates: Requirements 9.5**

  - [ ] 10.10 Tulis property test untuk kriteria selesai AL
    - **Property 23: Kriteria selesai active learning**
    - "Selesai" jika dan hanya jika ≥ 2 putaran, semua Sub_Label non-Kelas_Kecil memenuhi `precision_random >= 0.90` dan `n_random >= 20`, perubahan fallback `< 0.05`; pada `AL_MAX_ROUNDS` daftar gagal sama persis dengan Sub_Label yang gagal.
    - **Validates: Requirements 9.6, 9.7, 9.8**

- [ ] 11. [MANUSIA] Checkpoint S4a-b: putaran active learning
  - Jalankan `run_s4a(1)`; buka `al_round_1.csv` dan grid, isi `keputusan` seluruh baris; jalankan `run_s4b(1)`. Ulangi `run_s4a(2)`, `run_s4b(2)`, dst. sampai `al_status.json` menyatakan selesai atau `AL_MAX_ROUNDS` tercapai (periksa Sub_Label yang belum memenuhi precision 0,90). Agent tidak mengisi keputusan.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. S4c-d: Eval_Set_Terverifikasi untuk val dan test
  - [ ] 12.1 Tambah cell **S4c** (`sl_build_eval_queue`, `run_s4c`)
    - Dari `sl_compose()` terakhir, per split (val, test) dan per strata (Kategori, Sub_Label termasuk `lainnya`): target `EVAL_N_PER_SUBLABEL` gambar terverifikasi per strata (hitung yang sudah terverifikasi dulu, mis. gold lolos Spot_Check; kekurangan diambil acak dari yang belum), ditambah semua anggota strata Kelas_Kecil (menurut jumlah anggota: `< MIN_EVAL_SUBLABEL` di split itu atau `< MIN_TRAIN_SUBHEAD` di train). Gold_Set aktif tidak masuk antrian.
    - Tulis `eval_val.csv` dan `eval_test.csv` (`no, path, split, orig_idx, kategori, sub_label_saat_ini, sumber, asal_pilihan` ∈ `strata`/`kelas_kecil`, `keputusan`). Val/test tidak pernah dipakai untuk training probe atau Sub_Head.
    - _Requirements: 10.1, 10.2, 10.4_

  - [ ] 12.2 Tambah cell **S4d** (`run_s4d`)
    - Gerbang `sl_gate_queue` untuk `eval_val.csv` dan `eval_test.csv` (masing-masing); validasi `keputusan` seperti S4b; tambahkan ke `audit_log.csv` (`tahap='eval_val'|'eval_test'`).
    - _Requirements: 9.4, 10.3, 10.4, 15.5_

  - [ ] 12.3 Tulis property test untuk komposisi antrian evaluasi
    - **Property 24: Komposisi antrian evaluasi**
    - Antrian hanya berisi split yang bersangkutan; per strata jumlah terverifikasi setelah review = `min(EVAL_N_PER_SUBLABEL, anggota)`; Gold_Set aktif tidak muncul; semua anggota strata Kelas_Kecil muncul.
    - **Validates: Requirements 10.1, 10.2**

- [ ] 13. [MANUSIA] Checkpoint S4c-d: review Eval_Set_Terverifikasi
  - Jalankan `run_s4c()`, isi `keputusan` pada `eval_val.csv` dan `eval_test.csv` (label test hanya untuk evaluasi akhir), lalu jalankan `run_s4d()`.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. S5: output sublabels.csv dan kebijakan Kelas_Kecil
  - [ ] 14.1 Tambah cell markdown S5 + cell **S5a** (`sl_kelas_kecil`, `run_s5a`)
    - `sl_kelas_kecil(n_train_ver, n_val_ver, cfg)`: Kelas_Kecil jika `n_train < MIN_TRAIN_SUBHEAD` atau `n_val < MIN_EVAL_SUBLABEL`; terapkan kebijakan (`rule_only`, `merge:<target>`, `latih`); cetak Sub_Label, jumlah train, jumlah val, dan kebijakan.
    - `run_s5a()`: `sl_compose()` → `sublabels.csv` (11 kolom), `ringkasan_sublabel.csv` (Kategori × Sub_Label × split × `sumber`, jumlah `lainnya`, jumlah per flag; ditampilkan), `kelas_kecil.csv`, `output_manifest.json` (7 file wajib + jumlah baris: `sublabels.csv`, `taxonomy.json`, `cek_kategori.csv`, `audit_log.csv`, `di_luar_manifest.csv`, `lintas_split_sumber.csv`, `kalibrasi_tau.csv`). Tulis lewat `write_sublabel_table`.
    - _Requirements: 11.5, 11.8, 11.9, 12.1, 12.2, 12.3, 12.4_

  - [ ] 14.2 Tambah cell **S5b** (`validate_outputs`) dan cell **SV-S5**
    - Assert: kunci (`split`,`orig_idx`) unik dan jumlah = `index_view0.csv`; `sub_label` terdaftar dan sesuai `kategori`; semua view satu `orig_idx` berlabel sama (via `sl_propagate` pada `.npz` train); `write -> read` `sublabels.csv` identik; `output_manifest.json` mendaftar 7 file wajib.
    - _Requirements: 2.4, 11.3, 11.6, 11.7, 11.8, 15.6_

  - [ ] 14.3 Tulis property test untuk round-trip Sublabel_Table
    - **Property 29: Round-trip Sublabel_Table**
    - Tabel sintetis dengan `confidence`/`margin` NaN, `flag` kosong, `source_group` khusus, nama file `NA.jpg`; `write_sublabel_table` lalu `read_sublabel_table` identik dengan tipe `orig_idx` int64, `confidence`/`margin` float64, `terverifikasi` bool.
    - **Validates: Requirements 11.7**

- [ ] 15. [MANUSIA] Checkpoint S5
  - Jalankan S5a, S5b, SV-S5. Tinjau `ringkasan_sublabel.csv`, `kelas_kecil.csv`, jumlah `lainnya`, dan proporsi sumber terverifikasi vs pseudo sebelum melatih Sub_Head.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. S6: Sub_Head, Flat_Head, perbandingan val, evaluasi test sekali
  - [ ] 16.1 Tambah cell markdown S6 + cell **S6-core** (helper murni S6)
    - `sl_output_classes(scope, cfg)` (Sub_Label Kategori dikurangi `rule_only`, dikurangi yang di-`merge`, urut alfabetis; `flat` = gabungan 10 kelas), `sl_train_rows(tab, cfg, scope)` (train, non-fallback, bukan `cek_kategori`, dan (`terverifikasi` atau (`sumber ∈ {zeroshot, probe}` dan `margin >= tau`)); buang `rule_only`, petakan `merge`), `sl_class_weights(counts)` = `N / (C × N_c)`, `sl_assert_group_folds(folds, groups)`, `sl_macro_f1_reliable(y, p, names, min_n)` (hanya kelas `n_eval >= MIN_EVAL_SUBLABEL`, flag `tidak_reliabel`).
    - _Requirements: 10.5, 10.6, 12.2, 12.3, 13.2, 13.3, 13.4, 13.9_

  - [ ] 16.2 Tulis property test untuk himpunan training Probe_AL dan Sub_Head
    - **Property 19: Himpunan training Probe_AL dan Sub_Head**
    - Probe_AL ⊆ train Verified non-fallback; Sub_Head ⊆ baris layak (aturan di atas) dan semua baris layak terpilih; tidak ada val/test; `rule_only` dibuang; `merge` memakai label target.
    - **Validates: Requirements 9.1, 9.9, 10.4, 12.2, 12.3, 13.3**

  - [ ] 16.3 Tulis property test untuk deteksi Kelas_Kecil dan kebijakan
    - **Property 30: Deteksi Kelas_Kecil dan kebijakan**
    - Kelas_Kecil jika dan hanya jika `n_train < MIN_TRAIN_SUBHEAD` atau `n_val < MIN_EVAL_SUBLABEL`; flag `kelas_kecil` terpasang; kelas output tidak memuat `rule_only` maupun yang di-`merge`; `merge` idempoten; label asli di Sublabel_Table tidak berubah.
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4**

  - [ ] 16.4 Tulis property test untuk class weight
    - **Property 31: Class weight**
    - `w_c = N / (C × N_c)` sehingga `w_c × N_c = N / C` dan `Σ w_c × N_c = N`.
    - **Validates: Requirements 13.4**

  - [ ] 16.5 Tulis property test untuk fold tanpa kebocoran group
    - **Property 33: Fold tanpa kebocoran group**
    - `StratifiedGroupKFold` dengan group `orig_idx` menghasilkan fold dengan `orig_idx` saling lepas; `sl_assert_group_folds` gagal bila satu `orig_idx` ada di dua fold.
    - **Validates: Requirements 13.9**

  - [ ] 16.6 Tulis property test untuk metrik hanya dari Eval_Set_Terverifikasi
    - **Property 25: Metrik hanya dari Eval_Set_Terverifikasi**
    - Mengacak label/prediksi baris tidak terverifikasi tidak mengubah metrik; `n_eval` benar; Sub_Label dengan `n_eval < MIN_EVAL_SUBLABEL` `tidak_reliabel` dan macro F1 utama hanya dari yang reliabel.
    - **Validates: Requirements 10.3, 10.5, 10.6**

  - [ ] 16.7 Tambah cell **S6a** (`run_s6a`): Sub_Head per Kategori
    - Per `scope` ∈ Recylable/Organik/B3: kelas output dari `sl_output_classes`; Kategori dengan satu kelas output ditetapkan langsung tanpa training. Muat `.npz` train penuh, `sl_propagate` label View_0 ke semua view, `sl_train_rows`, `norm = fit_input_norm(...)` (statistik train Kategori itu), `class_weight = sl_class_weights(bincount(y_tr))` dari baris termasuk view augmentasi. Val = View_0 val terverifikasi dengan label ∈ kelas output Kategori itu (batas val < 30 → peringatan).
    - Buat `ctx` (`X_tr, y_tr, X_va, y_va, in_dim, num_classes, idx_to_label, label_to_idx, class_weight, head_dir, norm`) dan panggil `train_head(ht, f'{SUB_RUN_NAME}__{scope}__{ht}', seed, verbose=False, ctx=ctx)` untuk `ht` ∈ `SUB_HEADS_TO_COMPARE` (`linear`, `mlp`, `mlp2`), plus kNN cosine (`knn_probs(..., num_classes=C)`). `register_result(f'{scope}::{ht}', ..., registry=SUB_REGISTRY, y_val=..., num_classes=C, idx_to_label=...)`; simpan probabilitas **seluruh val** (untuk mode `end_to_end`), **tanpa** probabilitas test.
    - Cek `SUB_HEAD_DIR` sudah berisi `history_*` → peringatan ganti `SUB_RUN_NAME` sebelum `os.makedirs`. Tulis `run_config_<SUB_RUN_NAME>__<scope>.json` (proporsi Verified vs Pseudo per kelas, class weight, jumlah baris). Jangan memicu training dari shell; cell ini dijalankan manusia di Colab GPU.
    - _Requirements: 10.4, 13.1, 13.2, 13.3, 13.4, 13.5, 13.7, 13.8, 13.9, 13.10, 13.12_

  - [ ] 16.8 Tambah cell **S6b** (`run_s6b`): Flat_Head 10 kelas
    - `scope='flat'` dengan konfigurasi dan jenis head identik dengan S6a; norm dan class weight dari seluruh baris train layak; daftarkan ke `SUB_REGISTRY` (`flat::<h>`) tanpa menghitung probabilitas test.
    - _Requirements: 13.5, 13.10, 13.11, 13.12_

  - [ ] 16.9 Tambah cell **S6c** (`sl_route_hier`, `sl_sort_comparison`, `sl_b3_turun`, `run_s6c`)
    - Tiga mode di val: `hier_oracle::<h>`, `hier_e2e::<h>` (Kategori dari head 3-kelas terpilih; sumber: `results[FINAL_MODEL]`, cache `kategori_pred_val_<SUB_RUN_NAME>.csv`, atau `load_head(...)`; cetak sumber yang dipakai), `flat::<h>`; `<h>` ∈ `SUB_HEADS_TO_COMPARE` ∪ {`knn`} ∪ {`best_per_kategori`}.
    - Tabel kolom: `model, mode, head, macro_F1_sub, n_eval, n_sublabel_dinilai, minR_B3sub, R_/P_/n_ untuk 4 Sub_Label B3, recall_kat_B3, prec_kat_B3, macro_F1_sub_nodup, minR_B3sub_nodup, B3_turun, detik, catatan`; urut `macro_F1_sub` menurun lalu `minR_B3sub` menurun; `B3_turun` pada Flat_Head bila `recall_kat_B3` lebih rendah dari recall B3 head 3-kelas terpilih. Mask `_nodup` = bukan `lintas_split_sumber` dan bukan `near_dup`.
    - Simpan `perbandingan_subhead_<SUB_RUN_NAME>.csv` dan `perbandingan_subhead_per_sublabel_<SUB_RUN_NAME>.csv` di `head_runs/<SUB_RUN_NAME>/`. Pemilihan model dari val saja.
    - _Requirements: 10.5, 10.6, 14.1, 14.2, 14.3, 14.4, 14.5, 14.7_

  - [ ] 16.10 Tulis property test untuk perutean hierarkis
    - **Property 34: Perutean hierarkis**
    - Bila Kategori prediksi = Kategori sebenarnya untuk semua gambar, `end_to_end` identik dengan `oracle`; bila kelas sebenarnya tidak ada di Sub_Head Kategori prediksi, dihitung salah.
    - **Validates: Requirements 14.1**

  - [ ] 16.11 Tulis property test untuk urutan tabel dan `B3_turun`
    - **Property 35: Urutan tabel perbandingan dan tanda `B3_turun`**
    - Tabel terurut menurun menurut `macro_F1_sub` lalu `minR_B3sub`; `B3_turun` jika dan hanya jika `recall_kat_B3` Flat_Head `<` recall B3 head 3-kelas terpilih.
    - **Validates: Requirements 14.3, 14.4**

  - [ ] 16.12 Tambah cell **S6d** (`run_s6d(name)`): evaluasi test sekali
    - `run_s6d(name)` hanya dijalankan untuk model pilihan manusia: panggil `mark_test_use(name, tracker=_SUB_TEST_REPORTED)` lebih dulu (peringatan bila ada model sub-label lain yang sudah memakai test), lalu hitung metrik pada Eval_Set_Terverifikasi test (`eval_test.csv` + gold test lolos). Tahap dilewati dengan instruksi bila `eval_test.csv` belum lengkap.
    - Keluaran: classification report per Sub_Label dengan `n_eval`, confusion matrix Sub_Label per Kategori, dan matriks 10x10 gabungan.
    - _Requirements: 8.7, 10.3, 14.5, 14.6, 14.8, 15.5_

  - [ ] 16.13 Tulis property test untuk penjaga pemakaian test
    - **Property 36: Penjaga pemakaian test**
    - `mark_test_use(name, tracker=_SUB_TEST_REPORTED)` memperingatkan jika dan hanya jika tracker sub-label sudah berisi nama lain; isi `_TEST_REPORTED` (3-kelas) tidak memengaruhi.
    - **Validates: Requirements 14.6**

- [ ] 17. [MANUSIA] Checkpoint S6: pilih model dari val, evaluasi test sekali
  - Jalankan S6a, S6b, S6c (Colab GPU). Pilih model sub-label final dari tabel val (perhatikan `minR_B3sub`, kolom `_nodup`, `n_eval`, flag `tidak_reliabel`, dan `B3_turun`). Jalankan `run_s6d(<nama_model>)` **satu kali**. Ganti `SUB_RUN_NAME` untuk setiap eksperimen baru.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 18. Dokumentasi
  - [ ] 18.1 Perbarui `documentation.md`
    - Tambahkan bagian Bahasa Indonesia: alur tahap S0-S6, lokasi `Output_Dir` dan `head_runs/<SUB_RUN_NAME>/`, arti 11 kolom Sublabel_Table, nilai `keputusan` Review_Queue, catatan `TAXONOMY_VERSION` (versi, alasan, perbedaan Sub_Label), dan hasil evaluasi terakhir **hanya dari angka yang benar-benar dihasilkan user** (jangan mengarang; beri placeholder "menunggu hasil" bila belum ada). Catat perubahan signature fungsi Cell 11-12/16.
    - _Requirements: 16.1, 16.3_

  - [ ] 18.2 Perbarui steering `.kiro/steering/solusi-auto-labeling-siglip.md`
    - Ganti taksonomi draft v1 dengan taksonomi final 13 Sub_Label, ubah `EMBEDDING_DIR` menjadi `siglip-so400m-patch14-384_naug_2_2_5_v3clean`, perbarui status dari "rencana" menjadi status implementasi dan checklist "Langkah selanjutnya" sesuai tahap yang benar-benar selesai.
    - _Requirements: 16.2_

- [ ] 19. Final checkpoint
  - **[MANUSIA]** Restart kernel → jalankan Cell 1-12 → Bagian C tanpa Cell 12b-21 untuk memastikan Bagian C mandiri; jalankan Cell 13-21 sekali untuk memastikan tidak ada regresi 3-kelas; set `RUN_PBT=True` dan jalankan seluruh SV-PBT; pastikan `output_manifest.json` mendaftar 7 file wajib dan hanya `run_s6d` yang menyentuh test.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks bertanda `*` bersifat opsional (property-based test dengan `hypothesis` di dalam notebook) dan dapat dilewati untuk MVP; test-nya tetap harus ditulis sebagai cell di bawah scaffold SV-PBT (task 2.4) bila dikerjakan.
- Task **[MANUSIA]** dan checkpoint tidak masuk grafik dependensi; semua keputusan Review_Queue dan teks verbatim berasal dari user.
- Property 9 diuji bersama pemilihan AL (task 10.7); Property 19 dan 30 diuji setelah helper S6 tersedia (task 16.2, 16.3).
- Setiap tugas notebook menulis ke file `.ipynb` yang sama, sehingga seluruh task notebook ditempatkan pada wave berbeda; hanya task dokumentasi (file berbeda) berbagi satu wave.
- Property-based test hanya untuk fungsi murni; kualitas cluster, training head, dan render grid diverifikasi lewat cell SV `assert` dan pengecekan manual (lihat Testing Strategy di design).
- Cell 1-21 tidak berubah perilaku; satu-satunya pengecualian adalah argumen opsional pada task 2.3.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1"] },
    { "id": 1, "tasks": ["2.2"] },
    { "id": 2, "tasks": ["2.3"] },
    { "id": 3, "tasks": ["2.4"] },
    { "id": 4, "tasks": ["2.5"] },
    { "id": 5, "tasks": ["2.6"] },
    { "id": 6, "tasks": ["2.7"] },
    { "id": 7, "tasks": ["2.8"] },
    { "id": 8, "tasks": ["2.9"] },
    { "id": 9, "tasks": ["2.10"] },
    { "id": 10, "tasks": ["2.11"] },
    { "id": 11, "tasks": ["2.12"] },
    { "id": 12, "tasks": ["2.13"] },
    { "id": 13, "tasks": ["4.1"] },
    { "id": 14, "tasks": ["4.2"] },
    { "id": 15, "tasks": ["4.3"] },
    { "id": 16, "tasks": ["4.4"] },
    { "id": 17, "tasks": ["4.5"] },
    { "id": 18, "tasks": ["4.6"] },
    { "id": 19, "tasks": ["4.7"] },
    { "id": 20, "tasks": ["4.8"] },
    { "id": 21, "tasks": ["6.1"] },
    { "id": 22, "tasks": ["6.2"] },
    { "id": 23, "tasks": ["6.3"] },
    { "id": 24, "tasks": ["8.1"] },
    { "id": 25, "tasks": ["8.2"] },
    { "id": 26, "tasks": ["8.3"] },
    { "id": 27, "tasks": ["8.4"] },
    { "id": 28, "tasks": ["8.5"] },
    { "id": 29, "tasks": ["8.6"] },
    { "id": 30, "tasks": ["8.7"] },
    { "id": 31, "tasks": ["8.8"] },
    { "id": 32, "tasks": ["8.9"] },
    { "id": 33, "tasks": ["8.10"] },
    { "id": 34, "tasks": ["10.1"] },
    { "id": 35, "tasks": ["10.2"] },
    { "id": 36, "tasks": ["10.3"] },
    { "id": 37, "tasks": ["10.4"] },
    { "id": 38, "tasks": ["10.5"] },
    { "id": 39, "tasks": ["10.6"] },
    { "id": 40, "tasks": ["10.7"] },
    { "id": 41, "tasks": ["10.8"] },
    { "id": 42, "tasks": ["10.9"] },
    { "id": 43, "tasks": ["10.10"] },
    { "id": 44, "tasks": ["12.1"] },
    { "id": 45, "tasks": ["12.2"] },
    { "id": 46, "tasks": ["12.3"] },
    { "id": 47, "tasks": ["14.1"] },
    { "id": 48, "tasks": ["14.2"] },
    { "id": 49, "tasks": ["14.3"] },
    { "id": 50, "tasks": ["16.1"] },
    { "id": 51, "tasks": ["16.2"] },
    { "id": 52, "tasks": ["16.3"] },
    { "id": 53, "tasks": ["16.4"] },
    { "id": 54, "tasks": ["16.5"] },
    { "id": 55, "tasks": ["16.6"] },
    { "id": 56, "tasks": ["16.7"] },
    { "id": 57, "tasks": ["16.8"] },
    { "id": 58, "tasks": ["16.9"] },
    { "id": 59, "tasks": ["16.10"] },
    { "id": 60, "tasks": ["16.11"] },
    { "id": 61, "tasks": ["16.12"] },
    { "id": 62, "tasks": ["16.13"] },
    { "id": 63, "tasks": ["18.1", "18.2"] }
  ]
}
```
