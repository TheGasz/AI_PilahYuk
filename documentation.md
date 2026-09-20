# Dokumentasi Proyek: Klasifikasi Sampah dengan SigLIP

Dokumen ini berisi rekam jejak pekerjaan, alur pemrosesan data, serta spesifikasi pelatihan model yang telah dilakukan di dalam *workspace* ini. Fokus utama dari tahap pengerjaan ini adalah melakukan pra-pemrosesan data secara lokal (menghindari proses *upload* data kotor yang berat ke *cloud*) dan menyusun *pipeline* pelatihan model visi komputer berkinerja tinggi menggunakan *backbone* SigLIP di Google Colab.

---

## 1. Inisialisasi Proyek & *Version Control*
- Menginisialisasi repositori Git lokal di dalam *workspace* proyek (`d:\UGM\Semester 5\Senpro\Project\Senpro`).
- Membuat dan menggunakan *branch* `Data_Preprocessing` untuk memisahkan pekerjaan pembersihan data dan pembuatan *script* dari struktur utama (*main*).
- Menjalankan inspeksi awal pada folder `Dataset` yang memiliki tiga kelas: `0_Recylable`, `1_Organik`, dan `2_B3`.
- Menganalisis berkas hasil Exploratory Data Analysis (EDA) sebelumnya yang berada di folder `EDA/` (`eda_kualitas_gambar.csv`, `eda_properti_teknis.csv`, dll.) untuk memahami masalah yang ada (seperti duplikat identik, resolusi rendah, dan ketidakseimbangan kelas).

---

## 2. Pra-pemrosesan Data Lokal (`preprocessing_lokal.ipynb`)
Mengingat dataset berada di mesin lokal dengan ukuran yang cukup besar, tahap pembersihan (*cleaning*) dan persiapan (*preparation*) dilakukan sepenuhnya di lokal melalui *notebook* `preprocessing_lokal.ipynb`. 

Tahapan yang dijalankan di dalam *script* ini meliputi:
1. **Deduplikasi Gambar**: 
   - Melakukan deteksi gambar ganda menggunakan dua metode *hashing*: MD5 untuk byte mentah file dan SHA256 untuk pixel RGB.
   - Menyimpan gambar dengan resolusi tertinggi sebagai perwakilan (representatif) grup duplikat.
   - Memindahkan sisa file duplikat ke dalam folder `Dataset/duplicates_removed` agar data mentah tidak hilang namun tidak ikut dilatih.
   - Menyimpan log duplikasi di `dedup_log.csv`.
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
   - Mendefinisikan resep augmentasi (terutama augmentasi yang lebih agresif untuk kelas minoritas `B3` guna menghindari *overfitting*). Resep ini disimpan di `augmentasi_config.json` sebagai rujukan untuk DataLoader.

**Status**: Tahap ini *telah berhasil dieksekusi* oleh pengguna. Folder keluaran `preprocessing_output` telah terbuat beserta isi-isinya.

---

## 3. Persiapan Pelatihan SigLIP di Colab (`siglip_training_colab.ipynb`)
Sebagai tahap lanjutan, *notebook* `siglip_training_colab.ipynb` dikembangkan untuk mengeksekusi pelatihan model (*training*) yang sesungguhnya. *Notebook* ini disiapkan dengan anggapan bahwa data bersih dan folder hasil pra-pemrosesan telah diunggah ke Google Drive.

Fitur utama pada *pipeline* pelatihan ini meliputi:
1. **Arsitektur Model**:
   - Menggunakan *checkpoint* `google/siglip-so400m-patch14-384`.
   - Hanya mengambil *vision encoder* (membuang bagian teks) dan menempelkan *classification head* berupa Multi-Layer Perceptron (MLP) berlapis tunggal (Linear → GELU → Dropout → Linear).
   - Opsi pembekuan tulang punggung (*freeze backbone*) sehingga hanya melatih beban di *classification head* (menghemat VRAM dan jauh lebih stabil pada data latih yang tidak terlalu besar).
2. **Pipeline Dataloader Khusus SigLIP**:
   - Pembuatan kelas `WasteDataset` di PyTorch yang memuat gambar (*on-the-fly*).
   - Augmentasi PIL diterapkan di DataLoader (hanya pada bagian latih / *train*), yang dirancang untuk memperkaya keragaman gambar secara langsung di memori.
   - Data dilempar ke `SiglipImageProcessor` untuk dipotong/diperbesar (resize) menjadi *384x384* dan dinormalisasi menjadi skala `[-1, 1]`, menghindari pemakaian normalisasi ImageNet tradisional.
3. **Optimasi Berbasis Performa & Memori (GPU Keterbatasan Colab)**:
   - Menggunakan optimasi **Mixed Precision Training (AMP)** via `torch.cuda.amp.autocast` dan `GradScaler`. AMP mengurangi penggunaan VRAM yang sangat krusial saat bekerja dengan SigLIP berparameter besar (`so400m`) di Colab GPU (T4).
   - Mendukung penggunaan dua variasi fungsi *Loss*: Cross-Entropy standar yang berbobot (*weighted*) atau Focal Loss (untuk *imbalance* yang parah).
4. **Mekanisme Anti-Kehilangan Data & Overfitting**:
   - **Early Stopping**: Pengecekan otomatis performa. Menghentikan siklus *epoch* bila nilai *loss* validasi berhenti menurun selama sekian *epoch*.
   - **Checkpointing Periodik**: Menyimpan *state* model, pengoptimal (*optimizer*), penjadwal (*scheduler*), dan `GradScaler` ke Google Drive per N *epoch*.
   - **Sinkronisasi History**: Menulis semua kurva metrik (*train/val loss/acc*) ke dalam *file* JSON (`training_history.json`) di GDrive sehingga riwayat kurva dapat dilanjutkan saat *session disconnect*.
5. **Pemantauan Kualitas Terperinci (*Granular Tracking*)**:
   - Pengecekan otomatis *sanity check* secara visual dengan mem-plot perbandingan gambar sebelum dan sesudah augmentasi untuk memastikan transformasinya wajar.
   - Melacak performa khusus setiap kelas (F1, Precision, Recall) dari dalam siklus per *epoch* dengan fokus pada parameter `Recall B3`.
6. **Ekstraksi Fitur/Embedding Lanjutan**:
   - Modul `extract_embeddings` disediakan di akhir *notebook* yang akan melewatkan setiap dataset pada *vision encoder* untuk mendapatkan vektor `1152-dimensi`.
   - Vektor-vektor ini disimpan menjadi wujud array bertipe `.npz` (misal: `embeddings_train.npz`). Ini ditujukan sebagai aset analisis tingkat lanjut (seperti pengelompokan K-Means, visualisasi kluster memakai algoritma UMAP, ataupun klasifikasi metode hibrida).

---

## 4. Penutup & Status Pengerjaan Terkini
Kedua file implementasi kode di atas (`preprocessing_lokal.ipynb` dan `siglip_training_colab.ipynb`) telah selesai diuji sintaksnya (*linted and validated*) lalu di-commit pada sistem manajemen versi (Git).

- **Total File Notebook Dihasilkan**: 2
- **Posisi Branch**: `Data_Preprocessing`
- **Tugas Selanjutnya**: 
  1. *Upload* hasil dari `preprocessing_output` ke *cloud storage* (Google Drive).
  2. Mulai sesi eksekusi di Google Colab.
