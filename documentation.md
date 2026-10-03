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

---

## 4. Penutup & Status Pengerjaan Terkini
Kedua file implementasi kode di atas (`preprocessing_lokal.ipynb` dan `siglip_embedding_probe_colab.ipynb`) telah disiapkan untuk menjalankan seluruh siklus pemrosesan data dan pemodelan secara menyeluruh.

- **Total File Notebook Dihasilkan**: 2
- **Posisi Branch**: `Data_Preprocessing`
- **Tugas Selanjutnya**: 
  1. *Upload* hasil dari `preprocessing_output` ke *cloud storage* (Google Drive).
  2. Mulai sesi eksekusi di Google Colab.
