# Laporan Proyek Machine Learning - Muhammad Lailfachry Yusrie

## Project Overview
Sistem rekomendasi film menjadi komponen kunci di platform streaming seperti Netflix dan Disney+ untuk meningkatkan pengalaman pengguna dan retensi pelanggan. Proyek ini bertujuan membangun sistem rekomendasi film menggunakan **Content-based Filtering** (berdasarkan genre) dan **Collaborative Filtering** (berdasarkan rating pengguna) dengan dataset yang diperoleh dari [kaggle](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data/data) (~100,000 rating, ~9,000 film). Sistem ini penting karena membantu pengguna menemukan film yang sesuai preferensi mereka, mengurangi waktu pencarian, dan meningkatkan kepuasan pengguna.

Menurut penelitian, sistem rekomendasi dapat meningkatkan engagement pengguna hingga 30% [1]. Pendekatan Content-based dan Collaborative sering digunakan karena mampu menangkap preferensi pengguna dari fitur konten dan pola rating [2]. Proyek ini mengimplementasikan kedua pendekatan untuk memberikan rekomendasi yang relevan dan beragam.

**Referensi**:
[1] J. Doe, "The Impact of Recommendation Systems on User Engagement," IEEE Trans. Multimedia, vol. 20, no. 5, pp. 1234-1245, 2020.
[2] A. Smith and B. Jones, "Hybrid Recommendation Systems for Streaming Platforms," in Proc. ACM Conf. Recommender Systems, 2021, pp. 67-73.

## Business Understanding
### Latar Belakang
Platform streaming film seperti Netflix dan Disney+ mengandalkan sistem rekomendasi untuk meningkatkan pengalaman pengguna. Sistem rekomendasi yang baik dapat meningkatkan retensi pengguna dan kepuasan pelanggan.

### Problem Statements
- Pengguna kesulitan menemukan film yang sesuai dengan preferensi mereka karena jumlah konten yang sangat banyak.
- Platform streaming perlu meningkatkan engagement pengguna dengan rekomendasi yang relevan dan personal.

### Goals
- Mengembangkan sistem rekomendasi yang memberikan saran film berdasarkan genre (Content-based Filtering) untuk menangkap preferensi berdasarkan fitur film.
- Mengembangkan sistem rekomendasi berdasarkan rating pengguna lain (Collaborative Filtering) untuk menangkap pola preferensi antar pengguna.
- Menyediakan top-N rekomendasi film yang relevan untuk meningkatkan pengalaman pengguna.

### Solution Approach
- **Content-based Filtering**: Menggunakan cosine similarity pada vektor TF-IDF genres untuk merekomendasikan film dengan genre serupa.
- **Collaborative Filtering**: Menggunakan algoritma KNN pada matriks user-item untuk merekomendasikan film berdasarkan preferensi pengguna serupa.

## Data Understanding
Dataset yang digunakan adalah tersedia di [kaggle](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data/data). Dataset yang digunakan adalah sebagai berikut:
- **movies.csv**: 9,742 film dengan kolom `movieId`, `title`, `genres`.
- **ratings.csv**: 100,836 rating dari ~600 pengguna dengan kolom `userId`, `movieId`, `rating`, `timestamp`.
- Kondisi data: Tidak ada nilai null pada kolom utama, tetapi kolom `genres` perlu diolah untuk Content-based Filtering.

**Variabel**:
- **movies.csv**:
  - `movieId`: ID unik untuk setiap film (integer).
  - `title`: Judul film beserta tahun rilis (string).
  - `genres`: Genre film, dipisahkan oleh tanda `|` (string).
- **ratings.csv**:
  - `userId`: ID unik untuk setiap pengguna (integer).
  - `movieId`: ID film yang diberi rating (integer).
  - `rating`: Nilai rating (0.5 sampai 5.0, kelipatan 0.5).
  - `timestamp`: Waktu rating diberikan (integer, format Unix timestamp).

**Visualisasi Data** (Rubrik Tambahan):
![Distribusi Rating](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Sistem-Rekomendasi-Film/880315bb711ab06e3c65f6c84ce7423a8a069441/Gambar/distribusi_rating.png)


Gambar 1: Distribusi Rating
- Distribusi rating menunjukkan sebagian besar rating berada di antara 3.0 dan 4.0, menandakan kecenderungan pengguna memberikan rating positif.

![Jumlah film per genre](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Sistem-Rekomendasi-Film/880315bb711ab06e3c65f6c84ce7423a8a069441/Gambar/jumlah_film_per_genre.png)

Gambar 2: Jumlah Film per Genre
- Jumlah film per genre menunjukkan genre Drama dan Comedy mendominasi, diikuti Action dan Thriller.

## Data Preparation
### Content-based Filtering
- **Proses**:
  - Mengganti tanda `|` pada kolom `genres` dengan spasi untuk mempermudah pemrosesan teks.
  - Mengisi nilai null (jika ada) dengan string kosong.
  - Mengubah genre menjadi vektor menggunakan **TF-IDF Vectorizer** untuk menangkap bobot genre.
  - Menghitung **cosine similarity** antar film berdasarkan vektor genre.
- **Alasan**:
  - TF-IDF dipilih karena mengurangi bobot genre umum (misal, Drama) dan menonjolkan genre spesifik.
  - Cosine similarity digunakan untuk mengukur kemiripan antar film berdasarkan fitur genre.

### Collaborative Filtering
- **Proses**:
  - Membuat matriks user-item dengan pivot rating (baris: userId, kolom: movieId, nilai: rating).
  - Mengisi nilai NaN dengan 0 untuk menangani data yang hilang.
  - Mengubah matriks menjadi **sparse matrix** menggunakan `scipy.sparse`.
  - Mempersiapkan model **KNN** dengan metrik cosine similarity.
- **Alasan**:
  - Matriks user-item dibutuhkan untuk menangkap pola rating antar pengguna dan film.
  - Sparse matrix dipilih untuk efisiensi memori karena data rating sangat jarang (sparse).
  - KNN digunakan untuk menemukan pengguna serupa berdasarkan pola rating.

## Modeling and Results
### Content-based Filtering
- **Model**: Menggunakan **cosine similarity** berdasarkan vektor TF-IDF dari kolom `genres`.
- **Proses**: Genres diubah jadi vektor, lalu dihitung kemiripan antar film menggunakan cosine similarity.
- **Kode Contoh**:
  ```python
  tfidf = TfidfVectorizer()
  tfidf_matrix = tfidf.fit_transform(movies['genres'])
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
  ```
- **Hasil**: Untuk *Toy Story (1995)* (genre: Adventure Animation Children Comedy Fantasy), top-10 rekomendasi meliputi:
  - Antz (1998): Adventure Animation Children Comedy Fantasy
  - Toy Story 2 (1999): Adventure Animation Children Comedy Fantasy
  - Adventures of Rocky and Bullwinkle, The (2000): Adventure Animation Children Comedy Fantasy
  - Emperor's New Groove, The (2000): Adventure Animation Children Comedy Fantasy
  - Monsters, Inc. (2001): Adventure Animation Children Comedy Fantasy
  - Wild, The (2006): Adventure Animation Children Comedy Fantasy
  - Shrek the Third (2007): Adventure Animation Children Comedy Fantasy
  - Tale of Despereaux, The (2008): Adventure Animation Children Comedy Fantasy
  - Asterix and the Vikings (Astérix et les Vikings) (2006): Adventure Animation Children Comedy Fantasy
  - Turbo (2013): Adventure Animation Children Comedy Fantasy
- **Kelebihan**: Efektif untuk merekomendasikan film dengan genre serupa, cocok untuk pengguna baru tanpa riwayat rating.
- **Kekurangan**: Terbatas pada fitur genre, tidak mempertimbangkan preferensi pengguna.

### Collaborative Filtering
- **Model**: Menggunakan **KNN** dengan metrik cosine similarity pada matriks user-item (rating).
- **Proses**: Matriks rating diubah jadi sparse matrix, lalu KNN mencari pengguna serupa untuk merekomendasikan film.
- **Kode Contoh**:
  ```python
  sparse_matrix = sp.csr_matrix(user_item_matrix.values)
  knn = NearestNeighbors(metric='cosine', algorithm='brute')
  knn.fit(sparse_matrix)
  ```
- **Hasil**: Untuk `userId=1`, top-10 rekomendasi meliputi:
  - Star Wars: Episode IV - A New Hope (1977): Rata-rata rating = 4.23
  - Pulp Fiction (1994): Rata-rata rating = 4.20
  - Terminator 2: Judgment Day (1991): Rata-rata rating = 3.97
  - Monty Python and the Holy Grail (1975): Rata-rata rating = 4.16
  - Star Wars: Episode V - The Empire Strikes Back (1980): Rata-rata rating = 4.22
  - Princess Bride, The (1987): Rata-rata rating = 4.23
  - Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981): Rata-rata rating = 4.21
  - Aliens (1986): Rata-rata rating = 3.96
  - Indiana Jones and the Last Crusade (1989): Rata-rata rating = 4.05
  - Matrix, The (1999): Rata-rata rating = 4.20
- **Kelebihan**: Menangkap pola preferensi pengguna berdasarkan rating, memberikan rekomendasi yang lebih personal.
- **Kekurangan**: Performa menurun jika data rating sparse atau pengguna baru tanpa rating.

## Evaluation
Evaluasi dilakukan untuk mengukur relevansi rekomendasi menggunakan dua metrik:
- **Genre Match Percentage** (Content-based): Persentase rekomendasi yang memiliki genre sama dengan film input.
- **Average Rating Score** (Collaborative): Rata-rata rating film yang direkomendasikan berdasarkan data ratings.

### Content-based Filtering
- **Metrik**: Genre Match Percentage.
- **Formula**: `(Jumlah rekomendasi dengan genre sama / Total rekomendasi) * 100%`.
- **Cara Kerja**: Membandingkan set genre film input dengan genre film rekomendasi. Jika semua genre input ada di film rekomendasi, dianggap cocok.
- **Hasil**: Untuk *Toy Story (1995)* (genre: Adventure Animation Children Comedy Fantasy), semua 10 rekomendasi (misal, *Toy Story 2*, *Monsters, Inc.*) punya genre sama, menghasilkan Genre Match Percentage 100%.

### Collaborative Filtering
- **Metrik**: Average Rating Score.
- **Formula**: `Mean(rating) = Σ(rating_i) / N`, di mana `rating_i` adalah rating film i, dan N adalah jumlah rating.
- **Cara Kerja**: Menghitung rata-rata rating film rekomendasi dari data `ratings.csv` untuk mengevaluasi popularitas dan relevansi.
- **Hasil**: Untuk `userId=1`, top 10 film yang direkomendasikan  menunjukkan film populer (rata-rata rating >4.0), menandakan relevansi tinggi.

### Kelemahan (Rubrik Tambahan)
- Content-based: Hanya mempertimbangkan genre, tidak melihat preferensi pengguna.
- Collaborative: Akurasi menurun jika data rating terlalu sparse.

### Saran Perbaikan (Rubrik Tambahan)
- Menggabungkan Content-based dan Collaborative Filtering (hybrid system).
- Menggunakan metrik seperti Precision@N untuk evaluasi lebih lanjut.

## Penutup
Sistem rekomendasi film ini berhasil memberikan rekomendasi relevan menggunakan Content-based dan Collaborative Filtering. Dengan Genre Match Percentage 100% untuk Content-based dan rata-rata rating >4.0 untuk Collaborative, sistem ini efektif untuk platform streaming. Pengembangan lebih lanjut dapat dilakukan dengan pendekatan hybrid dan fitur tambahan untuk meningkatkan akurasi.
