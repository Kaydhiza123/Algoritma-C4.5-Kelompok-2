# 🎓 Dashboard Prediksi Kelulusan Mahasiswa S1 Matematika UNAIR Menggunakan Algoritma C4.5
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://algoritma-c45-kelompok-2-prediksikelulusan.streamlit.app/)

## 📖 Deskripsi Proyek

Aplikasi ini merupakan dashboard berbasis **Streamlit** yang digunakan untuk melakukan prediksi status kelulusan mahasiswa S1 Matematika Universitas Airlangga menggunakan algoritma **C4.5** yang dihitung secara manual.

Program mampu:
- Membaca data mentah lulusan dalam format Excel.
- Melakukan parsing data multi-header secara otomatis.
- Melakukan cleaning data.
- Menghasilkan fitur tambahan (Semester).
- Menghitung algoritma C4.5 secara manual menggunakan:
  - Entropy
  - Information Gain
  - Split Information
  - Gain Ratio
- Menampilkan proses perhitungan setiap iterasi.
- Membentuk pohon keputusan secara rekursif.
- Menampilkan visualisasi pohon keputusan dinamis menggunakan Graphviz.
- Menghitung akurasi model menggunakan data uji.
- Menampilkan confusion matrix.
- Melakukan kalkulasi otomatis waktu studi (Lama Studi & Semester) berdasarkan input tanggal.
- Melakukan prediksi terhadap data mahasiswa baru melalui dashboard interaktif 2-kolom yang simetris.

---

## 🎯 Tujuan Penelitian

Penelitian ini bertujuan untuk mengimplementasikan algoritma **C4.5** dalam memprediksi kelulusan mahasiswa berdasarkan beberapa faktor akademik, yaitu:
- IPK
- Jumlah SKS
- Skor SKP
- Tanggal Masuk *(Kini ikut dihitung sebagai atribut penentu node)*
- Tanggal Yudisium
- Lama Studi *(Berupa nilai kontinu ordinal/riil)*
- Semester

Hasil prediksi terdiri dari dua kelas:
- **Lulus Tepat Waktu**
- **Tidak Lulus Tepat Waktu**

---

## ⚙️ Fitur Interaktif & Validasi Dashboard

Untuk meningkatkan akurasi prediktif dan pengalaman pengguna (*user experience*), form prediksi mahasiswa baru pada dashboard dilengkapi dengan aturan dinamis:
1. **Otomatisasi Lama Studi:** Pengguna hanya perlu memasukkan `Tanggal Masuk` dan `Tanggal Yudisium`. Sistem akan otomatis mengonversi selisih hari menjadi format tahun desimal (Lama Studi) dan jumlah Semester.
2. **Validasi Kronologis:** Sistem akan memblokir proses prediksi dan menampilkan pesan *error* jika pengguna memasukkan `Tanggal Yudisium` yang mendahului `Tanggal Masuk`.
3. **Penyelarasan Regulasi SKP Dinamis:** Validasi minimal skor SKP disesuaikan secara otomatis berbasis tahun angkatan masuk mahasiswa untuk mengakomodasi perubahan regulasi internal kampus:
   - **Angkatan $\le$ 2015:** SKP minimal **75**.
   - **Angkatan 2016 - 2019:** SKP minimal **100**.
   - **Angkatan 2020 - 2025:** SKP minimal **140**.
   
   Sistem akan memblokir proses komputasi dan memberikan pesan peringatan edukatif jika input SKP tidak memenuhi kriteria angkatan tersebut.
4. **Layout Simetris:** Form disusun menggunakan visualisasi 2 kolom (Data Akademik & Garis Waktu Studi) demi menjaga kerapian antarmuka Streamlit.

---

## 🛠️ Teknologi yang Digunakan

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Graphviz

---

## 📂 Struktur Program

```text
project/
│
├── c4.5.py
├── README.md
├── requirements.txt
