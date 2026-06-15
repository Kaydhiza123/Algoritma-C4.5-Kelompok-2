# 🎓 Dashboard Prediksi Kelulusan Mahasiswa S1 Matematika UNAIR Menggunakan Algoritma C4.5
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://algoritma-c45-kelompok-2-prediksikelulusan.streamlit.app/])

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
- Menampilkan visualisasi pohon keputusan.
- Menghitung akurasi model menggunakan data uji.
- Menampilkan confusion matrix.
- Melakukan prediksi terhadap data mahasiswa baru melalui dashboard interaktif.

---

## 🎯 Tujuan Penelitian

Penelitian ini bertujuan untuk mengimplementasikan algoritma **C4.5** dalam memprediksi kelulusan mahasiswa berdasarkan beberapa faktor akademik, yaitu:

- IPK
- Jumlah SKS
- Skor SKP
- Lama Studi
- Tanggal Yudisium
- Semester

Hasil prediksi terdiri dari dua kelas:

- **Lulus Tepat Waktu**
- **Tidak Lulus Tepat Waktu**

---

## ⚙️ Teknologi yang Digunakan

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
