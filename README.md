# 🎓 Dashboard Prediksi Kelulusan C4.5 S1 Matematika UNAIR

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://algoritma-c45-kelompok-2-proyekdatamining.streamlit.app/)

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi kelulusan tepat waktu mahasiswa S1 Matematika Universitas Airlangga menggunakan algoritma pohon keputusan **C4.5 (Decision Tree dengan kriteria Entropy/Information Gain)**.

---

## 📋 Fitur Utama

- **Parsing otomatis** file Excel multi-header (data dari beberapa batch wisuda)
- **Cleaning data otomatis** termasuk parsing kolom Jenis Kelamin dari format centang (L/P)
- **Kategorisasi fitur** sesuai pedoman akademik UNAIR
- **Visualisasi pohon keputusan** C4.5
- **Evaluasi model** (akurasi, confusion matrix, classification report)
- **Form prediksi interaktif** untuk data mahasiswa baru

---

## 🗂️ Struktur Dataset

File Excel yang diunggah harus memiliki kolom berikut:

| Kolom | Keterangan |
|-------|-----------|
| `NO` | Nomor urut |
| `NIM` | Nomor Induk Mahasiswa |
| `NAMA` | Nama mahasiswa |
| `L` | Kolom jenis kelamin laki-laki (isi `v` jika laki-laki) |
| `P` | Kolom jenis kelamin perempuan (isi `v` jika perempuan) |
| `ASAL` | Kota/kabupaten asal mahasiswa |
| `TANGGAL MASUK` | Tanggal masuk kuliah |
| `TANGGAL YUDISIUM` | Tanggal yudisium/wisuda |
| `IPK` | Indeks Prestasi Kumulatif (min. 2.00) |
| `SKS` | Jumlah SKS yang ditempuh (min. 144) |
| `ELPT` | Skor English Language Proficiency Test (min. 450) |
| `SKP` | Skor Kreativitas dan Prestasi (min. 140) |
| `LAMA STUDI` / `MASA STUDI` | Lama studi dalam tahun (desimal) |

> File dapat berisi beberapa tabel dengan header berulang (per batch wisuda) — aplikasi akan menggabungkannya secara otomatis.

---

## 🔢 Kategorisasi Atribut

### Asal Mahasiswa
| Kategori | Wilayah |
|----------|---------|
| Gerbangkertosusila | Surabaya, Sidoarjo, Gresik, Bangkalan, Mojokerto, Lamongan |
| Lokal | Kota/kabupaten lain di Jawa Timur |
| Jabodetabek | Jakarta, Bogor, Depok, Tangerang, Bekasi |
| Non-lokal | Seluruh wilayah di luar kategori di atas |

### IPK
| Kategori | Rentang |
|----------|---------|
| Memuaskan | 2,00 – 2,75 |
| Sangat Memuaskan | 2,76 – 3,50 |
| Cumlaude | 3,51 – 4,00 |

### SKP
| Kategori | Rentang |
|----------|---------|
| Cukup | 140 – 250 |
| Baik | 251 – 400 |
| Sangat Baik | > 400 |

### Label Target
| Label | Kriteria |
|-------|---------|
| Lulus Tepat Waktu | Lama studi ≤ 4 tahun, atau lama studi < 4,5 tahun dengan yudisium ≤ Juni |
| Tidak Lulus Tepat Waktu | Lama studi ≥ 4,5 tahun, atau lama studi < 4,5 tahun dengan yudisium > Juni |

---

## ⚙️ Instalasi

### 1. Prasyarat
- Python 3.9 atau lebih baru
- pip

### 2. Clone atau download project
```bash
git clone <url-repo>
cd <nama-folder>
```

### 3. Install dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi
```bash
streamlit run c4.5.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 📦 Dependensi

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
```

---

## 🚀 Cara Penggunaan

1. Jalankan aplikasi dengan `streamlit run c4.5.py`
2. Unggah file Excel (`.xlsx`) data lulusan di sidebar kiri
3. Aplikasi otomatis memproses data melalui tahapan:
   - **Parsing** → menggabungkan semua tabel dari berbagai batch wisuda
   - **Cleaning** → memperbaiki format, menghapus data tidak lengkap
   - **Kategorisasi** → mengklasifikasikan atribut sesuai pedoman akademik
   - **Modeling** → membangun pohon keputusan C4.5
4. Lihat hasil evaluasi model dan visualisasi pohon keputusan
5. Gunakan form prediksi di bagian bawah untuk memprediksi mahasiswa baru

---

## 📊 Output Aplikasi

| Output | Keterangan |
|--------|-----------|
| Tabel parsing | Preview data setelah digabung dari semua batch |
| Ringkasan cleaning | Jumlah data masuk, dihapus, dan siap diproses |
| Tabel kategorisasi | Data dengan kolom kategori hasil preprocessing |
| Akurasi model | Persentase akurasi pada data uji (20%) |
| Classification report | Precision, recall, F1-score per kelas |
| Confusion matrix | Heatmap prediksi vs aktual |
| Pohon keputusan | Visualisasi C4.5 dengan kedalaman maksimal 5 |
| Form prediksi | Input data mahasiswa baru → hasil prediksi langsung |

---

## 👨‍💻 Teknologi

- **Python** — bahasa pemrograman utama
- **Streamlit** — framework dashboard web
- **scikit-learn** — implementasi algoritma C4.5 (Decision Tree Entropy)
- **pandas & numpy** — manipulasi dan preprocessing data
- **matplotlib & seaborn** — visualisasi grafik

---

## 📝 Catatan

- Algoritma C4.5 diimplementasikan menggunakan `DecisionTreeClassifier` dengan `criterion='entropy'` dari scikit-learn
- Split data: 80% training, 20% testing dengan stratifikasi kelas
- Kedalaman pohon maksimal (`max_depth`) = 5
