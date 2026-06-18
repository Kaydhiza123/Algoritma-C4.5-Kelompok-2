import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="C4.5 Graduation Predictor", layout="wide")

# ==========================================
# 1. PARSING DATA MENTAH MULTI-HEADER
# ==========================================
def parse_raw_excel(uploaded_file):
    df_raw = pd.read_excel(uploaded_file, header=None)
    
    header_rows = []
    for i, row in df_raw.iterrows():
        vals = row.astype(str).str.upper().tolist()
        if 'NO' in vals and 'NIM' in vals:
            header_rows.append(i)
    
    if not header_rows:
        return None, "Tidak ditemukan header 'NO' dan 'NIM' di file."
    
    all_data = []
    for idx, h_row in enumerate(header_rows):
        cols = [str(c).strip() for c in df_raw.iloc[h_row].tolist()]
        end = header_rows[idx + 1] if idx + 1 < len(header_rows) else len(df_raw)
        
        for r in range(h_row + 1, end):
            row_vals = df_raw.iloc[r].tolist()
            if pd.isna(row_vals[0]) or str(row_vals[0]).strip() in ['nan', '']:
                continue
            try:
                int(float(str(row_vals[0])))
            except:
                continue
            row_dict = {cols[c]: row_vals[c] for c in range(len(cols))}
            all_data.append(row_dict)
    
    return pd.DataFrame(all_data), None

# ==========================================
# 2. CLEANING DATA
# ==========================================
def cleaning_data(df):
    df_clean = df.copy()
    
    # Jenis Kelamin: kolom L/P dengan isi 'v'
    def parse_jenis_kelamin(row):
        l_val = str(row.get('L', '')).strip().lower()
        p_val = str(row.get('P', '')).strip().lower()
        if l_val == 'v':
            return 'L'
        elif p_val == 'v':
            return 'P'
        else:
            return np.nan
    
    df_clean['Jenis Kelamin'] = df_clean.apply(parse_jenis_kelamin, axis=1)
    
    # Cari kolom Lama Studi (bisa MASA STUDI atau LAMA STUDI tergantung batch)
    lama_studi_col = None
    for c in df_clean.columns:
        if 'STUDI' in c.upper() or 'LAMA' in c.upper() or 'MASA' in c.upper():
            lama_studi_col = c
            break
    
    if lama_studi_col:
        df_clean['Lama Studi'] = df_clean[lama_studi_col]
    else:
        return None, "Kolom Lama Studi / Masa Studi tidak ditemukan."
    
    # Normalisasi nama kolom
    rename_map = {
        'ASAL': 'Asal',
        'TANGGAL YUDISIUM': 'Tanggal Yudisium',
        'TANGGAL MASUK': 'Tanggal Masuk',
        'NIM': 'NIM',
        'NAMA': 'Nama',
    }
    df_clean.rename(columns=rename_map, inplace=True)
    
    # Konversi numerik
    for col in ['IPK', 'SKS', 'SKP', 'Lama Studi']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Konversi tanggal
    for col in ['Tanggal Yudisium', 'Tanggal Masuk']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Drop baris dengan nilai kritis yang kosong
    kritis = ['Jenis Kelamin', 'IPK', 'SKS', 'SKP', 'Lama Studi', 'Tanggal Yudisium', 'Tanggal Masuk']
    kritis_ada = [k for k in kritis if k in df_clean.columns]
    before = len(df_clean)
    df_clean.dropna(subset=kritis_ada, inplace=True)
    after = len(df_clean)
    
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean, before - after

# ==========================================
# 3. KATEGORISASI FITUR
# ==========================================
def kategorisasi_data(df):
    df_kat = df.copy()

    # Hitung Semester dari selisih Tanggal Masuk → Tanggal Yudisium (dalam bulan / 6)
    def hitung_semester(row):
        try:
            masuk = pd.to_datetime(row['Tanggal Masuk'])
            yudisium = pd.to_datetime(row['Tanggal Yudisium'])
            selisih_bulan = (yudisium.year - masuk.year) * 12 + (yudisium.month - masuk.month)
            return round(selisih_bulan / 6)
        except:
            return np.nan
    df_kat['Semester'] = df_kat.apply(hitung_semester, axis=1)
    
    # Hitung Rasio SKP untuk setiap baris data
    def hitung_rasio_skp(row):
        try:
            skp_min = get_skp_minimal(row['Tanggal Masuk'])
            skp_asli = float(row['SKP'])
            return round(skp_asli / skp_min, 4)
        except:
            return np.nan
    df_kat['SKP Ratio'] = df_kat.apply(hitung_rasio_skp, axis=1)
    return df_kat

def get_skp_minimal(tanggal_masuk):
    try:
        # Ekstrak tahun jika berupa objek datetime/Timestamp
        tahun = pd.to_datetime(tanggal_masuk).year
    except:
        # Jika sudah berupa angka ordinal atau int
        try:
            tahun = pd.to_datetime(tanggal_masuk, unit='D').year
        except:
            tahun = 2020 # Default jika gagal parsing
            
    if tahun <= 2015:
        return 75
    elif 2016 <= tahun <= 2019:
        return 100
    elif 2020 <= tahun <= 2025:
        return 140
    else:
        return 140 # Default untuk tahun > 2025

# ==========================================================
# FUNGSI PERHITUNGAN MATEMATIKA C4.5 SECARA MANUAL
# ==========================================================

# 1. Hitung Entropi suatu himpunan target (y)
def hitung_entropi_manual(y_label):
    counts = pd.Series(y_label).value_counts()
    total = len(y_label)
    if total == 0:
        return 0
    entropi = 0
    for count in counts:
        p = count / total
        entropi -= p * np.log2(p)
    return entropi

# 2. Cari Split Point (titik potong) terbaik untuk atribut kontinu/numerik
def cari_split_terbaik(X_col, y_curr):
    total_data = len(y_curr)
    entropi_total = hitung_entropi_manual(y_curr)
    
    values = X_col.dropna().unique()
    values.sort()
    
    best_gain_ratio = -1
    best_threshold = None
    best_gain = 0
    
    # Cek setiap kemungkinan titik tengah (midpoint) sebagai threshold
    for val in values:
        # Menjadikan angka aktual saat ini sebagai batas pembagi saklek
        threshold = val
        
        left_mask = X_col <= threshold
        right_mask = X_col > threshold
        
        y_left = y_curr[left_mask]
        y_right = y_curr[right_mask]
        
        # Lewati jika pembagian tidak menghasilkan 2 kubu (misal angka terbesar di dataset)
        if len(y_left) == 0 or len(y_right) == 0:
            continue
            
        entropi_left = hitung_entropi_manual(y_left)
        entropi_right = hitung_entropi_manual(y_right)
        
        # Hitung Information Gain
        gain = entropi_total - (
            (len(y_left) / total_data) * entropi_left + 
            (len(y_right) / total_data) * entropi_right
        )
        
        # Hitung Split Information
        p_left = len(y_left) / total_data
        p_right = len(y_right) / total_data
        split_info = - (p_left * np.log2(p_left) + p_right * np.log2(p_right))
        
        gain_ratio = gain / split_info if split_info > 0 else 0
        
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_threshold = threshold
            best_gain = gain
            
    return best_gain_ratio, best_threshold, best_gain

# 3. Fungsi Rekursif untuk Membangun Pohon & Menyimpan Langkah Hitungannya ke Tabel
def bangun_pohon_manual(X_curr, y_curr, list_tabel_iterasi, info_node="Root"):
    # Jika data sudah homogen (isinya cuma 1 kelas status)
    if len(y_curr.unique()) <= 1:
        return {"n_samples": len(y_curr), "prediksi": y_curr.iloc[0] if len(y_curr) > 0 else "Tidak Diketahui"}
        
    # Jika atribut habis
    if X_curr.shape[1] == 0:
        return {"n_samples": len(y_curr), "prediksi": y_curr.mode()[0]}
        
    entropi_node = hitung_entropi_manual(y_curr)
    
    catatan_atribut = []
    best_attr = None
    best_thresh = None
    max_gr = -1
    
    # Hitung semua atribut untuk kompetisi di node ini
    for col in X_curr.columns:
        gr, thresh, gain = cari_split_terbaik(X_curr[col], y_curr)
        catatan_atribut.append({
            "Node/Iterasi": info_node,
            "Atribut": col,
            "Threshold": f"<= {thresh:.3f}" if thresh else "-",
            "Information Gain": round(gain, 4),
            "Gain Ratio (C4.5)": round(gr, 4)
        })
        
        if gr > max_gr and thresh is not None:
            max_gr = gr
            best_attr = col
            best_thresh = thresh
            
    # Simpan hasil kompetisi atribut ke list global untuk ditampilkan di Streamlit nanti
    if catatan_atribut:
        list_tabel_iterasi.append(pd.DataFrame(catatan_atribut))
        
    if best_attr is None:
        return {"n_samples": len(y_curr), "prediksi": y_curr.mode()[0]}
        
    # Split data secara riil untuk cabang berikutnya
    mask_kiri = X_curr[best_attr] <= best_thresh
    mask_kanan = X_curr[best_attr] > best_thresh
    
    # Hapus atribut terpilih (best_attr) agar tidak bisa dipilih lagi pada percabangan di bawahnya
    X_kiri_baru = X_curr[mask_kiri].drop(columns=[best_attr])
    X_kanan_baru = X_curr[mask_kanan].drop(columns=[best_attr])
    
    # Buat cabang anak (Child Node) dengan melemparkan dataframe yang atributnya sudah berkurang
    cabang_kiri = bangun_pohon_manual(X_kiri_baru, y_curr[mask_kiri], list_tabel_iterasi, f"Cabang: {best_attr} <= {best_thresh:.2f}")
    cabang_kanan = bangun_pohon_manual(X_kanan_baru, y_curr[mask_kanan], list_tabel_iterasi, f"Cabang: {best_attr} > {best_thresh:.2f}")
    
    return {
        "atribut": best_attr,
        "threshold": best_thresh,
        "n_samples": len(y_curr),
        "entropi": round(entropi_node, 4),
        "kiri": cabang_kiri,
        "kanan": cabang_kanan
    }

# Fungsi untuk menelusuri hasil input mahasiswa baru ke dalam pohon manual
def prediksi_pohon_manual(pohon, row_input):
    if "prediksi" in pohon:
        return pohon["prediksi"]
        
    attr = pohon["atribut"]
    thresh = pohon["threshold"]
    
    if row_input[attr].iloc[0] <= thresh:
        return prediksi_pohon_manual(pohon["kiri"], row_input)
    else:
        return prediksi_pohon_manual(pohon["kanan"], row_input)

# ==========================================================
# FUNGSI EVALUASI PERFORMA POHON MANUAL (CONFUSION MATRIX)
# ==========================================================
def evaluasi_pohon_manual(pohon, X_test, y_test):
    y_pred_manual = []
    
    # Lakukan prediksi baris demi baris pada data uji
    for idx in range(len(X_test)):
        row_input = X_test.iloc[[idx]]
        prediksi = prediksi_pohon_manual(pohon, row_input)
        y_pred_manual.append(prediksi)
        
    y_pred_manual = np.array(y_pred_manual)
    y_true = np.array(y_test)
    
    # Hitung nilai akurasi manual
    benar = np.sum(y_pred_manual == y_true)
    total = len(y_true)
    akurasi = benar / total if total > 0 else 0
    
    return akurasi, y_true, y_pred_manual

# ==========================================================
# FUNGSI UNTUK MENGGAMBAR GRAPH POHON MANUAL
# ==========================================================
def generate_graphviz_pohon(pohon, dot=None, parent_id=None, edge_label=""):
    import graphviz
    
    if dot is None:
        dot = graphviz.Digraph(comment='Pohon Keputusan C4.5 Manual')
        dot.attr(rankdir='TB', size='10,10')
        # Atur style visual node agar rapi dan profesional
        dot.attr('node', shape='box', style='filled,rounded', color='black', fontname='Arial', fontsize='11')
        dot.attr('edge', fontname='Arial', fontsize='10', color='gray')
        
    # Ambil ID unik untuk node saat ini menggunakan hash memori objek
    current_id = str(id(pohon))
    
    # Jika node adalah Daun/Leaf (Hasil keputusan akhir)
    if "prediksi" in pohon:
        label_daun = f"STATUS:\n{pohon['prediksi']}\n(Samples: {pohon['n_samples']})"
        # Warnai hijau jika Lulus Tepat Waktu, oranye jika Tidak Lulus Tepat Waktu
        warna = '#C2EABD' if pohon['prediksi'] == 'Lulus Tepat Waktu' else '#FFD3B6'
        dot.node(current_id, label_daun, fillcolor=warna, shape='ellipse', style='filled')
    else:
        # Jika node adalah cabang/kondisi atribut
        label_cabang = f"Apakah {pohon['atribut']}?\n(Samples: {pohon['n_samples']}\nEntropi: {pohon['entropi']})"
        dot.node(current_id, label_cabang, fillcolor='#E3F2FD')
        
    # Hubungkan ke node di atasnya (parent node) jika ada
    if parent_id is not None:
        dot.edge(parent_id, current_id, label=edge_label)
        
    # Rekursif ke cabang anak kiri dan kanan jika bukan leaf node
    if "prediksi" not in pohon:
        # Cabang Kiri (Memenuhi kondisi <= threshold)
        generate_graphviz_pohon(pohon["kiri"], dot, current_id, f"<= {pohon['threshold']:.2f}")
        # Cabang Kanan (Lebih besar dari > threshold)
        generate_graphviz_pohon(pohon["kanan"], dot, current_id, f"> {pohon['threshold']:.2f}")
        
    return dot
    
# ==========================================
# 4. ANTARMUKA DASHBOARD
# ==========================================
st.title("🎓 Dashboard Prediksi Kelulusan C4.5 S1 Matematika UNAIR")
st.write("Upload data mentah Excel wisuda — aplikasi otomatis membersihkan dan membangun pohon keputusan C4.5.")

st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah file Excel mentah (.xlsx)", type=['xlsx'])

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.encoded_categories = {}
    st.session_state.features = []
    st.session_state.target_map = {}

if uploaded_file is not None:

    # STEP 1: PARSING
    df_parsed, parse_err = parse_raw_excel(uploaded_file)
    if parse_err:
        st.error(parse_err)
        st.stop()

    st.subheader("📋 Data Hasil Parsing (Sebelum Cleaning)")
    st.dataframe(df_parsed.head(10))
    st.caption(f"Total baris terparsing: **{len(df_parsed)}**")

    # STEP 2: CLEANING
    result = cleaning_data(df_parsed)
    if result[0] is None:
        st.error(result[1])
        st.stop()
    df_clean, baris_dihapus = result

    st.subheader("🧹 Hasil Cleaning Data")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Data Masuk", len(df_parsed))
    col_info2.metric("Baris Dihapus (missing kritis)", baris_dihapus)
    col_info3.metric("Data Siap Diproses", len(df_clean))

    st.write("**Distribusi Jenis Kelamin (hasil parsing kolom L/P):**")
    st.dataframe(df_clean['Jenis Kelamin'].value_counts().rename_axis('Jenis Kelamin').reset_index(name='Jumlah'))

    tampil_cols = ['NIM', 'Nama', 'Jenis Kelamin', 'IPK', 'SKS', 'SKP', 'Lama Studi', 'Tanggal Masuk', 'Tanggal Yudisium']
    tampil_ada = [c for c in tampil_cols if c in df_clean.columns]
    st.dataframe(df_clean[tampil_ada].head(10))

    # STEP 3: KATEGORISASI
    df_processed = kategorisasi_data(df_clean)
    
    st.subheader("✨ Hasil Kategorisasi Fitur")
    kat_cols = ['Jenis Kelamin', 'IPK', 'SKS', 'SKP', 'SKP Ratio', 'Lama Studi', 'Tanggal Yudisium', 'Semester', 'STATUS']
    kat_cols_ada = [c for c in kat_cols if c in df_processed.columns]
    st.dataframe(df_processed[kat_cols_ada].head(10))

    st.write("**Distribusi Target (Status Kelulusan):**")
    st.dataframe(df_processed['STATUS'].value_counts().rename_axis('Status').reset_index(name='Jumlah'))

    # STEP 4: MODELING C4.5
    # Fitur: IPK (numerik), SKS, SKP (numerik), Lama Studi, Tanggal Yudisium, Semester
    feature_cols = ['IPK', 'SKS', 'SKP', 'Lama Studi', 'Tanggal Masuk','Tanggal Yudisium', 'Semester']
    df_model = df_processed.copy()

    # Timpa nilai SKP mentah dengan SKP Ratio yang sudah dihitung di STEP 3
    if 'SKP Ratio' in df_model.columns:
        df_model['SKP'] = df_model['SKP Ratio']
    else:
        # Antisipasi darurat jika kolom gagal terbuat di fungsi atas
        def hitung_rasio_skp_darurat(row):
            skp_min = get_skp_minimal(row['Tanggal Masuk'])
            return round(float(row['SKP']) / skp_min, 3) if pd.notna(row['SKP']) else np.nan
        
        df_model['SKP'] = df_model.apply(hitung_rasio_skp_darurat, axis=1)

    # Konversi Tanggal Yudisium ke nilai numerik (ordinal)
    df_model['Tanggal Yudisium'] = pd.to_datetime(df_model['Tanggal Yudisium']).map(
        lambda x: x.toordinal() if pd.notna(x) else np.nan
    )

    # Konversi Tanggal Masuk ke nilai numerik (ordinal)
    df_model['Tanggal Masuk'] = pd.to_datetime(df_model['Tanggal Masuk']).map(
        lambda x: x.toordinal() if pd.notna(x) else np.nan
    )

    # Hitung Rasio SKP untuk setiap baris di dataset berdasarkan Tanggal Masuk
    def hitung_rasio_skp(row):
        skp_min = get_skp_minimal(row['Tanggal Masuk'])
        return row['SKP'] / skp_min

    # Ubah nilai SKP menjadi nilai rasio SKP/SKP_Minimal
    df_model['SKP'] = df_model.apply(hitung_rasio_skp, axis=1)

    X = df_model[feature_cols].copy()
    y = df_model['STATUS'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Wadah untuk menampung tabel-tabel iterasi manual
    st.session_state.list_tabel_iterasi = []

    # Jalankan perhitungan pohon manual
    pohon_keputusan_manual = bangun_pohon_manual(X, y, st.session_state.list_tabel_iterasi, "Root Node (Awal)")
    st.session_state.pohon_manual = pohon_keputusan_manual  # Simpan hasil struktur pohonnya

    # 📊 TAMPILKAN LANGKAH PERHITUNGAN MANUAL DI DASHBOARD
    st.write("---")
    st.header("🧮 Proses Perhitungan Manual Algoritma C4.5")
    st.write("Berikut merupakan hasil iterasi perhitungan *Gain Ratio* untuk setiap atribut di setiap percabangan:")

    for idx, df_iterasi in enumerate(st.session_state.list_tabel_iterasi):
        node_name = df_iterasi["Node/Iterasi"].iloc[0]
        st.subheader(f"📍 Iterasi {idx + 1}: Perhitungan pada {node_name}")
        
        # Cari pemenang di iterasi ini
        df_sorted = df_iterasi.sort_values(by="Gain Ratio (C4.5)", ascending=False).reset_index(drop=True)
        st.dataframe(df_sorted)
        
        pemenang = df_sorted.iloc[0]
        st.info(f"🏆 Atribut **{pemenang['Atribut']}** dipilih sebagai node karena memiliki nilai **Gain Ratio tertinggi ({pemenang['Gain Ratio (C4.5)']})** dengan batasan/threshold `{pemenang['Threshold']}`.")
   
    # ==============================================================
    # 🌳 VISUALISASI POHON KEPUTUSAN DARI HASIL HITUNGAN MANUAL
    # ==============================================================
    st.write("---")
    st.header("🌳 Visualisasi Pohon Keputusan C4.5 (Versi Manual)")
    st.write("Grafik di bawah ini digenerate secara dinamis murni mengikuti alur percabangan matematika manual di atas:")
    
    try:
        # Panggil fungsi generator graphviz menggunakan pohon manual kita
        grafik_pohon = generate_graphviz_pohon(st.session_state.pohon_manual)
        
        # Tampilkan grafik ke dashboard Streamlit
        st.graphviz_chart(grafik_pohon)
        
    except Exception as e:
        st.error(f"Gagal memvisualisasikan pohon manual. Pastikan library graphviz sudah terinstal. Error: {e}")
    
    # ==============================================================
    # 📊 PERFORMA MODEL & CONFUSION MATRIX
    # ==============================================================
    st.write("---")
    st.header("📊 Performa Model & Confusion Matrix (Versi Manual)")
    
    # Jalankan fungsi evaluasi menggunakan data uji (X_test dan y_test)
    acc_manual, y_true, y_pred = evaluasi_pohon_manual(st.session_state.pohon_manual, X_test, y_test)
    
    col_perf1, col_perf2 = st.columns(2)
    with col_perf1:
        st.metric(label="🎯 Akurasi Model Manual (Data Uji 20%)", value=f"{acc_manual * 100:.2f} %")
        st.write(f"Detail: **{np.sum(y_pred == y_true)}** prediksi benar dari **{len(y_true)}** total data uji.")
    
    with col_perf2:
        st.write("**Confusion Matrix (Prediksi vs Aktual):**")
        
        # Membuat DataFrame Confusion Matrix secara manual menggunakan pd.crosstab
        df_eval = pd.DataFrame({
            'Status Aktual (Asli)': y_true,
            'Hasil Prediksi Model': y_pred
        })
        
        # crosstab ini berfungsi sama persis dengan confusion matrix, namun berbasis teks tabel pandas
        confusion_matrix_manual = pd.crosstab(
            df_eval['Status Aktual (Asli)'], 
            df_eval['Hasil Prediksi Model'], 
            margins=True, # Menambahkan baris/kolom 'All' untuk total data
            margins_name="Total"
        )
        st.dataframe(confusion_matrix_manual, use_container_width=True)

    # Encode target
    le_target = LabelEncoder()
    le_target.fit(['Lulus Tepat Waktu', 'Tidak Lulus Tepat Waktu'])
    y = le_target.transform(y.astype(str))
    target_map = dict(enumerate(le_target.classes_))

    st.session_state.target_map = target_map
    st.session_state.features = feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # FORM PREDIKSI
    st.write("---")
    st.header("🔮 Prediksi Mahasiswa Baru")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📊 Data Akademik")
            input_ipk = st.number_input("IPK (min. 2.00)", min_value=2.00, max_value=4.00, value=2.00, step=0.01, format="%.2f")
            input_sks = st.number_input("Jumlah SKS (min. 144)", min_value=144, max_value=160, value=144)
            input_skp = st.number_input("Skor SKP (<=2015 min. 75, 2016-2019 min. 100, 2020-2025 min. 140)", min_value=75, value=75)
        with c2:
            st.markdown("##### 📅 Garis Waktu Studi")
            
            # Mendefinisikan batas minimum dan maksimum tanggal yang bisa dipilih
            import datetime
            min_date = datetime.date(2000, 1, 1)
            max_date = datetime.date(2030, 12, 31)
            
            # Tambahkan parameter min_value dan max_value
            input_tanggal_masuk = st.date_input(
                "Tanggal Masuk", 
                value=datetime.date(2020, 1, 1), # Nilai default awal saat form dimuat
                min_value=min_date, 
                max_value=max_date
            )
            input_tanggal_yudisium = st.date_input(
                "Tanggal Yudisium", 
                value=datetime.date(2024, 1, 1), # Nilai default awal saat form dimuat
                min_value=min_date, 
                max_value=max_date
            )

        submit_btn = st.form_submit_button("Prediksi Kelulusan", use_container_width=True)

        if submit_btn:
            # 1. Ambil tahun dari tanggal masuk
            tahun_masuk = input_tanggal_masuk.year
            
            # 2. Tentukan batasan SKP berdasarkan aturan tahun masuk
            if tahun_masuk <= 2015:
                skp_minimal = 75
            elif 2016 <= tahun_masuk <= 2019:
                skp_minimal = 100
            elif 2020 <= tahun_masuk <= 2025:
                skp_minimal = 140
            else:
                skp_minimal = 140  # Nilai default untuk tahun > 2025 (jika ada)

            # 3. Validasi SKS minimal
            if input_sks < 144:
                st.warning("⚠️ SKS minimal untuk kelulusan adalah 144.")
                st.stop()

            # 4. Validasi SKP dinamis sesuai tahun masuk
            if input_skp < skp_minimal:
                st.error(f"❌ Untuk Angkatan {tahun_masuk}, SKP minimal adalah {skp_minimal}. Skor SKP Anda saat ini: {input_skp}.")
                st.stop()

            # Hitung Lama Studi (Tahun) secara otomatis dari selisih tanggal
            try:
                selisih_hari = (input_tanggal_yudisium - input_tanggal_masuk).days
                # Validasi jika tanggal yudisium mendahului tanggal masuk
                if selisih_hari < 0:
                    st.error("⚠️ Tanggal Yudisium tidak boleh sebelum Tanggal Masuk.")
                    st.stop()
                
                # Konversi hari ke tahun (dibagi 365.25 untuk akurasi tahun kabisat)
                hitung_lama_studi = round(selisih_hari / 365.25, 1)
            except:
                hitung_lama_studi = 4.0 # Nilai default jika terjadi error
            
            # Hitung semester dari input tanggal
            try:
                masuk = pd.to_datetime(input_tanggal_masuk)
                yudisium = pd.to_datetime(input_tanggal_yudisium)
                selisih_bulan = (yudisium.year - masuk.year) * 12 + (yudisium.month - masuk.month)
                input_semester = round(selisih_bulan / 6)
            except:
                input_semester = 8

            input_tgl_yudisium_ordinal = pd.to_datetime(input_tanggal_yudisium).toordinal()
            input_tgl_masuk_ordinal = pd.to_datetime(input_tanggal_masuk).toordinal()
            input_skp_rasio = input_skp / skp_minimal

            st.info(f"📅 **Kalkulasi Otomatis Waktu Studi:**\n"
                    f"* Lama Studi: **{hitung_lama_studi} Tahun**\n"
                    f"* Semester: **{input_semester} Semester**")
            
            input_row = pd.DataFrame([{
                'IPK': input_ipk,
                'SKS': input_sks,
                'SKP': input_skp_rasio,
                'Lama Studi': hitung_lama_studi,
                'Tanggal Masuk': input_tgl_masuk_ordinal,
                'Tanggal Yudisium': input_tgl_yudisium_ordinal,
                'Semester': input_semester,
            }])[feature_cols]

            #(MANUAL):
            hasil = prediksi_pohon_manual(st.session_state.pohon_manual, input_row)

            if hasil == 'Lulus Tepat Waktu':
                st.success(f"🎉 Hasil Prediksi: **{hasil}**")
            else:
                st.error(f"⚠️ Hasil Prediksi: **{hasil}**")
else:
    st.info("💡 Silakan unggah file Excel data lulusan (.xlsx) di sidebar kiri untuk memulai.")
