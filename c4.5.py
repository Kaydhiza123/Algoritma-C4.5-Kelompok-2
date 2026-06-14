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
    for col in ['IPK', 'SKS', 'ELPT', 'SKP', 'Lama Studi']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Konversi tanggal
    for col in ['Tanggal Yudisium', 'Tanggal Masuk']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Bersihkan Asal
    if 'Asal' in df_clean.columns:
        df_clean['Asal'] = df_clean['Asal'].astype(str).str.strip().str.rstrip(',').str.strip()
    
    # Drop baris dengan nilai kritis yang kosong
    kritis = ['Jenis Kelamin', 'IPK', 'SKS', 'ELPT', 'SKP', 'Lama Studi', 'Asal']
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

    gerbangkertosusila = ['surabaya', 'sidoarjo', 'gresik', 'bangkalan', 'mojokerto', 'lamongan']
    jabodetabek = ['jakarta', 'bogor', 'depok', 'tangerang', 'bekasi']
    jawa_timur = [
    # Kota
    'malang', 'kediri', 'blitar', 'probolinggo',
    'pasuruan', 'madiun', 'batu',
    # Kabupaten
    'sampang', 'pamekasan', 'sumenep',
    'bojonegoro', 'tuban', 'ngawi', 'magetan',
    'ponorogo', 'pacitan', 'trenggalek', 'tulungagung',
    'nganjuk', 'jombang',
    'lumajang', 'jember', 'bondowoso',
    'situbondo', 'banyuwangi'
    ]

    def klasifikasi_asal(asal):
        asal_str = str(asal).strip().lower()
        if asal_str in gerbangkertosusila:
            return 'Gerbangkertosusila'
        elif asal_str in jabodetabek:
            return 'Jabodetabek'
        elif asal_str in jawa_timur:
            return 'Lokal'
        else:
            return 'Non-lokal'

    df_kat['Asal_Kat'] = df_kat['Asal'].apply(klasifikasi_asal)

    df_kat['IPK_Kat'] = df_kat['IPK'].apply(
        lambda x: 'Memuaskan' if 2.00 <= x <= 2.75
        else ('Sangat Memuaskan' if 2.76 <= x <= 3.50
        else ('Cumlaude' if 3.51 <= x <= 4.00 else 'Lainnya'))
    )

    df_kat['SKP_Kat'] = df_kat['SKP'].apply(
        lambda x: 'SANGAT BAIK' if x > 400
        else ('BAIK' if 251 <= x <= 400 else 'CUKUP')
    )

    def klasifikasi_status(row):
        lama_studi = float(row['Lama Studi'])
        try:
            bulan_yudisium = pd.to_datetime(row['Tanggal Yudisium']).month
        except:
            bulan_yudisium = 6
        if lama_studi > 4.00:
            if lama_studi < 4.50:
                return 'Tidak Lulus Tepat Waktu' if bulan_yudisium > 6 else 'Lulus Tepat Waktu'
            else:
                return 'Tidak Lulus Tepat Waktu'
        else:
            return 'Lulus Tepat Waktu'

    df_kat['Status'] = df_kat.apply(klasifikasi_status, axis=1)
    return df_kat

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

    tampil_cols = ['NIM', 'Nama', 'Jenis Kelamin', 'Asal', 'IPK', 'SKS', 'ELPT', 'SKP', 'Lama Studi', 'Tanggal Yudisium']
    tampil_ada = [c for c in tampil_cols if c in df_clean.columns]
    st.dataframe(df_clean[tampil_ada].head(10))

    # STEP 3: KATEGORISASI
    df_processed = kategorisasi_data(df_clean)

    st.subheader("✨ Hasil Kategorisasi Fitur")
    kat_cols = ['Jenis Kelamin', 'Asal', 'Asal_Kat', 'IPK', 'IPK_Kat', 'SKS', 'ELPT', 'SKP', 'SKP_Kat', 'Lama Studi', 'Status']
    st.dataframe(df_processed[kat_cols].head(10))

    st.write("**Distribusi Target (Status Kelulusan):**")
    st.dataframe(df_processed['Status'].value_counts().rename_axis('Status').reset_index(name='Jumlah'))

    # STEP 4: MODELING C4.5
    feature_cols = ['Jenis Kelamin', 'Asal_Kat', 'IPK_Kat', 'SKS', 'ELPT', 'SKP_Kat']
    X = df_processed[feature_cols].copy()
    y = df_processed['Status'].copy()

    # Encode semua kolom — paksa semua ke numerik tanpa cek dtype
    # (Python 3.14 pakai dtype 'str' bukan 'object', jadi cek dtype tidak reliable)
    encoded_maps = {}
    label_encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoded_maps[col] = dict(enumerate(le.classes_))
        label_encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    le_target.fit(['Lulus Tepat Waktu', 'Tidak Lulus Tepat Waktu'])
    y = le_target.transform(y.astype(str))
    target_map = dict(enumerate(le_target.classes_))

    st.session_state.encoded_categories = encoded_maps
    st.session_state.label_encoders = label_encoders
    st.session_state.target_map = target_map
    st.session_state.features = feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    st.session_state.model = clf

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # PERFORMA MODEL
    st.write("---")
    st.header("📊 Performa Model Decision Tree C4.5")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎯 Akurasi Model (Data Uji 20%)", value=f"{acc * 100:.2f} %")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=list(target_map.values())))
    with col2:
        st.write("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(target_map.values()),
                    yticklabels=list(target_map.values()), ax=ax_cm)
        ax_cm.set_ylabel('Aktual')
        ax_cm.set_xlabel('Prediksi')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # VISUALISASI POHON
    st.write("---")
    st.header("🌿 Visualisasi Pohon Keputusan (Decision Tree)")
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(clf, feature_names=feature_cols, class_names=list(target_map.values()),
              filled=True, rounded=True, fontsize=9, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # FORM PREDIKSI
    st.write("---")
    st.header("🔮 Prediksi Mahasiswa Baru")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_jk = st.selectbox("Jenis Kelamin", options=list(encoded_maps['Jenis Kelamin'].values()))
            input_asal = st.selectbox("Kategori Asal Daerah", options=list(encoded_maps['Asal_Kat'].values()))
        with c2:
            input_ipk = st.number_input("IPK (min. 2.00)", min_value=2.00, max_value=4.00, value=2.00, step=0.01, format="%.2f")
        with c3:
            input_sks = st.number_input("Jumlah SKS (min. 144)", min_value=144, max_value=160, value=144)
            input_elpt = st.number_input("Skor ELPT (min. 450)", min_value=450, max_value=677, value=450)
            input_skp = st.number_input("Skor SKP (min. 140)", min_value=140, max_value=600, value=140)

        submit_btn = st.form_submit_button("Prediksi Kelulusan")

        if submit_btn:
            def get_key(val, my_dict):
                for k, v in my_dict.items():
                    if val == v:
                        return k
                return 0

            def kategorisasi_skp_input(skp):
                if skp > 400:
                    return 'SANGAT BAIK'
                elif 251 <= skp <= 400:
                    return 'BAIK'
                else:
                    return 'CUKUP'

            def kategorisasi_ipk_input(ipk):
                ipk = float(ipk)
                if 2.00 <= ipk <= 2.75:
                    return 'Memuaskan'
                elif 2.76 <= ipk <= 3.50:
                    return 'Sangat Memuaskan'
                else:
                    return 'Cumlaude'
            
            if input_sks < 144 or input_elpt < 450 or input_skp < 140:
                st.warning("⚠️ SKS minimal 144, ELPT minimal 450, SKP minimal 140.")
                st.stop()

            input_row = pd.DataFrame([{
                'Jenis Kelamin': get_key(input_jk, encoded_maps['Jenis Kelamin']),
                'Asal_Kat': get_key(input_asal, encoded_maps['Asal_Kat']),
                'IPK_Kat': get_key(kategorisasi_ipk_input(input_ipk), encoded_maps['IPK_Kat']),
                'SKS': input_sks,
                'ELPT': input_elpt,
                'SKP_Kat': get_key(kategorisasi_skp_input(input_skp), encoded_maps['SKP_Kat'])
            }])[feature_cols]

            pred_code = st.session_state.model.predict(input_row)[0]
            hasil = target_map[pred_code]

            if hasil == 'Lulus Tepat Waktu':
                st.success(f"🎉 Hasil Prediksi: **{hasil}**")
            else:
                st.error(f"⚠️ Hasil Prediksi: **{hasil}**")
else:
    st.info("💡 Silakan unggah file Excel data lulusan (.xlsx) di sidebar kiri untuk memulai.")
