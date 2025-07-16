import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.signal import find_peaks
from PIL import Image
import io
import base64

st.set_page_config(page_title="Clustering Perceraian Jawa Timur", page_icon="💔", layout="wide")

st.markdown("""
<style>
    /* GLOBAL TABLE STYLING (untuk semua tabel) */
    table {
        background-color: white;
        border-collapse: collapse;
        width: 100%;
        font-size: 16px;
        border: 1px solid #ccc;
    }

    th {
        background-color: #d62828;
        color: white;
        padding: 10px;
        text-align: left;
        font-size: 16px;
        font-weight: bold;
        border: 1px solid #ccc;
    }

    td {
        background-color: #ffffff;
        padding: 10px;
        border: 1px solid #ddd;
        color: #212529;
    }

    tr:hover td {
        background-color: #f8f9fa;
    }

    /* Untuk dataframe (st.dataframe) agar tidak pudar */
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }

    /* Sidebar juga rapikan */
    [data-testid="stSidebar"] {
        background-color: #343a40;
        color: white;
    }

    /* Label atau badge wilayah dalam tabel */
    .css-1wivap2, .css-qrbaxs, .css-1cpxqw2 {
        background-color: #edf2f4 !important;
        color: #2b2d42 !important;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 2px;
        display: inline-block;
        font-weight: 500;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# CUSTOM BACKGROUND & STYLING
# =============================
st.markdown("""
    <style>
        /* ===== LATAR BELAKANG UTAMA ===== */
        .stApp {
            background-color: #fff5f5;
            background-image: linear-gradient(to bottom right, #fff5f5, #ffeaea);
        }

        /* ===== SIDEBAR (MERAH MUDA GELAP) ===== */
        [data-testid="stSidebar"] {
            background-color: #dca8a8; /* merah muda agak gelap */
            color: #3b0000;
        }

        [data-testid="stSidebar"] .css-1v0mbdj {
            color: #3b0000;
        }

        /* Sidebar teks */
        .css-h5rgaw, .css-10trblm {
            color: #3b0000 !important;
        }

        /* HEADER DAN TEKS */
        h1, h2, h3 {
            color: #8b0000 !important;
        }

        /* Tombol */
        .stButton>button {
            background-color: #d62828;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }

        .stButton>button:hover {
            background-color: #a61e1e;
            color: white;
        }

        /* Tabel dan DataFrame */
        table {
            background-color: white;
            border-collapse: collapse;
            width: 100%;
            font-size: 16px;
            border: 1px solid #ccc;
        }

        th {
            background-color: #d62828;
            color: white;
            padding: 10px;
            text-align: left;
            font-size: 16px;
            font-weight: bold;
            border: 1px solid #ccc;
        }

        td {
            background-color: #ffffff;
            padding: 10px;
            border: 1px solid #ddd;
            color: #212529;
        }

        tr:hover td {
            background-color: #f8f9fa;
        }

        .stDataFrame {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# HEADER DAN LOGO
# =============================
st.sidebar.image("logo_fix.png", width=200)
st.markdown("""
<h1 style='text-align: center; color: #d62828;'>
APLIKASI PENGELOMPOKAN KABUPATEN/KOTA <br>
BERDASARKAN FAKTOR PENYEBAB PERCERAIAN DI JAWA TIMUR
</h1>
<hr style='border: 2px solid #d62828;'>
""", unsafe_allow_html=True)

# =============================
# MENU
# =============================
menu = st.sidebar.selectbox("🔍 Pilih Menu", [
    "Beranda",
    "Upload Data & Pilih Faktor",
    "Preprocessing",
    "Pemodelan OPTICS",
    "Evaluasi Model",
    "Ringkasan Hasil"
])

# Variabel global sementara (simpan di session_state jika perlu)
if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'df_prop' not in st.session_state:
    st.session_state.df_prop = None
if 'X_std' not in st.session_state:
    st.session_state.X_std = None
if 'labels_op' not in st.session_state:
    st.session_state.labels_op = None

# =============================
# BERANDA
# =============================
if menu == "Beranda":
    st.header("📖 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk **mengelompokkan kabupaten/kota di Provinsi Jawa Timur** berdasarkan kemiripan distribusi faktor penyebab perceraian. 
    Dengan menggunakan algoritma **OPTICS Clustering**, aplikasi ini membantu mengungkap pola perceraian yang dominan di setiap wilayah sebagai bahan pertimbangan dalam perumusan kebijakan sosial yang lebih tepat sasaran.

    ### 🧠 Apa itu OPTICS?
    **OPTICS** (Ordering Points To Identify the Clustering Structure) adalah metode klasterisasi yang dikembangkan sebagai **penyempurnaan dari algoritma DBSCAN**. Keduanya termasuk dalam keluarga **density-based clustering**, yaitu metode yang mengelompokkan data berdasarkan kepadatan titik-titik di sekitarnya.
    Jika DBSCAN hanya mampu mendeteksi klaster dengan satu tingkat kepadatan dan memerlukan parameter epsilon (jarak maksimum antar titik), maka OPTICS lebih fleksibel. OPTICS **tidak membutuhkan nilai epsilon yang pasti di awal** dan mampu menemukan **klaster dengan berbagai bentuk dan kepadatan berbeda secara otomatis**.
    Keunggulan utama OPTICS adalah kemampuannya membuat **Reachability Plot**, yaitu grafik yang membantu kita **melihat struktur klaster secara visual**, termasuk titik-titik yang dianggap sebagai noise.
    """)
    
    # 💡 GUNAKAN INI UNTUK MENAMPILKAN GAMBAR
    # Buat layout: kolom kosong - kolom isi - kolom kosong
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("alur_op.png", caption="Diagram Alur Proses OPTICS", width=200)

    # Penjelasan alur proses
    st.markdown("""
    ### 🔄 Alur Proses OPTICS
    
    Berikut adalah tahapan utama dalam algoritma OPTICS sebagaimana tergambar pada diagram di atas:
    
    1. **OPTICS**
       - Inisialisasi algoritma dengan parameter `min_samples` dan `xi`.
       - Mulai proses pemetaan kepadatan data.
    
    2. **Deteksi Titik Inti (Core Distance)**
       - Hitung jarak dari setiap titik ke tetangga ke-`min_samples`.
       - Jika titik memiliki cukup tetangga dalam radius tertentu, maka dianggap sebagai titik inti (*core point*).
    
    3. **Jarak Keterjangkauan dan Pengurutan Klaster**
       - Hitung **reachability distance**: jarak aktual atau jarak ke core point yang lebih besar.
       - Urutkan titik berdasarkan keterjangkauannya untuk membentuk urutan pemrosesan (cluster ordering).
    
    4. **Ekstraksi Klaster (Reachability Plot)**
       - Buat **Reachability Plot** dari urutan titik tersebut.
       - Visualisasi ini menunjukkan struktur klaster sebagai lembah (klaster padat) dan puncak (batas/noise).
    
    5. **Evaluasi Hasil**
       - Gunakan metrik evaluasi seperti:
         - **Silhouette Score**: mengukur seberapa baik setiap titik berada di dalam klasternya.
         - **Davies-Bouldin Index (DBI)**: mengukur rasio jarak intra dan antar klaster.
       - Semakin tinggi Silhouette dan semakin rendah DBI → semakin baik hasil klasterisasi.
    """)

    st.markdown("""
    ### ✨ Mengapa OPTICS?
    - Dapat mendeteksi klaster dengan bentuk dan kepadatan yang beragam
    - Tidak perlu menentukan jumlah klaster dari awal
    - Menghasilkan plot visualisasi **Reachability Plot** untuk interpretasi hasil yang lebih mendalam
    - Lebih fleksibel dibanding DBSCAN dalam menangani data kompleks

    ### 💔 Tentang Perceraian
    Perceraian dapat terjadi karena berbagai penyebab. Dalam aplikasi ini, Anda dapat mengeksplorasi beberapa faktor utama yang sering menjadi penyebab perceraian, seperti:
    - **Perselisihan dan Pertengkaran**
        : Ketidakharmonisan yang terus-menerus antara pasangan, sering kali berupa konflik yang tidak terselesaikan.
    - **Masalah Ekonomi**
        : Ketidakmampuan memenuhi kebutuhan hidup yang menimbulkan ketegangan dalam rumah tangga.
    - **Kekerasan Dalam Rumah Tangga (KDRT)**
        : Tindakan kekerasan fisik, psikis, atau seksual yang dilakukan salah satu pihak terhadap pasangannya.
    - **Meninggalkan salah satu pihak**
        : Salah satu pasangan meninggalkan rumah tangga tanpa izin atau alasan yang sah dalam jangka waktu lama.
    - **Zina**
        : Pelanggaran kesetiaan dalam pernikahan melalui hubungan di luar ikatan resmi, seperti perselingkuhan.

    ### 🚀 Alur Penggunaan
    1. Upload data
    2. Pilih faktor
    3. Preprocessing data
    4. Jalankan OPTICS
    5. Evaluasi hasil
    6. Lihat ringkasan dan unduh hasil
    """)

# =============================
# UPLOAD DATA
# =============================
elif menu == "Upload Data & Pilih Faktor":
    st.header("📤 Upload Data & Pilih Faktor")
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            # Rename kolom
            df.rename(columns={
                'Kabupaten/Kota': 'wilayah',
                'Fakor Perceraian - Perselisihan dan Pertengkaran Terus Menerus': 'perselisihan dan pertengkaran',
                'Fakor Perceraian - Ekonomi': 'ekonomi',
                'Fakor Perceraian - Kekerasan Dalam Rumah Tangga': 'KDRT',
                'Fakor Perceraian - Meninggalkan Salah satu Pihak': 'meninggalkan salah satu pihak',
                'Fakor Perceraian - Zina': 'zina',
            }, inplace=True)

            if 'wilayah' not in df.columns:
                st.error("❌ Kolom 'Kabupaten/Kota' tidak ditemukan atau gagal diubah jadi 'wilayah'.")
            else:
                st.session_state.df = df
                st.success("✅ Data berhasil diunggah!")

                all_factors = ['perselisihan dan pertengkaran', 'ekonomi', 'KDRT', 'meninggalkan salah satu pihak', 'zina']
                selected = st.multiselect("Pilih faktor yang ingin digunakan", all_factors, default=all_factors)
                st.session_state.selected_features = selected

                st.subheader("🧾 Pratinjau Data")
                st.dataframe(df[['wilayah'] + selected].set_index('wilayah'))

        except Exception as e:
            st.error(f"❌ Gagal membaca file. Pastikan format kolom sesuai. Error: {e}")

# =============================
# PREPROCESSING
# =============================
elif menu == "Preprocessing":
    if st.session_state.df is None:
        st.warning("Silakan unggah data terlebih dahulu.")
    else:
        df = st.session_state.df.copy()
        selected = st.session_state.selected_features

        expected_cols = ['wilayah', 'Jumlah Cerai'] + selected
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Kolom berikut tidak ditemukan dalam data: {missing_cols}")
        else:
            df = df[expected_cols].copy()
            df = df[df['Jumlah Cerai'] > 0]

            Q1 = df[selected].quantile(0.25)
            Q3 = df[selected].quantile(0.75)
            IQR = Q3 - Q1

            # Navigasi antar tahapan preprocessing
            subtab = st.sidebar.radio("Tahapan Preprocessing", [
                "Statistik", "Cek Missing", "Tangani Missing",
                "Proporsi", "Cek Outlier", "Tangani Outlier",
                "Korelasi", "Standarisasi"])

            if subtab == "Statistik":
                st.subheader("📋 Struktur Dataset")
                jenis_data = ["Nominal" if c.lower() in ["wilayah", "kabupaten/kota"] else "Rasio" for c in df.columns]
                df_info = pd.DataFrame({
                    "Kolom": df.columns,
                    "Jumlah Non-Null": df.notnull().sum().values,
                    "Jenis Data": jenis_data
                })
                st.dataframe(df_info)

                st.subheader("📊 Statistik Deskriptif")
                st.dataframe(df.describe())

                st.markdown("### 🔥 Distribusi Faktor")
                fig, axs = plt.subplots(1, len(selected), figsize=(16, 4))
                for i, col in enumerate(selected):
                    sns.histplot(df[col], kde=True, ax=axs[i])
                    axs[i].set_title(col)
                st.pyplot(fig)

            elif subtab == "Cek Missing":
                st.subheader("🔍 Jumlah Missing Value Tiap Faktor")
                st.dataframe(df[selected].isna().sum().rename("Jumlah Missing"))

            elif subtab == "Tangani Missing":
                df.fillna(0, inplace=True)
                st.session_state.df = df.copy()
                st.subheader("🧽 Setelah Penanganan Missing Value")
                st.dataframe(df[selected].isna().sum().rename("Setelah Ditangani"))
                st.info("Missing value diisi dengan 0 untuk menjaga integritas data.")

            elif subtab == "Proporsi":
                for col in selected:
                    df[col] = df[col] / df['Jumlah Cerai']
                st.session_state.df = df.copy()
                st.session_state.df_prop = df[selected]
                st.subheader("📐 Data dalam Bentuk Proporsi")
                st.info("Data telah diproporsikan berdasarkan jumlah perceraian.")
                st.dataframe(df.set_index('wilayah').head())

            elif subtab == "Cek Outlier":
                st.subheader("🚨 Deteksi Outlier")
                outliers = ((df[selected] < (Q1 - 1.5 * IQR)) | (df[selected] > (Q3 + 1.5 * IQR))).sum()
                st.dataframe(outliers.rename("Jumlah Outlier"))
                fig, axs = plt.subplots(1, len(selected), figsize=(16, 4))
                for i, col in enumerate(selected):
                    sns.boxplot(y=df[col], ax=axs[i])
                    axs[i].set_title(col)
                st.pyplot(fig)

            elif subtab == "Tangani Outlier":
                for col in selected:
                    lower = Q1[col] - 1.5 * IQR[col]
                    upper = Q3[col] + 1.5 * IQR[col]
                    df[col] = np.where(df[col] < lower, lower, df[col])
                    df[col] = np.where(df[col] > upper, upper, df[col])
                st.session_state.df_prop = df[selected]
                st.subheader("🧰 Penanganan Outlier (Winsorization)")
                st.success("Outlier ditangani dengan metode Winsorization.")
                fig, axs = plt.subplots(1, len(selected), figsize=(16, 4))
                for i, col in enumerate(selected):
                    sns.boxplot(y=df[col], ax=axs[i])
                    axs[i].set_title(col)
                st.pyplot(fig)

            elif subtab == "Korelasi":
                st.subheader("🔗 Korelasi Pearson (Setelah Proporsi & Outlier)")
                if "df_prop" not in st.session_state:
                    st.warning("Silakan lakukan proporsi dan penanganan outlier terlebih dahulu.")
                elif len(selected) < 2:
                    st.warning("Silakan pilih minimal 2 faktor untuk analisis korelasi.")
                else:
                    df_prop = st.session_state.df_prop
                    corr = df_prop.corr(method='pearson')
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)

                    def interpretasi_korelasi(r):
                        if 0.00 <= abs(r) <= 0.20:
                            return "Tidak ada korelasi"
                        elif 0.21 <= abs(r) <= 0.40:
                            return "Korelasi lemah"
                        elif 0.41 <= abs(r) <= 0.60:
                            return "Korelasi sedang"
                        elif 0.61 <= abs(r) <= 0.80:
                            return "Korelasi kuat"
                        elif 0.81 <= abs(r) <= 1.00:
                            return "Korelasi sangat kuat"
                        else:
                            return "Nilai tidak valid"

                    st.markdown("#### 📋 Kriteria Interpretasi Koefisien Korelasi")
                    kriteria_korelasi = pd.DataFrame({
                        "Nilai r (Koefisien)": [
                            "0,00 – 0,20", "0,21 – 0,40", "0,41 – 0,60",
                            "0,61 – 0,80", "0,81 – 1,00"
                        ],
                        "Interpretasi": [
                            "Tidak ada korelasi", "Korelasi lemah", "Korelasi sedang",
                            "Korelasi kuat", "Korelasi sempurna"
                        ]
                    })
                    st.dataframe(kriteria_korelasi, use_container_width=True)

                    data_interpretasi = []
                    for i in range(len(corr.columns)):
                        for j in range(i + 1, len(corr.columns)):
                            kolom_1 = corr.columns[i]
                            kolom_2 = corr.columns[j]
                            r = corr.iloc[i, j]
                            interpretasi = interpretasi_korelasi(r)
                            pasangan = f"{kolom_1} vs {kolom_2}"
                            data_interpretasi.append({
                                "Pasangan Variabel": pasangan,
                                "Nilai r": round(r, 3),
                                "Interpretasi": interpretasi
                            })
                    df_interpretasi = pd.DataFrame(data_interpretasi)
                    st.markdown("#### 📑 Tabel Interpretasi Korelasi Antar Faktor")
                    st.dataframe(df_interpretasi, use_container_width=True)

            elif subtab == "Standarisasi":
                st.subheader("📏 Standarisasi Data")
                if "df_prop" not in st.session_state:
                    st.warning("Silakan lakukan proporsi dan penanganan outlier terlebih dahulu.")
                else:
                    df_prop = st.session_state.df_prop
                    X_std = StandardScaler().fit_transform(df_prop)
                    st.session_state.X_std = X_std
                    before = pd.DataFrame(df_prop).describe().loc[['mean', 'std']]
                    after = pd.DataFrame(X_std, columns=selected).describe().loc[['mean', 'std']]
                    st.markdown("**Sebelum Standarisasi**")
                    st.dataframe(before)
                    st.markdown("**Setelah Standarisasi**")
                    st.dataframe(after)
                    st.info("Standardisasi penting agar semua fitur memiliki skala yang sama.")

# =============================
# PEMODELAN OPTICS
# =============================
elif menu == "Pemodelan OPTICS":
    if st.session_state.X_std is None:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
    else:
        st.header("⚙️ Pemodelan OPTICS Clustering")

        st.markdown("### ✏️ Parameter OPTICS")
        st.markdown("""
        - **`min_samples`**: Jumlah minimum tetangga (titik terdekat) yang diperlukan agar suatu titik dianggap sebagai **titik inti** (*core point*). Semakin besar nilainya, semakin ketat syarat pembentukan klaster.
        
        - **`xi`**: Mengatur tingkat toleransi terhadap perubahan kepadatan saat menentukan batas antar klaster. Nilai **lebih kecil** membuat algoritma **lebih sensitif** dalam memisahkan klaster.
        
        - **`min_cluster_size`**: Ukuran **minimum klaster** yang diizinkan, dinyatakan dalam **persentase dari total data** (misalnya 0.1 = 10%). Berguna untuk mencegah terbentuknya klaster yang terlalu kecil atau tidak bermakna.
        """)

        min_samples = st.slider("min_samples", 2, 20, 5)
        xi = st.slider("xi", 0.01, 0.3, 0.05, step=0.01)
        min_cluster_size = st.slider("min_cluster_size (proporsi)", 0.01, 0.5, 0.1, step=0.01)

        if st.button("🚀 Jalankan Pemodelan OPTICS"):
            optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
            optics.fit(st.session_state.X_std)
            st.session_state.labels_op = optics.labels_
            # Setelah clustering berhasil:
            st.session_state.min_samples = min_samples
            st.session_state.xi = xi
            st.session_state.min_cluster_size = min_cluster_size

            reachability = optics.reachability_
            ordering = optics.ordering_
            labels_op = optics.labels_

            st.session_state.reachability = optics.reachability_
            st.session_state.ordering = optics.ordering_

            st.markdown("### 📏 Core Distance")
            st.markdown("""
            **Core Distance** menunjukkan jarak minimum yang diperlukan agar suatu titik dianggap sebagai **titik inti (core point)**. 
            Nilai ini dihitung berdasarkan jarak ke tetangga ke-`min_samples` terdekat. 
            Jika suatu titik tidak memiliki cukup tetangga, maka nilainya adalah 0 atau tidak terdefinisi.
            
            Statistik berikut memberikan gambaran umum mengenai distribusi nilai core distance dari seluruh data.
            """)
            
            core_distances = optics.core_distances_
            st.write(f"Total data points: {len(core_distances)}")
            st.write(f"Core points found: {(core_distances > 0).sum()}")
            st.write(f"Non-core points: {(core_distances == 0).sum()}")
            st.write(pd.Series(core_distances).describe()[['mean','std','min','max']].rename("Core Distance Stats"))
            
            st.markdown("### 🔗 Reachability Distance")
            st.markdown("""
            **Reachability Distance** menunjukkan seberapa mudah suatu titik dijangkau dari titik inti lainnya. 
            Nilai ini mempertimbangkan jarak antar titik dan kepadatan di sekitar titik asal. 
            Semakin kecil nilainya, semakin dekat atau padat hubungan antar titik tersebut.
            
            Reachability digunakan dalam **Reachability Plot**, yang merupakan visualisasi penting untuk mendeteksi struktur klaster pada data.
            """)
            
            reachability = optics.reachability_
            reachability_clean = reachability[np.isfinite(reachability)]
            st.write(f"Finite reachability distances: {len(reachability_clean)}")
            st.write(f"Infinite reachability distances: {(~np.isfinite(reachability)).sum()}")
            st.write(pd.Series(reachability_clean).describe()[['mean','std','min','max']].rename("Reachability Stats"))

            # Distribusi klaster
            labels = optics.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            st.markdown("### 📊 Distribusi Klaster")
            dist_df = pd.Series(labels).value_counts().sort_index()
            for k, v in dist_df.items():
                name = f"Cluster {k}" if k != -1 else "Noise"
                st.write(f"{name}: {v} points ({(v / len(labels) * 100):.1f}%)")
            # Reachability plot update with detailed visualization
            reachability_clean = np.where(np.isinf(reachability), np.nan, reachability)
            finite_reach = reachability_clean[~np.isnan(reachability_clean)]

            if len(finite_reach) > 0:
                max_reach = np.nanmax(reachability_clean)
                reachability_plot = np.where(np.isinf(reachability), max_reach * 1.2, reachability)
            else:
                max_reach = 1.0
                reachability_plot = np.where(np.isinf(reachability), 1.2, reachability)

            reachability_ordered = reachability_plot[ordering]
            labels_ordered = labels_op[ordering]
            space = np.arange(len(labels_ordered))

            unique_labels_ordered = np.unique(labels_ordered)
            cluster_labels = sorted(label for label in unique_labels_ordered if label != -1)
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            label_to_color = {-1: 'black'}
            for i, label in enumerate(cluster_labels):
                label_to_color[label] = colors[i % 10]

            plt.figure(figsize=(15, 8))
            plt.plot(space, reachability_ordered, 'k-', linewidth=1, alpha=0.3, label='Reachability Profile')
            for label in sorted(label_to_color.keys()):
                mask = labels_ordered == label
                color = label_to_color[label]
                label_text = 'Noise' if label == -1 else f'Cluster {label}'
                plt.scatter(space[mask], reachability_ordered[mask], c=[color], s=40, alpha=0.9, label=label_text, edgecolors='black', linewidths=0.3)

            def detect_peaks_valleys(data, prominence_factor=0.1):
                from scipy.signal import find_peaks
                data_range = np.max(data) - np.min(data)
                prominence = data_range * prominence_factor
                peaks, peak_props = find_peaks(data, prominence=prominence, distance=5)
                valleys, valley_props = find_peaks(-data, prominence=prominence, distance=5)
                return peaks, valleys, peak_props, valley_props

            peaks, valleys, peak_props, valley_props = detect_peaks_valleys(reachability_ordered, prominence_factor=0.15)

            if len(peaks) > 0:
                plt.scatter(space[peaks], reachability_ordered[peaks], marker='^', s=100, c='red', alpha=0.8,
                            label=f'Peaks ({len(peaks)}) - Cluster Boundaries', edgecolors='darkred', linewidth=2, zorder=5)
                for i, idx in enumerate(peaks):
                    plt.annotate(f'P{i+1}', xy=(space[idx], reachability_ordered[idx]),
                                 xytext=(5, 10), textcoords='offset points', fontsize=9, fontweight='bold', color='red',
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if len(valleys) > 0:
                plt.scatter(space[valleys], reachability_ordered[valleys], marker='v', s=100, c='blue', alpha=0.8,
                            label=f'Valleys ({len(valleys)}) - Cluster Cores', edgecolors='darkblue', linewidth=2, zorder=5)
                for i, idx in enumerate(valleys):
                    plt.annotate(f'V{i+1}', xy=(space[idx], reachability_ordered[idx]),
                                 xytext=(5, -15), textcoords='offset points', fontsize=9, fontweight='bold', color='blue',
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if len(finite_reach) > 0:
                xi_threshold = np.percentile(finite_reach, 85)
                plt.axhline(y=xi_threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                            label=f'Xi Threshold (≈{xi_threshold:.3f})')
                plt.axhspan(0, xi_threshold, alpha=0.1, color='green', label='Cluster Extraction Zone')

            cluster_boundaries = []
            for i in range(len(unique_labels_ordered) - 1):
                current_label = unique_labels_ordered[i]
                next_label = unique_labels_ordered[i + 1]
                for j in range(len(labels_ordered) - 1):
                    if labels_ordered[j] == current_label and labels_ordered[j + 1] == next_label:
                        cluster_boundaries.append(j + 0.5)
                        break

            for boundary in cluster_boundaries:
                plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)

            plt.title(f"OPTICS Reachability Plot with Peaks & Valleys Analysis", fontsize=16, fontweight='bold', pad=20)
            plt.xlabel("Data Point Index (OPTICS Ordering)", fontsize=12)
            plt.ylabel("Reachability Distance", fontsize=12)
            plt.legend(title="Cluster", fontsize=9, title_fontsize=10, loc='upper right', bbox_to_anchor=(1.25, 1))
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            st.pyplot(plt)

            st.markdown("### 🔧 Parameter yang digunakan:")
            st.write(f"min_samples: {min_samples}")
            st.write(f"xi: {xi}")
            st.write(f"min_cluster_size: {min_cluster_size}")

            st.success("✅ Pemodelan dengan OPTICS berhasil dilakukan!")
            st.markdown("""
            Untuk mengetahui seberapa baik hasil pengelompokan yang terbentuk, 
            silahkan buka menu **Evaluasi Model**!!  
            Di sana Anda dapat melihat penilaian kualitas klaster menggunakan metrik seperti **Silhouette Score** dan **Davies-Bouldin Index**.
            """)

# =============================
# EVALUASI MODEL
# =============================
elif menu == "Evaluasi Model":
    st.header("🧪 Evaluasi Hasil Clustering")
    labels = st.session_state.labels_op
    X = st.session_state.X_std

    if labels is None:
        st.warning("Belum ada hasil clustering.")
    else:
        mask = labels != -1
        if mask.sum() < 2:
            st.warning("Terlalu banyak noise, tidak bisa mengevaluasi.")
        else:
            sil = silhouette_score(X[mask], labels[mask])
            dbi = davies_bouldin_score(X[mask], labels[mask])

            # Silhouette Score
            st.markdown("### 📈 Silhouette Score")
            st.markdown("""
            **Silhouette Score** mengukur seberapa baik setiap data berada dalam klasternya.
            """)
            st.write(f"Nilai Silhouette: **{sil:.3f}**")
            st.dataframe(pd.DataFrame({
                'Rentang': ['> 0.7', '0.5 - 0.7', '0.25 - 0.5', '< 0.25'],
                'Interpretasi': ['Sangat baik', 'Baik', 'Cukup', 'Buruk']
            }))

            # Davies-Bouldin Index
            st.markdown("### 📉 Davies-Bouldin Index")
            st.markdown("""
            **Davies-Bouldin Index (DBI)** mengukur seberapa baik klaster terpisah satu sama lain.  
            Nilai DBI **lebih rendah lebih baik**, karena menunjukkan antar-klaster yang lebih terpisah dan kompak.
            """)
            st.write(f"Nilai DBI: **{dbi:.3f}**")
            st.dataframe(pd.DataFrame({
                'Rentang': ['0 - 1', '1 - 2', '> 2'],
                'Interpretasi': ['Sangat baik', 'Cukup', 'Buruk']
            }))



# =============================
# RINGKASAN HASIL
# =============================
elif menu == "Ringkasan Hasil":
    st.header("📌 Ringkasan Akhir")

    df = st.session_state.df
    labels = st.session_state.labels_op

    if df is None or labels is None:
        st.warning("Belum ada hasil clustering.")
    elif len(df) != len(labels):
        st.error(f"❌ Jumlah data ({len(df)}) dan label hasil clustering ({len(labels)}) tidak cocok. "
                 f"Pastikan preprocessing dan pemodelan dijalankan ulang.")
    else:
        df_result = df.copy()
        df_result['Cluster'] = labels

        st.markdown("### 🔧 Parameter yang Digunakan")
        st.write(f"min_samples: {st.session_state.get('min_samples', '-')}")
        st.write(f"xi: {st.session_state.get('xi', '-')}")
        st.write(f"min_cluster_size: {st.session_state.get('min_cluster_size', '-')}")

        st.markdown("### 📊 Distribusi Klaster")
        dist_df = pd.Series(labels).value_counts().sort_index()
        for k, v in dist_df.items():
            name = f"Cluster {k}" if k != -1 else "Noise"
            st.write(f"{name}: {v} data ({(v / len(labels) * 100):.1f}%)")

        # Reachability Plot
        st.markdown("### 📉 Reachability Plot")
        reachability = st.session_state.reachability
        ordering = st.session_state.ordering

        reachability_clean = np.where(np.isinf(reachability), np.nan, reachability)
        finite_reach = reachability_clean[~np.isnan(reachability_clean)]

        if len(finite_reach) > 0:
            max_reach = np.nanmax(reachability_clean)
            reachability_plot = np.where(np.isinf(reachability), max_reach * 1.2, reachability)
        else:
            max_reach = 1.0
            reachability_plot = np.where(np.isinf(reachability), 1.2, reachability)

        reachability_ordered = reachability_plot[ordering]
        labels_ordered = labels[ordering]
        space = np.arange(len(labels_ordered))

        unique_labels_ordered = np.unique(labels_ordered)
        cluster_labels = sorted(label for label in unique_labels_ordered if label != -1)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        label_to_color = {-1: 'black'}
        for i, label in enumerate(cluster_labels):
            label_to_color[label] = colors[i % 10]

        plt.figure(figsize=(15, 8))
        plt.plot(space, reachability_ordered, 'k-', linewidth=1, alpha=0.3, label='Reachability Profile')

        for label in sorted(label_to_color.keys()):
            mask = labels_ordered == label
            color = label_to_color[label]
            label_text = 'Noise' if label == -1 else f'Cluster {label}'
            plt.scatter(space[mask], reachability_ordered[mask], c=[color], s=40, alpha=0.9, label=label_text, edgecolors='black', linewidths=0.3)

        def detect_peaks_valleys(data, prominence_factor=0.1):
            from scipy.signal import find_peaks
            data_range = np.max(data) - np.min(data)
            prominence = data_range * prominence_factor
            peaks, _ = find_peaks(data, prominence=prominence, distance=5)
            valleys, _ = find_peaks(-data, prominence=prominence, distance=5)
            return peaks, valleys

        peaks, valleys = detect_peaks_valleys(reachability_ordered, prominence_factor=0.15)

        if len(peaks) > 0:
            plt.scatter(space[peaks], reachability_ordered[peaks], marker='^', s=100, c='red', alpha=0.8, label=f'Peaks ({len(peaks)})', edgecolors='darkred', linewidth=2, zorder=5)
        if len(valleys) > 0:
            plt.scatter(space[valleys], reachability_ordered[valleys], marker='v', s=100, c='blue', alpha=0.8, label=f'Valleys ({len(valleys)})', edgecolors='darkblue', linewidth=2, zorder=5)

        if len(finite_reach) > 0:
            xi_threshold = np.percentile(finite_reach, 85)
            plt.axhline(y=xi_threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Xi Threshold (≈{xi_threshold:.3f})')
            plt.axhspan(0, xi_threshold, alpha=0.1, color='green', label='Cluster Extraction Zone')

        cluster_boundaries = []
        for i in range(len(unique_labels_ordered) - 1):
            current_label = unique_labels_ordered[i]
            next_label = unique_labels_ordered[i + 1]
            for j in range(len(labels_ordered) - 1):
                if labels_ordered[j] == current_label and labels_ordered[j + 1] == next_label:
                    cluster_boundaries.append(j + 0.5)
                    break

        for boundary in cluster_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        plt.title(f"OPTICS Reachability Plot with Peaks & Valleys Analysis", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Data Point Index (OPTICS Ordering)", fontsize=12)
        plt.ylabel("Reachability Distance", fontsize=12)
        plt.legend(title="Cluster", fontsize=9, title_fontsize=10, loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        st.pyplot(plt)
        
        # Evaluasi
        st.markdown("### 🧪 Evaluasi Model")
        mask = labels != -1
        if mask.sum() >= 2:
            sil = silhouette_score(st.session_state.X_std[mask], labels[mask])
            dbi = davies_bouldin_score(st.session_state.X_std[mask], labels[mask])
            st.write(f"Silhouette Score: **{sil:.3f}**")
            st.write(f"Davies-Bouldin Index: **{dbi:.3f}**")
        else:
            st.warning("Terlalu banyak noise, evaluasi tidak dapat dilakukan.")

        # Tabel wilayah tiap klaster
        if 'wilayah' in df.columns:
            st.markdown("### 📍 Wilayah Tiap Klaster")
            wilayah_klaster = df.copy()
            wilayah_klaster['Cluster'] = labels
            tabel = wilayah_klaster.groupby('Cluster')['wilayah'].apply(list).rename("Wilayah dalam Klaster")
            st.dataframe(tabel)

            # Interpretasi Karakteristik Tiap Klaster
            st.markdown("### 🧠 Interpretasi Karakteristik Klaster")
    
            def analyze_cluster_characteristics_streamlit(df_std, top_n=3):
                cluster_stats = {}
                unique_labels = df_std['cluster'].unique()
                overall_means = df_std.drop(columns='cluster').mean()
    
                for label in sorted(unique_labels):
                    cluster_data = df_std[df_std['cluster'] == label]
    
                    if label == -1:
                        cluster_stats[label] = {
                            'name': 'Noise / Outliers',
                            'size': len(cluster_data),
                            'percentage': (len(cluster_data) / len(df_std)) * 100,
                            'dominant': []
                        }
                        continue
    
                    cluster_means = cluster_data.drop(columns='cluster').mean()
    
                    dominant_features = []
                    for feature in cluster_means.index:
                        if overall_means[feature] != 0 and cluster_means[feature] > overall_means[feature]:
                            ratio = cluster_means[feature] / overall_means[feature]
                            dominant_features.append((feature, ratio))
    
                    dominant_features.sort(key=lambda x: x[1], reverse=True)
    
                    # Nama deskriptif
                    if dominant_features:
                        top_feature = dominant_features[0][0]
                        name_map = {
                            'ekonomi': 'Faktor Ekonomi Dominan',
                            'perselisihan dan pertengkaran': 'Perselisihan sebagai Faktor Utama',
                            'KDRT': 'Kekerasan dalam Rumah Tangga',
                            'meninggalkan salah satu pihak': 'Dominansi Faktor Meninggalkan',
                            'zina': 'Dominansi Zina / Perselingkuhan'
                        }
                        cluster_name = name_map.get(top_feature, f'Dominasi {top_feature}')
                    else:
                        cluster_name = "Tidak Ada Dominansi Jelas"
    
                    cluster_stats[label] = {
                        'name': cluster_name,
                        'size': len(cluster_data),
                        'percentage': (len(cluster_data) / len(df_std)) * 100,
                        'dominant': dominant_features[:top_n]
                    }
    
                return cluster_stats
    
            # Gabungkan X_std_df dengan label
            X_std_df = st.session_state.X_std.copy()
            X_std_df = pd.DataFrame(X_std_df, columns=st.session_state.X_.columns)
            X_std_df['cluster'] = labels
    
            stats = analyze_cluster_characteristics_streamlit(X_std_df)
    
            for label, info in stats.items():
                if label == -1:
                    st.markdown(f"#### 🔴 Noise (n = {info['size']}, {info['percentage']:.1f}%)")
                    st.write("Terdiri dari data outlier yang tidak termasuk klaster manapun.")
                else:
                    st.markdown(f"#### 🔵 Cluster {label} (n = {info['size']}, {info['percentage']:.1f}%)")
                    st.write(f"**Interpretasi**: {info['name']}")
                    if info['dominant']:
                        st.markdown("**Faktor Dominan:**")
                        for feat, ratio in info['dominant']:
                            st.write(f"- `{feat}`: {ratio:.2f}× dari rerata keseluruhan")
                    else:
                        st.write("Tidak ada fitur dominan yang signifikan.")

            
        # Unduh Excel
        if 'wilayah' in df_result.columns:
            df_download = df_result[['wilayah', 'Cluster']].copy()
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_download.to_excel(writer, index=False, sheet_name='Hasil Klaster')
            buffer.seek(0)
            st.download_button(
                label="📥 Unduh Ringkasan Hasil (.xlsx)",
                data=buffer,
                file_name="hasil_klaster.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
