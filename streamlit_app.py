# Streamlit GUI untuk Clustering Perceraian Jawa Timur dengan OPTICS
# ==============================================================
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
import io
import base64

st.set_page_config(page_title="Clustering Perceraian Jawa Timur", page_icon="💔", layout="wide")

# =============================
# HEADER DAN LOGO
# =============================
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Coat_of_arms_of_East_Java.svg/800px-Coat_of_arms_of_East_Java.svg.png", width=100)
st.markdown("""
<h1 style='text-align: center; color: #d62828;'>
APLIKASI PENGELOMPOKAN KABUPATEN/KOTA <br>
BERDASARKAN FAKTOR PENYEBAB PERCERAIAN DI JAWA TIMUR
</h1>
<hr style='border: 2px solid #d62828;'>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR MENU
# =============================
menu = st.sidebar.radio("📌 Navigasi", [
    "Beranda", 
    "Upload Data & Pilih Faktor", 
    "Preprocessing", 
    "Pemodelan OPTICS", 
    "Evaluasi Model", 
    "Ringkasan Hasil"])

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
    Aplikasi ini bertujuan untuk **mengelompokkan kabupaten/kota di Provinsi Jawa Timur** berdasarkan proporsi faktor-faktor penyebab perceraian menggunakan algoritma **OPTICS Clustering**.

    ### ✨ Mengapa OPTICS?
    - Dapat mendeteksi **klaster dengan bentuk arbitrer**
    - Tidak perlu menentukan jumlah klaster di awal
    - Menghasilkan plot **Reachability** untuk interpretasi mendalam

    ### 💔 Tentang Perceraian
    Perceraian di Indonesia bisa disebabkan oleh berbagai faktor. Dalam aplikasi ini, Anda dapat memilih faktor-faktor yang dianggap penting seperti:
    - Perselisihan dan Pertengkaran
    - Masalah Ekonomi
    - Kekerasan Dalam Rumah Tangga (KDRT)
    - Meninggalkan salah satu pihak
    - Zina

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

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df.rename(columns={
            'Kabupaten/Kota': 'wilayah',
            'Fakor Perceraian - Perselisihan dan Pertengkaran Terus Menerus': 'perselisihan dan pertengkaran',
            'Fakor Perceraian - Ekonomi': 'ekonomi',
            'Fakor Perceraian - Kekerasan Dalam Rumah Tangga': 'KDRT',
            'Fakor Perceraian - Meninggalkan Salah satu Pihak': 'meninggalkan salah satu pihak',
            'Fakor Perceraian - Zina': 'zina',
        }, inplace=True)
        st.session_state.df = df

        st.success("✅ Data berhasil diunggah!")
        all_factors = ['perselisihan dan pertengkaran', 'ekonomi', 'KDRT', 'meninggalkan salah satu pihak', 'zina']
        selected = st.multiselect("Pilih faktor yang ingin digunakan", all_factors, default=all_factors)
        st.session_state.selected_features = selected

        st.subheader("🧾 Pratinjau Data")
        st.dataframe(df[['wilayah'] + selected].set_index('wilayah'))

# =============================
# PREPROCESSING
# =============================
elif menu == "Preprocessing":
    if st.session_state.df is None:
        st.warning("Silakan unggah data terlebih dahulu.")
    else:
        df = st.session_state.df.copy()
        selected = st.session_state.selected_features
        df = df[df['Jumlah Cerai'] > 0].copy()
        df = df[selected + ['Jumlah Cerai']]

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Statistik & Korelasi", "Cek Missing Value", "Tangani Missing Value",
            "Proporsi", "Cek Outlier", "Tangani Outlier", "Standarisasi"])

        with tab1:
            st.markdown("### 📊 Statistik Deskriptif")
            st.dataframe(df.describe())
            st.markdown("### 🔥 Distribusi Faktor")
            fig, axs = plt.subplots(1, len(selected), figsize=(16, 4))
            for i, col in enumerate(selected):
                sns.histplot(df[col], kde=True, ax=axs[i])
                axs[i].set_title(col)
            st.pyplot(fig)

            st.markdown("### 🔗 Korelasi Pearson")
            corr = df[selected].corr(method='pearson')
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)

        with tab2:
            st.markdown("### 🔍 Jumlah Missing Value Tiap Faktor")
            st.dataframe(df[selected].isna().sum().rename("Jumlah Missing"))

        with tab3:
            df.fillna(0, inplace=True)
            st.dataframe(df[selected].isna().sum().rename("Setelah Ditangani"))
            st.info("Missing value diisi dengan 0 untuk menjaga integritas data.")

        with tab4:
            for col in selected:
                df[col] = df[col] / df['Jumlah Cerai']
            st.session_state.df_prop = df[selected]
            st.info("Data diproporsikan untuk menjaga skala dan menghindari bias karena jumlah perceraian berbeda-beda di tiap daerah.")
            st.dataframe(st.session_state.df_prop.head())

        with tab5:
            Q1 = df[selected].quantile(0.25)
            Q3 = df[selected].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[selected] < (Q1 - 1.5 * IQR)) | (df[selected] > (Q3 + 1.5 * IQR))).sum()
            st.dataframe(outliers.rename("Jumlah Outlier"))
            fig, axs = plt.subplots(1, len(selected), figsize=(16, 4))
            for i, col in enumerate(selected):
                sns.boxplot(y=df[col], ax=axs[i])
                axs[i].set_title(col)
            st.pyplot(fig)

        with tab6:
            for col in selected:
                lower = Q1[col] - 1.5 * IQR[col]
                upper = Q3[col] + 1.5 * IQR[col]
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])
            st.session_state.df_prop = df[selected]
            st.success("Outlier ditangani dengan metode Winsorization.")
            fig, axs = plt.subplots(1, len(selected), figsize=(16, 4))
            for i, col in enumerate(selected):
                sns.boxplot(y=df[col], ax=axs[i])
                axs[i].set_title(col)
            st.pyplot(fig)

        with tab7:
            X_std = StandardScaler().fit_transform(df[selected])
            st.session_state.X_std = X_std
            before = pd.DataFrame(df[selected]).describe().loc[['mean', 'std']]
            after = pd.DataFrame(X_std, columns=selected).describe().loc[['mean', 'std']]
            st.subheader("📊 Sebelum Standarisasi")
            st.dataframe(before)
            st.subheader("📈 Setelah Standarisasi")
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
        - **min_samples**: jumlah minimum tetangga untuk menjadi titik inti
        - **xi**: toleransi perubahan kepadatan (semakin kecil, semakin sensitif)
        - **min_cluster_size**: proporsi minimum ukuran klaster dari total data
        """)

        min_samples = st.slider("min_samples", 2, 20, 5)
        xi = st.slider("xi", 0.01, 0.3, 0.05, step=0.01)
        min_cluster_size = st.slider("min_cluster_size (proporsi)", 0.01, 0.5, 0.1, step=0.01)

        if st.button("🚀 Jalankan Pemodelan OPTICS"):
            optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
            optics.fit(st.session_state.X_std)
            st.session_state.labels_op = optics.labels_

            # Core distances
            core_distances = optics.core_distances_
            st.markdown("### 📏 Core Distance")
            st.write(f"Total data points: {len(core_distances)}")
            st.write(f"Core points found: {(core_distances > 0).sum()}")
            st.write(f"Non-core points: {(core_distances == 0).sum()}")
            st.write(pd.Series(core_distances).describe()[['mean','std','min','max']].rename("Core Distance Stats"))

            # Reachability distances
            reachability = optics.reachability_
            reachability_clean = reachability[np.isfinite(reachability)]
            st.markdown("### 🔗 Reachability Distance")
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

            # Reachability plot
            st.markdown("### 📉 Reachability Plot")
            space = np.arange(len(st.session_state.X_std))
            fig, ax = plt.subplots(figsize=(14, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for klass, color in zip(range(0, n_clusters), colors):
                Xk = space[labels == klass]
                Rk = reachability[labels == klass]
                ax.plot(Xk, Rk, marker='.', linestyle='', ms=5, label=f"Cluster {klass}", color=color)
            ax.plot(space[labels == -1], reachability[labels == -1], 'k.', label='Noise')
            ax.set_ylabel('Reachability Distance')
            ax.set_title('Reachability Plot')
            ax.legend()
            st.pyplot(fig)

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

            st.markdown("### 📈 Silhouette Score")
            st.write(f"Nilai Silhouette: **{sil:.3f}**")
            st.dataframe(pd.DataFrame({
                'Rentang': ['> 0.7', '0.5 - 0.7', '0.25 - 0.5', '< 0.25'],
                'Interpretasi': ['Sangat baik', 'Baik', 'Cukup', 'Buruk']
            }))

            st.markdown("### 📉 Davies-Bouldin Index")
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
    if st.session_state.labels_op is None:
        st.warning("Belum ada hasil clustering.")
    else:
        df = st.session_state.df.copy()
        df['Cluster'] = st.session_state.labels_op
        st.markdown("### 📍 Parameter yang Digunakan")
        st.write(f"min_samples: {min_samples}")
        st.write(f"xi: {xi}")
        st.write(f"min_cluster_size: {min_cluster_size}")

        st.markdown("### 📊 Distribusi Klaster")
        st.dataframe(df.groupby('Cluster')['wilayah'].apply(list).rename("Wilayah dalam Klaster"))

        if 'X_std' in st.session_state and 'labels_op' in st.session_state:
            mask = st.session_state.labels_op != -1
            if mask.sum() >= 2:
                sil = silhouette_score(st.session_state.X_std[mask], st.session_state.labels_op[mask])
                dbi = davies_bouldin_score(st.session_state.X_std[mask], st.session_state.labels_op[mask])
                st.write(f"Silhouette Score: {sil:.3f}")
                st.write(f"Davies-Bouldin Index: {dbi:.3f}")

        csv = df[['wilayah', 'Cluster']].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="hasil_klaster.csv">📥 Unduh Ringkasan Hasil (.csv)</a>'
        st.markdown(href, unsafe_allow_html=True)
