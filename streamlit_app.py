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
st.sidebar.image("logo_jatim.png", width=120)
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

# (Lanjutan dari file sebelumnya...)
# ==============================================================
# Bagian PEMODELAN OPTICS, EVALUASI, dan RINGKASAN HASIL

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

            reachability = optics.reachability_
            ordering = optics.ordering_
            labels_op = optics.labels_

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
    if st.session_state.labels_op is None or st.session_state.df is None:
        st.warning("Belum ada hasil clustering atau data tidak tersedia.")
    else:
        df = st.session_state.df.copy()
        labels = st.session_state.labels_op

        if len(df) != len(labels):
            st.error("❌ Jumlah data dan label tidak sesuai. Pastikan preprocessing dan pemodelan telah dijalankan dengan benar.")
        else:
            df['Cluster'] = labels
            st.markdown("### 📍 Parameter yang Digunakan")
            st.write(f"min_samples: {min_samples}")
            st.write(f"xi: {xi}")
            st.write(f"min_cluster_size: {min_cluster_size}")

            st.markdown("### 📊 Distribusi Klaster")
            st.dataframe(df.groupby('Cluster')['wilayah'].apply(list).rename("Wilayah dalam Klaster"))

            if 'X_std' in st.session_state and 'labels_op' in st.session_state:
                mask = labels != -1
                if mask.sum() >= 2:
                    sil = silhouette_score(st.session_state.X_std[mask], labels[mask])
                    dbi = davies_bouldin_score(st.session_state.X_std[mask], labels[mask])
                    st.write(f"Silhouette Score: {sil:.3f}")
                    st.write(f"Davies-Bouldin Index: {dbi:.3f}")

            csv = df[['wilayah', 'Cluster']].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_klaster.csv">📥 Unduh Ringkasan Hasil (.csv)</a>'
            st.markdown(href, unsafe_allow_html=True)
