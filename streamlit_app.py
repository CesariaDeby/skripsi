import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from scipy.stats import skew
import warnings

warnings.filterwarnings("ignore")

# ===================================
# Konfigurasi Tampilan
# ===================================
st.set_page_config(layout="wide", page_title="Clustering Perceraian", page_icon="💔")
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #FFE4E1 !important;
        color: black !important;
    }
    .custom-title {
        font-family: 'Georgia', serif;
        font-size: 20px;
        font-weight: bold;
        display: block;
        color: black !important;
    }
    .move-down {
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# ===================================
# Sidebar & Navigasi
# ===================================
with st.sidebar:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.image("https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png", width=140)
    with col2:
        st.markdown("<span class='custom-title move-down'>Clustering Perceraian</span>", unsafe_allow_html=True)

    menu = ["Beranda", "Upload Data", "Clustering", "Ringkasan"]
    selected = st.radio("Navigasi", menu)

# ===================================
# Halaman: Beranda
# ===================================
if selected == "Beranda":
    st.title("💔 Clustering Faktor Perceraian di Kabupaten/Kota")
    st.markdown("""
    Aplikasi ini dirancang untuk mengelompokkan kabupaten/kota berdasarkan faktor penyebab perceraian menggunakan algoritma OPTICS. 
    Anda dapat mengunggah data, melakukan preprocessing, dan melakukan clustering serta meninjau hasil pengelompokannya.
    """)

# ===================================
# Halaman: Upload Data
# ===================================
elif selected == "Upload Data":
    st.title("📤 Upload & Preprocessing Data")
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

        selected_features = ['perselisihan dan pertengkaran', 'ekonomi', 'KDRT', 'meninggalkan salah satu pihak', 'zina']
        df = df[df['Jumlah Cerai'] > 0].copy()
        df.fillna(0, inplace=True)
        for col in selected_features:
            df[col] = df[col] / df['Jumlah Cerai']

        df_prop = df[selected_features].copy()
        st.subheader("🔢 Data Proporsi Faktor Perceraian")
        st.dataframe(df[['wilayah'] + selected_features].set_index('wilayah'))

        st.subheader("📊 Distribusi dan Outlier")
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        for i, col in enumerate(selected_features):
            ax = axs[i // 3][i % 3]
            sns.histplot(df_prop[col], kde=True, bins=20, ax=ax)
            ax.set_title(f"{col}\nSkewness: {skew(df_prop[col]):.2f}")
        st.pyplot(fig)

        # Winsorization
        Q1 = df_prop.quantile(0.25)
        Q3 = df_prop.quantile(0.75)
        IQR = Q3 - Q1
        for col in df_prop.columns:
            lower = Q1[col] - 1.5 * IQR[col]
            upper = Q3[col] + 1.5 * IQR[col]
            df_prop[col] = np.where(df_prop[col] < lower, lower, df_prop[col])
            df_prop[col] = np.where(df_prop[col] > upper, upper, df_prop[col])

        st.session_state.df_prop = df_prop
        st.success("Data berhasil diproses dan disimpan.")

# ===================================
# Halaman: Clustering
# ===================================
elif selected == "Clustering":
    st.title("🔍 Clustering dengan OPTICS")
    if 'df_prop' in st.session_state:
        df_prop = st.session_state.df_prop
        X = df_prop.values
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        st.sidebar.markdown("### ⚙️ Parameter OPTICS")
        min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
        xi = st.sidebar.slider("xi", 0.01, 0.3, 0.05, step=0.01)
        min_cluster_size = st.sidebar.slider("min_cluster_size (proporsi)", 0.01, 0.5, 0.1, step=0.01)

        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        optics.fit(X_std)
        labels = optics.labels_
        reachability = optics.reachability_
        ordering = optics.ordering_

        st.subheader("📈 Reachability Plot")
        fig, ax = plt.subplots(figsize=(12, 5))
        space = np.arange(len(X_std))
        colors = plt.cm.tab10(labels.astype(float) / (max(labels) + 1))
        ax.bar(space, reachability[ordering], color=colors[ordering])
        ax.set_ylabel('Reachability')
        ax.set_title('Reachability Plot')
        st.pyplot(fig)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.success(f"Jumlah Klaster: {n_clusters}")

        if n_clusters > 1:
            sil_score = silhouette_score(X_std[labels != -1], labels[labels != -1])
            st.markdown(f"Silhouette Score: **{sil_score:.4f}**")

        df_result = df_prop.copy()
        df_result['cluster'] = labels
        st.session_state.df_result = df_result
        st.dataframe(df_result)
    else:
        st.warning("Silakan unggah dan proses data terlebih dahulu.")

# ===================================
# Halaman: Ringkasan
# ===================================
elif selected == "Ringkasan":
    st.title("📋 Ringkasan Hasil Clustering")
    if 'df_result' in st.session_state:
        df_result = st.session_state.df_result
        for label in sorted(df_result['cluster'].unique()):
            cluster_df = df_result[df_result['cluster'] == label]
            st.markdown(f"### 🟢 Klaster {label}")
            st.write(cluster_df.describe().T)
            st.bar_chart(cluster_df[selected_features])
    else:
        st.warning("Hasil clustering belum tersedia.")
