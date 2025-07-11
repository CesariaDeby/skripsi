import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings("ignore")

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(page_title="Clustering Perceraian", layout="wide", page_icon="💔")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966485.png", width=80)
    st.title("💔 Clustering Perceraian")
    st.markdown("""
    Aplikasi ini mengelompokkan kabupaten/kota di Jawa Timur berdasarkan proporsi faktor penyebab perceraian menggunakan algoritma OPTICS.
    
    **Langkah-langkah:**
    1. Unggah data Excel
    2. Lihat distribusi & preprocessing
    3. Atur parameter clustering
    4. Lihat ringkasan hasil
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ by Anda")

# ===============================
# HEADER
# ===============================
st.title("📌 Clustering Faktor Penyebab Perceraian di Jawa Timur")

# ===============================
# UNGGAH DATA
# ===============================
st.header("1. 📤 Unggah Data")
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

    selected_features = [
        'perselisihan dan pertengkaran',
        'ekonomi',
        'KDRT',
        'meninggalkan salah satu pihak',
        'zina'
    ]

    df = df[df['Jumlah Cerai'] > 0].copy()
    df.fillna(0, inplace=True)

    for col in selected_features:
        df[col] = df[col] / df['Jumlah Cerai']

    df_prop = df[selected_features].copy()

    # ===============================
    # TABS UNTUK ANALISIS & CLUSTERING
    # ===============================
    tabs = st.tabs(["📊 Distribusi", "📉 Clustering OPTICS", "📌 Ringkasan"])

    # ===============================
    # TAB 1: DISTRIBUSI & WINSORIZATION
    # ===============================
    with tabs[0]:
        st.subheader("Distribusi dan Outlier")
        fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 4))
        for i, col in enumerate(selected_features):
            sns.histplot(df_prop[col], kde=True, bins=15, ax=axs[i])
            axs[i].set_title(col)
        st.pyplot(fig)

        Q1 = df_prop.quantile(0.25)
        Q3 = df_prop.quantile(0.75)
        IQR = Q3 - Q1

        for column in df_prop.columns:
            lower = Q1[column] - 1.5 * IQR[column]
            upper = Q3[column] + 1.5 * IQR[column]
            df_prop[column] = np.where(df_prop[column] < lower, lower, df_prop[column])
            df_prop[column] = np.where(df_prop[column] > upper, upper, df_prop[column])

        st.markdown("### Boxplot Setelah Winsorization")
        fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 4))
        for i, col in enumerate(selected_features):
            sns.boxplot(y=df_prop[col], ax=axs[i])
            axs[i].set_title(col)
        st.pyplot(fig)

    # ===============================
    # TAB 2: CLUSTERING
    # ===============================
    with tabs[1]:
        st.subheader("OPTICS Clustering")
        X_std = StandardScaler().fit_transform(df_prop)

        col1, col2, col3 = st.columns(3)
        with col1:
            min_samples = st.slider("min_samples", 2, 20, 5)
        with col2:
            xi = st.slider("xi", 0.01, 0.3, 0.05)
        with col3:
            min_cluster_size = st.slider("min_cluster_size", 0.01, 0.5, 0.1)

        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        optics.fit(X_std)
        labels = optics.labels_
        reachability = optics.reachability_
        ordering = optics.ordering_

        st.markdown(f"**Jumlah Cluster:** {len(set(labels)) - (1 if -1 in labels else 0)}")

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for klass, color in zip(range(0, len(set(labels))), colors):
            Xk = ordering[labels[ordering] == klass]
            ax.plot(Xk, reachability[ordering][labels[ordering] == klass], '.', c=color)
        ax.set_ylabel("Reachability Distance")
        ax.set_title("Reachability Plot")
        st.pyplot(fig)

    # ===============================
    # TAB 3: RINGKASAN HASIL
    # ===============================
    with tabs[2]:
        st.subheader("Ringkasan")
        st.markdown(f"- Jumlah kabupaten/kota: **{len(df)}**")
        st.markdown(f"- Jumlah fitur: **{len(selected_features)}**")
        st.markdown(f"- Jumlah cluster: **{len(set(labels)) - (1 if -1 in labels else 0)}**")

else:
    st.info("📥 Silakan unggah file terlebih dahulu.")
