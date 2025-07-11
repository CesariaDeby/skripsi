import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from scipy.stats import skew
from scipy.signal import find_peaks
import matplotlib.patches as mpatches

st.set_page_config(layout="wide", page_title="Clustering Perceraian", page_icon="💔")

# ================================
# Sidebar Navigasi & Styling
# ================================
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #F0F5F9 !important;
    }
    .custom-title {
        font-family: 'Arial Black';
        font-size: 20px;
        color: #0D3B66;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659730.png", width=100)
    st.markdown("<div class='custom-title'>Clustering Perceraian</div>", unsafe_allow_html=True)
    menu = st.radio("Navigasi", ["Beranda", "Upload & Preprocessing", "Clustering OPTICS", "Ringkasan"])

# ================================
# Tab: Beranda
# ================================
if menu == "Beranda":
    st.title("💔 Clustering Faktor Perceraian di Kabupaten/Kota")
    st.markdown("""
    Aplikasi ini bertujuan untuk mengelompokkan wilayah berdasarkan faktor penyebab perceraian menggunakan algoritma OPTICS.
    Anda dapat mengunggah data, menangani outlier, memilih parameter, dan menampilkan visualisasi interaktif.
    """)

# ================================
# Tab: Upload & Preprocessing
# ================================
elif menu == "Upload & Preprocessing":
    st.title("📂 Upload & Pra-pemrosesan")
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
        st.session_state.df_prop = df_prop
        st.session_state.wilayah = df['wilayah']

        st.subheader("📊 Distribusi & Outlier")
        tab1, tab2 = st.tabs(["Distribusi", "Boxplot"])

        with tab1:
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for i, col in enumerate(selected_features):
                ax = axs[i // 3][i % 3]
                sns.histplot(df_prop[col], kde=True, ax=ax)
                ax.set_title(f"{col}\nSkewness: {skew(df_prop[col]):.2f}")
            st.pyplot(fig)

        with tab2:
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for i, col in enumerate(selected_features):
                ax = axs[i // 3][i % 3]
                sns.boxplot(y=df_prop[col], ax=ax)
                ax.set_title(f"Boxplot: {col}")
            st.pyplot(fig)

        st.subheader("🔄 Winsorization")
        Q1 = df_prop.quantile(0.25)
        Q3 = df_prop.quantile(0.75)
        IQR = Q3 - Q1
        for col in selected_features:
            lower = Q1[col] - 1.5 * IQR[col]
            upper = Q3[col] + 1.5 * IQR[col]
            df_prop[col] = np.clip(df_prop[col], lower, upper)
        st.session_state.df_prop = df_prop
        st.success("Outlier berhasil ditangani dengan Winsorization")

# ================================
# Tab: Clustering OPTICS
# ================================
elif menu == "Clustering OPTICS":
    st.title("💡 Clustering dengan OPTICS")

    if 'df_prop' in st.session_state:
        df_prop = st.session_state.df_prop

        st.sidebar.subheader("Parameter OPTICS")
        min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
        xi = st.sidebar.slider("xi", 0.01, 0.3, 0.05)
        min_cluster_size = st.sidebar.slider("min_cluster_size", 0.01, 0.5, 0.1)

        X = StandardScaler().fit_transform(df_prop)
        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        optics.fit(X)

        labels = optics.labels_
        reachability = optics.reachability_
        ordering = optics.ordering_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette = silhouette_score(X[labels != -1], labels[labels != -1]) if n_clusters > 1 else None

        st.success(f"Jumlah Klaster: {n_clusters}")
        if silhouette:
            st.info(f"Silhouette Score: {silhouette:.3f}")

        st.subheader("🔻 Reachability Plot")
        fig, ax = plt.subplots(figsize=(15, 6))
        space = np.arange(len(X))
        reach_plot = np.where(np.isinf(reachability), np.nanmax(reachability) * 1.2, reachability)
        for klass in set(labels):
            mask = labels[ordering] == klass
            color = 'k' if klass == -1 else plt.cm.tab10(klass % 10)
            ax.plot(space[mask], reach_plot[ordering][mask], '.', markerfacecolor=color)
        ax.set_ylabel('Reachability Distance')
        ax.set_title('Reachability Plot')
        st.pyplot(fig)

        # Simpan hasil
        st.session_state.labels = labels
        st.session_state.optics_model = optics
    else:
        st.warning("Silakan unggah dan pra-proses data terlebih dahulu.")

# ================================
# Tab: Ringkasan
# ================================
elif menu == "Ringkasan":
    st.title("📄 Ringkasan Hasil Clustering")

    if 'labels' in st.session_state and 'wilayah' in st.session_state:
        df_summary = pd.DataFrame({
            'Wilayah': st.session_state.wilayah,
            'Cluster': st.session_state.labels
        })
        st.dataframe(df_summary)
    else:
        st.warning("Belum ada hasil clustering untuk diringkas.")
