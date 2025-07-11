import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
from scipy.stats import skew
import warnings

warnings.filterwarnings("ignore")

# ==========================
# Page Config & CSS Styling
# ==========================
st.set_page_config(layout="wide", page_title="💔 Clustering Perceraian")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab"] {font-size: 16px; padding: 10px 20px;}
    .css-18ni7ap {flex-direction: row;}
    .stSlider > div {color: #2c3e50; font-weight: 500;}
    </style>
""", unsafe_allow_html=True)

# ==========================
# Sidebar Navigation
# ==========================
st.sidebar.title("📋 Menu Navigasi")
menu = st.sidebar.radio("Pilih Tahapan: ", [
    "Unggah Data",
    "Distribusi & Outlier",
    "Clustering OPTICS",
    "Ringkasan Hasil"
])

# ==========================
# Load Dataset
# ==========================
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.df_prop = None
    st.session_state.selected_features = [
        'perselisihan dan pertengkaran', 'ekonomi', 'KDRT', 'meninggalkan salah satu pihak', 'zina']

if menu == "Unggah Data":
    st.title("💔 Clustering Faktor Perceraian di Kabupaten/Kota")
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

        df = df[df['Jumlah Cerai'] > 0].copy()
        df.fillna(0, inplace=True)

        # Hitung proporsi
        for col in st.session_state.selected_features:
            df[col] = df[col] / df['Jumlah Cerai']

        st.session_state.df = df
        st.session_state.df_prop = df[st.session_state.selected_features].copy()

        st.success("File berhasil dimuat dan diproses!")
        st.dataframe(df[['wilayah'] + st.session_state.selected_features].set_index('wilayah'))

elif menu == "Distribusi & Outlier":
    st.title("📊 Visualisasi Distribusi & Deteksi Outlier")
    if st.session_state.df is None:
        st.warning("Silakan unggah file data terlebih dahulu.")
    else:
        tabs = st.tabs(["Distribusi", "Boxplot", "Post-Winsorization", "Heatmap Korelasi"])
        df_prop = st.session_state.df_prop.copy()
        features = st.session_state.selected_features

        with tabs[0]:
            st.markdown("##### Distribusi Data Tiap Faktor")
            fig, axs = plt.subplots(2, 3, figsize=(18, 8))
            for i, col in enumerate(features):
                ax = axs[i // 3][i % 3]
                sns.histplot(df_prop[col], kde=True, bins=20, ax=ax)
                ax.set_title(f"{col}\nSkewness: {skew(df_prop[col]):.2f}")
            plt.tight_layout()
            st.pyplot(fig)

        with tabs[1]:
            st.markdown("##### Deteksi Outlier Sebelum Winsorization")
            fig, axs = plt.subplots(2, 3, figsize=(18, 8))
            for i, col in enumerate(features):
                ax = axs[i // 3][i % 3]
                sns.boxplot(y=df_prop[col], ax=ax)
                ax.set_title(f"{col}")
            plt.tight_layout()
            st.pyplot(fig)

        Q1 = df_prop.quantile(0.25)
        Q3 = df_prop.quantile(0.75)
        IQR = Q3 - Q1
        for column in df_prop.columns:
            lower = Q1[column] - 1.5 * IQR[column]
            upper = Q3[column] + 1.5 * IQR[column]
            df_prop[column] = np.where(df_prop[column] < lower, lower, df_prop[column])
            df_prop[column] = np.where(df_prop[column] > upper, upper, df_prop[column])
        st.session_state.df_prop = df_prop

        with tabs[2]:
            st.markdown("##### Setelah Winsorization")
            fig, axs = plt.subplots(2, 3, figsize=(18, 8))
            for i, col in enumerate(features):
                ax = axs[i // 3][i % 3]
                sns.boxplot(y=df_prop[col], ax=ax)
                ax.set_title(f"{col}")
            plt.tight_layout()
            st.pyplot(fig)

        with tabs[3]:
            st.markdown("##### Korelasi Pearson")
            corr_matrix = df_prop.corr(method='pearson')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax)
            ax.set_title("Heatmap Korelasi antar Faktor")
            st.pyplot(fig)

elif menu == "Clustering OPTICS":
    st.title("📈 Clustering dengan OPTICS")
    if st.session_state.df is None:
        st.warning("Silakan unggah file data terlebih dahulu.")
    else:
        X = st.session_state.df_prop.values
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        st.sidebar.markdown("### ⚙️ Parameter OPTICS")
        min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
        xi = st.sidebar.slider("xi", 0.01, 0.3, 0.05, step=0.01)
        min_cluster_size = st.sidebar.slider("min_cluster_size (proporsi)", 0.01, 0.5, 0.1, step=0.01)

        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        optics.fit(X_std)

        labels = optics.labels_
        ordering = optics.ordering_
        reachability = optics.reachability_

        # Reachability plot
        reach_clean = np.where(np.isinf(reachability), np.nan, reachability)
        finite_reach = reach_clean[~np.isnan(reach_clean)]
        max_reach = np.nanmax(reach_clean) if len(finite_reach) > 0 else 1.0
        reach_plot = np.where(np.isinf(reachability), max_reach * 1.2, reachability)

        reach_ordered = reach_plot[ordering]
        labels_ordered = labels[ordering]
        space = np.arange(len(labels_ordered))

        # Deteksi peaks & valleys
        def detect_peaks_valleys(data, prom=0.15):
            rng = np.max(data) - np.min(data)
            p = rng * prom
            peaks, _ = find_peaks(data, prominence=p, distance=5)
            valleys, _ = find_peaks(-data, prominence=p, distance=5)
            return peaks, valleys

        peaks, valleys = detect_peaks_valleys(reach_ordered)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(space, reach_ordered, 'k-', alpha=0.3)
        for label in np.unique(labels_ordered):
            color = 'black' if label == -1 else plt.cm.tab10(label % 10)
            mask = labels_ordered == label
            ax.scatter(space[mask], reach_ordered[mask], c=[color], label=f"{'Noise' if label==-1 else f'Cluster {label}'}", s=40, alpha=0.8, edgecolors='black', linewidths=0.3)

        ax.scatter(space[peaks], reach_ordered[peaks], marker='^', c='red', s=100, label='Peaks', edgecolors='darkred')
        ax.scatter(space[valleys], reach_ordered[valleys], marker='v', c='blue', s=100, label='Valleys', edgecolors='darkblue')

        ax.set_title("Reachability Plot with Peaks & Valleys")
        ax.set_xlabel("Data Index (OPTICS Ordering)")
        ax.set_ylabel("Reachability Distance")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        st.pyplot(fig)

elif menu == "Ringkasan Hasil":
    st.title("📌 Ringkasan Hasil Clustering")
    st.write("Silakan jalankan proses clustering pada menu sebelumnya untuk melihat ringkasan hasil.")
