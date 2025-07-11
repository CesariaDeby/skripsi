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

st.set_page_config(layout="wide", page_title="Clustering Perceraian", page_icon="💔")
st.title("💔 Clustering Faktor Perceraian di Kabupaten/Kota")

# ===============================
# MENU UTAMA
# ===============================
menu = st.sidebar.selectbox(
    "📋 Menu", [
        "Unggah & Pratinjau Data",
        "Distribusi & Outlier",
        "Clustering OPTICS",
        "Ringkasan Hasil"
    ]
)

# ===============================
# FILE UPLOADER
# ===============================
uploaded_file = st.sidebar.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

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
        'perselisihan dan pertengkaran', 'ekonomi', 'KDRT',
        'meninggalkan salah satu pihak', 'zina'
    ]

    df = df[df['Jumlah Cerai'] > 0].copy()
    df.fillna(0, inplace=True)

    for col in selected_features:
        df[col] = df[col] / df['Jumlah Cerai']

    df_prop = df[selected_features].copy()

    if menu == "Unggah & Pratinjau Data":
        st.header("📄 Pratinjau Data")
        st.dataframe(df[['wilayah'] + selected_features].set_index('wilayah'))

    elif menu == "Distribusi & Outlier":
        st.header("📊 Distribusi & Deteksi Outlier")
        tabs = st.tabs(["Distribusi", "Boxplot", "Post-Winsorization", "Korelasi"])

        with tabs[0]:
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for i, col in enumerate(selected_features):
                ax = axs[i // 3][i % 3]
                sns.histplot(df_prop[col], kde=True, bins=20, ax=ax)
                ax.set_title(f"{col}\nSkewness: {skew(df_prop[col]):.2f}")
            st.pyplot(fig)

        with tabs[1]:
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for i, col in enumerate(selected_features):
                ax = axs[i // 3][i % 3]
                sns.boxplot(y=df_prop[col], ax=ax)
                ax.set_title(f"Outlier Check: {col}")
            st.pyplot(fig)

        Q1, Q3 = df_prop.quantile(0.25), df_prop.quantile(0.75)
        IQR = Q3 - Q1
        for col in df_prop.columns:
            lower = Q1[col] - 1.5 * IQR[col]
            upper = Q3[col] + 1.5 * IQR[col]
            df_prop[col] = np.clip(df_prop[col], lower, upper)

        with tabs[2]:
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for i, col in enumerate(selected_features):
                ax = axs[i // 3][i % 3]
                sns.boxplot(y=df_prop[col], ax=ax)
                ax.set_title(f"Post-Winsorization: {col}")
            st.pyplot(fig)

        with tabs[3]:
            corr = df_prop.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax)
            ax.set_title("Heatmap Korelasi antar Faktor")
            st.pyplot(fig)

    elif menu == "Clustering OPTICS":
        st.header("📈 Clustering Menggunakan OPTICS")
        st.sidebar.markdown("### ⚙️ Parameter OPTICS")
        min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
        xi = st.sidebar.slider("xi", 0.01, 0.3, 0.05, step=0.01)
        min_cluster_size = st.sidebar.slider("min_cluster_size", 0.01, 0.5, 0.1, step=0.01)

        scaler = StandardScaler()
        X_std = scaler.fit_transform(df_prop)

        model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        model.fit(X_std)

        labels = model.labels_
        ordering = model.ordering_
        reachability = model.reachability_

        reachability_clean = np.where(np.isinf(reachability), np.nan, reachability)
        finite_reach = reachability_clean[~np.isnan(reachability_clean)]
        max_reach = np.nanmax(finite_reach) if len(finite_reach) > 0 else 1.0
        reachability_plot = np.where(np.isinf(reachability), max_reach * 1.2, reachability)

        reach_ordered = reachability_plot[ordering]
        labels_ordered = labels[ordering]
        space = np.arange(len(labels_ordered))

        def detect_peaks_valleys(data, prominence_factor=0.1):
            data_range = np.max(data) - np.min(data)
            prominence = data_range * prominence_factor
            peaks, _ = find_peaks(data, prominence=prominence, distance=5)
            valleys, _ = find_peaks(-data, prominence=prominence, distance=5)
            return peaks, valleys

        peaks, valleys = detect_peaks_valleys(reach_ordered, prominence_factor=0.15)

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(space, reach_ordered, 'k-', alpha=0.3, linewidth=1)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, label in enumerate(np.unique(labels_ordered)):
            mask = labels_ordered == label
            color = 'black' if label == -1 else colors[i % 10]
            label_text = "Noise" if label == -1 else f"Cluster {label}"
            ax.scatter(space[mask], reach_ordered[mask], c=[color], s=40, label=label_text, edgecolors='black')

        ax.scatter(space[peaks], reach_ordered[peaks], marker='^', s=100, c='red', label='Peaks')
        ax.scatter(space[valleys], reach_ordered[valleys], marker='v', s=100, c='blue', label='Valleys')

        ax.axhline(np.percentile(finite_reach, 85), color='orange', linestyle='--', label='Xi Threshold')
        ax.set_title("Reachability Plot (Peaks & Valleys)", fontsize=16)
        ax.legend()
        st.pyplot(fig)

        st.session_state['labels'] = labels
        st.session_state['X_std'] = X_std

    elif menu == "Ringkasan Hasil":
        st.header("📌 Ringkasan Hasil Clustering")
        if 'labels' in st.session_state:
            labels = st.session_state['labels']
            X_std = st.session_state['X_std']
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            valid = labels != -1
            sil_score = silhouette_score(X_std[valid], labels[valid]) if np.sum(valid) > 1 else None

            st.markdown(f"- Jumlah klaster: **{n_clusters}**")
            st.markdown(f"- Titik noise: **{n_noise}**")
            st.markdown(f"- Cakupan data: **{(len(labels)-n_noise)/len(labels)*100:.1f}%**")
            if sil_score:
                st.markdown(f"- Silhouette Score: **{sil_score:.3f}**")
            else:
                st.warning("Silhouette Score tidak dapat dihitung.")
        else:
            st.info("Silakan lakukan proses clustering terlebih dahulu.")
else:
    st.info("💡 Silakan unggah file Excel terlebih dahulu untuk memulai.")
