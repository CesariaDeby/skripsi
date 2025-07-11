import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.signal import find_peaks
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from sklearn.decomposition import PCA
from scipy.stats import skew
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Clustering Perceraian", page_icon="💔")
st.title("💔 Clustering Faktor Perceraian di Kabupaten/Kota")

uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    # ================================
    # 1. Load dan Pra-pemrosesan Data
    # ================================
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # ================================
    # 2. Rename Kolom
    # ================================
    df.rename(columns={
        'Kabupaten/Kota': 'wilayah',
        'Fakor Perceraian - Perselisihan dan Pertengkaran Terus Menerus': 'perselisihan dan pertengkaran',
        'Fakor Perceraian - Ekonomi': 'ekonomi',
        'Fakor Perceraian - Kekerasan Dalam Rumah Tangga': 'KDRT',
        'Fakor Perceraian - Meninggalkan Salah satu Pihak': 'meninggalkan salah satu pihak',
        'Fakor Perceraian - Zina': 'zina',
    }, inplace=True)

    # ================================
    # 3. Preprocessing
    # ================================
    selected_features = [
        'perselisihan dan pertengkaran',
        'ekonomi',
        'KDRT',
        'meninggalkan salah satu pihak',
        'zina'
    ]

    df = df[df['Jumlah Cerai'] > 0].copy()
    df.fillna(0, inplace=True)

    # Buat kolom proporsi
    for col in selected_features:
        df[col] = df[col] / df['Jumlah Cerai']

    df_prop = df[selected_features].copy()
    st.subheader("🔢 Data Proporsi Faktor Perceraian")
    st.dataframe(df[['wilayah'] + selected_features].set_index('wilayah'))

    # ================================
    # 4. Visualisasi Distribusi & Outlier
    # ================================
    st.markdown("### 📊 Distribusi dan Outlier Tiap Faktor")

    tabs = st.tabs(["Distribusi", "Boxplot", "Post-Winsorization", "Heatmap Korelasi"])

    with tabs[0]:
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        for i, col in enumerate(selected_features):
            ax = axs[i // 3][i % 3]
            sns.histplot(df_prop[col], kde=True, bins=20, ax=ax)
            ax.set_title(f"{col}\nSkewness: {skew(df_prop[col]):.2f}")
        plt.tight_layout()
        st.pyplot(fig)

    with tabs[1]:
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        for i, col in enumerate(selected_features):
            ax = axs[i // 3][i % 3]
            sns.boxplot(y=df_prop[col], ax=ax)
            ax.set_title(f"Outlier Check: {col}")
        plt.tight_layout()
        st.pyplot(fig)

    # ================================
    # 5. Winsorization
    # ================================
    Q1 = df_prop.quantile(0.25)
    Q3 = df_prop.quantile(0.75)
    IQR = Q3 - Q1

    outlier_counts = ((df_prop < (Q1 - 1.5 * IQR)) | (df_prop > (Q3 + 1.5 * IQR))).sum()
    st.markdown("#### ⚠️ Jumlah Outlier per Fitur")
    st.write(outlier_counts)

    for column in df_prop.columns:
        lower = Q1[column] - 1.5 * IQR[column]
        upper = Q3[column] + 1.5 * IQR[column]
        df_prop[column] = np.where(df_prop[column] < lower, lower, df_prop[column])
        df_prop[column] = np.where(df_prop[column] > upper, upper, df_prop[column])

    with tabs[2]:
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        for i, col in enumerate(selected_features):
            ax = axs[i // 3][i % 3]
            sns.boxplot(y=df_prop[col], ax=ax)
            ax.set_title(f"Post-Winsorization: {col}")
        plt.tight_layout()
        st.pyplot(fig)

    # ================================
    # 6. Korelasi Pearson
    # ================================
    corr_matrix = df_prop.corr(method='pearson')

    with tabs[3]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax)
        ax.set_title("Heatmap Korelasi antar Faktor")
        st.pyplot(fig)

    st.markdown("#### 📈 Korelasi Pearson > 0.6")
    high_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0],dtype=bool)).stack()
    high_corr = high_corr[abs(high_corr) > 0.6].sort_values(ascending=False)
    st.dataframe(high_corr.round(2))

    if selected_features:
        st.success("Fitur terpilih: " + ", ".join(selected_features))

        # ===============================
        # PHASE 2: STANDARISASI DATA
        # ===============================
        X = df_prop.values
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

    
        # ===============================
        # PHASE 4: CLUSTERING DENGAN PARAMETER MANUAL (USER INPUT)
        # ===============================
        st.subheader("✏️ Pilih Parameter OPTICS Sendiri")
        
        with st.sidebar:
            st.markdown("### ⚙️ Parameter Manual OPTICS")
            user_min_samples = st.slider("min_samples", min_value=2, max_value=20, value=5)
            user_xi = st.slider("xi", min_value=0.01, max_value=0.3, step=0.01, value=0.05)
            user_min_cluster_size = st.slider("min_cluster_size (proporsi)", min_value=0.01, max_value=0.5, step=0.01, value=0.1)
        
        optics = OPTICS(
            min_samples=user_min_samples,
            xi=user_xi,
            min_cluster_size=user_min_cluster_size
        )
        optics = OPTICS(min_samples=user_min_samples, xi=user_xi, min_cluster_size=user_min_cluster_size)
        optics.fit(X_std)

        labels_op = optics.labels_
        ordering = optics.ordering_
        reachability = optics.reachability_

        n_clusters = len(set(labels_op)) - (1 if -1 in labels_op else 0)
        n_noise = np.sum(labels_op == -1)
        valid_mask = labels_op != -1
        actual_silhouette = silhouette_score(X_std[valid_mask], labels_op[valid_mask]) if valid_mask.sum() >= 2 else None

        # ===============================
        # PHASE 5: REACHABILITY PLOT (ADVANCED VERSION)
        # ===============================
        st.subheader("📉 Reachability Plot (Peaks & Valleys Analysis)")
        
        # 5.1 Clean and prepare reachability
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
        
        # 5.2 Mapping warna
        unique_labels_ordered = np.unique(labels_ordered)
        cluster_labels = sorted(label for label in unique_labels_ordered if label != -1)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        label_to_color = {-1: 'black'}
        for i, label in enumerate(cluster_labels):
            label_to_color[label] = colors[i % 10]
        
        # 5.3 Deteksi peaks dan valleys
        def detect_peaks_valleys(data, prominence_factor=0.1):
            from scipy.signal import find_peaks
            data_range = np.max(data) - np.min(data)
            prominence = data_range * prominence_factor
            peaks, peak_props = find_peaks(data, prominence=prominence, distance=5)
            valleys, valley_props = find_peaks(-data, prominence=prominence, distance=5)
            return peaks, valleys, peak_props, valley_props
        
        peaks, valleys, peak_props, valley_props = detect_peaks_valleys(reachability_ordered, prominence_factor=0.15)
        
        # 5.4 Visualisasi
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(space, reachability_ordered, 'k-', linewidth=1, alpha=0.3, label='Reachability Profile')
        
        # Plot masing-masing cluster
        for label in sorted(label_to_color.keys()):
            mask = labels_ordered == label
            color = label_to_color[label]
            label_text = 'Noise' if label == -1 else f'Cluster {label}'
            ax.scatter(space[mask], reachability_ordered[mask],
                       c=[color], s=40, alpha=0.9,
                       label=label_text, edgecolors='black', linewidths=0.3)
        
        # Plot peaks
        if len(peaks) > 0:
            ax.scatter(space[peaks], reachability_ordered[peaks], marker='^', s=100, c='red', alpha=0.8,
                       label=f'Peaks ({len(peaks)}) - Cluster Boundaries',
                       edgecolors='darkred', linewidth=2, zorder=5)
            for i, idx in enumerate(peaks):
                ax.annotate(f'P{i+1}', xy=(space[idx], reachability_ordered[idx]),
                            xytext=(5, 10), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='red',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Plot valleys
        if len(valleys) > 0:
            ax.scatter(space[valleys], reachability_ordered[valleys], marker='v', s=100, c='blue', alpha=0.8,
                       label=f'Valleys ({len(valleys)}) - Cluster Cores',
                       edgecolors='darkblue', linewidth=2, zorder=5)
            for i, idx in enumerate(valleys):
                ax.annotate(f'V{i+1}', xy=(space[idx], reachability_ordered[idx]),
                            xytext=(5, -15), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='blue',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # 5.5 Tambahan visualisasi: xi threshold & boundaries
        if len(finite_reach) > 0:
            xi_threshold = np.percentile(finite_reach, 85)
            ax.axhline(y=xi_threshold, color='orange', linestyle='--', alpha=0.7,
                       linewidth=2, label=f'Xi Threshold (≈{xi_threshold:.3f})')
            ax.axhspan(0, xi_threshold, alpha=0.1, color='green', label='Cluster Extraction Zone')
        
        # Tambah garis batas antar cluster
        cluster_boundaries = []
        for i in range(len(unique_labels_ordered) - 1):
            current_label = unique_labels_ordered[i]
            next_label = unique_labels_ordered[i + 1]
            for j in range(len(labels_ordered) - 1):
                if labels_ordered[j] == current_label and labels_ordered[j + 1] == next_label:
                    cluster_boundaries.append(j + 0.5)
                    break
        
        for boundary in cluster_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Finalisasi plot
        ax.set_title("OPTICS Reachability Plot with Peaks & Valleys Analysis", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Data Point Index (OPTICS Ordering)", fontsize=12)
        ax.set_ylabel("Reachability Distance", fontsize=12)
        ax.legend(title="Cluster", fontsize=9, title_fontsize=10, loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        st.pyplot(fig)


        # ===============================
        # PHASE 6: RINGKASAN HASIL
        # ===============================
        st.header("\U0001F4CC Ringkasan Hasil & Langkah Selanjutnya")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("\U0001F4CA Ringkasan Hasil Akhir")
            st.markdown(f"- Jumlah klaster: **{n_clusters}**")
            st.markdown(f"- Titik noise: **{n_noise}**")
            st.markdown(f"- Cakupan data: **{((len(labels_op) - n_noise) / len(labels_op) * 100):.1f}%**")
            if actual_silhouette:
                st.markdown(f"- Silhouette Score: **{actual_silhouette:.3f}**")
            else:
                st.warning("Silhouette Score tidak dapat dihitung.")

        with col2:
            st.subheader("⚙️ Parameter OPTICS")
            st.markdown(f"- `min_samples`: {user_min_samples}")
            st.markdown(f"- `xi`: {user_xi}")
            st.markdown(f"- `min_cluster_size`: {user_min_cluster_size}")

    else:
        st.warning("Silakan pilih minimal satu fitur terlebih dahulu.")
else:
    st.info("Silakan unggah file Excel terlebih dahulu.")
