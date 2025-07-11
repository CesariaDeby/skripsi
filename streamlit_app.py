# STREAMLIT VERSION OF PHASE 1-5
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
    df['wilayah'] = df['wilayah'].str.upper().str.strip()
    selected_features = st.sidebar.multiselect("Pilih Fitur", df.columns.drop('wilayah'))

    if selected_features:
        st.success("Fitur terpilih: " + ", ".join(selected_features))

        # ===============================
        # PHASE 2: STANDARISASI DATA
        # ===============================
        X = df[selected_features].values
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # ===============================
        # PHASE 3: GRID SEARCH PARAMETER OPTICS
        # ===============================
        st.subheader("Pemilihan Parameter Terbaik (Grid Search)")
        min_samples_list = [2, 3, 5, 7]
        xi_list = [0.03, 0.05, 0.07]
        min_cluster_size_list = [0.05, 0.1, 0.2]

        best_results = []
        for ms in min_samples_list:
            for xi in xi_list:
                for mcs in min_cluster_size_list:
                    optics = OPTICS(min_samples=ms, xi=xi, min_cluster_size=mcs)
                    labels = optics.fit_predict(X_std)
                    valid = labels != -1
                    if valid.sum() < 2 or len(set(labels)) <= 1:
                        continue
                    score = silhouette_score(X_std[valid], labels[valid])
                    best_results.append({
                        'min_samples': ms, 'xi': xi, 'min_cluster_size': mcs,
                        'score': score,
                        'clusters': len(set(labels)) - (1 if -1 in labels else 0),
                        'noise': np.sum(labels == -1)
                    })

        top_results = sorted(best_results, key=lambda x: x['score'], reverse=True)[:5]
        st.write(pd.DataFrame(top_results))

        # ===============================
        # PHASE 4: CLUSTERING DENGAN PARAMETER TERBAIK
        # ===============================
        param = top_results[0]
        optics = OPTICS(min_samples=param['min_samples'], xi=param['xi'], min_cluster_size=param['min_cluster_size'])
        optics.fit(X_std)

        labels_op = optics.labels_
        ordering = optics.ordering_
        reachability = optics.reachability_

        n_clusters = len(set(labels_op)) - (1 if -1 in labels_op else 0)
        n_noise = np.sum(labels_op == -1)
        valid_mask = labels_op != -1
        actual_silhouette = silhouette_score(X_std[valid_mask], labels_op[valid_mask]) if valid_mask.sum() >= 2 else None

        # ===============================
        # PHASE 5: REACHABILITY PLOT
        # ===============================
        st.subheader("Reachability Plot")
        space = np.arange(len(X_std))
        reachability_clean = np.where(np.isinf(reachability), np.nan, reachability)
        max_reach = np.nanmax(reachability_clean)
        reach_plot = np.where(np.isinf(reachability), max_reach * 1.2, reachability)
        reach_ordered = reach_plot[ordering]
        labels_ordered = labels_op[ordering]

        fig, ax = plt.subplots(figsize=(14, 6))
        for klass in np.unique(labels_ordered):
            mask = labels_ordered == klass
            color = 'k' if klass == -1 else plt.cm.tab10(klass % 10)
            label = "Noise" if klass == -1 else f"Cluster {klass}"
            ax.plot(space[mask], reach_ordered[mask], marker='.', linestyle='', ms=5, label=label)
        ax.set_ylabel("Reachability Distance")
        ax.set_title("Reachability Plot (OPTICS)")
        ax.legend()
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
            st.markdown(f"- `min_samples`: {param['min_samples']}")
            st.markdown(f"- `xi`: {param['xi']}")
            st.markdown(f"- `min_cluster_size`: {param['min_cluster_size']}")

    else:
        st.warning("Silakan pilih minimal satu fitur terlebih dahulu.")
else:
    st.info("Silakan unggah file Excel terlebih dahulu.")
