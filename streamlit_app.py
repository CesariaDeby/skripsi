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

st.set_page_config(layout="wide", page_title="Manual OPTICS Clustering", page_icon="💔")
st.title("💔 Clustering Faktor Perceraian - Pilih Parameter OPTICS Sendiri")

uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

if uploaded_file:
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

    selected_features = [
        'perselisihan dan pertengkaran',
        'ekonomi',
        'KDRT',
        'meninggalkan salah satu pihak',
        'zina'
    ]

    df = df[df['Jumlah Cerai'] > 0].copy()
    df.fillna(0, inplace=True)

    # Hitung proporsi
    for col in selected_features:
        df[col] = df[col] / df['Jumlah Cerai']

    df_prop = df[selected_features].copy()

    st.subheader("📌 Pilih Parameter OPTICS")
    with st.sidebar:
        st.markdown("### ⚙️ Parameter OPTICS")
        min_samples = st.slider("min_samples", min_value=2, max_value=10, value=5)
        xi = st.slider("xi", min_value=0.01, max_value=0.2, step=0.01, value=0.05)
        min_cluster_size = st.slider("min_cluster_size (proporsi)", min_value=0.01, max_value=0.5, step=0.01, value=0.1)

    # Standardisasi
    X = df_prop.values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Clustering dengan OPTICS berdasarkan input user
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    optics.fit(X_std)

    labels_op = optics.labels_
    ordering = optics.ordering_
    reachability = optics.reachability_

    n_clusters = len(set(labels_op)) - (1 if -1 in labels_op else 0)
    n_noise = np.sum(labels_op == -1)
    valid_mask = labels_op != -1
    silhouette = silhouette_score(X_std[valid_mask], labels_op[valid_mask]) if valid_mask.sum() >= 2 else None

    # Visualisasi Reachability Plot
    st.subheader("📈 Reachability Plot")
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
    ax.set_title("Reachability Plot (OPTICS Manual)")
    ax.legend()
    st.pyplot(fig)

    # Ringkasan Hasil
    st.subheader("📋 Ringkasan Hasil")
    st.markdown(f"- Jumlah klaster: **{n_clusters}**")
    st.markdown(f"- Jumlah noise: **{n_noise}**")
    st.markdown(f"- Cakupan data: **{((len(labels_op) - n_noise) / len(labels_op) * 100):.1f}%**")
    if silhouette:
        st.markdown(f"- Silhouette Score: **{silhouette:.3f}**")
    else:
        st.warning("Silhouette Score tidak dapat dihitung (jumlah cluster < 2)")

else:
    st.info("Silakan unggah file Excel terlebih dahulu.")
