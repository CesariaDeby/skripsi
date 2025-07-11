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
from shapely.geometry import Point
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Clustering Perceraian", page_icon="💔")
st.title("💔 Clustering Faktor Perceraian di Kabupaten/Kota")

uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

if uploaded_file:
    # ================================
    # 1. Load dan Pra-pemrosesan Data
    # ================================
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

    # ================================
    # 2. Visualisasi Distribusi Awal
    # ================================
    st.subheader("Distribusi Proporsi Tiap Faktor")
    fig, ax = plt.subplots(1, len(selected_features), figsize=(20, 4))
    for i, col in enumerate(selected_features):
        sns.histplot(df_prop[col], kde=True, bins=20, ax=ax[i])
        ax[i].set_title(col)
    st.pyplot(fig)

    # ================================
    # 3. Winsorization dan Boxplot
    # ================================
    Q1 = df_prop.quantile(0.25)
    Q3 = df_prop.quantile(0.75)
    IQR = Q3 - Q1
    for col in selected_features:
        lower = Q1[col] - 1.5 * IQR[col]
        upper = Q3[col] + 1.5 * IQR[col]
        df_prop[col] = np.clip(df_prop[col], lower, upper)

    # ================================
    # 4. Korelasi Pearson
    # ================================
    st.subheader("Heatmap Korelasi dan Korelasi Pearson")
    corr = df_prop.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.write("Matriks Korelasi Pearson antar Faktor:")
    st.dataframe(corr.round(2))

    # ================================
    # 5. Standarisasi Data
    # ================================
    X = df_prop[selected_features].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # ================================
    # 6. Grid Search Parameter OPTICS
    # ================================
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

    # Gunakan parameter terbaik pertama
    param = top_results[0]
    optics = OPTICS(min_samples=param['min_samples'], xi=param['xi'], min_cluster_size=param['min_cluster_size'])
    optics.fit(X_std)
    labels_op = optics.labels_
    ordering = optics.ordering_
    reachability = optics.reachability_

    # ================================
    # 7. Reachability Plot
    # ================================
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

    # ================================
    # 8. Evaluasi Clustering
    # ================================
    st.subheader("Evaluasi Clustering")
    valid_mask = labels_op != -1
    if valid_mask.sum() >= 2:
        sil = silhouette_score(X_std[valid_mask], labels_op[valid_mask])
        dbi = davies_bouldin_score(X_std[valid_mask], labels_op[valid_mask])
        st.write(f"Silhouette Score: {sil:.3f}")
        st.write(f"Davies-Bouldin Index: {dbi:.3f}")
    else:
        st.warning("Tidak cukup titik valid untuk evaluasi clustering.")

else:
    st.info("Silakan unggah file Excel terlebih dahulu.")
# Assumsi: Data telah tersedia di variabel berikut:
# df: DataFrame dengan kolom 'wilayah' dan fitur-fitur
# X_std: Hasil standardisasi dari data numerik
# labels_op: Hasil label klaster dari OPTICS
# selected_features: daftar nama fitur yang digunakan
# n_clusters, n_noise, actual_silhouette, min_samples, xi, min_cluster_size, ordering, reachability, core_distances, cluster_counts tersedia

# ===============================
# STREAMLIT APP STARTS HERE
# ===============================
st.set_page_config(layout="wide")
st.title("Analisis Klaster Perceraian di Jawa Timur dengan OPTICS")

# PHASE 6: SUMMARY & NEXT STEPS
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
    st.markdown(f"- `min_samples`: {min_samples}")
    st.markdown(f"- `xi`: {xi}")
    st.markdown(f"- `min_cluster_size`: {min_cluster_size}")

st.markdown("---")

# PHASE 7: CLUSTER INTERPRETATION
st.header("🔍 Interpretasi Karakteristik Klaster")
df_std = pd.DataFrame(X_std, columns=selected_features)
df_std['cluster'] = labels_op

def analyze_cluster_characteristics(df_std, top_n=3):
    cluster_stats = {}
    unique_labels = df_std['cluster'].unique()
    overall_means = df_std.drop(columns='cluster').mean()

    for label in sorted(unique_labels):
        cluster_data = df_std[df_std['cluster'] == label]
        cluster_means = cluster_data.drop(columns='cluster').mean()

        dominant_features = []
        for feature in cluster_means.index:
            if overall_means[feature] != 0 and cluster_means[feature] > overall_means[feature]:
                ratio = cluster_means[feature] / overall_means[feature]
                dominant_features.append((feature, ratio))

        dominant_features.sort(key=lambda x: x[1], reverse=True)

        if dominant_features:
            top_feature = dominant_features[0][0]
            name_map = {
                'ekonomi': 'Faktor Ekonomi Dominan',
                'perselisihan dan pertengkaran': 'Perselisihan sebagai Faktor Utama',
                'KDRT': 'Kekerasan dalam Rumah Tangga',
                'meninggalkan salah satu pihak': 'Dominansi Faktor Meninggalkan',
                'zina': 'Dominansi Zina / Perselingkuhan'
            }
            cluster_name = name_map.get(top_feature, f'Dominasi {top_feature}')
        else:
            cluster_name = "Tidak Ada Dominansi Jelas"

        cluster_stats[label] = {
            'name': cluster_name,
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(df_std)) * 100
        }

    return cluster_stats

cluster_stats = analyze_cluster_characteristics(df_std)

# PHASE 8: PCA SCATTER PLOT
st.header("\U0001F4CD Visualisasi Scatter Plot (PCA)")
pca = PCA(n_components=2)
X_plot = pca.fit_transform(X_std)

unique_labels = sorted(df_std['cluster'].unique())
cluster_labels = sorted(label for label in unique_labels if label != -1)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
label_to_color = {-1: 'black'}
for i, label in enumerate(cluster_labels):
    label_to_color[label] = colors[i % 10]

fig_pca, ax_pca = plt.subplots(figsize=(12, 7))
for label in unique_labels:
    mask = df_std['cluster'] == label
    color = label_to_color[label]
    label_text = "Noise" if label == -1 else f"Cluster {label}"
    ax_pca.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[color], label=label_text,
                   s=90 if label != -1 else 70, alpha=0.85, edgecolor='black', linewidth=0.5)
ax_pca.set_title("2D PCA Scatter Plot (OPTICS)", fontsize=14)
ax_pca.set_xlabel("PCA Komponen 1")
ax_pca.set_ylabel("PCA Komponen 2")
ax_pca.legend()
ax_pca.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig_pca)
