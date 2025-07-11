import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Clustering Perceraian", page_icon="💔")

# ================================
# 1. Judul dan Upload Data
# ================================
st.title("💔 Aplikasi Pengelompokan Kabupaten/Kota Jawa Timur Berdasarkan Faktor Penyebab Perceraian")
st.markdown("### Upload Dataset")

uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

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

else:
    st.info("Silakan unggah file Excel (.xlsx) yang berisi data perceraian.")

