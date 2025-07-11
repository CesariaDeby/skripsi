import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Clustering Perceraian", page_icon="💔")

# 1. Judul dan Upload Data
st.title("💔 Aplikasi Pengelompokan Kabupaten/Kota Jawa Timur Berdasarkan Faktor Penyebab Perceraian")
st.markdown("### Upload Dataset")

uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
