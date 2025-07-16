from pathlib import Path
import streamlit as st
import pandas as pd
from octoanalytics.core import plot_forecast, eval_forecast

st.set_page_config(
    page_title='Premium risk calculation',
    layout='wide',
    initial_sidebar_state='collapsed'  # sidebar fermée
)

st.markdown(
    """
    <style>
    /* Fond noir global de l'app */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    /* Fond noir + texte blanc sidebar (fermée mais style conservé) */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Inputs et boutons */
    input, select, textarea, button {
        background-color: #222222 !important;
        color: white !important;
        border-color: #444444 !important;
    }
    /* Titre fichier centré et blanc */
    .file-title {
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🗂️ Premium risk calculation")

uploaded_files = st.file_uploader(
    "1 • Chargez un ou plusieurs fichiers CSV (timestamp, MW)",
    type="csv",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Glissez vos fichiers CSV ou cliquez pour les sélectionner.")
    st.stop()

file_names = [f.name for f in uploaded_files]
file_choice = st.selectbox("2 • Choisissez le fichier", file_names)
file_obj = next(f for f in uploaded_files if f.name == file_choice)

try:
    df = pd.read_csv(file_obj, parse_dates=['timestamp'])
except Exception as e:
    st.error(f"Erreur de lecture : {e}")
    st.stop()

expected_cols = {'timestamp', 'MW'}
if set(df.columns) != expected_cols:
    st.error(f"Le fichier doit contenir uniquement les colonnes {expected_cols} "
             f"(colonnes trouvées : {list(df.columns)})")
    st.stop()

file_stem = Path(file_choice).stem
st.markdown(f'<div class="file-title">{file_stem}</div>', unsafe_allow_html=True)

fig = plot_forecast(df, datetime_col='timestamp', target_col='MW')
st.plotly_chart(fig, use_container_width=True, height=700)
