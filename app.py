import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("Prediksi Tingkat Obesitas")

# --- Load model ---
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# --- Daftar fitur sesuai saat training model ---
fitur_model = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O',
    'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
]

# --- Upload file CSV ---
uploaded = st.file_uploader("Upload file CSV data pasien", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Data preview:")
    st.dataframe(df.head())

    st.write("Isi input manual (satu pasien):")
    inputs = {}
    for col in fitur_model:
        if col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                default = float(df[col].mean())
                inputs[col] = st.number_input(col, value=default)
            else:
                uniques = df[col].dropna().unique().tolist()
                if not uniques:
                    uniques = ["N/A"]
                inputs[col] = st.selectbox(col, uniques)
        else:
            st.warning(f"Kolom '{col}' tidak ditemukan di CSV.")
            inputs[col] = ""

    X = pd.DataFrame([inputs])
    X = X[fitur_model]
    X = X.fillna(0)

    st.write("Input untuk prediksi:")
    st.json(inputs)

    if st.button("Prediksi"):
        try:
            yhat = model.predict(X)[0]
            st.success(f"Prediksi obesitas: **{yhat}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                st.write("Probabilitas per kelas:")
                st.json(dict(zip(model.classes_, [float(p) for p in probs])))
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
