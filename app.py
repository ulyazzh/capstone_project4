import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("Prediksi Tingkat Obesitas")

# --- Load model dan encoder ---
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("model.pkl")

@st.cache(allow_output_mutation=True)
def load_encoder():
    return joblib.load("le_gender.pkl")  # Ganti jika ada banyak encoder

model = load_model()
le_gender = load_encoder()

# --- Upload file CSV ---
uploaded = st.file_uploader("Upload file CSV data pasien", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Data preview:")
    st.dataframe(df.head())

    st.write("Isi input manual:")
    inputs = {}
    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64]:
            default = float(df[col].mean())
            inputs[col] = st.number_input(col, value=default)
        else:
            uniques = df[col].dropna().unique().tolist()
            if not uniques:
                uniques = ["N/A"]
            inputs[col] = st.selectbox(col, uniques)

    X = pd.DataFrame([inputs])

    # --- Encode kolom kategorikal ---
    if 'Gender' in X.columns:
        try:
            X['Gender'] = le_gender.transform(X['Gender'])
        except:
            st.error("Nilai 'Gender' tidak dikenali oleh encoder. Pastikan input sesuai data training.")

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
