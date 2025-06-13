
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("Prediksi Tingkat Obesitas")

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("model.pkl")

model = load_model()

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
            uniques = df[col].unique().tolist()
            inputs[col] = st.selectbox(col, uniques)

    X = pd.DataFrame([inputs])
    st.write("Input untuk prediksi:")
    st.json(inputs)

    if st.button("Prediksi"):
        yhat = model.predict(X)[0]
        st.success(f"Prediksi obesitas: **{yhat}**")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            st.write("Probabilitas per kelas:")
            st.json(dict(zip(model.classes_, [float(p) for p in probs])))
