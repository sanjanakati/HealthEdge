"""HealthEdge - Streamlit interface. All inference runs locally (Edge AI).

Run:  streamlit run app.py
"""
import os
import numpy as np
import joblib
import streamlit as st

MODEL_PATH = "healthedge_model.joblib"

st.set_page_config(page_title="HealthEdge", page_icon="🩺")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)  # loaded once, reused for every reading


st.title("HealthEdge - Vital-Sign Anomaly Detection")
st.caption("Edge AI prototype: the model runs locally on this device. "
           "Educational use only - not a medical diagnosis.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()

bpm = st.number_input("Pulse rate (BPM)", 20.0, 220.0, 76.0)
temp = st.number_input("Temperature (C)", 30.0, 43.0, 36.8)

if st.button("Check Vitals"):
    x = np.array([[bpm, temp]])
    score = model.decision_function(x)[0]
    if model.predict(x)[0] == -1:
        st.error(f"ALERT: Anomaly detected (score {score:.3f})")
    else:
        st.success(f"Normal (score {score:.3f})")
    st.caption("Processed locally. Educational prototype, not a diagnosis.")
