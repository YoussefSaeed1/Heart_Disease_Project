"""Heart Disease Risk Predictor (Streamlit).

Run:  streamlit run app.py
The app trains and saves the model on first run if models/heart_disease_model.joblib is missing.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from heart_disease import config  # noqa: E402
from heart_disease.predict import load_model, predict_proba  # noqa: E402
from heart_disease.validate import check_inputs  # noqa: E402

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")
st.caption("Logistic regression trained on the UCI Cleveland dataset (303 patients). For learning only, not a diagnostic tool.")

model = st.cache_resource(load_model)()


def coded(col):
    codes = config.CODES[col]
    return st.selectbox(config.LABELS[col], list(codes), format_func=lambda k: f"{codes[k]} ({k})")


with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input(config.LABELS["age"], 18, 100, 55)
        sex = coded("sex")
        cp = coded("cp")
        trestbps = st.number_input(config.LABELS["trestbps"], 70, 250, 130)
        chol = st.number_input(config.LABELS["chol"], 80, 700, 245)
        fbs = coded("fbs")
        restecg = coded("restecg")
    with c2:
        thalach = st.number_input(config.LABELS["thalach"], 50, 230, 150)
        exang = coded("exang")
        oldpeak = st.number_input(config.LABELS["oldpeak"], 0.0, 10.0, 1.0, step=0.1)
        slope = coded("slope")
        ca = st.selectbox(config.LABELS["ca"], [0, 1, 2, 3])
        thal = coded("thal")
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    row = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
               exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
    for msg in check_inputs(row):
        st.warning(msg)
    p = float(predict_proba([row], model)[0])
    st.metric("Estimated probability of heart disease", f"{p:.0%}")
    if p >= 0.5:
        st.error("Above the 50% threshold: the model would flag this patient for follow-up.")
    else:
        st.success("Below the 50% threshold.")
