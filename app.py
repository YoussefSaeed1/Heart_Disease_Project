# --- Heart Disease Risk Predictor App ---
import os
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient attributes to predict heart disease risk using the trained model.")

# --- 1. Load trained model ---
MODEL_PATH = os.path.join("..", "models", "final_model.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found at ../models/final_model.pkl.\nPlease train and export the model first (run your notebook).")
    st.stop()

model = load(MODEL_PATH)

# --- 2. Column names (same as in notebooks) ---
# Use same feature set used for training
DATA_PATH = os.path.join("..", "data", "heart_disease.data")
if os.path.exists(DATA_PATH):
    # Read .data without headers and assign names as done in notebooks
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","num"
    ]
    # Create binary target like notebooks and drop 'num'
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    X_cols = [c for c in df.columns if c != "target"]
else:
    # fallback if data is missing
    X_cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
              "thalach","exang","oldpeak","slope","ca","thal"]

# --- 3. Build input form ---
inputs = {}
with st.form("heart_form"):
    st.subheader("Patient Inputs")
    for c in X_cols:
        # Use selectboxes for categorical-like fields for safer input
        if c in ["sex"]:
            val = st.selectbox(f"{c} (0=female,1=male)", [0,1])
        elif c in ["cp"]:
            val = st.selectbox(f"{c} (chest pain type 0-3)", [0,1,2,3])
        elif c in ["fbs","restecg","exang","slope","ca","thal"]:
            val = st.selectbox(f"{c}", [0,1,2,3])
        else:
            val = st.number_input(f"{c}", value=0.0, step=1.0)
        inputs[c] = val
    submitted = st.form_submit_button("Predict")

# --- 4. Make prediction ---
if submitted:
    X_row = pd.DataFrame([inputs])
    try:
        proba = model.predict_proba(X_row)[:,1][0]
    except AttributeError:
        st.error("Loaded model does not support predict_proba. Retrain using a classifier with probability estimates.")
        st.stop()

    pred = int(proba >= 0.5)
    st.metric("Predicted Probability", f"{proba:.3f}")
    st.write("**Prediction:**", "❤️ **Heart Disease**" if pred==1 else "✅ **No Heart Disease**")
