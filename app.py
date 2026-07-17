import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wafer Fault Detection",
    page_icon="🧇",
    layout="wide"
)

st.title("🧇 Wafer Fault Detection")
st.write("Upload a CSV file for Batch Prediction")

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_objects():
    with open("artifacts/model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("artifacts/scalar.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_objects()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Remove unwanted column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Remove target column if uploaded accidentally
    if "Good/Bad" in df.columns:
        df.drop(columns=["Good/Bad"], inplace=True)

    # Fill missing values
    df.fillna(0, inplace=True)

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(X_scaled)

    # Convert back to original labels
    prediction = pd.Series(prediction).map({
        0: -1,
        1: 1
    })

    result = df.copy()
    result["Prediction"] = prediction

    st.subheader("Prediction Result")
    st.dataframe(result)

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction",
        data=csv,
        file_name="Wafer_Prediction.csv",
        mime="text/csv"
    )