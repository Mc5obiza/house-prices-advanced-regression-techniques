from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("House Price Prediction Frontend")
st.caption("Upload a CSV file and get predictions from the FastAPI backend.")

with st.sidebar:
    st.header("Backend Settings")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL).strip().rstrip("/")

    if st.button("Check Backend Health"):
        try:
            response = requests.get(f"{api_base_url}/health", timeout=10)
            if response.status_code == 200:
                st.success(f"Backend reachable: {response.json()}")
            else:
                st.error(f"Health check failed ({response.status_code}): {response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach backend: {exc}")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        input_df = pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        st.stop()

    st.subheader("Input Preview")
    st.dataframe(input_df.head(20), use_container_width=True)
    st.write(f"Rows: {len(input_df)}")

    if st.button("Run Prediction", type="primary"):
        try:
            files = {"file": (uploaded_file.name, uploaded_bytes, "text/csv")}
            response = requests.post(f"{api_base_url}/predict", files=files, timeout=120)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        if response.status_code != 200:
            st.error(f"Prediction failed ({response.status_code}): {response.text}")
            st.stop()

        try:
            payload = response.json()
            prediction_df = pd.DataFrame(payload.get("predictions", []))
        except Exception as exc:
            st.error(f"Invalid response from backend: {exc}")
            st.stop()

        st.subheader("Predictions")
        st.dataframe(prediction_df, use_container_width=True)
        st.write(f"Predicted rows: {payload.get('rows', len(prediction_df))}")

        csv_bytes = prediction_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

        if "SalePrice" in prediction_df.columns:
            st.subheader("Prediction Distribution")
            st.bar_chart(prediction_df["SalePrice"])
