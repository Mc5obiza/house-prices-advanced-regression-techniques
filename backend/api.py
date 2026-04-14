from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

try:
    from .xgb_pipeline import MODEL_OUTPUT_PATH, load_model
except ImportError:
    from xgb_pipeline import MODEL_OUTPUT_PATH, load_model

app = FastAPI(
    title="House Pricing Predictor API",
    version="1.0.0",
    description="Upload a CSV file and get SalePrice predictions using the trained XGBoost pipeline.",
)


MODEL = None
MODEL_LOAD_ERROR = None

try:
    MODEL = load_model(MODEL_OUTPUT_PATH)
except Exception as exc:  # pragma: no cover
    MODEL_LOAD_ERROR = str(exc)

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the House Pricing Predictor API. Use /predict to get predictions."}
@app.get("/health")
def health() -> dict[str, str]:
    if MODEL is None:
        return {"status": "error", "detail": f"Model failed to load: {MODEL_LOAD_ERROR}"}
    return {"status": "ok", "detail": "Model loaded"}


@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, object]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {MODEL_LOAD_ERROR}")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        file_bytes = await file.read()
        input_df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    if input_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    features_df = input_df.drop(columns=["SalePrice"], errors="ignore")

    try:
        pred_log = MODEL.predict(features_df)
        pred = np.expm1(pred_log)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    output_df = pd.DataFrame({"SalePrice": pred})
    if "Id" in input_df.columns:
        output_df.insert(0, "Id", input_df["Id"].tolist())

    return {
        "rows": int(len(output_df)),
        "predictions": output_df.to_dict(orient="records"),
    }
