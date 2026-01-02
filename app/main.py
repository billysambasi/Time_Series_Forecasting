from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Bitcoin Price Forecast API")

# Load trained model once at startup
try:
    model = joblib.load("model.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define input schema with Pydantic
class Features(BaseModel):
    Open: float
    High: float
    Low: float
    Volume: float
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float
    lag_7: float

@app.get("/", summary="Health check", description="Returns API status")
def home():
    return {"message": "Bitcoin Time Series Forecast API is running"}

@app.post("/predict", summary="Predict closing price", description="Provide features to get forecasted close price")
def predict(features: Features):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        df = pd.DataFrame([features.dict()])
        prediction = model.predict(df)
        logging.info(f"Prediction request: {features.dict()} -> {prediction[0]}")
        return {"predicted_close": float(prediction[0])}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/predict_batch", summary="Batch prediction", description="Submit multiple rows of features for batch forecasts")
def predict_batch(batch: List[Features]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        df = pd.DataFrame([item.dict() for item in batch])
        predictions = model.predict(df)
        logging.info(f"Batch prediction request: {len(batch)} rows -> {predictions.tolist()}")
        return {"predicted_close": [float(p) for p in predictions]}
    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")