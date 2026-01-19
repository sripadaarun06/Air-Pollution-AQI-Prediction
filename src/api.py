import os
from pyexpat import features
from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI(title="Air Pollution AQI Prediction API")
rf_model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("features.pkl")
model = joblib.load("final_model.pkl")
scaler = None
FEATURES = None
IGNORED_FEATURES = ["City", "Date"]

try:
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    if os.path.exists("features.pkl"):
        FEATURES = joblib.load("features.pkl")
except:
    pass
@app.get("/")
def home():
    return {"message": "Air Pollution AQI API is running"}

@app.post("/predict")
def predict_aqi(data: dict):
    try:
        IGNORED_FEATURES = ["City", "Date", "AQI_Bucket"]
        
        for feature in FEATURES:
            if feature in IGNORED_FEATURES:
                continue
            if feature not in data:
                return {"error": f"Missing feature: '{feature}'"}
            input_values = [
                float(data[f])
    for f in FEATURES
    if f not in ["City", "Date", "AQI_Bucket"]
]

        X = [input_values]
        X_scaled = scaler.transform(X)
        print("rf_model type:", type(rf_model))
        pred_value = rf_model.predict(X_scaled)[0]
        prediction = float(pred_value)
        
        if prediction < 50:
            risk = "GOOD"
        elif prediction < 100:
            risk = "MODERATE"
        else:
            risk = "UNHEALTHY" if prediction < 150 else "VERY UNHEALTHY" if prediction < 200 else "HAZARDOUS"

        return {"prediction": prediction, "risk": risk}
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
