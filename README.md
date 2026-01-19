# Air Pollution AQI Prediction API

An end-to-end machine learning project that predicts **Air Quality Index (AQI)** from air pollution data and exposes the prediction through a **production-ready FastAPI REST API** with explainability and model comparison.

---

## 🚀 Project Overview

Air pollution poses serious health risks, especially in urban areas. Most monitoring systems provide only descriptive insights.  
This project builds a **predictive system** that:

- Predicts AQI using pollutant concentration data
- Classifies pollution risk levels (GOOD → HAZARDOUS)
- Provides real-time predictions via an API
- Explains model behavior using SHAP

The system is designed to be **modular, scalable, and deployment-ready**.

---

## 🧠 Features

- Automated data preprocessing & feature engineering
- Multiple model comparison (Linear, Random Forest, Gradient Boosting, XGBoost)
- Best model selection based on evaluation metrics
- Explainable AI using SHAP
- FastAPI-based REST API
- Swagger (OpenAPI) documentation
- Automated API testing scripts

---

## 🗂️ Project Structure

Air-Pollution-AQI-Prediction/
├── data/
│ └── city_day.csv
├── models/
│ ├── final_model.pkl
│ ├── scaler.pkl
│ └── features.pkl
├── output/
│ ├── shap_summary.png
│ └── Figure_1.png
├── src/
│ ├── api.py
│ ├── main.py
│ ├── main_fast.py
│ ├── preprocessing.py
│ ├── rf_model.py
│ ├── evaluation.py
│ ├── explainability.py
│ ├── inference.py
│ └── eda.py
├── tests/
│ ├── test.py
│ └── test_api_new.py
├── DOCUMENTATION.md
├── model_comparison.md
├── API_README.md
├── README.md
├── requirements.txt
└── .gitignore

---

## 📊 Data Description

The dataset contains daily air pollution measurements across Indian cities.

**Key features include:**
- PM2.5, PM10
- NO, NO2, NOx
- NH3, CO, SO2, O3
- Benzene, Toluene, Xylene

**Target:**
- AQI (Air Quality Index)

Non-numeric fields such as `City` and `Date` are treated as metadata and excluded from modeling.

---

## 🧪 Model Training & Evaluation

### Models evaluated:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### Metrics used:
- R² Score
- RMSE
- MAE

**XGBoost** achieved the best generalization performance and was selected for deployment.

---

## 🔍 Explainability

SHAP (SHapley Additive exPlanations) is used to:
- Identify dominant pollution contributors
- Improve transparency and trust
- Support risk-aware decision making

---

## 🌐 API Usage

### Start the API
```bash
uvicorn src.api:app --reload
```

---

## Swagger (OpenAPI) UI

### Open in browser:
- http://127.0.0.1:8000/docs

## API Usage Example

### Request (POST /predict)
{
  "PM2.5": 180,
  "PM10": 250,
  "NO": 60,
  "NO2": 80,
  "NOx": 120,
  "NH3": 35,
  "CO": 2.1,
  "SO2": 40,
  "O3": 60,
  "Benzene": 4.2,
  "Toluene": 6.5,
  "Xylene": 2.1
}

---

## Installation

- pip install -r requirements.txt
