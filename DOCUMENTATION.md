# Air Pollution AQI Prediction – Project Documentation

---

## 1. Business Problem Definition

Air pollution is a critical environmental and public health issue, especially in urban regions. Traditional air quality monitoring systems are mostly descriptive and do not provide predictive insights for proactive decision-making.

The objective of this project is to build a machine learning–based system that predicts the Air Quality Index (AQI) using pollutant concentration data and classifies the associated health risk level (GOOD, MODERATE, POOR, SEVERE, HAZARDOUS).

This solution enables:
- Early identification of high pollution risk
- Data-driven environmental and health decisions
- Easy integration with applications via an API

---

## 2. Data Source & Schema Explanation

The dataset used in this project contains daily air pollution measurements collected across multiple Indian cities. Each record represents pollutant concentrations for a specific city and date.

### Key features include:
- **PM2.5, PM10** – Particulate matter concentrations
- **NO, NO2, NOx** – Nitrogen-based pollutants
- **NH3** – Ammonia
- **CO** – Carbon monoxide
- **SO2** – Sulfur dioxide
- **O3** – Ozone
- **Benzene, Toluene, Xylene** – Volatile organic compounds

### Target variable:
- **AQI (Air Quality Index)** – A numerical indicator of overall air pollution severity

Non-numeric fields such as **City** and **Date** were treated as metadata and excluded from numerical modeling to prevent categorical leakage and ensure consistent inference.

---

## 3. Feature Engineering Decisions

Feature engineering focused on ensuring numerical consistency, robustness, and alignment between training and inference.

Key decisions:
- Removed non-numeric attributes (City, Date) from model inputs
- Handled missing values using statistical imputation
- Detected and treated outliers using IQR-based filtering
- Applied standard scaling to normalize feature ranges
- Preserved feature order to maintain training–inference consistency

All preprocessing components (scaler and feature list) were serialized to guarantee reliable predictions during deployment.

---

## 4. Model Comparison & Selection Rationale

Multiple regression models were evaluated to ensure fair comparison and robust performance:

- **Linear Regression** – Baseline model
- **Random Forest Regressor** – Bagging-based ensemble model
- **Gradient Boosting Regressor** – Classical boosting approach
- **XGBoost Regressor** – Optimized gradient boosting model

### Evaluation metrics:
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Ensemble-based models significantly outperformed linear regression. XGBoost demonstrated the best generalization performance and was selected as the final deployment model due to its accuracy and stability.

---

## 5. Explainability & Risk Analysis

Model explainability was achieved using SHAP (SHapley Additive exPlanations) to understand feature contributions to AQI predictions.

Key insights:
- PM2.5 and PM10 were the strongest contributors to AQI
- Gaseous pollutants influenced AQI under specific conditions

Risk levels were derived after prediction to avoid target leakage:
- **GOOD:** AQI ≤ 50
- **MODERATE:** 51–100
- **POOR:** 101–200
- **SEVERE:** 201–300
- **HAZARDOUS:** >300

This approach ensures transparent, interpretable, and decision-ready predictions.

---

## 6. Deployment & Lifecycle Plan

The trained model is deployed using a **FastAPI-based REST service**.

### Deployment features:
- Serialized model, scaler, and feature loading
- Input validation and error handling
- Real-time AQI prediction via POST endpoint
- Auto-generated OpenAPI (Swagger) documentation for testing and integration

### Lifecycle considerations:
- Monitoring prediction drift
- Periodic retraining with updated data
- Scalable integration with dashboards and external systems

The architecture is modular and supports future enhancements without major redesign.