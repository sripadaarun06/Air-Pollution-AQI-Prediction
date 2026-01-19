# Model Comparison Report – AQI Prediction

## Objective
The objective of this study is to predict Air Quality Index (AQI) using multiple regression models and select the most reliable model for deployment in a real-world air quality monitoring system.

---

## Models Evaluated

### 1. Linear Regression
Linear Regression was used as a baseline model to understand linear relationships between pollutant concentrations and AQI.

**Observations:**
- Fast and interpretable
- Underperformed due to non-linear relationships in pollution data

---

### 2. Random Forest Regressor
Random Forest was used to capture non-linear interactions between features using an ensemble of decision trees.

**Observations:**
- Strong performance
- Reduced overfitting compared to linear models
- Good balance between accuracy and stability

---

### 3. XGBoost Regressor
XGBoost was evaluated as an advanced boosting-based ensemble model.

**Observations:**
- High predictive power
- Better handling of complex feature interactions
- Slightly higher computational cost

---

## Evaluation Metrics
Models were compared using:
- R² Score
- Root Mean Squared Error (RMSE)
- Train vs Test performance gap

These metrics ensured both accuracy and generalization.

---

## Final Model Selection
Based on evaluation results, the best-performing model was selected considering:
- Predictive accuracy
- Generalization capability
- Deployment feasibility

The selected model was serialized and integrated into an API for real-time inference.

---

## Conclusion
Ensemble-based regression models significantly outperformed linear approaches for AQI prediction. Model comparison ensured robustness and justified the final deployment choice.