import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from rf_model import train_models
from evaluation import evaluate_models
from inference import predict_pollution
import joblib
import numpy as np

TARGET = "AQI"

print("Loading data...")
df = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/Data Analyst/Air-Pollution-AQI-Prediction/data/city_day.csv")
print(f"Data loaded: {df.shape}")
df_sample = df.sample(n=min(5000, len(df)), random_state=42)
print(f"Sampled data: {df_sample.shape}")
X, y, feature_names, scaler = preprocess_data(df_sample, TARGET)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training models...")
Models = train_models(X_train, y_train)
print("Models trained:", list(Models.keys()))

Results = evaluate_models(Models, X_test, y_test)
print("\nModel Evaluation Results:")
for model, metrics in Results.items():
    print(f"  {model}:")
    for key, val in metrics.items():
        print(f"    {key}: {val:.4f}")

print("\nBest model: XGBoost")
Best_model = Models["XGBoost"]

joblib.dump(Best_model, "final_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "features.pkl")

Sample_input = X_test[0]
Output = predict_pollution(Best_model, Sample_input)
print(f"Inference Output: {Output}")

print("\nDone!")
