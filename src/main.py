print("Starting main.py")
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from rf_model import train_models
from evaluation import evaluate_models
from explainability import explain_model
from inference import predict_pollution
import joblib

TARGET = "AQI"

print("Loading data...")
df = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/Data Analyst/Air-Pollution-AQI-Prediction/data/city_day.csv")
print(f"Data loaded: {df.shape}")
X, y, feature_names, scaler = preprocess_data(df, TARGET)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
joblib.dump(feature_names, "features.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training features:", list(feature_names))

Models = train_models(X_train, y_train)

print("Models keys:", list(Models.keys()))
print("XGBoost type:", type(Models["XGBoost"]))

Best_model = Models["XGBoost"]

print("Best_model type:", type(Best_model))

joblib.dump(Best_model, "final_model.pkl")

# explain_model(Best_model, X_train, feature_names)

Sample_input = X_test[0]
Output = predict_pollution(Best_model, Sample_input)
print("Inference Output:", Output)
print("Done!")