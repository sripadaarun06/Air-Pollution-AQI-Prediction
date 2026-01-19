import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def train_models(X_train, y_train, models=None):
    if models is None:
        models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    }

    for model in models.values():
        model.fit(X_train, y_train)
    return models
def evaluate_and_save(models, X_train, y_train, X_test, y_test, out_dir="output"):
    records = []
    for name, m in models.items():
        y_pred_train = m.predict(X_train)
        y_pred_test = m.predict(X_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        records.append({
            "model": name,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse
        })
    os.makedirs(out_dir, exist_ok=True)
    metrics_df = pd.DataFrame.from_records(records)
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    best = metrics_df.sort_values("test_r2", ascending=False).iloc[0]
    best_name = best["model"]
    return metrics_df, best_name
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demo models, save metrics and best model")
    parser.add_argument("--full", action="store_true", help="Train on full data instead of subsampling")
    parser.add_argument("--out-dir", default="output", help="Directory to save metrics and models")
    args = parser.parse_args()
    try:
        from preprocessing import load_csv, preprocess_data
    except Exception:
        print("Could not import preprocessing. Make sure preprocessing.py is present.")
        raise
    print("MAIN: loading data...")
    df = load_csv()
    target = "AQI" if "AQI" in df.columns else df.columns[-1]
    print("Using target:", target)
    X, y, cols, scaler = preprocess_data(df, target)
    if X.size == 0:
        print("No numeric features found to train on.")
        raise SystemExit(1)
    X = np.asarray(X)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if not args.full:
        max_rows = 2000
        if X_train.shape[0] > max_rows:
            idx = np.random.RandomState(42).choice(X_train.shape[0], max_rows, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
    print("Training models...")
    try:
        models = train_models(X_train, y_train)
        print("Models trained:", list(models.keys()))
        metrics_df, best_name = evaluate_and_save(models, X_train, y_train, X_test, y_test, out_dir=args.out_dir)
        print("Saved metrics to:", os.path.join(args.out_dir, "metrics.csv"))
        best_model = models[best_name]
        model_dir = os.path.join(args.out_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"best_model_{best_name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"Saved best model ({best_name}) to: {model_path}")
    except Exception as e:
        print("Error training models:", e)