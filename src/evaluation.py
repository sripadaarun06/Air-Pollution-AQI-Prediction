from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_models(models, X_test, y_test):
    Results = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        Results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        }
    return Results
def print_evaluation_results(results):
    """Pretty-print the evaluation results dictionary produced by evaluate_models."""
    if not results:
        print("No results to display.")
        return
    for name, metrics in results.items():
        print(f"Model: {name}")
        for k, v in metrics.items():
            try:
                print(f"  {k}: {v:.4f}")
            except Exception:
                print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    print("evaluation.py loaded. This module defines `evaluate_models(models, X_test, y_test)`.")
    print("To display results when you call it, pass the returned dict to `print_evaluation_results(results)`).")