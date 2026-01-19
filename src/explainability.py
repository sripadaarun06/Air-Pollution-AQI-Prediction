import shap
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def explain_model(model, X_train, feature_names):
    X_sample = X_train[:1000]
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.savefig("output/shap_summary.png")
    plt.close()


if __name__ == "__main__":
    print("explainability.py loaded. This module defines `explain_model(model, X_train, feature_names)`.")
    print("Use it to generate SHAP summary plots for model interpretability.")
