def predict_pollution(model, input_data):
    prediction = model.predict([input_data])[0]
    if prediction < 50:
        Risk = "GOOD"
    elif prediction < 100:
        Risk = "MODERATE"
    elif prediction < 200:
        Risk = "POOR"
    else:
        Risk = "SEVERE"
    return {
        "Predicted_AQI": round(prediction, 2),
        "Risk_Level": Risk
    }

if __name__ == "__main__":
    print("inference.py loaded. This module defines `predict_pollution(model, input_data)`.")
    print("Returns: {'Predicted_AQI': value, 'Risk_Level': category}")