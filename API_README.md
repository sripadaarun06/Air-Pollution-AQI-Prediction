"""
API Test Summary
================

✅ API Module: api.py
- Loads trained model: final_model.pkl (RandomForest)
- Framework: FastAPI
- Port: 8000 (or 8001 if 8000 is busy)

Endpoints:
----------
1. GET /
   - Health check endpoint
   - Returns: {"message": "Air Pollution AQI API is running"}

2. POST /predict
   - Accepts pollution measurements as JSON dictionary
   - Expects 12-13 feature values
   - Returns: {"Predicted_AQI": float, "Risk_Level": str}
   
   Example Input:
   {
       "PM2_5": 120,
       "PM10": 180,
       "NO": 25,
       "NO2": 40,
       "NOx": 50,
       "NH3": 10,
       "CO": 1.4,
       "SO2": 12,
       "O3": 30,
       "Benzene": 5,
       "Toluene": 8,
       "Xylene": 3
   }
   
   Example Output:
   {
       "Predicted_AQI": 72.51,
       "Risk_Level": "MODERATE"
   }

Risk Levels:
- GOOD: AQI < 50
- MODERATE: 50 <= AQI < 100
- POOR: 100 <= AQI < 200
- SEVERE: AQI >= 200

To Run the API:
---------------
uvicorn api:app --host 127.0.0.1 --port 8000

Then test with:
curl http://127.0.0.1:8000/
python test_api_new.py
"""

print(__doc__)
