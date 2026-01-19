import requests
import json

BASE_URL = "http://127.0.0.1:8000"
print("Test 1: GET /")
try:
    response = requests.get(f"{BASE_URL}/", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
except Exception as e:
    print(f"Error: {e}\n")
print("Test 2: POST /predict")
test_data = {
    "PM2.5": 120,
    "PM10": 180,
    "NO2": 40,
    "SO2": 12,
    "CO": 1.4,
    "O3": 30,
    "NO": 25,
    "NOx": 50,
    "NH3": 10,
    "Benzene": 5,
    "Toluene": 8,
    "Xylene": 3
}
try:
    response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
except Exception as e:
    print(f"Error: {e}\n")

print("✅ API tests completed!")
