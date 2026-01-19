import requests

url = "http://127.0.0.1:8000/predict"
data = {
  "City": "Delhi",
  "PM2.5": 120,
  "PM10": 180,
  "NO": 15,
  "NO2": 40,
  "NOx": 55,
  "NH3": 8,
  "CO": 1.4,
  "SO2": 12,
  "O3": 30,
  "Benzene": 3,
  "Toluene": 5,
  "Xylene": 2
}


response = requests.post(url, json=data)
print(response.json())
