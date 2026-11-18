from pathlib import Path
import joblib
import pandas as pd

project_root = Path(__file__).resolve().parents[1]

model = joblib.load(
    project_root / "models/air_random_forest.pkl"
)

sample = pd.DataFrame([{
    "City": "Delhi",
    "PM10": 180,
    "NO": 25,
    "NO2": 40,
    "NOx": 55,
    "NH3": 30,
    "CO": 1.5,
    "SO2": 15,
    "O3": 35
}])

prediction = model.predict(sample)

print("ECOPULSE - AIR PREDICTION TEST")
print("City:", sample["City"].iloc[0])
print("Predicted PM2.5:", round(prediction[0], 2), "µg/m³")
print("\nPrediction completed successfully!")