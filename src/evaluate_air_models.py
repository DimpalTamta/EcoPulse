from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

project_root = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    project_root / "data/processed/air_pm25.csv",
    parse_dates=["Date"]
)

df = df.sort_values("Date")

features = [
    "City", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3"
]

split = int(len(df) * 0.8)

test_df = df.iloc[split:]

X_test = test_df[features]
y_test = test_df["PM2.5"]

models = {
    "Multiple Linear Regression": "air_linear_regression.pkl",
    "Ridge Regression": "air_ridge_regression.pkl",
    "Lasso Regression": "air_lasso_regression.pkl",
    "Decision Tree": "air_decision_tree.pkl",
    "Random Forest": "air_random_forest.pkl"
}

results = []

for name, filename in models.items():

    model = joblib.load(
        project_root / "models" / filename
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Algorithm": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    })

results_df = pd.DataFrame(results)

print("\nAIR MODEL COMPARISON")
print(results_df.to_string(index=False))

output_dir = project_root / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    project_root / "reports" / "air_model_comparison.csv",
    index=False
)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

metrics = ["MAE", "RMSE", "R2"]

for ax, metric in zip(axes, metrics):

    ax.bar(
        results_df["Algorithm"],
        results_df[metric]
    )

    ax.set_title(f"{metric} Comparison")
    ax.tick_params(axis="x", rotation=65)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

plt.tight_layout()

plt.savefig(
    output_dir / "air_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison CSV and graph saved successfully!")