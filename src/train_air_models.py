from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

project_root = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    project_root / "data/processed/air_pm25.csv",
    parse_dates=["Date"]
)

# Predict PM2.5 from other pollutant measurements
features = ["City", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
target = "PM2.5"

df = df.sort_values("Date")

# Earlier observations for training, later observations for testing
split = int(len(df) * 0.8)

train_df = df.iloc[:split]
test_df = df.iloc[split:]

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df[target]
y_test = test_df[target]

numeric_features = [col for col in features if col != "City"]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "city",
        OneHotEncoder(handle_unknown="ignore"),
        ["City"]
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MULTIPLE LINEAR REGRESSION RESULTS")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

joblib.dump(
    model,
    project_root / "models/air_linear_regression.pkl"
)

print("\nModel saved successfully!")