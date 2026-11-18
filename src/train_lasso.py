from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

project_root = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    project_root / "data/processed/air_pm25.csv",
    parse_dates=["Date"]
)

features = ["City", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
target = "PM2.5"

df = df.sort_values("Date")

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
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
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
    ("regressor", Lasso(alpha=1.0, max_iter=10000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("LASSO REGRESSION RESULTS")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

joblib.dump(
    model,
    project_root / "models/air_lasso_regression.pkl"
)

print("\nLasso model saved successfully!")