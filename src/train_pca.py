from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    project_root / "data/processed/air_pm25.csv"
)

features = [
    "PM2.5", "PM10", "NO", "NO2",
    "NOx", "NH3", "CO", "SO2", "O3"
]

X = df[features]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))
])

X_reduced = model.fit_transform(X)

pca = model.named_steps["pca"]

print("PCA RESULTS")
print("Original features:", X.shape[1])
print("Reduced components:", X_reduced.shape[1])

print("\nExplained variance ratio:")
for i, variance in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {variance:.4f}")

print(
    "\nTotal explained variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

component_columns = [
    f"PC{i}" for i in range(1, X_reduced.shape[1] + 1)
]

pca_df = pd.DataFrame(
    X_reduced,
    columns=component_columns
)

pca_df.insert(0, "City", df["City"].values)
pca_df.insert(1, "Date", df["Date"].values)

pca_df.to_csv(
    project_root / "data/processed/air_pca.csv",
    index=False
)

joblib.dump(
    model,
    project_root / "models/air_pca.pkl"
)

print("\nPCA model saved successfully!")