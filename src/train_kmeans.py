from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
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
    ("kmeans", KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ))
])

clusters = model.fit_predict(X)

df["Pollution_Cluster"] = clusters

print("K-MEANS CLUSTERING RESULTS")
print("Total records:", len(df))
print("Number of clusters:", len(np.unique(clusters)))

print("\nCluster distribution:")
print(df["Pollution_Cluster"].value_counts().sort_index())

sample_size = min(5000, len(df))

rng = np.random.default_rng(42)
sample_indices = rng.choice(
    len(df),
    size=sample_size,
    replace=False
)

X_transformed = model[:-1].transform(X)

score = silhouette_score(
    X_transformed[sample_indices],
    clusters[sample_indices]
)

print("\nSilhouette Score:", round(score, 4))

df.to_csv(
    project_root / "data/processed/air_clusters.csv",
    index=False
)

joblib.dump(
    model,
    project_root / "models/air_kmeans.pkl"
)

print("\nK-Means model saved successfully!")