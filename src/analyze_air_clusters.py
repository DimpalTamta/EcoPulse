from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    project_root / "data/processed/air_clusters.csv"
)

features = [
    "PM2.5", "PM10", "NO", "NO2",
    "NOx", "NH3", "CO", "SO2", "O3"
]

cluster_summary = (
    df.groupby("Pollution_Cluster")[features]
    .mean()
    .round(2)
)

cluster_summary.insert(
    0,
    "Records",
    df.groupby("Pollution_Cluster").size()
)

print("\nK-MEANS CLUSTER ANALYSIS")
print(cluster_summary.to_string())

output_dir = project_root / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

cluster_summary.to_csv(
    output_dir / "air_cluster_summary.csv"
)

print("\nCluster summary saved successfully!")