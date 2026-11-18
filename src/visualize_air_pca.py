from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]

pca_df = pd.read_csv(
    project_root / "data/processed/air_pca.csv"
)

cluster_df = pd.read_csv(
    project_root / "data/processed/air_clusters.csv"
)

if len(pca_df) != len(cluster_df):
    raise ValueError("PCA and cluster datasets have different row counts.")

pca_df["Pollution_Cluster"] = cluster_df["Pollution_Cluster"].values

output_dir = project_root / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(11, 7))

colors = {
    0: "#328B9B",
    1: "#C86B42",
    2: "#8B5CF6"
}

labels = {
    0: "Lower Pollution Profile",
    1: "High Particulate Pollution",
    2: "Elevated Gaseous Pollution"
}

for cluster in sorted(pca_df["Pollution_Cluster"].unique()):

    subset = pca_df[
        pca_df["Pollution_Cluster"] == cluster
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=8,
        alpha=0.35,
        color=colors[cluster],
        label=labels[cluster]
    )

plt.title("PCA Visualization of Air Pollution Clusters")
plt.xlabel("Principal Component 1 (37.14% variance)")
plt.ylabel("Principal Component 2 (16.43% variance)")

plt.legend(markerscale=3)
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    output_dir / "air_pca_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPCA cluster visualization saved successfully!")