from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "data/raw/air/city_day.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCities:", df["City"].nunique())
print("Date range:", df["Date"].min(), "to", df["Date"].max())