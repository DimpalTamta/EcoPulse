from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "data/raw/air/city_day.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

print("Original rows:", len(df))

duplicates = df.duplicated(subset=["City", "Date"]).sum()

print("Duplicate city-date records:", duplicates)

df = df.drop_duplicates(subset=["City", "Date"])

print("Rows after removing duplicates:", len(df))

print("\nMissing values percentage:")

missing_percentage = df.isnull().mean() * 100

print(missing_percentage.round(2))

print("\nPM2.5 available records:")
print(df["PM2.5"].notna().sum())

print("\nPM2.5 available records by city:")
print(
    df.groupby("City")["PM2.5"]
    .count()
    .sort_values(ascending=False)
)

pm25_df = df.dropna(subset=["PM2.5"]).copy()

pm25_df = pm25_df.sort_values(["City", "Date"])

output_path = project_root / "data/processed/air_pm25.csv"

pm25_df.to_csv(output_path, index=False)

print("\nPM2.5 dataset saved!")
print("Rows:", len(pm25_df))
print("File:", output_path)