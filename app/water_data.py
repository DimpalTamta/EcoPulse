"""Shared water-quality helpers for EcoPulse.

* loads the cleaned CPCB river-monitoring data
* removes physically impossible values (typing errors in the raw records)
* derives a simplified CPCB "designated best use" class (A / B / C)

Nothing here draws anything; pages import from this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
WATER_DIR = ROOT / "data" / "processed" / "water"

# --------------------------------------------------------------------------
# Colours / classes
# --------------------------------------------------------------------------

CLASS_ORDER = ["Class A", "Class B", "Class C", "Below Class C"]
CLASS_COLOURS = {
    "Class A": "#1E9E5A",
    "Class B": "#7CB82F",
    "Class C": "#E6B422",
    "Below Class C": "#D9453D",
}
CLASS_MEANING = {
    "Class A": "Drinking water source after disinfection only",
    "Class B": "Outdoor bathing (organised)",
    "Class C": "Drinking water source after conventional treatment",
    "Below Class C": "Does not meet the simple Class A–C criteria",
}
# Plausible physical ranges. Anything outside is treated as a data-entry error.
PLAUSIBLE = {
    "Temperature (°C)": (0, 50),
    "Dissolved Oxygen (mg/L)": (0, 20),
    "pH": (0, 14),
    "Conductivity (µmho/cm)": (0, 60000),
    "BOD (mg/L)": (0, 500),
    "Nitrate N (mg/L)": (0, 500),
    "Fecal Coliform (MPN/100ml)": (0, 1e8),
    "Total Coliform (MPN/100ml)": (0, 1e8),
    "Fecal Streptococci (MPN/100ml)": (0, 1e8),
}
PARAMS = list(PLAUSIBLE.keys())

SHORT = {
    "Temperature (°C)": "Temperature",
    "Dissolved Oxygen (mg/L)": "Dissolved oxygen",
    "pH": "pH",
    "Conductivity (µmho/cm)": "Conductivity",
    "BOD (mg/L)": "BOD",
    "Nitrate N (mg/L)": "Nitrate-N",
    "Fecal Coliform (MPN/100ml)": "Fecal coliform",
    "Total Coliform (MPN/100ml)": "Total coliform",
    "Fecal Streptococci (MPN/100ml)": "Fecal streptococci",
}


STATE_FIX = {
    "Chattisgarh": "Chhattisgarh", "Jharkha Nd": "Jharkhand", "Odis-Ha": "Odisha",
    "Orissa": "Odisha", "Rajastha N": "Rajasthan", "Tamilnadu": "Tamil Nadu",
    "Uttarakh And": "Uttarakhand", "Cpcb, Delhi": "Delhi", "Rd_Bangalore": "Karnataka",
    "Unknown": "Unspecified", "Inter-State": "Inter-State",
}


def _clean_state(name):
    name = " ".join(str(name).split()).title()
    return STATE_FIX.get(name, name)


def _classify(row):
    do, ph, bod, tc = row["DO"], row["pH_mid"], row["BOD"], row["TC"]
    if pd.isna(do) or pd.isna(ph) or pd.isna(bod):
        return np.nan
    tc_ok = lambda lim: pd.isna(tc) or tc <= lim  # noqa: E731
    if do >= 6 and 6.5 <= ph <= 8.5 and bod <= 2 and tc_ok(50):
        return "Class A"
    if do >= 5 and 6.5 <= ph <= 8.5 and bod <= 3 and tc_ok(500):
        return "Class B"
    if do >= 4 and 6.0 <= ph <= 9.0 and bod <= 3 and tc_ok(5000):
        return "Class C"
    return "Below Class C"


@st.cache_data(show_spinner=False)
def load_water():
    """Cleaned station-year records + mid-point columns + screening class."""
    df = pd.read_csv(WATER_DIR / "EcoPulse_Cleaned_Water_Dataset.csv")
    df["State"] = df["State"].fillna("Unspecified").map(_clean_state)
    df = df[df["State"] != "-"].copy()
    df = df[~df["State"].isin(["Unspecified", "Inter-State"])].copy()

    for base, (lo, hi) in PLAUSIBLE.items():
        for side in ("Min", "Max"):
            col = f"{base} {side}"
            if col in df:
                bad = (df[col] < lo) | (df[col] > hi)
                df.loc[bad, col] = np.nan

    # mid-point of the reported min/max for every parameter
    for base in PARAMS:
        lo, hi = f"{base} Min", f"{base} Max"
        if lo in df and hi in df:
            df[SHORT[base]] = df[[lo, hi]].mean(axis=1)

    df["DO"] = df["Dissolved oxygen"]
    df["pH_mid"] = df["pH"]
    df["BOD"] = df["BOD"]
    df["TC"] = df["Total coliform"]
    df["Screening class"] = df.apply(_classify, axis=1)
    df["Year"] = df["Year"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_water_file(name):
    return pd.read_csv(WATER_DIR / name)


def class_counts(df):
    s = df["Screening class"].value_counts().reindex(CLASS_ORDER).fillna(0)
    return s.astype(int)


def share_good(df):
    """% of records meeting Class A, B or C (i.e. usable after treatment)."""
    d = df.dropna(subset=["Screening class"])
    if d.empty:
        return float("nan")
    return float((d["Screening class"] != "Below Class C").mean() * 100)
