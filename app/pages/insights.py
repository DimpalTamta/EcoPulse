"""Air × Water: combines both datasets at state level."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pages.air_dashboard import load_air_data
from theme import HEAT_SCALE_R, TEAL, callout, html_table, kpi_row, page_header, style_fig
from water_data import load_water, share_good

CITY_STATE = {
    "Ahmedabad": "Gujarat", "Aizawl": "Mizoram", "Amaravati": "Andhra Pradesh",
    "Amritsar": "Punjab", "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Brajrajnagar": "Odisha", "Chandigarh": "Chandigarh", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Delhi": "Delhi", "Ernakulam": "Kerala",
    "Gurugram": "Haryana", "Guwahati": "Assam", "Hyderabad": "Telangana",
    "Jaipur": "Rajasthan", "Jorapokhar": "Jharkhand", "Kochi": "Kerala",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra",
    "Patna": "Bihar", "Shillong": "Meghalaya", "Talcher": "Odisha",
    "Thiruvananthapuram": "Kerala", "Visakhapatnam": "Andhra Pradesh",
}


def _air_score(pm25):
    """100 at/below the WHO guideline (15), 0 at/above 150 µg/m³."""
    return float(np.clip(100 * (1 - (pm25 - 15) / (150 - 15)), 0, 100))


@st.cache_data(show_spinner=False)
def state_table():
    air = load_air_data()
    air["State"] = air["City"].map(CITY_STATE)
    a = air.groupby("State").agg(PM25=("PM2.5", "mean"), Cities=("City", "nunique")).reset_index()
    a["Air score"] = a["PM25"].map(_air_score)

    w = load_water()
    g = w.groupby("State")
    wt = pd.DataFrame({
        "Water records": g.size(),
        "Usable water %": g.apply(share_good),
        "Median BOD": g["BOD"].median(),
        "Median DO": g["Dissolved oxygen"].median(),
    }).reset_index()
    wt = wt[wt["Water records"] >= 10]

    m = a.merge(wt, on="State", how="inner")
    m["EcoPulse score"] = (m["Air score"] + m["Usable water %"]) / 2
    return m.sort_values("EcoPulse score", ascending=False).reset_index(drop=True)


def score_band(s):
    return ("#2C6FB7", "Favourable") if s >= 65 else ("#8FBBE3", "Moderate") if s >= 45 else ("#A81E2D", "Stressed")


def show_insights():
    page_header(
        "AIR × WATER / COMBINED INSIGHTS",
        "Integrated Air–Water Analytics",
        "Where air quality and river water quality move together — and the "
        "EcoPulse Score that combines them for each state.",
    )
    m = state_table()
    if m.empty:
        st.warning("No overlapping states between the two datasets.")
        return

    best, worst = m.iloc[0], m.iloc[-1]
    corr = m["PM25"].corr(m["Usable water %"], method="spearman")
    kpi_row([
        ("States compared", f"{len(m)}", "Have both air and water data", TEAL),
        ("Highest composite score", f"{best['EcoPulse score']:.0f}", best["State"], "#2C6FB7"),
        ("Lowest composite score", f"{worst['EcoPulse score']:.0f}", worst["State"], "#A81E2D"),
        ("Air–water link", f"{corr:+.2f}", "Spearman ρ: PM2.5 vs usable water", "#5B8FC7"),
    ])

    t1, t2, t3 = st.tabs(["Composite Index", "Air–Water Relationship", "State Scorecard"])

    with t1:
        d = m.sort_values("EcoPulse score")
        fig = go.Figure()
        fig.add_bar(y=d["State"], x=d["Air score"] / 2, orientation="h", name="Air sub-index (50%)",
                    marker_color="#1F5FA8", marker_line_width=0)
        fig.add_bar(y=d["State"], x=d["Usable water %"] / 2, orientation="h", name="Water sub-index (50%)",
                    marker_color="#8FBBE3", marker_line_width=0)
        fig.update_layout(barmode="stack", bargap=0.35, legend=dict(orientation="h", y=1.06, x=0))
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(title="EcoPulse Score (0–100)", range=[0, 100])
        st.plotly_chart(style_fig(fig, max(380, 30 * len(d))), use_container_width=True)
        callout(
            "<b>Methodology:</b> Air sub-index = 100 at the WHO guideline (15 µg/m³ PM2.5) falling to 0 at "
            "150 µg/m³. Water score = % of river records meeting at least Class C. "
            "EcoPulse Score = average of the two. It is a simple comparison tool, not an official index.",
            tone=TEAL,
        )

    with t2:
        fig = px.scatter(
            m, x="PM25", y="Usable water %", size="Water records", color="EcoPulse score",
            hover_name="State", text="State", size_max=42,
            color_continuous_scale=HEAT_SCALE_R,
            labels={"PM25": "Average PM2.5 (µg/m³)", "Usable water %": "Water records usable (A–C, %)"},
        )
        fig.update_traces(textposition="top center", textfont=dict(size=11), marker=dict(line=dict(width=1, color="#FFFFFF"), opacity=0.9))
        fig.add_vline(x=m["PM25"].median(), line_dash="dot", line_color="#7A8CA3")
        fig.add_hline(y=m["Usable water %"].median(), line_dash="dot", line_color="#7A8CA3")
        st.plotly_chart(style_fig(fig, 520, legend=False), use_container_width=True)
        quad = m[(m["PM25"] > m["PM25"].median()) & (m["Usable water %"] < m["Usable water %"].median())]
        if len(quad):
            callout("Double burden (above-median PM2.5 <i>and</i> below-median water quality): "
                    + ", ".join(f"<b>{s}</b>" for s in quad["State"]) + ".", tone="#A81E2D")
        st.caption("Air data is from city monitors (2015–2020); water data is from river stations (2012–2023). "
                   "Periods differ, so read this as a pattern, not a cause.")

    with t3:
        show = m[["State", "Cities", "PM25", "Air score", "Water records", "Usable water %",
                  "Median BOD", "Median DO", "EcoPulse score"]].round(1)
        show = show.rename(columns={"PM25": "Avg PM2.5"})
        html_table(show, 60)
        st.download_button("Download state table (CSV)", show.to_csv(index=False).encode(),
                           file_name="ecopulse_state_scores.csv", mime="text/csv")
