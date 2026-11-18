from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    donut_3d,
    AQI_BUCKET_COLOURS, AQI_BUCKET_ORDER, HEAT_SCALE_R, INDIA_LIMIT, INK, TEAL,
    clean_html, inject_theme, kpi_row, nav_card, nav_group, palette_picker,
    pm25_band, section_title, sidebar_brand, style_fig,
)
from water_data import CLASS_COLOURS, class_counts, load_water
from pages.water_dashboard import show_water_dashboard
from pages.water_trends import show_water_trends
from pages.water_analytics import show_water_analytics
from pages.air_analytics import show_air_analytics
from pages.insights import show_insights, state_table, score_band
from pages.ml_comparison import show_ml_comparison
from pages.report_center import show_report_center
from pages.air_dashboard import show_air_dashboard
from pages.air_prediction import show_air_prediction
from pages.air_trends import show_air_trends
from pages.air_clusters import show_air_clusters
from pages.pollution_map import show_pollution_map
from pages.pollution_hotspots import render as show_pollution_hotspots


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EcoPulse | Air & Water Intelligence",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]

# One shared stylesheet for the whole app (see theme.py). The Home page also
# gets a very slow, soft animated gradient behind the atlas sections.
inject_theme(ambient=st.session_state.get("page", "Home") == "Home")


@st.cache_data
def load_home_stats():
    df = pd.read_csv(
        ROOT / "data" / "processed" / "air_pm25.csv",
        usecols=["City", "Date", "PM2.5", "AQI_Bucket"],
        parse_dates=["Date"],
    )
    by_city = df.groupby("City")["PM2.5"].mean().sort_values()
    yearly = df.groupby(df["Date"].dt.year)["PM2.5"].mean()
    buckets = df["AQI_Bucket"].value_counts().reindex(AQI_BUCKET_ORDER).fillna(0)

    return {
        "cities": int(df["City"].nunique()),
        "records": int(len(df)),
        "start": int(df["Date"].min().year),
        "end": int(df["Date"].max().year),
        "worst_city": by_city.index[-1],
        "worst_val": float(by_city.iloc[-1]),
        "best_city": by_city.index[0],
        "best_val": float(by_city.iloc[0]),
        "yearly": yearly,
        "buckets": buckets,
    }


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

VALID_PAGES = {
    "Home",
    "Air Dashboard", "Air Prediction", "Air Trends", "Air Clusters", "Air Analytics",
    "Water Dashboard", "Water Trends", "Water Analytics",
    "Insights", "Pollution Map", "Pollution Hotspots",
    "ML Comparison", "Download Report",
}

if (
    "page" not in st.session_state
    or st.session_state.page not in VALID_PAGES
):
    st.session_state.page = "Home"


def navigate(page):
    st.session_state.page = page
    st.rerun()


def sidebar_button(label, page, icon=None):
    active = st.session_state.page == page

    if st.button(
        label,
        icon=icon,
        use_container_width=True,
        type="primary" if active else "secondary",
        key=f"nav_{page}",
    ):
        navigate(page)


with st.sidebar:
    sidebar_brand()

    sidebar_button("Home", "Home", ":material/home:")

    nav_group("Air Quality")
    sidebar_button("Air Dashboard", "Air Dashboard", ":material/air:")
    sidebar_button("PM2.5 Prediction", "Air Prediction", ":material/online_prediction:")
    sidebar_button("Historical Trends", "Air Trends", ":material/show_chart:")
    sidebar_button("Air Analytics", "Air Analytics", ":material/query_stats:")
    sidebar_button("Clusters & PCA", "Air Clusters", ":material/bubble_chart:")

    nav_group("Water Quality")
    sidebar_button("Water Dashboard", "Water Dashboard", ":material/water_drop:")
    sidebar_button("Water Trends & States", "Water Trends", ":material/trending_up:")
    sidebar_button("Water Analytics", "Water Analytics", ":material/analytics:")

    nav_group("Combined Analysis")
    sidebar_button("Integrated Air–Water Analytics", "Insights", ":material/compare_arrows:")
    sidebar_button("India Pollution Map", "Pollution Map", ":material/map:")
    sidebar_button("Pollution Hotspots", "Pollution Hotspots", ":material/local_fire_department:")

    nav_group("Models & Reports")
    sidebar_button("Air ML Comparison", "ML Comparison", ":material/model_training:")
    sidebar_button("Download Centre", "Download Report", ":material/download:")

    nav_group("Appearance")
    palette_picker()

    st.markdown(
        clean_html("""
        <div class="eco-side-note">
            <b>About the data</b>
            Historical CPCB air and river-water records for India.
            Results are estimates, not live readings.
        </div>
        """),
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# HOME DASHBOARD
# --------------------------------------------------

def _home_hero():
    st.markdown(
        clean_html("""
        <div class="ep-header anim">
            <div class="ep-eyebrow">Air &amp; Water Atlas of India</div>
            <div class="ep-title">The air we breathe and the water we use</div>
            <div class="ep-sub">
                Historical PM2.5 in 26 cities and river water quality across
                Indian states, with analytics, machine-learning models and a
                combined EcoPulse Score.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def _home_page():
    stats = load_home_stats()
    water = load_water()
    states = state_table()

    _home_hero()

    _, worst_colour, _ = pm25_band(stats["worst_val"])
    _, best_colour, _ = pm25_band(stats["best_val"])
    kpi_row(
        [
            ("Cities monitored", f"{stats['cities']}",
             f"{stats['records']:,} daily records", TEAL),
            ("Most polluted city", stats["worst_city"],
             f"Average PM2.5 {stats['worst_val']:.0f} µg/m³", worst_colour),
            ("Cleanest city", stats["best_city"],
             f"Average PM2.5 {stats['best_val']:.0f} µg/m³", best_colour),
            ("River stations", f"{water['Station Code'].nunique():,}",
             f"{water['State'].nunique()} states · {water['Year'].min()}–{water['Year'].max()}",
             "#4C5BD4"),
        ]
    )

    # ---------------- air ----------------
    section_title("Air quality", f"PM2.5 in {stats['cities']} cities, {stats['start']}–{stats['end']}")
    g1, g2 = st.columns([1.35, 1], gap="large")
    with g1:
        st.markdown("### Annual Mean PM2.5")
        y = stats["yearly"].reset_index()
        y.columns = ["Year", "PM2.5"]
        fig = px.area(y, x="Year", y="PM2.5", markers=True,
                      color_discrete_sequence=[TEAL],
                      labels={"PM2.5": "PM2.5 (µg/m³)"})
        fig.update_traces(line_width=3, fillcolor="rgba(14,124,134,0.22)",
                          marker=dict(size=9))
        fig.add_hline(y=INDIA_LIMIT, line_dash="dash", line_color="#D9453D",
                      annotation_text=f"India limit {INDIA_LIMIT}",
                      annotation_position="top left")
        fig.update_xaxes(type="category")
        st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)
    with g2:
        st.markdown("### AQI Category Distribution")
        b = stats["buckets"]
        st.plotly_chart(donut_3d(list(b.index), list(b.values), [AQI_BUCKET_COLOURS[x] for x in b.index], 340), use_container_width=True)

    # ---------------- water ----------------
    section_title("River water quality", "CPCB monitoring stations, simple Class A / B / C screening")
    w1, w2 = st.columns([1, 1.35], gap="large")
    d = water.dropna(subset=["Screening class"]).copy()
    d["ok"] = d["Screening class"] != "Below Class C"
    with w1:
        st.markdown("### Screening-Class Distribution")
        c = class_counts(water)
        st.plotly_chart(donut_3d(list(c.index), list(c.values), [CLASS_COLOURS[x] for x in c.index], 340), use_container_width=True)
    with w2:
        st.markdown("### Treatable Records by Year (Class A–C)")
        yy = d.groupby("Year")["ok"].mean().mul(100).reset_index()
        yy.columns = ["Year", "Usable (%)"]
        fig = px.bar(yy, x="Year", y="Usable (%)", color="Usable (%)",
                     color_continuous_scale=HEAT_SCALE_R,
                     range_color=[0, 100], text=yy["Usable (%)"].round(0).astype(int))
        fig.update_traces(textposition="outside", textfont=dict(size=13, color=INK))
        fig.update_xaxes(type="category")
        fig.update_yaxes(range=[0, 115], title="% of records in Class A–C")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)

    # ---------------- combined ----------------
    if len(states):
        section_title("EcoPulse Score by state",
                      "Average of an air score and a water score, 0–100. Top 5 and bottom 5 shown.")
        pick = pd.concat([states.head(5), states.tail(5)]).drop_duplicates("State")
        pick = pick.sort_values("EcoPulse score")
        fig = go.Figure(go.Bar(
            x=pick["EcoPulse score"], y=pick["State"], orientation="h",
            marker_color=[score_band(v)[0] for v in pick["EcoPulse score"]],
            text=[f"{v:.0f}" for v in pick["EcoPulse score"]], textposition="outside",
            textfont=dict(size=14, color=INK)))
        fig.update_xaxes(range=[0, 112], title="EcoPulse Score (0–100)")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style_fig(fig, 400, legend=False), use_container_width=True)

    # ---------------- explore ----------------
    section_title("Explore")
    cards = [
        ("Air Dashboard", "City trends, calendar heatmap and pollutant mix.",
         "Open Air Dashboard", "Air Dashboard", "#0E7C86"),
        ("Water Dashboard", "State and year filters, parameters and station ranking.",
         "Open Water Dashboard", "Water Dashboard", "#4C5BD4"),
        ("Integrated Air–Water Analytics", "Where poor air and poor water overlap.",
         "Open Insights", "Insights", "#F28C28"),
        ("India Pollution Map", "City-level PM2.5 on an interactive map.",
         "Open Pollution Map", "Pollution Map", "#D6336C"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (title, text, button, page, tone) in zip(cols, cards):
        with col:
            nav_card(title, text, tone)
            if st.button(button, use_container_width=True, key=f"home_{page}"):
                navigate(page)


# --------------------------------------------------
# ROUTING
# --------------------------------------------------

page = st.session_state.page

if page == "Home":
    _home_page()
elif page == "Air Dashboard":
    show_air_dashboard()
elif page == "Air Prediction":
    show_air_prediction()
elif page == "Air Trends":
    show_air_trends()
elif page == "Air Clusters":
    show_air_clusters()
elif page == "Air Analytics":
    show_air_analytics()
elif page == "Water Dashboard":
    show_water_dashboard()
elif page == "Water Trends":
    show_water_trends()
elif page == "Water Analytics":
    show_water_analytics()
elif page == "Insights":
    show_insights()
elif page == "Pollution Map":
    show_pollution_map()
elif page == "Pollution Hotspots":
    show_pollution_hotspots()
elif page == "ML Comparison":
    show_ml_comparison()
elif page == "Download Report":
    show_report_center()
else:
    st.warning("Page not found. Please choose a page from the sidebar.")
    if st.button("Return Home"):
        navigate("Home")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    clean_html("""
    <div class="eco-footer">
        EcoPulse · Air &amp; Water Intelligence Platform · Historical analysis for India
    </div>
    """),
    unsafe_allow_html=True,
)
