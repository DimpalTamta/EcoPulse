
from pathlib import Path
import html

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import HEAT_SCALE, MAP_STYLES, fit_zoom, india_map_layers, page_header, show_chart, style_fig


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "air_pm25.csv"

CITY_INFO = {
    "Ahmedabad": ("Gujarat", 23.0225, 72.5714),
    "Aizawl": ("Mizoram", 23.7271, 92.7176),
    "Amaravati": ("Andhra Pradesh", 16.5745, 80.3575),
    "Amritsar": ("Punjab", 31.6340, 74.8723),
    "Bengaluru": ("Karnataka", 12.9716, 77.5946),
    "Bhopal": ("Madhya Pradesh", 23.2599, 77.4126),
    "Brajrajnagar": ("Odisha", 21.8167, 83.9167),
    "Chandigarh": ("Chandigarh", 30.7333, 76.7794),
    "Chennai": ("Tamil Nadu", 13.0827, 80.2707),
    "Coimbatore": ("Tamil Nadu", 11.0168, 76.9558),
    "Delhi": ("Delhi", 28.6139, 77.2090),
    "Ernakulam": ("Kerala", 9.9816, 76.2999),
    "Gurugram": ("Haryana", 28.4595, 77.0266),
    "Guwahati": ("Assam", 26.1445, 91.7362),
    "Hyderabad": ("Telangana", 17.3850, 78.4867),
    "Jaipur": ("Rajasthan", 26.9124, 75.7873),
    "Jorapokhar": ("Jharkhand", 23.7167, 86.4000),
    "Kochi": ("Kerala", 9.9312, 76.2673),
    "Kolkata": ("West Bengal", 22.5726, 88.3639),
    "Lucknow": ("Uttar Pradesh", 26.8467, 80.9462),
    "Mumbai": ("Maharashtra", 19.0760, 72.8777),
    "Patna": ("Bihar", 25.5941, 85.1376),
    "Shillong": ("Meghalaya", 25.5788, 91.8933),
    "Talcher": ("Odisha", 20.9500, 85.2333),
    "Thiruvananthapuram": ("Kerala", 8.5241, 76.9366),
    "Visakhapatnam": ("Andhra Pradesh", 17.6868, 83.2185),
}

COLORS = {
    "background": "#F4F8F3",
    "dark": "#0B3B44",
    "teal": "#0E7C86",
    "mint": "#B9E6D4",
    "peach": "#F0D2BF",
    "text": "#0B2A2E",
}

SEVERITY_COLORS = {
    "Low": "#2C6FB7",
    "Moderate": "#8FBBE3",
    "Elevated": "#E9A58F",
    "High": "#D2604F",
    "Very High": "#8E1B2F",
}


@st.cache_data
def load_hotspot_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {"City", "Date", "PM2.5"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["City"] = df["City"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )
    df["PM2.5"] = pd.to_numeric(
        df["PM2.5"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["City", "Date", "PM2.5"]
    ).copy()

    df = df[df["PM2.5"] >= 0].copy()

    df["Year"] = df["Date"].dt.year
    df["State"] = df["City"].map(
        lambda city: CITY_INFO.get(
            city, ("Unknown", None, None)
        )[0]
    )

    return df


def severity_label(value):
    """
    Descriptive categories for aggregated PM2.5 values.
    These are not official AQI categories.
    """
    if value <= 30:
        return "Low"
    elif value <= 60:
        return "Moderate"
    elif value <= 90:
        return "Elevated"
    elif value <= 120:
        return "High"
    return "Very High"


def apply_chart_theme(fig, height=420):
    fig.update_layout(legend_title_text="")
    return style_fig(fig, height=height)


def render_html_table(df):
    if df.empty:
        st.info("No records available.")
        return

    display_df = df.copy()

    for column in display_df.columns:
        if pd.api.types.is_float_dtype(
            display_df[column]
        ):
            display_df[column] = (
                display_df[column]
                .map(lambda value: f"{value:,.2f}")
            )

    table_html = display_df.to_html(
        index=False,
        escape=True,
        classes="hotspot-table"
    )

    st.markdown(
        f'<div class="hotspot-table-wrap">'
        f'{table_html}</div>',
        unsafe_allow_html=True
    )


def render():
    page_header(
        "AIR INTELLIGENCE / HOTSPOTS",
        "Pollution Hotspots",
        "Discover cities with elevated historical PM2.5 concentrations, "
        "compare pollution patterns and explore hotspots across India.",
    )

    try:
        df = load_hotspot_data()
    except Exception as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.warning("No valid pollution records found.")
        return

    st.markdown("### Hotspot Filters")

    years = sorted(df["Year"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_year = st.selectbox(
            "Select year",
            ["All Years"] + years,
            key="hotspot_year"
        )

    with col2:
        statistic = st.selectbox(
            "Pollution measurement",
            [
                "Average PM2.5",
                "Maximum PM2.5"
            ],
            key="hotspot_statistic"
        )

    with col3:
        top_n = st.slider(
            "Number of ranked cities",
            min_value=3,
            max_value=26,
            value=10,
            key="hotspot_top_n"
        )

    filtered = df.copy()

    if selected_year != "All Years":
        filtered = filtered[
            filtered["Year"] == selected_year
        ].copy()

    available_states = sorted(
        filtered.loc[
            filtered["State"] != "Unknown",
            "State"
        ].unique().tolist()
    )

    selected_states = st.multiselect(
        "Filter by states / union territories",
        options=available_states,
        default=[],
        placeholder="All states",
        key="hotspot_states"
    )

    if selected_states:
        filtered = filtered[
            filtered["State"].isin(selected_states)
        ].copy()

    available_cities = sorted(
        filtered["City"].unique().tolist()
    )

    selected_cities = st.multiselect(
        "Filter by cities",
        options=available_cities,
        default=[],
        placeholder="All cities",
        key="hotspot_cities"
    )

    if selected_cities:
        filtered = filtered[
            filtered["City"].isin(selected_cities)
        ].copy()

    if filtered.empty:
        st.warning(
            "No pollution records match your filters."
        )
        return

    aggregation = (
        "mean"
        if statistic == "Average PM2.5"
        else "max"
    )

    city_stats = (
        filtered.groupby(
            ["City", "State"],
            as_index=False
        )
        .agg(
            PM25=("PM2.5", aggregation),
            Observations=("PM2.5", "count"),
            First_Date=("Date", "min"),
            Last_Date=("Date", "max")
        )
        .sort_values(
            "PM25",
            ascending=False
        )
        .reset_index(drop=True)
    )

    city_stats["Severity"] = (
        city_stats["PM25"].apply(severity_label)
    )

    city_stats["Rank"] = (
        city_stats.index + 1
    )

    city_stats["Latitude"] = (
        city_stats["City"].map(
            lambda city: CITY_INFO.get(
                city, (None, None, None)
            )[1]
        )
    )

    city_stats["Longitude"] = (
        city_stats["City"].map(
            lambda city: CITY_INFO.get(
                city, (None, None, None)
            )[2]
        )
    )

    highest = city_stats.iloc[0]

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Cities analyzed",
            f"{len(city_stats):,}"
        )

    with m2:
        st.metric(
            "Highest hotspot",
            highest["City"]
        )

    with m3:
        st.metric(
            "Highest PM2.5",
            f'{highest["PM25"]:.2f} µg/m³'
        )

    with m4:
        st.metric(
            "Observations",
            f"{len(filtered):,}"
        )

    st.caption(
        "PM2.5 values are historical measurements "
        "in µg/m³. The highest hotspot is based on "
        f"{statistic.lower()} for the selected period."
    )

    st.markdown("### Top-Ranked PM2.5 Hotspots")

    ranked = city_stats.head(top_n).copy()

    fig_bar = px.bar(
        ranked.sort_values("PM25"),
        x="PM25",
        y="City",
        orientation="h",
        color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        hover_data=[
            "State",
            "Observations"
        ],
        labels={
            "PM25": "PM2.5 (µg/m³)",
            "City": ""
        },
        title="Highest PM2.5 concentrations"
    )

    fig_bar.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=ranked.sort_values(
                "PM25"
            )["City"].tolist()
        )
    )

    apply_chart_theme(
        fig_bar,
        height=max(380, len(ranked) * 36 + 110)
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    st.markdown("### India Hotspot Map")

    map_df = city_stats.dropna(
        subset=["Latitude", "Longitude"]
    ).copy()

    if not map_df.empty:
        map_df["Marker_Size"] = (
            map_df["PM25"].clip(lower=0) + 1
        )

        _c, _z = fit_zoom(map_df["Latitude"], map_df["Longitude"], 3.6)
        _sty = st.selectbox("Base map", list(MAP_STYLES), key="hs_base_style")
        _zadj = st.slider("Zoom level", -2.0, 4.0, 0.0, 0.25, key="hs_zoom_adj")
        _z = min(max(_z + _zadj, 2.5), 11)
        fig_map = px.scatter_map(
            map_df,
            lat="Latitude",
            lon="Longitude",
            size="Marker_Size",
            color="PM25",
            hover_name="City",
            hover_data={
                "State": True,
                "PM25": ":.2f",
                "Observations": True,
                "Latitude": False,
                "Longitude": False,
                "Marker_Size": False
            },
            color_continuous_scale=HEAT_SCALE,
            zoom=_z,
            center=_c,
            size_max=35,
            map_style=MAP_STYLES[_sty],
            title="Historical city-level PM2.5 hotspots"
        )

        fig_map.update_layout(
            map_layers=india_map_layers("#0B2A4A" if "Dark" not in _sty else "#9CC3EE"),
            height=560,
            margin=dict(
                l=0,
                r=0,
                t=45,
                b=0
            ),
            coloraxis_colorbar=dict(
                title="PM2.5"
            )
        )

        show_chart(fig_map)
    else:
        st.info(
            "No geographical coordinates are "
            "available for the selected cities."
        )

    # ---------------- advanced hotspot indicators ----------------
    st.markdown("### Hotspot Diagnostics")
    _g = filtered.groupby("City")["PM2.5"]
    diag = pd.DataFrame({
        "Mean PM2.5": _g.mean(),
        "Median PM2.5": _g.median(),
        "P95 PM2.5": _g.quantile(0.95),
        "Exceedance rate (%)": _g.apply(lambda v: (v > 60).mean() * 100),
        "Peak-to-mean ratio": _g.max() / _g.mean(),
        "Variability (CV)": _g.std() / _g.mean(),
    })
    _mon = filtered.assign(_m=filtered["Date"].dt.month)
    _w = _mon[_mon["_m"].isin([11, 12, 1, 2])].groupby("City")["PM2.5"].mean()
    _s = _mon[_mon["_m"].isin([6, 7, 8, 9])].groupby("City")["PM2.5"].mean()
    diag["Winter-to-monsoon ratio"] = (_w / _s)
    diag = diag.reindex(city_stats["City"]).dropna(how="all")

    def _persist(r):
        if r["Exceedance rate (%)"] >= 60:
            return "Chronic hotspot"
        if r["Winter-to-monsoon ratio"] >= 2.2:
            return "Seasonal hotspot"
        if r["Peak-to-mean ratio"] >= 6:
            return "Episodic hotspot"
        return "Non-hotspot"
    diag["Hotspot class"] = diag.apply(_persist, axis=1)
    diag = diag.round(2).reset_index()
    HC = {"Chronic hotspot": "#8E1B2F", "Seasonal hotspot": "#D2604F",
          "Episodic hotspot": "#8FBBE3", "Non-hotspot": "#2C6FB7"}
    d1, d2 = st.columns([1, 1.5], gap="large")
    with d1:
        _cnt = diag["Hotspot class"].value_counts().reindex(list(HC)).dropna()
        from theme import donut_3d
        st.plotly_chart(donut_3d(list(_cnt.index), list(_cnt.values), [HC[k] for k in _cnt.index], 360),
                        use_container_width=True)
    with d2:
        _sc = px.scatter(diag, x="Exceedance rate (%)", y="Winter-to-monsoon ratio", size="P95 PM2.5",
                         color="Hotspot class", color_discrete_map=HC, hover_name="City", size_max=34)
        apply_chart_theme(_sc, height=360)
        st.plotly_chart(_sc, use_container_width=True)
    render_html_table(diag)
    st.caption("Exceedance rate = share of days above the India 24-hour limit (60 µg/m³). Chronic: exceeds on 60%+ of days. "
               "Seasonal: winter mean at least 2.2x monsoon mean. Episodic: peak at least 6x the mean. "
               "P95 = 95th-percentile daily PM2.5. CV = coefficient of variation.")

    st.markdown("### Hotspot Rankings")

    ranking_table = city_stats[
        [
            "Rank",
            "City",
            "State",
            "PM25",
            "Severity",
            "Observations",
            "First_Date",
            "Last_Date"
        ]
    ].copy()

    ranking_table["First_Date"] = (
        ranking_table["First_Date"]
        .dt.strftime("%d %b %Y")
    )

    ranking_table["Last_Date"] = (
        ranking_table["Last_Date"]
        .dt.strftime("%d %b %Y")
    )

    ranking_table = ranking_table.rename(
        columns={
            "PM25": "PM2.5 (µg/m³)",
            "First_Date": "First Observation",
            "Last_Date": "Last Observation"
        }
    )

    render_html_table(ranking_table)

    st.download_button(
        "Download hotspot rankings (CSV)",
        data=ranking_table.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="ecopulse_hotspot_rankings.csv",
        mime="text/csv",
        key="hotspot_csv_download"
    )

    st.markdown("### Severity Class Distribution")

    severity_counts = (
        city_stats["Severity"]
        .value_counts()
        .rename_axis("Severity")
        .reset_index(name="Cities")
    )

    fig_severity = px.bar(
        severity_counts,
        x="Severity",
        y="Cities",
        color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={
            "Severity": [
                "Low",
                "Moderate",
                "Elevated",
                "High",
                "Very High"
            ]
        },
        title="Number of cities by severity category"
    )

    apply_chart_theme(fig_severity, height=350)

    fig_severity.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )

    st.info(
        "Severity bands on this page are descriptive "
        "ranges applied to aggregated PM2.5 values. "
        "They are not official Indian AQI categories. "
        "The dataset contains historical observations, "
        "not live pollution readings."
    )


if __name__ == "__main__":
    render()
