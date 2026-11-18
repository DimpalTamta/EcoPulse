
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import COLORWAY, DIVERGE_SCALE, page_header, show_chart


ROOT = Path(__file__).resolve().parents[2]

POLLUTANTS = {
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "Nitric Oxide (NO)": "NO",
    "Nitrogen Dioxide (NO2)": "NO2",
    "Nitrogen Oxides (NOx)": "NOx",
    "Ammonia (NH3)": "NH3",
    "Carbon Monoxide (CO)": "CO",
    "Sulphur Dioxide (SO2)": "SO2",
    "Ozone (O3)": "O3",
}

SEASON_ORDER = [
    "Winter",
    "Pre-Monsoon",
    "Monsoon",
    "Post-Monsoon",
]

PLOT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font_color": "#0B2A2E",
    "margin": dict(l=10, r=10, t=35, b=20),
    "xaxis": dict(
        showgrid=False,
        zeroline=False,
    ),
    "yaxis": dict(
        gridcolor="#E4EEE9",
        zeroline=False,
    ),
}


@st.cache_data
def load_trends_data():
    df = pd.read_csv(
        ROOT / "data" / "processed" / "air_pm25.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df.dropna(subset=["Date", "City"])

    return df


def identify_season(month):
    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Pre-Monsoon"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Post-Monsoon"


def show_air_trends():
    df = load_trends_data()

    page_header(
        'AIR INTELLIGENCE / HISTORICAL ANALYSIS',
        'Air Pollution Trends',
        'Investigate historical pollution patterns, seasonal variations, and relationships between environmental measurements across Indian cities.',
    )

    st.caption(
        "Historical dataset: 2015–2020. "
        "These are recorded observations, not live measurements."
    )

    available = {
        label: column
        for label, column in POLLUTANTS.items()
        if column in df.columns
    }

    if not available:
        st.error("No supported pollutant columns were found.")
        return

    cities = sorted(df["City"].unique().tolist())

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_cities = st.multiselect(
            "Select cities",
            options=cities,
            default=(
                ["Delhi"]
                if "Delhi" in cities
                else cities[:1]
            ),
            max_selections=5,
            key="trends_cities",
        )

    with filter_col2:
        pollutant_label = st.selectbox(
            "Select pollutant",
            options=list(available.keys()),
            key="trends_pollutant",
        )

    if not selected_cities:
        st.warning("Select at least one city to continue.")
        return

    pollutant = available[pollutant_label]

    filtered = df[
        df["City"].isin(selected_cities)
    ].copy()

    filtered[pollutant] = pd.to_numeric(
        filtered[pollutant],
        errors="coerce",
    )

    filtered = filtered.dropna(
        subset=[pollutant]
    )

    if filtered.empty:
        st.warning(
            "No observations are available for this selection."
        )
        return

    date_min = filtered["Date"].min().date()
    date_max = filtered["Date"].max().date()

    date_range = st.date_input(
        "Historical date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
        key="trends_date_range",
    )

    if len(date_range) != 2:
        st.info("Select a start and end date.")
        return

    start_date, end_date = date_range

    filtered = filtered[
        filtered["Date"].dt.date.between(
            start_date,
            end_date,
        )
    ].copy()

    if filtered.empty:
        st.warning(
            "No observations exist within the selected dates."
        )
        return

    st.divider()

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Average Concentration",
            f"{filtered[pollutant].mean():.2f}",
        )

    with metric2:
        st.metric(
            "Maximum Concentration",
            f"{filtered[pollutant].max():.2f}",
        )

    with metric3:
        st.metric(
            "Available Observations",
            f"{len(filtered):,}",
        )

    unit = (
        "mg/m³"
        if pollutant == "CO"
        else "µg/m³"
    )

    st.caption(
        f"Concentration unit: {unit}. "
        "Statistics are calculated from the selected "
        "historical observations."
    )

    st.markdown("### Historical Concentration Timeline")

    granularity = st.segmented_control(
        "Time aggregation",
        options=["Daily", "Monthly", "Yearly"],
        default="Monthly",
        key="trends_granularity",
    )

    frequency = {
        "Daily": "D",
        "Monthly": "MS",
        "Yearly": "YS",
    }[granularity]

    timeline = (
        filtered.groupby(
            [
                "City",
                pd.Grouper(
                    key="Date",
                    freq=frequency,
                ),
            ]
        )[pollutant]
        .mean()
        .reset_index()
    )

    def _alpha(hex_colour, a):
        h = hex_colour.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

    fig_timeline = go.Figure()
    single = timeline["City"].nunique() == 1
    for i, (city_name, part) in enumerate(timeline.groupby("City")):
        colour = COLORWAY[i % len(COLORWAY)]
        fig_timeline.add_trace(
            go.Scatter(
                x=part["Date"],
                y=part[pollutant],
                name=city_name,
                mode="lines" if granularity == "Daily" else "lines+markers",
                line=dict(
                    color=colour,
                    width=1.8 if granularity == "Daily" else 3.5,
                    shape="spline" if granularity != "Daily" else "linear",
                    smoothing=0.6,
                ),
                marker=dict(size=8, color=colour, line=dict(color="#FFFFFF", width=2)),
                fill="tozeroy" if single else None,
                fillcolor=_alpha(colour, 0.14),
                hovertemplate=(
                    f"{city_name}<br>%{{x|%b %Y}}: %{{y:.1f}} {unit}<extra></extra>"
                ),
            )
        )

    if pollutant == "PM2.5":
        fig_timeline.add_hline(
            y=60, line_dash="dash", line_color="#D9453D",
            annotation_text="India limit 60", annotation_position="top left",
            annotation_font=dict(color="#D9453D", size=13),
        )
        fig_timeline.add_hline(
            y=15, line_dash="dot", line_color="#1E9E5A",
            annotation_text="WHO guideline 15", annotation_position="bottom left",
            annotation_font=dict(color="#1E9E5A", size=13),
        )

    fig_timeline.update_xaxes(title="Observation date")
    fig_timeline.update_yaxes(title=f"{pollutant} ({unit})", rangemode="tozero")
    fig_timeline.update_layout(
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )

    show_chart(
        fig_timeline,
        use_container_width=True,
    )

    st.markdown("### Seasonal Pollution Patterns")

    seasonal = filtered.copy()

    seasonal["Season"] = seasonal["Date"].dt.month.map(
        identify_season
    )

    seasonal_summary = (
        seasonal.groupby(
            ["City", "Season"],
            as_index=False,
        )[pollutant]
        .mean()
    )

    seasonal_summary["Season"] = pd.Categorical(
        seasonal_summary["Season"], categories=SEASON_ORDER, ordered=True
    )
    seasonal_summary = seasonal_summary.sort_values(["Season", "City"])

    fig_season = go.Figure()
    for i, (city_name, part) in enumerate(seasonal_summary.groupby("City", observed=True)):
        fig_season.add_trace(
            go.Bar(
                x=part["Season"].astype(str),
                y=part[pollutant],
                name=city_name,
                marker=dict(
                    color=COLORWAY[i % len(COLORWAY)],
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                text=part[pollutant].round(0).astype(int),
                textposition="outside",
                textfont=dict(size=13),
                hovertemplate=(
                    f"{city_name} · %{{x}}<br>Average: %{{y:.1f}} {unit}<extra></extra>"
                ),
            )
        )

    if pollutant == "PM2.5":
        fig_season.add_hline(
            y=60, line_dash="dash", line_color="#D9453D",
            annotation_text="India limit 60", annotation_position="top left",
            annotation_font=dict(color="#D9453D", size=13),
        )

    fig_season.update_xaxes(
        categoryorder="array", categoryarray=SEASON_ORDER, title="Season"
    )
    fig_season.update_yaxes(title=f"Average {pollutant} ({unit})", rangemode="tozero")
    fig_season.update_layout(
        barmode="group",
        bargap=0.25,
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )

    show_chart(
        fig_season,
        use_container_width=True,
    )

    st.caption(
        "Seasons use approximate calendar-month groupings. "
        "Seasonal averages may reflect unequal observation "
        "coverage across years and cities."
    )

    st.markdown("### Pollutant Relationships")

    correlation_columns = [
        column
        for column in POLLUTANTS.values()
        if column in filtered.columns
    ]

    correlation_data = filtered[
        correlation_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlations = correlation_data.corr()

    if correlations.notna().any().any():
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=correlations.values,
                x=correlations.columns,
                y=correlations.index,
                colorscale=DIVERGE_SCALE,
                texttemplate="%{z:.2f}",
                textfont=dict(size=12),
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="Correlation",
                ),
                hovertemplate=(
                    "%{y} vs %{x}"
                    "<extra></extra>"
                    "Correlation: %{z:.2f}"
                ),
            )
        )

        fig_corr.update_layout(
            **PLOT_LAYOUT,
            height=470,
        )

        show_chart(
            fig_corr,
            use_container_width=True,
        )

        st.caption(
            "Pearson correlations describe statistical "
            "relationships between measured pollutants. "
            "They do not establish causation."
        )
    else:
        st.info(
            "Not enough valid measurements are available "
            "to calculate pollutant correlations."
        )
