from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    HEAT_SCALE,
    clean_matrix,
    donut_3d,
    AQI_BUCKET_COLOURS,
    AQI_BUCKET_ORDER,
    INDIA_LIMIT,
    TEAL,
    WHO_LIMIT,
    aqi_legend,
    callout,
    html_table,
    kpi_row,
    page_header,
    pm25_band,
    style_fig,
)


ROOT = Path(__file__).resolve().parents[2]

POLLUTANTS = ["PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Stepped colour scale so heatmap colours match the CPCB categories exactly.
_ZMAX = 200
HEATMAP_SCALE = HEAT_SCALE   # blue (clean) -> red (severe)


@st.cache_data
def load_air_data():
    df = pd.read_csv(
        ROOT / "data" / "processed" / "air_pm25.csv"
    )

    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _band_colour(value):
    return pm25_band(value)[1]


def show_air_dashboard():
    df = load_air_data()

    page_header(
        "AIR INTELLIGENCE / HISTORICAL EXPLORER",
        "Air Quality Dashboard",
        "Explore historical PM2.5 pollution across Indian cities "
        "using environmental observations from 2015–2020.",
    )

    cities = sorted(df["City"].dropna().unique())
    years = sorted(df["Date"].dt.year.unique().tolist())

    # ---------------- controls ----------------
    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        selected_city = st.selectbox(
            "Explore a city",
            cities,
            index=cities.index("Delhi") if "Delhi" in cities else 0,
        )

    with c2:
        if len(years) > 1:
            year_from, year_to = st.select_slider(
                "Year range",
                options=years,
                value=(years[0], years[-1]),
            )
        else:
            year_from = year_to = years[0]

    city_all = df[df["City"] == selected_city].sort_values("Date")
    city_df = city_all[
        city_all["Date"].dt.year.between(year_from, year_to)
    ].copy()

    if city_df.empty or city_df["PM2.5"].dropna().empty:
        st.warning(
            "No PM2.5 observations for this city in the selected years. "
            "Try widening the year range."
        )
        return

    st.divider()

    # ---------------- key numbers ----------------
    avg_pm25 = city_df["PM2.5"].mean()
    max_pm25 = city_df["PM2.5"].max()
    max_date = city_df.loc[city_df["PM2.5"].idxmax(), "Date"]
    observations = len(city_df)
    share_above = (city_df["PM2.5"] > INDIA_LIMIT).mean() * 100

    label, colour, advice = pm25_band(avg_pm25)
    max_label, max_colour, _ = pm25_band(max_pm25)

    kpi_row(
        [
            ("Average PM2.5", f"{avg_pm25:.1f} µg/m³", label, colour),
            (
                "Highest Recorded PM2.5",
                f"{max_pm25:.1f} µg/m³",
                f"{max_date:%d %b %Y} · {max_label}",
                max_colour,
            ),
            (
                "Days above India limit",
                f"{share_above:.0f}%",
                f"Limit = {INDIA_LIMIT} µg/m³ (24-hour)",
                "#EE8A1F",
            ),
            (
                "Historical Observations",
                f"{observations:,}",
                f"{year_from}–{year_to}",
                TEAL,
            ),
        ]
    )

    callout(
        f"<b>{selected_city}</b> averaged <b>{avg_pm25:.1f} µg/m³</b> "
        f"({avg_pm25 / WHO_LIMIT:.1f}× the WHO guideline of {WHO_LIMIT}). "
        f"Typical category: <b>{label}</b>. {advice}",
        tone=colour,
    )

    st.caption(
        "Statistics reflect available historical observations, "
        "not live air-quality measurements."
    )

    tab_trend, tab_heat, tab_mix, tab_compare, tab_data = st.tabs(
        [
            "Trend",
            "Calendar heatmap",
            "Pollutant mix",
            "Compare cities",
            "Data",
        ]
    )

    # ---------------- trend ----------------
    with tab_trend:
        st.markdown("### PM2.5 Pollution Over Time")

        view = st.radio(
            "Show values as",
            ["Monthly average", "Daily readings", "Yearly average"],
            horizontal=True,
        )

        series = city_df.set_index("Date")["PM2.5"]
        if view == "Monthly average":
            plot_df = series.resample("MS").mean().reset_index()
        elif view == "Yearly average":
            plot_df = series.resample("YS").mean().reset_index()
        else:
            plot_df = series.reset_index()

        fig = px.line(
            plot_df,
            x="Date",
            y="PM2.5",
            markers=view != "Daily readings",
            labels={
                "Date": "Date",
                "PM2.5": "Average PM2.5 (µg/m³)",
            },
        )

        fig.update_traces(
            line_color=TEAL,
            marker_color="#34B3A0",
            line_width=3 if view != "Daily readings" else 1.4,
        )

        fig.add_hline(
            y=INDIA_LIMIT,
            line_dash="dash",
            line_color="#D9453D",
            annotation_text=f"India limit {INDIA_LIMIT}",
            annotation_position="top left",
        )
        fig.add_hline(
            y=WHO_LIMIT,
            line_dash="dot",
            line_color="#1E9E5A",
            annotation_text=f"WHO guideline {WHO_LIMIT}",
            annotation_position="bottom left",
        )

        style_fig(fig, height=420, legend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Dashed red line: India's 24-hour national standard. "
            "Dotted green line: WHO 2021 guideline."
        )

    # ---------------- heatmap ----------------
    with tab_heat:
        st.markdown("### Temporal PM2.5 Intensity: Year × Month")

        heat = city_df.assign(
            Year=city_df["Date"].dt.year,
            Month=city_df["Date"].dt.month,
        ).pivot_table(
            index="Year", columns="Month", values="PM2.5", aggfunc="mean"
        )
        heat = heat.reindex(columns=range(1, 13))
        heat = clean_matrix(heat)

        fig_heat = go.Figure(
            go.Heatmap(
                z=heat.values,
                x=MONTHS,
                y=[str(y) for y in heat.index],
                colorscale=HEATMAP_SCALE,
                zmin=0,
                zmax=_ZMAX,
                hoverongaps=False,
                xgap=3,
                ygap=3,
                colorbar=dict(title="µg/m³"),
                hovertemplate=(
                    "%{y} %{x}<br>Average PM2.5: %{z:.1f} µg/m³"
                    "<extra></extra>"
                ),
            )
        )
        fig_heat.update_yaxes(autorange="reversed")
        style_fig(fig_heat, height=130 + 55 * max(len(heat), 2), legend=False)
        st.plotly_chart(fig_heat, use_container_width=True)

        aqi_legend()
        st.caption(
            "Each cell is the monthly average. Blank cells mean no data "
            "was recorded for that month."
        )

    # ---------------- pollutant mix ----------------
    with tab_mix:
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### AQI Category Distribution")

            if "AQI_Bucket" in city_df.columns:
                counts = (
                    city_df["AQI_Bucket"]
                    .value_counts()
                    .reindex(AQI_BUCKET_ORDER)
                    .dropna()
                    .reset_index()
                )
                counts.columns = ["Category", "Days"]

                _lab = counts["Category"].tolist()
                fig_pie = donut_3d(_lab, counts["Days"].tolist(), [AQI_BUCKET_COLOURS[x] for x in _lab], 380)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("AQI category column not found in the dataset.")

        with right:
            st.markdown("### Benchmark Against National Mean")

            present = [p for p in POLLUTANTS if p in df.columns]
            city_mean = city_df[present].mean()
            india_mean = df[present].mean()
            relative = (city_mean / india_mean * 100).dropna().reset_index()
            relative.columns = ["Pollutant", "Index"]

            fig_rel = go.Figure(
                go.Bar(
                    x=relative["Pollutant"],
                    y=relative["Index"],
                    marker_color=[
                        "#D9453D" if v > 100 else "#1E9E5A"
                        for v in relative["Index"]
                    ],
                    hovertemplate=(
                        "%{x}: %{y:.0f}% of the India average"
                        "<extra></extra>"
                    ),
                )
            )
            fig_rel.add_hline(
                y=100, line_dash="dash", line_color="#3C5559",
                annotation_text="India average = 100",
            )
            fig_rel.update_yaxes(title="Index (India average = 100)")
            style_fig(fig_rel, height=380, legend=False)
            st.plotly_chart(fig_rel, use_container_width=True)

        st.caption(
            "Red bars: the city is above the all-India average for that "
            "pollutant. Green bars: below average."
        )

    # ---------------- compare cities ----------------
    with tab_compare:
        st.markdown("### Inter-City Comparison")

        city_summary = (
            df.groupby("City", as_index=False)["PM2.5"]
            .mean()
            .sort_values("PM2.5", ascending=False)
        )

        fig_cities = go.Figure(
            go.Bar(
                x=city_summary["City"],
                y=city_summary["PM2.5"],
                marker=dict(
                    color=[_band_colour(v) for v in city_summary["PM2.5"]],
                    line=dict(
                        color=[
                            "#0B3B44" if c == selected_city else "rgba(0,0,0,0)"
                            for c in city_summary["City"]
                        ],
                        width=3,
                    ),
                ),
                hovertemplate=(
                    "%{x}<br>Average PM2.5: %{y:.1f} µg/m³<extra></extra>"
                ),
            )
        )
        fig_cities.add_hline(
            y=INDIA_LIMIT, line_dash="dash", line_color="#D9453D",
            annotation_text=f"India limit {INDIA_LIMIT}",
        )
        fig_cities.update_layout(xaxis_tickangle=-45)
        fig_cities.update_yaxes(title="Average PM2.5 (µg/m³)")
        style_fig(fig_cities, height=460, legend=False)
        st.plotly_chart(fig_cities, use_container_width=True)

        aqi_legend()
        st.caption(
            f"The dark outline marks {selected_city}. "
            "City averages are calculated from available records. "
            "Cities have different observation periods and coverage, "
            "so this comparison is descriptive rather than a "
            "like-for-like ranking."
        )

    # ---------------- data ----------------
    with tab_data:
        st.markdown(f"### {selected_city} data ({year_from}–{year_to})")

        export = city_df.copy()
        export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")

        st.download_button(
            "Download this city's data (CSV)",
            export.to_csv(index=False).encode("utf-8"),
            file_name=f"ecopulse_{selected_city.lower()}_{year_from}_{year_to}.csv",
            mime="text/csv",
        )

        keep = ["Date", "PM2.5"] + [p for p in POLLUTANTS if p in export.columns]
        if "AQI_Bucket" in export.columns:
            keep.append("AQI_Bucket")
        preview = export[keep].round(2).fillna("–")
        html_table(preview.tail(100).iloc[::-1], max_rows=100)
        st.caption("Latest 100 rows shown. Download the CSV for the full table.")
