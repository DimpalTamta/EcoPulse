from pathlib import Path
import html

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import MAP_STYLES, fit_zoom, india_map_layers, HEAT_SCALE, page_header, show_chart

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "air_pm25.csv"

# Approximate city centres, not air-quality monitoring station coordinates.
CITY_LOCATIONS = {
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


@st.cache_data
def load_map_data():
    df = pd.read_csv(DATA_PATH)
    required = {"City", "Date", "PM2.5"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["City", "Date", "PM2.5"]).copy()
    df["City"] = df["City"].astype(str).str.strip()
    df["Year"] = df["Date"].dt.year.astype(int)
    df["State"] = df["City"].map(lambda c: CITY_LOCATIONS.get(c, (None,))[0])
    df["Latitude"] = df["City"].map(lambda c: CITY_LOCATIONS.get(c, (None, None, None))[1])
    df["Longitude"] = df["City"].map(lambda c: CITY_LOCATIONS.get(c, (None, None, None))[2])
    return df


def set_draft_checks(prefix, options, checked):
    """Select/clear draft checkboxes without applying filters."""
    for option in options:
        st.session_state[f"{prefix}_draft::{option}"] = checked


def apply_picker(prefix, options):
    """Commit all checked options, including those hidden by search."""
    st.session_state[f"{prefix}_applied"] = [
        option for option in options
        if st.session_state.get(f"{prefix}_draft::{option}", False)
    ]


def check_picker(title, prefix, options):
    """Searchable multi-checkbox popover with explicit Apply action."""
    options = sorted(options)
    applied_key = f"{prefix}_applied"
    if applied_key not in st.session_state:
        st.session_state[applied_key] = options.copy()

    # New cities become selected by default only when they first appear.
    for option in options:
        draft_key = f"{prefix}_draft::{option}"
        if draft_key not in st.session_state:
            st.session_state[draft_key] = True

    # Only valid options can be applied after state filtering changes.
    applied = [o for o in st.session_state[applied_key] if o in options]
    st.session_state[applied_key] = applied
    label = (
        "All selected" if len(applied) == len(options)
        else "None selected" if not applied
        else f"{len(applied)} selected"
    )

    with st.popover(f"{title}: {label}", use_container_width=True):
        st.caption("Tick multiple options, then press Apply. Search does not erase checked items.")
        search = st.text_input(
            f"Search {title.lower()}",
            placeholder="Search by name...",
            key=f"{prefix}_search",
        )
        left, right = st.columns(2)
        with left:
            st.button(
                "Select all", key=f"{prefix}_all", use_container_width=True,
                on_click=set_draft_checks, args=(prefix, options, True),
            )
        with right:
            st.button(
                "Clear all", key=f"{prefix}_clear", use_container_width=True,
                on_click=set_draft_checks, args=(prefix, options, False),
            )

        matches = [o for o in options if search.casefold() in o.casefold()]
        if not matches:
            st.caption("No matching names.")
        else:
            with st.container(height=230, border=False):
                for option in matches:
                    st.checkbox(option, key=f"{prefix}_draft::{option}")

        draft_count = sum(bool(st.session_state[f"{prefix}_draft::{o}"]) for o in options)
        st.caption(f"{draft_count} checked out of {len(options)}. Changes are not applied yet.")
        st.button(
            f"Apply selected {title.lower()}",
            key=f"{prefix}_apply", use_container_width=True,
            on_click=apply_picker, args=(prefix, options),
            type="primary",
        )

    return st.session_state[applied_key]


def html_table(df):
    """HTML avoids the PyArrow import blocked on the user's Windows machine."""
    table = df.to_html(index=False, escape=True, classes="eco-peach-table", border=0)
    st.markdown(f'<div class="ep-table-wrap">{table}</div>', unsafe_allow_html=True)


def show_pollution_map():
    page_header(
        'AIR INTELLIGENCE / GEOGRAPHIC EXPLORATION',
        'India Pollution Map',
        'Explore historical PM2.5 measurements across Indian cities on an interactive map.',
    )
    st.caption("Markers use approximate city centres, not monitoring-station locations. Data are historical, not live.")

    try:
        data = load_map_data()
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        st.error("Could not load the historical air-quality dataset.")
        st.code(str(exc))
        return

    unknown = sorted(data.loc[data["State"].isna(), "City"].unique().tolist())
    if unknown:
        st.warning("Cities without configured map locations: " + ", ".join(unknown))
    data = data.dropna(subset=["State", "Latitude", "Longitude"])
    if data.empty:
        st.warning("No mapped observations are available.")
        return

    st.markdown("### Spatial Filters")
    left, right = st.columns(2, gap="large")
    states = sorted(data["State"].unique().tolist())
    with left:
        st.markdown("**States / Union Territories**")
        selected_states = check_picker("States", "map_state", states)

    eligible_cities = sorted(data.loc[data["State"].isin(selected_states), "City"].unique().tolist())
    with right:
        st.markdown("**Cities**")
        if eligible_cities:
            selected_cities = check_picker("Cities", "map_city", eligible_cities)
        else:
            st.caption("Select a state to choose cities.")
            selected_cities = []

    year_col, stat_col = st.columns(2, gap="large")
    years = sorted(data["Year"].unique().tolist())
    with year_col:
        selected_year = st.selectbox("Historical year", ["All Years"] + years, key="map_year")
    with stat_col:
        statistic = st.selectbox("Pollution statistic", ["Average PM2.5", "Maximum PM2.5"], key="map_statistic")

    if not selected_states or not selected_cities:
        st.info("Select states and cities in the dropdowns, then click each Apply button to display pollution observations.")
        return

    filtered = data.loc[data["State"].isin(selected_states) & data["City"].isin(selected_cities)].copy()
    if selected_year != "All Years":
        filtered = filtered.loc[filtered["Year"] == selected_year].copy()
    if filtered.empty:
        st.info("No historical observations match these filters.")
        return

    agg = "mean" if statistic == "Average PM2.5" else "max"
    summary = filtered.groupby(["State", "City", "Latitude", "Longitude"], as_index=False).agg(
        PM25=("PM2.5", agg),
        Observations=("PM2.5", "size"),
        FirstDate=("Date", "min"),
        LastDate=("Date", "max"),
    )
    summary["PM25"] = summary["PM25"].round(2)
    summary["Period"] = summary["FirstDate"].dt.strftime("%d %b %Y") + " – " + summary["LastDate"].dt.strftime("%d %b %Y")

    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Cities on Map", f"{len(summary):,}")
    with m2:
        st.metric("Mapped Observations", f"{summary['Observations'].sum():,}")
    with m3:
        st.metric("Highest City Value", f"{summary['PM25'].max():.2f} µg/m³")

    st.markdown("### Geographic PM2.5 Distribution")
    mc1, mc2 = st.columns([1, 1.4], gap="large")
    with mc1:
        style_name = st.selectbox("Base map", list(MAP_STYLES), key="map_base_style")
    with mc2:
        zoom_adj = st.slider("Zoom level (auto-fit to selection, adjust to taste)", -2.0, 4.0, 0.0, 0.25, key="map_zoom_adj")
    centre, auto_zoom = fit_zoom(summary["Latitude"], summary["Longitude"])
    fig = px.scatter_map(
        summary, lat="Latitude", lon="Longitude", color="PM25", size="PM25",
        hover_name="City",
        custom_data=["State", "PM25", "Observations", "Period"],
        color_continuous_scale=HEAT_SCALE,
        size_max=23,
        zoom=min(max(auto_zoom + zoom_adj, 2.5), 11),
        center=centre,
        map_style=MAP_STYLES[style_name],
        labels={"PM25": "PM2.5 (µg/m³)"},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "State: %{customdata[0]}<br>"
            "PM2.5: <b>%{customdata[1]:.2f} µg/m³</b><br>"
            "Observations: %{customdata[2]:,}<br>"
            "Period: %{customdata[3]}<extra></extra>"
        ),
        marker=dict(opacity=0.87),
    )
    fig.update_layout(
        map_layers=india_map_layers("#0B2A4A" if "Dark" not in style_name else "#9CC3EE"),
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="PM2.5", tickfont=dict(color="#0B2A2E", size=13)),
    )
    show_chart(fig, use_container_width=True)
    st.caption("Marker size and colour show the selected historical PM2.5 statistic. Values describe available city observations, not entire states.")

    st.markdown("### City-Level PM2.5 Comparison")
    ranking = summary.sort_values("PM25", ascending=False)
    chart = px.bar(
        ranking, x="PM25", y="City", orientation="h", color="PM25",
        color_continuous_scale=HEAT_SCALE,
        labels={"PM25": f"{statistic} (µg/m³)"},
        hover_data={"State": True, "Observations": True},
    )
    chart.update_layout(
        height=max(390, len(ranking) * 28),
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    show_chart(chart, use_container_width=True)
    st.caption("Comparisons are descriptive; monitoring coverage and observation counts differ by city.")

    with st.expander("View geographic data table"):
        display = ranking[["State", "City", "PM25", "Observations", "Period"]].rename(
            columns={"PM25": "PM2.5 (µg/m³)", "Period": "Observation period"}
        )
        html_table(display)
