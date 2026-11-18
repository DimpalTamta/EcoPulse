"""Download centre: PDF report + CSV export of the historical air-quality data.

Uses the existing `generate_air_report` function from report_generator.py.
If `reportlab` is not installed the page still opens and explains how to
enable the PDF button, so the rest of the app is never affected.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from theme import kpi_row, page_header, callout

try:
    from pages.report_generator import generate_air_report
    _REPORT_ERROR = None
except Exception as error:  # reportlab missing, etc.
    generate_air_report = None
    _REPORT_ERROR = error


ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_report_data():
    df = pd.read_csv(ROOT / "data" / "processed" / "air_pm25.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _show_air_exports():
    df = load_report_data()

    cities = ["All Cities"] + sorted(df["City"].dropna().unique().tolist())

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        city = st.selectbox(
            "City",
            cities,
            index=cities.index("Delhi") if "Delhi" in cities else 0,
        )

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    with c2:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Pick both a start date and an end date to continue.")
        return

    start, end = date_range

    mask = (df["Date"] >= pd.Timestamp(start)) & (
        df["Date"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    )
    if city != "All Cities":
        mask &= df["City"] == city
    subset = df.loc[mask]

    if subset.empty:
        st.warning(
            "No records match this city and date range. "
            "Try a wider range."
        )
        return

    kpi_row(
        [
            ("Records included", f"{len(subset):,}", f"{start} → {end}"),
            ("Cities", f"{subset['City'].nunique()}", city),
            (
                "Average PM2.5",
                f"{subset['PM2.5'].mean():.1f} µg/m³",
                "Over the selected range",
            ),
        ]
    )

    st.markdown("### Your downloads")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**PDF summary report**")
        st.caption("Summary statistics and city comparison tables.")

        if generate_air_report is None:
            callout(
                "The PDF generator needs the <b>reportlab</b> package. "
                "Run <b>pip install reportlab</b> in your terminal, then "
                "restart the app."
            )
        else:
            if st.button("Prepare PDF report", use_container_width=True):
                try:
                    pdf_bytes = generate_air_report(
                        df,
                        city=city,
                        start_date=start,
                        end_date=end,
                    )
                    st.session_state["ep_pdf"] = (
                        pdf_bytes,
                        f"ecopulse_report_{city.lower().replace(' ', '_')}.pdf",
                    )
                except Exception as error:
                    st.error(f"Could not create the report: {error}")

            ready = st.session_state.get("ep_pdf")
            if ready:
                st.download_button(
                    "Download PDF",
                    ready[0],
                    file_name=ready[1],
                    mime="application/pdf",
                    use_container_width=True,
                )

    with right:
        st.markdown("**Cleaned data (CSV)**")
        st.caption("Exactly the rows that match your filters.")

        export = subset.copy()
        export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
        st.download_button(
            "Download CSV",
            export.to_csv(index=False).encode("utf-8"),
            file_name=f"ecopulse_data_{city.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "Reports describe historical observations (2015–2020) and are for "
        "educational use, not official CPCB assessments."
    )


def _show_water_exports():
    from water_data import load_water, class_counts, share_good

    df = load_water()
    states = ["All states"] + sorted(df["State"].unique())
    years = sorted(df["Year"].unique().tolist())

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        state = st.selectbox("State", states, key="rc_wstate")
    with c2:
        y0, y1 = st.select_slider("Year range", options=years,
                                  value=(years[0], years[-1]), key="rc_wyears")

    d = df[df["Year"].between(y0, y1)]
    if state != "All states":
        d = d[d["State"] == state]
    if d.empty:
        st.warning("No records match this selection.")
        return

    kpi_row([
        ("Records included", f"{len(d):,}", f"{y0} → {y1}"),
        ("Stations", f"{d['Station Code'].nunique():,}", state),
        ("Usable after treatment", f"{share_good(d):.0f}%", "Class A, B or C"),
    ])

    st.markdown("### Your downloads")
    left, right = st.columns(2, gap="large")
    slug = state.lower().replace(" ", "_")

    with left:
        st.markdown("**PDF summary report**")
        st.caption("Class mix, key medians and state comparison.")
        if st.button("Prepare water PDF", use_container_width=True, key="rc_wpdf_btn"):
            try:
                from pages.water_report import generate_water_report
                st.session_state["ep_wpdf"] = (
                    generate_water_report(df, state, y0, y1),
                    f"ecopulse_water_report_{slug}.pdf",
                )
            except Exception as error:
                st.error(f"Could not create the report: {error}")
        ready = st.session_state.get("ep_wpdf")
        if ready:
            st.download_button("Download PDF", ready[0], file_name=ready[1],
                               mime="application/pdf", use_container_width=True)

    with right:
        st.markdown("**Cleaned data (CSV)**")
        st.caption("Exactly the rows that match your filters.")
        st.download_button(
            "Download CSV", d.to_csv(index=False).encode("utf-8"),
            file_name=f"ecopulse_water_{slug}.csv", mime="text/csv",
            use_container_width=True,
        )

    st.caption("Water screening classes are simplified CPCB-style labels, "
               "not official certifications.")


def show_report_center():
    page_header(
        "EXPORT / REPORTS",
        "Download Centre",
        "Create a PDF summary or export cleaned data for air quality "
        "or river water quality.",
    )
    air_tab, water_tab = st.tabs(["Air Quality", "Water Quality"])
    with air_tab:
        _show_air_exports()
    with water_tab:
        _show_water_exports()
