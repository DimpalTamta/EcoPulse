import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    donut_3d,
    HEAT_SCALE, HEAT_SCALE_R, TEAL, callout, html_table, kpi_row, page_header, style_fig,
)
from water_data import (
    CLASS_COLOURS, CLASS_MEANING, CLASS_ORDER, PARAMS, SHORT,
    class_counts, load_water, share_good,
)

# Reference limits drawn on the charts (CPCB outdoor-bathing / drinking-water use)
LIMITS = {
    "Dissolved oxygen": (5.0, "Min. 5 mg/L (Class B)"),
    "BOD": (3.0, "Max. 3 mg/L (Class B/C)"),
    "Total coliform": (500, "Max. 500 MPN/100ml (Class B)"),
}
LOG_PARAMS = {"Conductivity", "Fecal coliform", "Total coliform",
              "Fecal streptococci", "BOD", "Nitrate-N"}


def show_water_dashboard():
    df = load_water()

    page_header(
        "WATER INTELLIGENCE / RIVER MONITORING EXPLORER",
        "Water Quality Dashboard",
        "Explore CPCB river-monitoring records across Indian states, "
        "with a simple A / B / C screening class for every record.",
    )

    states = ["All states"] + sorted(df["State"].unique())
    years = sorted(df["Year"].unique().tolist())

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        state = st.selectbox("State", states, key="wd_state")
    with c2:
        y0, y1 = st.select_slider(
            "Year range", options=years, value=(years[0], years[-1]), key="wd_years"
        )

    d = df[df["Year"].between(y0, y1)]
    if state != "All states":
        d = d[d["State"] == state]

    if d.empty:
        st.warning("No records for this selection. Try a wider year range.")
        return

    st.divider()

    good = share_good(d)
    tone = "#1E9E5A" if good >= 60 else "#E6B422" if good >= 40 else "#D9453D"
    kpi_row([
        ("Station-year records", f"{len(d):,}", f"{y0}–{y1} · {state}", TEAL),
        ("Monitoring stations", f"{d['Station Code'].nunique():,}",
         f"{d['State'].nunique()} state(s)", "#34B3A0"),
        ("Median dissolved oxygen", f"{d['Dissolved oxygen'].median():.1f} mg/L",
         "Higher is healthier", "#0E7C86"),
        ("Median BOD", f"{d['BOD'].median():.1f} mg/L",
         "Lower is healthier", "#EE8A1F"),
        ("Usable after treatment", f"{good:.0f}%", "Records in Class A, B or C", tone),
    ])

    callout(
        f"In <b>{state}</b> ({y0}–{y1}), <b>{good:.0f}%</b> of records meet at least "
        f"the simple <b>Class C</b> criteria. Median fecal coliform is "
        f"<b>{d['Fecal coliform'].median():,.0f} MPN/100ml</b>.",
        tone=tone,
    )
    st.caption(
        "Screening classes use simplified CPCB criteria on the mid-point of the reported "
        "minimum and maximum. They are not official certifications."
    )

    t_mix, t_param, t_state, t_station, t_data = st.tabs([
        "Class mix", "Parameters", "States", "Stations", "Data",
    ])

    # ------------------------------------------------ class mix
    with t_mix:
        left, right = st.columns([1, 1.5], gap="large")
        counts = class_counts(d)
        with left:
            st.markdown("### Screening-Class Distribution")
            st.plotly_chart(donut_3d(list(counts.index), list(counts.values), [CLASS_COLOURS[c] for c in counts.index], 360), use_container_width=True)
        with right:
            st.markdown("### Screening-Class Composition by Year")
            by_year = (
                d.dropna(subset=["Screening class"])
                .groupby(["Year", "Screening class"]).size().reset_index(name="n")
            )
            by_year["share"] = by_year["n"] / by_year.groupby("Year")["n"].transform("sum") * 100
            fig = px.bar(
                by_year, x="Year", y="share", color="Screening class",
                color_discrete_map=CLASS_COLOURS,
                category_orders={"Screening class": CLASS_ORDER},
                labels={"share": "% of records"},
            )
            fig.update_xaxes(type="category")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)

        st.markdown("### Class Definitions")
        cols = st.columns(4, gap="small")
        for col, c in zip(cols, CLASS_ORDER):
            with col:
                st.markdown(
                    f'<div class="ep-callout" style="--tone:{CLASS_COLOURS[c]}">'
                    f"<b>{c}</b><br>{CLASS_MEANING[c]}</div>",
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------ parameters
    with t_param:
        p = st.selectbox("Parameter", PARAMS, index=1, format_func=lambda x: x, key="wd_param")
        col = SHORT[p]
        vals = d[col].dropna()
        if vals.empty:
            st.info("No readings for this parameter in the selection.")
        else:
            a, b = st.columns(2, gap="large")
            use_log = col in LOG_PARAMS and (vals > 0).any()
            with a:
                st.markdown(f"### Distribution of {col.lower()}")
                plot_vals = vals[vals > 0] if use_log else vals
                fig = px.histogram(plot_vals, nbins=50, log_x=use_log,
                                   color_discrete_sequence=[TEAL],
                                   labels={"value": p, "count": "Records"})
                fig.update_layout(showlegend=False, yaxis_title="Records")
                if col in LIMITS:
                    fig.add_vline(x=LIMITS[col][0], line_dash="dash", line_color="#D9453D",
                                  annotation_text=LIMITS[col][1], annotation_position="top")
                st.plotly_chart(style_fig(fig, 380, legend=False), use_container_width=True)
            with b:
                st.markdown(f"### {col} by screening class")
                dd = d.dropna(subset=[col, "Screening class"])
                dd = dd[dd[col] > 0] if use_log else dd
                fig = px.box(dd, x="Screening class", y=col, color="Screening class",
                             color_discrete_map=CLASS_COLOURS, log_y=use_log,
                             category_orders={"Screening class": CLASS_ORDER},
                             labels={col: p})
                st.plotly_chart(style_fig(fig, 380, legend=False), use_container_width=True)

            s = vals.describe(percentiles=[.1, .5, .9])
            kpi_row([
                ("Median", f"{s['50%']:,.2f}", p, TEAL),
                ("10th – 90th percentile", f"{s['10%']:,.1f} – {s['90%']:,.1f}", "Typical range", "#34B3A0"),
                ("Readings", f"{int(s['count']):,}", f"of {len(d):,} records", "#0E7C86"),
            ])

    # ------------------------------------------------ states
    with t_state:
        metric = st.radio(
            "Rank states by",
            ["Share below Class C (%)", "Median BOD (mg/L)", "Median dissolved oxygen (mg/L)",
             "Median fecal coliform (MPN/100ml)"],
            horizontal=True, key="wd_rank",
        )
        base = df[df["Year"].between(y0, y1)]
        g = base.groupby("State")
        table = pd.DataFrame({
            "Records": g.size(),
            "Share below Class C (%)": g["Screening class"].apply(
                lambda s: (s.dropna() == "Below Class C").mean() * 100 if s.notna().any() else np.nan),
            "Median BOD (mg/L)": g["BOD"].median(),
            "Median dissolved oxygen (mg/L)": g["Dissolved oxygen"].median(),
            "Median fecal coliform (MPN/100ml)": g["Fecal coliform"].median(),
        })
        table = table[table["Records"] >= 10].dropna(subset=[metric])
        asc = metric == "Median dissolved oxygen (mg/L)"
        table = table.sort_values(metric, ascending=asc).head(15).iloc[::-1]
        st.markdown(f"### Top 15 states · {metric}")
        st.caption("Only states with at least 10 records in the selected years.")
        fig = px.bar(table.reset_index(), x=metric, y="State", orientation="h",
                     color=metric,
                     color_continuous_scale=HEAT_SCALE if not asc else HEAT_SCALE_R,
                     hover_data=["Records"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 520, legend=False), use_container_width=True)

    # ------------------------------------------------ stations
    with t_station:
        st.markdown("### Stations with Highest Pollution Load")
        st.caption("Ranked by median BOD, with at least 2 records in the selection.")
        sg = d.groupby(["Station Code", "Monitoring Location", "State"])
        st_tbl = sg.agg(
            Records=("Year", "size"),
            BOD=("BOD", "median"),
            DO=("Dissolved oxygen", "median"),
            FecalColiform=("Fecal coliform", "median"),
        ).reset_index()
        st_tbl = st_tbl[st_tbl["Records"] >= 2].sort_values("BOD", ascending=False).head(20)
        st_tbl = st_tbl.rename(columns={
            "BOD": "Median BOD (mg/L)", "DO": "Median DO (mg/L)",
            "FecalColiform": "Median fecal coliform",
        }).round(2)
        html_table(st_tbl, 20)

    # ------------------------------------------------ data
    with t_data:
        cols = ["Station Code", "Monitoring Location", "State", "Year",
                "Dissolved oxygen", "pH", "BOD", "Conductivity", "Nitrate-N",
                "Fecal coliform", "Total coliform", "Screening class"]
        show = d[cols].round(2)
        html_table(show, 150)
        st.download_button(
            "Download this selection (CSV)",
            show.to_csv(index=False).encode("utf-8"),
            file_name="ecopulse_water_selection.csv",
            mime="text/csv",
        )
