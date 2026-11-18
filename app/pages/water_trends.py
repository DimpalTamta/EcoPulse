import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from theme import (
    clean_matrix,
    COLORWAY, HEAT_SCALE, HEAT_SCALE_R, INK, MUTED, TEAL, callout, html_table, kpi_row,
    page_header, style_fig,
)
from water_data import CLASS_COLOURS, CLASS_ORDER, load_water

GOOD, BAD = "#1E9E5A", "#D9453D"

# (column, title, unit, log?, guideline line (value, label) or None)
PANELS = [
    ("Dissolved oxygen", "Dissolved oxygen", "mg/L", False, (5, "min 5")),
    ("BOD", "Biochemical oxygen demand", "mg/L", False, (3, "max 3")),
    ("pH", "pH", "", False, None),
    ("Nitrate-N", "Nitrate-N", "mg/L", False, None),
    ("Conductivity", "Conductivity", "µmho/cm", True, None),
    ("Fecal coliform", "Fecal coliform", "MPN/100ml", True, (2500, "limit 2,500")),
]

# metric -> (label, value column or special, higher_is_better)
METRICS = {
    "Usable water (% in Class A–C)": ("usable", True, "%"),
    "Median dissolved oxygen (mg/L)": ("Dissolved oxygen", True, "mg/L"),
    "Median BOD (mg/L)": ("BOD", False, "mg/L"),
    "Median fecal coliform (MPN/100ml)": ("Fecal coliform", False, "MPN/100ml"),
}


def _alpha(hex_colour, a):
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _state_year(df, metric, min_n=3):
    col, _, _ = METRICS[metric]
    d = df.copy()
    if col == "usable":
        d = d.dropna(subset=["Screening class"])
        d["v"] = (d["Screening class"] != "Below Class C") * 100.0
    else:
        d["v"] = d[col]
    g = d.dropna(subset=["v"]).groupby(["State", "Year"])["v"].agg(["median", "mean", "size"]).reset_index()
    g["value"] = g["mean"] if col == "usable" else g["median"]
    return g[g["size"] >= min_n][["State", "Year", "value", "size"]]


def show_water_trends():
    df = load_water()
    page_header(
        "WATER QUALITY / TRENDS & STATES",
        "Water Trends & State Comparison",
        "How river water quality changed from 2012 to 2023, and how states compare.",
    )

    yearly = df.groupby("Year").agg(
        Records=("Year", "size"), BOD=("BOD", "median"),
    ).reset_index()
    best = yearly.loc[yearly["BOD"].idxmin()]
    worst = yearly.loc[yearly["BOD"].idxmax()]
    kpi_row([
        ("Years covered", f"{yearly['Year'].min()}–{yearly['Year'].max()}",
         f"{len(yearly)} years with data (2015–16 missing)", TEAL),
        ("Lowest median BOD", f"{best['BOD']:.1f} mg/L", f"Year {int(best['Year'])}", "#1E9E5A"),
        ("Highest median BOD", f"{worst['BOD']:.1f} mg/L", f"Year {int(worst['Year'])}", "#D9453D"),
        ("Latest records", f"{int(yearly.iloc[-1]['Records']):,}", f"Year {int(yearly.iloc[-1]['Year'])}", "#34B3A0"),
    ])

    t_year, t_class, t_heat, t_cov = st.tabs([
        "Yearly trends", "Class share", "State × year", "Coverage",
    ])

    # ------------------------------------------------------------ yearly trends
    with t_year:
        states = ["All states"] + sorted(df["State"].unique())
        c1, _ = st.columns([1, 2])
        with c1:
            sel = st.selectbox("Area", states, key="wt_area")
        d = df if sel == "All states" else df[df["State"] == sel]
        st.markdown(f"### Six key parameters over time · {sel}")
        fig = make_subplots(rows=2, cols=3, vertical_spacing=0.2, horizontal_spacing=0.07,
                            subplot_titles=[f"<b>{t}</b>" + (f" ({u})" if u else "") for _, t, u, _, _ in PANELS])
        for i, (col, title, unit, log, ref) in enumerate(PANELS):
            r, c = divmod(i, 3)
            g = d.groupby("Year")[col].quantile([.25, .5, .75]).unstack().reset_index()
            g.columns = ["Year", "q1", "med", "q3"]
            g = g.dropna()
            colour = COLORWAY[i % len(COLORWAY)]
            fig.add_trace(go.Scatter(x=g["Year"], y=g["q3"], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"), row=r + 1, col=c + 1)
            fig.add_trace(go.Scatter(x=g["Year"], y=g["q1"], mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor=_alpha(colour, .2), showlegend=False,
                                     hoverinfo="skip"), row=r + 1, col=c + 1)
            fig.add_trace(go.Scatter(x=g["Year"], y=g["med"], mode="lines+markers", showlegend=False,
                                     line=dict(color=colour, width=3.5),
                                     marker=dict(size=8, line=dict(color="#FFFFFF", width=1.5)),
                                     hovertemplate=f"{title} %{{x}}: %{{y:.2f}} {unit}<extra></extra>"),
                          row=r + 1, col=c + 1)
            if ref:
                fig.add_hline(y=ref[0], line_dash="dash", line_color=BAD, row=r + 1, col=c + 1,
                              annotation_text=ref[1], annotation_font=dict(color=BAD, size=11),
                              annotation_position="top left")
            if col == "pH":
                fig.add_hrect(y0=6.5, y1=8.5, fillcolor=_alpha(GOOD, .10), line_width=0, row=r + 1, col=c + 1)
            fig.update_yaxes(type="log" if log else "linear", row=r + 1, col=c + 1,
                             showgrid=True, tickfont=dict(size=12))
            fig.update_xaxes(tickfont=dict(size=12), dtick=2, row=r + 1, col=c + 1)
        fig.update_annotations(font=dict(size=14, color=INK))
        st.plotly_chart(style_fig(fig, 640, legend=False), use_container_width=True)
        st.caption("Line = yearly median; shaded band = middle half of stations (25th–75th percentile). "
                   "Red dashed line = guideline (dissolved oxygen: stay above; BOD and coliform: stay below). "
                   "Green band on pH = the 6.5–8.5 target range. Conductivity and coliform use a log scale.")
        first, last = d["Year"].min(), d["Year"].max()
        bod = d.groupby("Year")["BOD"].median()
        do = d.groupby("Year")["Dissolved oxygen"].median()
        callout(f"Median BOD moved from <b>{bod.iloc[0]:.1f}</b> to <b>{bod.iloc[-1]:.1f} mg/L</b> and dissolved "
                f"oxygen from <b>{do.iloc[0]:.1f}</b> to <b>{do.iloc[-1]:.1f} mg/L</b> between {first} and {last}. "
                "Year-to-year jumps partly reflect which stations were sampled.", tone=TEAL)

    # ------------------------------------------------------------ class share
    with t_class:
        c1, _ = st.columns([1, 2])
        with c1:
            sel = st.selectbox("Area", ["All states"] + sorted(df["State"].unique()), key="wt_class_area")
        d = df.dropna(subset=["Screening class"])
        if sel != "All states":
            d = d[d["State"] == sel]
        g = d.groupby(["Year", "Screening class"]).size().reset_index(name="n")
        g["share"] = g["n"] / g.groupby("Year")["n"].transform("sum") * 100
        g["label"] = np.where(g["share"] >= 6, g["share"].round(0).astype(int).astype(str) + "%", "")
        st.markdown("### Screening-Class Composition by Year")
        fig = px.bar(g, x="Year", y="share", color="Screening class", color_discrete_map=CLASS_COLOURS,
                     category_orders={"Screening class": CLASS_ORDER}, text="label",
                     labels={"share": "% of records", "Screening class": "Class"})
        fig.update_traces(textposition="inside", textfont=dict(size=13, color="#FFFFFF"),
                          marker_line=dict(color="#FFFFFF", width=1))
        usable = d.assign(ok=d["Screening class"] != "Below Class C").groupby("Year")["ok"].mean() * 100
        fig.add_trace(go.Scatter(x=usable.index, y=usable.values, mode="lines+markers", name="Usable (A–C)",
                                 line=dict(color=INK, width=3), marker=dict(size=9, color="#FFFFFF",
                                                                               line=dict(color=INK, width=2.5))))
        fig.update_xaxes(type="category")
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(style_fig(fig, 450), use_container_width=True)
        trend = usable.iloc[-1] - usable.iloc[0]
        callout(
            f"Records usable after treatment (Class A–C) went from <b>{usable.iloc[0]:.0f}%</b> "
            f"in {usable.index[0]} to <b>{usable.iloc[-1]:.0f}%</b> in {usable.index[-1]} "
            f"({trend:+.0f} points). The black line tracks that share; small years can swing widely.",
            tone=TEAL,
        )

    # ------------------------------------------------------------ state x year
    with t_heat:
        c1, c2, c3 = st.columns([1.4, 1, 1], gap="large")
        with c1:
            metric = st.selectbox("Metric", list(METRICS), key="wt_metric")
        with c2:
            n_states = st.slider("States shown", 6, 30, 15, key="wt_nstates")
        with c3:
            sort_by = st.selectbox("Order states by", ["Best first", "Worst first", "Most records"], key="wt_sort")
        col, hib, unit = METRICS[metric]
        sy = _state_year(df, metric)
        counts = df.groupby("State").size().sort_values(ascending=False)
        keep = [s for s in counts.index if s in sy["State"].unique()][:n_states]
        sy = sy[sy["State"].isin(keep)]
        pv = sy.pivot(index="State", columns="Year", values="value")
        all_years = list(range(int(df["Year"].min()), int(df["Year"].max()) + 1))
        pv = pv.reindex(columns=all_years)
        avg = pv.mean(axis=1)
        if sort_by == "Most records":
            order = [s for s in keep if s in pv.index]
        else:
            order = avg.sort_values(ascending=not hib if sort_by == "Best first" else hib).index.tolist()
        pv = clean_matrix(pv.loc[order])
        lo, hi = np.nanpercentile(pv.values, 5), np.nanpercentile(pv.values, 95)
        scale = HEAT_SCALE_R if hib else HEAT_SCALE
        fmt = "%{text:.0f}" if unit in ("%", "MPN/100ml") else "%{text:.1f}"
        st.markdown(f"### {metric} by state and year")
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=[str(y) for y in pv.columns], y=list(pv.index), colorscale=scale,
            zmin=lo, zmax=hi, xgap=3, ygap=3, hoverongaps=False,
            text=pv.values, texttemplate=fmt, textfont=dict(size=12, color="#0B2A2E"),
            colorbar=dict(title=unit), hovertemplate="%{y} · %{x}<br>%{z:,.1f} " + unit + "<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(type="category", side="top", showgrid=False)
        fig.update_layout(plot_bgcolor="#EEF2F2")
        fig = style_fig(fig, 150 + 36 * len(pv), legend=False)
        fig.update_layout(plot_bgcolor="#E9EEEE")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(("Blue = better, red = worse. " if True else "") +
                   "Years without enough readings are interpolated from neighbouring years. Colours are capped at the 5th–95th percentile "
                   "so extreme readings do not wash out the rest.")

        st.markdown("### Net Change: Baseline vs Latest Year")
        rows = []
        for s, d in sy.groupby("State"):
            d = d.sort_values("Year")
            if len(d) < 2:
                continue
            rows.append({"State": s, "From": int(d["Year"].iloc[0]), "To": int(d["Year"].iloc[-1]),
                         "Start": d["value"].iloc[0], "End": d["value"].iloc[-1]})
        ch = pd.DataFrame(rows)
        if not ch.empty:
            ch["Change"] = ch["End"] - ch["Start"]
            ch["Better"] = (ch["Change"] > 0) if hib else (ch["Change"] < 0)
            ch = ch.sort_values("Change", ascending=hib)
            fig = go.Figure()
            for r in ch.itertuples():
                fig.add_trace(go.Scatter(x=[r.Start, r.End], y=[r.State, r.State], mode="lines", showlegend=False,
                                         line=dict(color=GOOD if r.Better else BAD, width=5), opacity=.55,
                                         hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=ch["Start"], y=ch["State"], mode="markers", name="First year",
                                     marker=dict(size=13, color="#FFFFFF", line=dict(color=INK, width=2.5)),
                                     customdata=ch["From"],
                                     hovertemplate="%{y} · %{customdata}: %{x:,.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=ch["End"], y=ch["State"], mode="markers", name="Latest year",
                                     marker=dict(size=14, color=[GOOD if b else BAD for b in ch["Better"]]),
                                     customdata=ch["To"],
                                     hovertemplate="%{y} · %{customdata}: %{x:,.1f}<extra></extra>"))
            fig.update_yaxes(categoryorder="array", categoryarray=ch["State"].tolist()[::-1], showgrid=True)
            fig.update_xaxes(title=f"{metric}", showgrid=True)
            fig.update_layout(legend=dict(orientation="h", y=1.07))
            st.plotly_chart(style_fig(fig, 130 + 32 * len(ch)), use_container_width=True)
            best = ch.sort_values("Change", ascending=not hib).iloc[0]
            callout(f"Largest improvement: <b>{best['State']}</b> ({best['Change']:+.1f} {unit}, "
                    f"{best['From']}→{best['To']}). Green line = improved, red = worsened.", tone=GOOD)

        st.markdown("### State-Level Time Series")
        picks = st.multiselect("States", list(pv.index), default=list(pv.index)[:3], key="wt_follow")
        if picks:
            fig = go.Figure()
            for i, s in enumerate(picks):
                d = sy[sy["State"] == s].sort_values("Year")
                fig.add_trace(go.Scatter(x=d["Year"], y=d["value"], name=s, mode="lines+markers",
                                         line=dict(width=3.5, color=COLORWAY[i % len(COLORWAY)]),
                                         marker=dict(size=9, line=dict(color="#FFFFFF", width=2))))
            fig.update_xaxes(type="category", title="Year")
            fig.update_yaxes(title=metric)
            fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    # ------------------------------------------------------------ coverage
    with t_cov:
        cov = df.groupby("State").agg(
            Records=("Year", "size"), Stations=("Station Code", "nunique"),
            First=("Year", "min"), Last=("Year", "max"),
        ).reset_index().sort_values("Records", ascending=False)
        n = st.slider("States shown", 8, min(35, len(cov)), 20, key="wt_cov_n")
        top = cov.head(n).iloc[::-1]
        st.markdown("### Station Coverage by State")
        fig = go.Figure(go.Bar(
            x=top["Records"], y=top["State"], orientation="h",
            marker=dict(color=top["Stations"], colorscale=[[0, "#CFE2F5"], [.5, "#5B8FC7"], [1, "#12355B"]],
                        colorbar=dict(title="Stations")),
            text=top["Records"].map("{:,}".format), textposition="outside", textfont=dict(size=13, color=INK),
            customdata=np.stack([top["Stations"], top["First"], top["Last"]], axis=1),
            hovertemplate="%{y}<br>%{x:,} records · %{customdata[0]} stations<br>"
                          "%{customdata[1]}–%{customdata[2]}<extra></extra>"))
        fig.update_xaxes(title="Station-year records", range=[0, top["Records"].max() * 1.15])
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style_fig(fig, 120 + 28 * n, legend=False), use_container_width=True)
        st.caption("Bar length = records; bar colour = number of monitoring stations. "
                   "States with few records give less reliable averages.")
        with st.expander("Full coverage table"):
            html_table(cov, 80)
