"""Air Analytics: what the PM2.5 record says about cities, seasons and categories.

  City trends      -> Which cities got cleaner or dirtier over time?
  Seasonality      -> Which months are worst, and by how much?
  AQI categories   -> How many days were Good ... Severe?
  Spread           -> How variable is the air, and how often is the limit broken?
  Pollutants       -> Which pollutants rise and fall together?
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    clean_matrix,
    AQI_BUCKET_COLOURS, AQI_BUCKET_ORDER, COLORWAY, DIVERGE_SCALE, INDIA_LIMIT, INK,
    MUTED, TEAL, WHO_LIMIT, callout, kpi_row, page_header, pm25_band, style_fig,
)
from pages.air_dashboard import HEATMAP_SCALE, _ZMAX, load_air_data

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
POLL = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
MIN_DAYS = 90
GOOD, BAD = "#1E9E5A", "#D9453D"   # improved / worsened (meaning, not decoration)


def _alpha(hex_colour, a):
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


@st.cache_data(show_spinner=False)
def _annual(include_partial):
    df = load_air_data().dropna(subset=["PM2.5"]).copy()
    df["Year"] = df["Date"].dt.year
    g = df.groupby(["City", "Year"])["PM2.5"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= MIN_DAYS]
    if not include_partial:
        g = g[g["Year"] < df["Year"].max()]
    return g.rename(columns={"mean": "PM2.5"})


def _line_chart(annual, cities):
    fig = go.Figure()
    nat = annual.groupby("Year")["PM2.5"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=nat["Year"], y=nat["PM2.5"], name="All-city average", mode="lines",
        line=dict(color=MUTED, width=3, dash="dot"),
        hovertemplate="All cities %{x}: %{y:.0f} µg/m³<extra></extra>"))
    for i, c in enumerate(cities):
        d = annual[annual["City"] == c]
        fig.add_trace(go.Scatter(
            x=d["Year"], y=d["PM2.5"], name=c, mode="lines+markers",
            line=dict(width=3.5, color=COLORWAY[i % len(COLORWAY)]),
            marker=dict(size=10, line=dict(color="#FFFFFF", width=2)),
            hovertemplate=c + " %{x}: %{y:.0f} µg/m³<extra></extra>"))
    fig.add_hline(y=INDIA_LIMIT, line_dash="dash", line_color=BAD,
                  annotation_text=f"India limit {INDIA_LIMIT}", annotation_position="top left",
                  annotation_font=dict(color=BAD, size=13))
    fig.add_hline(y=WHO_LIMIT, line_dash="dot", line_color=GOOD,
                  annotation_text=f"WHO guideline {WHO_LIMIT}", annotation_position="bottom left",
                  annotation_font=dict(color=GOOD, size=13))
    fig.update_xaxes(type="category", title="Year")
    fig.update_yaxes(title="Annual average PM2.5 (µg/m³)", rangemode="tozero")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    return style_fig(fig, 450)


def _dumbbell(annual, show):
    rows = []
    for city, d in annual.groupby("City"):
        d = d.sort_values("Year")
        if len(d) < 2:
            continue
        a, b = d.iloc[0], d.iloc[-1]
        rows.append({"City": city, "From": int(a["Year"]), "To": int(b["Year"]),
                     "Start": a["PM2.5"], "End": b["PM2.5"], "Change": b["PM2.5"] - a["PM2.5"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return None, t
    if show == "Cleaner now":
        t = t[t["Change"] < 0]
    elif show == "Dirtier now":
        t = t[t["Change"] > 0]
    if t.empty:
        return None, t
    t = t.sort_values("Change", ascending=False)
    fig = go.Figure()
    for r in t.itertuples():
        fig.add_trace(go.Scatter(
            x=[r.Start, r.End], y=[r.City, r.City], mode="lines", showlegend=False,
            line=dict(color=GOOD if r.Change < 0 else BAD, width=5), hoverinfo="skip", opacity=.55))
    fig.add_trace(go.Scatter(
        x=t["Start"], y=t["City"], mode="markers", name="First full year",
        marker=dict(size=13, color="#FFFFFF", line=dict(color=INK, width=2.5)),
        customdata=t["From"], hovertemplate="%{y} · %{customdata}: %{x:.0f} µg/m³<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t["End"], y=t["City"], mode="markers+text", name="Latest full year",
        marker=dict(size=14, color=[GOOD if c < 0 else BAD for c in t["Change"]]),
        text=[f"{c:+.0f}" for c in t["Change"]], textposition="middle right",
        textfont=dict(size=13, color=INK),
        customdata=t["To"], hovertemplate="%{y} · %{customdata}: %{x:.0f} µg/m³<extra></extra>"))
    fig.add_vline(x=INDIA_LIMIT, line_dash="dash", line_color=BAD, opacity=.6)
    fig.update_yaxes(categoryorder="array", categoryarray=t["City"].tolist()[::-1], showgrid=True)
    fig.update_xaxes(title="Annual average PM2.5 (µg/m³)", rangemode="tozero", showgrid=True)
    fig.update_layout(legend=dict(orientation="h", y=1.06))
    return style_fig(fig, max(380, 30 * len(t) + 110)), t


def show_air_analytics():
    df = load_air_data()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    page_header(
        "AIR QUALITY / DEEPER ANALYTICS",
        "Air Analytics",
        "Which cities are improving, which months are worst, and how often the "
        "air breaks the limit.",
    )

    by_city = df.groupby("City")["PM2.5"].mean().dropna()
    above = (df["PM2.5"] > INDIA_LIMIT).mean() * 100
    kpi_row([
        ("Cities", f"{by_city.size}", "With PM2.5 readings", TEAL),
        ("Days above India limit", f"{above:.0f}%", f"Limit {INDIA_LIMIT} µg/m³", "#EE8A1F"),
        ("Days above WHO guideline", f"{(df['PM2.5'] > WHO_LIMIT).mean() * 100:.0f}%",
         f"Guideline {WHO_LIMIT} µg/m³", "#D9453D"),
        ("Worst month (all cities)", MONTHS[int(df.groupby('Month')['PM2.5'].mean().idxmax()) - 1],
         "Highest average PM2.5", "#7A1F2B"),
    ])

    t1, t2, t3, t4, t5 = st.tabs([
        "City trends", "Seasonality", "AQI categories", "Spread & weekdays", "Pollutants",
    ])

    # ------------------------------------------------------------ city trends
    with t1:
        c1, c2 = st.columns([2, 1], gap="large")
        with c2:
            partial = st.toggle("Include partial 2020 (Jan–Jul)", value=False, key="aa_partial")
        annual = _annual(partial)
        all_cities = sorted(annual["City"].unique())
        defaults = [c for c in ["Delhi", "Mumbai", "Chennai", "Bengaluru", "Kolkata"] if c in all_cities]
        with c1:
            picks = st.multiselect("Cities to compare", all_cities, default=defaults,
                                   max_selections=8, key="aa_cities")

        st.markdown("### Annual average PM2.5")
        if picks:
            st.plotly_chart(_line_chart(annual, picks), use_container_width=True)
            st.caption("Dotted grey line = average of all cities. Each point needs at least "
                       f"{MIN_DAYS} daily readings in that year.")
        else:
            st.info("Pick at least one city to draw its trend.")

        st.markdown("### Change in Annual Mean PM2.5 (Start vs End Year)")
        show = st.radio("Show", ["All cities", "Cleaner now", "Dirtier now"], horizontal=True, key="aa_show")
        fig, t = _dumbbell(annual, show)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            s = t.sort_values("Change")
            best, worst = s.iloc[0], s.iloc[-1]
            callout(
                f"Biggest improvement: <b>{best['City']}</b> ({best['Change']:+.0f} µg/m³, "
                f"{best['From']}→{best['To']}). Biggest rise: <b>{worst['City']}</b> "
                f"({worst['Change']:+.0f} µg/m³, {worst['From']}→{worst['To']}).",
                tone=GOOD if best["Change"] < 0 else BAD)
            st.caption("White dot = first full year on record for that city; coloured dot = latest full "
                       "year. Green line = cleaner, red line = dirtier. Cities have different start years.")
        else:
            st.info("No cities match this filter.")

    # ------------------------------------------------------------ seasonality
    with t2:
        cities_s = sorted(by_city.index)
        picks = st.multiselect("Highlight cities", cities_s,
                               default=[c for c in ["Delhi", "Mumbai", "Chennai"] if c in cities_s],
                               max_selections=6, key="aa_season_cities")
        mm = df.groupby(["City", "Month"])["PM2.5"].mean().unstack("Month").reindex(columns=range(1, 13))
        nat = mm.mean()
        lo, hi = mm.quantile(.25), mm.quantile(.75)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MONTHS, y=hi.values, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=MONTHS, y=lo.values, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_alpha(TEAL, .18),
                                 name="Middle half of cities", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=MONTHS, y=nat.values, mode="lines+markers", name="All-city average",
                                 line=dict(color=TEAL, width=4), marker=dict(size=9)))
        for i, c in enumerate(picks):
            fig.add_trace(go.Scatter(x=MONTHS, y=mm.loc[c].values, mode="lines+markers", name=c,
                                     line=dict(width=3, color=COLORWAY[(i + 1) % len(COLORWAY)])))
        fig.add_hline(y=INDIA_LIMIT, line_dash="dash", line_color=BAD,
                      annotation_text=f"India limit {INDIA_LIMIT}", annotation_font=dict(color=BAD, size=13))
        fig.update_yaxes(title="Average PM2.5 (µg/m³)", rangemode="tozero")
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.15))
        st.markdown("### Seasonal Profile (Monthly Mean)")
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
        worst_m, best_m = nat.idxmax(), nat.idxmin()
        callout(f"Across cities, <b>{MONTHS[int(worst_m) - 1]}</b> is the worst month "
                f"(<b>{nat.max():.0f} µg/m³</b>) and <b>{MONTHS[int(best_m) - 1]}</b> the cleanest "
                f"(<b>{nat.min():.0f} µg/m³</b>) — a swing of <b>{nat.max() / nat.min():.1f}×</b>.", tone=TEAL)

        st.markdown("### City × Month PM2.5 Intensity Matrix")
        n_show = st.slider("Cities shown (most polluted first)", 6, len(mm), min(12, len(mm)), key="aa_heat_n")
        order = by_city.sort_values(ascending=False).head(n_show).index
        pv = clean_matrix(mm.loc[order])
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=MONTHS, y=list(pv.index), colorscale=HEATMAP_SCALE, zmin=0, zmax=_ZMAX,
            xgap=3, ygap=3, text=np.round(pv.values, 0), texttemplate="%{text:.0f}",
            textfont=dict(size=12, color="#0B2A2E"), hoverongaps=False,
            colorbar=dict(title="µg/m³"),
            hovertemplate="%{y} · %{x}<br>%{z:.0f} µg/m³<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(style_fig(fig, 90 + 38 * n_show, legend=False), use_container_width=True)
        st.caption("Blue = lower PM2.5, red = higher PM2.5; gaps in monitoring are interpolated. "
                   "Numbers are monthly average PM2.5 in µg/m³.")

    # ------------------------------------------------------------ AQI categories
    with t3:
        d = df.dropna(subset=["AQI_Bucket"])
        st.markdown("### AQI Category Share by Year")
        g = d.groupby(["Year", "AQI_Bucket"]).size().reset_index(name="n")
        g["share"] = g["n"] / g.groupby("Year")["n"].transform("sum") * 100
        g["label"] = np.where(g["share"] >= 4, g["share"].round(0).astype(int).astype(str) + "%", "")
        fig = px.bar(g, x="Year", y="share", color="AQI_Bucket", color_discrete_map=AQI_BUCKET_COLOURS,
                     category_orders={"AQI_Bucket": AQI_BUCKET_ORDER},
                     labels={"share": "% of days", "AQI_Bucket": "Category"}, text="label")
        fig.update_traces(textposition="inside", textfont=dict(size=12, color="#FFFFFF"),
                          marker_line=dict(color="#FFFFFF", width=1))
        fig.update_xaxes(type="category")
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

        st.markdown("### City-Level Breakdown")
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            yr = st.selectbox("Year", ["All years"] + sorted(d["Year"].unique().tolist(), reverse=True),
                              key="aa_aqi_year")
        dd = d if yr == "All years" else d[d["Year"] == yr]
        poor = (dd.assign(p=dd["AQI_Bucket"].isin(["Poor", "Very Poor", "Severe"]))
                .groupby("City")["p"].mean().mul(100).sort_values(ascending=False))
        with c2:
            n = st.slider("Cities shown (most 'Poor or worse' days first)", 5, max(len(poor), 6),
                          min(12, len(poor)), key="aa_aqi_n")
        top = poor.head(n).index
        g = dd[dd["City"].isin(top)].groupby(["City", "AQI_Bucket"]).size().reset_index(name="n")
        g["share"] = g["n"] / g.groupby("City")["n"].transform("sum") * 100
        fig = px.bar(g, x="share", y="City", color="AQI_Bucket", orientation="h",
                     color_discrete_map=AQI_BUCKET_COLOURS,
                     category_orders={"AQI_Bucket": AQI_BUCKET_ORDER, "City": list(top)},
                     labels={"share": "% of days", "AQI_Bucket": "Category"})
        fig.update_traces(marker_line=dict(color="#FFFFFF", width=1))
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(style_fig(fig, 110 + 36 * n), use_container_width=True)
        st.caption("Sorted so the cities with the most Poor, Very Poor and Severe days are at the top.")

    # ------------------------------------------------------------ spread & weekdays
    with t4:
        st.markdown("### Dispersion of Daily PM2.5 by City")
        n = st.slider("Cities shown (highest median first)", 5, len(by_city), 10, key="aa_box_n")
        med = df.groupby("City")["PM2.5"].median().sort_values(ascending=False)
        top = med.head(n).index.tolist()
        cap = float(df[df["City"].isin(top)]["PM2.5"].quantile(.995))
        fig = go.Figure()
        for c in top[::-1]:
            vals = df.loc[df["City"] == c, "PM2.5"].dropna()
            colour = pm25_band(float(vals.median()))[1]
            fig.add_trace(go.Box(
                x=vals, name=c, orientation="h", boxmean=True, boxpoints=False,
                line=dict(color=colour, width=2), fillcolor=_alpha(colour, .35),
                hovertemplate=c + "<br>%{x:.0f} µg/m³<extra></extra>"))
        fig.add_vline(x=INDIA_LIMIT, line_dash="dash", line_color=BAD,
                      annotation_text=f"India limit {INDIA_LIMIT}", annotation_position="top",
                      annotation_font=dict(color=BAD, size=13))
        fig.add_vline(x=WHO_LIMIT, line_dash="dot", line_color=GOOD,
                      annotation_text=f"WHO {WHO_LIMIT}", annotation_position="top",
                      annotation_font=dict(color=GOOD, size=13))
        fig.update_xaxes(title="Daily PM2.5 (µg/m³)", range=[0, cap], showgrid=True)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style_fig(fig, 120 + 44 * n, legend=False), use_container_width=True)
        st.caption("Box = middle half of days, line = median, dashed line = mean, whiskers = typical range. "
                   "Box colour shows the CPCB category of the median. The axis is clipped at the 99.5th "
                   "percentile so rare extreme days do not squash the boxes.")

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("### NAAQS Exceedance Days")
            share = (df[df["City"].isin(top)].assign(x=lambda t: t["PM2.5"] > INDIA_LIMIT)
                     .groupby("City")["x"].mean().mul(100).reindex(top))
            fig = go.Figure(go.Bar(
                x=share.values, y=list(share.index), orientation="h",
                marker_color=[pm25_band(float(med[c]))[1] for c in share.index],
                text=[f"{v:.0f}%" for v in share.values], textposition="outside",
                textfont=dict(size=13, color=INK),
                hovertemplate="%{y}: %{x:.0f}% of days<extra></extra>"))
            fig.update_xaxes(title="% of days above 60 µg/m³", range=[0, 112])
            fig.update_yaxes(autorange="reversed", showgrid=False)
            st.plotly_chart(style_fig(fig, 120 + 36 * n, legend=False), use_container_width=True)
        with b:
            st.markdown("### Day-of-Week Seasonality")
            wd = df.groupby(df["Date"].dt.dayofweek)["PM2.5"].mean()
            dev = (wd / wd.mean() - 1) * 100
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig = go.Figure(go.Bar(
                x=names, y=dev.values, marker_color=[BAD if v > 0 else GOOD for v in dev.values],
                text=[f"{v:+.1f}%" for v in dev.values], textposition="outside",
                textfont=dict(size=13, color=INK),
                hovertemplate="%{x}: %{y:+.1f}% vs average day<extra></extra>"))
            lim = max(5.0, float(np.abs(dev).max()) * 2.2)
            fig.update_yaxes(title="Difference from the average day (%)", range=[-lim, lim],
                             zeroline=True, zerolinecolor=INK)
            st.plotly_chart(style_fig(fig, 120 + 36 * n, legend=False), use_container_width=True)
        spread = float(dev.max() - dev.min())
        callout(f"Weekdays differ by only <b>{spread:.1f} percentage points</b> from top to bottom. "
                "Pollution here follows <b>season and city</b>, not the day of the week.", tone=TEAL)

    # ------------------------------------------------------------ pollutants
    with t5:
        cols = [c for c in POLL if c in df]
        corr = df[cols].corr(method="spearman").round(2)
        st.markdown("### Pollutant Correlation Matrix")
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=DIVERGE_SCALE)
        fig.update_traces(textfont=dict(size=14))
        st.plotly_chart(style_fig(fig, 500, legend=False), use_container_width=True)
        pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
        top3 = pairs.sort_values(ascending=False).head(3)
        txt = "; ".join(f"<b>{a} ↔ {b}</b> ({v:+.2f})" for (a, b), v in top3.items())
        callout(f"Strongest links: {txt}. Pollutants that rise together usually share a source "
                "(for example traffic or burning).", tone=TEAL)
        st.caption("Spearman rank correlation of daily readings across all cities. "
                   "Correlation does not prove one pollutant causes another.")
