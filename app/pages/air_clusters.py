"""Clusters & PCA: what kinds of polluted air exist, and how do they differ?

K-Means groups every daily record by its mix of nine pollutants.  PCA is a way
of drawing that nine-dimensional mix on a flat chart.  This page names each
cluster from its pollutant profile and explains every chart in plain words.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    HEAT_SCALE,
    AQI_BUCKET_COLOURS, AQI_BUCKET_ORDER, COLORWAY, DIVERGE_SCALE, INDIA_LIMIT, INK, TEAL,
    callout, html_table, kpi_row, page_header, style_fig,
)

ROOT = Path(__file__).resolve().parents[2]
CLUSTERS_PATH = ROOT / "data" / "processed" / "air_clusters.csv"
PCA_PATH = ROOT / "data" / "processed" / "air_pca.csv"
PCA_MODEL = ROOT / "models" / "air_pca.pkl"

POLLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PC_MEANING = {  # plain-language labels of the usual pollutant groups
    "PM2.5": "fine dust & smoke", "PM10": "coarse dust", "NO": "traffic exhaust",
    "NO2": "traffic exhaust", "NOx": "traffic exhaust", "NH3": "agriculture / waste",
    "CO": "combustion", "SO2": "industry / coal", "O3": "sunlight chemistry",
}


@st.cache_data(show_spinner=False)
def load_data():
    c = pd.read_csv(CLUSTERS_PATH, parse_dates=["Date"])
    p = pd.read_csv(PCA_PATH, parse_dates=["Date"])
    pcs = [x for x in p.columns if x.startswith("PC")]
    if len(c) == len(p) and (c["City"].values == p["City"].values).all() \
            and (c["Date"].values == p["Date"].values).all():
        c = pd.concat([c.reset_index(drop=True), p[pcs].reset_index(drop=True)], axis=1)
    else:
        c = c.merge(p[["City", "Date"] + pcs], on=["City", "Date"], how="left")
    c["Cluster"] = c["Pollution_Cluster"].astype(int)
    c["Year"] = c["Date"].dt.year
    c["Month"] = c["Date"].dt.month
    return c, pcs


@st.cache_resource(show_spinner=False)
def load_pca_model():
    try:
        import joblib
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = joblib.load(PCA_MODEL)
        pca = m[-1]
        feats = list(m[0].feature_names_in_) if hasattr(m[0], "feature_names_in_") else POLLS
        return pca.components_, pca.explained_variance_ratio_, feats
    except Exception:
        return None


def name_clusters(df):
    """Name each cluster from how its pollutants compare with the all-data average."""
    mean_all = df[POLLS].mean()
    prof = df.groupby("Cluster")[POLLS].mean()
    idx = prof / mean_all * 100
    names, why = {}, {}
    pm_rank = prof["PM2.5"].sort_values(ascending=False).index.tolist()
    gas_rank = (idx[["CO", "SO2", "NO2"]].mean(axis=1)).sort_values(ascending=False).index.tolist()
    heavy = pm_rank[0]
    gas = next(c for c in gas_rank if c != heavy) if len(gas_rank) > 1 else None
    for c in prof.index:
        if c == heavy:
            names[c] = "Particulate-heavy"
        elif c == gas and idx.loc[c, ["CO", "SO2"]].max() > 200:
            names[c] = "Gas & industrial"
        else:
            names[c] = "Moderate background"
    for c in prof.index:
        top = idx.loc[c].sort_values(ascending=False).head(3)
        parts = ", ".join(f"{p} ({v:.0f}% of average)" for p, v in top.items())
        meaning = {
            "Particulate-heavy": "Smoggy days dominated by fine and coarse dust. This is the unhealthiest pattern "
                                 "for lungs and is typical of winter smog and burning episodes.",
            "Gas & industrial": "Days with very high carbon monoxide, sulphur dioxide and nitrogen dioxide. "
                                "This points to industrial, coal-burning or heavy-traffic sources rather than dust.",
            "Moderate background": "Everyday city air. All pollutants sit near or below the national average, "
                                   "so this is the most common and the mildest pattern.",
        }[names[c]]
        why[c] = f"{meaning} Highest relative to average: {parts}."
    return names, why, idx


def show_air_clusters():
    page_header(
        "AIR QUALITY / PATTERN DISCOVERY",
        "Clusters & PCA",
        "What kinds of polluted air exist? K-Means sorts every day into a few "
        "pollution patterns; PCA draws them on one flat map.",
    )
    try:
        df, pcs = load_data()
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        st.error("The saved clustering or PCA dataset could not be loaded.")
        st.code(str(error))
        return

    names, why, idx = name_clusters(df)
    order = idx["PM2.5"].sort_values().index.tolist()          # mild -> severe
    cols = {c: COLORWAY[i] for i, c in enumerate(order)}
    label = {c: f"{names[c]}" for c in order}
    df["Pattern"] = df["Cluster"].map(label)

    # ------------------------------------------------------------ filters
    f1, f2, f3 = st.columns([1.3, 1.3, 1], gap="large")
    with f1:
        pick = st.multiselect("Pollution patterns", [label[c] for c in order],
                              default=[label[c] for c in order], key="ac_patterns")
    with f2:
        cities = sorted(df["City"].unique())
        pick_c = st.multiselect("Cities (empty = all)", cities, default=[], key="ac_cities")
    with f3:
        y0, y1 = int(df["Year"].min()), int(df["Year"].max())
        yr = st.slider("Years", y0, y1, (y0, y1), key="ac_years")

    f = df[df["Pattern"].isin(pick) & df["Year"].between(*yr)]
    if pick_c:
        f = f[f["City"].isin(pick_c)]
    if f.empty:
        st.warning("No observations match these filters.")
        return

    fe = pd.Series({c: n for c, n in zip(order, [label[c] for c in order])})
    heavy = order[-1]
    kpi_row([
        ("Patterns found", f"{df['Cluster'].nunique()}", "K-Means, 9 pollutants", TEAL),
        ("Days in view", f"{len(f):,}", f"of {len(df):,} total", "#34B3A0"),
        ("Share that is particulate-heavy", f"{(f['Cluster'] == heavy).mean() * 100:.0f}%",
         "Days dominated by dust and smoke", "#D9453D"),
        ("Cluster quality", "0.45", "Silhouette score (fair separation)", "#4C5BD4"),
    ])

    t1, t2, t3, t4 = st.tabs(["Regime profiles", "Spatio-temporal distribution", "PCA diagnostics", "Summary statistics"])

    # ------------------------------------------------------------ profiles
    with t1:
        st.markdown("### Pollution Regimes Identified by K-Means")
        cc = st.columns(len(order), gap="medium")
        for col, c in zip(cc, order):
            sub = f[f["Cluster"] == c]
            with col:
                share = len(sub) / len(f) * 100
                pm = sub["PM2.5"].mean() if len(sub) else float("nan")
                st.markdown(
                    f'<div class="ep-callout" style="--tone:{cols[c]};min-height:215px">'
                    f'<b style="font-size:1.15rem">{names[c]}</b><br>'
                    f'<span style="font-size:1.7rem;font-weight:800">{share:.0f}%</span> '
                    f'of days · PM2.5 avg <b>{pm:.0f}</b> µg/m³<br>'
                    f'<span style="font-size:.95rem">{why[c].split(" Highest")[0]}</span></div>',
                    unsafe_allow_html=True)

        st.markdown("### Regime Deviation from Baseline (Index, All-Days = 100)")
        prof = f.groupby("Pattern")[POLLS].mean()
        base = df[POLLS].mean()
        rel = (prof / base * 100).reindex([label[c] for c in order if label[c] in prof.index])
        fig = go.Figure(go.Heatmap(
            z=rel.values, x=POLLS, y=list(rel.index), zmin=0, zmax=300, xgap=3, ygap=3,
            colorscale=HEAT_SCALE,
            text=np.round(rel.values, 0), texttemplate="%{text:.0f}",
            textfont=dict(size=15, color="#0B2A2E"), colorbar=dict(title="% of average"),
            hovertemplate="%{y}<br>%{x}: %{z:.0f}% of the all-days average<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(style_fig(fig, 110 + 70 * len(rel), legend=False), use_container_width=True)
        st.caption("100 = the average day across all data. 200 means twice the average for that pollutant. "
                   "Using percentages lets pollutants measured in different units sit on one scale.")

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("### Mean PM2.5 and PM10 by Regime")
            tv = f.groupby("Pattern")[["PM2.5", "PM10"]].mean().reindex(rel.index)
            fig = go.Figure()
            for i, p in enumerate(["PM2.5", "PM10"]):
                fig.add_trace(go.Bar(x=tv.index, y=tv[p], name=p, marker_color=COLORWAY[[0, 2][i]],
                                     text=tv[p].round(0), textposition="outside",
                                     textfont=dict(size=13, color=INK)))
            fig.add_hline(y=INDIA_LIMIT, line_dash="dash", line_color="#D9453D",
                          annotation_text="India PM2.5 limit 60", annotation_font=dict(color="#D9453D", size=12))
            fig.update_yaxes(title="µg/m³")
            fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        with b:
            st.markdown("### CPCB AQI Category Composition per Regime")
            ct = f.dropna(subset=["AQI_Bucket"]).groupby(["Pattern", "AQI_Bucket"]).size().reset_index(name="n")
            ct["share"] = ct["n"] / ct.groupby("Pattern")["n"].transform("sum") * 100
            fig = px.bar(ct, x="share", y="Pattern", color="AQI_Bucket", orientation="h",
                         color_discrete_map=AQI_BUCKET_COLOURS,
                         category_orders={"AQI_Bucket": AQI_BUCKET_ORDER, "Pattern": list(rel.index)},
                         labels={"share": "% of days", "AQI_Bucket": "Category"})
            fig.update_traces(marker_line=dict(color="#FFFFFF", width=1))
            fig.update_xaxes(range=[0, 100])
            fig.update_yaxes(showgrid=False)
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
            st.caption("Shows that the patterns are not just AQI labels: they describe *what* the pollution is made of.")

    # ------------------------------------------------------------ where & when
    with t2:
        pat_order = [label[c] for c in order if label[c] in f["Pattern"].unique()]
        cmap = {label[c]: cols[c] for c in order}

        st.markdown("### Regime Frequency by City")
        n = st.slider("Cities shown (most particulate-heavy first)", 5, f["City"].nunique(),
                      min(14, f["City"].nunique()), key="ac_city_n")
        g = f.groupby(["City", "Pattern"]).size().reset_index(name="n")
        g["share"] = g["n"] / g.groupby("City")["n"].transform("sum") * 100
        heavy_lbl = label[heavy]
        rank = g[g["Pattern"] == heavy_lbl].set_index("City")["share"].reindex(f["City"].unique()).fillna(0)
        top = rank.sort_values(ascending=False).head(n).index.tolist()
        fig = px.bar(g[g["City"].isin(top)], x="share", y="City", color="Pattern", orientation="h",
                     color_discrete_map=cmap, category_orders={"City": top, "Pattern": pat_order},
                     labels={"share": "% of the city's days"})
        fig.update_traces(marker_line=dict(color="#FFFFFF", width=1))
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(style_fig(fig, 120 + 34 * n), use_container_width=True)

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("### Intra-Annual Regime Share")
            m = f.groupby(["Month", "Pattern"]).size().reset_index(name="n")
            m["share"] = m["n"] / m.groupby("Month")["n"].transform("sum") * 100
            m["Month name"] = m["Month"].map(lambda x: MONTHS[x - 1])
            fig = px.area(m, x="Month name", y="share", color="Pattern", color_discrete_map=cmap,
                          category_orders={"Month name": MONTHS, "Pattern": pat_order},
                          labels={"share": "% of days", "Month name": ""})
            fig.update_yaxes(range=[0, 100])
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        with b:
            st.markdown("### Inter-Annual Regime Share")
            y = f.groupby(["Year", "Pattern"]).size().reset_index(name="n")
            y["share"] = y["n"] / y.groupby("Year")["n"].transform("sum") * 100
            fig = px.bar(y, x="Year", y="share", color="Pattern", color_discrete_map=cmap,
                         category_orders={"Pattern": pat_order}, labels={"share": "% of days"})
            fig.update_traces(marker_line=dict(color="#FFFFFF", width=1))
            fig.update_xaxes(type="category")
            fig.update_yaxes(range=[0, 100])
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        mm = f.assign(h=f["Cluster"] == heavy).groupby("Month")["h"].mean() * 100
        if len(mm):
            callout(f"<b>{label[heavy]}</b> days peak in <b>{MONTHS[int(mm.idxmax()) - 1]}</b> "
                    f"({mm.max():.0f}% of that month's days) and are rarest in "
                    f"<b>{MONTHS[int(mm.idxmin()) - 1]}</b> ({mm.min():.0f}%).", tone="#D9453D")

    # ------------------------------------------------------------ PCA
    with t3:
        st.markdown(
            "### What is PCA showing?\n"
            "Each day is described by **nine pollutant readings**, which cannot be drawn on a flat chart. "
            "PCA builds a few **new axes (PC1, PC2 …)**, each a blend of pollutants, so that most of the "
            "difference between days survives. **Days close together on the map have similar air; days far apart "
            "differ.** The axes are blends, not single pollutants.")
        model = load_pca_model()
        if model is None:
            comps, evr, feats = None, None, POLLS
        else:
            comps, evr, feats = model

        # --- variance explained
        if evr is not None:
            a, b = st.columns([1, 1.2], gap="large")
            with a:
                st.markdown("#### Explained Variance per Principal Component")
                cum = np.cumsum(evr) * 100
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[f"PC{i + 1}" for i in range(len(evr))], y=evr * 100,
                                     name="This axis", marker_color=TEAL,
                                     text=[f"{v * 100:.0f}%" for v in evr], textposition="outside",
                                     textfont=dict(size=13, color=INK)))
                fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(evr))], y=cum,
                                         name="Running total", mode="lines+markers",
                                         line=dict(color="#F28C28", width=3), marker=dict(size=8)))
                fig.add_hline(y=95, line_dash="dot", line_color="#3C5559",
                              annotation_text="95%", annotation_position="bottom right")
                fig.update_yaxes(title="% of the variation in the data", range=[0, 112])
                fig.update_layout(legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(style_fig(fig, 400), use_container_width=True)
                st.caption(f"The first two axes keep {cum[1]:.0f}% of the information; eight axes keep {cum[-1]:.0f}%.")
            with b:
                st.markdown("#### Principal Component Loadings")
                k = min(3, len(comps))
                load = pd.DataFrame(comps[:k].T, index=feats, columns=[f"PC{i + 1}" for i in range(k)])
                fig = go.Figure(go.Heatmap(
                    z=load.values, x=list(load.columns), y=list(load.index), zmin=-0.7, zmax=0.7, xgap=3, ygap=3,
                    colorscale=DIVERGE_SCALE, text=np.round(load.values, 2), texttemplate="%{text:.2f}",
                    textfont=dict(size=14), colorbar=dict(title="Weight"),
                    hovertemplate="%{y} in %{x}: %{z:.2f}<extra></extra>"))
                fig.update_yaxes(autorange="reversed", showgrid=False)
                st.plotly_chart(style_fig(fig, 400, legend=False), use_container_width=True)
                st.caption("Weight shows how strongly a pollutant pushes a day to the right (+) or left (−) on that axis.")

            def describe(i):
                w = pd.Series(comps[i], index=feats)
                if (w > 0).all() or (w < 0).all():
                    top = w.abs().sort_values(ascending=False).head(3).index.tolist()
                    return f"<b>PC{i + 1}</b> = overall pollution level: every pollutant pushes the same way, " \
                           f"led by {', '.join(top)}. Further right = more polluted day."
                pos = w[w > 0.2].sort_values(ascending=False).head(3).index.tolist()
                neg = w[w < -0.2].sort_values().head(3).index.tolist()
                return f"<b>PC{i + 1}</b> = a contrast: {', '.join(pos) or '—'} ({PC_MEANING.get(pos[0], '')}) " \
                       f"against {', '.join(neg) or '—'} ({PC_MEANING.get(neg[0], '') if neg else ''})."
            callout("<br>".join(describe(i) for i in range(min(3, len(comps)))), tone=TEAL)

        # --- projection
        st.markdown("#### PCA Embedding Coloured by Regime")
        have = [p for p in pcs if f[p].notna().any()]
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2], gap="medium")
        with c1:
            ax = st.selectbox("Horizontal axis", have, index=0, key="ac_px")
        with c2:
            ay = st.selectbox("Vertical axis", have, index=min(1, len(have) - 1), key="ac_py")
        with c3:
            npts = st.slider("Points drawn", 500, min(8000, len(f)) if len(f) > 500 else 501,
                             min(3000, len(f)) if len(f) > 500 else 501, step=250, key="ac_pts")
        with c4:
            arrows = st.toggle("Show pollutant arrows", value=True, key="ac_arrows",
                               help="Arrows point in the direction each pollutant increases.")
        samp = f.dropna(subset=[ax, ay]).sample(min(npts, len(f)), random_state=7)
        fig = go.Figure()
        for c in order:
            nm = label[c]
            s = samp[samp["Cluster"] == c]
            if s.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=s[ax], y=s[ay], mode="markers", name=nm,
                marker=dict(size=6, color=cols[c], opacity=.5),
                customdata=np.stack([s["City"], s["Date"].dt.strftime("%Y-%m-%d"), s["PM2.5"].round(0)], axis=1),
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>PM2.5 %{customdata[2]} µg/m³<extra>" + nm + "</extra>"))
        for c in order:
            s = f[f["Cluster"] == c].dropna(subset=[ax, ay])
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=[s[ax].mean()], y=[s[ay].mean()], mode="markers+text", showlegend=False,
                marker=dict(size=22, color=cols[c], symbol="star", line=dict(color="#FFFFFF", width=2.5)),
                text=[f"<b>{names[c]}</b>"], textposition="top center", textfont=dict(size=14, color=INK),
                hovertemplate=f"Centre of {names[c]}<extra></extra>"))
        if arrows and comps is not None:
            ix, iy = int(ax[2:]) - 1, int(ay[2:]) - 1
            if ix < len(comps) and iy < len(comps):
                scale = 0.85 * min(samp[ax].abs().quantile(.97), samp[ay].abs().quantile(.97)) / \
                    max(np.abs(comps[[ix, iy]]).max(), 1e-9)
                for j, fe_ in enumerate(feats):
                    x1, y1 = comps[ix][j] * scale, comps[iy][j] * scale
                    if np.hypot(x1, y1) < 0.25 * scale * 0.5:
                        continue
                    fig.add_annotation(x=x1, y=y1, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                                       showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#0B3B44",
                                       text=f"<b>{fe_}</b>", font=dict(size=13, color="#0B3B44"),
                                       bgcolor="rgba(255,255,255,.75)")
        fig.add_hline(y=0, line_color="#CADBD8", line_width=1)
        fig.add_vline(x=0, line_color="#CADBD8", line_width=1)
        vx = f"{evr[int(ax[2:]) - 1] * 100:.0f}% of information" if evr is not None else ""
        vy = f"{evr[int(ay[2:]) - 1] * 100:.0f}% of information" if evr is not None else ""
        fig.update_xaxes(title=f"{ax} · {vx}", showgrid=True)
        fig.update_layout(legend=dict(orientation="h", y=1.08, title_text="Pattern  "))
        fig.update_yaxes(title=f"{ay} · {vy}")
        st.plotly_chart(style_fig(fig, 560), use_container_width=True)
        st.caption("Stars mark the centre of each pattern. Arrows show which pollutants increase in that direction: "
                   "a day far along an arrow is high in that pollutant.")

        # --- clusters vs components
        st.markdown("#### Regime Distribution along Principal Components")
        k = min(3, len(have))
        cmap = {label[c]: cols[c] for c in order}
        fig = go.Figure()
        for c in order:
            s = f[f["Cluster"] == c]
            for j, p in enumerate(have[:k]):
                fig.add_trace(go.Box(
                    y=s[p], x=[p] * len(s), name=label[c], legendgroup=label[c], showlegend=(j == 0),
                    marker_color=cols[c], boxpoints=False, line=dict(width=2),
                    offsetgroup=label[c]))
        fig.update_layout(boxmode="group", legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(title="Score on the axis", zeroline=True, zerolinecolor="#5E7477")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        sc = f.groupby("Pattern")[have[:k]].mean().reindex([label[c] for c in order if label[c] in f["Pattern"].unique()]).round(2)
        html_table(sc.reset_index(), 10)
        callout("If the boxes for different patterns barely overlap on an axis, that axis separates them well. "
                "The table gives each pattern's <b>average score</b> on the first axes.", tone=TEAL)

    # ------------------------------------------------------------ summary
    with t4:
        st.markdown("### Cluster Summary Statistics")
        rows = []
        for c in order:
            s = f[f["Cluster"] == c]
            if s.empty:
                continue
            aqi = s["AQI_Bucket"].mode()
            rows.append({
                "Pattern": names[c], "Cluster ID": c, "Days": f"{len(s):,}",
                "Share": f"{len(s) / len(f) * 100:.0f}%",
                "PM2.5": round(s["PM2.5"].mean(), 1), "PM10": round(s["PM10"].mean(), 1),
                "NO2": round(s["NO2"].mean(), 1), "SO2": round(s["SO2"].mean(), 1),
                "CO": round(s["CO"].mean(), 2), "O3": round(s["O3"].mean(), 1),
                "Days above India limit": f"{(s['PM2.5'] > INDIA_LIMIT).mean() * 100:.0f}%",
                "Most common AQI": aqi.iloc[0] if len(aqi) else "–",
            })
        tab = pd.DataFrame(rows)
        html_table(tab, 10)
        st.download_button("⬇ Download summary (CSV)", tab.to_csv(index=False).encode(),
                           file_name="ecopulse_air_cluster_summary.csv", mime="text/csv")
        st.caption("Averages are in µg/m³ except CO (mg/m³). Cluster IDs are K-Means labels; names are assigned from "
                   "each pattern's pollutant profile.")

        st.markdown("### Regime Interpretation")
        for c in order:
            st.markdown(f'<div class="ep-callout" style="--tone:{cols[c]}"><b>{names[c]}</b> — {why[c]}</div>',
                        unsafe_allow_html=True)

        with st.expander("How does K-Means clustering work?"):
            st.write("K-Means places three centre points among the days and assigns each day to the nearest one, "
                     "using all nine pollutants after putting them on a common scale. The centres then move to the "
                     "middle of their groups and the process repeats until it settles. The numbers 0, 1, 2 are just "
                     "labels, not severity ranks.")
        with st.expander("How good are the clusters?"):
            st.write("The saved three-cluster model has a silhouette score of about 0.45 (1 = perfectly separated, "
                     "0 = overlapping). That means the patterns are real but their edges blur, as expected with "
                     "continuous pollution data.")
        with st.expander("Why does PCA only show part of the picture?"):
            st.write("A flat map must drop information. The two chosen axes keep only part of the variation, so two "
                     "days that look close on the map can still differ in other pollutants. Use the axis selectors to "
                     "look at other angles.")
