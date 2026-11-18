import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from theme import (
    donut_3d,
    COLORWAY, DIVERGE_SCALE, INK, SEQ_SCALE, TEAL, callout, html_table, page_header,
    style_fig,
)
from water_data import CLASS_ORDER, SHORT, PARAMS, load_water

CLUSTER_COLS = ["Dissolved oxygen", "BOD", "pH", "Conductivity", "Nitrate-N",
                "Fecal coliform", "Total coliform"]
LOGGED = {"BOD", "Conductivity", "Nitrate-N", "Fecal coliform", "Total coliform"}
PALETTE = COLORWAY


def _a(hex_colour, a):
    h = hex_colour.lstrip('#')
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


@st.cache_data(show_spinner=False)
def _clusters(k):
    df = load_water()
    d = df.dropna(subset=CLUSTER_COLS).copy()
    X = d[CLUSTER_COLS].copy()
    for c in LOGGED:
        X[c] = np.log10(X[c].clip(lower=0) + 1)
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    d["Group"] = km.labels_
    xy = PCA(n_components=2, random_state=42).fit(Z)
    pts = xy.transform(Z)
    d["PC1"], d["PC2"] = pts[:, 0], pts[:, 1]
    # name groups by median BOD (cleanest first)
    order = d.groupby("Group")["BOD"].median().sort_values().index.tolist()
    names = {g: f"Group {i + 1}" for i, g in enumerate(order)}
    d["Group"] = d["Group"].map(names)
    return d, xy.explained_variance_ratio_


def show_water_analytics():
    df = load_water()
    page_header(
        "WATER INTELLIGENCE / ANALYTICS",
        "Water Analytics",
        "How the measured parameters relate to each other and how monitoring "
        "records group together.",
    )

    t_corr, t_clu, t_dist = st.tabs([
        "Correlations", "Water groups & PCA", "Parameter spread",
    ])

    with t_corr:
        cols = [SHORT[p] for p in PARAMS]
        X = df[cols].copy()
        for c in ["Conductivity", "Fecal coliform", "Total coliform", "Fecal streptococci",
                  "BOD", "Nitrate-N"]:
            X[c] = np.log10(X[c].clip(lower=0) + 1)
        corr = X.corr(method="spearman").round(2)
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=DIVERGE_SCALE)
        st.plotly_chart(style_fig(fig, 520, legend=False), use_container_width=True)
        st.caption("Spearman rank correlation of mid-point values (robust to extreme readings).")
        pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
        strongest = pairs.abs().sort_values(ascending=False).head(3)
        txt = "; ".join(f"<b>{a} ↔ {b}</b> ({pairs[(a, b)]:+.2f})" for a, b in strongest.index)
        callout(f"Strongest relationships: {txt}.", tone="#0E7C86")

    with t_clu:
        k = st.slider("Number of groups", 2, 5, 3, key="wa_k")
        d, var = _clusters(k)
        a, b = st.columns([1.4, 1], gap="large")
        with a:
            st.markdown("### PCA Projection of Water Records")
            samp = d.sample(min(len(d), 3000), random_state=1)
            fig = px.scatter(samp, x="PC1", y="PC2", color="Group", opacity=0.6,
                             color_discrete_sequence=PALETTE,
                             hover_data=["State", "Year"],
                             labels={"PC1": f"PC1 ({var[0] * 100:.0f}% of variation)",
                                     "PC2": f"PC2 ({var[1] * 100:.0f}% of variation)"})
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        with b:
            st.markdown("### Cluster Size Distribution")
            n = d["Group"].value_counts().sort_index()
            fig = donut_3d(list(n.index), list(n.values), (PALETTE * 3)[:len(n)], 380)
            st.plotly_chart(fig, use_container_width=True)

        prof = d.groupby("Group")[CLUSTER_COLS].median().round(2)
        prof.insert(0, "Records", d.groupby("Group").size())
        prof["Usable (A–C) %"] = d.groupby("Group")["Screening class"].apply(
            lambda s: (s != "Below Class C").mean() * 100).round(0)
        st.markdown("### Cluster Centroid Profile (Median)")
        html_table(prof.reset_index(), 10)
        st.caption("Groups are ordered from lowest to highest median BOD. "
                   "Coliform, BOD, conductivity and nitrate are log-scaled before clustering "
                   "so a few extreme readings cannot dominate.")

    with t_dist:
        GUIDE = {  # parameter -> list of (value, label)
            "Dissolved oxygen": [(5, "min 5 mg/L")],
            "BOD": [(3, "max 3 mg/L")],
            "pH": [(6.5, "6.5"), (8.5, "8.5")],
            "Total coliform": [(500, "max 500")],
        }
        c1, c2, c3 = st.columns([1.3, 1, 1], gap="large")
        with c1:
            p = st.selectbox("Parameter", PARAMS, index=4, key="wa_param")
        with c2:
            by = st.radio("Compare by", ["State", "Year", "Screening class"], key="wa_by")
        with c3:
            style = st.radio("Chart", ["Range plot", "Distribution curves", "Violin"], key="wa_style")
        col = SHORT[p]
        log = col in LOGGED
        dd = df.dropna(subset=[col, "Screening class"] if by == "Screening class" else [col]).copy()
        if log:
            dd = dd[dd[col] > 0]
        if by == "State":
            n = st.slider("States shown (most records first)", 5, 25, 12, key="wa_nst")
            keep = dd.groupby("State").size().sort_values(ascending=False).head(n).index
            dd = dd[dd["State"].isin(keep)]
        elif by == "Year":
            dd["Year"] = dd["Year"].astype(str)
        gcol = by
        order_med = dd.groupby(gcol)[col].median().sort_values(ascending=False)
        if by == "Year":
            order_med = order_med.sort_index(key=lambda x: x.astype(int))
        elif by == "Screening class":
            order_med = order_med.reindex([c for c in CLASS_ORDER if c in order_med.index])
        groups = order_med.index.tolist()
        if dd.empty:
            st.info("No readings for this selection.")
        else:
            st.markdown(f"### {p} · spread by {by.lower()}")
            fig = go.Figure()
            if style == "Range plot":
                q = dd.groupby(gcol)[col].quantile([.1, .25, .5, .75, .9]).unstack()
                q = q.reindex(groups if by != "State" else order_med.index)
                ypos = list(q.index)
                for g_, r in q.iterrows():
                    fig.add_trace(go.Scatter(x=[r[.1], r[.9]], y=[g_, g_], mode="lines", showlegend=False,
                                             line=dict(color="#5E7477", width=2.5), hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=[r[.25], r[.75]], y=[g_, g_], mode="lines", showlegend=False,
                                             line=dict(color=TEAL, width=14), hoverinfo="skip", opacity=.55))
                fig.add_trace(go.Scatter(
                    x=q[.5], y=ypos, mode="markers", name="Median",
                    marker=dict(size=15, color="#FFFFFF", line=dict(color=INK, width=3)),
                    customdata=np.stack([q[.1], q[.25], q[.75], q[.9]], axis=1),
                    hovertemplate="%{y}<br>Median %{x:,.2f}<br>Middle half %{customdata[1]:,.2f}–%{customdata[2]:,.2f}"
                                  "<br>Typical range %{customdata[0]:,.2f}–%{customdata[3]:,.2f}<extra></extra>"))
                fig.update_yaxes(categoryorder="array", categoryarray=ypos[::-1], showgrid=True)
                fig.update_xaxes(type="log" if log else "linear", title=p, showgrid=True)
                height = 140 + 42 * len(ypos)
                cap = "Dot = median · thick bar = middle half of readings · thin line = 10th–90th percentile."
            elif style == "Distribution curves":
                few = groups if len(groups) <= 6 else None
                sel = few or st.multiselect(f"{by} to compare", groups, default=groups[:5], key="wa_ecdf")
                for i, g_ in enumerate(sel):
                    v = np.sort(dd.loc[dd[gcol] == g_, col].values)
                    fig.add_trace(go.Scatter(x=v, y=np.arange(1, len(v) + 1) / len(v) * 100, mode="lines", name=str(g_),
                                             line=dict(width=3.5, color=COLORWAY[i % len(COLORWAY)], shape="hv")))
                fig.update_xaxes(type="log" if log else "linear", title=p, showgrid=True)
                fig.update_yaxes(title="% of readings at or below this value", range=[0, 100])
                fig.update_layout(legend=dict(orientation="h", y=-0.2))
                height = 480
                cap = ("Each curve shows what share of readings fall below a value. A curve further to the left "
                       "means lower readings; a steep curve means readings are similar.")
            else:
                for i, g_ in enumerate(groups):
                    v = dd.loc[dd[gcol] == g_, col]
                    lo_, hi_ = v.quantile(.01), v.quantile(.99)
                    v = v[(v >= lo_) & (v <= hi_)]
                    fig.add_trace(go.Violin(
                        x=np.log10(v) if log else v, y=[str(g_)] * len(v), name=str(g_), orientation="h",
                        side="positive", width=1.6, points=False, meanline_visible=True,
                        box_visible=True, line_color=COLORWAY[i % len(COLORWAY)],
                        fillcolor=_a(COLORWAY[i % len(COLORWAY)], .35), showlegend=False))
                fig.update_yaxes(categoryorder="array", categoryarray=[str(g) for g in groups][::-1], showgrid=False)
                fig.update_xaxes(title=(f"log10 of {p}" if log else p), showgrid=True)
                height = 140 + 50 * len(groups)
                cap = ("Width = how common a value is; inner box = middle half; line = mean. "
                       "The most extreme 1% at each end is trimmed for readability.")
            if p in [k for k in (SHORT[x] for x in PARAMS)] and col in GUIDE and style != "Violin":
                for val, lab in GUIDE[col]:
                    fig.add_vline(x=val, line_dash="dash", line_color="#D9453D",
                                  annotation_text=lab, annotation_position="top",
                                  annotation_font=dict(color="#D9453D", size=12))
            st.plotly_chart(style_fig(fig, height, legend=(style == "Distribution curves")), use_container_width=True)
            st.caption(cap + (" Log scale is used because this parameter is strongly skewed." if log else "")
                       + (" Red dashed line = guideline." if col in GUIDE and style != "Violin" else ""))
            summ = dd.groupby(gcol)[col].agg(
                Readings="size", Median="median",
                P10=lambda s: s.quantile(.1), P90=lambda s: s.quantile(.9)).reindex(groups).round(2).reset_index()
            hi_g, lo_g = order_med.index[0], order_med.index[-1]
            callout(f"Highest median: <b>{hi_g}</b> ({order_med.iloc[0]:,.2f}); lowest: <b>{lo_g}</b> "
                    f"({order_med.iloc[-1]:,.2f}). Wider bars or flatter curves mean more variation between stations.",
                    tone=TEAL)
            with st.expander("Summary numbers"):
                html_table(summ, 40)
