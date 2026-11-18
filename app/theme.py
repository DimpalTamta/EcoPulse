"""Shared look-and-feel for EcoPulse.

Everything visual lives here so every page looks the same:
  * colours (brand, CPCB PM2.5 categories, chart palettes)
  * ONE stylesheet (sidebar, buttons, inputs, tabs, cards, tables)
  * small HTML helpers (page header, section title, KPI cards, callouts)
  * style_fig(): gives every Plotly chart the same clean white look with
    dark, readable labels

Rule used throughout: dark backgrounds always get white text, light
backgrounds always get near-black text.

Nothing in this file touches data or models.
"""

import html
import re

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# COLOURS
# --------------------------------------------------------------------------

INK = "#0B2A2E"        # main text on light backgrounds (very high contrast)
MUTED = "#3C5559"      # secondary text on light backgrounds (~8:1 on white)
DARK = "#0B3B44"       # deep brand colour (headers, table heads)
TEAL = "#0E7C86"       # main chart colour
MINT = "#34B3A0"
SEAFOAM = "#BFE8E0"
BORDER = "#D5E3E1"
CARD = "#FFFFFF"
ACCENT = "#E8590C"     # action buttons
INDIGO = "#3B4BC0"     # download buttons
AMBER = "#FBBF24"      # sidebar highlight

# Categorical chart colours (distinct, readable on white).
COLORWAY = ["#1F5FA8", "#0E7C86", "#C0392B", "#5B7FA6",
            "#E07B5F", "#7A8CA3", "#2E8B9A", "#A94438"]

# Continuous gradients used by bars / heatmaps / maps.
SEQ_SCALE = [[0.0, "#CDEFE8"], [0.5, "#2AA6A0"], [1.0, "#0B3B44"]]
# Blue -> red theme used by every heatmap / map / gradient bar in the app.
HEAT_SCALE = [[0.0, "#2C6FB7"], [0.25, "#8FBBE3"], [0.5, "#F3EEE6"], [0.75, "#E9806B"], [1.0, "#A81E2D"]]
BLUE_RED_SCALE = HEAT_SCALE
DIVERGE_SCALE = [[0.0, "#2C6FB7"], [0.5, "#FFFFFF"], [1.0, "#A81E2D"]]
# Same heat gradient flipped (green = high), for "higher is healthier" values.
HEAT_SCALE_R = [[round(1.0 - pos, 2), col] for pos, col in reversed(HEAT_SCALE)]

# CPCB (India) 24-hour PM2.5 breakpoints in ug/m3, with traffic-light colours.
PM25_BANDS = [
    (30, "Good", "#1E9E5A",
     "Air is clean. Outdoor activity is fine for everyone."),
    (60, "Satisfactory", "#7CB82F",
     "Minor discomfort is possible for very sensitive people."),
    (90, "Moderate", "#E6B422",
     "People with asthma or heart disease may feel discomfort."),
    (120, "Poor", "#EE8A1F",
     "Breathing discomfort on long exposure. Limit heavy outdoor exercise."),
    (250, "Very Poor", "#D9453D",
     "Respiratory illness on prolonged exposure. Avoid outdoor exertion."),
    (float("inf"), "Severe", "#7A1F2B",
     "Affects healthy people too. Stay indoors and use a mask outside."),
]

# Same colours keyed by the AQI_Bucket names that exist in the dataset.
AQI_BUCKET_COLOURS = {name: colour for _, name, colour, _ in PM25_BANDS}
AQI_BUCKET_ORDER = [name for _, name, _, _ in PM25_BANDS]

# Reference limits for PM2.5 (24-hour average, ug/m3).
WHO_LIMIT = 15      # WHO 2021 guideline
INDIA_LIMIT = 60    # India NAAQS


def pm25_band(value):
    """Return (label, colour, advice) for a PM2.5 value in ug/m3."""
    for upper, label, colour, advice in PM25_BANDS:
        if value <= upper:
            return label, colour, advice
    return PM25_BANDS[-1][1:]



# --------------------------------------------------------------------------
# PALETTES  (user-selectable; applied to CSS *and* every Plotly chart)
# --------------------------------------------------------------------------
# The default look is written throughout this file (and the pages) with the
# DEFAULT colours below.  When another palette is chosen, every default colour
# is swapped for its palette equivalent at render time (CSS, KPI tones and
# Plotly figures), so one design system drives the whole app.  Traffic-light
# colours (AQI bands, water classes, limit lines) are never swapped because
# their meaning must not change.

DEFAULT_PALETTE = "EcoPulse Teal"

_ROLES = [
    "primary", "dark", "side1", "side2", "hdr_mid", "hdr_end", "mint", "mint2",
    "seafoam", "seq_light", "hover", "stripe", "border", "border2", "grid",
    "rowline", "bg1", "bg2", "bg3", "inputb", "muted", "ink", "axis",
    "acc", "acc1", "acc2", "ind1", "ind2", "ind3", "hi", "hi2", "hi3",
]

_DEFAULT_ROLES = dict(zip(_ROLES, [
    "#0E7C86", "#0B3B44", "#0A2F3A", "#09242D", "#0E6E7A", "#2A9D8F", "#34B3A0", "#2AA6A0",
    "#BFE8E0", "#CDEFE8", "#D7EFEB", "#F2F8F7", "#D5E3E1", "#CADBD8", "#E1EAE8",
    "#E1ECEA", "#E6F0EE", "#F4F8F7", "#EEF4F3", "#6F9A9E", "#3C5559", "#0B2A2E", "#5E7477",
    "#E8590C", "#C2410C", "#EA580C", "#2F3FA8", "#4C5BD4", "#3B4BC0", "#FBBF24", "#FBD66B", "#FDE68A",
]))

_DEFAULT_COLORWAY = ["#0E7C86", "#F28C28", "#4C5BD4", "#2FA866",
                     "#D6336C", "#8B5CF6", "#E0A800", "#64748B"]


def _pal(values, colorway):
    d = dict(zip(_ROLES, values))
    assert colorway[0] == d["primary"] and colorway[2] == d["ind2"]
    return {"roles": d, "colorway": colorway}


PALETTES = {
    DEFAULT_PALETTE: {"roles": dict(_DEFAULT_ROLES), "colorway": list(_DEFAULT_COLORWAY),
                      "swatch": ["#0B3B44", "#0E7C86", "#34B3A0", "#FBBF24"]},
    "Ocean Blue": _pal([
        "#0A6DB0", "#082F49", "#0B2A4A", "#071E36", "#0B5FA0", "#06A3C2", "#22B8CF", "#1B8FD0",
        "#BAE6FD", "#D2ECFB", "#DCEEFB", "#F1F7FC", "#D3E2EF", "#C5D8E8", "#E0EAF3",
        "#E3EDF6", "#E4EEF8", "#F3F8FC", "#EAF2F9", "#6C93B5", "#3D5568", "#0A2236", "#5C7489",
        "#0284C7", "#0369A1", "#0EA5E9", "#0F766E", "#6366F1", "#4F46E5", "#7DD3FC", "#A5E1FD", "#BAE6FD",
    ], ["#0A6DB0", "#06B6D4", "#6366F1", "#0D9488", "#F59E0B", "#EC4899", "#64748B", "#22C55E"]),
    "Forest Green": _pal([
        "#15803D", "#0F3D2E", "#0F3D2E", "#0A2B20", "#17703F", "#2FA66A", "#34D399", "#22A06B",
        "#BBF7D0", "#D3F5DE", "#DCF5E4", "#F2F9F4", "#D3E6DA", "#C6DDCE", "#E1EEE5",
        "#E3F0E7", "#E4F1E8", "#F3F9F5", "#EAF4ED", "#6E9C80", "#3D5A49", "#0B2A1A", "#5C7867",
        "#D97706", "#B45309", "#F59E0B", "#0E7490", "#0891B2", "#0E7490", "#A3E635", "#BEF264", "#D9F99D",
    ], ["#15803D", "#D97706", "#0891B2", "#65A30D", "#B45309", "#7C3AED", "#10B981", "#64748B"]),
    "Purple Modern": _pal([
        "#6D28D9", "#2E1065", "#2E1065", "#1E0A4A", "#5B21B6", "#818CF8", "#A78BFA", "#7C3AED",
        "#DDD6FE", "#E7E0FD", "#EDE9FE", "#F6F4FD", "#E0DAF2", "#D3CBEA", "#E9E5F5",
        "#ECE8F7", "#EDE9F8", "#F7F5FD", "#F1EEFB", "#8E7CC3", "#4C4568", "#1B1033", "#6A6288",
        "#DB2777", "#BE185D", "#EC4899", "#1D4ED8", "#2563EB", "#1E40AF", "#F9A8D4", "#FBCFE8", "#FCE7F3",
    ], ["#6D28D9", "#F59E0B", "#2563EB", "#0D9488", "#DB2777", "#8B5CF6", "#EA580C", "#64748B"]),
    "Clean Professional": _pal([
        "#2F5D9B", "#1F2937", "#1F2937", "#111827", "#334155", "#475569", "#60A5FA", "#3B7DD8",
        "#DBEAFE", "#E3EEFC", "#E8EEF7", "#F5F7FA", "#E2E6EC", "#D5DBE3", "#E9EDF2",
        "#EBEEF3", "#EEF1F5", "#F7F8FA", "#F1F3F6", "#94A3B8", "#475569", "#111827", "#64748B",
        "#2563EB", "#1D4ED8", "#3B82F6", "#334155", "#475569", "#1E293B", "#93C5FD", "#BFDBFE", "#DBEAFE",
    ], ["#2F5D9B", "#E67E22", "#475569", "#16A085", "#C0392B", "#8E44AD", "#F1C40F", "#7F8C8D"]),
}

def _mix(a, b, t):
    """Blend hex colour a toward b by t (0..1)."""
    a, b = a.lstrip("#"), b.lstrip("#")
    ca = [int(a[i:i + 2], 16) for i in (0, 2, 4)]
    cb = [int(b[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))


def _auto_pal(primary, dark, mid, accent, second, colorway_rest):
    """Build a full palette from a few base colours (neutrals are tints of the primary)."""
    W, K = "#FFFFFF", "#000000"
    roles = [
        primary, dark, dark, _mix(dark, K, .25), _mix(primary, dark, .25), mid, mid, _mix(primary, W, .15),
        _mix(mid, W, .72), _mix(primary, W, .82), _mix(primary, W, .88), _mix(primary, W, .95),
        _mix(primary, W, .80), _mix(primary, W, .74), _mix(primary, W, .88),
        _mix(primary, W, .86), _mix(primary, W, .90), _mix(primary, W, .95), _mix(primary, W, .92),
        _mix(primary, W, .45), _mix(dark, W, .25), _mix(dark, K, .35), _mix(dark, W, .40),
        accent, _mix(accent, K, .2), _mix(accent, W, .12), _mix(second, K, .25), second, _mix(second, K, .12),
        _mix(mid, W, .35), _mix(mid, W, .55), _mix(mid, W, .75),
    ]
    return _pal(roles, [primary, accent, second] + colorway_rest)


PALETTES["Sunset Orange"] = _auto_pal("#D9631E", "#431A08", "#F08A3C", "#1F5FA8", "#7C3AED",
                                      ["#2E8B9A", "#C0392B", "#E0A800", "#64748B", "#16A085"])
PALETTES["Royal Violet"] = _auto_pal("#6B2FBF", "#26104D", "#9B6BE0", "#F08A3C", "#2563EB",
                                     ["#D6336C", "#0D9488", "#E0A800", "#64748B", "#8B5CF6"])
PALETTES["Orchid Magenta"] = _auto_pal("#A6288F", "#3A0A36", "#D45FC0", "#F59E0B", "#4F46E5",
                                       ["#0E7C86", "#C0392B", "#E0A800", "#64748B", "#10B981"])
PALETTES["Crimson Red"] = _auto_pal("#B3262E", "#3A0A0E", "#E2606A", "#1F5FA8", "#475569",
                                    ["#E0A800", "#0E7C86", "#D6336C", "#64748B", "#F28C28"])
PALETTES["Amber Gold"] = _auto_pal("#C98A0B", "#3F2A04", "#EDB43F", "#1E3A8A", "#6D28D9",
                                   ["#0E7C86", "#C0392B", "#2FA866", "#64748B", "#D6336C"])
PALETTES["Midnight Navy"] = _auto_pal("#1E3A8A", "#0B1437", "#4F6FD0", "#F28C28", "#0E7C86",
                                      ["#D6336C", "#8B5CF6", "#E0A800", "#64748B", "#2FA866"])
PALETTES["Terracotta Sand"] = _auto_pal("#B5603E", "#3E2216", "#D9946F", "#0E7C86", "#4C5BD4",
                                        ["#E0A800", "#8B5CF6", "#C0392B", "#64748B", "#2FA866"])
PALETTES["Emerald Lime"] = _auto_pal("#0F9D58", "#073B22", "#4CC38A", "#E8590C", "#2563EB",
                                     ["#8B5CF6", "#E0A800", "#D6336C", "#64748B", "#0891B2"])

for _n, _p in PALETTES.items():
    _p.setdefault("swatch", [_p["roles"]["dark"], _p["roles"]["primary"],
                             _p["roles"]["mint"], _p["roles"]["hi"]])


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


_COLOUR_MAPS = {}


def current_palette_name():
    try:
        name = st.session_state.get("palette_choice", DEFAULT_PALETTE)
    except Exception:
        name = DEFAULT_PALETTE
    return name if name in PALETTES else DEFAULT_PALETTE


def colour_map(name=None):
    """{DEFAULT HEX (upper) -> palette hex} for the chosen palette."""
    name = name or current_palette_name()
    if name not in _COLOUR_MAPS:
        pal = PALETTES[name]
        m = {}
        for role in _ROLES:
            m[_DEFAULT_ROLES[role].upper()] = pal["roles"][role]
        for old, new in zip(_DEFAULT_COLORWAY, pal["colorway"]):
            m[old.upper()] = new
        _COLOUR_MAPS[name] = m
    return _COLOUR_MAPS[name]


def rc(colour):
    """Recolour one colour string for the active palette."""
    if isinstance(colour, str):
        return colour_map().get(colour.strip().upper(), colour)
    return colour


def palette_colorway():
    return PALETTES[current_palette_name()]["colorway"]


_HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")


def recolor_text(text):
    """Swap default colours inside CSS / HTML text for the active palette."""
    name = current_palette_name()
    if name == DEFAULT_PALETTE:
        return text
    m = colour_map(name)
    text = _HEX_RE.sub(lambda mo: m.get(mo.group(0).upper(), mo.group(0)), text)
    rgb_pairs = {}
    for role in ("primary", "dark", "acc1", "ind1", "hi"):
        rgb_pairs[_hex_to_rgb(_DEFAULT_ROLES[role])] = _hex_to_rgb(PALETTES[name]["roles"][role])
    for old, new in rgb_pairs.items():
        text = text.replace(f"rgba({old},", f"rgba({new},")
    return text


def _recolor_scale(scale, m):
    if isinstance(scale, (list, tuple)):
        out = []
        for item in scale:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append([item[0], m.get(str(item[1]).upper(), item[1])])
            elif isinstance(item, str):
                out.append(m.get(item.upper(), item))
            else:
                out.append(item)
        return out
    return scale


def _recolor_obj(obj, m, attrs):
    for a in attrs:
        try:
            v = getattr(obj, a)
        except Exception:
            continue
        if isinstance(v, str):
            nv = m.get(v.strip().upper())
            if nv:
                setattr(obj, a, nv)
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            setattr(obj, a, [m.get(x.strip().upper(), x) for x in v])


def recolor_fig(fig):
    """Swap default colours in a Plotly figure for the active palette.
    Semantic colours (AQI, classes, limit lines) are not in the map."""
    name = current_palette_name()
    tag = f"ep:{name}"
    try:
        if fig.layout.meta == tag:
            return fig
        fig.update_layout(meta=tag)
    except Exception:
        pass
    if name == DEFAULT_PALETTE:
        return fig
    m = colour_map(name)
    for tr in fig.data:
        for part in ("marker", "line", "textfont", "fillcolor"):
            obj = getattr(tr, part, None) if part != "fillcolor" else tr
            if obj is None:
                continue
            if part == "fillcolor":
                _recolor_obj(tr, m, ["fillcolor"])
                continue
            _recolor_obj(obj, m, ["color"])
            inner = getattr(obj, "line", None)
            if inner is not None and not isinstance(inner, str):
                _recolor_obj(inner, m, ["color"])
        try:
            sc = tr.colorscale
            if sc is not None:
                tr.colorscale = _recolor_scale(sc, m)
        except Exception:
            pass
        try:
            mc = tr.marker.colorscale
            if mc is not None:
                tr.marker.colorscale = _recolor_scale(mc, m)
        except Exception:
            pass
    try:
        ca = fig.layout.coloraxis
        if ca.colorscale is not None:
            ca.colorscale = _recolor_scale(ca.colorscale, m)
    except Exception:
        pass
    cw = fig.layout.colorway
    if cw:
        fig.layout.colorway = [m.get(c.upper(), c) for c in cw]
    for sh in (fig.layout.shapes or []):
        try:
            _recolor_obj(sh.line, m, ["color"])
        except Exception:
            pass
    return fig


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

def clean_html(markup):
    """Strip indentation so Markdown does not treat HTML as a code block."""
    return "\n".join(line.strip() for line in markup.strip().splitlines())


_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

:root { color-scheme: light; }

/* ================= base ================= */
html, body, .stApp {
    font-family: 'DM Sans', 'Segoe UI', Roboto, Arial, sans-serif;
    font-size: 17px;
    color: #0B2A2E;
}
.stApp { background: linear-gradient(180deg, #E6F0EE 0%, #F4F8F7 45%, #EEF4F3 100%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }
.block-container { max-width: 1320px; padding: 1.8rem 2.4rem 3rem; }
[data-testid="stVerticalBlock"] { gap: 1.05rem; }

p, li, label, .stMarkdown { line-height: 1.65; }
[data-testid="stMarkdownContainer"] { color: #0B2A2E; }
[data-testid="stMarkdownContainer"] p { font-size: 1.02rem; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p { color: #3C5559 !important; font-size: .97rem !important; }
hr { border-color: #CADBD8 !important; margin: 1.2rem 0 !important; }

h1, h2, h3, h4 {
    font-family: 'Outfit', 'DM Sans', sans-serif !important;
    color: #0B3B44 !important; letter-spacing: -0.02em; font-weight: 700 !important;
}
h1 { font-size: 2.0rem !important; }
h2 { font-size: 1.7rem !important; }
h3 { font-size: 1.5rem !important; margin: 1.1rem 0 .3rem !important; padding: 0 !important; }
h4 { font-size: 1.2rem !important; }

/* ================= sidebar ================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B3B44 0%, #0A2F3A 55%, #09242D 100%);
    border-right: 1px solid rgba(255,255,255,.10);
}
[data-testid="stSidebar"][aria-expanded="true"] { min-width: 336px; max-width: 336px; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.5rem 1.1rem 1.6rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .3rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #E9F6F4; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16) !important; margin: .9rem 0 !important; }
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] svg { color: #FFFFFF !important; fill: #FFFFFF !important; }

/* ---- brand ---- */
.eco-brand { display: flex; align-items: center; gap: .85rem; padding: .35rem .4rem .2rem; margin-bottom: 1.1rem; }
.eco-brand .mark {
    width: 54px; height: 54px; border-radius: 16px; flex: 0 0 54px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #34B3A0 0%, #0E7C86 55%, #FBBF24 140%);
    box-shadow: 0 8px 22px rgba(0,0,0,.38), inset 0 0 0 1px rgba(255,255,255,.28);
}
.eco-brand .mark svg { width: 34px; height: 34px; }
.eco-brand .name {
    font-family: 'Outfit', 'DM Sans', sans-serif; font-size: 2.35rem; font-weight: 800;
    color: #FFFFFF; letter-spacing: -.045em; line-height: 1;
}
.eco-brand .name span { color: #FBD66B; }
.eco-brand .sub {
    font-size: .74rem; font-weight: 700; letter-spacing: .2em; color: #BFE8E0;
    margin-top: 6px; white-space: nowrap; text-transform: uppercase;
}

/* ---- section headings ---- */
.eco-nav-group {
    display: flex; align-items: center; gap: .6rem;
    color: #FFFFFF; font-family: 'Outfit', 'DM Sans', sans-serif;
    font-size: 1.02rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase;
    margin: 1.5rem .2rem .45rem; padding-bottom: .5rem;
    border-bottom: 2px solid rgba(251,191,36,.55);
}
.eco-nav-group .dot { width: 11px; height: 11px; border-radius: 3px; background: #FBBF24; flex: 0 0 11px; }

/* ---- menu items ---- */
[data-testid="stSidebar"] .stButton > button {
    width: 100%; justify-content: flex-start; text-align: left;
    padding: .62rem 1rem !important; min-height: 50px;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 12px; box-shadow: none !important;
    transition: background .15s ease, transform .15s ease;
}
[data-testid="stSidebar"] .stButton > button > div { justify-content: flex-start !important; width: 100%; gap: .75rem; }
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span {
    color: #EAF7F5 !important; font-size: 1.1rem !important; font-weight: 600 !important;
    text-align: left !important; margin: 0 !important;
}
[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] { font-size: 1.4rem !important; }
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,.13) !important; transform: translateX(3px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: #FBBF24 !important; border-color: #FBBF24 !important;
    box-shadow: 0 8px 20px rgba(251,191,36,.32) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] span,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] span {
    color: #0B2A2E !important; font-weight: 800 !important;
}

/* ---- appearance (palette picker) in sidebar ---- */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #FFFFFF !important; font-size: 1.02rem !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: rgba(255,255,255,.96) !important; }
.eco-swatch { display: flex; gap: 5px; margin: .15rem .1rem .2rem; }
.eco-swatch i { flex: 1; height: 12px; border-radius: 6px; display: block; }
.eco-side-note {
    margin-top: .8rem; padding: 1rem 1.05rem; border-radius: 14px;
    background: rgba(251,191,36,.12); border: 1px solid rgba(251,191,36,.38);
    color: #F2FBFA; font-size: .97rem; line-height: 1.55;
}
.eco-side-note b { display: block; color: #FBD66B; font-size: .8rem; letter-spacing: .14em;
                   text-transform: uppercase; margin-bottom: .3rem; }

/* ================= buttons (main area) ================= */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #C2410C 0%, #EA580C 100%) !important;
    border: none !important; border-radius: 12px; min-height: 48px;
    padding: .65rem 1.4rem; box-shadow: 0 6px 16px rgba(194,65,12,.28);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton > button p, .stButton > button span,
[data-testid="stFormSubmitButton"] > button p {
    color: #FFFFFF !important; font-size: 1.04rem !important; font-weight: 800 !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px); filter: brightness(1.08);
    box-shadow: 0 10px 22px rgba(194,65,12,.38);
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #2F3FA8 0%, #4C5BD4 100%) !important;
    border: none !important; border-radius: 12px; min-height: 48px;
    padding: .65rem 1.4rem; box-shadow: 0 6px 16px rgba(47,63,168,.28);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stDownloadButton > button p, .stDownloadButton > button span {
    color: #FFFFFF !important; font-size: 1.04rem !important; font-weight: 800 !important;
}
.stDownloadButton > button:hover { transform: translateY(-2px); filter: brightness(1.1); }

/* popover trigger + body (map page filters) */
[data-testid="stPopover"] > button, [data-testid="stPopover"] button {
    background: #FFFFFF !important; border: 1.5px solid #6F9A9E !important;
    border-radius: 12px; min-height: 48px; box-shadow: none !important;
}
[data-testid="stPopover"] button p, [data-testid="stPopover"] button span {
    color: #0B2A2E !important; font-weight: 700 !important; font-size: 1rem !important;
}
[data-testid="stPopoverBody"] {
    background: #FFFFFF !important; border: 1px solid #CADBD8 !important;
    border-radius: 16px !important; box-shadow: 0 18px 45px rgba(11,42,46,.22) !important;
}
[data-testid="stPopoverBody"] p, [data-testid="stPopoverBody"] label,
[data-testid="stPopoverBody"] span { color: #0B2A2E; }
[data-testid="stPopoverBody"] .stButton > button p,
[data-testid="stPopoverBody"] .stButton > button span { color: #FFFFFF !important; }

/* ================= inputs ================= */
[data-testid="stWidgetLabel"] p {
    font-weight: 700 !important; color: #0B2A2E !important; font-size: 1.04rem !important;
}
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="textarea"] > div, [data-testid="stNumberInput"] > div > div {
    background: #FFFFFF !important; border: 1.5px solid #6F9A9E !important;
    border-radius: 12px !important; min-height: 46px;
    transition: border-color .15s ease, box-shadow .15s ease;
}
[data-baseweb="select"] > div:hover, [data-baseweb="input"] > div:hover { border-color: #0E7C86 !important; }
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"] > div:focus-within {
    border-color: #0E7C86 !important; box-shadow: 0 0 0 3px rgba(14,124,134,.22) !important;
}
[data-baseweb="select"] *, [data-baseweb="input"] input, .stTextInput input,
.stNumberInput input, .stDateInput input {
    color: #0B2A2E !important; font-size: 1rem !important; font-weight: 600 !important;
}
.stTextInput input, .stNumberInput input, .stDateInput input { background: #FFFFFF !important; }
[data-baseweb="select"] svg { fill: #0B2A2E !important; }
[data-baseweb="select"] [data-baseweb="tag"], [data-baseweb="tag"] {
    background: #0E7C86 !important; border-radius: 8px !important;
}
[data-baseweb="select"] [data-baseweb="tag"] *, [data-baseweb="tag"] * {
    color: #FFFFFF !important; font-weight: 700 !important;
}
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] {
    background: #FFFFFF !important;
}
[role="option"], [role="option"] * { color: #0B2A2E !important; font-weight: 600; font-size: 1rem; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background: #D7EFEB !important; }
[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {
    color: #0B2A2E !important; font-weight: 600 !important; font-size: 1rem !important;
}
[data-testid="stThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
    color: #0B2A2E !important; font-weight: 700 !important; font-size: .95rem !important;
}

/* ================= tabs ================= */
[data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #CADBD8; }
button[data-baseweb="tab"] {
    padding: 12px 20px; border-radius: 12px 12px 0 0; transition: background .15s ease;
}
button[data-baseweb="tab"] p {
    font-size: 1.06rem !important; font-weight: 700 !important; color: #3C5559 !important;
}
button[data-baseweb="tab"]:hover { background: rgba(14,124,134,.10); }
button[data-baseweb="tab"][aria-selected="true"] { background: #FFFFFF; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #0B2A2E !important; }
[data-baseweb="tab-highlight"] { background: #E8590C !important; height: 4px !important; }

/* ================= cards, alerts, expanders ================= */
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stAlert"] p { color: #0B2A2E !important; font-size: 1rem; }
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #D5E3E1; border-left: 6px solid #0E7C86;
    border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 6px 20px rgba(11,59,68,.07);
}
[data-testid="stMetricLabel"] p { color: #3C5559 !important; font-weight: 700 !important; font-size: .95rem !important; }
[data-testid="stMetricValue"] { color: #0B3B44 !important; font-family: 'Outfit', 'DM Sans', sans-serif; font-weight: 700; }
[data-testid="stExpander"] {
    background: #FFFFFF; border: 1px solid #D5E3E1; border-radius: 14px;
}
[data-testid="stExpander"] summary p { font-weight: 700 !important; font-size: 1.02rem !important; color: #0B2A2E !important; }
[data-testid="stPlotlyChart"] {
    background: #FFFFFF; border: 1px solid #D5E3E1; border-radius: 18px;
    padding: .7rem .8rem .3rem; box-shadow: 0 6px 22px rgba(11,59,68,.07);
}

/* ================= page header (dark -> white text) ================= */
.ep-header {
    padding: 1.35rem 1.9rem; border-radius: 20px; margin-bottom: .6rem;
    background: linear-gradient(115deg, #0B3B44 0%, #0E6E7A 62%, #2A9D8F 100%);
    box-shadow: 0 12px 30px rgba(11,59,68,.20);
}
.ep-header .ep-eyebrow {
    color: #FDE68A !important; font-size: .8rem; font-weight: 700;
    letter-spacing: .16em; text-transform: uppercase;
}
.ep-header .ep-title {
    font-family: 'Outfit', 'DM Sans', sans-serif; color: #FFFFFF !important;
    font-size: 2.0rem; font-weight: 700; letter-spacing: -.03em;
    line-height: 1.15; margin: .3rem 0 .35rem;
}
.ep-header .ep-sub { color: #E6F6F4 !important; font-size: 1.04rem; max-width: 820px; }

/* ================= section title ================= */
.ep-section { display: flex; align-items: center; gap: .8rem; margin: 1.6rem 0 .2rem; }
.ep-section .bar { width: 6px; height: 34px; border-radius: 4px;
                   background: linear-gradient(180deg, #E8590C, #FBBF24); }
.ep-section .t { font-family: 'Outfit', 'DM Sans', sans-serif; font-size: 1.6rem; font-weight: 700;
                 color: #0B3B44; letter-spacing: -.02em; line-height: 1.15; }
.ep-section .s { color: #3C5559; font-size: .98rem; margin-top: 2px; }

/* ================= KPI cards ================= */
.ep-kpi {
    background: #FFFFFF; border: 1px solid #D5E3E1; border-radius: 18px;
    padding: 1.05rem 1.25rem; height: 100%; min-height: 150px; box-sizing: border-box;
    display: flex; flex-direction: column; justify-content: flex-start;
    border-top: 6px solid var(--tone, #0E7C86);
    box-shadow: 0 6px 22px rgba(11,59,68,.08);
    transition: transform .18s ease, box-shadow .18s ease;
}
/* every box in a row gets the same width and height, whatever the text length */
div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { flex: 1 1 0 !important; min-width: 0 !important; }
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div:has(> div > .ep-kpi),
div[data-testid="stColumn"] div[data-testid="stElementContainer"]:has(.ep-kpi),
div[data-testid="stColumn"] div[data-testid="stElementContainer"]:has(.ep-nav-card),
div[data-testid="stColumn"] div[data-testid="stElementContainer"]:has(.ep-step) { height: 100%; }
div[data-testid="stColumn"] div[data-testid="stMarkdownContainer"]:has(.ep-kpi),
div[data-testid="stColumn"] div[data-testid="stMarkdownContainer"]:has(.ep-nav-card),
div[data-testid="stColumn"] div[data-testid="stMarkdownContainer"]:has(.ep-step) { height: 100%; }
.ep-nav-card, .ep-step { min-height: 150px; box-sizing: border-box; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #D5E3E1; border-radius:18px; padding:1rem 1.2rem; min-height:110px; box-shadow:0 6px 22px rgba(11,59,68,.08); }
.ep-kpi:hover { transform: translateY(-3px); box-shadow: 0 14px 30px rgba(11,59,68,.14); }
.ep-kpi .l { color: #3C5559; font-size: .82rem; font-weight: 800;
             letter-spacing: .08em; text-transform: uppercase; }
.ep-kpi .v { font-family: 'Outfit', 'DM Sans', sans-serif; color: #0B3B44;
             font-size: 2.0rem; font-weight: 700; line-height: 1.2; margin: .2rem 0; }
.ep-kpi .s { color: #3C5559; font-size: .94rem; }

/* ================= badges, callouts, nav cards ================= */
.ep-badge {
    display: inline-block; padding: .25rem .8rem; border-radius: 999px;
    color: #FFFFFF; font-weight: 700; font-size: .88rem; letter-spacing: .02em;
}
.ep-callout {
    border-radius: 14px; padding: 1rem 1.2rem; margin: .4rem 0 .6rem;
    background: #FFFFFF; border: 1px solid #D5E3E1;
    border-left: 7px solid var(--tone, #0E7C86); color: #0B2A2E; font-size: 1.03rem;
    box-shadow: 0 4px 16px rgba(11,59,68,.05);
}
.ep-callout b { color: #0B3B44; }
.ep-nav-card {
    background: #FFFFFF; border: 1px solid #D5E3E1; border-radius: 18px;
    border-top: 6px solid var(--tone, #0E7C86); padding: 1.15rem 1.25rem;
    min-height: 150px; box-shadow: 0 6px 22px rgba(11,59,68,.08);
}
.ep-nav-card .t { font-family: 'Outfit', 'DM Sans', sans-serif; font-size: 1.3rem; font-weight: 700; color: #0B3B44; }
.ep-nav-card .d { color: #3C5559; font-size: .98rem; margin-top: .35rem; line-height: 1.55; }

/* ================= AQI legend ================= */
.ep-legend { display: flex; gap: 6px; flex-wrap: wrap; margin: .3rem 0 .6rem; }
.ep-legend span {
    flex: 1 1 110px; text-align: center; color: #FFFFFF; font-weight: 800;
    font-size: .9rem; padding: .55rem .3rem; border-radius: 10px;
    text-shadow: 0 1px 2px rgba(0,0,0,.35);
}
.ep-legend small { display: block; font-weight: 600; opacity: .95; }

/* ================= steps ================= */
.ep-step { background: #FFFFFF; border: 1px solid #D5E3E1; border-radius: 16px; padding: 1.1rem 1.2rem; height: 100%; }
.ep-step .n {
    display: inline-flex; width: 34px; height: 34px; border-radius: 50%;
    background: #0E7C86; color: #FFFFFF; align-items: center;
    justify-content: center; font-weight: 800; margin-bottom: .5rem;
}
.ep-step .t { font-weight: 800; color: #0B3B44; }
.ep-step .d { color: #3C5559; font-size: .96rem; }

/* ================= html tables (no pyarrow needed) ================= */
.ep-table-wrap, .hotspot-table-wrap {
    overflow-x: auto; border-radius: 14px; border: 1px solid #D5E3E1; background: #FFFFFF;
    box-shadow: 0 4px 16px rgba(11,59,68,.05);
}
table.ep-table, table.hotspot-table, table.eco-peach-table {
    border-collapse: collapse; width: 100%; background: #FFFFFF; border: none;
}
table.ep-table th, table.hotspot-table th, table.eco-peach-table th {
    background: #0B3B44 !important; color: #FFFFFF !important; padding: .75rem .9rem;
    text-align: left !important; font-size: .92rem; font-weight: 700; white-space: nowrap;
}
table.ep-table td, table.hotspot-table td, table.eco-peach-table td {
    padding: .62rem .9rem; border-top: 1px solid #E1ECEA; color: #0B2A2E; font-size: .96rem;
}
table.ep-table tr:nth-child(even) td, table.hotspot-table tr:nth-child(even) td,
table.eco-peach-table tr:nth-child(even) td { background: #F2F8F7; }
table.ep-table tr:hover td, table.hotspot-table tr:hover td,
table.eco-peach-table tr:hover td { background: #D7EFEB; }

/* ================= footer ================= */
.eco-footer {
    text-align: center; color: #3C5559; font-size: .9rem; margin-top: 2.5rem;
    padding: 1.2rem 0; border-top: 1px solid #CADBD8;
}

@media (max-width: 768px) {
    .block-container { padding: 1.2rem 1rem 2rem; }
    .ep-header .ep-title { font-size: 1.6rem; }
}

/* ================= animated atlas backgrounds (subtle, home page) ================= */
@keyframes epDrift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.ep-header.anim {
    background: linear-gradient(115deg, #0B3B44, #0E6E7A, #2A9D8F, #0E7C86, #0B3B44);
    background-size: 320% 320%;
    animation: epDrift 22s ease-in-out infinite;
}
.ep-ambient .stApp, .stApp.ep-ambient {
    background: linear-gradient(120deg, #E6F0EE, #F4F8F7, #DCEFEA, #EEF4F3, #E6F0EE);
    background-size: 300% 300%;
    animation: epDrift 40s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
    .ep-header.anim, .ep-ambient .stApp { animation: none; }
}

/* ================= widget accent colour (sliders, radios, checkboxes) ================= */
[data-testid="stSlider"] [role="slider"] { background-color: #0E7C86 !important; box-shadow: 0 0 0 4px rgba(14,124,134,.2) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:nth-child(2) { background: #0E7C86 !important; }
[data-testid="stRadio"] [role="radio"][aria-checked="true"] > div:first-child,
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child[aria-checked="true"] { background-color: #0E7C86 !important; border-color: #0E7C86 !important; }
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
    background: #0E7C86 !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"] p { color: #FFFFFF !important; }
</style>
"""


def inject_theme(ambient=False):
    """Add the shared stylesheet (recoloured for the chosen palette).
    ambient=True adds the slow animated gradient page background (Home)."""
    css = recolor_text(_THEME_CSS)
    if ambient:
        css += "<style>.stApp { background: linear-gradient(120deg, " + rc("#E6F0EE") + ", " + rc("#F4F8F7") + \
               ", " + rc("#DCEFEA") + ", " + rc("#EEF4F3") + ", " + rc("#E6F0EE") + \
               "); background-size: 300% 300%; animation: epDrift 40s ease-in-out infinite; }</style>"
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HTML HELPERS
# --------------------------------------------------------------------------

def page_header(eyebrow, title, subtitle="", animated=False):
    st.markdown(
        clean_html(f"""
        <div class="ep-header{' anim' if animated else ''}">
            <div class="ep-eyebrow">{html.escape(eyebrow)}</div>
            <div class="ep-title">{html.escape(title)}</div>
            <div class="ep-sub">{html.escape(subtitle)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=""):
    """Large section heading with a colour bar (used to split long pages)."""
    sub = f'<div class="s">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        clean_html(f"""
        <div class="ep-section"><div class="bar"></div>
        <div><div class="t">{html.escape(title)}</div>{sub}</div></div>
        """),
        unsafe_allow_html=True,
    )


def kpi_card(label, value, sub="", tone=TEAL):
    """Return HTML for one KPI card (render inside st.markdown)."""
    return clean_html(f"""
    <div class="ep-kpi" style="--tone:{rc(tone)}">
        <div class="l">{html.escape(str(label))}</div>
        <div class="v">{html.escape(str(value))}</div>
        <div class="s">{html.escape(str(sub))}</div>
    </div>
    """)


def kpi_row(items):
    """items = [(label, value, sub, tone), ...] shown side by side."""
    cols = st.columns(len(items), gap="medium")
    for col, item in zip(cols, items):
        item = list(item)
        while len(item) < 3:
            item.append("")
        if len(item) < 4:
            item.append(TEAL)
        label, value, sub, tone = item[:4]
        with col:
            st.markdown(kpi_card(label, value, sub, tone), unsafe_allow_html=True)


def nav_card(title, text, tone=TEAL):
    """Small link-style card for the Home page (put a button under it)."""
    st.markdown(
        clean_html(f"""
        <div class="ep-nav-card" style="--tone:{rc(tone)}">
            <div class="t">{html.escape(title)}</div>
            <div class="d">{html.escape(text)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def badge(text, colour):
    return (
        f'<span class="ep-badge" style="background:{colour}">'
        f'{html.escape(str(text))}</span>'
    )


def callout(text_html, tone=TEAL):
    """text_html may contain <b>; pass only trusted strings."""
    st.markdown(
        f'<div class="ep-callout" style="--tone:{rc(tone)}">{text_html}</div>',
        unsafe_allow_html=True,
    )


def aqi_legend():
    parts = []
    lows = [0, 31, 61, 91, 121, 251]
    for (upper, label, colour, _), low in zip(PM25_BANDS, lows):
        rng = f"{low}+" if upper == float("inf") else f"{low}–{upper}"
        parts.append(
            f'<span style="background:{colour}">{label}<small>{rng} µg/m³</small></span>'
        )
    st.markdown(
        f'<div class="ep-legend">{"".join(parts)}</div>', unsafe_allow_html=True
    )


def html_table(df, max_rows=200):
    """Render a DataFrame as styled HTML (avoids the pyarrow requirement)."""
    shown = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    rows = []
    for _, row in shown.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f'<div class="ep-table-wrap"><table class="ep-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody>'
        f"</table></div>",
        unsafe_allow_html=True,
    )
    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows:,} of {len(df):,} rows.")


# --------------------------------------------------------------------------
# PLOTLY
# --------------------------------------------------------------------------

_AXIS_LINE = "#5E7477"
_GRID = "#E1EAE8"


def style_fig(fig, height=420, legend=True):
    """Give any Plotly figure the EcoPulse look: white background, dark and
    clearly visible labels, palette colours, hover-friendly."""
    ink, dark, border = rc(INK), rc(DARK), rc(BORDER)
    axis, grid = rc(_AXIS_LINE), rc(_GRID)
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=COLORWAY,
        font=dict(family="DM Sans, Segoe UI, Arial, sans-serif", color=ink, size=14),
        margin=dict(l=12, r=16, t=34, b=12),
        hoverlabel=dict(bgcolor=dark, font_color="#FFFFFF", font_size=14),
        showlegend=legend,
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)", bordercolor=border, borderwidth=1,
            font=dict(size=14, color=ink),
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=axis, linewidth=1.5,
        tickfont=dict(size=13, color=ink), title_font=dict(size=15, color=ink),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=grid, zeroline=False, linecolor=axis,
        linewidth=1.5, tickfont=dict(size=13, color=ink),
        title_font=dict(size=15, color=ink), automargin=True,
    )
    fig.update_layout(title_font=dict(size=17, color=ink))
    fig.update_annotations(font=dict(size=13, color=ink))
    fig.update_traces(textfont_size=14, selector=dict(type="pie"))
    recolor_fig(fig)
    diversify_bars(fig)
    return fig


def diversify_bars(fig):
    """Single-colour bar charts get a per-bar gradient (primary -> mid -> accent) so they
    are not monotonous. Charts that already colour by category/value are left alone."""
    try:
        bars_ = [t for t in fig.data if t.type == "bar"]
        if len(bars_) != 1 or len(fig.data) != 1:
            return fig
        tr = bars_[0]
        col = tr.marker.color
        if col is not None and not isinstance(col, str):
            return fig
        horiz = tr.orientation == "h"
        vals = tr.x if horiz else tr.y
        if vals is None or len(vals) < 3 or len(vals) > 80:
            return fig
        v = pd.to_numeric(pd.Series(list(vals)), errors="coerce").fillna(0)
        r = v.rank(method="first").sub(1).div(max(len(v) - 1, 1)).tolist()
        roles = PALETTES[current_palette_name()]["roles"]
        stops = [roles["primary"], roles["hdr_end"], roles["acc"]]

        def grad(t):
            return _mix(stops[0], stops[1], t * 2) if t < .5 else _mix(stops[1], stops[2], (t - .5) * 2)
        tr.marker.color = [grad(t) for t in r]
    except Exception:
        pass
    return fig


def show_chart(fig, **kwargs):
    """Apply the EcoPulse look to a finished figure (keeps its own height,
    legend setting and margins) and draw it full width."""
    height = fig.layout.height or 420
    legend = True if fig.layout.showlegend is None else fig.layout.showlegend
    margin = fig.layout.margin
    own_margin = [margin.l, margin.r, margin.t, margin.b] if margin and margin.l is not None else None
    style_fig(fig, height=height, legend=legend)
    if own_margin:
        fig.update_layout(margin=dict(l=own_margin[0], r=own_margin[1], t=own_margin[2], b=own_margin[3]))
    kwargs.setdefault("use_container_width", True)
    st.plotly_chart(fig, **kwargs)


# --------------------------------------------------------------------------
# Make every st.plotly_chart follow the chosen palette (even pages that do
# not call style_fig), without touching those pages.
# --------------------------------------------------------------------------

if not getattr(st, "_ep_plotly_patched", False):
    _orig_plotly_chart = st.plotly_chart

    def _themed_plotly_chart(figure_or_data, *args, **kwargs):
        try:
            recolor_fig(figure_or_data)
            diversify_bars(figure_or_data)
        except Exception:
            pass
        return _orig_plotly_chart(figure_or_data, *args, **kwargs)

    st.plotly_chart = _themed_plotly_chart
    st._ep_plotly_patched = True


# --------------------------------------------------------------------------
# SIDEBAR HELPERS
# --------------------------------------------------------------------------

_LOGO_SVG = (
    '<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M24 5c10 7 15 15 15 22a15 15 0 0 1-30 0c0-7 5-15 15-22Z" fill="rgba(255,255,255,.18)" '
    'stroke="#fff" stroke-width="2.4" stroke-linejoin="round"/>'
    '<path d="M8 27h9l4-9 6 17 4-8h9" stroke="#FBD66B" stroke-width="3" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def sidebar_brand():
    st.markdown(
        clean_html(f"""
        <div class="eco-brand">
            <div class="mark">{_LOGO_SVG}</div>
            <div>
                <div class="name">Eco<span>Pulse</span></div>
                <div class="sub">Air &amp; Water Intelligence</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def nav_group(title):
    st.markdown(
        f'<div class="eco-nav-group"><span class="dot"></span>{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def palette_picker():
    """Colour-palette selector (stored in session_state['palette_choice'])."""
    names = list(PALETTES)
    st.session_state.setdefault("palette_choice", DEFAULT_PALETTE)
    st.selectbox("Colour palette", names, key="palette_choice")
    sw = PALETTES[current_palette_name()]["swatch"]
    st.markdown(
        '<div class="eco-swatch">' + "".join(f'<i style="background:{c}"></i>' for c in sw) + "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# SHARED CHART HELPERS
# --------------------------------------------------------------------------

def donut_3d(labels, values, colours, height=340, hole=0.5):
    """Donut with a stacked-shadow depth effect (3D look) built from Plotly pies."""
    import plotly.graph_objects as go

    def _shade(hex_c, f):
        h = hex_c.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return "#%02X%02X%02X" % (int(r * f), int(g * f), int(b * f))

    fig = go.Figure()
    for i, off in enumerate((0.0, 0.012, 0.024, 0.036)):      # depth layers, back to front
        fig.add_trace(go.Pie(
            labels=labels, values=values, hole=hole, sort=False, direction="clockwise",
            domain=dict(x=[0.08, 0.92], y=[0.0 + 0.04 - off + 0.0, 0.90 - off]),
            marker=dict(colors=[_shade(c, 0.55) for c in colours], line=dict(color="#FFFFFF", width=1)),
            textinfo="none", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Pie(
        labels=labels, values=values, hole=hole, sort=False, direction="clockwise",
        domain=dict(x=[0.08, 0.92], y=[0.08, 0.98]),
        pull=[0.02] * len(values),
        marker=dict(colors=list(colours), line=dict(color="#FFFFFF", width=2)),
        textinfo="percent", textfont=dict(size=13, color="#FFFFFF"),
        hovertemplate="%{label}<br>%{value:,} (%{percent})<extra></extra>"))
    return style_fig(fig, height)


def clean_matrix(pv, fill=True):
    """Drop empty rows/columns and fill gaps left-right so no blank cells remain."""
    pv = pv.dropna(how="all").dropna(axis=1, how="all")
    if fill:
        pv = pv.T.interpolate(limit_direction="both").T
        pv = pv.apply(lambda r: r.fillna(r.mean()), axis=1)
    return pv


_INDIA_GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"


def india_state_geojson():
    """State boundary polygons (fetched once, cached). Returns None if offline."""
    import json, urllib.request
    cache = {}
    try:
        import streamlit as _st

        @_st.cache_data(show_spinner=False)
        def _load():
            with urllib.request.urlopen(_INDIA_GEOJSON_URL, timeout=8) as r:
                return json.loads(r.read().decode("utf-8"))
        return _load()
    except Exception:
        return None


def india_map_layers(line_colour="#12355B", fill_colour="rgba(44,111,183,0.07)"):
    """Mapbox-style layers drawing vivid state borders (and a faint fill) over the base map."""
    gj = india_state_geojson()
    if not gj:
        return []
    return [
        dict(sourcetype="geojson", source=gj, type="fill", color=fill_colour, below="traces"),
        dict(sourcetype="geojson", source=gj, type="line", color=line_colour,
             line=dict(width=1.6), below="traces"),
    ]


def fit_zoom(lat, lon, default=3.6):
    """Centre and zoom that frame the given coordinates (used to auto-zoom the map)."""
    import numpy as np
    lat, lon = list(lat), list(lon)
    if not lat:
        return {"lat": 22.5, "lon": 79.0}, default
    span = max(max(lat) - min(lat), (max(lon) - min(lon)) * 0.8, 0.5)
    zoom = float(np.clip(np.log2(360 / (span * 1.6)), 3.4, 9.0))
    return {"lat": (max(lat) + min(lat)) / 2, "lon": (max(lon) + min(lon)) / 2}, zoom


MAP_STYLES = {
    "Light (Carto Positron)": "carto-positron",
    "Street (OpenStreetMap)": "open-street-map",
    "Voyager (Carto)": "carto-voyager",
    "Dark (Carto Dark Matter)": "carto-darkmatter",
}
