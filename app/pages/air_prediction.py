from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (
    INDIA_LIMIT,
    WHO_LIMIT,
    PM25_BANDS,
    badge,
    callout,
    html_table,
    page_header,
    pm25_band,
    style_fig,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_FILES = {
    "Multiple Linear Regression": "air_linear_regression.pkl",
    "Ridge Regression": "air_ridge_regression.pkl",
    "Lasso Regression": "air_lasso_regression.pkl",
    "Decision Tree": "air_decision_tree.pkl",
    "Random Forest": "air_random_forest.pkl",
}

# (session-state key, label, step, default "typical day" value)
FIELDS = {
    "pm10": ("PM10 (µg/m³)", 1.0, 180.0),
    "no": ("NO (µg/m³)", 1.0, 25.0),
    "no2": ("NO₂ (µg/m³)", 1.0, 40.0),
    "nox": ("NOx (µg/m³)", 1.0, 55.0),
    "nh3": ("NH₃ (µg/m³)", 1.0, 30.0),
    "co": ("CO (mg/m³)", 0.1, 1.5),
    "so2": ("SO₂ (µg/m³)", 1.0, 15.0),
    "o3": ("O₃ (µg/m³)", 1.0, 35.0),
}

# Illustrative scenarios (not live readings).
PRESETS = {
    "Clean day": dict(pm10=45.0, no=5.0, no2=15.0, nox=20.0,
                         nh3=10.0, co=0.5, so2=8.0, o3=30.0),
    "Typical day": {k: v[2] for k, v in FIELDS.items()},
    "Winter smog": dict(pm10=420.0, no=60.0, no2=90.0, nox=140.0,
                           nh3=55.0, co=3.5, so2=28.0, o3=22.0),
}


@st.cache_resource
def load_air_model():
    return joblib.load(
        ROOT / "models" / "air_random_forest.pkl"
    )


@st.cache_resource
def load_named_model(filename):
    return joblib.load(ROOT / "models" / filename)


@st.cache_data
def load_cities():
    df = pd.read_csv(
        ROOT / "data" / "processed" / "air_pm25.csv",
        usecols=["City"],
    )

    return sorted(
        df["City"].dropna().unique().tolist()
    )


@st.cache_data
def load_city_averages():
    df = pd.read_csv(
        ROOT / "data" / "processed" / "air_pm25.csv",
        usecols=["City", "PM2.5"],
    )
    return df.groupby("City")["PM2.5"].mean().to_dict()


def _apply_preset(values):
    for key, value in values.items():
        st.session_state[f"ep_{key}"] = value


def _gauge(value):
    """Semi-circle gauge coloured with the CPCB categories."""
    lows = [0, 30, 60, 90, 120, 250]
    highs = [30, 60, 90, 120, 250, 300]
    steps = [
        dict(range=[lo, hi], color=band[2])
        for lo, hi, band in zip(lows, highs, PM25_BANDS)
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=min(value, 300),
            number=dict(suffix=" µg/m³", font=dict(size=34)),
            gauge=dict(
                axis=dict(range=[0, 300], tickvals=[0, 30, 60, 90, 120, 250]),
                bar=dict(color="#0B3B44", thickness=0.28),
                steps=steps,
                threshold=dict(
                    line=dict(color="#000000", width=4),
                    thickness=0.85,
                    value=INDIA_LIMIT,
                ),
            ),
        )
    )
    style_fig(fig, height=290, legend=False)
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=0))
    return fig


def show_air_prediction():

    page_header(
        "AIR INTELLIGENCE / MACHINE LEARNING",
        "PM2.5 Prediction",
        "Estimate particulate matter concentration "
        "using our trained Random Forest Regression model.",
    )

    st.info(
        "This model estimates PM2.5 from supplied pollutant "
        "measurements. It does not forecast future air quality."
    )

    # Make sure every input has a starting value (typical day).
    for key, (_, _, default) in FIELDS.items():
        st.session_state.setdefault(f"ep_{key}", default)

    top_left, top_right = st.columns([1, 2], gap="large")

    with top_left:
        city = st.selectbox(
            "Select City",
            load_cities(),
        )

    with top_right:
        st.markdown("**Quick scenarios** — fill the form in one click")
        preset_cols = st.columns(len(PRESETS))
        for col, (name, values) in zip(preset_cols, PRESETS.items()):
            with col:
                st.button(
                    name,
                    key=f"preset_{name}",
                    use_container_width=True,
                    on_click=_apply_preset,
                    args=(values,),
                )

    st.subheader("Pollutant Measurements")

    st.caption(
        "The default measurements are illustrative, "
        "not live environmental readings."
    )

    col1, col2, col3 = st.columns(
        3,
        gap="large",
    )

    def field(key):
        label, step, _ = FIELDS[key]
        return st.number_input(
            label,
            min_value=0.0,
            step=step,
            key=f"ep_{key}",
        )

    with col1:
        pm10 = field("pm10")
        no = field("no")
        no2 = field("no2")

    with col2:
        nox = field("nox")
        nh3 = field("nh3")
        co = field("co")

    with col3:
        so2 = field("so2")
        o3 = field("o3")

    if st.button(
        "Generate Prediction",
        use_container_width=True,
    ):

        sample = pd.DataFrame(
            [
                {
                    "City": city,
                    "PM10": pm10,
                    "NO": no,
                    "NO2": no2,
                    "NOx": nox,
                    "NH3": nh3,
                    "CO": co,
                    "SO2": so2,
                    "O3": o3,
                }
            ]
        )

        try:
            model = load_air_model()

            prediction = model.predict(sample)[0]
            value = max(0, prediction)

            st.success(
                "Prediction generated successfully!"
            )

            label, colour, advice = pm25_band(value)

            left, right = st.columns([1, 1], gap="large")

            with left:
                st.plotly_chart(_gauge(value), use_container_width=True)
                st.caption(
                    "Black marker = India's 24-hour limit "
                    f"({INDIA_LIMIT} µg/m³)."
                )

            with right:
                st.metric(
                    "Estimated PM2.5 Concentration",
                    f"{value:.2f} µg/m³",
                )

                st.markdown(
                    "Category: " + badge(label, colour),
                    unsafe_allow_html=True,
                )

                callout(
                    f"{advice}<br>"
                    f"That is <b>{value / WHO_LIMIT:.1f}×</b> the WHO "
                    f"guideline ({WHO_LIMIT} µg/m³) and "
                    f"<b>{value / INDIA_LIMIT:.1f}×</b> India's limit "
                    f"({INDIA_LIMIT} µg/m³).",
                    tone=colour,
                )

                city_avg = load_city_averages().get(city)
                if city_avg and city_avg > 0:
                    diff = (value - city_avg) / city_avg * 100
                    direction = "above" if diff >= 0 else "below"
                    st.caption(
                        f"{city}'s historical average is {city_avg:.1f} "
                        f"µg/m³, so this estimate is {abs(diff):.0f}% "
                        f"{direction} it."
                    )

            if pm10 > 0 and value > pm10:
                st.warning(
                    "The estimate is higher than the PM10 you entered. "
                    "PM2.5 is a part of PM10, so please double-check "
                    "your inputs."
                )

            st.caption(
                f"City: {city} | "
                "Model: Random Forest Regression"
            )

            with st.expander("Compare all five trained models on these inputs"):
                rows = []
                for name, filename in MODEL_FILES.items():
                    try:
                        other = load_named_model(filename)
                        rows.append(
                            {
                                "Model": name,
                                "Estimated PM2.5 (µg/m³)": round(
                                    max(0.0, float(other.predict(sample)[0])), 2
                                ),
                            }
                        )
                    except Exception:
                        continue

                if rows:
                    comp = pd.DataFrame(rows)
                    fig = go.Figure(
                        go.Bar(
                            x=comp["Model"],
                            y=comp["Estimated PM2.5 (µg/m³)"],
                            marker_color=[
                                pm25_band(v)[1]
                                for v in comp["Estimated PM2.5 (µg/m³)"]
                            ],
                            text=comp["Estimated PM2.5 (µg/m³)"],
                            textposition="outside",
                        )
                    )
                    fig.update_yaxes(title="PM2.5 (µg/m³)")
                    style_fig(fig, height=340, legend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    html_table(comp)
                    st.caption(
                        "Random Forest had the best test accuracy "
                        "(R² = 0.674) in the model evaluation, so it is the "
                        "model used above. Simpler linear models can "
                        "disagree noticeably."
                    )
                else:
                    st.info("The other model files could not be loaded.")

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )
