
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import html_table, page_header, show_chart


ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    ROOT / "reports" / "air_model_comparison.csv"
)

REGRESSION_RESULTS = [
    {
        "Model": "Multiple Linear Regression",
        "MAE": 22.61,
        "RMSE": 36.05,
        "R2": 0.2086,
    },
    {
        "Model": "Ridge Regression",
        "MAE": 22.10,
        "RMSE": 35.57,
        "R2": 0.2296,
    },
    {
        "Model": "Lasso Regression",
        "MAE": 19.74,
        "RMSE": 30.95,
        "R2": 0.4168,
    },
    {
        "Model": "Decision Tree",
        "MAE": 14.95,
        "RMSE": 25.01,
        "R2": 0.6191,
    },
    {
        "Model": "Random Forest",
        "MAE": 13.75,
        "RMSE": 23.12,
        "R2": 0.6744,
    },
]

MODEL_DESCRIPTIONS = {
    "Multiple Linear Regression": (
        "Models PM2.5 as a linear combination of the "
        "input pollutant measurements and city information."
    ),
    "Ridge Regression": (
        "Uses linear regression with L2 regularization "
        "to reduce sensitivity to large coefficients."
    ),
    "Lasso Regression": (
        "Uses L1 regularization, which can shrink some "
        "feature coefficients to zero."
    ),
    "Decision Tree": (
        "Learns nonlinear decision rules by splitting "
        "the input feature space."
    ),
    "Random Forest": (
        "Combines predictions from multiple decision "
        "trees to improve stability and accuracy."
    ),
}

# Light-theme chart style (dark text on white) so every label is readable.
CHART_STYLE = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font_color": "#0B2A2E",
    "margin": dict(l=10, r=10, t=35, b=20),
}


@st.cache_data
def load_regression_results():
    if RESULTS_PATH.exists():
        saved = pd.read_csv(RESULTS_PATH)

        column_lookup = {
            str(column).lower().replace("_", "").replace(" ", ""): column
            for column in saved.columns
        }

        required = {
            "model": "Model",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R2",
        }

        if all(key in column_lookup for key in required):
            renamed = saved.rename(
                columns={
                    column_lookup[key]: target
                    for key, target in required.items()
                }
            )

            result = renamed[
                ["Model", "MAE", "RMSE", "R2"]
            ].copy()

            for metric in ["MAE", "RMSE", "R2"]:
                result[metric] = pd.to_numeric(
                    result[metric],
                    errors="coerce",
                )

            result = result.dropna(
                subset=["Model", "MAE", "RMSE", "R2"]
            )

            if not result.empty:
                return result, "Saved model comparison CSV"

    return (
        pd.DataFrame(REGRESSION_RESULTS),
        "Previously recorded training results",
    )


def show_ml_comparison():
    page_header(
        'ENVIRONMENTAL INTELLIGENCE / MODEL EVALUATION',
        'Machine Learning Comparison Centre',
        "Explore the performance of EcoPulse's supervised learning models and the results of its unsupervised learning experiments.",
    )

    results, source = load_regression_results()

    st.caption(
        f"Regression results source: {source}. "
        "Metrics were obtained from the historical "
        "air-quality dataset, not live predictions."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Regression Models",
            "Clustering & PCA",
            "Model Methodology",
        ]
    )

    with tab1:
        st.markdown("### Regression Performance")

        st.write(
            "Five regression algorithms were evaluated "
            "for estimating PM2.5 concentration."
        )

        results = results.sort_values(
            "RMSE",
            ascending=True,
        ).reset_index(drop=True)

        best = results.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Lowest RMSE",
                f"{best['RMSE']:.2f}",
            )

        with col2:
            st.metric(
                "Best Model",
                str(best["Model"]),
            )

        with col3:
            st.metric(
                "R² of Best Model",
                f"{best['R2']:.4f}",
            )

        st.caption(
            "MAE and RMSE are measured in µg/m³. "
            "Lower error values indicate better performance. "
            "Higher R² indicates more explained variation."
        )

        st.markdown("### Compare Model Metrics")

        metric_options = {
            "Mean Absolute Error (MAE)": "MAE",
            "Root Mean Squared Error (RMSE)": "RMSE",
            "R² Score": "R2",
        }

        selected_metric = st.selectbox(
            "Select evaluation metric",
            options=list(metric_options.keys()),
            key="ml_metric_selector",
        )

        metric = metric_options[selected_metric]

        ascending = metric != "R2"

        chart_data = results.sort_values(
            metric,
            ascending=ascending,
        )

        fig = px.bar(
            chart_data,
            x=metric,
            y="Model",
            orientation="h",
            text=metric,
            labels={
                "R2": "R² Score",
                "MAE": "MAE (µg/m³)",
                "RMSE": "RMSE (µg/m³)",
            },
        )

        # Best model (first row after sorting) is highlighted in orange.
        bar_colours = [
            "#E8590C" if i == 0 else "#0E7C86"
            for i in range(len(chart_data))
        ]

        fig.update_traces(
            texttemplate=(
                "%{x:.4f}"
                if metric == "R2"
                else "%{x:.2f}"
            ),
            textposition="outside",
            marker_color=bar_colours,
        )

        fig.update_layout(
            **CHART_STYLE,
            height=440,
            xaxis=dict(
                gridcolor="#E3EEE8"
            ),
            yaxis=dict(showgrid=False),
        )

        show_chart(
            fig,
            use_container_width=True,
        )

        st.markdown("### Complete Evaluation Table")

        display_results = results.rename(
            columns={
                "R2": "R²",
            }
        )

        html_table(display_results.round(4))

        st.markdown("### Understand Each Model")

        selected_model = st.selectbox(
            "Choose a model",
            options=results["Model"].tolist(),
            key="ml_model_explanation",
        )

        st.write(
            MODEL_DESCRIPTIONS.get(
                selected_model,
                "This model was included in the saved "
                "evaluation results.",
            )
        )

        selected_row = results[
            results["Model"] == selected_model
        ].iloc[0]

        st.write(
            f"**MAE:** {selected_row['MAE']:.2f} µg/m³"
        )

        st.write(
            f"**RMSE:** {selected_row['RMSE']:.2f} µg/m³"
        )

        st.write(
            f"**R²:** {selected_row['R2']:.4f}"
        )

    with tab2:
        st.markdown("### K-Means Clustering")

        st.write(
            "K-Means was used to identify groups of "
            "historical air-quality observations with "
            "similar pollutant characteristics."
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Number of Clusters",
                "3",
            )

        with c2:
            st.metric(
                "Silhouette Score",
                "0.449",
            )

        st.caption(
            "The silhouette score describes how well "
            "observations are separated into clusters. "
            "It is not a prediction-accuracy percentage."
        )

        st.divider()

        st.markdown("### Principal Component Analysis")

        st.write(
            "PCA was used to reduce the dimensionality "
            "of the environmental feature space while "
            "retaining most of its variation."
        )

        p1, p2 = st.columns(2)

        with p1:
            st.metric(
                "Retained Components",
                "8",
            )

        with p2:
            st.metric(
                "Explained Variance",
                "98.02%",
            )

        st.caption(
            "PCA explained variance is not the same "
            "as model prediction accuracy."
        )

        st.info(
            "K-Means and PCA are unsupervised methods. "
            "Their results cannot be directly compared "
            "with regression MAE, RMSE, or R²."
        )

    with tab3:
        st.markdown("### Dataset and Evaluation")

        st.write(
            "The air-quality analysis uses the Indian "
            "city-day dataset covering 2015–2020."
        )

        st.metric(
            "Processed Observations",
            "24,933",
        )

        st.markdown("#### Regression Experiment")

        st.write(
            "Target variable: PM2.5 concentration."
        )

        st.write(
            "Input features: City, PM10, NO, NO2, NOx, "
            "NH3, CO, SO2, and O3."
        )

        st.write(
            "Training and testing used a chronological "
            "80/20 split across the processed dataset."
        )

        st.write(
            "Training observations: 19,946."
        )

        st.write(
            "Testing observations: 4,987."
        )

        st.markdown("#### Metric Definitions")

        with st.expander("Mean Absolute Error (MAE)"):
            st.write(
                "MAE measures the average absolute "
                "difference between predicted and "
                "observed PM2.5 values."
            )

        with st.expander("Root Mean Squared Error (RMSE)"):
            st.write(
                "RMSE penalizes larger prediction errors "
                "more heavily than MAE."
            )

        with st.expander("Coefficient of Determination (R²)"):
            st.write(
                "R² measures model performance relative "
                "to predicting the test-set mean. "
                "Values closer to 1 generally indicate "
                "better explanatory performance."
            )

        with st.expander("Silhouette Score"):
            st.write(
                "The silhouette score evaluates how "
                "similar observations are to their own "
                "cluster compared with other clusters."
            )

        with st.expander("PCA Explained Variance"):
            st.write(
                "Explained variance describes how much "
                "of the dataset's variation is represented "
                "by the retained principal components."
            )

        st.divider()

        st.caption(
            "EcoPulse is an educational and analytical "
            "project. Its models are not official air-quality "
            "monitoring or forecasting services."
        )
