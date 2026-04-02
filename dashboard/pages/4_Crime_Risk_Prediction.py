from pathlib import Path
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.shared import get_bundle, get_model_metadata, get_models
from crime_analytics.services.model_service import predict


def confidence_badge(confidence: float) -> tuple[str, str, str]:
    if confidence >= 0.75:
        return "High Confidence", "#14532d", "#bbf7d0"
    if confidence >= 0.60:
        return "Moderate Confidence", "#78350f", "#fef3c7"
    return "Low Confidence", "#7f1d1d", "#fecaca"


def render_class_probabilities(class_probabilities: dict[str, float], predicted_label: str) -> None:
    sorted_probs = sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)

    for label, prob in sorted_probs:
        label_col, bar_col, value_col = st.columns([2, 7, 2])
        label_text = f"**{label}**" if label == predicted_label else label
        label_col.markdown(label_text)
        bar_col.progress(float(prob))
        value_col.markdown(f"**{prob:.2%}**")


def build_forecast_trend(state_history: pd.DataFrame, state: str, start_year: int, population: float) -> pd.DataFrame:
    state_df = state_history[state_history["state"].astype(str).str.lower() == state.lower()].copy()
    if state_df.empty:
        return pd.DataFrame(columns=["year", "predicted_crime_rate"])

    state_df = state_df.sort_values("year")
    trend_source = state_df.get("previous_year_crime_rate")
    if trend_source is None:
        trend_source = state_df.get("prev_year_crime_rate")
    if trend_source is None:
        trend_source = state_df.get("crime_rate")
    if trend_source is None:
        return pd.DataFrame(columns=["year", "predicted_crime_rate"])

    trend_df = pd.DataFrame({"year": state_df["year"], "trend": pd.to_numeric(trend_source, errors="coerce")})
    if "crime_rate" in state_df.columns:
        trend_df["trend"] = trend_df["trend"].fillna(pd.to_numeric(state_df["crime_rate"], errors="coerce"))
    trend_df = trend_df.dropna(subset=["year", "trend"])
    if trend_df.empty:
        return pd.DataFrame(columns=["year", "predicted_crime_rate"])

    if len(trend_df) >= 2:
        slope, intercept = np.polyfit(trend_df["year"].astype(float), trend_df["trend"].astype(float), 1)
    else:
        slope = 0.0
        intercept = float(trend_df["trend"].iloc[-1])

    rows = []
    for step in range(1, 4):
        forecast_year = int(start_year) + step
        projected_prev_rate = max(0.0, float(slope * (forecast_year - 1) + intercept))
        forecast_result = predict(
            models=models,
            state=state,
            year=forecast_year,
            prev_year_crime_rate=projected_prev_rate,
            population=float(population),
        )
        rows.append({"year": forecast_year, "predicted_crime_rate": float(forecast_result["predicted_crime_rate"])})

    return pd.DataFrame(rows)

st.set_page_config(page_title="Crime Risk Prediction", page_icon="🤖", layout="wide")

st.title("Crime Risk Prediction")
st.caption("Random Forest based macro-level risk estimation")
st.info(
    "Risk category is assigned from predicted crime rate thresholds: "
    "LOW (<150), MEDIUM (150-299.99), HIGH (>=300)."
)
st.info(
    "Predictions for future years are forecast estimates based on historical trends and should be interpreted as advisory projections."
)

bundle = get_bundle()
models = get_models()
metadata = get_model_metadata()
state_df = bundle.state_year_data
max_year = int(state_df["year"].max())

st.info(f"Historical data available up to {max_year}.")
st.caption(
    "Historical prediction uses observed data patterns inside the dataset range. "
    "Forecast prediction extrapolates beyond the latest available year."
)

states = sorted(state_df["state"].unique().tolist())
state = st.selectbox("State", states)
year = st.number_input("Prediction year", min_value=1900, value=max_year + 1, step=1)

state_hist = state_df[state_df["state"] == state].sort_values("year")
default_prev_rate = float(state_hist.iloc[-1]["crime_rate"])
default_population = float(state_hist.iloc[-1]["population"]) if state_hist.iloc[-1]["population"] == state_hist.iloc[-1]["population"] else 0.0

prev_rate = st.number_input("Previous year crime rate", min_value=0.0, value=default_prev_rate)
population = st.number_input("Population", min_value=0.0, value=default_population)

if st.button("Predict risk"):
    is_forecast = int(year) > max_year
    prediction_type = "Forecast" if is_forecast else "Historical"
    result = predict(
        models=models,
        state=state,
        year=int(year),
        prev_year_crime_rate=float(prev_rate),
        population=float(population),
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Crime Rate", f"{result['predicted_crime_rate']:.2f}")
    c2.metric("Predicted Risk Category", result["predicted_risk_category"])
    c3.metric("Prediction Type", prediction_type)

    if is_forecast:
        st.warning("Forecast Mode: Prediction is based on extrapolation of historical trends.")
    else:
        st.success("Historical Mode: Prediction based on observed data patterns.")

    with st.container(border=True):
        st.markdown("**Historical vs Forecast Prediction**")
        st.markdown(
            "- Historical prediction uses years already present in the dataset and reflects observed patterns.\n"
            "- Forecast prediction extends beyond the latest dataset year and estimates future values from trend behavior.\n"
            "- Forecast results are advisory and should be reviewed with domain context before use."
        )

    st.caption(
        "Threshold rules: LOW (<150), MEDIUM (150 <= rate < 300), HIGH (>=300)."
    )

    badge_label, badge_text_color, badge_bg = confidence_badge(float(result["confidence"]))
    st.markdown(
        (
            f"<div style='padding:10px 12px;border-radius:8px;"
            f"background:{badge_bg};color:{badge_text_color};font-weight:700;'>"
            f"{badge_label}</div>"
        ),
        unsafe_allow_html=True,
    )

    if result["is_uncertain"]:
        st.warning(result["uncertainty_reason"])
    else:
        st.success("Prediction confidence is above the uncertainty threshold.")

    st.subheader("Class Probabilities")
    render_class_probabilities(
        class_probabilities=result["class_probabilities"],
        predicted_label=result["predicted_risk_category"],
    )

    forecast_df = build_forecast_trend(state_df, state=state, start_year=int(year), population=float(population))
    if not forecast_df.empty:
        st.subheader("3-Year Forecast Trend")
        fig = px.line(
            forecast_df,
            x="year",
            y="predicted_crime_rate",
            markers=True,
            labels={"year": "Year", "predicted_crime_rate": "Predicted Crime Rate"},
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Forecast trend is driven by the historical previous-year crime-rate pattern for the selected state.")

    st.subheader("Input Sensitivity Check")
    scenario_prev = predict(
        models=models,
        state=state,
        year=int(year),
        prev_year_crime_rate=float(prev_rate) * 1.10,
        population=float(population),
        include_forecast_trend=False,
    )
    scenario_pop = predict(
        models=models,
        state=state,
        year=int(year),
        prev_year_crime_rate=float(prev_rate),
        population=float(population) * 1.10,
        include_forecast_trend=False,
    )

    s1, s2 = st.columns(2)
    s1.metric(
        "If previous_year_crime_rate +10%",
        f"{scenario_prev['predicted_crime_rate']:.2f}",
        f"{scenario_prev['predicted_crime_rate'] - result['predicted_crime_rate']:.2f}",
    )
    s2.metric(
        "If population +10%",
        f"{scenario_pop['predicted_crime_rate']:.2f}",
        f"{scenario_pop['predicted_crime_rate'] - result['predicted_crime_rate']:.2f}",
    )

    st.caption(
        "These what-if results confirm prediction output responds to changes in "
        "previous_year_crime_rate and population."
    )

st.subheader("Model Performance Snapshot")
metrics = metadata.get("metrics", {})
mc1, mc2, mc3 = st.columns(3)
mc1.metric("MAE", f"{float(metrics.get('mae', 0.0)):.4f}")
mc2.metric("R2", f"{float(metrics.get('r2', 0.0)):.4f}")
mc3.metric("Classification Accuracy", f"{float(metrics.get('classification_accuracy', 0.0)):.2%}")

st.warning("Prediction output is advisory and should be interpreted with reporting-bias context.")
