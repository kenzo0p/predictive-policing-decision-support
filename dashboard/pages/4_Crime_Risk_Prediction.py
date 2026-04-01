from pathlib import Path
import sys

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

st.set_page_config(page_title="Crime Risk Prediction", page_icon="🤖", layout="wide")

st.title("Crime Risk Prediction")
st.caption("Random Forest based macro-level risk estimation")
st.info(
    "Risk category is assigned from predicted crime rate thresholds: "
    "LOW (<150), MEDIUM (150-299.99), HIGH (>=300)."
)

bundle = get_bundle()
models = get_models()
metadata = get_model_metadata()
state_df = bundle.state_year_data

states = sorted(state_df["state"].unique().tolist())
state = st.selectbox("State", states)
years = sorted(state_df["year"].unique().tolist())
year = st.selectbox("Prediction year", years, index=len(years) - 1)

state_hist = state_df[state_df["state"] == state].sort_values("year")
default_prev_rate = float(state_hist.iloc[-1]["crime_rate"])
default_population = float(state_hist.iloc[-1]["population"]) if state_hist.iloc[-1]["population"] == state_hist.iloc[-1]["population"] else 0.0

prev_rate = st.number_input("Previous year crime rate", min_value=0.0, value=default_prev_rate)
population = st.number_input("Population", min_value=0.0, value=default_population)

if st.button("Predict risk"):
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
    c3.metric("Model Confidence", f"{result['confidence']:.2%}")

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

    st.subheader("Input Sensitivity Check")
    scenario_prev = predict(
        models=models,
        state=state,
        year=int(year),
        prev_year_crime_rate=float(prev_rate) * 1.10,
        population=float(population),
    )
    scenario_pop = predict(
        models=models,
        state=state,
        year=int(year),
        prev_year_crime_rate=float(prev_rate),
        population=float(population) * 1.10,
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
