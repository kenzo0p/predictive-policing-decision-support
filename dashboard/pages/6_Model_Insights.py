from pathlib import Path
import sys

import plotly.express as px
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.shared import feature_importance_table, get_model_metadata


st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")

st.title("Model Insights")
st.caption("Feature importance and model performance diagnostics")

metadata = get_model_metadata()
metrics = metadata.get("metrics", {})
importance_df = feature_importance_table()

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"{float(metrics.get('mae', 0.0)):.4f}")
m2.metric("RMSE", f"{float(metrics.get('rmse', 0.0)):.4f}")
m3.metric("R2", f"{float(metrics.get('r2', 0.0)):.4f}")
m4.metric("Accuracy", f"{float(metrics.get('classification_accuracy', 0.0)):.2%}")

if importance_df.empty:
    st.warning("Feature importance data not found. Retrain the model to populate insights.")
    st.stop()

plot_df = importance_df.copy()
plot_df["pretty_feature"] = plot_df["feature"].astype(str)
plot_df = plot_df.sort_values("avg_importance", ascending=False)

st.subheader("Feature Importance Chart")
fig = px.bar(
    plot_df.head(20),
    x="avg_importance",
    y="pretty_feature",
    orientation="h",
    color="avg_importance",
    color_continuous_scale="Blues",
    labels={"avg_importance": "Average Importance", "pretty_feature": "Feature"},
)
fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, width="stretch")

state_mask = plot_df["feature"].astype(str).str.startswith("cat__state_")
state_total = float(plot_df.loc[state_mask, "avg_importance"].sum())

non_state = plot_df.loc[~state_mask].copy()
non_state["base_feature"] = (
    non_state["feature"].astype(str)
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

base_rows = [{"base_feature": "state (encoded)", "avg_importance": state_total}]
if not non_state.empty:
    grouped_non_state = (
        non_state.groupby("base_feature", as_index=False)["avg_importance"]
        .sum()
        .sort_values("avg_importance", ascending=False)
    )
    base_rows.extend(grouped_non_state.to_dict(orient="records"))

base_df = pd.DataFrame(base_rows).sort_values("avg_importance", ascending=False)

st.subheader("Most Influential Feature Groups")
fig_grouped = px.bar(
    base_df,
    x="base_feature",
    y="avg_importance",
    color="avg_importance",
    color_continuous_scale="Tealgrn",
    labels={"base_feature": "Feature Group", "avg_importance": "Importance"},
)
fig_grouped.update_layout(xaxis_tickangle=-20, height=450)
st.plotly_chart(fig_grouped, width="stretch")

st.subheader("Interpretation")
top3 = base_df.head(3)
lines = [
    f"- {row.base_feature} contributes {row.avg_importance:.3f} importance"
    for row in top3.itertuples()
]
st.markdown(
    "The model is influenced most by these feature groups:\n" + "\n".join(lines)
)

st.info(
    "State (encoded) often dominates because each state has its own one-hot feature. "
    "Use this together with population and temporal features for policy interpretation, "
    "not as a standalone causality claim."
)

st.subheader("Detailed Feature Importance Table")
st.dataframe(plot_df[["feature", "regressor_importance", "classifier_importance", "avg_importance"]], width="stretch")
