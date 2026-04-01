from pathlib import Path
import sys

import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.shared import bias_table

st.set_page_config(page_title="Reporting Bias Analysis", page_icon="⚖️", layout="wide")

st.title("Reporting Bias Analysis")
st.caption("Anomaly and statistical consistency checks at state level")

bias_df = bias_table()

fig = px.bar(
    bias_df.head(20),
    x="state",
    y="anomaly_score",
    color="bias_risk",
    title="Top States by Reporting Bias Anomaly Score",
    color_discrete_map={"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"},
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, width="stretch")

st.subheader("Bias Signals Table")
st.dataframe(
    bias_df[["state", "mean", "std", "count", "cv", "z_score_mean", "anomaly_score", "bias_risk"]],
    width="stretch",
)

st.warning(
    "High anomaly score suggests possible reporting irregularity, but does not prove under-reporting or over-reporting by itself."
)
