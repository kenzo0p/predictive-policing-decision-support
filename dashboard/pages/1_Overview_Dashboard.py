from pathlib import Path
import sys

import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.shared import heatmap_matrix, overview_metrics

st.set_page_config(page_title="Overview Dashboard", page_icon="📈", layout="wide")

st.title("Overview Dashboard")
st.caption("State-wise crime analytics and crime-rate heatmap")

metrics = overview_metrics()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("States", metrics["states"])
col2.metric("Years", metrics["years"])
col3.metric("Rows", metrics["rows"])
col4.metric("Total Crimes", f"{metrics['total_crimes']:.0f}")
col5.metric("Avg Crime Rate", f"{metrics['avg_crime_rate']:.2f}")

st.subheader("Crime Rate Heatmap")
matrix = heatmap_matrix()
fig = px.imshow(
    matrix,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    labels={"x": "Year", "y": "State", "color": "Crime Rate"},
)
fig.update_layout(height=700)
st.plotly_chart(fig, width="stretch")

st.info(
    "Interpretation note: higher reported crime rates can reflect improved reporting systems, "
    "not necessarily higher true crime prevalence."
)
