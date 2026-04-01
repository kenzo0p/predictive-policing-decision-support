from pathlib import Path
import sys

import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.shared import comparison, trends

st.set_page_config(page_title="Crime Trends", page_icon="📉", layout="wide")

st.title("Crime Trends")

all_trends = trends()
states = sorted(all_trends["state"].unique().tolist())

selected_state = st.selectbox("Select state", options=["All"] + states)
state_filter = None if selected_state == "All" else selected_state
trend_df = trends(state_filter)

fig = px.line(
    trend_df,
    x="year",
    y="crime_rate",
    color="state",
    markers=True,
    title="Crime Rate Trend by State",
)
st.plotly_chart(fig, width="stretch")

st.subheader("State-wise Comparison")
years = sorted(all_trends["year"].unique().tolist())
selected_year = st.selectbox("Filter comparison by year", options=[None] + years, format_func=lambda x: "All years" if x is None else str(x))
comp_df = comparison(selected_year)

fig_comp = px.bar(
    comp_df.head(20),
    x="state",
    y="avg_crime_rate",
    color="avg_crime_rate",
    color_continuous_scale="Viridis",
    title="Top 20 States by Average Crime Rate",
)
fig_comp.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_comp, width="stretch")

st.dataframe(comp_df, width="stretch")
