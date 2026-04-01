from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Limitations", page_icon="⚠️", layout="wide")

st.title("Limitations")
st.caption("Known constraints of data, model, and deployment scope")

st.markdown("- Data may contain reporting bias")
st.markdown("- Limited dataset years")
st.markdown("- Model depends on historical data")
st.markdown("- Predictions are macro-level only")
st.markdown("- Not suitable for individual prediction")

st.info(
    "These limitations should be reviewed before using outputs for policy decisions. "
    "Predictions are advisory, not definitive evidence."
)
