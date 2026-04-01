from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Future Scope", page_icon="🚀", layout="wide")

st.title("Future Scope")
st.caption("Planned enhancements for capability, robustness, and responsible AI")

st.markdown("- Add district-level data")
st.markdown("- Include socioeconomic features")
st.markdown("- Use time-series forecasting models")
st.markdown("- Deploy system to cloud")
st.markdown("- Add fairness and bias audit module")

st.success(
    "These roadmap items will improve prediction quality, interpretability, and operational readiness."
)
