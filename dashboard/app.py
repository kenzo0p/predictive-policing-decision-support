from pathlib import Path
import sys

import streamlit as st

# Ensure project-root imports work when running `streamlit run dashboard/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Predictive Policing Decision Support",
    page_icon="📊",
    layout="wide",
)

st.title("Predictive Policing Decision Support System")
st.caption("State-level analytics under reporting-bias and ethical constraints")

st.info(
    "Use the pages in the left sidebar: Overview Dashboard, Crime Trends, "
    "Reporting Bias Analysis, Crime Risk Prediction, and Ethical Considerations."
)

st.markdown(
    """
    ### Scope and Safeguards
    - Macro-level state analysis only
    - Not for individual profiling
    - Results are decision support, not automated enforcement
    - Higher reported crime can indicate better reporting coverage
    """
)
