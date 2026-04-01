from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crime_analytics.services.analytics_service import ethical_points

st.set_page_config(page_title="Ethical Considerations", page_icon="🧭", layout="wide")

st.title("Ethical Considerations")
st.caption("Fairness, transparency, accountability, and responsible use")

for point in ethical_points():
    st.markdown(f"- {point}")

st.subheader("Usage Constraints")
st.error("Do not use this system for individual profiling or automated punitive action.")
st.info("Treat predictions as advisory signals for policy planning and social interventions.")

st.subheader("Recommended Governance")
st.markdown("- Human-in-the-loop review for all high-impact decisions")
st.markdown("- Periodic bias audit and documentation")
st.markdown("- Public transparency around data limitations and uncertainty")
