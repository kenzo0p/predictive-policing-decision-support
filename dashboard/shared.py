from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure absolute package imports work when Streamlit runs modules directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crime_analytics.services.analytics_service import (
    detect_reporting_bias,
    get_heatmap_matrix,
    get_overview_metrics,
    get_state_comparison,
    get_state_trends,
    load_crime_dataset,
)
from crime_analytics.services.model_service import load_model_metadata, load_models

DATA_PATH = os.getenv("CRIME_DATA_PATH", "data/raw/dstrIPC_2013.csv")
MODEL_PATH = os.getenv("CRIME_MODEL_PATH", "models/crime_risk_model.joblib")
MODEL_METADATA_PATH = os.getenv("CRIME_MODEL_METADATA_PATH", "models/crime_risk_model_metadata.joblib")


@st.cache_resource
def get_bundle():
    return load_crime_dataset(DATA_PATH)


@st.cache_resource
def get_models():
    return load_models(MODEL_PATH)


@st.cache_resource
def get_model_metadata():
    return load_model_metadata(MODEL_PATH, MODEL_METADATA_PATH)


@st.cache_data
def overview_metrics() -> dict:
    return get_overview_metrics(get_bundle())


@st.cache_data
def trends(state: str | None = None):
    return get_state_trends(get_bundle(), state=state)


@st.cache_data
def comparison(year: int | None = None):
    return get_state_comparison(get_bundle(), year=year)


@st.cache_data
def heatmap_matrix():
    return get_heatmap_matrix(get_bundle())


@st.cache_data
def bias_table():
    return detect_reporting_bias(get_bundle())


@st.cache_data
def feature_importance_table() -> pd.DataFrame:
    metadata = get_model_metadata()
    rows = metadata.get("feature_importance", [])
    if not rows:
        return pd.DataFrame(columns=["feature", "regressor_importance", "classifier_importance", "avg_importance"])

    importance_df = pd.DataFrame(rows)
    for col in ["regressor_importance", "classifier_importance", "avg_importance"]:
        if col in importance_df.columns:
            importance_df[col] = pd.to_numeric(importance_df[col], errors="coerce").fillna(0.0)
    return importance_df
