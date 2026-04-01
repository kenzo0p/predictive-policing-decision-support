from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

AGGREGATE_DISTRICT_PATTERN = r"TOTAL|RLY|G\.R\.P|CID|STF|BIEO|R\.P\.O"


@dataclass
class DataBundle:
    district_data: pd.DataFrame
    state_year_data: pd.DataFrame


def _normalize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for src, dst in [
        ("STATE/UT", "state"),
        ("DISTRICT", "district"),
        ("YEAR", "year"),
        ("TOTAL IPC CRIMES", "total_crimes"),
        ("POPULATION", "population"),
    ]:
        if src in df.columns:
            rename_map[src] = dst

    normalized = df.rename(columns=rename_map).copy()

    required = {"state", "district", "year", "total_crimes"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    normalized["state"] = normalized["state"].astype(str).str.strip()
    normalized["district"] = normalized["district"].astype(str).str.strip()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce")
    normalized["total_crimes"] = pd.to_numeric(normalized["total_crimes"], errors="coerce")

    if "population" in normalized.columns:
        normalized["population"] = pd.to_numeric(normalized["population"], errors="coerce")
    else:
        normalized["population"] = np.nan

    normalized = normalized.dropna(subset=["state", "district", "year", "total_crimes"])
    normalized["year"] = normalized["year"].astype(int)
    normalized["total_crimes"] = normalized["total_crimes"].astype(float)

    normalized = normalized[
        ~normalized["district"].str.contains(AGGREGATE_DISTRICT_PATTERN, case=False, na=False)
    ]

    return normalized


def load_crime_dataset(path: str) -> DataBundle:
    raw = pd.read_csv(path)
    district_data = _normalize_base_columns(raw)

    state_year = (
        district_data.groupby(["state", "year"], as_index=False)
        .agg(total_crimes=("total_crimes", "sum"), population=("population", "sum"))
        .sort_values(["state", "year"])
    )

    valid_pop = state_year["population"].notna() & (state_year["population"] > 0)
    state_year["crime_rate"] = np.where(
        valid_pop,
        (state_year["total_crimes"] / state_year["population"]) * 100000.0,
        np.nan,
    )

    # Fallback for datasets without population: normalize by state mean to keep downstream pages usable.
    no_rate_rows = state_year["crime_rate"].isna()
    if no_rate_rows.any():
        state_means = state_year.groupby("state")["total_crimes"].transform("mean").replace(0, np.nan)
        state_year.loc[no_rate_rows, "crime_rate"] = (
            state_year.loc[no_rate_rows, "total_crimes"] / state_means[no_rate_rows]
        ) * 100.0

    state_year["prev_year_crime_rate"] = state_year.groupby("state")["crime_rate"].shift(1)
    state_year["prev_year_crime_rate"] = state_year["prev_year_crime_rate"].fillna(state_year["crime_rate"])

    return DataBundle(district_data=district_data, state_year_data=state_year)


def get_overview_metrics(bundle: DataBundle) -> Dict[str, float]:
    state_year = bundle.state_year_data
    return {
        "states": int(state_year["state"].nunique()),
        "years": int(state_year["year"].nunique()),
        "rows": int(len(bundle.district_data)),
        "total_crimes": float(state_year["total_crimes"].sum()),
        "avg_crime_rate": float(state_year["crime_rate"].mean()),
    }


def get_state_trends(bundle: DataBundle, state: Optional[str] = None) -> pd.DataFrame:
    data = bundle.state_year_data
    if state:
        data = data[data["state"].str.lower() == state.lower()]
    return data.sort_values(["state", "year"]).reset_index(drop=True)


def get_state_comparison(bundle: DataBundle, year: Optional[int] = None) -> pd.DataFrame:
    data = bundle.state_year_data
    if year is not None:
        data = data[data["year"] == year]

    comparison = (
        data.groupby("state", as_index=False)
        .agg(
            total_crimes=("total_crimes", "sum"),
            avg_crime_rate=("crime_rate", "mean"),
            avg_population=("population", "mean"),
        )
        .sort_values("avg_crime_rate", ascending=False)
    )
    return comparison.reset_index(drop=True)


def get_heatmap_matrix(bundle: DataBundle) -> pd.DataFrame:
    matrix = bundle.state_year_data.pivot_table(
        index="state",
        columns="year",
        values="crime_rate",
        aggfunc="mean",
    )
    return matrix.sort_index()


def detect_reporting_bias(bundle: DataBundle) -> pd.DataFrame:
    district = bundle.district_data

    state_stats = district.groupby("state")["total_crimes"].agg(["mean", "std", "count"]).reset_index()
    state_stats["cv"] = (state_stats["std"] / state_stats["mean"].replace(0, np.nan)) * 100.0
    std_mean = float(state_stats["mean"].std(ddof=0))
    std_mean = std_mean if std_mean != 0 else np.nan
    state_stats["z_score_mean"] = (state_stats["mean"] - state_stats["mean"].mean()) / std_mean
    state_stats["anomaly_score"] = np.abs(state_stats["z_score_mean"]).fillna(0) + (
        state_stats["cv"].fillna(0) / 100.0
    )

    def assign_risk(row: pd.Series) -> str:
        if row["anomaly_score"] >= 2.0 or row["cv"] >= 100:
            return "HIGH"
        if row["anomaly_score"] >= 1.2 or row["cv"] >= 60:
            return "MEDIUM"
        return "LOW"

    state_stats["bias_risk"] = state_stats.apply(assign_risk, axis=1)
    state_stats = state_stats.sort_values("anomaly_score", ascending=False)
    return state_stats.reset_index(drop=True)


def build_model_table(bundle: DataBundle) -> pd.DataFrame:
    model_df = bundle.state_year_data.copy()
    model_df = model_df.sort_values(["state", "year"])

    # When crime_rate has little/no variance (e.g., missing population fallback),
    # derive risk buckets from total_crimes to avoid collapsing to a single class.
    score = model_df["crime_rate"]
    if score.nunique(dropna=True) < 3:
        score = model_df["total_crimes"]

    if len(model_df) >= 3:
        ranked_score = score.rank(method="first")
        model_df["crime_risk_category"] = pd.qcut(
            ranked_score,
            q=3,
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)
    else:
        model_df["crime_risk_category"] = "LOW"

    model_df["population"] = model_df["population"].fillna(model_df["population"].median())
    return model_df.reset_index(drop=True)


def ethical_points() -> List[str]:
    return [
        "Predictions are macro-level decision support, not individual profiling.",
        "Higher reported crime may reflect better reporting infrastructure.",
        "All model outputs require human review and policy context.",
        "Bias and drift audits should be performed before operational use.",
        "The system prioritizes transparency through explainable aggregate features.",
    ]
