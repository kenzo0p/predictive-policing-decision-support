#!/usr/bin/env python3
"""Generate static screenshot assets for project documentation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from crime_analytics.services.analytics_service import (
    detect_reporting_bias,
    get_heatmap_matrix,
    get_state_trends,
    load_crime_dataset,
)
from crime_analytics.services.model_service import load_models, predict


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_overview_heatmap(bundle, out_dir: Path) -> None:
    matrix = get_heatmap_matrix(bundle)
    plt.figure(figsize=(12, 7))
    plt.imshow(matrix.fillna(0.0), aspect="auto")
    plt.colorbar(label="Crime Rate")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45)
    plt.title("Overview Heatmap: State vs Year Crime Rate")
    plt.tight_layout()
    plt.savefig(out_dir / "overview_heatmap.png", dpi=160)
    plt.close()


def _save_trends(bundle, out_dir: Path) -> None:
    trends = get_state_trends(bundle)
    top_states = (
        trends.groupby("state", as_index=False)["crime_rate"].mean()
        .sort_values("crime_rate", ascending=False)
        .head(5)["state"]
        .tolist()
    )

    plt.figure(figsize=(12, 7))
    for state in top_states:
        state_df = trends[trends["state"] == state].sort_values("year")
        plt.plot(state_df["year"], state_df["crime_rate"], marker="o", label=state)

    plt.title("Crime Trends: Top 5 States by Average Crime Rate")
    plt.xlabel("Year")
    plt.ylabel("Crime Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "crime_trends.png", dpi=160)
    plt.close()


def _save_bias_plot(bundle, out_dir: Path) -> None:
    bias_df = detect_reporting_bias(bundle).head(15)

    colors = bias_df["bias_risk"].map({"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"})
    plt.figure(figsize=(12, 7))
    plt.barh(bias_df["state"], bias_df["anomaly_score"], color=colors)
    plt.gca().invert_yaxis()
    plt.title("Reporting Bias: Top States by Anomaly Score")
    plt.xlabel("Anomaly Score")
    plt.tight_layout()
    plt.savefig(out_dir / "reporting_bias.png", dpi=160)
    plt.close()


def _save_prediction_card(bundle, out_dir: Path, model_path: Path) -> None:
    models = load_models(str(model_path))
    state_df = bundle.state_year_data.sort_values(["state", "year"])

    sample = state_df.iloc[-1]
    result = predict(
        models=models,
        state=str(sample["state"]),
        year=int(sample["year"]),
        prev_year_crime_rate=float(sample["prev_year_crime_rate"]),
        population=float(sample["population"]) if pd.notna(sample["population"]) else 0.0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    lines = [
        "Crime Risk Prediction Snapshot",
        f"State: {sample['state']}",
        f"Year: {int(sample['year'])}",
        f"Predicted Crime Rate: {result['predicted_crime_rate']:.2f}",
        f"Predicted Risk Category: {result['predicted_risk_category']}",
        f"Confidence: {result['confidence']:.2%}",
        f"Uncertain: {result['is_uncertain']}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "ml_prediction_snapshot.png", dpi=160)
    plt.close()


def _save_ethics_slide(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    points = [
        "Ethical Considerations",
        "- Macro-level analysis only",
        "- Not for individual profiling",
        "- Use outputs as advisory signals",
        "- Interpret with reporting-bias context",
        "- Human-in-the-loop decisions are mandatory",
    ]
    ax.text(0.02, 0.95, "\n".join(points), va="top", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / "ethics_section.png", dpi=160)
    plt.close()


def main() -> int:
    data_path = PROJECT_ROOT / "data/raw/dstrIPC_2013.csv"
    model_path = PROJECT_ROOT / "models/crime_risk_model.joblib"
    out_dir = PROJECT_ROOT / "artifacts/screenshots"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    _ensure_dir(out_dir)
    bundle = load_crime_dataset(str(data_path))

    _save_overview_heatmap(bundle, out_dir)
    _save_trends(bundle, out_dir)
    _save_bias_plot(bundle, out_dir)
    _save_prediction_card(bundle, out_dir, model_path)
    _save_ethics_slide(out_dir)

    print(f"Saved screenshots in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
