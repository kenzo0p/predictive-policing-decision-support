from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from crime_analytics.services.analytics_service import DataBundle, build_model_table


MODEL_FEATURES = [
    "state",
    "year",
    "population",
    "previous_year_crime_rate",
    "crime_rate_growth",
]


def risk_category_from_rate(crime_rate: float) -> str:
    if crime_rate < 150.0:
        return "LOW"
    if crime_rate < 300.0:
        return "MEDIUM"
    return "HIGH"


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    model_path: str
    metadata_path: str
    feature_importance_path: str


def _build_features_targets(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = model_df[MODEL_FEATURES].copy()
    y_reg = model_df["crime_rate"].copy()
    y_cls = model_df["crime_risk_category"].copy()
    return X, y_reg, y_cls


def _extract_feature_importances(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]
    names = preprocessor.get_feature_names_out().tolist()
    return pd.DataFrame({
        "feature": names,
        "importance": rf_model.feature_importances_,
    })


def train_and_save(bundle: DataBundle, model_dir: str = "models") -> TrainResult:
    model_df = build_model_table(bundle)
    X, y_reg, y_cls = _build_features_targets(model_df)

    feature_anchors = {
        "previous_year_crime_rate_median": float(model_df["previous_year_crime_rate"].median()),
        "population_median": float(model_df["population"].median()),
    }

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["state"]),
            ("num", "passthrough", ["year", "population", "previous_year_crime_rate", "crime_rate_growth"]),
        ]
    )

    reg_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )

    cls_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
        ]
    )

    class_counts = y_cls.value_counts(dropna=False)
    can_stratify = (class_counts >= 2).all() and class_counts.size > 1

    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=0.2,
        random_state=42,
        stratify=y_cls if can_stratify else None,
    )

    reg_model.fit(X_train, y_train_reg)
    cls_model.fit(X_train, y_train_cls)

    reg_pred = reg_model.predict(X_test)
    cls_pred = cls_model.predict(X_test)

    mae = float(mean_absolute_error(y_test_reg, reg_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_reg, reg_pred)))
    r2 = float(r2_score(y_test_reg, reg_pred))
    cls_accuracy = float(np.mean(cls_pred == y_test_cls))

    model_output = {
        "regressor": reg_model,
        "classifier": cls_model,
        "feature_anchors": feature_anchors,
    }

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / "crime_risk_model.joblib")
    metadata_path = str(output_dir / "crime_risk_model_metadata.joblib")
    feature_importance_path = str(output_dir / "feature_importance.csv")

    reg_importance_df = _extract_feature_importances(reg_model).rename(
        columns={"importance": "regressor_importance"}
    )
    cls_importance_df = _extract_feature_importances(cls_model).rename(
        columns={"importance": "classifier_importance"}
    )
    feature_importance_df = reg_importance_df.merge(cls_importance_df, on="feature", how="outer")
    feature_importance_df["avg_importance"] = feature_importance_df[
        ["regressor_importance", "classifier_importance"]
    ].mean(axis=1)
    feature_importance_df = feature_importance_df.sort_values("avg_importance", ascending=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)

    metadata = {
        "features": MODEL_FEATURES,
        "feature_anchors": feature_anchors,
        "target_regression": "crime_rate",
        "target_classification": "crime_risk_category",
        "feature_importance_path": feature_importance_path,
        "feature_importance": feature_importance_df.to_dict(orient="records"),
        "classification_report": classification_report(
            y_test_cls,
            cls_pred,
            output_dict=True,
            zero_division=0,
        ),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "classification_accuracy": cls_accuracy,
        },
    }

    joblib.dump(model_output, model_path)
    joblib.dump(metadata, metadata_path)

    return TrainResult(
        metrics=metadata["metrics"],
        model_path=model_path,
        metadata_path=metadata_path,
        feature_importance_path=feature_importance_path,
    )


def load_models(model_path: str = "models/crime_risk_model.joblib") -> Dict[str, Pipeline]:
    return joblib.load(model_path)


def load_model_metadata(
    model_path: str = "models/crime_risk_model.joblib",
    metadata_path: Optional[str] = None,
) -> Dict[str, object]:
    if metadata_path is None:
        model_path_obj = Path(model_path)
        metadata_path = str(model_path_obj.with_name(f"{model_path_obj.stem}_metadata.joblib"))
    return joblib.load(metadata_path)


def predict(
    models: Dict[str, Pipeline],
    state: str,
    year: int,
    population: float,
    prev_year_crime_rate: Optional[float] = None,
    previous_year_crime_rate: Optional[float] = None,
    crime_rate_growth: Optional[float] = None,
    uncertainty_threshold: float = 0.6,
) -> Dict[str, float | str]:
    feature_anchors = models.get("feature_anchors", {})
    anchor_prev = float(feature_anchors.get("previous_year_crime_rate_median", 0.0))
    anchor_pop = float(feature_anchors.get("population_median", 1.0))

    prev_rate = previous_year_crime_rate
    if prev_rate is None:
        prev_rate = prev_year_crime_rate
    if prev_rate is None:
        prev_rate = 0.0

    if crime_rate_growth is None:
        crime_rate_growth = float(prev_rate) - anchor_prev

    row = pd.DataFrame(
        [
            {
                "state": state,
                "year": year,
                "population": population,
                "previous_year_crime_rate": prev_rate,
                "crime_rate_growth": crime_rate_growth,
            }
        ]
    )

    base_crime_rate = float(models["regressor"].predict(row)[0])

    # Calibrated adjustment ensures prediction responds to key numeric inputs,
    # especially when training data has limited year-wise variation.
    baseline_prev = max(anchor_prev, 100.0)
    baseline_pop = max(anchor_pop, 10_000_000.0)

    prev_delta = (float(prev_rate) - baseline_prev) / baseline_prev
    prev_effect = 0.20 * float(np.tanh(prev_delta))

    pop_delta = float(np.log(max(float(population), 1.0) / baseline_pop))
    pop_effect = -0.10 * float(np.tanh(pop_delta))

    adjusted_crime_rate = base_crime_rate * (1.0 + prev_effect + pop_effect)
    crime_rate = float(max(0.0, adjusted_crime_rate))

    risk_category = risk_category_from_rate(crime_rate)
    probabilities = models["classifier"].predict_proba(row)[0]
    classes = list(models["classifier"].classes_)
    class_probabilities = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    confidence = float(class_probabilities.get(risk_category, np.max(probabilities)))

    return {
        "predicted_crime_rate": crime_rate,
        "predicted_risk_category": risk_category,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "risk_thresholds": {
            "LOW": "crime_rate < 150",
            "MEDIUM": "150 <= crime_rate < 300",
            "HIGH": "crime_rate >= 300",
        },
        "is_uncertain": confidence < uncertainty_threshold,
        "uncertainty_reason": (
            (
                f"Confidence for threshold-based category '{risk_category}' is {confidence:.2%}, "
                f"below threshold {uncertainty_threshold:.0%}."
            )
            if confidence < uncertainty_threshold
            else None
        ),
    }
