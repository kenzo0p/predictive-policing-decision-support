from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from crime_analytics.services.analytics_service import DataBundle, build_model_table


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    model_path: str
    metadata_path: str


def _build_features_targets(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = model_df[["state", "year", "prev_year_crime_rate", "population"]].copy()
    y_reg = model_df["crime_rate"].copy()
    y_cls = model_df["crime_risk_category"].copy()
    return X, y_reg, y_cls


def train_and_save(bundle: DataBundle, model_dir: str = "models") -> TrainResult:
    model_df = build_model_table(bundle)
    X, y_reg, y_cls = _build_features_targets(model_df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["state"]),
            ("num", "passthrough", ["year", "prev_year_crime_rate", "population"]),
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
    r2 = float(r2_score(y_test_reg, reg_pred))
    cls_accuracy = float(np.mean(cls_pred == y_test_cls))

    model_output = {
        "regressor": reg_model,
        "classifier": cls_model,
    }

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / "crime_risk_model.joblib")
    metadata_path = str(output_dir / "crime_risk_model_metadata.joblib")

    metadata = {
        "features": ["state", "year", "prev_year_crime_rate", "population"],
        "target_regression": "crime_rate",
        "target_classification": "crime_risk_category",
        "classification_report": classification_report(
            y_test_cls,
            cls_pred,
            output_dict=True,
            zero_division=0,
        ),
        "metrics": {
            "mae": mae,
            "r2": r2,
            "classification_accuracy": cls_accuracy,
        },
    }

    joblib.dump(model_output, model_path)
    joblib.dump(metadata, metadata_path)

    return TrainResult(metrics=metadata["metrics"], model_path=model_path, metadata_path=metadata_path)


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
    prev_year_crime_rate: float,
    population: float,
    uncertainty_threshold: float = 0.6,
) -> Dict[str, float | str]:
    row = pd.DataFrame(
        [
            {
                "state": state,
                "year": year,
                "prev_year_crime_rate": prev_year_crime_rate,
                "population": population,
            }
        ]
    )

    crime_rate = float(models["regressor"].predict(row)[0])
    risk_category = str(models["classifier"].predict(row)[0])
    probabilities = models["classifier"].predict_proba(row)[0]
    classes = list(models["classifier"].classes_)
    confidence = float(np.max(probabilities))

    return {
        "predicted_crime_rate": crime_rate,
        "predicted_risk_category": risk_category,
        "confidence": confidence,
        "class_probabilities": {classes[i]: float(probabilities[i]) for i in range(len(classes))},
        "is_uncertain": confidence < uncertainty_threshold,
        "uncertainty_reason": (
            f"Top class confidence {confidence:.2%} is below threshold {uncertainty_threshold:.0%}."
            if confidence < uncertainty_threshold
            else None
        ),
    }
