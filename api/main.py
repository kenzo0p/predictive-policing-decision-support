from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from api.database import mongo_available, mongo_configured, recent_predictions, save_prediction
from api.schemas import EthicalResponse, OverviewResponse, PredictionRequest, PredictionResponse
from crime_analytics.services.analytics_service import (
    detect_reporting_bias,
    ethical_points,
    get_heatmap_matrix,
    get_overview_metrics,
    get_state_comparison,
    get_state_trends,
    load_crime_dataset,
)
from crime_analytics.services.model_service import load_models, predict

DATA_PATH = os.getenv("CRIME_DATA_PATH", "data/raw/dstrIPC_2013.csv")
MODEL_PATH = os.getenv("CRIME_MODEL_PATH", "models/crime_risk_model.joblib")

app = FastAPI(
    title="Predictive Policing Decision Support API",
    description=(
        "State-level crime analytics and ethical decision support. "
        "Not intended for individual profiling or automated policing actions."
    ),
    version="1.0.0",
)


def _get_bundle():
    try:
        return load_crime_dataset(DATA_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {exc}")


def _get_models():
    try:
        return load_models(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model artifacts: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/database/status")
def database_status() -> dict:
    return {
        "mongo_driver_available": mongo_available(),
        "mongo_configured": mongo_configured(),
    }


@app.get("/analytics/overview", response_model=OverviewResponse)
def analytics_overview() -> OverviewResponse:
    bundle = _get_bundle()
    return OverviewResponse(**get_overview_metrics(bundle))


@app.get("/analytics/trends")
def analytics_trends(state: Optional[str] = Query(default=None)) -> list[dict]:
    bundle = _get_bundle()
    trends = get_state_trends(bundle, state=state)
    return trends.to_dict(orient="records")


@app.get("/analytics/comparison")
def analytics_comparison(year: Optional[int] = Query(default=None)) -> list[dict]:
    bundle = _get_bundle()
    comparison = get_state_comparison(bundle, year=year)
    return comparison.to_dict(orient="records")


@app.get("/analytics/heatmap")
def analytics_heatmap() -> dict:
    bundle = _get_bundle()
    matrix = get_heatmap_matrix(bundle)
    return {
        "states": matrix.index.tolist(),
        "years": [int(col) for col in matrix.columns.tolist()],
        "values": matrix.fillna(0.0).values.tolist(),
    }


@app.get("/analytics/bias")
def analytics_bias() -> list[dict]:
    bundle = _get_bundle()
    bias_df = detect_reporting_bias(bundle)
    return bias_df.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse)
def predict_crime_risk(request: PredictionRequest) -> PredictionResponse:
    models = _get_models()
    output = predict(
        models=models,
        state=request.state,
        year=request.year,
        prev_year_crime_rate=request.prev_year_crime_rate,
        population=request.population,
    )

    save_prediction(request.model_dump(), output)
    return PredictionResponse(**output)


@app.get("/ethics", response_model=EthicalResponse)
def ethics() -> EthicalResponse:
    return EthicalResponse(principles=ethical_points())


@app.get("/predictions/recent")
def predictions_recent(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return recent_predictions(limit=limit)
