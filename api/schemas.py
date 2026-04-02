from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    state: str = Field(..., description="State name")
    year: int = Field(..., ge=1900)
    prev_year_crime_rate: float = Field(..., ge=0)
    population: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_crime_rate: float
    predicted_risk_category: str
    confidence: float
    prediction_type: str
    is_forecast: bool
    dataset_max_year: Optional[int] = None
    class_probabilities: Dict[str, float]
    risk_thresholds: Dict[str, str]
    forecast_trend: List[Dict[str, float | int | str]] = Field(default_factory=list)
    is_uncertain: bool
    uncertainty_reason: Optional[str] = None


class OverviewResponse(BaseModel):
    states: int
    years: int
    rows: int
    total_crimes: float
    avg_crime_rate: float


class EthicalResponse(BaseModel):
    principles: List[str]
