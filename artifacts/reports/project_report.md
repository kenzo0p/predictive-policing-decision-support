# Project Report: Predictive Policing Decision Support System

## 1. Objective

Build an end-to-end, state-level crime analytics and prediction system that includes:
- data preprocessing
- visual analytics
- reporting-bias analysis
- machine learning prediction
- API serving
- dashboard interaction
- ethics-aware interpretation

## 2. Dataset

- Source: `data/raw/dstrIPC_2013.csv`
- Unit: district-level records
- Key fields used: state, district, year, total crimes, population (if available)

## 3. Data Preprocessing

Implemented in `src/data_pipeline.py` with:
- column inference from aliases
- optional custom column mapping (`config/column_map.example.json`)
- schema validation for required columns
- optional computation of `total_crimes`
- removal of aggregate district rows (TOTAL, RLY, CID, etc.)
- cleaned output to `data/processed/clean_crime_data.csv`

## 4. Analytics Dashboard

Analytics functionality includes:
- overview KPIs (states, years, total crimes, mean crime rate)
- year/state trend analysis
- state comparison tables
- state-year heatmap matrix

Implemented in:
- backend: `crime_analytics/services/analytics_service.py`
- frontend pages: `dashboard/pages/1_Overview_Dashboard.py`, `dashboard/pages/2_Crime_Trends.py`

## 5. Reporting Bias Analysis

Bias indicators are computed at state level using:
- mean and standard deviation of district crime totals
- coefficient of variation (CV)
- z-score based anomaly signal
- composite anomaly score
- categorical bias risk (LOW, MEDIUM, HIGH)

Implemented in:
- service: `detect_reporting_bias` in `crime_analytics/services/analytics_service.py`
- dashboard page: `dashboard/pages/3_Reporting_Bias_Analysis.py`

## 6. ML Prediction

Two-model approach:
- RandomForestRegressor for predicted crime rate
- RandomForestClassifier for risk category

Features:
- state
- year
- prev_year_crime_rate
- population

Outputs:
- predicted crime rate
- risk category
- class probabilities
- confidence score and uncertainty flag

Implemented in:
- training/inference: `crime_analytics/services/model_service.py`
- training script: `scripts/train_model.py`
- Streamlit page: `dashboard/pages/4_Crime_Risk_Prediction.py`

## 7. FastAPI Backend

Main file: `api/main.py`

Exposed endpoints:
- health and database status
- analytics overview/trends/comparison/heatmap
- reporting bias summary
- prediction endpoint
- ethics principles endpoint
- recent predictions endpoint

Schemas in `api/schemas.py`, optional Mongo persistence in `api/database.py`.

## 8. Streamlit Dashboard

Entry point: `dashboard/app.py`

Pages:
- Overview Dashboard
- Crime Trends
- Reporting Bias Analysis
- Crime Risk Prediction
- Ethical Considerations

Shared cached loaders in `dashboard/shared.py`.

## 9. Ethics and Responsible Use

Ethical guidance is integrated in both API and dashboard.

Core principles:
- no individual profiling
- outputs are advisory, not automated enforcement
- reporting quality influences observed crime levels
- human-in-the-loop governance required

Implemented in:
- `dashboard/pages/5_Ethical_Considerations.py`
- `crime_analytics/services/analytics_service.py` (`ethical_points`)
- `GET /ethics` endpoint

## 10. Screenshots and Artifacts

Generated screenshots are saved to:
- `artifacts/screenshots/`

Model artifacts:
- `models/crime_risk_model.joblib`
- `models/crime_risk_model_metadata.joblib`

Legacy/alternate model artifact:
- `artifacts/models/crime_risk_model.pkl`

## 11. Status Against 10 Required Items

1. Data preprocessing: Completed
2. Analytics dashboard: Completed
3. Reporting bias analysis: Completed
4. ML prediction: Completed
5. FastAPI backend: Completed
6. Streamlit dashboard: Completed
7. Ethics section: Completed
8. README: Completed
9. Screenshots: Completed
10. Report: Completed

## 12. Conclusion

All 10 requested components are now present in the project. The system is deployable for educational and policy-support use at aggregate level, with explicit bias and ethics caveats documented across code, dashboard, and API.
