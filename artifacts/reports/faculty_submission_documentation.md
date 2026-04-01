# Faculty Submission Document
## Predictive Policing Decision Support System

Prepared for: Project Evaluation / Faculty Review  
Project Type: End-to-End Data + ML + API + Dashboard System  
Date: 1 April 2026

---

## 1. Executive Summary

This project is an end-to-end predictive policing decision-support application at macro (state) level.  
It includes:
- data preprocessing and schema standardization
- analytics and visualization dashboard
- reporting-bias detection
- machine learning based risk prediction
- FastAPI backend services
- Streamlit interactive frontend
- ethics and responsible-use safeguards

The system is designed for policy support and resource planning, not for individual profiling or automated punitive action.

---

## 2. Problem Statement

Crime data often suffers from reporting variation across regions, making direct interpretation risky. The goal is to provide a transparent, ethical, and technically robust decision-support workflow that:
- normalizes and validates data
- presents interpretable analytics
- flags potential reporting-bias signals
- produces ML-based risk estimates with confidence and uncertainty
- communicates ethical constraints clearly

---

## 3. Architecture Overview

### 3.1 Pipeline Stages
1. Raw data ingestion from CSV
2. Data cleaning and schema validation
3. State-year aggregation and feature engineering
4. Model training and artifact storage
5. Backend API serving analytics/prediction/ethics
6. Frontend dashboard rendering and interaction
7. Documentation and screenshot artifacts generation

### 3.2 Core Layers
- Data layer: preprocessing and aggregation
- Service layer: analytics and ML logic
- API layer: FastAPI endpoints with schemas
- UI layer: Streamlit multipage dashboard
- Governance layer: ethics and bias interpretation constraints

---

## 4. Component-Wise Documentation (Each Built Module)

## 4.1 Data Preprocessing

Purpose:
- Convert raw IPC dataset into a clean, consistent schema usable by analytics and ML.

Implementation:
- Main pipeline: `src/data_pipeline.py`
- Wrapper runner: `src/preprocessing.py`
- Column map template: `config/column_map.example.json`

What it does:
- Infers aliases for required fields (`state`, `district`, `year`, `total_crimes`)
- Supports explicit user mapping via JSON
- Validates required schema
- Optionally computes `total_crimes` if missing
- Removes aggregate pseudo-district rows (TOTAL, RLY, CID, etc.)
- Writes cleaned output to `data/processed/clean_crime_data.csv`

Output:
- Cleaned tabular dataset suitable for downstream analytics and model training.

---

## 4.2 Analytics Dashboard Logic

Purpose:
- Provide high-level state-wise crime insights through metrics and trends.

Implementation:
- Service functions: `crime_analytics/services/analytics_service.py`
- Shared dashboard accessors: `dashboard/shared.py`
- UI pages: `dashboard/pages/1_Overview_Dashboard.py`, `dashboard/pages/2_Crime_Trends.py`

What it does:
- Computes summary KPIs (states, years, total crimes, average crime rate)
- Produces state trend series over time
- Produces state comparison views
- Builds state-year heatmap matrix

Output:
- Interactive visual analytics in Streamlit.

---

## 4.3 Reporting Bias Analysis

Purpose:
- Detect data irregularities that may indicate reporting inconsistency.

Implementation:
- Bias computation: `detect_reporting_bias` in `crime_analytics/services/analytics_service.py`
- Dashboard page: `dashboard/pages/3_Reporting_Bias_Analysis.py`

Methods used:
- state-level district mean/std/count
- coefficient of variation (CV)
- z-score based anomaly signal
- composite anomaly score
- risk labels: LOW / MEDIUM / HIGH

Important interpretation:
- Bias score indicates anomaly risk, not legal proof of under-reporting/over-reporting.

---

## 4.4 ML Prediction Module

Purpose:
- Estimate crime risk using interpretable aggregate features.

Implementation:
- Model service: `crime_analytics/services/model_service.py`
- Training script: `scripts/train_model.py`
- Prediction UI page: `dashboard/pages/4_Crime_Risk_Prediction.py`

Model design:
- RandomForestRegressor: predicts numeric crime rate
- RandomForestClassifier: predicts risk class (LOW/MEDIUM/HIGH)

Features:
- state
- year
- prev_year_crime_rate
- population

Outputs returned:
- predicted crime rate
- predicted risk category
- class probabilities
- confidence score
- uncertainty flag/reason

Artifacts:
- `models/crime_risk_model.joblib`
- `models/crime_risk_model_metadata.joblib`

UI improvement completed:
- Class probabilities now shown with ordered progress bars and percentages (instead of raw JSON).

---

## 4.5 FastAPI Backend

Purpose:
- Expose analytics, prediction, and ethics functions as HTTP APIs.

Implementation:
- API app: `api/main.py`
- Schemas: `api/schemas.py`
- Optional Mongo helper: `api/database.py`

Key endpoints:
- `GET /health`
- `GET /database/status`
- `GET /analytics/overview`
- `GET /analytics/trends`
- `GET /analytics/comparison`
- `GET /analytics/heatmap`
- `GET /analytics/bias`
- `POST /predict`
- `GET /ethics`
- `GET /predictions/recent`

Output:
- Structured JSON responses consumed by UI or external clients.

---

## 4.6 Streamlit Dashboard

Purpose:
- Provide non-technical users an interactive view of analytics and predictions.

Implementation:
- Entry point: `dashboard/app.py`
- Pages:
  - `dashboard/pages/1_Overview_Dashboard.py`
  - `dashboard/pages/2_Crime_Trends.py`
  - `dashboard/pages/3_Reporting_Bias_Analysis.py`
  - `dashboard/pages/4_Crime_Risk_Prediction.py`
  - `dashboard/pages/5_Ethical_Considerations.py`

Capabilities:
- multi-page navigation
- KPI + chart visualizations
- prediction input controls
- confidence and uncertainty communication
- ethics warnings and governance recommendations

---

## 4.7 Ethics Section and Responsible AI Constraints

Purpose:
- Ensure model output is interpreted responsibly and within policy boundaries.

Implementation:
- Ethics points source: `ethical_points` in `crime_analytics/services/analytics_service.py`
- Dashboard ethics page: `dashboard/pages/5_Ethical_Considerations.py`
- API endpoint: `GET /ethics`

Principles:
- no individual profiling
- advisory outputs only
- reporting-bias context required
- human-in-the-loop decision-making

---

## 4.8 Documentation Deliverables

Already available:
- Main project readme: `README.md`
- Technical report: `artifacts/reports/project_report.md`
- Demo/viva speaking script: `artifacts/reports/demo_script.md`
- This faculty submission document: `artifacts/reports/faculty_submission_documentation.md`

---

## 4.9 Screenshots and Artifacts

Screenshot generation script:
- `scripts/generate_screenshots.py`

Generated screenshots:
- `artifacts/screenshots/overview_heatmap.png`
- `artifacts/screenshots/crime_trends.png`
- `artifacts/screenshots/reporting_bias.png`
- `artifacts/screenshots/ml_prediction_snapshot.png`
- `artifacts/screenshots/ethics_section.png`

---

## 4.10 End-to-End Run Automation

Single command script:
- `scripts/run_all.sh`

What it runs:
1. data preprocessing
2. model training
3. screenshot generation

Dependency installation:
- `requirements.txt` (pinned package versions)

---

## 5. Exact Run Instructions (Reproducible)

1. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run full pipeline
```bash
bash scripts/run_all.sh
```

4. Run API
```bash
venv/bin/uvicorn api.main:app --reload
```

5. Run dashboard
```bash
venv/bin/streamlit run dashboard/app.py
```

---

## 6. Verification Checklist

1. Preprocessing output exists: `data/processed/clean_crime_data.csv`
2. Model files exist: `models/crime_risk_model.joblib`, `models/crime_risk_model_metadata.joblib`
3. Screenshot files exist in `artifacts/screenshots/`
4. API health works: `GET /health` returns `{"status":"ok"}`
5. Streamlit UI loads on local port
6. Prediction page shows probability bars and confidence/uncertainty

---

## 7. Limitations and Future Improvements

Current limitations:
- Data scope currently based on available dataset slice
- Prediction quality depends on data quality and coverage
- Reporting bias cannot be conclusively inferred from statistics alone

Recommended future work:
- incorporate multi-year richer features (socioeconomic factors where permissible)
- stronger model evaluation protocol and calibration
- formal bias/fairness dashboards with periodic audit logs
- CI pipeline for tests and model-quality checks

---

## 8. Final Statement

All requested project components are implemented and operational:
1. data preprocessing
2. analytics dashboard
3. reporting-bias analysis
4. ML prediction
5. FastAPI backend
6. Streamlit dashboard
7. ethics section
8. README
9. screenshots
10. report

The application is functioning end-to-end and documented for faculty submission.
