# Predictive Policing Decision Support System

This project provides a state-level crime analytics and decision-support workflow with explicit reporting-bias and ethical safeguards.

## Requirement Coverage (10/10)

1. Data preprocessing: Implemented in `src/data_pipeline.py` and `src/preprocessing.py`
2. Analytics dashboard: Implemented through analytics services and overview/trends pages
3. Reporting bias analysis: Implemented in `crime_analytics/services/analytics_service.py` and `dashboard/pages/3_Reporting_Bias_Analysis.py`
4. ML prediction: Implemented in `crime_analytics/services/model_service.py` and `dashboard/pages/4_Crime_Risk_Prediction.py`
5. FastAPI backend: Implemented in `api/main.py`
6. Streamlit dashboard: Implemented in `dashboard/app.py` + `dashboard/pages/`
7. Ethics section: Implemented in `dashboard/pages/5_Ethical_Considerations.py` and `/ethics` API endpoint
8. README: This file
9. Screenshots: Generated in `artifacts/screenshots/`
10. Report: Added in `artifacts/reports/project_report.md`

## Project Structure

- `api/`: FastAPI backend (analytics, prediction, ethics, optional Mongo persistence)
- `crime_analytics/services/`: Core analytics and ML service logic
- `dashboard/`: Streamlit multi-page app
- `src/`: Data preparation and standalone analysis scripts
- `scripts/`: Utility scripts to run dashboard, train model, and generate screenshots
- `artifacts/`: Generated assets (models, screenshots, reports)

## Setup

1. Create and activate virtual environment:
   - macOS/Linux: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`

## Run All Required Steps (Single Command)

```bash
bash scripts/run_all.sh
```

This command runs preprocessing, model training, and screenshot generation.

## Data Preprocessing

Run schema-aware pipeline:

```bash
python src/data_pipeline.py --input data/raw/dstrIPC_2013.csv --output data/processed/clean_crime_data.csv
```

Quick wrapper:

```bash
python src/preprocessing.py
```

## Model Training

```bash
python scripts/train_model.py --data data/raw/dstrIPC_2013.csv --model-dir models
```

## Run FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Key endpoints:
- `GET /health`
- `GET /analytics/overview`
- `GET /analytics/trends`
- `GET /analytics/comparison`
- `GET /analytics/heatmap`
- `GET /analytics/bias`
- `POST /predict`
- `GET /ethics`

## Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

or

```bash
bash scripts/run_dashboard.sh
```

## Generate Screenshots

```bash
python scripts/generate_screenshots.py
```

Generated files are stored in `artifacts/screenshots/`.

## Ethics and Safe-Use Notes

- Macro-level analysis only (state-level decision support)
- Not for individual profiling or punitive automation
- High reported crime may reflect better reporting infrastructure
- Human review is required for all policy decisions

## Deliverables

- Screenshots: `artifacts/screenshots/`
- Report: `artifacts/reports/project_report.md`
- Demo script: `artifacts/reports/demo_script.md`
- Trained models: `models/`
