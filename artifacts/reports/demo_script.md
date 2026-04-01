# Viva / Demo Script (5-7 minutes)

## 1) Problem Statement (30-45 sec)

This system is a predictive policing decision-support dashboard at macro state level. It combines analytics, reporting-bias checks, ML predictions, and ethics constraints. It is designed for policy support, not individual targeting.

## 2) Data and Preprocessing (45-60 sec)

- Raw input: `data/raw/dstrIPC_2013.csv`
- Pipeline: `src/data_pipeline.py`
- Steps:
  - auto-map columns into standard schema
  - validate required fields
  - drop aggregate rows (TOTAL, RLY, CID, etc.)
  - export cleaned data to `data/processed/clean_crime_data.csv`

## 3) Analytics Dashboard (60 sec)

Show Streamlit pages:
- Overview: KPIs and heatmap
- Trends: state and year trend lines
- Comparison: ranking by average crime rate

Mention that high reported crime can also mean better reporting quality.

## 4) Reporting Bias Analysis (45 sec)

Explain bias indicators:
- coefficient of variation (CV)
- z-score-based anomaly
- composite anomaly score
- LOW / MEDIUM / HIGH risk labels

Show `Reporting Bias Analysis` page.

## 5) ML Prediction (60 sec)

Model stack:
- RandomForestRegressor for crime rate
- RandomForestClassifier for risk category

Inputs: state, year, previous crime rate, population
Outputs: predicted crime rate, class, confidence, uncertainty flag

Show the `Crime Risk Prediction` page and highlight confidence interpretation.

## 6) FastAPI Backend (45 sec)

Show that backend serves all app functions:
- `/analytics/*`
- `/predict`
- `/ethics`

Explain that frontend and API share service-layer logic for consistency.

## 7) Ethics and Safeguards (45 sec)

State clearly:
- no individual profiling
- no automated punitive decisions
- human-in-the-loop required
- predictions are advisory signals

Show `Ethical Considerations` page.

## 8) Deliverables Checklist (30 sec)

- README: `README.md`
- Report: `artifacts/reports/project_report.md`
- Screenshots: `artifacts/screenshots/`
- Trained models: `models/`

## Quick Run Commands

```bash
source venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh
```

Then run:

```bash
venv/bin/uvicorn api.main:app --reload
venv/bin/streamlit run dashboard/app.py
```
