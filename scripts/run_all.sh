#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter at $PYTHON_BIN"
  echo "Create a venv first: python3 -m venv venv"
  exit 1
fi

echo "[1/4] Running data preprocessing"
"$PYTHON_BIN" "$ROOT_DIR/src/data_pipeline.py" \
  --input "$ROOT_DIR/data/raw/dstrIPC_2013.csv" \
  --output "$ROOT_DIR/data/processed/clean_crime_data.csv"

echo "[2/4] Training ML model"
"$PYTHON_BIN" "$ROOT_DIR/scripts/train_model.py" \
  --data "$ROOT_DIR/data/raw/dstrIPC_2013.csv" \
  --model-dir "$ROOT_DIR/models"

echo "[3/4] Generating screenshots"
"$PYTHON_BIN" "$ROOT_DIR/scripts/generate_screenshots.py"

echo "[4/4] Done"
echo "You can now start services:"
echo "  API:       cd $ROOT_DIR && $ROOT_DIR/venv/bin/uvicorn api.main:app --reload"
echo "  Streamlit: cd $ROOT_DIR && $ROOT_DIR/venv/bin/streamlit run dashboard/app.py"
