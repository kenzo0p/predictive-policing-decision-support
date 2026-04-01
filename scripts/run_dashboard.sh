#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_STREAMLIT="$ROOT_DIR/venv/bin/streamlit"

if [[ -x "$VENV_STREAMLIT" ]]; then
  exec "$VENV_STREAMLIT" run "$ROOT_DIR/dashboard/app.py"
fi

echo "Streamlit not found in ./venv. Please install dependencies first."
echo "Then run: ./venv/bin/streamlit run dashboard/app.py"
exit 1
