#!/usr/bin/env python3
"""Train and persist crime risk models for API/dashboard consumption."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from crime_analytics.services.analytics_service import load_crime_dataset
from crime_analytics.services.model_service import train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crime risk model")
    parser.add_argument(
        "--data",
        default="data/raw/dstrIPC_2013.csv",
        help="Path to input crime CSV",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to store joblib artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    bundle = load_crime_dataset(args.data)
    result = train_and_save(bundle, model_dir=args.model_dir)

    print("Training complete")
    print(f"Model saved at: {result.model_path}")
    print(f"Metadata saved at: {result.metadata_path}")
    print("Metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
