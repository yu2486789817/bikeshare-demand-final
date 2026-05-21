from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bikeshare.config import DEFAULT_MONTHS, PROCESSED_DIR, ensure_project_dirs
from bikeshare.data import build_processed_data
from bikeshare.features import make_features
from bikeshare.modeling import load_metrics, train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible bike-share model experiment.")
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS, help="Months like 202401 202402.")
    parser.add_argument("--reuse-features", action="store_true", help="Skip data download and reuse processed features.")
    parser.add_argument("--no-lstm", action="store_true", help="Skip the PyTorch LSTM model.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Temporal train split ratio.")
    parser.add_argument(
        "--output",
        default="reports/tables/experiment_leaderboard.csv",
        help="CSV path for the model leaderboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    features_path = PROCESSED_DIR / "model_features.csv"
    if args.reuse_features:
        if not features_path.exists():
            raise FileNotFoundError(f"{features_path} does not exist; run without --reuse-features first.")
        features = pd.read_csv(features_path, parse_dates=["timestamp"])
    else:
        paths = build_processed_data(args.months)
        hourly = pd.read_csv(paths.hourly_demand)
        features = make_features(hourly)
        features.to_csv(features_path, index=False)

    artifacts = train_models(features, train_ratio=args.train_ratio, include_lstm=not args.no_lstm)
    leaderboard = load_metrics(artifacts.metrics_path)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output_path)

    print("Experiment complete")
    print(leaderboard.to_string())
    with artifacts.training_info_path.open("r", encoding="utf-8") as fh:
        training_info = json.load(fh)
    if "lstm" in training_info:
        print(f"LSTM device: {training_info['lstm']['device']}")
    print(f"Leaderboard: {output_path}")


if __name__ == "__main__":
    main()
