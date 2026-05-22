from __future__ import annotations

import argparse

import pandas as pd

from .config import DEFAULT_MONTHS, PROCESSED_DIR, TABLES_DIR, ensure_project_dirs
from .data import DataPaths, build_processed_data
from .dispatch import export_dispatch_recommendations
from .features import make_features
from .modeling import train_models
from .reporting import export_report_assets
from .station_clustering import cluster_stations
from .station_modeling import train_station_models


def run_pipeline(
    months: list[str],
    force_download: bool = False,
    include_lstm: bool = True,
) -> None:
    ensure_project_dirs()
    print(f"Building processed data for {len(months)} month(s)...", flush=True)
    paths = build_processed_data(months, force_download=force_download)
    print("Creating model features...", flush=True)
    hourly = pd.read_csv(paths.hourly_demand)
    features = make_features(hourly)
    features_path = PROCESSED_DIR / "model_features.csv"
    features.to_csv(features_path, index=False)

    print("Training city-level demand models...", flush=True)
    artifacts = train_models(features, include_lstm=include_lstm)
    print("Building station-level dispatch outputs...", flush=True)
    station_hourly = pd.read_csv(paths.station_hourly)
    recommendations = export_dispatch_recommendations(station_hourly)
    station_clusters = cluster_stations(station_hourly)
    print("Training station-level demand models...", flush=True)
    station_artifacts = train_station_models(station_hourly, features)
    print("Exporting report assets...", flush=True)
    export_report_assets(
        features,
        recommendations,
        artifacts.metrics_path,
        artifacts.predictions_path,
        station_clusters=station_clusters,
        station_metrics_path=station_artifacts.metrics_path,
        station_predictions_path=station_artifacts.predictions_path,
    )

    print("Pipeline complete")
    print(f"- hourly demand: {paths.hourly_demand}")
    print(f"- model features: {features_path}")
    print(f"- metrics: {artifacts.metrics_path}")
    print(f"- predictions: {artifacts.predictions_path}")
    print(f"- station clusters: {TABLES_DIR / 'station_clusters.csv'}")
    print(f"- station predictions: {station_artifacts.predictions_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bike-share demand pipeline.")
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS, help="Months like 202401 202402.")
    parser.add_argument("--force-download", action="store_true", help="Redownload trip zip files.")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip the PyTorch LSTM model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.months, force_download=args.force_download, include_lstm=not args.skip_lstm)


if __name__ == "__main__":
    main()
