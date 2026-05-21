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


def run_pipeline(months: list[str], force_download: bool = False) -> None:
    ensure_project_dirs()
    paths = build_processed_data(months, force_download=force_download)
    hourly = pd.read_csv(paths.hourly_demand)
    features = make_features(hourly)
    features_path = PROCESSED_DIR / "model_features.csv"
    features.to_csv(features_path, index=False)

    artifacts = train_models(features)
    station_hourly = pd.read_csv(paths.station_hourly)
    recommendations = export_dispatch_recommendations(station_hourly)
    station_clusters = cluster_stations(station_hourly)
    station_artifacts = train_station_models(station_hourly, features)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.months, force_download=args.force_download)


if __name__ == "__main__":
    main()
