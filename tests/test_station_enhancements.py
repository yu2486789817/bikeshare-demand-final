from __future__ import annotations

import pandas as pd

from bikeshare.station_clustering import cluster_stations
from bikeshare.station_modeling import train_station_models


def _station_hourly() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=80, freq="h")
    rows = []
    for station in ["A", "B", "C", "D"]:
        for i, timestamp in enumerate(timestamps):
            pickup = (i % 8) + (5 if station in {"A", "C"} else 1)
            dropoff = (i % 6) + (5 if station in {"B", "D"} else 1)
            rows.append(
                {
                    "timestamp": timestamp,
                    "station_name": station,
                    "pickup_count": pickup,
                    "dropoff_count": dropoff,
                    "net_demand": pickup - dropoff,
                }
            )
    return pd.DataFrame(rows)


def _hourly() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=80, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cnt": range(80),
            "temperature_2m": [10.0] * 80,
            "relative_humidity_2m": [50.0] * 80,
            "precipitation": [0.0] * 80,
            "wind_speed_10m": [5.0] * 80,
            "is_holiday": [1 if timestamp.date() == pd.Timestamp("2024-01-01").date() else 0 for timestamp in timestamps],
            "is_pre_holiday": [0] * 80,
            "is_post_holiday": [0] * 80,
        }
    )


def test_station_clustering_exports_labels(tmp_path) -> None:
    output = tmp_path / "station_clusters.csv"

    clusters = cluster_stations(_station_hourly(), output_path=output, n_clusters=4)

    assert output.exists()
    assert {"station_name", "cluster_id", "cluster_label"}.issubset(clusters.columns)
    assert len(clusters) == 4


def test_station_prediction_outputs_metrics_and_predictions(tmp_path) -> None:
    metrics_path = tmp_path / "station_model_metrics.csv"
    predictions_path = tmp_path / "station_predictions.csv"

    train_station_models(
        _station_hourly(),
        _hourly(),
        top_n=2,
        train_ratio=0.75,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)

    assert metrics["station_name"].nunique() == 2
    assert {"ridge", "random_forest", "hist_gradient_boosting", "xgboost"}.issubset(set(metrics["model"]))
    assert {"timestamp", "station_name", "actual", "xgboost"}.issubset(predictions.columns)
