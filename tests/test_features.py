from __future__ import annotations

import pandas as pd

from bikeshare.dispatch import summarize_dispatch
from bikeshare.features import feature_columns, make_features, temporal_train_test_split


def test_make_features_adds_lags_and_time_features() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
            "cnt": range(48),
            "duration_mean_min": [8.0] * 48,
            "unique_start_stations": [3] * 48,
            "member_share": [0.7] * 48,
            "casual_share": [0.3] * 48,
            "temperature_2m": [5.0] * 48,
            "relative_humidity_2m": [60.0] * 48,
            "precipitation": [0.0] * 48,
            "wind_speed_10m": [12.0] * 48,
        }
    )

    features = make_features(hourly)

    assert {"hour_sin", "hour_cos", "lag_24h", "rolling_24h_mean", "is_holiday"}.issubset(features.columns)
    assert features["lag_24h"].isna().sum() == 0
    assert features.loc[features["timestamp"].dt.date == pd.Timestamp("2024-01-01").date(), "is_holiday"].eq(1).all()
    assert "timestamp" not in feature_columns(features)
    assert "cnt" not in feature_columns(features)


def test_temporal_split_preserves_order() -> None:
    features = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="h"), "cnt": range(10)})
    train, test = temporal_train_test_split(features, train_ratio=0.8)

    assert len(train) == 8
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()


def test_dispatch_recommendation_direction() -> None:
    station = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "station_name": ["A", "A", "B", "B"],
            "pickup_count": [10, 8, 1, 2],
            "dropoff_count": [2, 3, 8, 7],
            "net_demand": [8, 5, -7, -5],
        }
    )

    summary = summarize_dispatch(station, top_n=2).set_index("station_name")

    assert summary.loc["A", "recommended_action"] == "优先补车"
    assert summary.loc["B", "recommended_action"] == "优先清车"


def test_dispatch_recommendations_skip_unknown_station() -> None:
    station = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "station_name": ["Unknown station", "Unknown station", "A", "A"],
            "pickup_count": [100, 120, 10, 12],
            "dropoff_count": [5, 6, 2, 3],
            "net_demand": [95, 114, 8, 9],
        }
    )

    summary = summarize_dispatch(station, top_n=5)

    assert "Unknown station" not in set(summary["station_name"])
    assert summary["station_name"].tolist() == ["A"]
