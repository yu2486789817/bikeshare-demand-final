from __future__ import annotations

import numpy as np
import pandas as pd

from bikeshare.lstm_model import LSTMConfig, _make_sequence_arrays
from bikeshare.modeling import _score


def _feature_frame(rows: int = 64) -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=rows, freq="h")
    cnt = np.arange(rows, dtype=float) + 10
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "cnt": cnt,
            "duration_mean_min": np.full(rows, 12.0),
            "unique_start_stations": np.full(rows, 5.0),
            "member_share": np.full(rows, 0.65),
            "casual_share": np.full(rows, 0.35),
            "temperature_2m": np.linspace(0, 20, rows),
            "relative_humidity_2m": np.full(rows, 50.0),
            "precipitation": np.zeros(rows),
            "wind_speed_10m": np.full(rows, 8.0),
            "hour": timestamp.hour,
            "weekday": timestamp.weekday,
            "month": timestamp.month,
            "is_weekend": timestamp.weekday.isin([5, 6]).astype(int),
            "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
            "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
            "weekday_sin": np.sin(2 * np.pi * timestamp.weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * timestamp.weekday / 7),
            "lag_1h": cnt,
            "lag_24h": cnt,
            "rolling_24h_mean": cnt,
            "rolling_168h_mean": cnt,
            "electric_share": np.full(rows, 0.5),
            "classic_share": np.full(rows, 0.5),
        }
    )


def test_lstm_sequence_builder_respects_train_boundary() -> None:
    features = _feature_frame()
    train_x, train_y, test_x, test_y, timestamps, _, _ = _make_sequence_arrays(
        features,
        columns=[c for c in features.columns if c not in {"timestamp", "cnt"}],
        train_end_idx=48,
        config=LSTMConfig(lookback=24),
    )

    assert train_x.shape == (24, 24, 22)
    assert len(train_y) == 24
    assert test_x.shape[0] == 16
    assert len(test_y) == 16
    assert timestamps[0] == pd.Timestamp("2024-01-03 00:00:00")


def test_score_computes_expected_metrics() -> None:
    metrics = _score(pd.Series([100, 200]), np.array([110, 190]))

    assert metrics["MAE"] == 10
    assert round(metrics["MAPE"], 2) == 7.5
