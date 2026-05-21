from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

TARGET_COLUMN = "cnt"


def make_features(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["hour"] = data["timestamp"].dt.hour
    data["weekday"] = data["timestamp"].dt.weekday
    data["month"] = data["timestamp"].dt.month
    data["is_weekend"] = data["weekday"].isin([5, 6]).astype(int)
    holiday_calendar = holidays.US(years=range(data["timestamp"].dt.year.min(), data["timestamp"].dt.year.max() + 1))
    dates = data["timestamp"].dt.date
    data["is_holiday"] = dates.isin(holiday_calendar).astype(int)
    data["is_pre_holiday"] = (dates + pd.Timedelta(days=1)).isin(holiday_calendar).astype(int)
    data["is_post_holiday"] = (dates - pd.Timedelta(days=1)).isin(holiday_calendar).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)
    data["lag_1h"] = data[TARGET_COLUMN].shift(1)
    data["lag_24h"] = data[TARGET_COLUMN].shift(24)
    data["rolling_24h_mean"] = data[TARGET_COLUMN].shift(1).rolling(24, min_periods=3).mean()
    data["rolling_168h_mean"] = data[TARGET_COLUMN].shift(1).rolling(168, min_periods=24).mean()

    for column in ["member", "casual", "classic_bike", "electric_bike", "docked_bike"]:
        if column not in data.columns:
            data[column] = 0
    data["electric_share"] = data["electric_bike"] / data[TARGET_COLUMN].clip(lower=1)
    data["classic_share"] = data["classic_bike"] / data[TARGET_COLUMN].clip(lower=1)

    numeric_columns = data.select_dtypes(include=["number"]).columns
    data[numeric_columns] = data[numeric_columns].ffill().bfill().fillna(0)
    return data


def feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        TARGET_COLUMN,
        "member",
        "casual",
        "classic_bike",
        "electric_bike",
        "docked_bike",
    }
    return [
        column
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]


def temporal_train_test_split(
    features: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    split_at = max(1, int(len(features) * train_ratio))
    if split_at >= len(features):
        split_at = len(features) - 1
    train = features.iloc[:split_at].copy()
    test = features.iloc[split_at:].copy()
    return train, test
