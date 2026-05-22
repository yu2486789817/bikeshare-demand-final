from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR, TABLES_DIR, WEATHER_COLUMNS
from .features import TARGET_COLUMN


@dataclass(frozen=True)
class StationModelArtifacts:
    metrics_path: Path = TABLES_DIR / "station_model_metrics.csv"
    predictions_path: Path = MODELS_DIR / "station_predictions.csv"


def _mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.abs(np.asarray(y_true)), 1, None)
    return float(np.mean(np.abs((np.asarray(y_true) - y_pred) / denominator)) * 100)


def _score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAPE": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _build_station_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=220,
            random_state=random_state,
            l2_regularization=0.02,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError as exc:
        raise RuntimeError("xgboost is required for station-level prediction.") from exc
    return models


def _add_station_features(station: pd.DataFrame, hourly: pd.DataFrame, station_name: str) -> pd.DataFrame:
    station = station.copy()
    hourly = hourly.copy()
    station["timestamp"] = pd.to_datetime(station["timestamp"])
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])

    full_index = pd.date_range(hourly["timestamp"].min(), hourly["timestamp"].max(), freq="h")
    station = (
        station.set_index("timestamp")
        .reindex(full_index)
        .rename_axis("timestamp")
        .reset_index()
    )
    station["station_name"] = station_name
    for column in ["pickup_count", "dropoff_count", "net_demand"]:
        station[column] = station[column].fillna(0)

    weather_cols = [column for column in WEATHER_COLUMNS if column in hourly.columns]
    station = station.merge(hourly[["timestamp", *weather_cols]], on="timestamp", how="left")
    station[weather_cols] = station[weather_cols].ffill().bfill().fillna(0)
    station["hour"] = station["timestamp"].dt.hour
    station["weekday"] = station["timestamp"].dt.weekday
    station["month"] = station["timestamp"].dt.month
    station["is_weekend"] = station["weekday"].isin([5, 6]).astype(int)
    for column in ["is_holiday", "is_pre_holiday", "is_post_holiday"]:
        if column in hourly.columns:
            station[column] = hourly.set_index("timestamp").reindex(station["timestamp"])[column].fillna(0).to_numpy()
        else:
            station[column] = 0
    station["hour_sin"] = np.sin(2 * np.pi * station["hour"] / 24)
    station["hour_cos"] = np.cos(2 * np.pi * station["hour"] / 24)
    station["weekday_sin"] = np.sin(2 * np.pi * station["weekday"] / 7)
    station["weekday_cos"] = np.cos(2 * np.pi * station["weekday"] / 7)
    station["lag_1h"] = station["net_demand"].shift(1)
    station["lag_24h"] = station["net_demand"].shift(24)
    station["rolling_24h_mean"] = station["net_demand"].shift(1).rolling(24, min_periods=3).mean()
    station["rolling_168h_mean"] = station["net_demand"].shift(1).rolling(168, min_periods=24).mean()
    numeric_columns = station.select_dtypes(include=["number"]).columns
    station[numeric_columns] = station[numeric_columns].ffill().bfill().fillna(0)
    return station


def _feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "station_name", "pickup_count", "dropoff_count", "net_demand", TARGET_COLUMN}
    return [
        column
        for column in data.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]


def train_station_models(
    station_hourly: pd.DataFrame,
    hourly: pd.DataFrame,
    top_n: int = 20,
    train_ratio: float = 0.8,
    metrics_path: Path = TABLES_DIR / "station_model_metrics.csv",
    predictions_path: Path = MODELS_DIR / "station_predictions.csv",
) -> StationModelArtifacts:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    station_hourly = station_hourly.copy()
    station_hourly["timestamp"] = pd.to_datetime(station_hourly["timestamp"])
    volumes = (
        station_hourly.assign(volume=station_hourly["pickup_count"] + station_hourly["dropoff_count"])
        .groupby("station_name")["volume"]
        .sum()
        .sort_values(ascending=False)
    )
    stations = volumes.head(top_n).index.tolist()

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for station_idx, station_name in enumerate(stations, start=1):
        print(f"Training station {station_idx}/{len(stations)}: {station_name}...", flush=True)
        station_data = station_hourly[station_hourly["station_name"] == station_name]
        features = _add_station_features(station_data, hourly, station_name)
        split_at = max(1, int(len(features) * train_ratio))
        if split_at >= len(features):
            split_at = len(features) - 1
        train = features.iloc[:split_at].copy()
        test = features.iloc[split_at:].copy()
        columns = _feature_columns(features)
        x_train = train[columns]
        y_train = train["net_demand"]
        x_test = test[columns]
        y_test = test["net_demand"]

        station_predictions = pd.DataFrame(
            {
                "timestamp": test["timestamp"],
                "station_name": station_name,
                "actual": y_test.to_numpy(),
            }
        )
        for model_name, model in _build_station_models().items():
            print(f"  Training station model: {model_name}...", flush=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            station_predictions[model_name] = y_pred
            score = _score(y_test, y_pred)
            metrics.append({"station_name": station_name, "model": model_name, **score})
        predictions.append(station_predictions)

    artifacts = StationModelArtifacts(metrics_path=metrics_path, predictions_path=predictions_path)
    artifacts.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(artifacts.predictions_path, index=False)
    pd.DataFrame(metrics).to_csv(artifacts.metrics_path, index=False)
    return artifacts
