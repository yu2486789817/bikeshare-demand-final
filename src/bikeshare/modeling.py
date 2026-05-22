from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR
from .features import TARGET_COLUMN, feature_columns, temporal_train_test_split
from .lstm_model import LSTMConfig, train_lstm_model


@dataclass(frozen=True)
class ModelArtifacts:
    metrics_path: Path = MODELS_DIR / "metrics.json"
    predictions_path: Path = MODELS_DIR / "predictions.csv"
    training_info_path: Path = MODELS_DIR / "training_info.json"


class SeasonalNaiveRegressor:
    """Predicts demand from the same hour on the previous day when available."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveRegressor":
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if "lag_24h" in x.columns:
            return x["lag_24h"].fillna(self.global_mean_).to_numpy()
        return np.full(len(x), self.global_mean_)


def _build_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "seasonal_naive": SeasonalNaiveRegressor(),
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=300,
            random_state=random_state,
            l2_regularization=0.02,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required by the project plan. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return models


def _mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.asarray(y_true), 1, None)
    return float(np.mean(np.abs((np.asarray(y_true) - y_pred) / denominator)) * 100)


def _score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAPE": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_models(
    features: pd.DataFrame,
    train_ratio: float = 0.8,
    include_lstm: bool = True,
) -> ModelArtifacts:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    features = features.sort_values("timestamp").reset_index(drop=True)
    train, test = temporal_train_test_split(features, train_ratio=train_ratio)
    columns = feature_columns(features)
    x_train = train[columns]
    y_train = train[TARGET_COLUMN]
    x_test = test[columns]
    y_test = test[TARGET_COLUMN]

    metrics: dict[str, dict[str, float]] = {}
    predictions = pd.DataFrame({"timestamp": test["timestamp"], "actual": y_test.to_numpy()})

    for name, model in _build_models().items():
        print(f"Training model: {name}...", flush=True)
        model.fit(x_train, y_train)
        y_pred = np.clip(model.predict(x_test), 0, None)
        metrics[name] = _score(y_test, y_pred)
        predictions[name] = y_pred
        joblib.dump({"model": model, "feature_columns": columns}, MODELS_DIR / f"{name}.joblib")

    artifacts = ModelArtifacts()
    training_info: dict[str, object] = {
        "train_ratio": train_ratio,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns": columns,
    }
    if include_lstm:
        print("Training model: lstm...", flush=True)
        lstm_metrics, lstm_predictions, lstm_metadata = train_lstm_model(
            features,
            train_end_idx=len(train),
            config=LSTMConfig(),
        )
        metrics["lstm"] = lstm_metrics
        predictions = predictions.merge(
            lstm_predictions[["timestamp", "lstm"]],
            on="timestamp",
            how="left",
        )
        training_info["lstm"] = lstm_metadata

    with artifacts.metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    with artifacts.training_info_path.open("w", encoding="utf-8") as fh:
        json.dump(training_info, fh, ensure_ascii=False, indent=2)
    predictions.to_csv(artifacts.predictions_path, index=False)
    return artifacts


def load_metrics(path: Path = MODELS_DIR / "metrics.json") -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return pd.DataFrame(metrics).T.sort_values("RMSE")


def best_model_name(metrics_path: Path = MODELS_DIR / "metrics.json") -> str:
    metrics = load_metrics(metrics_path)
    return str(metrics.index[0])
