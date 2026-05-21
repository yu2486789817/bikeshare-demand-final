from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR
from .features import TARGET_COLUMN, feature_columns


@dataclass(frozen=True)
class LSTMConfig:
    lookback: int = 24
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 128
    epochs: int = 35
    learning_rate: float = 0.001
    random_state: int = 42


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the LSTM model. Install it with "
            "`python -m pip install torch` or a CUDA-enabled PyTorch build."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.clip(y_true, 1, None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAPE": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _make_sequence_arrays(
    features: pd.DataFrame,
    columns: list[str],
    train_end_idx: int,
    config: LSTMConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[pd.Timestamp],
    StandardScaler,
    StandardScaler,
]:
    ordered = features.sort_values("timestamp").reset_index(drop=True)
    x_all = ordered[columns].to_numpy(dtype=np.float32)
    y_all = ordered[TARGET_COLUMN].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(x_all[:train_end_idx])
    x_scaled = scaler.transform(x_all).astype(np.float32)
    target_scaler = StandardScaler()
    target_scaler.fit(y_all[:train_end_idx].reshape(-1, 1))
    y_scaled = target_scaler.transform(y_all.reshape(-1, 1)).ravel().astype(np.float32)

    train_x, train_y, test_x, test_y, test_timestamps = [], [], [], [], []
    for target_idx in range(config.lookback, len(ordered)):
        start_idx = target_idx - config.lookback
        sequence = x_scaled[start_idx:target_idx]
        target = y_scaled[target_idx]
        if target_idx < train_end_idx:
            train_x.append(sequence)
            train_y.append(target)
        else:
            test_x.append(sequence)
            test_y.append(y_all[target_idx])
            test_timestamps.append(ordered.loc[target_idx, "timestamp"])

    if not train_x or not test_x:
        raise ValueError("Not enough rows to build LSTM train/test sequences")

    return (
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.float32),
        np.asarray(test_x, dtype=np.float32),
        np.asarray(test_y, dtype=np.float32),
        test_timestamps,
        scaler,
        target_scaler,
    )


def train_lstm_model(
    features: pd.DataFrame,
    train_end_idx: int,
    config: LSTMConfig | None = None,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, object]]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    config = config or LSTMConfig()

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    columns = feature_columns(features)
    train_x, train_y, test_x, test_y, test_timestamps, scaler, target_scaler = _make_sequence_arrays(
        features, columns, train_end_idx, config
    )

    class DemandLSTM(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(config.hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandLSTM(input_size=len(columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator()
    generator.manual_seed(config.random_state)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    model.train()
    last_loss = 0.0
    for _ in range(config.epochs):
        batch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(batch_losses))

    model.eval()
    test_tensor = torch.from_numpy(test_x).to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_tensor), config.batch_size):
            batch_pred = model(test_tensor[start : start + config.batch_size])
            preds.append(batch_pred.detach().cpu().numpy())
    y_pred_scaled = np.concatenate(preds).reshape(-1, 1)
    y_pred = np.clip(target_scaler.inverse_transform(y_pred_scaled).ravel(), 0, None)

    metrics = _score(test_y, y_pred)
    prediction_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test_timestamps),
            "actual": test_y,
            "lstm": y_pred,
        }
    )
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "lookback": config.lookback,
        "epochs": config.epochs,
        "last_train_loss": last_loss,
        "feature_columns": columns,
    }
    joblib.dump(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "target_scaler": target_scaler,
            "config": config,
            "metadata": metadata,
        },
        MODELS_DIR / "lstm.joblib",
    )
    return metrics, prediction_frame, metadata
