from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TABLES_DIR


def build_station_profiles(station_hourly: pd.DataFrame) -> pd.DataFrame:
    data = station_hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["is_weekend"] = data["timestamp"].dt.weekday.isin([5, 6]).astype(int)
    data["volume"] = data["pickup_count"] + data["dropoff_count"]
    data["morning_net"] = np.where(data["hour"].between(6, 10), data["net_demand"], 0)
    data["evening_net"] = np.where(data["hour"].between(16, 19), data["net_demand"], 0)
    data["weekend_volume"] = np.where(data["is_weekend"] == 1, data["volume"], 0)

    profiles = (
        data.groupby("station_name")
        .agg(
            pickup_total=("pickup_count", "sum"),
            dropoff_total=("dropoff_count", "sum"),
            volume_total=("volume", "sum"),
            avg_net_demand=("net_demand", "mean"),
            avg_abs_net_demand=("net_demand", lambda s: float(np.abs(s).mean())),
            peak_outflow=("net_demand", "max"),
            peak_inflow=("net_demand", "min"),
            morning_net=("morning_net", "mean"),
            evening_net=("evening_net", "mean"),
            weekend_volume=("weekend_volume", "sum"),
            active_hours=("timestamp", "nunique"),
        )
        .reset_index()
    )
    profiles["weekend_share"] = profiles["weekend_volume"] / profiles["volume_total"].clip(lower=1)
    profiles["pickup_dropoff_ratio"] = profiles["pickup_total"] / profiles["dropoff_total"].clip(lower=1)
    return profiles.sort_values("volume_total", ascending=False).reset_index(drop=True)


def _label_clusters(clustered: pd.DataFrame) -> dict[int, str]:
    cluster_summary = (
        clustered.groupby("cluster_id")
        .agg(
            avg_net_demand=("avg_net_demand", "mean"),
            morning_net=("morning_net", "mean"),
            evening_net=("evening_net", "mean"),
            weekend_share=("weekend_share", "mean"),
        )
        .reset_index()
    )
    weekend_threshold = cluster_summary["weekend_share"].median()
    labels: dict[int, str] = {}
    for row in cluster_summary.itertuples(index=False):
        if row.weekend_share > weekend_threshold and abs(row.avg_net_demand) < 0.35:
            label = "景点休闲型"
        elif row.avg_net_demand >= 0.05 or row.morning_net > row.evening_net:
            label = "通勤流出型"
        elif row.avg_net_demand <= -0.05 or row.evening_net > row.morning_net:
            label = "通勤流入型"
        else:
            label = "混合型"
        labels[int(row.cluster_id)] = label

    used = set()
    for cluster_id in sorted(labels):
        label = labels[cluster_id]
        if label in used:
            labels[cluster_id] = "混合型"
        used.add(labels[cluster_id])
    return labels


def cluster_stations(
    station_hourly: pd.DataFrame,
    output_path: Path = TABLES_DIR / "station_clusters.csv",
    n_clusters: int = 4,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles = build_station_profiles(station_hourly)
    if profiles.empty:
        profiles.to_csv(output_path, index=False)
        return profiles

    feature_columns = [
        "pickup_total",
        "dropoff_total",
        "avg_net_demand",
        "avg_abs_net_demand",
        "peak_outflow",
        "peak_inflow",
        "morning_net",
        "evening_net",
        "weekend_share",
        "pickup_dropoff_ratio",
    ]
    cluster_count = min(n_clusters, len(profiles), len(profiles[feature_columns].drop_duplicates()))
    if cluster_count < 1:
        profiles["cluster_id"] = 0
        profiles["cluster_label"] = "混合型"
        profiles.to_csv(output_path, index=False)
        return profiles
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=cluster_count, random_state=42, n_init=20)),
        ]
    )
    profiles["cluster_id"] = pipeline.fit_predict(profiles[feature_columns])
    labels = _label_clusters(profiles)
    profiles["cluster_label"] = profiles["cluster_id"].map(labels)
    profiles.to_csv(output_path, index=False)
    return profiles
