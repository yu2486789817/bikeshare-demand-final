from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TABLES_DIR

UNKNOWN_STATION_NAMES = {"", "Unknown station", "unknown station", "nan", "None"}


def summarize_dispatch(station_hourly: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    data = station_hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data[~data["station_name"].astype(str).str.strip().isin(UNKNOWN_STATION_NAMES)]
    summary = (
        data.groupby("station_name")
        .agg(
            pickup_total=("pickup_count", "sum"),
            dropoff_total=("dropoff_count", "sum"),
            avg_net_demand=("net_demand", "mean"),
            peak_outflow=("net_demand", "max"),
            peak_inflow=("net_demand", "min"),
        )
        .reset_index()
    )
    summary["imbalance_score"] = summary["peak_outflow"].abs() + summary["peak_inflow"].abs()
    summary["recommended_action"] = summary["avg_net_demand"].apply(
        lambda value: "优先补车" if value > 0.25 else ("优先清车" if value < -0.25 else "保持观测")
    )
    return summary.sort_values("imbalance_score", ascending=False).head(top_n)


def export_dispatch_recommendations(
    station_hourly: pd.DataFrame,
    output_path: Path = TABLES_DIR / "dispatch_recommendations.csv",
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations = summarize_dispatch(station_hourly)
    recommendations.to_csv(output_path, index=False)
    return recommendations
