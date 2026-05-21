from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR
from .modeling import load_metrics


def export_report_figures(
    hourly: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    station_clusters: pd.DataFrame | None = None,
    station_metrics: pd.DataFrame | None = None,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly = hourly.copy()
    predictions = predictions.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

    daily = hourly.assign(date=hourly["timestamp"].dt.date).groupby("date", as_index=False)["cnt"].sum()
    daily_fig = px.line(daily, x="date", y="cnt", title="Daily bike-share demand")
    daily_path = output_dir / "daily_demand.html"
    daily_fig.write_html(daily_path)

    metrics_long = metrics.reset_index(names="model").melt(
        id_vars="model",
        value_vars=["MAE", "RMSE", "MAPE"],
        var_name="metric",
        value_name="value",
    )
    metrics_fig = px.bar(
        metrics_long,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model error comparison",
    )
    metrics_path = output_dir / "model_metrics.html"
    metrics_fig.write_html(metrics_path)

    best_model = str(metrics.index[0])
    comparison_fig = go.Figure()
    tail = predictions.tail(24 * 21)
    comparison_fig.add_trace(go.Scatter(x=tail["timestamp"], y=tail["actual"], name="Actual"))
    comparison_fig.add_trace(go.Scatter(x=tail["timestamp"], y=tail[best_model], name=best_model))
    if "lstm" in tail.columns and best_model != "lstm":
        comparison_fig.add_trace(go.Scatter(x=tail["timestamp"], y=tail["lstm"], name="lstm"))
    comparison_fig.update_layout(
        title="Prediction comparison on the latest test window",
        xaxis_title="Time",
        yaxis_title="Hourly rides",
    )
    comparison_path = output_dir / "prediction_comparison.html"
    comparison_fig.write_html(comparison_path)

    figure_paths = [daily_path, metrics_path, comparison_path]

    if {"is_holiday"}.issubset(hourly.columns):
        holiday = hourly.assign(day_type=hourly["is_holiday"].map({1: "Holiday", 0: "Non-holiday"}))
        holiday_fig = px.bar(
            holiday.groupby("day_type", as_index=False)["cnt"].mean(),
            x="day_type",
            y="cnt",
            title="Average hourly demand on holidays vs non-holidays",
        )
        holiday_path = output_dir / "holiday_demand.html"
        holiday_fig.write_html(holiday_path)
        figure_paths.append(holiday_path)

    if station_clusters is not None and not station_clusters.empty:
        cluster_fig = px.scatter(
            station_clusters,
            x="avg_net_demand",
            y="weekend_share",
            size="volume_total",
            color="cluster_label",
            hover_name="station_name",
            title="Station cluster profile",
        )
        cluster_path = output_dir / "station_clusters.html"
        cluster_fig.write_html(cluster_path)
        figure_paths.append(cluster_path)

    if station_metrics is not None and not station_metrics.empty:
        station_metric_fig = px.box(
            station_metrics,
            x="model",
            y="RMSE",
            points="all",
            title="Top station net-demand prediction RMSE",
        )
        station_metric_path = output_dir / "station_model_metrics.html"
        station_metric_fig.write_html(station_metric_path)
        figure_paths.append(station_metric_path)

    return figure_paths


def export_report_assets(
    hourly: pd.DataFrame,
    recommendations: pd.DataFrame,
    metrics_path: Path,
    predictions_path: Path | None = None,
    station_clusters: pd.DataFrame | None = None,
    station_metrics_path: Path | None = None,
    station_predictions_path: Path | None = None,
    summary_path: Path = REPORTS_DIR / "project_summary.md",
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(metrics_path)
    metrics.to_csv(TABLES_DIR / "model_metrics.csv")
    if predictions_path is not None and predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        station_metrics = (
            pd.read_csv(station_metrics_path)
            if station_metrics_path is not None and station_metrics_path.exists()
            else None
        )
        figure_paths = export_report_figures(hourly, predictions, metrics, station_clusters, station_metrics)
    else:
        figure_paths = []

    with metrics_path.open("r", encoding="utf-8") as fh:
        raw_metrics = json.load(fh)
    best = min(raw_metrics, key=lambda name: raw_metrics[name]["RMSE"])

    start = pd.to_datetime(hourly["timestamp"]).min()
    end = pd.to_datetime(hourly["timestamp"]).max()
    content = f"""# 项目摘要：共享单车需求预测与调度分析

## 数据与范围

- 骑行数据：Capital Bikeshare 月度 trip history。
- 天气数据：Open-Meteo Historical Weather API。
- 时间范围：{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}。
- 小时级样本数：{len(hourly):,}。

## 方法

- 将骑行记录聚合为小时级总需求，并提取会员比例、车辆类型比例、历史滞后需求和滚动均值。
- 按小时融合温度、湿度、降水、风速等天气变量。
- 使用时间切分评估 Seasonal Naive、Ridge、Random Forest、HistGradientBoosting、XGBoost 和 LSTM。
- 增加美国联邦节假日特征、Top 站点聚类和 Top 20 站点净需求预测。

## 结果

- 当前 RMSE 最低模型：{best}。
- 站点调度建议已导出到 `reports/tables/dispatch_recommendations.csv`。
- 站点聚类画像已导出到 `reports/tables/station_clusters.csv`。
- Top 20 站点预测指标已导出到 `reports/tables/station_model_metrics.csv`。
- 模型指标已导出到 `reports/tables/model_metrics.csv`。
- 报告图表已导出到 `reports/figures/`：{", ".join(path.name for path in figure_paths) if figure_paths else "未生成"}。

## 可展示结论

- 需求存在明显小时周期和工作日/周末差异。
- 天气变量和历史滞后需求共同影响短期需求预测。
- Top 站点存在稳定的净流出或净流入模式，可转化为补车/清车建议。
- 站点聚类可区分通勤流出、通勤流入、景点休闲和混合站点。
"""
    summary_path.write_text(content, encoding="utf-8")
