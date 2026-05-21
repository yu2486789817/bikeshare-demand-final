from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bikeshare.config import MODELS_DIR, PROCESSED_DIR, TABLES_DIR
from bikeshare.modeling import load_metrics


st.set_page_config(
    page_title="共享单车需求预测与调度分析",
    page_icon="🚲",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --accent: #0f766e;
        --accent-soft: #ccfbf1;
        --ink: #10201d;
        --muted: #5f6f6b;
        --line: #d7e4df;
        --surface: #f7fbf9;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0 !important;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
    }
    .project-note {
        border-left: 4px solid var(--accent);
        background: var(--accent-soft);
        padding: 12px 16px;
        border-radius: 6px;
        color: var(--ink);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        hourly = load_csv(PROCESSED_DIR / "hourly_demand.csv")
        features = load_csv(PROCESSED_DIR / "model_features.csv")
        station = load_csv(PROCESSED_DIR / "station_hourly.csv")
        predictions = load_csv(MODELS_DIR / "predictions.csv")
        metrics = load_metrics(MODELS_DIR / "metrics.json")
    except FileNotFoundError as exc:
        st.error(f"缺少输出文件：{exc}. 请先运行 `python -m bikeshare.pipeline --months 202401`。")
        st.stop()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    station["timestamp"] = pd.to_datetime(station["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    return hourly, features, station, predictions, metrics


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_csv(path)


@st.cache_data(show_spinner=False)
def load_training_info() -> dict:
    path = MODELS_DIR / "training_info.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


hourly_df, features_df, station_df, predictions_df, metrics_df = require_outputs()
training_info = load_training_info()

st.title("共享单车需求预测与调度分析")
st.caption("Capital Bikeshare trip history + Open-Meteo weather | 多模型预测 | 站点调度建议")

total_rides = int(hourly_df["cnt"].sum())
date_range = f"{hourly_df['timestamp'].min():%Y-%m-%d} 至 {hourly_df['timestamp'].max():%Y-%m-%d}"
best_model = str(metrics_df.index[0])
best_rmse = metrics_df.iloc[0]["RMSE"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("总骑行量", f"{total_rides:,}")
col2.metric("小时样本", f"{len(hourly_df):,}")
col3.metric("最佳模型", best_model)
col4.metric("最佳 RMSE", f"{best_rmse:.1f}")

tab_overview, tab_predict, tab_compare, tab_dispatch, tab_about = st.tabs(
    ["数据概览", "需求预测", "模型对比", "站点调度", "项目说明"]
)

with tab_overview:
    st.subheader("数据概览")
    st.markdown(f"<div class='project-note'>分析时间范围：{date_range}</div>", unsafe_allow_html=True)
    daily = hourly_df.assign(date=hourly_df["timestamp"].dt.date).groupby("date", as_index=False)["cnt"].sum()
    fig_daily = px.line(daily, x="date", y="cnt", title="每日骑行总量")
    st.plotly_chart(fig_daily, width="stretch")

    left, right = st.columns(2)
    hour_pattern = hourly_df.groupby(hourly_df["timestamp"].dt.hour)["cnt"].mean().reset_index()
    hour_pattern.columns = ["hour", "avg_demand"]
    left.plotly_chart(
        px.bar(hour_pattern, x="hour", y="avg_demand", title="小时平均需求"),
        width="stretch",
    )
    weather_cols = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
    right.plotly_chart(
        px.imshow(
            hourly_df[["cnt", *weather_cols]].corr(numeric_only=True),
            text_auto=".2f",
            title="需求与天气变量相关性",
            color_continuous_scale="Teal",
        ),
        width="stretch",
    )
    if "is_holiday" in features_df.columns:
        holiday_view = features_df.assign(
            day_type=features_df["is_holiday"].map({1: "节假日", 0: "非节假日"})
        )
        st.plotly_chart(
            px.bar(
                holiday_view.groupby("day_type", as_index=False)["cnt"].mean(),
                x="day_type",
                y="cnt",
                title="节假日 vs 非节假日平均小时需求",
            ),
            width="stretch",
        )

with tab_predict:
    st.subheader("需求预测")
    model_columns = [c for c in predictions_df.columns if c not in {"timestamp", "actual"}]
    model_choice = st.selectbox("选择模型", model_columns, index=model_columns.index(best_model))
    display_days = st.slider("显示最近测试天数", min_value=3, max_value=30, value=14)
    cutoff = predictions_df["timestamp"].max() - pd.Timedelta(days=display_days)
    view = predictions_df[predictions_df["timestamp"] >= cutoff]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["actual"], name="真实需求", mode="lines"))
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view[model_choice], name="预测需求", mode="lines"))
    fig.update_layout(title=f"{model_choice} 测试集预测对比", xaxis_title="时间", yaxis_title="小时骑行量")
    st.plotly_chart(fig, width="stretch")

    error_view = predictions_df.assign(error=lambda d: d[model_choice] - d["actual"])
    st.plotly_chart(
        px.histogram(error_view, x="error", nbins=40, title="预测误差分布"),
        width="stretch",
    )

with tab_compare:
    st.subheader("模型对比")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}", "R2": "{:.3f}"}))
    metrics_long = metrics_df.reset_index(names="model").melt(id_vars="model", value_vars=["MAE", "RMSE", "MAPE"])
    st.plotly_chart(
        px.bar(metrics_long, x="model", y="value", color="variable", barmode="group", title="模型误差指标对比"),
        width="stretch",
    )
    lstm_info = training_info.get("lstm", {})
    if lstm_info:
        st.info(
            f"LSTM 使用过去 {lstm_info.get('lookback')} 小时窗口训练，"
            f"设备：{lstm_info.get('device')}，PyTorch：{lstm_info.get('torch_version')}。"
        )
    st.markdown("特征解释重点：小时周期、上一天同小时需求、滚动均值与天气变量共同影响预测结果。")

with tab_dispatch:
    st.subheader("站点调度")
    recommendations_path = TABLES_DIR / "dispatch_recommendations.csv"
    recommendations = load_csv(recommendations_path) if recommendations_path.exists() else pd.DataFrame()
    clusters = load_optional_csv(TABLES_DIR / "station_clusters.csv")
    station_metrics = load_optional_csv(TABLES_DIR / "station_model_metrics.csv")
    station_predictions = load_optional_csv(MODELS_DIR / "station_predictions.csv")
    if recommendations.empty:
        st.warning("未找到调度建议，请重新运行 pipeline。")
    else:
        st.dataframe(recommendations)
        if not clusters.empty:
            left, right = st.columns([1, 2])
            left.plotly_chart(
                px.histogram(clusters, x="cluster_label", title="站点聚类分布"),
                width="stretch",
            )
            right.plotly_chart(
                px.scatter(
                    clusters,
                    x="avg_net_demand",
                    y="weekend_share",
                    size="volume_total",
                    color="cluster_label",
                    hover_name="station_name",
                    title="站点画像聚类",
                ),
                width="stretch",
            )
            st.dataframe(
                clusters[
                    [
                        "station_name",
                        "cluster_label",
                        "volume_total",
                        "avg_net_demand",
                        "weekend_share",
                        "peak_outflow",
                        "peak_inflow",
                    ]
                ].head(20)
            )
        station_choice = st.selectbox("选择站点查看净需求", recommendations["station_name"].tolist())
        station_view = station_df[station_df["station_name"] == station_choice].copy()
        station_view = station_view.sort_values("timestamp").tail(24 * 14)
        st.plotly_chart(
            px.line(
                station_view,
                x="timestamp",
                y="net_demand",
                title=f"{station_choice} 最近测试窗口净需求",
            ),
            width="stretch",
        )
        st.markdown(
            "净需求为 pickup - dropoff；正值表示车辆流出更多，优先补车；负值表示车辆流入更多，优先清车。"
        )
        if not station_predictions.empty and not station_metrics.empty:
            station_predictions["timestamp"] = pd.to_datetime(station_predictions["timestamp"])
            model_scores = station_metrics.groupby("model", as_index=False)["RMSE"].mean().sort_values("RMSE")
            station_model_columns = [c for c in station_predictions.columns if c not in {"timestamp", "station_name", "actual"}]
            default_station_model = (
                model_scores.iloc[0]["model"]
                if not model_scores.empty and model_scores.iloc[0]["model"] in station_model_columns
                else station_model_columns[0]
            )
            station_model = st.selectbox(
                "站点预测模型",
                station_model_columns,
                index=station_model_columns.index(default_station_model),
            )
            prediction_station = st.selectbox(
                "选择站点查看预测",
                sorted(station_predictions["station_name"].unique()),
                index=0,
            )
            pred_view = station_predictions[station_predictions["station_name"] == prediction_station].tail(24 * 14)
            fig_station_pred = go.Figure()
            fig_station_pred.add_trace(
                go.Scatter(x=pred_view["timestamp"], y=pred_view["actual"], name="真实净需求", mode="lines")
            )
            fig_station_pred.add_trace(
                go.Scatter(x=pred_view["timestamp"], y=pred_view[station_model], name="预测净需求", mode="lines")
            )
            fig_station_pred.update_layout(
                title=f"{prediction_station} Top 20 站点净需求预测",
                xaxis_title="时间",
                yaxis_title="net demand",
            )
            st.plotly_chart(fig_station_pred, width="stretch")
            st.dataframe(
                station_metrics[station_metrics["station_name"] == prediction_station].sort_values("RMSE")
            )

with tab_about:
    st.subheader("项目说明")
    st.markdown(
        """
        **定位**：对应期末项目选项 A + C，多源数据融合挖掘与端到端应用系统开发。

        **流程**：下载骑行数据 -> 清洗时间与站点字段 -> 小时聚合 -> 融合历史天气 -> 构造周期/节假日/滞后/滚动特征 -> 传统模型、集成模型与 LSTM 训练评估 -> 站点聚类与 Top 20 站点预测 -> 导出调度建议 -> Dashboard 展示。

        **运行命令**：

        ```powershell
        python -m pip install -r requirements.txt
        python -m bikeshare.pipeline
        streamlit run app/dashboard.py
        ```
        """
    )
    with (MODELS_DIR / "metrics.json").open("r", encoding="utf-8") as fh:
        st.json(json.load(fh))
