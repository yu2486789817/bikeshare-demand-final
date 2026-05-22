from __future__ import annotations

import json
import sys
import zipfile
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bikeshare.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, TABLES_DIR
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


@st.cache_data(show_spinner=False)
def load_station_locations(station_names: tuple[str, ...]) -> pd.DataFrame:
    cached_station = load_optional_csv(PROCESSED_DIR / "station_hourly.csv")
    if {"station_name", "latitude", "longitude"}.issubset(cached_station.columns):
        locations = cached_station[["station_name", "latitude", "longitude"]].dropna().drop_duplicates()
        if not locations.empty:
            return locations

    target_stations = set(station_names)
    if not target_stations:
        return pd.DataFrame(columns=["station_name", "latitude", "longitude"])

    location_frames: list[pd.DataFrame] = []
    usecols = [
        "start_station_name",
        "end_station_name",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
    ]
    found_stations: set[str] = set()
    for zip_path in sorted(RAW_DIR.glob("*capitalbikeshare-tripdata.zip"), reverse=True):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_names = [
                    name
                    for name in zf.namelist()
                    if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
                ]
                if not csv_names:
                    continue
                with zf.open(csv_names[0]) as fh:
                    chunks = pd.read_csv(fh, usecols=usecols, chunksize=200_000)
                    for chunk in chunks:
                        starts = chunk.loc[
                            chunk["start_station_name"].isin(target_stations),
                            ["start_station_name", "start_lat", "start_lng"],
                        ].rename(
                            columns={
                                "start_station_name": "station_name",
                                "start_lat": "latitude",
                                "start_lng": "longitude",
                            }
                        )
                        ends = chunk.loc[
                            chunk["end_station_name"].isin(target_stations),
                            ["end_station_name", "end_lat", "end_lng"],
                        ].rename(
                            columns={
                                "end_station_name": "station_name",
                                "end_lat": "latitude",
                                "end_lng": "longitude",
                            }
                        )
                        sample = pd.concat([starts, ends], ignore_index=True).dropna()
                        if not sample.empty:
                            location_frames.append(sample)
                            found_stations.update(sample["station_name"].unique())
                        if found_stations.issuperset(target_stations):
                            break
        except (KeyError, ValueError, zipfile.BadZipFile):
            continue
        if found_stations.issuperset(target_stations):
            break

    if not location_frames:
        return pd.DataFrame(columns=["station_name", "latitude", "longitude"])

    locations = pd.concat(location_frames, ignore_index=True)
    locations = locations[locations["station_name"].isin(target_stations)]
    return locations.groupby("station_name", as_index=False).agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
    )


def add_weather_scenario(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    precipitation = data["precipitation"] if "precipitation" in data else pd.Series(0, index=data.index)
    temperature = data["temperature_2m"] if "temperature_2m" in data else pd.Series(20, index=data.index)
    wind_speed = data["wind_speed_10m"] if "wind_speed_10m" in data else pd.Series(0, index=data.index)
    conditions = [
        precipitation >= 0.5,
        temperature >= 30,
        temperature <= 5,
        wind_speed >= 25,
    ]
    labels = ["降水", "高温", "低温", "大风"]
    data["weather_scenario"] = np.select(conditions, labels, default="常规天气")
    return data


def weather_scenario_options() -> list[str]:
    return ["全部", "常规天气", "降水", "高温", "低温", "大风"]


def daily_weather_demand(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date
    daily = (
        data.groupby("date", as_index=False)
        .agg(
            daily_demand=("cnt", "sum"),
            precipitation=("precipitation", "sum"),
            max_temperature=("temperature_2m", "max"),
            min_temperature=("temperature_2m", "min"),
            max_wind_speed=("wind_speed_10m", "max"),
        )
    )
    conditions = [
        daily["precipitation"] >= 0.5,
        daily["max_temperature"] >= 30,
        daily["min_temperature"] <= 5,
        daily["max_wind_speed"] >= 25,
    ]
    labels = ["降水", "高温", "低温", "大风"]
    daily["weather_scenario"] = np.select(conditions, labels, default="常规天气")
    return daily


def weather_daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    daily = daily_weather_demand(hourly)
    summary = (
        daily.groupby("weather_scenario", as_index=False)
        .agg(
            daily_demand=("daily_demand", "mean"),
            days=("date", "nunique"),
        )
    )
    scenario_order = weather_scenario_options()[1:]
    summary = pd.DataFrame({"weather_scenario": scenario_order}).merge(summary, on="weather_scenario", how="left")
    normal_mean = summary.loc[summary["weather_scenario"] == "常规天气", "daily_demand"].dropna()
    baseline = float(normal_mean.iloc[0]) if not normal_mean.empty and normal_mean.iloc[0] > 0 else float(daily["daily_demand"].mean())
    summary["daily_demand"] = summary["daily_demand"].fillna(baseline)
    summary["days"] = summary["days"].fillna(0).astype(int)
    summary["relative_to_normal"] = summary["daily_demand"] / baseline if baseline > 0 else 1.0
    return summary


def weather_scenario_factors(hourly: pd.DataFrame) -> dict[str, float]:
    summary = weather_daily_summary(hourly)
    factors = {"全部": 1.0}
    for row in summary.itertuples(index=False):
        factors[row.weather_scenario] = float(row.relative_to_normal)
    return factors


def datetime_picker(
    label: str,
    timestamps: list[pd.Timestamp],
    key: str,
    default_index: int | None = None,
) -> pd.Timestamp:
    if not timestamps:
        return pd.Timestamp.now().floor("h")
    normalized = sorted(pd.Timestamp(timestamp).floor("h") for timestamp in timestamps)
    default_index = len(normalized) - 1 if default_index is None else default_index
    default_index = min(max(default_index, 0), len(normalized) - 1)
    default_timestamp = normalized[default_index]
    st.markdown(f"**{label}**")
    date_col, time_col = st.columns([2, 1])
    selected_date = date_col.date_input(
        "日期",
        value=default_timestamp.date(),
        min_value=normalized[0].date(),
        max_value=normalized[-1].date(),
        key=f"{key}_date",
    )
    selected_time = time_col.time_input(
        "时间",
        value=time(default_timestamp.hour, 0),
        step=timedelta(hours=1),
        key=f"{key}_time",
    )
    selected_hour = selected_time.hour
    return pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)


def add_map_fields(data: pd.DataFrame, value_col: str = "net_demand") -> pd.DataFrame:
    data = data.copy()
    data["net_inflow"] = -data[value_col]
    data["activity"] = data["net_inflow"].abs().clip(lower=1)
    data["flow_label"] = np.where(
        data["net_inflow"] > 0,
        "净流入",
        np.where(data["net_inflow"] < 0, "净流出", "平衡"),
    )
    return data


def station_map_figure(
    data: pd.DataFrame,
    title: str,
    animation_frame: str | None = None,
) -> go.Figure:
    limit = max(float(data["net_inflow"].abs().quantile(0.95)), 1.0)
    hover_data: dict[str, str | bool] = {
        "net_inflow": ":.1f",
        "latitude": False,
        "longitude": False,
        "activity": False,
        "flow_label": True,
    }
    if "pickup_count" in data.columns:
        hover_data["pickup_count"] = ":.1f"
    if "dropoff_count" in data.columns:
        hover_data["dropoff_count"] = ":.1f"
    fig = px.scatter_mapbox(
        data,
        lat="latitude",
        lon="longitude",
        color="net_inflow",
        size="activity",
        size_max=34,
        hover_name="station_name",
        hover_data=hover_data,
        color_continuous_scale=[
            (0.0, "#2563eb"),
            (0.5, "#f8fafc"),
            (1.0, "#dc2626"),
        ],
        range_color=(-limit, limit),
        animation_frame=animation_frame,
        zoom=12,
        center={"lat": float(data["latitude"].mean()), "lon": float(data["longitude"].mean())},
        title=title,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        coloraxis_colorbar={"title": "净流入"},
        height=650,
    )
    return fig


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

tab_overview, tab_predict, tab_compare, tab_detail, tab_about = st.tabs(
    ["数据概览", "需求预测", "模型对比", "站点详情", "项目说明"]
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
    right.markdown(
        """
        **相关性说明**：图中数值是皮尔逊相关系数，范围为 -1 到 1。越接近 1 表示两个变量越倾向于同向变化，越接近 -1 表示越倾向于反向变化，接近 0 表示线性关系较弱。`cnt` 是全系统每小时骑行量，`temperature_2m` 是 2 米气温，`relative_humidity_2m` 是相对湿度，`precipitation` 是降水量，`wind_speed_10m` 是 10 米风速。
        """
    )
    if "is_holiday" in features_df.columns:
        overview_daily = features_df.assign(
            date=features_df["timestamp"].dt.date,
            day_type=features_df["is_holiday"].map({1: "节假日", 0: "非节假日"}),
            weekday=features_df["timestamp"].dt.weekday,
            month=features_df["timestamp"].dt.month,
        )
        daily_summary = (
            overview_daily.groupby(["date", "day_type", "weekday", "month"], as_index=False)["cnt"]
            .sum()
            .rename(columns={"cnt": "daily_demand"})
        )
        holiday_daily, weekday_daily = st.columns(2)
        holiday_daily.plotly_chart(
            px.bar(
                daily_summary.groupby("day_type", as_index=False)["daily_demand"].mean(),
                x="day_type",
                y="daily_demand",
                title="节假日 vs 非节假日平均需求",
                labels={"day_type": "日期类型", "daily_demand": "日平均需求"},
            ),
            width="stretch",
        )
        weekday_order = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"]
        weekday_labels = {
            0: "星期一",
            1: "星期二",
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日",
        }
        weekday_view = daily_summary.assign(weekday_name=daily_summary["weekday"].map(weekday_labels))
        weekday_avg = weekday_view.groupby("weekday_name", as_index=False)["daily_demand"].mean()
        weekday_avg["weekday_name"] = pd.Categorical(
            weekday_avg["weekday_name"],
            categories=weekday_order,
            ordered=True,
        )
        weekday_avg = weekday_avg.sort_values("weekday_name")
        weekday_daily.plotly_chart(
            px.bar(
                weekday_avg,
                x="weekday_name",
                y="daily_demand",
                title="星期平均需求",
                labels={"weekday_name": "星期", "daily_demand": "日平均需求"},
            ),
            width="stretch",
        )

        month_avg = daily_summary.groupby("month", as_index=False)["daily_demand"].mean()
        st.plotly_chart(
            px.bar(
                month_avg,
                x="month",
                y="daily_demand",
                title="月每日平均需求",
                labels={"month": "月份", "daily_demand": "日平均需求"},
            ).update_xaxes(dtick=1),
            width="stretch",
        )
        weather_summary = weather_daily_summary(hourly_df)
        weather_fig = px.bar(
            weather_summary,
            x="weather_scenario",
            y="daily_demand",
            text=weather_summary["relative_to_normal"].map(lambda value: f"{value:.2f}x"),
            hover_data={"days": True, "relative_to_normal": ":.2f", "daily_demand": ":.0f"},
            title="不同天气的日平均需求",
            labels={
                "weather_scenario": "天气状态",
                "daily_demand": "日平均需求",
                "relative_to_normal": "相对常规天气倍数",
                "days": "样本天数",
            },
            color="weather_scenario",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        weather_fig.update_traces(textposition="outside")
        weather_fig.update_layout(showlegend=False)
        st.plotly_chart(weather_fig, width="stretch")
        st.caption(
            "柱顶倍数表示该天气状态的日平均需求相对常规天气的比例；"
            "“需求预测”页站点预测动画中的天气情景系数使用同一张表计算。"
        )

with tab_predict:
    st.subheader("需求预测")
    st.caption("这里预测的是全系统每小时骑行总量：每个时间点的 actual/预测值代表该小时内所有站点合计的出发骑行量。")
    model_columns = [c for c in predictions_df.columns if c not in {"timestamp", "actual"}]
    prediction_controls = st.columns([1, 1, 1])
    model_choice = prediction_controls[0].selectbox("选择模型", model_columns, index=model_columns.index(best_model))
    min_prediction_date = predictions_df["timestamp"].dt.date.min()
    max_prediction_date = predictions_df["timestamp"].dt.date.max()
    start_date = prediction_controls[1].date_input(
        "选择开始日期",
        value=min_prediction_date,
        min_value=min_prediction_date,
        max_value=max_prediction_date,
    )
    display_days = prediction_controls[2].slider("选择测试天数", min_value=1, max_value=30, value=14)
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = start_timestamp + pd.Timedelta(days=display_days)
    view = predictions_df[
        (predictions_df["timestamp"] >= start_timestamp)
        & (predictions_df["timestamp"] < end_timestamp)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["actual"], name="真实需求", mode="lines"))
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view[model_choice], name="预测需求", mode="lines"))
    fig.update_layout(
        title=f"{model_choice} 测试集预测对比（{start_timestamp:%Y-%m-%d} 起 {display_days} 天）",
        xaxis_title="时间",
        yaxis_title="小时骑行量",
    )
    if view.empty:
        st.warning("当前开始日期和测试天数没有可展示的预测数据。")
    else:
        st.plotly_chart(fig, width="stretch")

    error_view = predictions_df.assign(error=lambda d: d[model_choice] - d["actual"])
    st.plotly_chart(
        px.histogram(error_view, x="error", nbins=40, title="预测误差分布"),
        width="stretch",
    )

    st.subheader("站点预测动画")
    st.caption("底图使用 OpenStreetMap；颜色按“净流入 = dropoff - pickup”显示，红色为净流入更多，蓝色为净流出更多。")
    station_locations = load_station_locations(tuple(sorted(station_df["station_name"].dropna().unique())))
    if station_locations.empty:
        st.warning("未找到站点经纬度。请确认 raw 骑行 zip 文件存在，或重新运行 pipeline 生成带坐标的站点数据。")
    else:
        weather_columns = [
            column
            for column in ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]
            if column in hourly_df.columns
        ]
        station_predictions_map = load_optional_csv(MODELS_DIR / "station_predictions.csv")
        station_metrics_map = load_optional_csv(TABLES_DIR / "station_model_metrics.csv")
        if station_predictions_map.empty:
            st.warning("未找到站点预测结果，请先运行 pipeline 生成 `models/station_predictions.csv`。")
        else:
            station_predictions_map["timestamp"] = pd.to_datetime(station_predictions_map["timestamp"])
            prediction_weather = add_weather_scenario(
                station_predictions_map.merge(
                    hourly_df[["timestamp", *weather_columns]],
                    on="timestamp",
                    how="left",
                )
            )
            prediction_models = [
                column
                for column in prediction_weather.columns
                if column
                not in {
                    "timestamp",
                    "station_name",
                    "actual",
                    "temperature_2m",
                    "precipitation",
                    "wind_speed_10m",
                    "relative_humidity_2m",
                    "weather_scenario",
                }
            ]
            if not prediction_models:
                st.warning("站点预测文件中没有可用于动画展示的模型列。")
            else:
                if station_metrics_map.empty:
                    default_model = prediction_models[0]
                else:
                    model_scores = station_metrics_map.groupby("model", as_index=False)["RMSE"].mean().sort_values("RMSE")
                    default_model = (
                        model_scores.iloc[0]["model"]
                        if not model_scores.empty and model_scores.iloc[0]["model"] in prediction_models
                        else prediction_models[0]
                    )

                scenario_options = weather_scenario_options()
                scenario_factors = weather_scenario_factors(hourly_df)
                controls = st.columns([1, 1, 1])
                prediction_model = controls[0].selectbox(
                    "站点预测模型",
                    prediction_models,
                    index=prediction_models.index(default_model),
                    key="prediction_animation_model",
                )
                horizon_hours = controls[1].slider(
                    "预测周期",
                    min_value=1,
                    max_value=168,
                    value=24,
                    format="未来 %d 小时",
                    key="prediction_animation_horizon",
                )
                selected_scenario = controls[2].selectbox("天气状态", scenario_options, key="prediction_weather")
                timestamps = sorted(prediction_weather["timestamp"].dropna().unique())
                timestamp_options = [pd.Timestamp(timestamp) for timestamp in timestamps]
                default_start = max(0, len(timestamp_options) - horizon_hours)
                start_time = datetime_picker(
                    "预测起点",
                    timestamp_options,
                    key="prediction_animation_start",
                    default_index=default_start,
                )
                end_time = start_time + pd.Timedelta(hours=horizon_hours)

                animation_view = prediction_weather[
                    (prediction_weather["timestamp"] >= start_time)
                    & (prediction_weather["timestamp"] < end_time)
                ].copy()
                animation_view = animation_view.rename(columns={prediction_model: "net_demand"})
                animation_view["net_demand"] = animation_view["net_demand"] * scenario_factors[selected_scenario]
                animation_view = add_map_fields(animation_view, value_col="net_demand").merge(
                    station_locations,
                    on="station_name",
                    how="inner",
                )
                animation_view["frame_time"] = animation_view["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

                st.markdown(f"预测窗口：`{start_time:%Y-%m-%d %H:%M}` 至 `{end_time:%Y-%m-%d %H:%M}`")
                if selected_scenario != "全部":
                    st.caption(
                        f"天气情景 `{selected_scenario}` 使用历史全系统需求估计系数 "
                        f"{scenario_factors[selected_scenario]:.2f} 调整站点预测强度。"
                    )
                if animation_view.empty:
                    st.warning("当前预测起点和周期没有可展示的站点数据。")
                else:
                    st.plotly_chart(
                        station_map_figure(
                            animation_view,
                            f"{prediction_model} 站点净流入预测动画",
                            animation_frame="frame_time",
                        ),
                        width="stretch",
                    )
                    station_rank = (
                        animation_view.groupby("station_name", as_index=False)["net_inflow"]
                        .mean()
                        .sort_values("net_inflow", ascending=False)
                    )
                    st.dataframe(station_rank, width="stretch", hide_index=True)

with tab_compare:
    st.subheader("模型对比")
    st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}", "R2": "{:.3f}"}))
    metrics_long = metrics_df.reset_index(names="model").melt(id_vars="model", value_vars=["MAE", "RMSE", "MAPE"])
    st.plotly_chart(
        px.bar(metrics_long, x="model", y="value", color="variable", barmode="group", title="模型误差指标对比"),
        width="stretch",
    )
    st.caption("下方柱状图只展示误差指标 MAE、RMSE、MAPE；R2 是拟合优度指标，量纲和方向都不同，放在同一张误差柱状图里容易造成误读，因此保留在上方表格中查看。")
    lstm_info = training_info.get("lstm", {})
    if lstm_info:
        st.info(
            f"LSTM 使用过去 {lstm_info.get('lookback')} 小时窗口训练，"
            f"设备：{lstm_info.get('device')}，PyTorch：{lstm_info.get('torch_version')}。"
        )
    st.markdown("特征解释重点：小时周期、上一天同小时需求、滚动均值与天气变量共同影响预测结果。")

with tab_detail:
    st.subheader("站点详情")
    st.caption("下表展示了各站点的总体历史数据。")
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
        st.subheader("站点历史热力图")
        st.caption("底图使用 OpenStreetMap；颜色按“净流入 = dropoff - pickup”显示，红色为净流入更多，蓝色为净流出更多。")
        station_locations = load_station_locations(tuple(sorted(station_df["station_name"].dropna().unique())))
        missing_locations = sorted(set(station_df["station_name"].unique()) - set(station_locations["station_name"].unique()))
        if station_locations.empty:
            st.warning("未找到站点经纬度。请确认 raw 骑行 zip 文件存在，或重新运行 pipeline 生成带坐标的站点数据。")
        else:
            if missing_locations:
                st.info(f"有 {len(missing_locations)} 个站点缺少坐标，地图会先展示可定位站点。")

            weather_columns = [
                column
                for column in ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]
                if column in hourly_df.columns
            ]
            station_weather = station_df.merge(
                hourly_df[["timestamp", *weather_columns]],
                on="timestamp",
                how="left",
            )
            station_weather = add_weather_scenario(station_weather)
            heatmap_times = sorted(station_weather["timestamp"].dropna().unique())
            selected_time = datetime_picker(
                "选择日期时间",
                [pd.Timestamp(timestamp) for timestamp in heatmap_times],
                key="dispatch_map_datetime",
                default_index=max(len(heatmap_times) - 1, 0),
            )

            map_view = station_weather[
                station_weather["timestamp"] == selected_time
            ].copy()
            map_view = add_map_fields(map_view).merge(station_locations, on="station_name", how="inner")

            if map_view.empty:
                st.warning("当前日期时间没有可展示的站点数据。")
            else:
                summary_cols = st.columns(4)
                summary_cols[0].metric("地图站点", f"{map_view['station_name'].nunique():,}")
                summary_cols[1].metric("总 pickup", f"{map_view['pickup_count'].sum():.0f}")
                summary_cols[2].metric("总 dropoff", f"{map_view['dropoff_count'].sum():.0f}")
                summary_cols[3].metric("净流入合计", f"{map_view['net_inflow'].sum():.0f}")
                st.plotly_chart(
                    station_map_figure(
                        map_view,
                        f"{selected_time:%Y-%m-%d %H:00} 站点净流入热力图",
                    ),
                    width="stretch",
                )
                st.dataframe(
                    map_view.sort_values("net_inflow", ascending=False)[
                        ["station_name", "pickup_count", "dropoff_count", "net_inflow", "flow_label", "weather_scenario"]
                    ],
                    width="stretch",
                    hide_index=True,
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
            st.subheader("模型预测与真实值对比")
            station_compare_controls = st.columns([1, 1, 1])
            prediction_station = station_compare_controls[0].selectbox(
                "选择站点",
                sorted(station_predictions["station_name"].unique()),
                index=0,
                key="dispatch_station_prediction_station",
            )
            station_model_options = ["None", *station_model_columns]
            station_model = station_compare_controls[1].selectbox(
                "选择预测模型",
                station_model_options,
                index=station_model_options.index(default_station_model),
                key="dispatch_station_prediction_model",
            )
            station_horizon_hours = station_compare_controls[2].slider(
                "预测周期",
                min_value=1,
                max_value=168,
                value=24 * 14,
                format="%d 小时",
                key="dispatch_station_prediction_horizon",
            )
            station_prediction_times = sorted(station_predictions["timestamp"].dropna().unique())
            station_start_time = datetime_picker(
                "预测起点（日期时间）",
                [pd.Timestamp(timestamp) for timestamp in station_prediction_times],
                key="dispatch_station_prediction_start",
                default_index=max(0, len(station_prediction_times) - station_horizon_hours),
            )
            station_end_time = station_start_time + pd.Timedelta(hours=station_horizon_hours)
            pred_view = station_predictions[
                (station_predictions["station_name"] == prediction_station)
                & (station_predictions["timestamp"] >= station_start_time)
                & (station_predictions["timestamp"] < station_end_time)
            ].copy()
            fig_station_pred = go.Figure()
            fig_station_pred.add_trace(
                go.Scatter(x=pred_view["timestamp"], y=pred_view["actual"], name="真实净需求", mode="lines")
            )
            if station_model != "None":
                fig_station_pred.add_trace(
                    go.Scatter(x=pred_view["timestamp"], y=pred_view[station_model], name="模型预测的净需求", mode="lines")
                )
            fig_station_pred.update_layout(
                title=f"{prediction_station} 站点净需求（{station_start_time:%Y-%m-%d %H:00} 起 {station_horizon_hours} 小时）",
                xaxis_title="时间",
                yaxis_title="net demand",
            )
            if pred_view.empty:
                st.warning("当前站点、预测起点和预测周期没有可展示的数据。")
            else:
                st.plotly_chart(fig_station_pred, width="stretch")
            metric_view = station_metrics[station_metrics["station_name"] == prediction_station].sort_values("RMSE")
            if station_model != "None":
                metric_view = metric_view[metric_view["model"] == station_model]
            st.dataframe(metric_view)

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
