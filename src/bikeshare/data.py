from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .config import (
    CAPITAL_BIKESHARE_URL,
    DC_LATITUDE,
    DC_LONGITUDE,
    PROCESSED_DIR,
    RAW_DIR,
    TIMEZONE,
WEATHER_COLUMNS,
)

UNKNOWN_STATION_NAME = "Unknown station"


@dataclass(frozen=True)
class DataPaths:
    hourly_demand: Path = PROCESSED_DIR / "hourly_demand.csv"
    station_hourly: Path = PROCESSED_DIR / "station_hourly.csv"
    weather_hourly: Path = PROCESSED_DIR / "weather_hourly.csv"


def download_trip_zip(month: str, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{month}-capitalbikeshare-tripdata.zip"
    if target.exists() and target.stat().st_size > 0 and not force:
        try:
            with zipfile.ZipFile(target) as zf:
                if zf.testzip() is None:
                    return target
        except zipfile.BadZipFile:
            pass
    if target.exists():
        target.unlink()

    url = CAPITAL_BIKESHARE_URL.format(month=month)
    temp_target = target.with_suffix(".zip.part")
    for attempt in range(3):
        temp_target.unlink(missing_ok=True)
        try:
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temp_target.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            with zipfile.ZipFile(temp_target) as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"Corrupt file inside zip: {bad_file}")
            temp_target.replace(target)
            return target
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            temp_target.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"Failed to download valid trip zip for {month}: {url}") from exc
    return target


def _read_trip_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        ]
        if not csv_names:
            raise ValueError(f"No CSV file found in {zip_path}")
        with zf.open(csv_names[0]) as fh:
            return pd.read_csv(fh)


def load_trips(months: list[str], raw_dir: Path = RAW_DIR, force_download: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        print(f"Loading trip data for {month}...", flush=True)
        zip_path = download_trip_zip(month, raw_dir=raw_dir, force=force_download)
        frame = _read_trip_zip(zip_path)
        frame["source_month"] = month
        frames.append(frame)
    if not frames:
        raise ValueError("No trip months were provided")
    return pd.concat(frames, ignore_index=True)


def clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ride_id",
        "rideable_type",
        "started_at",
        "ended_at",
        "start_station_name",
        "end_station_name",
        "member_casual",
    }
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError(f"Trip data is missing columns: {sorted(missing)}")

    cleaned = trips.copy()
    cleaned["started_at"] = pd.to_datetime(cleaned["started_at"], errors="coerce", format="mixed")
    cleaned["ended_at"] = pd.to_datetime(cleaned["ended_at"], errors="coerce", format="mixed")
    cleaned = cleaned.dropna(subset=["started_at", "ended_at"])
    cleaned["duration_min"] = (cleaned["ended_at"] - cleaned["started_at"]).dt.total_seconds() / 60
    cleaned = cleaned[(cleaned["duration_min"] >= 1) & (cleaned["duration_min"] <= 24 * 60)]
    cleaned["started_hour"] = cleaned["started_at"].dt.floor("h")
    cleaned["start_station_name"] = cleaned["start_station_name"].fillna(UNKNOWN_STATION_NAME)
    cleaned["end_station_name"] = cleaned["end_station_name"].fillna(UNKNOWN_STATION_NAME)
    return cleaned


def aggregate_city_hourly(trips: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()
    city = (
        trips.groupby("started_hour")
        .agg(
            cnt=("ride_id", "count"),
            duration_mean_min=("duration_min", "mean"),
            unique_start_stations=("start_station_name", "nunique"),
        )
        .reset_index()
        .rename(columns={"started_hour": "timestamp"})
    )

    members = (
        pd.crosstab(trips["started_hour"], trips["member_casual"])
        .rename_axis("timestamp")
        .reset_index()
    )
    bikes = (
        pd.crosstab(trips["started_hour"], trips["rideable_type"])
        .rename_axis("timestamp")
        .reset_index()
    )
    hourly = city.merge(members, on="timestamp", how="left").merge(bikes, on="timestamp", how="left")

    full_index = pd.date_range(hourly["timestamp"].min(), hourly["timestamp"].max(), freq="h")
    hourly = hourly.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()
    count_cols = [c for c in hourly.columns if c not in {"timestamp", "duration_mean_min"}]
    hourly[count_cols] = hourly[count_cols].fillna(0)
    hourly["duration_mean_min"] = hourly["duration_mean_min"].interpolate().ffill().bfill()
    hourly["member_share"] = hourly.get("member", 0) / hourly["cnt"].clip(lower=1)
    hourly["casual_share"] = hourly.get("casual", 0) / hourly["cnt"].clip(lower=1)
    return hourly


def aggregate_station_hourly(trips: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    known_starts = trips.loc[trips["start_station_name"] != UNKNOWN_STATION_NAME, "start_station_name"]
    known_ends = trips.loc[trips["end_station_name"] != UNKNOWN_STATION_NAME, "end_station_name"]
    station_volume = pd.concat(
        [
            known_starts.rename("station_name"),
            known_ends.rename("station_name"),
        ],
        ignore_index=True,
    ).value_counts()
    top_stations = station_volume.head(top_n).index.tolist()
    filtered = trips[
        trips["start_station_name"].isin(top_stations) | trips["end_station_name"].isin(top_stations)
    ].copy()

    pickups = (
        filtered[filtered["start_station_name"].isin(top_stations)]
        .groupby(["started_hour", "start_station_name"])
        .size()
        .rename("pickup_count")
        .reset_index()
        .rename(columns={"started_hour": "timestamp", "start_station_name": "station_name"})
    )
    dropoffs = (
        filtered[filtered["end_station_name"].isin(top_stations)]
        .assign(ended_hour=lambda d: d["ended_at"].dt.floor("h"))
        .groupby(["ended_hour", "end_station_name"])
        .size()
        .rename("dropoff_count")
        .reset_index()
        .rename(columns={"ended_hour": "timestamp", "end_station_name": "station_name"})
    )
    station = pickups.merge(dropoffs, on=["timestamp", "station_name"], how="outer").fillna(0)
    station["net_demand"] = station["pickup_count"] - station["dropoff_count"]
    station["action"] = station["net_demand"].apply(
        lambda value: "补车" if value > 0 else ("清车" if value < 0 else "观测")
    )
    location_columns = {"start_lat", "start_lng", "end_lat", "end_lng"}
    if location_columns.issubset(trips.columns):
        start_locations = trips.loc[
            trips["start_station_name"].isin(top_stations),
            ["start_station_name", "start_lat", "start_lng"],
        ].rename(
            columns={
                "start_station_name": "station_name",
                "start_lat": "latitude",
                "start_lng": "longitude",
            }
        )
        end_locations = trips.loc[
            trips["end_station_name"].isin(top_stations),
            ["end_station_name", "end_lat", "end_lng"],
        ].rename(
            columns={
                "end_station_name": "station_name",
                "end_lat": "latitude",
                "end_lng": "longitude",
            }
        )
        locations = pd.concat([start_locations, end_locations], ignore_index=True).dropna()
        if not locations.empty:
            locations = locations.groupby("station_name", as_index=False).agg(
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
            )
            station = station.merge(locations, on="station_name", how="left")
    return station.sort_values(["timestamp", "station_name"]).reset_index(drop=True)


def fetch_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "latitude": DC_LATITUDE,
        "longitude": DC_LONGITUDE,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "hourly": ",".join(WEATHER_COLUMNS),
        "timezone": TIMEZONE,
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"Open-Meteo response did not include hourly data: {payload}")
    weather = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    return weather


def build_processed_data(months: list[str], force_download: bool = False) -> DataPaths:
    raw = load_trips(months, force_download=force_download)
    trips = clean_trips(raw)
    hourly = aggregate_city_hourly(trips)
    weather = fetch_weather(hourly["timestamp"].min(), hourly["timestamp"].max())
    merged = hourly.merge(weather, on="timestamp", how="left")
    weather_missing = merged[WEATHER_COLUMNS].isna().mean().max()
    if weather_missing > 0.05:
        raise ValueError(f"Weather missing rate is too high: {weather_missing:.1%}")
    merged[WEATHER_COLUMNS] = merged[WEATHER_COLUMNS].interpolate().ffill().bfill()
    station = aggregate_station_hourly(trips)

    paths = DataPaths()
    merged.to_csv(paths.hourly_demand, index=False)
    station.to_csv(paths.station_hourly, index=False)
    weather.to_csv(paths.weather_hourly, index=False)
    return paths


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python -m bikeshare.pipeline` first.")
    return pd.read_csv(path)
