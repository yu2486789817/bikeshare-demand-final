from __future__ import annotations

import pandas as pd

from bikeshare.data import aggregate_city_hourly, aggregate_station_hourly, clean_trips


def test_clean_trips_accepts_mixed_timestamp_precision() -> None:
    trips = pd.DataFrame(
        {
            "ride_id": ["a", "b"],
            "rideable_type": ["classic_bike", "electric_bike"],
            "started_at": ["2024-06-01 08:00:00", "2024-06-01 09:30:00.123"],
            "ended_at": ["2024-06-01 08:12:00", "2024-06-01 09:45:00.456"],
            "start_station_name": ["S1", "S2"],
            "end_station_name": ["S2", "S3"],
            "member_casual": ["member", "casual"],
        }
    )

    cleaned = clean_trips(trips)

    assert len(cleaned) == 2
    assert cleaned["started_hour"].tolist() == [pd.Timestamp("2024-06-01 08:00:00"), pd.Timestamp("2024-06-01 09:00:00")]


def test_aggregate_city_hourly_fills_missing_hours() -> None:
    trips = pd.DataFrame(
        {
            "ride_id": ["a", "b"],
            "rideable_type": ["classic_bike", "classic_bike"],
            "started_at": pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 10:00:00"]),
            "ended_at": pd.to_datetime(["2024-01-01 08:10:00", "2024-01-01 10:10:00"]),
            "started_hour": pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 10:00:00"]),
            "duration_min": [10.0, 10.0],
            "start_station_name": ["S1", "S2"],
            "end_station_name": ["S2", "S3"],
            "member_casual": ["member", "member"],
        }
    )

    hourly = aggregate_city_hourly(trips)

    assert len(hourly) == 3
    assert hourly.loc[hourly["timestamp"] == pd.Timestamp("2024-01-01 09:00:00"), "cnt"].item() == 0


def test_aggregate_station_hourly_excludes_unknown_station_from_top_stations() -> None:
    trips = pd.DataFrame(
        {
            "ride_id": ["a", "b", "c", "d"],
            "started_hour": pd.to_datetime(
                ["2024-01-01 08:00:00", "2024-01-01 08:00:00", "2024-01-01 09:00:00", "2024-01-01 09:00:00"]
            ),
            "ended_at": pd.to_datetime(
                ["2024-01-01 08:10:00", "2024-01-01 08:15:00", "2024-01-01 09:10:00", "2024-01-01 09:15:00"]
            ),
            "start_station_name": ["Unknown station", "Unknown station", "A", "B"],
            "end_station_name": ["Unknown station", "A", "B", "Unknown station"],
        }
    )

    station = aggregate_station_hourly(trips, top_n=2)

    assert "Unknown station" not in set(station["station_name"])
    assert set(station["station_name"]) == {"A", "B"}
