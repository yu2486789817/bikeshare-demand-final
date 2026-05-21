from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

DEFAULT_MONTHS = [f"{year}{month:02d}" for year in range(2023, 2026) for month in range(1, 13)]

CAPITAL_BIKESHARE_URL = (
    "https://s3.amazonaws.com/capitalbikeshare-data/{month}-capitalbikeshare-tripdata.zip"
)

DC_LATITUDE = 38.9072
DC_LONGITUDE = -77.0369
TIMEZONE = "America/New_York"

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]


def ensure_project_dirs() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
