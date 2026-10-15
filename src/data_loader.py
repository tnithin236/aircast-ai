"""Loading and validating data: air-quality CSVs, cached weather, Open-Meteo fetch."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

RAW_REQUIRED = ["City", "Date", "PM2.5", "PM10"]
PROCESSED_REQUIRED = ["city", "date", "pm25", "pm10"]


class SchemaError(ValueError):
    """Raised when a file does not have the columns the pipeline needs."""


def validate_schema(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` or place city_day.csv in data/raw/."
        )
    df = pd.read_csv(path)
    validate_schema(df, RAW_REQUIRED)
    return df


def load_processed(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned city-day table written by preprocessing (or by the Colab notebook)."""
    path = Path(path) if path else config.get_paths().processed_csv
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}.")
    df = pd.read_csv(path)
    validate_schema(df, PROCESSED_REQUIRED)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ("pm25", "pm10"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["city", "date"]).reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summary of coverage and missingness per city (observed days vs calendar span)."""
    rows = []
    for city, g in df.groupby("city"):
        span = (g["date"].max() - g["date"].min()).days + 1
        rows.append(
            dict(
                city=city,
                first_date=g["date"].min().date().isoformat(),
                last_date=g["date"].max().date().isoformat(),
                calendar_days=span,
                pm25_observed=int(g["pm25"].notna().sum()),
                pm25_missing_pct=round(100 * (1 - g["pm25"].notna().sum() / span), 1),
                pm10_observed=int(g["pm10"].notna().sum()),
            )
        )
    return {"per_city": pd.DataFrame(rows), "rows": len(df), "cities": df["city"].nunique()}


# ---------------------------------------------------------------- weather
# Approximate city-centre coordinates (documented assumption; not station locations).
COORDS: dict[str, tuple[float, float]] = {
    "Ahmedabad": (23.03, 72.58), "Aizawl": (23.73, 92.72), "Amaravati": (16.57, 80.36),
    "Amritsar": (31.63, 74.87), "Bengaluru": (12.97, 77.59), "Bhopal": (23.26, 77.41),
    "Brajrajnagar": (21.82, 83.92), "Chandigarh": (30.73, 76.78), "Chennai": (13.08, 80.27),
    "Coimbatore": (11.02, 76.96), "Delhi": (28.61, 77.21), "Ernakulam": (9.98, 76.28),
    "Gurugram": (28.46, 77.03), "Guwahati": (26.14, 91.74), "Hyderabad": (17.39, 78.49),
    "Jaipur": (26.91, 75.79), "Jorapokhar": (23.71, 86.41), "Kochi": (9.93, 76.27),
    "Kolkata": (22.57, 88.36), "Lucknow": (26.85, 80.95), "Mumbai": (19.08, 72.88),
    "Patna": (25.59, 85.14), "Shillong": (25.58, 91.89), "Talcher": (20.95, 85.22),
    "Thiruvananthapuram": (8.52, 76.94), "Visakhapatnam": (17.69, 83.22),
}
WX_VARS = {
    "temperature_2m_mean": "temp",
    "relative_humidity_2m_mean": "rh",
    "wind_speed_10m_max": "wind",
    "precipitation_sum": "precip",
    "surface_pressure_mean": "pressure",
}
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_TZ = "Asia/Kolkata"


def weather_path(city: str, raw_dir: Path | None = None) -> Path:
    return (raw_dir or config.get_paths().data_raw) / f"weather_{city}.csv"


def load_weather(city: str, raw_dir: Path | None = None) -> pd.DataFrame | None:
    """Return cached weather for a city, or None when it has not been downloaded."""
    p = weather_path(city, raw_dir)
    if not p.exists():
        return None
    w = pd.read_csv(p, parse_dates=["date"])
    missing = [c for c in WX_VARS.values() if c not in w.columns]
    if missing:
        log.warning("Weather cache for %s lacks columns %s; ignoring it", city, missing)
        return None
    return w


def load_weather_many(cities: list[str], raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for c in cities:
        w = load_weather(c, raw_dir)
        if w is not None:
            out[c] = w
    return out


def fetch_weather(city: str, start: str, end: str, raw_dir: Path | None = None,
                  retries: int = 3, timeout: int = 30) -> pd.DataFrame | None:
    """Download daily weather from Open-Meteo (no API key) and cache it. Returns None on failure."""
    import requests

    cached = load_weather(city, raw_dir)
    if cached is not None:
        return cached
    if city not in COORDS:
        log.warning("No coordinates for %s; skipping weather", city)
        return None
    lat, lon = COORDS[city]
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                  daily=",".join(WX_VARS), timezone=WEATHER_TZ)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(WEATHER_URL, params=params, timeout=timeout)
            r.raise_for_status()
            daily = r.json().get("daily")
            if not daily or "time" not in daily or any(v not in daily for v in WX_VARS):
                raise ValueError(f"Unexpected response keys: {list((daily or {}).keys())}")
            w = pd.DataFrame(daily).rename(columns={"time": "date", **WX_VARS})
            w["date"] = pd.to_datetime(w["date"])
            p = weather_path(city, raw_dir)
            p.parent.mkdir(parents=True, exist_ok=True)
            w.to_csv(p, index=False)
            return w
        except Exception as exc:  # noqa: BLE001 - network/JSON/validation errors all retried
            log.warning("Weather %s attempt %d/%d failed: %s", city, attempt, retries, exc)
            time.sleep(2 * attempt)
    return None
