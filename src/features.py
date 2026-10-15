"""Leakage-safe feature engineering for the direct multi-horizon strategy.

For a forecast origin t every feature uses only data observed on or before day t.
Calendar features describe the *target* date (known in advance), never future values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 7, 14, 30]
WX_COLS = ["temp", "rh", "wind", "precip", "pressure"]
WX_FEATURES = [f"{c}_t" for c in WX_COLS] + ["temp_roll3", "rh_roll3", "precip_roll3"]
CALENDAR_FEATURES = ["month_sin", "month_cos", "dow_sin", "dow_cos", "dayofmonth", "quarter"]
NON_FEATURE_COLS = ["y", "origin", "target_date", "city", "seasonal_naive"]


class FeatureError(ValueError):
    """Raised when a frame does not match the saved feature schema."""


def city_origin_features(g: pd.DataFrame, city: str, weather: pd.DataFrame | None = None):
    """Features at every origin day for one city. Returns (features, daily_series)."""
    s = g.set_index("date")["pm25"].asfreq("D")  # calendar grid; gaps stay NaN
    f = pd.DataFrame(index=s.index)
    f["pm25_t"] = s
    for k in LAGS:
        f[f"lag_{k}"] = s.shift(k)
    f["roll7_mean"] = s.rolling(7, min_periods=4).mean()
    f["roll14_mean"] = s.rolling(14, min_periods=7).mean()
    f["roll7_std"] = s.rolling(7, min_periods=4).std()
    f["roll7_min"] = s.rolling(7, min_periods=4).min()
    f["roll7_max"] = s.rolling(7, min_periods=4).max()
    f["trend_7"] = f["pm25_t"] - f["roll7_mean"]
    if weather is not None:
        w = weather.set_index("date").reindex(f.index)
        for c in WX_COLS:
            f[f"{c}_t"] = w[c]
        f["temp_roll3"] = w["temp"].rolling(3, min_periods=1).mean()
        f["rh_roll3"] = w["rh"].rolling(3, min_periods=1).mean()
        f["precip_roll3"] = w["precip"].rolling(3, min_periods=1).sum()
    else:
        for c in WX_FEATURES:
            f[c] = np.nan
    f["city"] = city
    return f, s


def calendar_features(target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * target_dates.month / 12),
            "month_cos": np.cos(2 * np.pi * target_dates.month / 12),
            "dow_sin": np.sin(2 * np.pi * target_dates.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * target_dates.dayofweek / 7),
            "dayofmonth": target_dates.day,
            "quarter": target_dates.quarter,
        },
        index=target_dates,
    )


def build_city_horizon_frame(g: pd.DataFrame, city: str, weather: pd.DataFrame | None, h: int) -> pd.DataFrame:
    """Training/backtest rows for one city and horizon h (target = PM2.5 at origin + h days)."""
    f, s = city_origin_features(g, city, weather)
    f = f.copy()
    f["y"] = s.shift(-h)
    tdate = f.index + pd.Timedelta(days=h)
    cal = calendar_features(tdate)
    for c in CALENDAR_FEATURES:
        f[c] = cal[c].to_numpy()
    f["seasonal_naive"] = s.shift(7 - h)  # same weekday last week; observed at origin since h <= 7
    f["origin"] = f.index
    f["target_date"] = tdate
    f = f.reset_index(drop=True)
    return f.dropna(subset=["pm25_t", "y"])  # origin must be observed; targets are never fabricated


def build_horizon_frame(df: pd.DataFrame, weather: dict[str, pd.DataFrame] | None, h: int) -> pd.DataFrame:
    weather = weather or {}
    parts = [build_city_horizon_frame(g, city, weather.get(city), h) for city, g in df.groupby("city")]
    return pd.concat(parts, ignore_index=True)


def base_feature_names(cities_unused: list[str] | None = None) -> list[str]:
    """Names of non-city-dummy features, in a fixed order."""
    names = ["pm25_t"] + [f"lag_{k}" for k in LAGS]
    names += ["roll7_mean", "roll14_mean", "roll7_std", "roll7_min", "roll7_max", "trend_7"]
    return names + WX_FEATURES + CALENDAR_FEATURES


def make_design(frame: pd.DataFrame, features: list[str], base_features: list[str], cities: list[str]) -> pd.DataFrame:
    """Build the model matrix with exactly the saved feature names and order."""
    missing = [c for c in base_features if c not in frame.columns]
    if missing:
        raise FeatureError(f"Frame is missing feature columns: {missing}")
    dummies = pd.DataFrame({f"city_{c}": (frame["city"] == c).astype(int) for c in cities}, index=frame.index)
    X = pd.concat([frame[base_features], dummies], axis=1)
    absent = [f for f in features if f not in X.columns]
    if absent:
        raise FeatureError(f"Schema features not producible: {absent}")
    return X[features]


def compute_split_dates(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.85):
    """Chronological split boundaries on origin date (70 / 15 / 15)."""
    dates = pd.Series(sorted(df["date"].unique()))
    return dates.iloc[int(len(dates) * train_frac)], dates.iloc[int(len(dates) * val_frac)]


def split_frame(frame: pd.DataFrame, val_start, test_start, gap_days: int = 7):
    """Train / validation / test with a purge gap so targets never cross a boundary."""
    gap = pd.Timedelta(days=gap_days)
    o = frame["origin"]
    return (
        frame[o < val_start - gap],
        frame[(o >= val_start) & (o < test_start - gap)],
        frame[o >= test_start],
    )
