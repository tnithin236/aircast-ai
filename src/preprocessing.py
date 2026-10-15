"""Cleaning raw city_day.csv into the processed table used for modelling."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import RAW_REQUIRED, validate_schema

log = logging.getLogger(__name__)
RENAME = {"City": "city", "Date": "date", "PM2.5": "pm25", "PM10": "pm10"}


def clean_air_quality(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate, rename, parse and de-duplicate. Target values are never imputed.

    Policies (also reported):
      * unparseable dates -> row dropped
      * negative concentrations -> set to NaN (physically impossible)
      * duplicate city+date rows -> averaged
      * extreme high values are KEPT (they may be genuine pollution events)
    """
    validate_schema(raw, RAW_REQUIRED)
    df = raw[RAW_REQUIRED].rename(columns=RENAME).copy()
    report: dict = {"rows_in": len(df)}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    report["bad_dates_dropped"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    for c in ("pm25", "pm10"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        neg = df[c] < 0
        report[f"{c}_negative_to_nan"] = int(neg.sum())
        df.loc[neg, c] = np.nan

    report["duplicate_city_dates_averaged"] = int(df.duplicated(["city", "date"]).sum())
    df = df.groupby(["city", "date"], as_index=False)[["pm25", "pm10"]].mean()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    report["rows_out"] = len(df)
    for k, v in report.items():
        if v and k.endswith(("dropped", "nan", "averaged")):
            log.warning("%s: %s", k, v)
    return df, report


def detect_gaps(df: pd.DataFrame, column: str = "pm25") -> pd.DataFrame:
    """Per city: number of calendar days with no observed value and the longest gap."""
    rows = []
    for city, g in df.groupby("city"):
        s = g.set_index("date")[column].asfreq("D")
        missing = s.isna()
        run = (missing != missing.shift()).cumsum()
        longest = int(missing.groupby(run).sum().max()) if len(s) else 0
        rows.append(dict(city=city, calendar_days=len(s), missing_days=int(missing.sum()),
                         longest_gap_days=longest))
    return pd.DataFrame(rows)


def select_cities(df: pd.DataFrame, min_valid_days: int = 1000) -> list[str]:
    counts = df.groupby("city")["pm25"].count()
    return counts[counts >= min_valid_days].sort_values(ascending=False).index.tolist()


def preprocess_raw(raw_csv: str | Path, out_csv: str | Path, min_valid_days: int = 1000) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_csv)
    df, report = clean_air_quality(raw)
    cities = select_cities(df, min_valid_days)
    df = df[df["city"].isin(cities)].reset_index(drop=True)
    report.update(cities_kept=cities, min_valid_days=min_valid_days)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, report
