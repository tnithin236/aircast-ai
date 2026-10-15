"""Inference: load saved artifacts and produce 7-day forecasts. Never trains."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import (CALENDAR_FEATURES, build_city_horizon_frame, calendar_features,
                       city_origin_features, make_design)

log = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Model artifacts are missing or incomplete."""


class UnsupportedCityError(ValueError):
    """City has no data or no trained model."""


@dataclass
class Artifacts:
    models: dict
    schema: dict
    metadata: dict

    @property
    def cities(self) -> list[str]:
        return list(self.schema["cities"])

    @property
    def model_name(self) -> str:
        return self.metadata.get("model", "unknown")

    @property
    def uses_weather(self) -> bool:
        return self.metadata.get("weather", "none") != "none"

    def design(self, frame: pd.DataFrame) -> pd.DataFrame:
        return make_design(frame, self.schema["features"], self.schema["base_features"], self.cities)


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    features: dict[int, pd.DataFrame] = field(default_factory=dict)  # design row per horizon (for SHAP)
    origin_date: pd.Timestamp | None = None


def load_artifacts(models_dir: Path | None = None) -> Artifacts:
    """Load the model bundle written by training. Only load artifacts you produced yourself (joblib = pickle)."""
    d = Path(models_dir) if models_dir else config.get_paths().models
    files = [d / "pm25_models.joblib", d / "feature_schema.json", d / "model_metadata.json"]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise ModelNotFoundError(f"Missing model artifacts in {d}: {missing}. Copy them from Colab or run scripts/train_model.py.")
    models = joblib.load(files[0])
    schema = json.loads(files[1].read_text())
    metadata = json.loads(files[2].read_text())
    if not isinstance(models, dict) or not models:
        raise ModelNotFoundError("pm25_models.joblib must contain a dict of {horizon: model}.")
    return Artifacts({int(k): v for k, v in models.items()}, schema, metadata)


def forecast_city(city: str, data: pd.DataFrame, art: Artifacts, weather: pd.DataFrame | None = None,
                  as_of: pd.Timestamp | str | None = None, today: pd.Timestamp | None = None) -> ForecastResult:
    """Seven-day PM2.5 forecast from the last observed day (or the last observed day <= as_of).

    Only data up to the origin is used, so later rows cannot influence the result.
    Negative outputs are clipped to 0; both raw and clipped values are returned.
    """
    if city not in art.cities:
        raise UnsupportedCityError(f"No trained model for city '{city}'. Supported: {art.cities}")
    g = data[data["city"] == city]
    if as_of is not None:
        g = g[g["date"] <= pd.Timestamp(as_of)]
    if g["pm25"].notna().sum() == 0:
        raise UnsupportedCityError(f"No PM2.5 observations for city '{city}'.")
    f, s = city_origin_features(g, city, weather)
    origin = s.dropna().index.max()
    row = f.loc[[origin]]
    warnings: list[str] = []
    if art.uses_weather and (weather is None or row[["temp_t", "rh_t", "wind_t", "precip_t", "pressure_t"]].isna().all(axis=None)):
        warnings.append("Weather features are unavailable for this city and origin date although the model was "
                        "trained with weather. Run scripts/download_weather.py; forecasts may be degraded.")
    now = pd.Timestamp(today) if today is not None else pd.Timestamp.now().normalize()
    age = (now - origin).days
    if age > 3:
        warnings.append(f"Forecast origin is {origin.date()} ({age} days ago). This is a historical dataset, not live air quality.")

    rows, feats = [], {}
    generated = pd.Timestamp.now().isoformat(timespec="seconds")
    for h in sorted(art.models):
        tdate = origin + pd.Timedelta(days=h)
        r = row.copy()
        cal = calendar_features(pd.DatetimeIndex([tdate]))
        for c in CALENDAR_FEATURES:
            r[c] = cal[c].iloc[0]
        X = art.design(r)
        raw = float(art.models[h].predict(X)[0])
        feats[h] = X
        rows.append(dict(city=city, origin_date=origin.date(), forecast_date=tdate.date(), horizon=h,
                         pm25_pred=max(raw, 0.0), pm25_pred_raw=raw, clipped=raw < 0,
                         unit=config.TARGET_UNIT, model=art.model_name, generated_at=generated))
    return ForecastResult(pd.DataFrame(rows), warnings, feats, origin)


def backtest_city(city: str, data: pd.DataFrame, art: Artifacts, weather: pd.DataFrame | None, horizon: int) -> pd.DataFrame:
    """Predictions on the held-out test period (origins >= test start) using the saved models.

    The saved models were fit on origins before the test period, so these are out-of-sample.
    """
    if city not in art.cities:
        raise UnsupportedCityError(f"No trained model for city '{city}'.")
    if horizon not in art.models:
        raise ValueError(f"Horizon {horizon} not available. Have: {sorted(art.models)}")
    frame = build_city_horizon_frame(data[data["city"] == city], city, weather, horizon)
    test_start = art.metadata.get("train_range", [None, None])[1]
    if test_start:
        frame = frame[frame["origin"] >= pd.Timestamp(test_start)]
    if frame.empty:
        return pd.DataFrame(columns=["target_date", "y", "pred", "persistence"])
    pred = art.models[horizon].predict(art.design(frame))
    return pd.DataFrame(dict(origin=frame["origin"].to_numpy(), target_date=frame["target_date"].to_numpy(),
                             y=frame["y"].to_numpy(), pred=np.clip(pred, 0, None),
                             persistence=frame["pm25_t"].to_numpy()))
