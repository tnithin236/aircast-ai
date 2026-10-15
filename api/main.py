"""Optional FastAPI service. Run: uvicorn api.main:app --reload"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src import alerts as al  # noqa: E402
from src import config, data_loader  # noqa: E402
from src import forecast as fc  # noqa: E402

app = FastAPI(title="AirCast AI API", version="1.0",
              description="PM2.5 forecasts from historical data. Not an official warning service.")
_CACHE: dict = {}


def get_context() -> tuple[pd.DataFrame, fc.Artifacts]:
    """Load data and model once. Raises ModelNotFoundError / FileNotFoundError when unavailable."""
    if "ctx" not in _CACHE:
        data = data_loader.load_processed()
        _CACHE["ctx"] = (data, fc.load_artifacts())
    return _CACHE["ctx"]


def _ctx_or_503():
    try:
        return get_context()
    except (fc.ModelNotFoundError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _forecast_or_http(city: str):
    data, art = _ctx_or_503()
    try:
        return data, art, fc.forecast_city(city, data, art, data_loader.load_weather(city))
    except fc.UnsupportedCityError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class ForecastRequest(BaseModel):
    location: str = Field(min_length=1, description="City name, see GET /locations")
    horizon: int = Field(7, ge=1, le=7, description="Number of forecast days (1-7)")


class AlertRequest(ForecastRequest):
    threshold: float = Field(60.0, gt=0, le=1000, description="PM2.5 threshold in µg/m³")


@app.get("/health")
def health():
    try:
        get_context()
        return {"status": "ok", "model_loaded": True}
    except (fc.ModelNotFoundError, FileNotFoundError) as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.get("/locations")
def locations():
    data, art = _ctx_or_503()
    return {"locations": sorted(set(art.cities) & set(data["city"].unique()))}


@app.get("/model-info")
def model_info():
    _, art = _ctx_or_503()
    return {k: v for k, v in art.metadata.items()}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    _, art, res = _forecast_or_http(req.location)
    f = res.forecast[res.forecast["horizon"] <= req.horizon]
    return {
        "location": req.location, "unit": config.TARGET_UNIT, "model_version": art.metadata.get("trained_at"),
        "model": art.model_name, "origin_date": str(res.origin_date.date()),
        "generated_at": f["generated_at"].iloc[0], "warnings": res.warnings,
        "forecast": [dict(forecast_date=str(r.forecast_date), horizon=int(r.horizon), pm25_pred=round(r.pm25_pred, 2))
                     for r in f.itertuples()],
    }


@app.post("/alerts")
def alerts(req: AlertRequest):
    data, art, res = _forecast_or_http(req.location)
    f = res.forecast[res.forecast["horizon"] <= req.horizon]
    last = data[(data["city"] == req.location) & data["pm25"].notna()]["date"].max()
    tm = art.metadata.get("test_metrics") or {}
    out = al.evaluate_alerts(f, al.AlertConfig(threshold=req.threshold), last,
                             quality={"beats_baseline": tm.get("beats_baseline_on_MAE")})
    return {"location": req.location, "alerts": [a.to_dict() for a in out], "warnings": res.warnings}
