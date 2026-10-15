"""Threshold-based forecast alerts. Dashboard-only: nothing is sent to external services.

Alerts are decision-support flags on predicted PM2.5, not official warnings.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# CPCB National AQI sub-index categories for PM2.5 (24-hour average, µg/m³).
# Source: https://www.airquality.cpcb.gov.in/ccr_docs/About_AQI.pdf
# This labels the PM2.5 concentration only; it is NOT the overall AQI (which needs other pollutants).
CPCB_PM25_UPPER = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"), (120, "Poor"), (250, "Very Poor")]
CPCB_LAST = "Severe"
SEVERITY_ORDER = {"info": 0, "watch": 1, "warning": 2, "high": 3}


def cpcb_pm25_category(value: float) -> str:
    for upper, label in CPCB_PM25_UPPER:
        if value <= upper:
            return label
    return CPCB_LAST


@dataclass
class AlertConfig:
    threshold: float = 60.0          # default: 24-h PM2.5 standard of the CPCB "Satisfactory" upper bound
    high_threshold: float = 120.0    # severity "high" above this
    peak_threshold: float = 90.0     # peak alert level
    sustained_days: int = 3
    stale_days: int = 3
    quality_mae_limit: float | None = 30.0  # dashboard heuristic (µg/m³), configurable; not a validated standard


@dataclass
class Alert:
    alert_type: str
    severity: str
    location: str
    forecast_date: str
    predicted_pm25: float | None
    threshold: float | None
    message: str
    model: str
    forecast_generated_at: str
    alert_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def _id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def evaluate_alerts(forecast: pd.DataFrame, config: AlertConfig | None = None,
                    latest_obs_date: pd.Timestamp | None = None, today: pd.Timestamp | None = None,
                    quality: dict | None = None) -> list[Alert]:
    """Evaluate one city's forecast frame (columns from forecast_city). IDs are stable across refreshes."""
    cfg = config or AlertConfig()
    if forecast.empty:
        return []
    fc = forecast.sort_values("horizon").reset_index(drop=True)
    city, model = fc["city"].iloc[0], fc["model"].iloc[0]
    generated = str(fc["generated_at"].iloc[0])
    origin = fc["origin_date"].iloc[0]
    out: list[Alert] = []

    def add(kind, sev, date, val, thr, msg):
        out.append(Alert(kind, sev, city, str(date), None if val is None else round(float(val), 1), thr, msg,
                         model, generated, _id(city, origin, kind, date, thr)))

    for _, r in fc.iterrows():
        if r.pm25_pred >= cfg.threshold:
            sev = "high" if r.pm25_pred >= cfg.high_threshold else "watch"
            add("forecast_threshold_exceeded", sev, r.forecast_date, r.pm25_pred, cfg.threshold,
                f"Predicted PM2.5 {r.pm25_pred:.1f} µg/m³ is at or above {cfg.threshold:g} "
                f"(CPCB PM2.5 category: {cpcb_pm25_category(r.pm25_pred)}).")

    peak = fc.loc[fc["pm25_pred"].idxmax()]
    if peak.pm25_pred >= cfg.peak_threshold:
        add("predicted_peak", "warning", peak.forecast_date, peak.pm25_pred, cfg.peak_threshold,
            f"Forecast peak of {peak.pm25_pred:.1f} µg/m³ expected on {peak.forecast_date}.")

    run = []
    for r in [row for _, row in fc.iterrows()] + [None]:
        if r is not None and r.pm25_pred >= cfg.threshold:
            run.append(r)
            continue
        if len(run) >= cfg.sustained_days:
            add("sustained_exceedance", "warning", run[0].forecast_date, max(x.pm25_pred for x in run), cfg.threshold,
                f"Predicted PM2.5 stays at or above {cfg.threshold:g} for {len(run)} consecutive days "
                f"starting {run[0].forecast_date}.")
        run = []

    now = pd.Timestamp(today) if today is not None else pd.Timestamp.now().normalize()
    if latest_obs_date is not None:
        age = (now - pd.Timestamp(latest_obs_date)).days
        if age > cfg.stale_days:
            add("stale_observation", "info", pd.Timestamp(latest_obs_date).date(), None, None,
                f"Latest observation is {age} days old. Forecasts start from that date and are not live.")

    if quality:
        if quality.get("beats_baseline") is False:
            add("forecast_quality", "info", fc["forecast_date"].iloc[0], None, None,
                "On held-out test data the selected model did not beat the best baseline on MAE. Treat forecasts with caution.")
        mae = quality.get("horizon_mae") or {}
        if cfg.quality_mae_limit is not None:
            for _, r in fc.iterrows():
                m = mae.get(int(r.horizon))
                if m is not None and m > cfg.quality_mae_limit:
                    add("forecast_quality", "info", r.forecast_date, r.pm25_pred, cfg.quality_mae_limit,
                        f"Typical test error at horizon {int(r.horizon)} is {m:.1f} µg/m³ "
                        f"(above the {cfg.quality_mae_limit:g} reliability limit).")
    return sorted(out, key=lambda a: (-SEVERITY_ORDER[a.severity], a.forecast_date))


class AlertHistory:
    """Optional CSV persistence that skips alerts already stored (by alert_id)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def append_new(self, alerts: list[Alert]) -> int:
        existing = self.read()
        seen = set(existing["alert_id"]) if "alert_id" in existing else set()
        new = [a.to_dict() for a in alerts if a.alert_id not in seen]
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(self.path, index=False)
        return len(new)
