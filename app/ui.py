"""Shared Streamlit helpers: page setup, cached loaders, charts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, data_loader  # noqa: E402
from src import forecast as fc  # noqa: E402

OBSERVED, FORECAST, MUTED = "#4cc9f0", "#f4a261", "#8795ad"
ATTRIBUTION = ("Air quality: CPCB city-day data via Kaggle (rohanrao/air-quality-data-in-india). "
               "Weather: Open-Meteo historical archive. Educational forecasting tool, not an official measurement or health warning.")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stMetric"] {background:#121e33; border:1px solid #1f2f4d; border-radius:6px; padding:14px 16px;}
[data-testid="stMetricLabel"] {color:#8795ad;}
h1, h2, h3 {letter-spacing:-0.01em;}
.note {color:#8795ad; font-size:0.85rem;}
</style>
"""


def setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} | AirCast AI", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- loaders
@st.cache_data(show_spinner="Loading observations...")
def _load_data(path: str, mtime: float) -> pd.DataFrame:
    return data_loader.load_processed(Path(path))


def get_data() -> pd.DataFrame:
    p = config.get_paths().processed_csv
    if not p.exists():
        st.error(f"Processed data not found at `{p}`. Copy `city_day_clean.csv` from the Colab artifacts into data/processed/ "
                 "or run `python scripts/preprocess_data.py`.")
        st.stop()
    try:
        return _load_data(str(p), p.stat().st_mtime)
    except (data_loader.SchemaError, ValueError) as exc:
        st.error(f"Could not read processed data: {exc}")
        st.stop()


@st.cache_resource(show_spinner="Loading model...")
def _load_artifacts(models_dir: str, stamp: float) -> fc.Artifacts:
    return fc.load_artifacts(Path(models_dir))


def get_artifacts(required: bool = True) -> fc.Artifacts | None:
    d = config.get_paths().models
    try:
        stamp = max((f.stat().st_mtime for f in d.glob("*") if f.is_file()), default=0.0)
        return _load_artifacts(str(d), stamp)
    except fc.ModelNotFoundError as exc:
        if required:
            st.error(f"{exc}")
            st.stop()
        return None
    except Exception as exc:  # noqa: BLE001 - version mismatch etc.
        if required:
            st.error(f"Model artifacts could not be loaded: {exc}. Check that scikit-learn / xgboost / lightgbm versions "
                     "match `models/model_metadata.json` (versions).")
            st.stop()
        return None


@st.cache_data
def _weather(city: str, raw_dir: str, mtime: float):
    return data_loader.load_weather(city, Path(raw_dir))


def get_weather(city: str):
    d = config.get_paths().data_raw
    p = data_loader.weather_path(city, d)
    return _weather(city, str(d), p.stat().st_mtime if p.exists() else 0.0)


@st.cache_data(show_spinner="Running backtest...")
def backtest(city: str, horizon: int, stamp: str, _data, _art, _weather) -> pd.DataFrame:
    return fc.backtest_city(city, _data, _art, _weather, horizon)


def select_city(data: pd.DataFrame, art: fc.Artifacts) -> str:
    """Sidebar city selector. Only cities with both data and a trained model are offered."""
    cities = sorted(set(art.cities) & set(data["city"].unique()))
    if not cities:
        st.error("No city has both observations and a trained model.")
        st.stop()
    prev = st.session_state.get("selected_city")
    idx = cities.index(prev) if prev in cities else (cities.index("Delhi") if "Delhi" in cities else 0)
    city = st.sidebar.selectbox("City", cities, index=idx)
    st.session_state["selected_city"] = city
    return city


def city_history(data: pd.DataFrame, city: str, column: str = "pm25") -> pd.DataFrame:
    g = data[data["city"] == city].dropna(subset=[column])
    if g.empty:
        st.warning(f"No {column} observations for {city}.")
        st.stop()
    return g


def freshness(last_obs: pd.Timestamp) -> tuple[str, int]:
    days = (pd.Timestamp.now().normalize() - pd.Timestamp(last_obs)).days
    return ("Live / recent" if days <= 3 else "Historical dataset (not live)"), days


# ---------------------------------------------------------------- charts
def base_layout(fig: go.Figure, title: str, ytitle: str = "PM2.5 (µg/m³)", height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", title=title, height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", yaxis_title=ytitle, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.1, x=0), hovermode="x unified")
    fig.update_xaxes(gridcolor="#1c2a45")
    fig.update_yaxes(gridcolor="#1c2a45")
    return fig


def history_forecast_chart(hist: pd.DataFrame, fcast: pd.DataFrame, days: int = 120, threshold: float | None = None) -> go.Figure:
    h = hist.tail(days)
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["pm25"], name="Observed", mode="lines", line=dict(color=OBSERVED, width=2))
    origin = pd.Timestamp(fcast["origin_date"].iloc[0])
    last = hist[hist["date"] == origin]["pm25"]
    xs = [origin] + list(pd.to_datetime(fcast["forecast_date"]))
    ys = ([float(last.iloc[0])] if len(last) else [fcast["pm25_pred"].iloc[0]]) + list(fcast["pm25_pred"])
    fig.add_scatter(x=xs, y=ys, name="Forecast", mode="lines+markers", line=dict(color=FORECAST, width=2, dash="dash"))
    if threshold:
        fig.add_hline(y=threshold, line=dict(color=MUTED, dash="dot"), annotation_text=f"Threshold {threshold:g}")
    return base_layout(fig, "Observed (solid) and forecast (dashed)")


def footer() -> None:
    st.divider()
    st.markdown(f"<p class='note'>{ATTRIBUTION}</p>", unsafe_allow_html=True)
