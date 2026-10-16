"""AirCast AI: Dashboard page (entry point). Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import alerts as al  # noqa: E402
from src import forecast as fc  # noqa: E402

ui.setup("Dashboard")

data, art = ui.get_data(), ui.get_artifacts()
city = ui.select_city(data, art)
hist = ui.city_history(data, city)
last = hist.iloc[-1]
status, age = ui.freshness(last["date"])
live = status.startswith("Live")

try:
    res = fc.forecast_city(city, data, art, ui.get_weather(city))
except (fc.UnsupportedCityError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
f = res.forecast
peak = f.loc[f["pm25_pred"].idxmax()]
cfg = al.AlertConfig()
tm = art.metadata.get("test_metrics") or {}

# ------------------------------------------------------------------ hero
ui.hero(live, last["date"])

# ------------------------------------------------------------------ KPI cards
prev = hist["pm25"].iloc[-2] if len(hist) > 1 else None
if prev:
    delta = (last["pm25"] - prev) / prev * 100
    obs_chip = (f"{'▼' if delta < 0 else '▲'} {abs(delta):.0f}% vs previous reading", "down" if delta < 0 else "up")
else:
    obs_chip = ("", "neutral")

c1, c2, c3, c4 = st.columns(4, gap="medium")
ui.render(ui.kpi_card("blue", "wind", "Latest observed PM2.5", f"{last['pm25']:.1f}", "µg/m³", *obs_chip[:1], obs_chip[1],
                      spark=ui.spark_bars(hist["pm25"].tail(8), "#29b6ff")), c1)
ui.render(ui.kpi_card("purple", "calendar", "Data status", "Live data" if live else "Historical data", "",
                      f"{age:,} days old", "neutral", "clock",
                      spark=f'<span class="spark">{ui.icon("database", 42, "#8b7bff", 1.6)}</span>'), c2)
ui.render(ui.kpi_card("green", "trend", f"Forecast for {f['forecast_date'].iloc[0]}", f"{f['pm25_pred'].iloc[0]:.1f}", "µg/m³",
                      "Next day prediction", "neutral", spark=ui.spark_line(f["pm25_pred"], "#2ee6a8")), c3)
ui.render(ui.kpi_card("orange", "star", "Forecast peak", f"{peak['pm25_pred']:.1f}", "µg/m³",
                      f"Expected on {peak['forecast_date']}", "neutral", spark=ui.spark_bars(f["pm25_pred"], "#ff9f43")), c4)

if not live:
    st.info(f"The newest record for {city} is from {last['date'].date()}. Forecasts start from that date; "
            "this app is running in historical-data mode.")
for w in res.warnings:
    st.warning(w)

st.write("")


# ------------------------------------------------------------------ trend chart (fragment: range changes rerun only this block)
@st.fragment
def trend_panel(hist: pd.DataFrame, f: pd.DataFrame, threshold: float) -> None:
    with st.container(key="panel_trend"):
        left, right = st.columns([3, 2], vertical_alignment="center")
        ui.render(ui.panel_title("chart", "Air Quality Trend"), left)
        choice = right.radio("Range", ["7D", "30D", "90D", "All"], index=1, horizontal=True,
                             label_visibility="collapsed", key="trend_range")
        days = {"7D": 7, "30D": 30, "90D": 90, "All": len(hist)}[choice]
        st.plotly_chart(ui.history_forecast_chart(hist, f, days, threshold, title=None), use_container_width=True)


trend_panel(hist, f, cfg.threshold)
st.write("")

# ------------------------------------------------------------------ model performance + insights
left, right = st.columns([1.15, 1], gap="medium")

m = ui.perf_metrics(tm)
beats = tm.get("beats_baseline_on_MAE")
mae_text, mae_kind = ui.mae_chip(m, beats)
fmt = lambda v: f"{v:.2f}" if v is not None else "n/a"  # noqa: E731

with left.container(key="panel_perf"):
    ui.render(ui.panel_title("cpu", "Model Performance"))
    p1, p2, p3 = st.columns(3)
    ui.render(ui.mini_stat(f"{art.model_name} MAE", fmt(m["mae"]), "µg/m³" if m["mae"] is not None else "", mae_text, mae_kind), p1)
    ui.render(ui.mini_stat("RMSE", fmt(m["rmse"]), "µg/m³" if m["rmse"] is not None else "", "Test set", "neutral"), p2)
    ui.render(ui.mini_stat("R² score", f"{m['r2']:.3f}" if m["r2"] is not None else "n/a", "", "",
                           visual=ui.ring(m["r2"]) if m["r2"] is not None else ""), p3)

first = f.iloc[0]
band, band_color = ui.aqi_band(first["pm25_pred"])
over = int((f["pm25_pred"] > cfg.threshold).sum())
items = [
    ("ok", f"Forecast for {first['forecast_date']} is {first['pm25_pred']:.1f} µg/m³, in the {band.lower()} band of India's CPCB scale."),
    ("ok", f"No forecast day exceeds the {cfg.threshold:g} µg/m³ alert threshold.") if over == 0
    else ("warn", f"{over} of {len(f)} forecast days exceed the {cfg.threshold:g} µg/m³ alert threshold."),
]
if beats is False:
    items.append(("warn", "The model does not beat the baseline on MAE, so treat these forecasts with caution."))

with right.container(key="panel_insights"):
    ui.render(ui.panel_title("bulb", "Forecast Insights"))
    ui.render(ui.insights(items, band, band_color))

st.write("")

# ------------------------------------------------------------------ alerts
with st.container(key="panel_alerts"):
    ui.render(ui.panel_title("bell", "Upcoming alerts"))
    quality = {"beats_baseline": beats}
    alerts = al.evaluate_alerts(f, cfg, last["date"], quality=quality)
    if alerts:
        st.dataframe(pd.DataFrame([a.to_dict() for a in alerts])[
                         ["severity", "alert_type", "forecast_date", "predicted_pm25", "threshold", "message"]],
                     use_container_width=True, hide_index=True)
    else:
        st.success("No alerts at the default threshold. Adjust thresholds on the Alerts page.")
    st.caption(f"Model: {art.model_name} | forecast generated {f['generated_at'].iloc[0]} | default threshold {cfg.threshold:g} µg/m³")

ui.footer()