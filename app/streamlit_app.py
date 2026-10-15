"""AirCast AI: Overview page (entry point). Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import alerts as al  # noqa: E402
from src import forecast as fc  # noqa: E402

ui.setup("Overview")
st.title("AirCast AI")
st.caption("Seven-day PM2.5 forecasts and threshold alerts for Indian cities. Educational decision-support, not an official warning.")

data, art = ui.get_data(), ui.get_artifacts()
city = ui.select_city(data, art)
hist = ui.city_history(data, city)
last = hist.iloc[-1]
status, age = ui.freshness(last["date"])

try:
    res = fc.forecast_city(city, data, art, ui.get_weather(city))
except (fc.UnsupportedCityError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
f = res.forecast
peak = f.loc[f["pm25_pred"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest observed PM2.5", f"{last['pm25']:.1f} µg/m³", help="Observed measurement from the dataset.")
c1.caption(f"Measured {last['date'].date()}")
c2.metric("Data status", status, f"{age} days old", delta_color="off")
c3.metric(f"Forecast for {f['forecast_date'].iloc[0]}", f"{f['pm25_pred'].iloc[0]:.1f} µg/m³", help="Predicted, not measured.")
c3.caption("Next day, model output")
c4.metric("Forecast peak", f"{peak['pm25_pred']:.1f} µg/m³")
c4.caption(f"Expected {peak['forecast_date']}")

if status != "Live / recent":
    st.info(f"The newest record for {city} is from {last['date'].date()}. Forecasts start from that date; this app is running in historical-data mode.")
for w in res.warnings:
    st.warning(w)

cfg = al.AlertConfig()
st.plotly_chart(ui.history_forecast_chart(hist, f, 120, cfg.threshold), use_container_width=True)

st.subheader("Upcoming alerts")
tm = art.metadata.get("test_metrics") or {}
quality = {"beats_baseline": tm.get("beats_baseline_on_MAE")}
alerts = al.evaluate_alerts(f, cfg, last["date"], quality=quality)
if alerts:
    st.dataframe(pd.DataFrame([a.to_dict() for a in alerts])[["severity", "alert_type", "forecast_date", "predicted_pm25", "threshold", "message"]],
                 use_container_width=True, hide_index=True)
else:
    st.success("No alerts at the default threshold. Adjust thresholds on the Alerts page.")
st.caption(f"Model: {art.model_name} | forecast generated {f['generated_at'].iloc[0]} | default threshold {cfg.threshold:g} µg/m³")
ui.footer()
