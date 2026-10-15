import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import alerts as al  # noqa: E402
from src import config  # noqa: E402
from src import forecast as fc  # noqa: E402

ui.setup("Alerts")
st.title("Forecast alerts")
st.caption("Alerts flag predicted PM2.5 above thresholds you choose. They are not official warnings, and nothing is sent outside this dashboard.")
data, art = ui.get_data(), ui.get_artifacts()
city = ui.select_city(data, art)

st.sidebar.subheader("Thresholds (µg/m³)")
cfg = al.AlertConfig(
    threshold=st.sidebar.number_input("Alert threshold", 10.0, 500.0, 60.0, 5.0),
    high_threshold=st.sidebar.number_input("High-severity threshold", 10.0, 500.0, 120.0, 5.0),
    peak_threshold=st.sidebar.number_input("Peak alert level", 10.0, 500.0, 90.0, 5.0),
    sustained_days=st.sidebar.slider("Sustained days", 2, 7, 3),
    stale_days=st.sidebar.slider("Stale after (days)", 1, 14, 3),
)
if cfg.high_threshold < cfg.threshold:
    st.sidebar.error("High-severity threshold should be at least the alert threshold.")

try:
    res = fc.forecast_city(city, data, art, ui.get_weather(city))
except (fc.UnsupportedCityError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
last = ui.city_history(data, city).iloc[-1]
tm = art.metadata.get("test_metrics") or {}
quality = {"beats_baseline": tm.get("beats_baseline_on_MAE")}
alerts = al.evaluate_alerts(res.forecast, cfg, last["date"], quality=quality)

st.subheader(f"Active alerts for {city}")
if alerts:
    df = pd.DataFrame([a.to_dict() for a in alerts])
    st.dataframe(df[["severity", "alert_type", "forecast_date", "predicted_pm25", "threshold", "message", "model", "forecast_generated_at"]],
                 use_container_width=True, hide_index=True)
else:
    st.success("No alerts at these thresholds.")

st.subheader("Forecast against CPCB PM2.5 categories")
cat = res.forecast[["forecast_date", "pm25_pred"]].copy()
cat["CPCB PM2.5 category"] = cat["pm25_pred"].map(al.cpcb_pm25_category)
st.dataframe(cat.rename(columns={"pm25_pred": "PM2.5 (µg/m³)"}).round(1), use_container_width=True, hide_index=True)
st.caption("Category of the PM2.5 sub-index only (CPCB breakpoints). It is not the overall AQI, which also needs other pollutants.")

st.subheader("Alert history")
hist = al.AlertHistory(config.get_paths().alert_history)
if st.button("Save current alerts to history") and alerts:
    try:
        st.success(f"Saved {hist.append_new(alerts)} new alert(s); identical alerts are skipped.")
    except OSError as exc:
        st.error(f"Could not write history: {exc}")
past = hist.read()
st.dataframe(past, use_container_width=True, hide_index=True) if len(past) else st.caption("No saved alerts yet.")
ui.footer()
