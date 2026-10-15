import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import forecast as fc  # noqa: E402

ui.setup("Forecast")
st.title("Air quality forecast")
data, art = ui.get_data(), ui.get_artifacts()
city = ui.select_city(data, art)

pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5"], help="PM10 forecasting is not implemented; only PM2.5 models are trained.")
n_days = st.sidebar.slider("Forecast horizon (days)", 1, max(art.models), max(art.models))
hist = ui.city_history(data, city)
weather = ui.get_weather(city)

try:
    res = fc.forecast_city(city, data, art, weather)
except (fc.UnsupportedCityError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
for w in res.warnings:
    st.warning(w)
f = res.forecast[res.forecast["horizon"] <= n_days]

st.plotly_chart(ui.history_forecast_chart(hist, f, 90), use_container_width=True)
st.caption(f"Model: {art.model_name} | Strategy: direct, one model per horizon | Generated: {f['generated_at'].iloc[0]} | Unit: µg/m³")

st.subheader("Forecast values")
show = f[["forecast_date", "horizon", "pm25_pred", "pm25_pred_raw", "clipped"]].rename(
    columns={"pm25_pred": "PM2.5 forecast (µg/m³)", "pm25_pred_raw": "Raw model output", "clipped": "Clipped at 0"})
st.dataframe(show.round(2), use_container_width=True, hide_index=True)
st.download_button("Download forecast CSV", f.to_csv(index=False), file_name=f"aircast_forecast_{city}.csv", mime="text/csv")

st.subheader("Actual vs predicted on the held-out test period")
hz = st.selectbox("Backtest horizon (days ahead)", sorted(art.models), index=0)
bt = ui.backtest(city, hz, str(art.metadata.get("trained_at")), data, art, weather)
if bt.empty:
    st.info("No held-out test rows for this city.")
else:
    bt = bt.sort_values("target_date")
    fig = go.Figure()
    fig.add_scatter(x=bt["target_date"], y=bt["y"], name="Actual", line=dict(color=ui.OBSERVED))
    fig.add_scatter(x=bt["target_date"], y=bt["pred"], name=f"Predicted ({hz}-day ahead)", line=dict(color=ui.FORECAST))
    fig.add_scatter(x=bt["target_date"], y=bt["persistence"], name="Persistence baseline", line=dict(color=ui.MUTED, dash="dot"))
    st.plotly_chart(ui.base_layout(fig, f"{city}: test period, horizon {hz} day(s)"), use_container_width=True)
    m, b = ev.metrics(bt["y"], bt["pred"]), ev.metrics(bt["y"], bt["persistence"])
    c1, c2 = st.columns(2)
    c1.metric("Model MAE (this city)", f"{m['MAE']:.1f} µg/m³", f"{m['MAE'] - b['MAE']:+.1f} vs persistence", delta_color="inverse")
    c2.metric("Model RMSE (this city)", f"{m['RMSE']:.1f} µg/m³", f"{m['RMSE'] - b['RMSE']:+.1f} vs persistence", delta_color="inverse")
ui.footer()
