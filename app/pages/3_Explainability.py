import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import config  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import explainability as xai  # noqa: E402
from src import forecast as fc  # noqa: E402

ui.setup("Explainability")
st.title("Explainability (SHAP)")
st.info("SHAP shows how much each input moved the model's output relative to its average output. "
        "It describes the model, not the atmosphere: contributions do not establish causality.")
data, art = ui.get_data(), ui.get_artifacts()
city = ui.select_city(data, art)

st.subheader("Global feature importance (day-1 model, test sample)")
imp = ev.load_reports()["shap_importance"]
if imp is None:
    st.info("`reports/shap_importance_h1.csv` not found. It is produced during training.")
else:
    top = imp.head(15).iloc[::-1]
    fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h")
    fig.update_traces(marker_color=ui.OBSERVED)
    st.plotly_chart(ui.base_layout(fig, "Mean |SHAP| value", "", 460), use_container_width=True)
    summary = config.get_paths().figures / "shap_summary_h1.png"
    if summary.exists():
        with st.expander("SHAP summary plot"):
            st.image(str(summary))

st.subheader(f"Explain a forecast for {city}")
try:
    res = fc.forecast_city(city, data, art, ui.get_weather(city))
except (fc.UnsupportedCityError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
h = st.selectbox("Forecast horizon (days ahead)", sorted(art.models))
exp = xai.explain_row(art.models[h], res.features[h])
if exp is None:
    st.warning("An explanation is not available for this model type or environment. Forecasts are unaffected.")
else:
    st.markdown(xai.plain_english(exp))
    c1, c2 = st.columns(2)
    c1.metric("Model average output", f"{exp.base_value:.1f} µg/m³")
    c2.metric("This forecast (before clipping)", f"{exp.prediction:.1f} µg/m³")
    t = exp.table.head(12).iloc[::-1]
    fig = px.bar(t, x="shap_value", y="feature", orientation="h", color="effect",
                 color_discrete_map={"raises forecast": ui.FORECAST, "lowers forecast": ui.OBSERVED})
    st.plotly_chart(ui.base_layout(fig, "Feature contributions (µg/m³)", "", 460), use_container_width=True)
    st.dataframe(exp.table.round(3), use_container_width=True, hide_index=True)
ui.footer()
