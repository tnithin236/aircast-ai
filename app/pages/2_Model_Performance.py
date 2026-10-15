import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import config  # noqa: E402
from src import evaluate as ev  # noqa: E402

ui.setup("Model performance")
st.title("Model performance")
art = ui.get_artifacts(required=False)
rep = ev.load_reports()
if rep["comparison"] is None:
    st.error("No saved evaluation found in reports/. Copy the Colab `reports` folder here or run `python scripts/train_model.py`.")
    st.stop()

tm = rep["test_metrics"] or {}
if tm:
    sel, base = tm["selected"], tm["best_baseline"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{tm['selected_model']} MAE", f"{sel['MAE']:.2f} µg/m³", f"{sel['MAE'] - base['MAE']:+.2f} vs {base['name']}", delta_color="inverse")
    c2.metric("RMSE", f"{sel['RMSE']:.2f} µg/m³", f"{sel['RMSE'] - base['RMSE']:+.2f}", delta_color="inverse")
    c3.metric("R²", f"{sel['R2']:.3f}")
    if tm.get("beats_baseline_on_MAE") is False:
        st.warning("The selected model did not beat the best baseline on overall MAE on the test set.")
    st.caption(f"Test period starts {tm.get('test_split_from')}. The model was selected on validation MAE only.")

st.subheader("Model comparison (test set)")
st.dataframe(rep["comparison"].round(3), use_container_width=True, hide_index=True)
st.caption("Persistence and SeasonalNaive7 are baselines. `*_unclipped` columns show metrics before non-negative clipping.")

if rep["by_horizon"] is not None:
    st.subheader("Error by forecast horizon")
    cols = st.columns(2)
    for col, metric in zip(cols, ["MAE", "RMSE"]):
        fig = px.line(rep["by_horizon"], x="horizon", y=metric, color="model", markers=True)
        col.plotly_chart(ui.base_layout(fig, f"{metric} by horizon", f"{metric} (µg/m³)", 320), use_container_width=True)

c1, c2 = st.columns(2)
if rep["by_city"] is not None:
    c1.subheader("By city (selected model)")
    c1.dataframe(rep["by_city"].round(2), use_container_width=True, hide_index=True)
if rep["by_month"] is not None:
    c2.subheader("By month of target date")
    c2.dataframe(rep["by_month"].round(2), use_container_width=True, hide_index=True)

hp = tm.get("high_pollution")
if hp and "MAE" in hp:
    st.subheader("High-pollution days")
    st.write(f"Days at or above the training 90th percentile ({hp['threshold']:.1f} µg/m³), n={hp.get('n')}: "
             f"MAE {hp['MAE']:.1f}, RMSE {hp['RMSE']:.1f} µg/m³.")

st.subheader("Saved evaluation figures")
figs = [config.get_paths().figures / n for n in ("actual_vs_pred_h1.png", "actual_vs_pred_h7.png", "residuals.png")]
for p in figs:
    if p.exists():
        st.image(str(p))

st.subheader("Model metadata")
if art:
    meta = {k: v for k, v in art.metadata.items() if k != "test_metrics"}
    st.json(meta, expanded=False)
else:
    st.info("Model artifacts not loaded; metadata unavailable.")
ui.footer()
