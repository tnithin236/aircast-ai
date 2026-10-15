import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from src import data_loader  # noqa: E402

ui.setup("Data explorer")
st.title("Data explorer")
data = ui.get_data()
cities = sorted(data["city"].unique())

sel = st.sidebar.multiselect("Cities", cities, default=cities[:1])
pollutant = st.sidebar.selectbox("Pollutant", ["pm25", "pm10"], format_func=lambda c: {"pm25": "PM2.5", "pm10": "PM10"}[c])
lo, hi = data["date"].min().date(), data["date"].max().date()
rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
if not sel:
    st.info("Pick at least one city in the sidebar.")
    st.stop()
if not isinstance(rng, tuple) or len(rng) != 2:
    st.info("Select both a start and an end date.")
    st.stop()

f = data[data["city"].isin(sel) & data["date"].between(str(rng[0]), str(rng[1]))]
if f.empty:
    st.warning("No rows match these filters.")
    st.stop()

st.caption("Observed values from the cleaned dataset. Extreme values are kept because they may be real pollution events.")
fig = px.line(f, x="date", y=pollutant, color="city")
st.plotly_chart(ui.base_layout(fig, f"{pollutant.upper()} observations", f"{pollutant.upper()} (µg/m³)"), use_container_width=True)

st.subheader("Missing values in selection")
q = data_loader.data_quality_report(f)["per_city"]
st.dataframe(q, use_container_width=True, hide_index=True)
st.subheader("Preview")
st.dataframe(f.head(500), use_container_width=True, hide_index=True)
st.download_button("Download filtered data", f.to_csv(index=False), file_name="aircast_filtered.csv", mime="text/csv")
ui.footer()
