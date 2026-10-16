"""Shared Streamlit helpers: theme, page setup, cached loaders, charts and dashboard components.

Requires Streamlit >= 1.39 (keyed containers, st.fragment, material icons in page links).
"""
from __future__ import annotations

import base64
import re
import sys
from functools import lru_cache
from numbers import Real
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, data_loader  # noqa: E402
from src import forecast as fc  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"  # optional: drop a globe image at app/assets/hero.png

OBSERVED, FORECAST, MUTED = "#29b6ff", "#ff9f43", "#8ea4d6"
GOOD, WARN, BAD = "#17c98b", "#ff9f43", "#ff5d5d"
USER_INITIALS = "NT"
ATTRIBUTION = ("Air quality: CPCB city-day data via Kaggle (rohanrao/air-quality-data-in-india). "
               "Weather: Open-Meteo historical archive. Educational forecasting tool, not an official measurement or health warning.")

# CPCB 24-hour PM2.5 bands (µg/m³): upper bound, name, colour
CPCB_BANDS = [(30, "Good", "#17c98b"), (60, "Satisfactory", "#8bd450"), (90, "Moderate", "#f4c542"),
              (120, "Poor", "#ff9f43"), (250, "Very poor", "#ff5d5d"), (float("inf"), "Severe", "#c2185b")]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,800&display=swap');
.stApp {font-family:'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; color:#e8eefc;
  background: radial-gradient(1100px 520px at 88% -8%, #12348a55 0%, transparent 62%),
              radial-gradient(900px 500px at 0% 100%, #0a2a6a3d 0%, transparent 60%), #050b1f;}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.2rem; max-width:1320px;}
hr {border-color:#12234a !important;}
[data-testid="stAlert"] {border-radius:12px;}
.note {color:#8795ad; font-size:0.82rem;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background:linear-gradient(180deg,#07112c 0%,#050b1f 100%); border-right:1px solid #12234a;}
[data-testid="stSidebarNav"] {display:none;}
.brand {text-align:center; padding:.2rem 0 1.1rem;}
.brand .logo {display:flex; justify-content:center; margin-bottom:.35rem; filter:drop-shadow(0 0 12px #29b6ff88);}
.brand .name {font-size:1.6rem; font-weight:800; letter-spacing:-.01em;}
.brand .tag {font-size:.78rem; color:#9fb2dc;}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {border-radius:10px; padding:.5rem .8rem; color:#c3d0ef; margin-bottom:2px;}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {background:#0f2250;}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
  background:linear-gradient(90deg,#1d4ed8,#1e40af); color:#fff; box-shadow:0 0 18px #2563eb66;}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {font-weight:600; font-size:.95rem;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#0b1736; border:1px solid #1d3a7a; border-radius:10px;}
[data-testid="stSidebar"] label p {color:#c3d0ef; font-weight:600;}
.navsep {height:1px; background:#12234a; margin:1rem 0 .8rem;}
.side-foot {margin-top:2.2rem; text-align:center; color:#7fe3b0; font-style:italic; font-size:.9rem;
  padding:3.2rem 0 .4rem; background:radial-gradient(220px 90px at 50% 100%, #1d4fb855, transparent 70%);}

/* ---------- hero ---------- */
.hero {position:relative; min-height:200px; padding:.2rem 0 1.2rem; overflow:hidden;}
.hero .pill, .hero .title, .hero p {position:relative; z-index:2;}
.pill {display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .85rem; border:1px solid #1f3d85;
  background:#0a1a47aa; border-radius:999px; font-size:.8rem; color:#cfe0ff;}
.hero .title {font-size:4rem; line-height:1.05; font-weight:800; font-style:italic; letter-spacing:-.02em; margin:.7rem 0 .6rem;}
.hero .title span {background:linear-gradient(90deg,#29b6ff,#6a5cff); -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero p {max-width:600px; color:#b8c7ea; font-size:1rem; margin:0; line-height:1.5;}
.orb {position:absolute; right:4%; top:-34px; width:330px; height:330px; border-radius:50%; z-index:0;
  background:radial-gradient(circle at 36% 30%, #2f80ff 0%, #0d2f8f 34%, #06143d 64%, transparent 72%);
  box-shadow:0 0 90px 8px #2f6bff55, inset -30px -20px 60px #00000099; opacity:.92;}
.orb.custom {width:640px; height:320px; right:0; top:-20px; border-radius:0; box-shadow:none; background-color:transparent;
  -webkit-mask-image:linear-gradient(90deg, transparent 0%, #000 35%); mask-image:linear-gradient(90deg, transparent 0%, #000 35%);}
.ring {position:absolute; border:1px solid #3b82f67a; border-radius:50%; pointer-events:none; z-index:1;}
.ring.r1 {right:0; top:20px; width:520px; height:170px; transform:rotate(-12deg);}
.ring.r2 {right:40px; top:50px; width:430px; height:130px; transform:rotate(-12deg); opacity:.55;}
.status {position:absolute; top:.1rem; right:0; display:flex; gap:.8rem; align-items:center; z-index:3;}
.status .box {display:flex; gap:.6rem; align-items:center; padding:.45rem .9rem; border:1px solid #1f3d85; border-radius:12px;
  background:#08153acc; font-size:.82rem;}
.status .sub {color:#9fb2dc; font-size:.75rem;}
.status .sep {width:1px; height:16px; background:#1f3d85;}
.dot {width:9px; height:9px; border-radius:50%; display:inline-block;}
.avatar {width:38px; height:38px; border-radius:50%; background:linear-gradient(135deg,#2563eb,#4f46e5); display:grid; place-items:center;
  font-weight:700; font-size:.85rem;}

/* ---------- KPI cards ---------- */
.kpi {position:relative; overflow:hidden; border-radius:18px; padding:1.05rem 1.15rem; min-height:128px; display:flex; gap:1rem;
  align-items:flex-start; border:1px solid #1d4fb8;}
.kpi .ic {flex:none; width:54px; height:54px; border-radius:50%; display:grid; place-items:center; color:#fff;}
.kpi .lbl {font-size:.85rem; color:#c9d6f2;}
.kpi .val {font-size:1.85rem; font-weight:800; line-height:1.15; margin:.15rem 0 .55rem;}
.kpi .val small {font-size:.95rem; font-weight:600; margin-left:.25rem; color:#dbe6ff;}
.kpi .spark {position:absolute; right:14px; bottom:12px; opacity:.95;}
.kpi.blue {background:linear-gradient(135deg,#0c2d6e,#081a45); border-color:#1d4fb8; box-shadow:0 0 26px #1d4fb833;}
.kpi.blue .ic {background:radial-gradient(circle at 35% 30%,#2a7bff,#1244b5);}
.kpi.purple {background:linear-gradient(135deg,#2b1670,#150b3a); border-color:#6d4aff; box-shadow:0 0 26px #6d4aff30;}
.kpi.purple .ic {background:radial-gradient(circle at 35% 30%,#8b5cf6,#5b34d6);}
.kpi.green {background:linear-gradient(135deg,#064a3b,#05261f); border-color:#17c98b; box-shadow:0 0 26px #17c98b2e;}
.kpi.green .ic {background:radial-gradient(circle at 35% 30%,#1fd8a0,#0b8f68);}
.kpi.orange {background:linear-gradient(135deg,#4d2c08,#2a1704); border-color:#ff9f43; box-shadow:0 0 26px #ff9f4330;}
.kpi.orange .ic {background:radial-gradient(circle at 35% 30%,#ffb347,#d9770d);}
.chip {display:inline-flex; align-items:center; gap:.35rem; font-size:.74rem; font-weight:600; padding:.22rem .65rem; border-radius:999px;}
.chip.neutral {background:#ffffff1a; color:#dbe6ff;}
.chip.down {background:#0e3a35; color:#6ee7b7;}
.chip.up {background:#4a1a1a; color:#ff9d9d;}

/* ---------- panels ---------- */
[class*="st-key-panel"] {background:linear-gradient(180deg,#0a1636ee,#07112b); border:1px solid #1c3b80; border-radius:18px;
  padding:1rem 1.2rem 1.1rem; box-shadow:0 0 30px #1d4fb822;}
.ptitle {display:flex; align-items:center; gap:.6rem; font-weight:700; font-size:1.05rem;}
.ptitle .pi {width:30px; height:30px; border-radius:8px; background:#12306e; display:grid; place-items:center; color:#7cc4ff;}
.mini {position:relative; background:#0a1431; border:1px solid #1b3470; border-radius:12px; padding:.85rem 1rem; min-height:108px;}
.mini .lbl {color:#c3d0ef; font-size:.85rem;}
.mini .val {font-size:1.5rem; font-weight:800; margin:.2rem 0 .5rem;}
.mini .val small {font-size:.85rem; font-weight:600; margin-left:.2rem; color:#c3d0ef;}
.mini .visual {position:absolute; right:14px; top:50%; transform:translateY(-50%);}
.insight-wrap {display:flex; align-items:center; justify-content:space-between; gap:1rem;}
.insight-list {flex:1;}
.insight {display:flex; gap:.7rem; align-items:flex-start; margin-bottom:.7rem; color:#dbe6ff; font-size:.92rem; line-height:1.45;}
.insight .tick {flex:none; width:22px; height:22px; border-radius:50%; display:grid; place-items:center; margin-top:1px;}
.insight .tick.ok {background:#17c98b; color:#04261c;}
.insight .tick.warn {background:#ff9f43; color:#2a1704;}
.badge {flex:none; width:158px; height:158px; border-radius:50%; display:flex; flex-direction:column; align-items:center;
  justify-content:center; text-align:center; border:2px solid var(--c); background:radial-gradient(circle at 50% 35%, #0e3a3a99, #07151f);
  box-shadow:0 0 36px color-mix(in srgb, var(--c) 40%, transparent);}
.badge b {font-size:1.25rem; line-height:1.1; margin-top:.2rem;}
.badge span {font-size:.74rem; color:#c3d0ef; margin-top:.15rem;}

/* ---------- range selector (radio) ---------- */
[class*="st-key-panel"] [data-testid="stRadio"] [role="radiogroup"] {justify-content:flex-end; gap:0; background:#08122e;
  border:1px solid #1c3b80; border-radius:999px; padding:3px; width:fit-content; margin-left:auto;}
[class*="st-key-panel"] [data-testid="stRadio"] label {padding:.25rem .95rem; border-radius:999px; margin:0; cursor:pointer;}
[class*="st-key-panel"] [data-testid="stRadio"] label > div:first-child {display:none;}
[class*="st-key-panel"] [data-testid="stRadio"] label:has(input:checked) {background:linear-gradient(90deg,#1d4ed8,#2563eb);}
[class*="st-key-panel"] [data-testid="stRadio"] label p {font-size:.82rem; color:#c3d0ef; margin:0;}
[class*="st-key-panel"] [data-testid="stRadio"] label:has(input:checked) p {color:#fff; font-weight:700;}

/* ---------- native metrics (used on other pages) ---------- */
[data-testid="stMetric"] {background:#0a1431; border:1px solid #1b3470; border-radius:12px; padding:14px 16px;}
[data-testid="stMetricLabel"] {color:#8ea4d6;}

@media (max-width: 900px) {
  .hero .title {font-size:2.6rem;} .orb, .ring {opacity:.35;} .status {position:relative; margin-bottom:.8rem;}
  .insight-wrap {flex-direction:column;}
}
</style>
"""

_ICONS = {
    "wind": '<path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6M10 22h4"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "alert": '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
}


def icon(name: str, size: int = 22, color: str = "currentColor", stroke: float = 2) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')


def _h(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never turns the HTML into a code block."""
    return "\n".join(s.strip() for s in markup.splitlines() if s.strip())


def render(markup: str, target=st) -> None:
    target.markdown(_h(markup), unsafe_allow_html=True)


# ---------------------------------------------------------------- page setup + sidebar
_NAV_ICONS = {"forecast": ":material/show_chart:", "performance": ":material/bar_chart:", "explain": ":material/lightbulb:",
              "alert": ":material/notifications:", "data": ":material/database:", "model": ":material/bar_chart:"}


def _nav_items() -> list[tuple[str, str, str]]:
    """Dashboard + every file in app/pages, labelled from its file name ('2_Model_Performance.py' -> 'Model Performance')."""
    items = [("Dashboard", ":material/home:", "streamlit_app.py")]
    for p in sorted((APP_DIR / "pages").glob("*.py")):
        label = re.sub(r"^\d+[_\s-]*", "", p.stem).replace("_", " ").strip()
        ic = next((v for k, v in _NAV_ICONS.items() if k in label.lower()), ":material/article:")
        items.append((label, ic, f"pages/{p.name}"))
    return items


def _sidebar() -> None:
    with st.sidebar:
        render(f"""
        <div class="brand">
          <div class="logo"><svg width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="url(#lg)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#29e0ff"/><stop offset="1" stop-color="#3b82f6"/></linearGradient></defs>
            {_ICONS['leaf']}</svg></div>
          <div class="name">AirCast AI</div>
          <div class="tag">Smarter Air. Healthier Tomorrow.</div>
        </div>""")
        for label, ic, target in _nav_items():
            try:
                st.page_link(target, label=label, icon=ic)
            except Exception:  # noqa: BLE001 - a page that cannot be linked is simply not listed
                continue
        render('<div class="navsep"></div>')


def setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} | AirCast AI", page_icon="🍃", layout="wide")
    st.markdown(_h(CSS), unsafe_allow_html=True)
    _sidebar()


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


# ---------------------------------------------------------------- metrics helpers
def _flat(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = re.sub(r"[^a-z0-9]", "", f"{prefix}{k}".lower())
        if isinstance(v, dict):
            out.update(_flat(v, key))
        elif isinstance(v, Real) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def perf_metrics(test_metrics) -> dict[str, float | None]:
    """Find MAE / RMSE / R² (and a baseline MAE) in the saved metadata, whatever the exact key spelling is."""
    flat = _flat(test_metrics) if isinstance(test_metrics, dict) else {}

    def is_base(k: str) -> bool:
        return any(w in k for w in ("baseline", "persist", "naive"))

    def pick(pattern: str, base: bool = False):
        return next((v for k, v in flat.items() if is_base(k) == base and re.search(pattern, k)), None)

    return {"mae": pick("mae"), "rmse": pick("rmse"), "r2": pick("r2|rsquared"), "base_mae": pick("mae", True)}


def mae_chip(m: dict, beats: bool | None) -> tuple[str, str]:
    if m["mae"] is not None and m["base_mae"] is not None:
        d = m["base_mae"] - m["mae"]
        return (f"▼ {d:.2f} lower than baseline", "down") if d > 0 else (f"▲ {abs(d):.2f} higher than baseline", "up")
    if beats is True:
        return "Beats baseline on MAE", "down"
    if beats is False:
        return "Does not beat baseline", "up"
    return "", "neutral"


def aqi_band(pm25: float) -> tuple[str, str]:
    for upper, name, color in CPCB_BANDS:
        if pm25 <= upper:
            return name, color
    return CPCB_BANDS[-1][1], CPCB_BANDS[-1][2]


# ---------------------------------------------------------------- components (HTML)
def spark_bars(values, color: str, w: int = 84, h: int = 46) -> str:
    vals = [float(v) for v in values if pd.notna(v)]
    if len(vals) < 2:
        return ""
    lo, span = min(vals), (max(vals) - min(vals)) or 1.0
    n = len(vals)
    bars = "".join(
        f'<rect x="{i * (w / n):.1f}" y="{h - (6 + (h - 6) * (v - lo) / span):.1f}" width="{w / (n * 1.6):.1f}" '
        f'height="{6 + (h - 6) * (v - lo) / span:.1f}" rx="1.5" fill="{color}" opacity="{0.4 + 0.6 * (i + 1) / n:.2f}"/>'
        for i, v in enumerate(vals))
    return f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{bars}</svg>'


def spark_line(values, color: str, w: int = 96, h: int = 40) -> str:
    vals = [float(v) for v in values if pd.notna(v)]
    if len(vals) < 2:
        return ""
    lo, span = min(vals), (max(vals) - min(vals)) or 1.0
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - 4 - (h - 8) * (v - lo) / span:.1f}" for i, v in enumerate(vals))
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline points="{pts}" fill="none" '
            f'stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def ring(frac: float, size: int = 64) -> str:
    frac = max(0.0, min(1.0, frac))
    c = 2 * 3.14159 * 26
    return (f'<svg class="visual" width="{size}" height="{size}" viewBox="0 0 64 64"><defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="#29b6ff"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs>'
            f'<circle cx="32" cy="32" r="26" fill="none" stroke="#16295a" stroke-width="7"/>'
            f'<circle cx="32" cy="32" r="26" fill="none" stroke="url(#rg)" stroke-width="7" stroke-linecap="round" '
            f'stroke-dasharray="{frac * c:.1f} {c:.1f}" transform="rotate(-90 32 32)"/></svg>')


@lru_cache(maxsize=1)
def _hero_image_css() -> str:
    p = ASSETS / "hero.png"
    if not p.exists():
        return ""
    return f"background-image:url(data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()});background-size:cover;background-position:center;"


def hero(live: bool, last_date) -> None:
    color, text = (GOOD, "Live data") if live else (WARN, "Historical data")
    img = _hero_image_css()
    decor = "" if img else '<div class="ring r1"></div><div class="ring r2"></div>'
    render(f"""
    <div class="hero">
      <div class="orb {'custom' if img else ''}" style="{img}"></div>{decor}
      <div class="status">
        <div class="box"><span class="dot" style="background:{color};box-shadow:0 0 8px {color}"></span><b>{text}</b>
          <span class="sep"></span><span class="sub">Latest record: {pd.Timestamp(last_date).strftime('%d %b %Y')}</span></div>
        <div class="avatar">{USER_INITIALS}</div>
      </div>
      <div class="pill">{icon('pulse', 14, '#7cc4ff')} Air Quality Analytics</div>
      <div class="title">AirCast <span>AI</span></div>
      <p>Advanced forecasting and monitoring for cleaner, healthier air in Indian cities using the power of Machine Learning.</p>
    </div>""")


def kpi_card(kind: str, icon_name: str, label: str, value: str, unit: str = "", chip: str = "", chip_kind: str = "neutral",
             chip_icon: str = "", spark: str = "") -> str:
    chip_html = f'<span class="chip {chip_kind}">{icon(chip_icon, 13) if chip_icon else ""}{chip}</span>' if chip else ""
    unit_html = f"<small>{unit}</small>" if unit else ""
    return f"""
    <div class="kpi {kind}">
      <div class="ic">{icon(icon_name, 26)}</div>
      <div><div class="lbl">{label}</div><div class="val">{value}{unit_html}</div>{chip_html}</div>
      {spark}
    </div>"""


def panel_title(icon_name: str, text: str) -> str:
    return f'<div class="ptitle"><span class="pi">{icon(icon_name, 17)}</span>{text}</div>'


def mini_stat(label: str, value: str, unit: str = "", chip: str = "", chip_kind: str = "neutral", visual: str = "") -> str:
    chip_html = f'<span class="chip {chip_kind}">{chip}</span>' if chip else ""
    unit_html = f"<small>{unit}</small>" if unit else ""
    return f'<div class="mini"><div class="lbl">{label}</div><div class="val">{value}{unit_html}</div>{chip_html}{visual}</div>'


def insights(items: list[tuple[str, str]], band: str, color: str) -> str:
    rows = "".join(f'<div class="insight"><span class="tick {k}">{icon("check" if k == "ok" else "alert", 13, stroke=3)}</span>'
                   f"<span>{t}</span></div>" for k, t in items)
    return f"""
    <div class="insight-wrap">
      <div class="insight-list">{rows}</div>
      <div class="badge" style="--c:{color}">{icon('leaf', 30, color)}<b>{band}</b><span>Next-day air quality</span></div>
    </div>"""


def sidebar_footer() -> None:
    render(f'<div class="side-foot">{icon("leaf", 16, "#7fe3b0")} Clean Air. Better Life.</div>', st.sidebar)


# ---------------------------------------------------------------- charts
def base_layout(fig: go.Figure, title: str | None, ytitle: str = "PM2.5 (µg/m³)", height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", title=dict(text=title) if title else None, height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title=ytitle,
                      font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", color="#c3d0ef"),
                      margin=dict(l=10, r=10, t=50 if title else 36, b=10),
                      legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"), hovermode="x unified")
    fig.update_xaxes(gridcolor="#15264d", zeroline=False)
    fig.update_yaxes(gridcolor="#15264d", zeroline=False, rangemode="tozero")
    return fig


def history_forecast_chart(hist: pd.DataFrame, fcast: pd.DataFrame, days: int = 120, threshold: float | None = None,
                           title: str | None = "Observed and forecast") -> go.Figure:
    h = hist.tail(days)
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["pm25"], name="Observed", mode="lines+markers" if len(h) <= 45 else "lines",
                    line=dict(color=OBSERVED, width=2.2), marker=dict(size=5), fill="tozeroy", fillcolor="rgba(41,182,255,0.10)")
    origin = pd.Timestamp(fcast["origin_date"].iloc[0])
    f_dates = pd.to_datetime(fcast["forecast_date"])
    last = hist[hist["date"] == origin]["pm25"]
    xs = [origin] + list(f_dates)
    ys = ([float(last.iloc[0])] if len(last) else [fcast["pm25_pred"].iloc[0]]) + list(fcast["pm25_pred"])
    fig.add_vrect(x0=origin, x1=f_dates.max(), fillcolor="rgba(255,159,67,0.08)", line_width=0, layer="below")
    fig.add_shape(type="line", x0=origin, x1=origin, y0=0, y1=1, yref="paper", line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=origin + (f_dates.max() - origin) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                       text="Forecast region", font=dict(size=11, color="#ffe2c2"), bgcolor="#4d2c08", borderpad=4)
    fig.add_scatter(x=xs, y=ys, name="Forecast", mode="lines+markers", line=dict(color=FORECAST, width=2.2, dash="dash"),
                    marker=dict(size=6))
    # Only draw the threshold when it is on the same scale as the data; otherwise it flattens the chart.
    if threshold and threshold <= 1.6 * max(h["pm25"].max(), max(ys)):
        fig.add_hline(y=threshold, line=dict(color=MUTED, dash="dot"), annotation_text=f"Threshold {threshold:g}")
    return base_layout(fig, title)


def footer() -> None:
    st.divider()
    st.markdown(f"<p class='note'>{ATTRIBUTION}</p>", unsafe_allow_html=True)
    sidebar_footer()