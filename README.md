# AirCast AI: Air Quality Forecasting & Pollution Alert System

<img width="1917" height="1020" alt="Screenshot 2026-10-16 111457" src="https://github.com/user-attachments/assets/236f2814-b846-4adf-9e84-7c7a438e8789" />


Seven-day daily **PM2.5** forecasts for Indian cities, built from historical CPCB city-day data and Open-Meteo weather, with SHAP explanations, configurable threshold alerts and a Streamlit dashboard.

> Educational forecasting and decision-support tool. Forecasts are model estimates, not official measurements, and alerts are not government or public-health warnings.

## Key features
- Real data: Kaggle *Air Quality Data in India* (`city_day.csv`, **daily** resolution) plus optional Open-Meteo historical weather.
- Direct multi-horizon forecasting (one model per day 1 to 7), chronological 70/15/15 split with a 7-day purge gap.
- Persistence and seasonal-naive baselines; Random Forest, XGBoost and LightGBM compared; model chosen on **validation MAE only**.
- SHAP global and per-forecast explanations (with a non-causality note).
- Alert engine: threshold exceeded, predicted peak, sustained exceedance, stale data, forecast-quality warning; CPCB PM2.5 category labels.
- 6-page Streamlit app, optional FastAPI service, pytest suite.

## Architecture
```
Kaggle city_day.csv ──► preprocessing ──► data/processed/city_day_clean.csv ─┐
Open-Meteo archive  ──► data/raw/weather_<city>.csv ─────────────────────────┤
                                                                             ▼
                          features.py ──► train.py ──► models/*.joblib, reports/*
                                                                             │
        Streamlit app / FastAPI ◄── forecast.py, alerts.py, explainability.py ◄┘
```
Training (`scripts/train_model.py`, or the Colab notebook) is separate from inference; the app never retrains.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt`

### Option A: use the artifacts from your Colab run
Copy the three folders from the downloaded `aircast_artifacts` into this project:
```powershell
$src = "$HOME\Downloads\aircast_artifacts"
Copy-Item "$src\models\*"         .\models\           -Force
Copy-Item "$src\reports\*"        .\reports\          -Recurse -Force
Copy-Item "$src\data\processed\*" .\data\processed\   -Force
python scripts/download_weather.py      # weather is NOT in the Colab zip; needed for faithful forecasts
streamlit run app/streamlit_app.py
```
Check `models/model_metadata.json` → `versions` and install matching scikit-learn / xgboost / lightgbm if the models fail to load.

### Option B: rebuild everything locally
```powershell
python scripts/download_data.py        # needs KAGGLE_API_TOKEN, or place city_day.csv in data/raw/
python scripts/preprocess_data.py
python scripts/download_weather.py
python scripts/train_model.py          # add --quick for a fast smoke test, --no-weather to skip weather
pytest -q
streamlit run app/streamlit_app.py
```
`scripts/train_model.py` ports the Colab pipeline; results can differ slightly from Colab if library versions differ.

## Data
| Source | Use | Notes |
|---|---|---|
| Kaggle `rohanrao/air-quality-data-in-india` (CPCB) | PM2.5 / PM10 target | Daily city-level. **Check the dataset page for its license**; it is not redistributed here. |
| Open-Meteo Historical Weather API | Temperature, humidity, wind, precipitation, pressure | Daily means/sums, time zone `Asia/Kolkata`, approximate city-centre coordinates (`src/data_loader.py`). Check Open-Meteo's terms for your use. |

Cleaned table columns: `city, date, pm25, pm10` (µg/m³). Policies: bad dates dropped, negatives → NaN, duplicate city-days averaged, target values never imputed, extreme values kept. Cities with fewer than 1000 observed PM2.5 days were excluded in the Colab run.

## Method
- **Features at origin day *t*:** `pm25_t`, lags (1, 2, 3, 7, 14, 30), rolling mean/std/min/max, 7-day trend, origin-day weather and 3-day weather rolls, cyclical calendar features of the *target* date, city one-hots. Everything uses data up to and including *t*.
- **Strategy:** direct. Model *h* predicts PM2.5 at *t+h*. No recursion, so no compounding of predicted inputs.
- **Split:** chronological on origin date; training origins stop 7 days before validation starts, and validation 7 days before test.
- **Final model:** refit on train+validation for deployment; reported test metrics come from the train-only fit.
- **Non-negativity:** predictions are clipped at 0. Both clipped and unclipped metrics are saved (`model_comparison.csv`).
- **AQI:** the app shows PM2.5 concentration and its CPCB **PM2.5 sub-index category**. It does not compute overall AQI.
- **Weather at forecast time:** only origin-day *observed* weather is used; no weather forecasts. For a real-time system, forecast weather would be needed for future days.

## Results
Open the **Model Performance** page, or run `python scripts/show_results.py` for a Markdown table from `reports/model_comparison.csv`, and paste your numbers here. Numbers are intentionally not hard-coded in this README, because they come from your training run. If the selected model does not beat the baseline, say so.

## Dashboard pages
Overview · Forecast · Model Performance · Explainability · Alerts · Data Explorer. Observed values are drawn solid, forecasts dashed. Take screenshots from your own running app for this section.

## Alerts
Defaults (editable on the Alerts page): threshold 60 µg/m³, high severity 120, peak level 90, sustained = 3 consecutive days, stale after 3 days. Quality warnings fire when the model fails to beat the baseline or the test error at a horizon exceeds a configurable limit (30 µg/m³, a dashboard heuristic). Alert IDs are deterministic, so saving to history skips duplicates. No email/SMS is sent.

## API (optional)
```powershell
uvicorn api.main:app --reload
```
`GET /health`, `GET /locations`, `GET /model-info`, `POST /forecast {"location":"Delhi","horizon":7}`, `POST /alerts {"location":"Delhi","threshold":60}`. Docs at `/docs`. Returns 404 for unsupported cities, 422 for invalid input, 503 if models are missing.

## Deployment
Streamlit Community Cloud can run `app/streamlit_app.py` if the repo contains `requirements.txt`, `models/`, `reports/`, and `data/processed/city_day_clean.csv` (small enough to commit if licensing allows; remove it from `.gitignore`) plus cached weather files if you want weather features. Not deployed or verified by this project. Always run locally first. Do not load model files from untrusted sources (joblib is pickle).

## Tests
`pytest -q` covers loading, schema errors, date parsing, duplicates, missing targets, features and ordering, chronological splits, forecast shape, non-negativity, no-future-data access, alerts, API validation, missing-model and unsupported-city behaviour, and a headless run of every Streamlit page. Test fixtures are synthetic and used only in tests.

## Known limitations
- Daily data only: no hourly forecast (not implemented; would need real hourly observations).
- Latest data in the dataset is historical, so forecasts start from that date and are **not live**.
- PM10 forecasting is not implemented.
- City coordinates are approximate; station-level weather differs.
- Uses observed origin-day weather, not forecast weather.
- Tree models are often only modestly better than persistence at short horizons; check your own metrics.
- LSTM, MLflow, live data and email alerts are not implemented.

## Responsible use
Do not use this for health decisions. Follow official CPCB/state advisories.

## Future work
Weather-forecast inputs, PM10 models, hourly data, rolling-origin validation, MLflow, live-data integration.
#
