"""Training pipeline (port of the Colab notebook): direct strategy, one model per horizon.

Model selection uses validation MAE only. Test metrics are computed once from train-only fits.
The saved production models are refit on train+validation; the test period stays untouched.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from . import config
from .evaluate import group_metrics, metrics, save_figures
from .features import (base_feature_names, build_horizon_frame, compute_split_dates, make_design, split_frame)

log = logging.getLogger(__name__)

GRIDS = {
    "RandomForest": [dict(n_estimators=200, max_depth=12, min_samples_leaf=5),
                     dict(n_estimators=300, max_depth=None, min_samples_leaf=3)],
    "XGBoost": [dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
                dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0)],
    "LightGBM": [dict(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
                 dict(n_estimators=600, num_leaves=15, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0)],
}
QUICK_GRIDS = {"RandomForest": [dict(n_estimators=30, max_depth=8, min_samples_leaf=5)]}
BASELINES = [("Persistence", "pm25_t"), ("SeasonalNaive7", "seasonal_naive")]


def make_model(name: str, params: dict):
    """Estimators are imported lazily so unused libraries are not required."""
    seed = config.SEED
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestRegressor(random_state=seed, n_jobs=-1, **params))
    if name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed, tree_method="hist", n_jobs=-1, **params)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
    raise ValueError(f"Unknown model: {name}")


@dataclass
class TrainResult:
    selected: str
    comparison: pd.DataFrame
    test_metrics: dict
    metadata: dict


def run_training(df: pd.DataFrame, weather: dict | None = None, grids: dict | None = None,
                 save: bool = True, shap_artifacts: bool = True) -> TrainResult:
    paths = config.get_paths()
    grids = grids or GRIDS
    weather = weather or {}
    cities = sorted(df["city"].unique())
    horizons = config.HORIZONS
    base = base_feature_names()
    features = base + [f"city_{c}" for c in cities]
    design = lambda fr: make_design(fr, features, base, cities)  # noqa: E731

    frames = {h: build_horizon_frame(df, weather, h) for h in horizons}
    val_start, test_start = compute_split_dates(df)
    splits = {h: split_frame(frames[h], val_start, test_start) for h in horizons}
    log.info("val_start=%s test_start=%s", val_start.date(), test_start.date())

    # ---- validation: best config per family, then pick family (validation only)
    best_cfg, val_rows = {}, []
    for name, grid in grids.items():
        scores = []
        for i, params in enumerate(grid):
            t0 = time.time()
            maes = []
            for h in horizons:
                tr, va, _ = splits[h]
                m = make_model(name, params).fit(design(tr), tr["y"])
                maes.append(mean_absolute_error(va["y"], m.predict(design(va))))
            scores.append(float(np.mean(maes)))
            log.info("%s cfg%d val MAE=%.2f (%.0fs)", name, i, scores[-1], time.time() - t0)
        k = int(np.argmin(scores))
        best_cfg[name] = grid[k]
        val_rows.append(dict(model=name, val_MAE=scores[k], cfg=grid[k]))
    for bname, col in BASELINES:
        maes = []
        for h in horizons:
            va = splits[h][1].dropna(subset=[col])
            maes.append(mean_absolute_error(va["y"], va[col]))
        val_rows.append(dict(model=bname, val_MAE=float(np.mean(maes)), cfg={}))
    val_df = pd.DataFrame(val_rows).sort_values("val_MAE")
    selected = val_df[val_df["model"].isin(grids)].iloc[0]["model"]

    # ---- test evaluation with train-only fits
    preds, train_only = [], {}
    for name in grids:
        train_only[name] = {}
        for h in horizons:
            tr, _, te = splits[h]
            m = make_model(name, best_cfg[name]).fit(design(tr), tr["y"])
            train_only[name][h] = m
            p = m.predict(design(te))
            d = te[["city", "origin", "target_date", "y"]].copy()
            d["pred_raw"], d["pred"] = p, np.clip(p, 0, None)
            d["model"], d["horizon"] = name, h
            preds.append(d)
    for bname, col in BASELINES:
        for h in horizons:
            te = splits[h][2].dropna(subset=[col])
            d = te[["city", "origin", "target_date", "y"]].copy()
            d["pred_raw"] = d["pred"] = te[col]
            d["model"], d["horizon"] = bname, h
            preds.append(d)
    P = pd.concat(preds, ignore_index=True)

    by_h = pd.DataFrame([dict(model=m, horizon=h, n=len(g), **metrics(g["y"], g["pred"]))
                         for (m, h), g in P.groupby(["model", "horizon"])])
    comp = []
    for m, g in P.groupby("model"):
        raw = metrics(g["y"], g["pred_raw"])
        comp.append(dict(model=m, n=len(g), **metrics(g["y"], g["pred"]),
                         MAE_unclipped=raw["MAE"], RMSE_unclipped=raw["RMSE"]))
    comp = pd.DataFrame(comp).merge(val_df[["model", "val_MAE"]], on="model").sort_values("MAE")
    comp["selected_on_validation"] = comp["model"] == selected

    S = P[P["model"] == selected].copy()
    S["month"] = pd.to_datetime(S["target_date"]).dt.month
    hi_thr = float(pd.concat([splits[h][0]["y"] for h in horizons]).quantile(0.90))
    hi = S[S["y"] >= hi_thr]
    by_city, by_month = group_metrics(S, "city"), group_metrics(S, "month")
    best_base = comp[comp["model"].isin([b for b, _ in BASELINES])].iloc[0]
    sel = comp[comp["model"] == selected].iloc[0]
    test_metrics = dict(
        selected_model=selected, test_split_from=str(test_start.date()),
        selected=dict(MAE=sel.MAE, RMSE=sel.RMSE, R2=sel.R2, sMAPE=sel.sMAPE),
        best_baseline=dict(name=best_base.model, MAE=best_base.MAE, RMSE=best_base.RMSE, R2=best_base.R2),
        beats_baseline_on_MAE=bool(sel.MAE < best_base.MAE),
        high_pollution=dict(threshold=hi_thr, n=len(hi), **(metrics(hi["y"], hi["pred"]) if len(hi) > 1 else {})),
        nonneg_clipping_note="Predictions clipped at 0; unclipped metrics are in model_comparison.csv",
    )

    # ---- refit on train+validation (test untouched) and persist
    final = {h: make_model(selected, best_cfg[selected]).fit(design(pd.concat(splits[h][:2])),
                                                             pd.concat(splits[h][:2])["y"]) for h in horizons}
    metadata = dict(
        model=selected, params=best_cfg[selected], target="PM2.5 (µg/m³)", strategy="direct, one model per horizon",
        horizons=horizons, train_range=[str(df.date.min().date()), str(test_start.date())],
        note="Final models refit on train+validation; test metrics come from the train-only fit.",
        trained_at=pd.Timestamp.now().isoformat(),
        source="Kaggle rohanrao/air-quality-data-in-india (city_day.csv)",
        weather="Open-Meteo archive, origin-day observations only" if weather else "none",
        weather_cities=sorted(weather), validation_MAE=float(val_df[val_df.model == selected].val_MAE.iloc[0]),
        test_metrics=test_metrics,
        versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                      **_lib_versions()),
    )
    if save:
        for d in (paths.models, paths.reports, paths.figures):
            d.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, paths.model_file)
        paths.schema_file.write_text(json.dumps(dict(features=features, base_features=base, cities=cities), indent=2))
        paths.metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
        comp.to_csv(paths.reports / "model_comparison.csv", index=False)
        by_h.to_csv(paths.reports / "metrics_by_horizon.csv", index=False)
        by_city.to_csv(paths.reports / "metrics_by_city.csv", index=False)
        by_month.to_csv(paths.reports / "metrics_by_month.csv", index=False)
        (paths.reports / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, default=float))
        save_figures(S, by_h, paths.figures, selected)
        if shap_artifacts:
            try:
                from .explainability import save_global_shap

                te = splits[1][2]
                save_global_shap(train_only[selected][1], design(te), paths.reports, paths.figures)
            except Exception as exc:  # noqa: BLE001
                log.warning("SHAP artifacts skipped: %s", exc)
    return TrainResult(selected, comp, test_metrics, metadata)


def _lib_versions() -> dict:
    out = {}
    for mod in ("sklearn", "xgboost", "lightgbm", "joblib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out
