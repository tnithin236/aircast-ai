"""Metrics, report loading and evaluation figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def smape(y, p, eps: float = 1.0) -> float:
    """Symmetric MAPE in %. `eps` (µg/m³) in the denominator guards near-zero values."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + eps)) * 100)


def metrics(y, p) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(
        MAE=float(mean_absolute_error(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        R2=float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        sMAPE=smape(y, p),
    )


def group_metrics(df: pd.DataFrame, by: str, y: str = "y", p: str = "pred") -> pd.DataFrame:
    rows = [dict(**{by: k}, n=len(g), **metrics(g[y], g[p])) for k, g in df.groupby(by)]
    return pd.DataFrame(rows)


def load_reports(reports_dir: Path | None = None) -> dict:
    """Load saved evaluation artifacts. Missing files are returned as None."""
    d = Path(reports_dir) if reports_dir else config.get_paths().reports
    out: dict = {}
    for key, name in [("comparison", "model_comparison.csv"), ("by_horizon", "metrics_by_horizon.csv"),
                      ("by_city", "metrics_by_city.csv"), ("by_month", "metrics_by_month.csv"),
                      ("shap_importance", "shap_importance_h1.csv")]:
        p = d / name
        out[key] = pd.read_csv(p) if p.exists() else None
    p = d / "test_metrics.json"
    out["test_metrics"] = json.loads(p.read_text()) if p.exists() else None
    return out


def save_figures(S: pd.DataFrame, by_h: pd.DataFrame, fig_dir: Path, model_name: str) -> None:
    """Static evaluation figures for the selected model. S has columns city, target_date, y, pred, horizon."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("MAE", "RMSE"):
        plt.figure(figsize=(7, 4))
        for m, g in by_h.groupby("model"):
            plt.plot(g["horizon"], g[metric], marker="o", label=m)
        plt.xlabel("Forecast horizon (days)")
        plt.ylabel(f"{metric} (µg/m³)")
        plt.title(f"{metric} by horizon (test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric.lower()}_by_horizon.png", dpi=150)
        plt.close()
    city = S["city"].value_counts().index[0]
    for h in (1, 7):
        g = S[(S.city == city) & (S.horizon == h)].sort_values("target_date")
        plt.figure(figsize=(10, 3.5))
        plt.plot(g["target_date"], g["y"], label="Actual")
        plt.plot(g["target_date"], g["pred"], label="Predicted", alpha=0.8)
        plt.title(f"{city}: actual vs predicted PM2.5, horizon {h}d ({model_name})")
        plt.ylabel("µg/m³")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"actual_vs_pred_h{h}.png", dpi=150)
        plt.close()
    resid = S["y"] - S["pred"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].hist(resid, bins=60)
    ax[0].set_title("Residual distribution")
    g = S[S.horizon == 1].assign(resid=resid).groupby("target_date")["resid"].mean()
    ax[1].plot(g.index, g.values)
    ax[1].set_title("Mean residual over time (h=1)")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals.png", dpi=150)
    plt.close()
