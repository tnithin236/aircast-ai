"""SHAP explanations for the saved tree models. All entry points fail soft."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Explanation:
    base_value: float
    prediction: float
    table: pd.DataFrame  # feature, value, shap_value, effect


def _unwrap(model):
    """Return (preprocessing step or None, tree estimator) for a plain model or sklearn Pipeline."""
    if hasattr(model, "steps"):
        return model[:-1], model.steps[-1][1]
    return None, model


def _transform(pre, X: pd.DataFrame) -> pd.DataFrame:
    if pre is None:
        return X
    arr = pre.transform(X)
    try:
        cols = list(pre.get_feature_names_out(X.columns))
    except Exception:  # noqa: BLE001
        cols = list(X.columns)
    return pd.DataFrame(arr, columns=cols if len(cols) == arr.shape[1] else None, index=X.index)


def explain_row(model, X_row: pd.DataFrame) -> Explanation | None:
    """Contribution of each feature to one prediction, relative to the model's base value."""
    try:
        import shap

        pre, est = _unwrap(model)
        Xs = _transform(pre, X_row)
        explainer = shap.TreeExplainer(est)
        sv = np.asarray(explainer.shap_values(Xs)).reshape(len(Xs), -1)[0]
        base = float(np.ravel(explainer.expected_value)[0])
        table = pd.DataFrame({"feature": Xs.columns, "value": Xs.iloc[0].to_numpy(), "shap_value": sv})
        table["effect"] = np.where(table["shap_value"] >= 0, "raises forecast", "lowers forecast")
        table = table.reindex(table["shap_value"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return Explanation(base, base + float(sv.sum()), table)
    except Exception as exc:  # noqa: BLE001
        log.warning("SHAP explanation unavailable: %s", exc)
        return None


def plain_english(exp: Explanation, top: int = 3) -> str:
    """Short description of the main drivers. Contributions are associations, not causes."""
    parts = []
    for _, r in exp.table.head(top).iterrows():
        direction = "pushes the forecast up" if r.shap_value > 0 else "pulls the forecast down"
        parts.append(f"**{r.feature}** (value {r.value:.2f}) {direction} by about {abs(r.shap_value):.1f} µg/m³")
    return ("Starting from the model's average output of "
            f"{exp.base_value:.1f} µg/m³, " + "; ".join(parts) + ". "
            "These are statistical contributions of the model, not proof of cause and effect.")


def save_global_shap(model, X: pd.DataFrame, reports_dir: Path, fig_dir: Path, sample: int = 1500, seed: int = 42) -> pd.DataFrame | None:
    """Write the summary plot and mean |SHAP| table for a (day-1) model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    pre, est = _unwrap(model)
    Xs = _transform(pre, X.sample(min(sample, len(X)), random_state=seed))
    sv = np.asarray(shap.TreeExplainer(est).shap_values(Xs))
    shap.summary_plot(sv, Xs, show=False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_h1.png", dpi=150, bbox_inches="tight")
    plt.close()
    imp = (pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(sv).mean(0)})
           .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(Path(reports_dir) / "shap_importance_h1.csv", index=False)
    return imp
