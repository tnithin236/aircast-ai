"""Shared fixtures. All data here is SYNTHETIC and exists only for unit tests.
It is never used as training data for the real project and must not be displayed as real pollution."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.train import QUICK_GRIDS, run_training  # noqa: E402


def make_synthetic(n_days: int = 500, cities=("TestCityA", "TestCityB"), seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    rows, wx = [], {}
    for k, city in enumerate(cities):
        t = np.arange(n_days)
        base = 60 + 30 * np.sin(2 * np.pi * t / 365) + 10 * k
        ar = np.zeros(n_days)
        for i in range(1, n_days):
            ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 8)
        pm25 = np.clip(base + ar, 5, None)
        rows.append(pd.DataFrame(dict(city=city, date=dates, pm25=pm25, pm10=pm25 * 1.6)))
        wx[city] = pd.DataFrame(dict(date=dates, temp=25 + 5 * np.sin(2 * np.pi * t / 365), rh=60 + rng.normal(0, 5, n_days),
                                     wind=10 + rng.normal(0, 2, n_days), precip=np.clip(rng.normal(0, 2, n_days), 0, None),
                                     pressure=1000 + rng.normal(0, 3, n_days)))
    return pd.concat(rows, ignore_index=True), wx


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic()


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory, synthetic):
    """Train tiny models on synthetic data into a temp dir and point config at it."""
    root = tmp_path_factory.mktemp("aircast")
    env = {"AIRCAST_MODELS_DIR": root / "models", "AIRCAST_REPORTS_DIR": root / "reports",
           "AIRCAST_DATA_PROCESSED": root / "processed", "AIRCAST_DATA_RAW": root / "raw"}
    mp = pytest.MonkeyPatch()
    for k, v in env.items():
        mp.setenv(k, str(v))
    df, wx = synthetic
    (root / "processed").mkdir()
    (root / "raw").mkdir()
    df.to_csv(root / "processed" / "city_day_clean.csv", index=False)
    for c, w in wx.items():
        w.to_csv(root / "raw" / f"weather_{c}.csv", index=False)
    result = run_training(df, wx, grids=QUICK_GRIDS, shap_artifacts=True)
    yield dict(root=root, result=result, df=df, wx=wx, paths=config.get_paths())
    mp.undo()
