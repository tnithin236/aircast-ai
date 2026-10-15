import numpy as np
import pandas as pd
import pytest

from src.features import (FeatureError, base_feature_names, build_city_horizon_frame, city_origin_features,
                          make_design, split_frame)


def city_df(n=60):
    return pd.DataFrame({"city": "A", "date": pd.date_range("2020-01-01", periods=n), "pm25": np.arange(n, dtype=float) + 1})


def test_lags_use_the_past():
    g = city_df()
    f, s = city_origin_features(g, "A")
    t = pd.Timestamp("2020-02-01")
    assert f.loc[t, "lag_1"] == s[t - pd.Timedelta(days=1)]
    assert f.loc[t, "lag_7"] == s[t - pd.Timedelta(days=7)]


def test_features_at_origin_ignore_future_values():
    g = city_df()
    f1, _ = city_origin_features(g, "A")
    g2 = g.copy()
    g2.loc[g2.date > "2020-02-01", "pm25"] = 9999.0
    f2, _ = city_origin_features(g2, "A")
    t = pd.Timestamp("2020-02-01")
    pd.testing.assert_series_equal(f1.loc[t].drop("city"), f2.loc[t].drop("city"), check_names=False)


def test_target_alignment_per_horizon():
    g = city_df()
    for h in (1, 4, 7):
        fr = build_city_horizon_frame(g, "A", None, h)
        row = fr.iloc[10]
        assert row["target_date"] - row["origin"] == pd.Timedelta(days=h)
        assert row["y"] == row["pm25_t"] + h  # pm25 rises by 1 per day in this fixture


def test_missing_target_rows_are_dropped_not_filled():
    g = city_df()
    g.loc[30, "pm25"] = np.nan
    fr = build_city_horizon_frame(g, "A", None, 1)
    assert fr["y"].notna().all() and fr["pm25_t"].notna().all()
    assert g.loc[30, "date"] not in set(fr["target_date"])


def test_feature_ordering_and_validation():
    g = city_df()
    fr = build_city_horizon_frame(g, "A", None, 3)
    base = base_feature_names()
    feats = base + ["city_A", "city_B"]
    X = make_design(fr, feats, base, ["A", "B"])
    assert list(X.columns) == feats
    assert (X["city_B"] == 0).all()
    with pytest.raises(FeatureError):
        make_design(fr.drop(columns=["lag_7"]), feats, base, ["A", "B"])


def test_split_is_chronological_with_purge_gap():
    g = city_df(200)
    fr = build_city_horizon_frame(g, "A", None, 7)
    val_start, test_start = fr["origin"].iloc[100], fr["origin"].iloc[150]
    tr, va, te = split_frame(fr, val_start, test_start)
    assert tr["target_date"].max() < val_start + pd.Timedelta(days=1)
    assert va["target_date"].max() < test_start + pd.Timedelta(days=1)
    assert tr["origin"].max() < va["origin"].min() < te["origin"].min()
