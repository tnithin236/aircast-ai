import numpy as np
import pandas as pd
import pytest

from src.forecast import ModelNotFoundError, UnsupportedCityError, backtest_city, forecast_city, load_artifacts


@pytest.fixture(scope="module")
def art(trained_env):
    return load_artifacts(trained_env["paths"].models)


def test_artifacts_saved(trained_env):
    p = trained_env["paths"]
    assert p.model_file.exists() and p.metadata_file.exists() and p.schema_file.exists()
    assert (p.reports / "model_comparison.csv").exists() and (p.reports / "test_metrics.json").exists()


def test_forecast_shape_and_horizons(trained_env, art):
    df, wx = trained_env["df"], trained_env["wx"]
    res = forecast_city("TestCityA", df, art, wx["TestCityA"], today=pd.Timestamp("2018-06-01"))
    f = res.forecast
    assert len(f) == 7 and list(f["horizon"]) == list(range(1, 8))
    assert pd.to_datetime(f["forecast_date"]).is_monotonic_increasing
    assert (pd.to_datetime(f["forecast_date"]) > pd.Timestamp(res.origin_date)).all()
    assert set(f["unit"]) == {"µg/m³"}


def test_predictions_non_negative_and_raw_reported(trained_env, art):
    df, wx = trained_env["df"], trained_env["wx"]
    f = forecast_city("TestCityB", df, art, wx["TestCityB"]).forecast
    assert (f["pm25_pred"] >= 0).all()
    assert (f["pm25_pred"] == np.maximum(f["pm25_pred_raw"], 0)).all()
    assert f["clipped"].dtype == bool


def test_forecast_does_not_use_future_observations(trained_env, art):
    df, wx = trained_env["df"], trained_env["wx"]
    cutoff = pd.Timestamp("2019-03-01")
    full = forecast_city("TestCityA", df, art, wx["TestCityA"], as_of=cutoff).forecast
    truncated = forecast_city("TestCityA", df[df.date <= cutoff], art, wx["TestCityA"]).forecast
    tampered = df.copy()
    tampered.loc[tampered.date > cutoff, "pm25"] = 9999.0
    tam = forecast_city("TestCityA", tampered, art, wx["TestCityA"], as_of=cutoff).forecast
    np.testing.assert_allclose(full["pm25_pred_raw"], truncated["pm25_pred_raw"])
    np.testing.assert_allclose(full["pm25_pred_raw"], tam["pm25_pred_raw"])


def test_multi_step_forecast_differs_by_horizon(trained_env, art):
    df, wx = trained_env["df"], trained_env["wx"]
    f = forecast_city("TestCityA", df, art, wx["TestCityA"]).forecast
    assert f["pm25_pred"].nunique() > 1


def test_unsupported_city(trained_env, art):
    with pytest.raises(UnsupportedCityError):
        forecast_city("Atlantis", trained_env["df"], art)


def test_missing_model_files(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_artifacts(tmp_path)


def test_missing_weather_gives_warning(trained_env, art):
    res = forecast_city("TestCityA", trained_env["df"], art, None)
    assert any("Weather" in w for w in res.warnings)


def test_backtest_is_out_of_sample(trained_env, art):
    bt = backtest_city("TestCityA", trained_env["df"], art, trained_env["wx"]["TestCityA"], 3)
    test_start = pd.Timestamp(art.metadata["train_range"][1])
    assert len(bt) > 0 and (pd.to_datetime(bt["origin"]) >= test_start).all()
    assert (bt["pred"] >= 0).all()
