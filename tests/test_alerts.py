import pandas as pd

from src.alerts import AlertConfig, AlertHistory, cpcb_pm25_category, evaluate_alerts


def fc(values, city="A", origin="2020-01-01"):
    o = pd.Timestamp(origin)
    return pd.DataFrame({"city": city, "origin_date": o.date(), "horizon": range(1, len(values) + 1),
                         "forecast_date": [(o + pd.Timedelta(days=i + 1)).date() for i in range(len(values))],
                         "pm25_pred": values, "model": "M", "generated_at": "2020-01-01T00:00:00"})


def kinds(alerts):
    return [a.alert_type for a in alerts]


def test_cpcb_categories():
    assert [cpcb_pm25_category(v) for v in (10, 30, 45, 75, 100, 200, 300)] == [
        "Good", "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def test_threshold_exceeded_and_severity():
    alerts = evaluate_alerts(fc([50, 70, 130, 40, 40, 40, 40]), AlertConfig(threshold=60, high_threshold=120),
                             today=pd.Timestamp("2020-01-02"))
    thr = [a for a in alerts if a.alert_type == "forecast_threshold_exceeded"]
    assert len(thr) == 2
    assert {a.severity for a in thr} == {"watch", "high"}
    assert all(a.threshold == 60 and a.location == "A" and a.model == "M" for a in thr)


def test_no_alerts_below_threshold():
    assert evaluate_alerts(fc([10] * 7), today=pd.Timestamp("2020-01-02"), latest_obs_date=pd.Timestamp("2020-01-01")) == []


def test_peak_and_sustained():
    a = evaluate_alerts(fc([20, 70, 80, 95, 20, 20, 20]), AlertConfig(threshold=60, peak_threshold=90, sustained_days=3),
                        today=pd.Timestamp("2020-01-02"))
    assert "predicted_peak" in kinds(a)
    s = [x for x in a if x.alert_type == "sustained_exceedance"]
    assert len(s) == 1 and s[0].forecast_date == "2020-01-03"
    assert "sustained_exceedance" not in kinds(evaluate_alerts(fc([70, 70, 10, 70, 70, 10, 10]), AlertConfig(sustained_days=3),
                                                               today=pd.Timestamp("2020-01-02")))


def test_stale_warning():
    a = evaluate_alerts(fc([10] * 7), latest_obs_date=pd.Timestamp("2020-01-01"), today=pd.Timestamp("2020-06-01"))
    assert "stale_observation" in kinds(a)
    a = evaluate_alerts(fc([10] * 7), latest_obs_date=pd.Timestamp("2020-01-01"), today=pd.Timestamp("2020-01-02"))
    assert "stale_observation" not in kinds(a)


def test_forecast_quality_warnings():
    a = evaluate_alerts(fc([10] * 7), today=pd.Timestamp("2020-01-02"),
                        quality={"beats_baseline": False, "horizon_mae": {7: 45.0}}, config=AlertConfig(quality_mae_limit=30))
    assert kinds(a).count("forecast_quality") == 2


def test_ids_stable_and_history_deduplicates(tmp_path):
    cfg = AlertConfig()
    a1 = evaluate_alerts(fc([90] * 7), cfg, today=pd.Timestamp("2020-01-02"))
    a2 = evaluate_alerts(fc([90] * 7), cfg, today=pd.Timestamp("2020-01-02"))
    assert [x.alert_id for x in a1] == [x.alert_id for x in a2]
    h = AlertHistory(tmp_path / "h.csv")
    assert h.append_new(a1) == len(a1) > 0
    assert h.append_new(a2) == 0
