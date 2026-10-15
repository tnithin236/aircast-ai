import pandas as pd
import pytest

from src.data_loader import SchemaError, load_processed, load_raw_csv, validate_schema
from src.preprocessing import clean_air_quality, detect_gaps


def raw_frame():
    return pd.DataFrame({"City": ["A", "A", "A", "B"], "Date": ["2020-01-01", "2020-01-01", "2020-01-03", "2020-01-01"],
                         "PM2.5": [10.0, 20.0, -5.0, None], "PM10": [20, 30, 40, 50], "NO2": [1, 2, 3, 4]})


def test_load_raw_csv_and_missing_file(tmp_path):
    p = tmp_path / "x.csv"
    raw_frame().to_csv(p, index=False)
    assert len(load_raw_csv(p)) == 4
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "nope.csv")


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"City": ["A"], "Date": ["2020-01-01"]}).to_csv(p, index=False)
    with pytest.raises(SchemaError, match="PM2.5"):
        load_raw_csv(p)
    with pytest.raises(SchemaError):
        validate_schema(pd.DataFrame({"a": [1]}), ["city"])


def test_date_parsing_drops_bad_dates():
    raw = raw_frame()
    raw.loc[3, "Date"] = "not-a-date"
    df, rep = clean_air_quality(raw)
    assert rep["bad_dates_dropped"] == 1
    assert "B" not in set(df["city"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_duplicates_averaged_negatives_nan_and_missing_target_not_imputed():
    df, rep = clean_air_quality(raw_frame())
    assert rep["duplicate_city_dates_averaged"] == 1
    a = df[df.city == "A"].set_index("date")["pm25"]
    assert a[pd.Timestamp("2020-01-01")] == 15.0
    assert pd.isna(a[pd.Timestamp("2020-01-03")])  # negative became NaN, not filled
    assert rep["pm25_negative_to_nan"] == 1
    assert df[df.city == "B"]["pm25"].isna().all()


def test_extreme_values_are_kept():
    raw = raw_frame()
    raw.loc[0, "PM2.5"] = 999.0
    df, _ = clean_air_quality(raw)
    assert df["pm25"].max() >= 500


def test_detect_gaps():
    df, _ = clean_air_quality(raw_frame())
    gaps = detect_gaps(df).set_index("city")
    assert gaps.loc["A", "missing_days"] == 2 and gaps.loc["A", "longest_gap_days"] == 2


def test_load_processed(tmp_path):
    p = tmp_path / "p.csv"
    pd.DataFrame({"city": ["A"], "date": ["2020-01-01"], "pm25": [1.0], "pm10": [2.0]}).to_csv(p, index=False)
    assert load_processed(p)["date"].iloc[0] == pd.Timestamp("2020-01-01")
