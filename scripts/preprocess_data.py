"""Clean data/raw/city_day.csv into data/processed/city_day_clean.csv."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, preprocessing  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
p = argparse.ArgumentParser()
p.add_argument("--min-valid-days", type=int, default=1000)
a = p.parse_args()
paths = config.get_paths()
src = paths.data_raw / "city_day.csv"
if not src.exists():
    sys.exit(f"{src} not found. Run scripts/download_data.py or place the file there.")
df, report = preprocessing.preprocess_raw(src, paths.processed_csv, a.min_valid_days)
print(json.dumps(report, indent=2, default=str))
print(preprocessing.detect_gaps(df).to_string(index=False))
