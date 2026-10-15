"""Train all models, evaluate on the chronological test set, save artifacts. Not run by the app."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data_loader, train  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
p = argparse.ArgumentParser()
p.add_argument("--no-weather", action="store_true", help="train without weather features")
p.add_argument("--quick", action="store_true", help="single small RandomForest (smoke test only)")
a = p.parse_args()
df = data_loader.load_processed()
wx = {} if a.no_weather else data_loader.load_weather_many(sorted(df["city"].unique()))
if not a.no_weather and not wx:
    logging.warning("No cached weather found; training without it. Run scripts/download_weather.py first to include weather.")
res = train.run_training(df, wx, grids=train.QUICK_GRIDS if a.quick else None)
print(res.comparison.round(3).to_string(index=False))
print("Selected on validation:", res.selected, "| beats baseline on test MAE:", res.test_metrics["beats_baseline_on_MAE"])
print("Artifacts written to", config.get_paths().models, "and", config.get_paths().reports)
