"""Download city_day.csv from Kaggle. Needs KAGGLE_API_TOKEN (or ~/.kaggle/kaggle.json)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

raw = config.get_paths().data_raw
raw.mkdir(parents=True, exist_ok=True)
cmd = ["kaggle", "datasets", "download", "-d", "rohanrao/air-quality-data-in-india", "-p", str(raw), "--unzip"]
try:
    subprocess.run(cmd, check=True)
except FileNotFoundError:
    sys.exit("The `kaggle` CLI is not installed. Run: python -m pip install -U kaggle")
except subprocess.CalledProcessError:
    sys.exit("Kaggle download failed. Check your token and that you accepted the dataset terms.\n"
             f"Manual option: download from https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india and place city_day.csv in {raw}")
print("Saved to", raw)
