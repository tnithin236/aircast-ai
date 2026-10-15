"""Download daily Open-Meteo weather for every city in the processed data (cached in data/raw)."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data_loader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
df = data_loader.load_processed()
start, end = df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")
ok = 0
for city in sorted(df["city"].unique()):
    w = data_loader.fetch_weather(city, start, end, config.get_paths().data_raw)
    ok += w is not None
    print(f"{city}: {'ok' if w is not None else 'FAILED'}")
    time.sleep(1)
print(f"Weather ready for {ok} cities | source: Open-Meteo archive | tz: {data_loader.WEATHER_TZ} | collected: {time.strftime('%Y-%m-%d')}")
