"""Print reports/model_comparison.csv as a Markdown table (paste into the README)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

p = config.get_paths().reports / "model_comparison.csv"
if not p.exists():
    sys.exit(f"{p} not found. Copy reports/ from Colab or run scripts/train_model.py.")
df = pd.read_csv(p)[["model", "n", "MAE", "RMSE", "R2", "sMAPE", "val_MAE", "selected_on_validation"]].round(3)
print("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns))
for r in df.itertuples(index=False):
    print("| " + " | ".join(str(x) for x in r) + " |")
