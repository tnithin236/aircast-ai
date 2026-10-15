"""Central configuration. Paths can be overridden with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HORIZONS = list(range(1, 8))
TARGET_UNIT = "µg/m³"


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def processed_csv(self) -> Path:
        return self.data_processed / "city_day_clean.csv"

    @property
    def model_file(self) -> Path:
        return self.models / "pm25_models.joblib"

    @property
    def metadata_file(self) -> Path:
        return self.models / "model_metadata.json"

    @property
    def schema_file(self) -> Path:
        return self.models / "feature_schema.json"

    @property
    def alert_history(self) -> Path:
        return self.reports / "alert_history.csv"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def get_paths() -> Paths:
    """Resolve paths at call time so tests and deployments can override them."""
    return Paths(
        root=ROOT,
        data_raw=_env_path("AIRCAST_DATA_RAW", ROOT / "data" / "raw"),
        data_processed=_env_path("AIRCAST_DATA_PROCESSED", ROOT / "data" / "processed"),
        models=_env_path("AIRCAST_MODELS_DIR", ROOT / "models"),
        reports=_env_path("AIRCAST_REPORTS_DIR", ROOT / "reports"),
    )
