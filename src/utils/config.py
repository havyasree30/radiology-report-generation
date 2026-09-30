"""Project configuration loading (dataset paths and analysis settings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
EDA_TABLES = EDA_DIR / "tables"
EDA_FIGURES = EDA_DIR / "figures"
EDA_SUMMARIES = EDA_DIR / "summaries"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

_PATH_KEYS = (
    "chexpert_root",
    "chexpert_train_csv",
    "chexpert_valid_csv",
    "chexpert_image_base",
    "iu_xray_root",
    "iu_reports_csv",
    "iu_projections_csv",
    "iu_images_dir",
)


@dataclass(frozen=True)
class DatasetPaths:
    chexpert_root: Path
    chexpert_train_csv: Path
    chexpert_valid_csv: Path
    chexpert_image_base: Path
    iu_xray_root: Path
    iu_reports_csv: Path
    iu_projections_csv: Path
    iu_images_dir: Path

    def as_dict(self) -> dict[str, str]:
        return {k: str(getattr(self, k)) for k in _PATH_KEYS}


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_paths(config_file: Path | None = None) -> DatasetPaths:
    """Load dataset locations from ``configs/paths.yaml``.

    Raises a clear error when the file is missing or still contains template
    placeholders, instead of silently falling back to guessed locations.
    """
    config_file = config_file or CONFIG_DIR / "paths.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"{config_file} not found. Copy configs/paths.example.yaml to "
            "configs/paths.yaml and fill in the dataset locations."
        )
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    missing = [k for k in _PATH_KEYS if not raw.get(k)]
    if missing:
        raise KeyError(f"{config_file} is missing keys: {missing}")
    placeholders = [k for k in _PATH_KEYS if str(raw[k]).startswith("PATH_TO")]
    if placeholders:
        raise ValueError(f"{config_file} still has template placeholders for: {placeholders}")
    return DatasetPaths(**{k: _resolve(str(raw[k])) for k in _PATH_KEYS})


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file from ``configs/`` by file name."""
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8")) or {}


def ensure_output_dirs() -> None:
    for d in (EDA_TABLES, EDA_FIGURES, EDA_SUMMARIES, SPLITS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
