"""CheXpert frontal-image dataset built from the Phase 1 split manifest.

* Labels are read from the ORIGINAL CSV and mapped in memory by the label policy;
  the CSV is never rewritten.
* Every sample yields (image, targets[14], mask[14]); mask == False entries must
  contribute zero loss and zero gradient.
* The locked test split cannot be loaded without an explicit unlock flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.classification.labels import LABELS, assert_matches
from src.data.chexpert import load_chexpert_csv
from src.data.label_policies import LabelPolicy, apply_label_policy
from src.utils.config import PROJECT_ROOT, DatasetPaths

_UNCERTAIN = {"ignore": "mask", "u_ignore": "mask", "zero": "zero", "u_zero": "zero", "one": "one", "u_one": "one"}
_BLANK = {"zero": "zero", "ignore": "mask"}


class LockedSplitError(PermissionError):
    """Raised when code tries to read the locked test partition."""


def label_policy_from_config(cfg: dict) -> LabelPolicy:
    try:
        return LabelPolicy(uncertain=_UNCERTAIN[cfg["uncertain"]], missing=_BLANK[cfg["blank"]])
    except KeyError as e:
        raise ValueError(f"unknown label policy value {e}; config={cfg}") from None


def load_split_frame(paths: DatasetPaths, data_cfg: dict, split: str, allow_locked: bool = False) -> pd.DataFrame:
    """Rows of one partition: Path, patient_id, raw labels (1/0/-1/NaN) in LABELS order."""
    if split in data_cfg.get("locked_splits", []) and not allow_locked:
        raise LockedSplitError(f"split '{split}' is locked; it may only be opened for the final approved evaluation")
    manifest = pd.read_csv(PROJECT_ROOT / data_cfg["manifest"])
    rows = manifest[manifest["split"] == split]
    if data_cfg.get("view"):
        rows = rows[rows["Frontal/Lateral"] == data_cfg["view"]]
    excl_path = data_cfg.get("exclusions")
    n_excluded = 0
    if excl_path:
        ex = pd.read_csv(PROJECT_ROOT / excl_path)
        drop = set(ex.loc[ex["recommendation"] == "exclude", "Path"])
        n_excluded = int(rows["Path"].isin(drop).sum())
        rows = rows[~rows["Path"].isin(drop)]
    source = paths.chexpert_valid_csv if split == "official_valid" else paths.chexpert_train_csv
    labels, targets = load_chexpert_csv(source, split)
    assert_matches(targets)
    df = rows[["Path", "patient_id", "study_id", "AP/PA"]].merge(labels[["Path", *LABELS]], on="Path", how="left",
                                                                  validate="one_to_one")
    if len(df) != len(rows):
        raise RuntimeError("label merge changed the row count")
    df = df.reset_index(drop=True)
    df.attrs["n_excluded"] = n_excluded
    return df


class CheXpertDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_base: Path, policy: LabelPolicy, transform):
        self.paths = frame["Path"].tolist()
        self.image_base = Path(image_base)
        raw = frame[list(LABELS)].to_numpy(dtype=np.float64)
        self.raw_labels = raw  # untouched copy for reporting
        t, m = apply_label_policy(raw, policy)
        self.targets = torch.from_numpy(t.astype(np.float32))
        self.mask = torch.from_numpy(m)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.image_base / self.paths[i]) as img:
            img.load()
            x = self.transform(img)
        return x, self.targets[i], self.mask[i], i


def seed_worker(worker_id: int) -> None:
    """Deterministic per-worker RNG (torch seeds each worker from the loader generator)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    import random
    random.seed(s)
