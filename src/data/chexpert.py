"""CheXpert metadata loading with label states preserved exactly as released.

Label states (never collapsed in this module):
    1.0  positive      0.0  negative      -1.0  uncertain      NaN  unmentioned
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.audit import AuditLog

POSITIVE, NEGATIVE, UNCERTAIN = 1.0, 0.0, -1.0
VALID_LABEL_VALUES = frozenset({POSITIVE, NEGATIVE, UNCERTAIN})
LABEL_STATES = ("positive", "negative", "uncertain", "missing")

# Columns that describe the image rather than an observation.
METADATA_COLUMNS = ("Path", "Sex", "Age", "Frontal/Lateral", "AP/PA")

# The 14 CheXpert observations. Used to *validate* the detected targets, not to
# replace detection: detection works from the actual CSV header and values.
EXPECTED_OBSERVATIONS = (
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
)

_PATH_RE = re.compile(
    r"(?P<patient>patient\d+)/(?P<study>study\d+)/view(?P<view_idx>\d+)_(?P<view>[A-Za-z]+)\.(?P<ext>\w+)$"
)


def detect_target_columns(df: pd.DataFrame, audit: AuditLog | None = None) -> list[str]:
    """Return observation columns in CSV order.

    A column is a target if it is not metadata and every non-null value is
    one of {-1, 0, 1} after numeric parsing. Non-metadata columns failing that
    test are reported (not silently dropped or coerced).
    """
    targets = []
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        non_numeric = df[col].notna() & values.isna()
        observed = set(values.dropna().unique().tolist())
        if non_numeric.any() or not observed <= VALID_LABEL_VALUES:
            if audit is not None:
                audit.add("non_label_column", col,
                          f"values={sorted(map(str, observed))[:10]} non_numeric={int(non_numeric.sum())}")
            continue
        targets.append(col)
    if audit is not None:
        unexpected = sorted(set(targets) - set(EXPECTED_OBSERVATIONS))
        absent = sorted(set(EXPECTED_OBSERVATIONS) - set(targets))
        if unexpected:
            audit.add("unexpected_target_column", None, str(unexpected))
        if absent:
            audit.add("expected_target_missing", None, str(absent), severity="error")
    return targets


def invalid_label_entries(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rows/columns whose raw value is not one of {-1, 0, 1, blank}."""
    rows = []
    for col in columns:
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce")
        bad = (raw.notna() & num.isna()) | (num.notna() & ~num.isin(list(VALID_LABEL_VALUES)))
        for idx in raw.index[bad]:
            rows.append({"row": idx, "column": col, "value": raw.loc[idx]})
    return pd.DataFrame(rows, columns=["row", "column", "value"])


def parse_path(path: str) -> dict[str, object]:
    """Extract patient / study / view identifiers from a CheXpert image path."""
    m = _PATH_RE.search(str(path).replace("\\", "/"))
    if m is None:
        return {"patient_id": None, "study_id": None, "view_index": None,
                "view_from_path": None, "extension": Path(str(path)).suffix.lower().lstrip(".") or None}
    return {
        "patient_id": m["patient"],
        # Study numbers restart for every patient, so a study is only unique
        # when qualified by its patient.
        "study_id": f"{m['patient']}/{m['study']}",
        "view_index": int(m["view_idx"]),
        "view_from_path": m["view"].lower(),
        "extension": m["ext"].lower(),
    }


def load_chexpert_csv(csv_path: Path, source_split: str, audit: AuditLog | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load one CheXpert CSV without altering any label value.

    Adds derived identifier columns; targets are cast to float so that the
    four states (1, 0, -1, NaN) remain distinct.
    """
    audit = audit or AuditLog("chexpert")
    if not csv_path.exists():
        raise FileNotFoundError(f"CheXpert CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    if df.empty:
        raise ValueError(f"CheXpert CSV is empty: {csv_path}")
    if "Path" not in df.columns:
        raise ValueError(f"{csv_path} has no 'Path' column; columns={list(df.columns)}")

    targets = detect_target_columns(df, audit)
    bad = invalid_label_entries(df, targets)
    for rec in bad.itertuples():
        audit.add("invalid_label_value", str(df.at[rec.row, "Path"]), f"{rec.column}={rec.value!r}", severity="error")
    for col in targets:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce") if "Age" in df.columns else np.nan
    parsed = pd.DataFrame([parse_path(p) for p in df["Path"]], index=df.index)
    df = pd.concat([df, parsed], axis=1)
    df["source_split"] = source_split

    for idx in df.index[df["patient_id"].isna()]:
        audit.add("unparseable_path", str(df.at[idx, "Path"]), "patient/study not extractable", severity="error")
    dup = df["Path"].duplicated(keep=False)
    for p in df.loc[dup, "Path"].unique():
        audit.add("duplicate_path_reference", str(p), f"{int((df['Path'] == p).sum())} rows")
    return df, targets


def label_state_frame(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Per-observation counts of the four label states (no conversion)."""
    n = len(df)
    rows = []
    for col in targets:
        v = df[col]
        counts = {
            "positive": int((v == POSITIVE).sum()),
            "negative": int((v == NEGATIVE).sum()),
            "uncertain": int((v == UNCERTAIN).sum()),
            "missing": int(v.isna().sum()),
        }
        row = {"observation": col, "n_images": n}
        for state, c in counts.items():
            row[f"{state}_count"] = c
            row[f"{state}_pct"] = 100.0 * c / n if n else np.nan
        row["other_count"] = n - sum(counts.values())
        rows.append(row)
    return pd.DataFrame(rows)


def positive_matrix(df: pd.DataFrame, targets: list[str]) -> np.ndarray:
    """Boolean matrix: True where the label is explicitly positive (== 1)."""
    return (df[targets].to_numpy() == POSITIVE)
