"""IU X-Ray (Indiana University / Open-I, Kaggle CSV layout) loading.

Layout discovered locally:
    indiana_reports.csv      one row per report, keyed by ``uid``
    indiana_projections.csv  one row per image: uid, filename, projection
    images/images_normalized/<filename>

The report ``uid`` is the study unit: every image of a report shares it.
The CSVs contain no patient identifier, so patient-level grouping is not
possible for this dataset (documented limitation).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.audit import AuditLog

REPORT_TEXT_COLUMNS = ("findings", "impression")
ANONYMIZATION_TOKEN = "XXXX"
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z\d]")


def clean_section(value: object) -> str:
    """Normalise a section to a stripped string; NaN/None become ''."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return _WS.sub(" ", str(value)).strip()


def normalize_for_duplicates(text: str) -> str:
    return _WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def approx_token_count(text: str) -> int:
    """Regex token count (words, numbers, punctuation). An approximation only:
    it is not the tokenizer of any particular language model."""
    return len(_TOKEN.findall(text)) if text else 0


def _uid_key(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def load_reports(csv_path: Path, audit: AuditLog | None = None) -> pd.DataFrame:
    audit = audit or AuditLog("iu_xray")
    if not csv_path.exists():
        raise FileNotFoundError(f"IU reports CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    if df.empty:
        raise ValueError(f"IU reports CSV is empty: {csv_path}")
    if "uid" not in df.columns:
        raise ValueError(f"{csv_path} has no 'uid' column; columns={list(df.columns)}")
    df["uid"] = df["uid"].map(_uid_key)
    for idx in df.index[df["uid"].isna()]:
        audit.add("missing_report_uid", f"row{idx}", "", severity="error")
    for col in REPORT_TEXT_COLUMNS:
        if col not in df.columns:
            audit.add("missing_report_column", col, "", severity="error")
            df[col] = ""
        df[col] = df[col].map(clean_section)
    dup = df["uid"].duplicated(keep=False) & df["uid"].notna()
    for uid in df.loc[dup, "uid"].unique():
        audit.add("duplicate_report_uid", uid, f"{int((df['uid'] == uid).sum())} rows")
    return df


def load_projections(csv_path: Path, audit: AuditLog | None = None) -> pd.DataFrame:
    audit = audit or AuditLog("iu_xray")
    if not csv_path.exists():
        raise FileNotFoundError(f"IU projections CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    for col in ("uid", "filename", "projection"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} lacks column {col!r}; columns={list(df.columns)}")
    df["uid"] = df["uid"].map(_uid_key)
    df["projection"] = df["projection"].fillna("Unknown").str.strip()
    df["extension"] = df["filename"].str.extract(r"\.([A-Za-z0-9]+)$", expand=False).str.lower()
    for idx in df.index[df["uid"].isna()]:
        audit.add("projection_missing_uid", str(df.at[idx, "filename"]), "", severity="error")
    for fn in df.loc[df["filename"].duplicated(keep=False), "filename"].unique():
        audit.add("duplicate_image_reference", fn, "")
    known = {"Frontal", "Lateral"}
    for idx in df.index[~df["projection"].isin(known)]:
        audit.add("unknown_projection", str(df.at[idx, "filename"]), str(df.at[idx, "projection"]))
    return df


def section_status(findings: str, impression: str) -> str:
    f, i = bool(findings), bool(impression)
    if f and i:
        return "both"
    if f:
        return "findings_only"
    if i:
        return "impression_only"
    return "neither"


def combined_text(findings: str, impression: str) -> str:
    parts = [p for p in (findings, impression) if p]
    return " ".join(parts)
