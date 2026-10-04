"""Lossless prediction storage.

Two artifacts per prediction set:
  <stem>.npz      canonical machine artifact: float64 probabilities (bit-exact),
                  raw labels, validity mask, image paths, label order
  <stem>.csv.gz   human-readable table, floats written with 17 significant
                  digits (sufficient for an exact float64 round trip)

A 16-digit CSV (pandas default) is NOT lossless and changed threshold statuses
for scores lying exactly on a Youden threshold; see the C1 verification report.
Reading must also be exact: pandas' default float parser can be 1 ULP off, so the
CSV is read with float_precision="round_trip".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.classification.labels import LABELS, assert_matches

FLOAT_FORMAT = "%.17g"


def save_predictions(out_dir: Path, stem: str, paths, patient_ids, probs: np.ndarray, raw_labels: np.ndarray,
                     valid: np.ndarray, meta: dict | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(paths), len(LABELS)):
        raise ValueError(f"probs shape {probs.shape} != ({len(paths)}, {len(LABELS)})")
    npz = out_dir / f"{stem}.npz"
    np.savez_compressed(npz, probs=probs, raw_labels=np.asarray(raw_labels, dtype=np.float64),
                        valid=np.asarray(valid, dtype=bool), paths=np.asarray(list(paths), dtype=str),
                        patient_ids=np.asarray(list(patient_ids), dtype=str), labels=np.asarray(LABELS, dtype=str),
                        meta=np.asarray(repr(meta or {}), dtype=str))
    df = pd.DataFrame({"Path": list(paths), "patient_id": list(patient_ids)})
    for j, lab in enumerate(LABELS):
        df[f"prob::{lab}"] = probs[:, j]
        df[f"label_raw::{lab}"] = raw_labels[:, j]
        df[f"valid::{lab}"] = np.asarray(valid)[:, j]
    csv = out_dir / f"{stem}.csv.gz"
    df.to_csv(csv, index=False, float_format=FLOAT_FORMAT, compression={"method": "gzip", "mtime": 0})
    return {"npz": npz, "csv": csv}


def load_predictions_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    assert_matches(list(d["labels"]))
    return {"probs": d["probs"], "raw_labels": d["raw_labels"], "valid": d["valid"],
            "paths": d["paths"].tolist(), "patient_ids": d["patient_ids"].tolist()}


def load_predictions_csv(path: Path) -> dict:
    # pandas' default C float parser is not correctly rounded (can be 1 ULP off even for
    # 17-digit input); "round_trip" uses Python's exact parser.
    df = pd.read_csv(path, float_precision="round_trip")
    return {"probs": df[[f"prob::{l}" for l in LABELS]].to_numpy(dtype=np.float64),
            "raw_labels": df[[f"label_raw::{l}" for l in LABELS]].to_numpy(dtype=np.float64),
            "valid": df[[f"valid::{l}" for l in LABELS]].to_numpy().astype(bool),
            "paths": df["Path"].tolist(), "patient_ids": df["patient_id"].tolist()}


def roundtrip_report(original: np.ndarray, reloaded: np.ndarray, thresholds: np.ndarray) -> dict:
    """Max absolute difference and number of `p >= threshold` status changes after reload."""
    original, reloaded = np.asarray(original, dtype=np.float64), np.asarray(reloaded, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    return {"max_abs_diff": float(np.abs(original - reloaded).max()),
            "bit_identical": bool(np.array_equal(original, reloaded)),
            "status_mismatches": int(((original >= t) != (reloaded >= t)).sum())}
