"""Final operating policy (C4): per-class thresholds + the No Finding consistency rule.

Rule "suppress_if_any_abnormal_positive":
    No Finding is positive only if (a) its own score >= its threshold AND (b) none of the abnormal
    observations is predicted positive. Only the final binary No Finding output can change.
    Scores and the abnormal-class predictions are never modified.

"Abnormal" for the rule = the 12 pathology observations (every class except No Finding and
Support Devices). Support Devices is excluded because the source labels themselves allow No Finding
together with Support Devices (a device is not a pathology); the project's earlier label-consistency
analysis uses the same definition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS, assert_matches
from src.classification.operating_points import apply_thresholds

RULE_NAME = "suppress_if_any_abnormal_positive"
KNOWN_RULES = {RULE_NAME}


def apply_no_finding_rule(pred: np.ndarray, abnormal_labels=PATHOLOGY_LABELS) -> np.ndarray:
    """Return a copy of the (n, 14) boolean predictions with No Finding suppressed when any abnormal is positive."""
    pred = np.asarray(pred, dtype=bool)
    if pred.ndim != 2 or pred.shape[1] != len(LABELS):
        raise ValueError(f"expected (n, {len(LABELS)}) boolean predictions, got {pred.shape}")
    nf = LABELS.index(NO_FINDING)
    ab = [LABELS.index(l) for l in abnormal_labels]
    out = pred.copy()
    out[:, nf] = pred[:, nf] & ~pred[:, ab].any(axis=1)
    return out


def save_final_policy(path: Path, thresholds: np.ndarray, provenance: dict) -> None:
    t = np.asarray(thresholds, dtype=np.float64)
    if t.shape != (len(LABELS),) or not np.all(np.isfinite(t)):
        raise ValueError("need 14 finite thresholds")
    payload = {**provenance, "label_order": list(LABELS), "no_finding_rule": RULE_NAME,
               "no_finding_rule_abnormal_labels": list(PATHOLOGY_LABELS),
               "classes": {lab: {"threshold": float(t[i])} for i, lab in enumerate(LABELS)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_final_policy(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_matches(d["label_order"])
    if d["no_finding_rule"] not in KNOWN_RULES:
        raise ValueError(f"unknown No Finding rule {d['no_finding_rule']!r}")
    if d.get("dataset_split_for_selection") != "validation":
        raise ValueError("final policy must have been selected on the validation split")
    missing = [l for l in LABELS if l not in d["classes"] or not np.isfinite(d["classes"][l]["threshold"])]
    if missing:
        raise ValueError(f"missing/invalid thresholds for {missing}")
    d["thresholds"] = np.array([d["classes"][l]["threshold"] for l in LABELS], dtype=np.float64)
    return d


def apply_final_policy(scores: np.ndarray, policy: dict) -> tuple[np.ndarray, np.ndarray]:
    """(raw thresholded predictions, final predictions after the No Finding rule). Scores are not altered."""
    raw = apply_thresholds(scores, policy["thresholds"])
    return raw, apply_no_finding_rule(raw)
