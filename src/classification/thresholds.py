"""Per-class Youden's J operating thresholds, fitted on VALIDATION data only.

J = TPR - FPR = sensitivity + specificity - 1; threshold_c = argmax_t J_c(t).
A sample is positive when probability >= threshold (sklearn roc_curve convention).
Classes whose ROC is undefined get threshold = None with a reason: never invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from src.classification.labels import LABELS, assert_matches

ALLOWED_SOURCE_SPLITS = frozenset({"val"})


def youden_thresholds(y: np.ndarray, p: np.ndarray, mask: np.ndarray, source_split: str) -> list[dict]:
    if source_split not in ALLOWED_SOURCE_SPLITS:
        raise ValueError(f"thresholds may only be fitted on {sorted(ALLOWED_SOURCE_SPLITS)}, not {source_split!r}")
    out = []
    for j, lab in enumerate(LABELS):
        m = mask[:, j].astype(bool)
        yj, pj = y[m, j], p[m, j]
        n_pos, n_neg = int((yj == 1).sum()), int((yj == 0).sum())
        entry = {"label": lab, "threshold": None, "sensitivity": None, "specificity": None, "youden_j": None,
                 "positive_support": n_pos, "negative_support": n_neg, "undefined_reason": None}
        if n_pos == 0 or n_neg == 0:
            entry["undefined_reason"] = "ROC undefined: no valid positives" if n_pos == 0 else "ROC undefined: no valid negatives"
            out.append(entry)
            continue
        fpr, tpr, thr = roc_curve(yj, pj, drop_intermediate=False)
        finite = np.isfinite(thr)
        j_stat = np.where(finite, tpr - fpr, -np.inf)
        k = int(np.argmax(j_stat))  # first maximum = highest threshold among ties
        entry.update(threshold=float(thr[k]), sensitivity=float(tpr[k]), specificity=float(1 - fpr[k]),
                     youden_j=float(j_stat[k]))
        out.append(entry)
    return out


def save_thresholds(entries: list[dict], path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": "youden_j", "labels": list(LABELS), "provenance": provenance,
                                "thresholds": entries}, indent=2), encoding="utf-8")


def load_thresholds(path: Path) -> np.ndarray:
    """Threshold vector in LABELS order. Fails if any is missing: no invented fallback."""
    if not Path(path).exists():
        raise FileNotFoundError(f"threshold file not found: {path}")
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_matches(d["labels"])
    if d.get("provenance", {}).get("source_split") not in ALLOWED_SOURCE_SPLITS:
        raise ValueError(f"threshold file {path} was not fitted on an allowed split")
    thr = [e["threshold"] for e in d["thresholds"]]
    missing = [e["label"] for e in d["thresholds"] if e["threshold"] is None]
    if missing:
        raise ValueError(f"undefined thresholds for {missing}; cannot produce binary statuses for them")
    return np.asarray(thr, dtype=np.float64)
