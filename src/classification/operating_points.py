"""Per-class operating points (C3): 0.50 baseline, Youden's J and F1-optimal thresholds.

Conventions
  * Decision rule: positive  <=>  score >= threshold  (same as src/classification/thresholds.py).
  * Candidate thresholds are the DISTINCT observed scores of the valid entries.
  * All counts are integers (patient-bootstrap weights are integer multiplicities), so
    ties are decided by exact integer/rational arithmetic, never by float comparison.
  * Youden J = TPR - FPR = sensitivity + specificity - 1.
      tie rule (exact J ties): higher sensitivity, then higher threshold.
  * F1 = 2TP / (2TP + FP + FN).
      tie rule (exact F1 ties): higher recall, then higher threshold.
  * Scores are model scores (sigmoid outputs). They are independent per class: nothing here
    applies softmax, normalises across classes, or constrains thresholds jointly.
  * Undefined cases return None/NaN with a reason; nothing is silently set to 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.classification.labels import LABELS, assert_matches

MIN_POSITIVES_FOR_THRESHOLD = 1
MIN_NEGATIVES_FOR_THRESHOLD = 1


class ScoreError(ValueError):
    """Raised for NaN/Inf/out-of-range scores or malformed inputs."""


def check_scores(p: np.ndarray, name: str = "scores") -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ScoreError(f"{name} contains NaN or Inf")
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ScoreError(f"{name} outside [0, 1]: min={p.min()}, max={p.max()}")
    return p


def check_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype == bool:
        return y
    if not np.all(np.isin(y, (0, 1))):
        raise ScoreError("labels must be binary (0/1) after masking")
    return y.astype(bool)


# ------------------------------------------------------------------ confusion / metrics
def confusion_counts(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y, p = check_labels(y), check_scores(p)
    pred = p >= threshold
    return {"tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()),
            "tn": int((~pred & ~y).sum()), "fn": int((~pred & y).sum())}


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Threshold-dependent metrics; undefined values are NaN (never 0)."""
    nan = float("nan")
    pos, neg = tp + fn, tn + fp
    sens = tp / pos if pos else nan
    spec = tn / neg if neg else nan
    prec = tp / (tp + fp) if (tp + fp) else nan          # undefined when nothing is predicted positive
    den = 2 * tp + fp + fn
    f1 = 2 * tp / den if den else nan                    # undefined only if no positives and no predictions
    bacc = (sens + spec) / 2 if pos and neg else nan
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "positives": pos, "negatives": neg,
            "sensitivity": sens, "recall": sens, "specificity": spec, "precision": prec, "f1": f1,
            "balanced_accuracy": bacc, "youden_j": (sens + spec - 1) if pos and neg else nan}


def operating_point(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    return {"threshold": float(threshold), **metrics_from_counts(**confusion_counts(y, p, threshold))}


# ------------------------------------------------------------------ candidate table
@dataclass
class Candidates:
    thresholds: np.ndarray   # distinct scores, descending
    tp: np.ndarray           # int64, cumulative (weighted) positives with score >= threshold
    fp: np.ndarray
    P: int                   # total (weighted) positives
    N: int                   # total (weighted) negatives


def sorted_order(p: np.ndarray) -> np.ndarray:
    return np.argsort(-p, kind="mergesort")


def candidate_table(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None,
                    order: np.ndarray | None = None) -> Candidates:
    y, p = check_labels(y), check_scores(p)
    order = sorted_order(p) if order is None else order
    ps, ys = p[order], y[order].astype(np.int64)
    w = np.ones(len(ps), dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)[order]
    pos_cum, neg_cum = np.cumsum(w * ys), np.cumsum(w * (1 - ys))
    last = np.r_[ps[1:] != ps[:-1], True] if len(ps) else np.zeros(0, bool)
    return Candidates(ps[last], pos_cum[last], neg_cum[last],
                      int(pos_cum[-1]) if len(ps) else 0, int(neg_cum[-1]) if len(ps) else 0)


def _degenerate(c: Candidates) -> str | None:
    if c.P < MIN_POSITIVES_FOR_THRESHOLD:
        return "no valid positives"
    if c.N < MIN_NEGATIVES_FOR_THRESHOLD:
        return "no valid negatives"
    return None


# ------------------------------------------------------------------ Youden J
def youden_from_candidates(c: Candidates) -> tuple[int, str | None]:
    """Index of the Youden-optimal candidate (exact arithmetic) or (-1, reason)."""
    reason = _degenerate(c)
    if reason:
        return -1, reason
    jnum = c.tp * c.N - c.fp * c.P                      # J * P * N, exact integers
    best = jnum.max()
    if best <= 0:
        return -1, "no threshold with J > 0 (scores uninformative or inverted)"
    tied = np.flatnonzero(jnum == best)
    # higher sensitivity first (larger tp), then higher threshold (smaller index: thresholds descend)
    k = tied[np.lexsort((tied, -c.tp[tied]))[0]]
    return int(k), None


def youden_threshold(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> dict:
    c = candidate_table(y, p, weights)
    k, reason = youden_from_candidates(c)
    if k < 0:
        return {"threshold": None, "undefined_reason": reason, "positives": c.P, "negatives": c.N}
    return {"threshold": float(c.thresholds[k]), "undefined_reason": None, "positives": c.P, "negatives": c.N,
            "n_tied_at_max_j": int((c.tp * c.N - c.fp * c.P == (c.tp * c.N - c.fp * c.P)[k]).sum())}


# ------------------------------------------------------------------ F1
def f1_threshold(y: np.ndarray, p: np.ndarray) -> dict:
    c = candidate_table(y, p)
    reason = _degenerate(c)
    if reason:
        return {"threshold": None, "undefined_reason": reason, "positives": c.P, "negatives": c.N}
    den = c.tp + c.fp + c.P                              # F1 = 2tp / (tp + fp + P)
    f1 = 2 * c.tp / den
    near = np.flatnonzero(f1 >= f1.max() - 1e-12)
    exact = [Fraction(2 * int(c.tp[i]), int(den[i])) for i in near]
    top = max(exact)
    tied = np.array([i for i, e in zip(near, exact) if e == top])
    k = tied[np.lexsort((tied, -c.tp[tied]))[0]]         # higher recall, then higher threshold
    return {"threshold": float(c.thresholds[k]), "undefined_reason": None, "positives": c.P, "negatives": c.N,
            "n_tied_at_max_f1": int(len(tied))}


# ------------------------------------------------------------------ whole-model helpers
def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binary predictions (n, 14): score >= class threshold. Scores are NOT normalised across classes."""
    s = check_scores(scores, "scores")
    t = np.asarray(thresholds, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != t.shape[0]:
        raise ScoreError(f"scores {s.shape} incompatible with {t.shape[0]} thresholds")
    if not np.all(np.isfinite(t)):
        raise ScoreError("thresholds contain NaN/Inf")
    return s >= t[None, :]


def per_class_table(scores: np.ndarray, raw_labels: np.ndarray, valid: np.ndarray, thresholds: list[float | None],
                    ) -> list[dict]:
    """Operating-point metrics for every class at the given per-class thresholds (None -> undefined row)."""
    rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        y = raw_labels[v, j] == 1
        row = {"observation": lab, "n_valid": int(v.sum()), "positives": int(y.sum()), "negatives": int((~y).sum()),
               "prevalence": float(y.mean()) if y.size else float("nan")}
        if thresholds[j] is None:
            row.update(threshold=None, status="undefined")
        else:
            row.update(operating_point(y, scores[v, j], thresholds[j]), status="ok")
        rows.append(row)
    return rows


def macro_summary(rows: list[dict]) -> dict:
    out = {}
    for m in ("precision", "recall", "specificity", "f1", "balanced_accuracy"):
        vals = np.array([r.get(m, np.nan) for r in rows if r.get("status") == "ok"], dtype=float)
        ok = vals[np.isfinite(vals)]
        out[f"macro_{m}"] = float(ok.mean()) if ok.size else float("nan")
        out[f"n_classes_defined_{m}"] = int(ok.size)
    return out


# ------------------------------------------------------------------ patient-level bootstrap
def patient_bootstrap_youden(scores: np.ndarray, raw_labels: np.ndarray, valid: np.ndarray, patient_ids,
                             n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, 14) Youden thresholds from resampling PATIENTS with replacement.

    Every resample draws the same patient multiset for all 14 classes. Counts are integer
    multiplicities, so the exact tie rule applies unchanged. NaN where a class has no valid
    positives or negatives (or J <= 0) in that resample.
    """
    s = check_scores(scores)
    pat = np.asarray(patient_ids)
    uniq, pat_idx = np.unique(pat, return_inverse=True)
    rng = np.random.default_rng(seed)
    orders, ys = [], []
    for j in range(len(LABELS)):
        v = np.flatnonzero(valid[:, j])
        o = sorted_order(s[v, j])
        orders.append((v, o))
        ys.append(raw_labels[v, j] == 1)
    out = np.full((n_boot, len(LABELS)), np.nan)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        for j in range(len(LABELS)):
            v, o = orders[j]
            c = candidate_table(ys[j], s[v, j], weights=mult[pat_idx[v]], order=o)
            k, reason = youden_from_candidates(c)
            if k >= 0:
                out[b, j] = c.thresholds[k]
    return out


# ------------------------------------------------------------------ threshold files
def save_threshold_file(path: Path, method: str, split: str, checkpoint: str, checkpoint_sha256: str,
                        classes: dict, extra: dict | None = None) -> None:
    for lab in classes:
        if lab not in LABELS:
            raise ValueError(f"unknown class {lab!r}")
    payload = {"method": method, "dataset_split": split, "checkpoint": checkpoint,
               "checkpoint_sha256": checkpoint_sha256, "label_order": list(LABELS),
               "decision_rule": "positive if score >= threshold; scores are independent sigmoid outputs",
               **(extra or {}), "classes": {lab: classes[lab] for lab in LABELS if lab in classes}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_threshold_file(path: Path) -> np.ndarray:
    """Thresholds in LABELS order from a C3 JSON. Fails loudly on undefined/missing thresholds."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_matches(d["label_order"])
    if d["dataset_split"] != "validation":
        raise ValueError(f"{path}: thresholds must come from the validation split, got {d['dataset_split']!r}")
    missing = [lab for lab in LABELS if lab not in d["classes"] or d["classes"][lab].get("threshold") is None]
    if missing:
        raise ValueError(f"{path}: undefined thresholds for {missing}")
    return np.array([d["classes"][lab]["threshold"] for lab in LABELS], dtype=np.float64)
