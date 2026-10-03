"""Validation-derived retrieval gates (R1 precision-aware query policy).

A retrieval gate is NOT a classifier threshold: the frozen binary decision never changes. It only decides whether a
classifier-positive finding is trusted strongly enough to enter a retrieval query.

Principle (declared before any result is inspected):
  * candidate gates = calibrated-probability quantiles (0, 5%, ..., 95%) of the frozen-positive validation images
    of that class (quantile 0 = the existing operating point);
  * the criterion is the precision-weighted F-score F_0.5 (precision counts twice as much as recall), because the
    aim is to reduce false-positive propagation into retrieval;
  * the gate with the highest F_0.5 on the full validation set is the candidate; it is ADOPTED only if its
    cross-fitted (patient-level K-fold) F_0.5 gain over the operating point has a patient-bootstrap 95% lower bound
    above zero; otherwise the class gets no gate (None) and is marked explicitly.
Only classifier-validation outputs are used; neither the classifier test set nor retrieval results are involved.
"""

from __future__ import annotations

import numpy as np

BETA = 0.5
QUANTILES = tuple(round(q, 2) for q in np.arange(0.0, 0.951, 0.05))


def fbeta(tp, fp, fn, beta: float = BETA):
    tp, fp, fn = np.asarray(tp, dtype=float), np.asarray(fp, dtype=float), np.asarray(fn, dtype=float)
    den = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
    return np.where(den > 0, (1 + beta ** 2) * tp / np.where(den > 0, den, 1), 0.0)


def candidate_gates(p_pos: np.ndarray) -> np.ndarray:
    """Distinct gate values from the quantiles of the calibrated probabilities of frozen-positive images."""
    return np.unique(np.quantile(np.asarray(p_pos, dtype=float), QUANTILES, method="lower"))


def gate_counts(p: np.ndarray, y: np.ndarray, pos: np.ndarray, gate: float) -> tuple[int, int]:
    """(tp, fp) among images that are frozen-positive AND have calibrated probability >= gate."""
    keep = pos & (p >= gate)
    return int((keep & y).sum()), int((keep & ~y).sum())


def select_gate(p: np.ndarray, y: np.ndarray, pos: np.ndarray) -> tuple[float, np.ndarray]:
    """Gate with the highest F_0.5 (ties -> the least strict gate). Recall is relative to ALL valid positives."""
    gates = candidate_gates(p[pos])
    n_pos = int(y.sum())
    scores = []
    for g in gates:
        tp, fp = gate_counts(p, y, pos, g)
        scores.append(float(fbeta(tp, fp, n_pos - tp)))
    scores = np.asarray(scores)
    return float(gates[int(np.argmax(scores))]), scores


def precision_recall(p, y, pos, gate: float) -> tuple[float, float]:
    tp, fp = gate_counts(p, y, pos, gate)
    n_pos = int(y.sum())
    return (tp / (tp + fp) if tp + fp else float("nan")), (tp / n_pos if n_pos else float("nan"))


def cross_fitted_gain(p: np.ndarray, y: np.ndarray, pos: np.ndarray, folds: np.ndarray, patient_idx: np.ndarray,
                      n_boot: int, seed: int) -> dict:
    """Out-of-fold F_0.5 gain (gate chosen on the other folds) versus the operating point, with a patient bootstrap.

    p, y, pos: arrays over the VALID images of one class; folds / patient_idx aligned with them.
    """
    op_gate = float(p[pos].min())
    tp_op, fp_op = (pos & y).astype(float), (pos & ~y).astype(float)
    tp_g, fp_g = np.zeros(len(p)), np.zeros(len(p))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        g, _ = select_gate(p[tr], y[tr], pos[tr])
        keep = pos[te] & (p[te] >= g)
        tp_g[te], fp_g[te] = (keep & y[te]), (keep & ~y[te])
    pos_y = y.astype(float)
    n_p = int(patient_idx.max()) + 1
    rng = np.random.default_rng(seed)

    def gain(w):
        P = float((w * pos_y).sum())
        f_g = fbeta((w * tp_g).sum(), (w * fp_g).sum(), P - (w * tp_g).sum())
        f_o = fbeta((w * tp_op).sum(), (w * fp_op).sum(), P - (w * tp_op).sum())
        return float(f_g - f_o)

    point = gain(np.ones(len(p)))
    draws = np.array([gain(np.bincount(rng.integers(0, n_p, n_p), minlength=n_p)[patient_idx].astype(float)) for _ in range(n_boot)])
    return {"oof_gain_f05": point, "ci95_low": float(np.percentile(draws, 2.5)), "ci95_high": float(np.percentile(draws, 97.5)),
            "operating_gate": op_gate}
