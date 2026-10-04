"""R2 query construction: fixed clinical phrase expansion, confidence weights, Top-N finding selection.

Nothing here changes a classifier decision: the inputs are the frozen classifier's final positive findings and the frozen
C5 calibrated probabilities (used only as query weights / for ordering).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.classification.labels import LABELS, NO_FINDING
from src.retrieval.queries import ABNORMAL

WEIGHTINGS = ("uniform", "probability", "margin")


def load_expansion(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(d["expansions"]) != set(ABNORMAL):
        raise ValueError("expansion mapping must cover exactly the 13 non-No-Finding classes")
    for lab, phrase in d["expansions"].items():
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase.split()) > 8:
            raise ValueError(f"{lab}: expansion must be a short non-empty phrase")
    return d


def phrase_text(findings, phrases: dict[str, str], normal_phrase: str) -> str:
    """Query text for a finding list in the fixed class order; {No Finding} -> the normal phrase; empty -> ''."""
    fs = set(findings)
    if not fs:
        return ""
    if fs == {NO_FINDING}:
        return normal_phrase
    return " ".join(phrases[l] for l in ABNORMAL if l in fs)


def select_top_n(findings: list[str], probs: dict[str, float], n: int | None) -> list[str]:
    """Keep the n most probable positive findings (ties -> fixed class order). n=None keeps all. Never pads."""
    if n is None or len(findings) <= n:
        return [l for l in ABNORMAL if l in set(findings)]
    ranked = sorted(findings, key=lambda l: (-probs[l], LABELS.index(l)))[:n]
    return [l for l in ABNORMAL if l in set(ranked)]


def finding_weights(kind: str, findings: list[str], probs: dict[str, float], cal_thresholds: dict[str, float]) -> dict[str, float]:
    """Query weights for positive findings.

    uniform     : 1 for every finding.
    probability : the calibrated probability p_i.
    margin      : (p_i - t_i) / (1 - t_i), the evidence above the frozen operating point t_i (calibrated-equivalent
                  threshold), in [0, 1]. The simple form max(p - 0.5, 0) is NOT used: most frozen positives have p < 0.5
                  because the calibrated-equivalent thresholds are far below 0.5, so it would zero most positives.
    If all weights of a query vanish (all findings exactly at their threshold) the weights fall back to uniform.
    """
    if kind == "uniform":
        w = {l: 1.0 for l in findings}
    elif kind == "probability":
        w = {l: float(probs[l]) for l in findings}
    elif kind == "margin":
        w = {l: max(0.0, (float(probs[l]) - cal_thresholds[l]) / (1.0 - cal_thresholds[l])) for l in findings}
    else:
        raise ValueError(f"unknown weighting {kind!r}")
    return w if sum(w.values()) > 0 else {l: 1.0 for l in findings}


def weighted_query_vector(phrase_vecs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """L2-normalised sum_i w_i * embedding(phrase_i)."""
    if not weights:
        raise ValueError("no findings to weight")
    v = sum(w * np.asarray(phrase_vecs[l], dtype=np.float64) for l, w in weights.items())
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("weighted query vector has zero norm")
    return (v / n).astype(np.float32)
