"""Quantify what a training-time sampler WOULD do before anyone enables one.

In multi-label data a sampler cannot upweight one label in isolation: an image
drawn because it carries a rare label also carries its co-occurring labels.
These functions compute the expected label distribution under a sampling
weight vector so that side effects are measured, not guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_prevalence_weights(Y: np.ndarray, how: str = "max") -> np.ndarray:
    """Per-image sampling weights from inverse label prevalence.

    Y: (N, C) boolean positive matrix. Images without any positive label are
    treated as their own group ("no positive label") so they are not dropped.
    how="max":  weight driven by the image's rarest positive label.
    how="mean": average inverse prevalence over the image's positive labels.
    how="sqrt_max": square root of "max" (tempered).
    """
    Y = np.asarray(Y, dtype=bool)
    n = len(Y)
    prev = Y.mean(axis=0)
    inv = np.where(prev > 0, 1.0 / np.maximum(prev, 1e-12), 0.0)
    none = ~Y.any(axis=1)
    inv_none = 1.0 / max(none.mean(), 1.0 / n)
    contrib = np.where(Y, inv[None, :], 0.0)
    if how in ("max", "sqrt_max"):
        w = contrib.max(axis=1)
    elif how == "mean":
        k = Y.sum(axis=1)
        w = np.divide(contrib.sum(axis=1), k, out=np.zeros(n), where=k > 0)
    else:
        raise ValueError(how)
    w = np.where(none, inv_none, w)
    return np.sqrt(w) if how == "sqrt_max" else w


def expected_prevalence(Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return (np.asarray(Y, dtype=float) * w[:, None]).sum(axis=0) / w.sum()


def kish_effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def weighted_conditional(Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """P_w(B positive | A positive) under sampling weights (rows A, cols B)."""
    Yf = np.asarray(Y, dtype=float)
    co = (Yf * w[:, None]).T @ Yf
    diag = np.diag(co)
    return np.divide(co, diag[:, None], out=np.full_like(co, np.nan), where=diag[:, None] > 0)


def sampler_effect_table(Y: np.ndarray, labels: list[str], schemes: dict[str, np.ndarray]) -> pd.DataFrame:
    base = Y.mean(axis=0)
    rows = []
    for name, w in schemes.items():
        ep = expected_prevalence(Y, w)
        for j, lab in enumerate(labels):
            rows.append({"scheme": name, "observation": lab, "original_prevalence": base[j],
                         "expected_prevalence": ep[j],
                         "inflation_factor": ep[j] / base[j] if base[j] > 0 else np.nan})
    return pd.DataFrame(rows)
