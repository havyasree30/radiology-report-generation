"""Multi-label structure statistics on a boolean positive matrix Y (N, C)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cardinality(Y: np.ndarray) -> float:
    """Mean number of positive labels per sample."""
    return float(np.asarray(Y, dtype=bool).sum(axis=1).mean())


def density(Y: np.ndarray) -> float:
    """Cardinality divided by the number of labels."""
    Y = np.asarray(Y, dtype=bool)
    return cardinality(Y) / Y.shape[1]


def count_distribution(Y: np.ndarray) -> pd.Series:
    k = np.asarray(Y, dtype=bool).sum(axis=1)
    return pd.Series(k).value_counts().sort_index()


def cooccurrence_counts(Y: np.ndarray) -> np.ndarray:
    Yi = np.asarray(Y, dtype=np.int64)
    return Yi.T @ Yi


def jaccard(Y: np.ndarray) -> np.ndarray:
    co = cooccurrence_counts(Y).astype(float)
    d = np.diag(co)
    union = d[:, None] + d[None, :] - co
    return np.divide(co, union, out=np.full_like(co, np.nan), where=union > 0)


def conditional_probability(Y: np.ndarray) -> np.ndarray:
    """P(col label positive | row label positive)."""
    co = cooccurrence_counts(Y).astype(float)
    d = np.diag(co)
    return np.divide(co, d[:, None], out=np.full_like(co, np.nan), where=d[:, None] > 0)


def phi_matrix(Y: np.ndarray) -> np.ndarray:
    """Phi coefficient (Pearson correlation of binary indicators)."""
    Yf = np.asarray(Y, dtype=float)
    sd = Yf.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(Yf, rowvar=False)
    c[(sd == 0)[:, None] | (sd == 0)[None, :]] = np.nan
    return c


def lift(Y: np.ndarray) -> np.ndarray:
    """P(A and B) / (P(A) P(B))."""
    Yf = np.asarray(Y, dtype=float)
    n = len(Yf)
    p = Yf.mean(axis=0)
    joint = cooccurrence_counts(Y) / n
    denom = p[:, None] * p[None, :]
    return np.divide(joint, denom, out=np.full_like(joint, np.nan), where=denom > 0)


def pairwise_long(Y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Long table of all unordered label pairs with every association measure."""
    co, jac, cond, phi, lf = cooccurrence_counts(Y), jaccard(Y), conditional_probability(Y), phi_matrix(Y), lift(Y)
    rows = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            rows.append({"label_a": labels[i], "label_b": labels[j], "n_a": int(co[i, i]), "n_b": int(co[j, j]),
                         "n_both": int(co[i, j]), "jaccard": jac[i, j], "p_b_given_a": cond[i, j],
                         "phi": phi[i, j], "lift": lf[i, j]})
    return pd.DataFrame(rows)
