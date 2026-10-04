"""Retrieval metrics based on independent finding agreement (not on the retriever's own similarity score)."""

from __future__ import annotations

import numpy as np

KS = (1, 3, 5, 10)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    u = a | b
    return float(len(a & b) / len(u)) if u else float("nan")


def dcg(gains) -> float:
    g = np.asarray(gains, dtype=float)
    return float((g / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(retrieved_gains, all_gains, k: int) -> float:
    """Graded nDCG@K. Ideal ranking = the K best gains among ALL corpus documents. NaN when no document has gain > 0."""
    ideal = np.sort(np.asarray(all_gains, dtype=float))[::-1][:k]
    idcg = dcg(ideal)
    return float(dcg(np.asarray(retrieved_gains, dtype=float)[:k]) / idcg) if idcg > 0 else float("nan")


def hit_and_rr(retrieved_exact, k: int) -> tuple[float, float]:
    """Hit@K and reciprocal rank (within the top-K list) for a boolean 'exact finding-set match' vector."""
    r = np.asarray(retrieved_exact, dtype=bool)[:k]
    if r.any():
        return 1.0, float(1.0 / (int(np.argmax(r)) + 1))
    return 0.0, 0.0
