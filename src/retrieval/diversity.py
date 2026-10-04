"""Maximal Marginal Relevance (Carbonell & Goldstein, 1998) over a candidate pool.

next = argmax_{d not in S}  lam * rel(d) - (1 - lam) * max_{s in S} sim(d, s)

rel is min-max normalised inside the pool (so it is on the same 0-1 scale as the cosine similarity); sim is the cosine
similarity of the report embeddings. The first pick is always the most relevant candidate; ties are broken by the original
rank, so the procedure is deterministic and prefix-consistent (top-3 of the MMR list equals MMR with k=3).
"""

from __future__ import annotations

import numpy as np


def mmr_rerank(candidates: list[str], relevance: np.ndarray, embeddings: np.ndarray, lam: float, k: int) -> list[str]:
    """candidates: doc ids best-first; relevance aligned with candidates; embeddings (n_candidates, d), unit norm."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    n = len(candidates)
    if n != len(relevance) or n != len(embeddings):
        raise ValueError("candidates, relevance and embeddings must be aligned")
    rel = np.asarray(relevance, dtype=np.float64)
    span = rel.max() - rel.min()
    rel_n = (rel - rel.min()) / span if span > 0 else np.zeros(n)
    sim = np.asarray(embeddings, dtype=np.float64) @ np.asarray(embeddings, dtype=np.float64).T
    chosen: list[int] = []
    remaining = list(range(n))
    while remaining and len(chosen) < k:
        if not chosen:
            best = max(remaining, key=lambda i: (rel_n[i], -i))
        else:
            best = max(remaining, key=lambda i: (lam * rel_n[i] - (1 - lam) * max(sim[i, j] for j in chosen), -i))
        chosen.append(best)
        remaining.remove(best)
    return [candidates[i] for i in chosen]
