"""Reciprocal Rank Fusion (Cormack et al., 2009) with full provenance.

score(d) = sum over rankers r of 1 / (k + rank_r(d)), rank 1-based, only documents inside each ranker's top-`depth`
contribute. Ties are broken by corpus order, so the result is fully deterministic.
"""

from __future__ import annotations

RRF_K = 60


def rrf_fuse(rankings: dict[str, list[str]], corpus_index: dict[str, int], k: int = RRF_K, depth: int | None = 100) -> list[dict]:
    """rankings: ranker name -> ordered list of doc ids (best first). Returns fused list of dicts (best first) with provenance."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    score: dict[str, float] = {}
    prov: dict[str, dict] = {}
    for name, ranked in rankings.items():
        for r, d in enumerate(ranked[:depth] if depth else ranked, start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r)
            prov.setdefault(d, {})[f"{name}_rank"] = r
    order = sorted(score, key=lambda d: (-score[d], corpus_index[d]))
    return [{"doc": d, "rrf_score": score[d], "rank": i + 1, **{f"{n}_rank": prov[d].get(f"{n}_rank") for n in rankings}} for i, d in enumerate(order)]
