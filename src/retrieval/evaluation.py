"""Per-query evaluation of one ranked list, identical in definition to R1 (finding agreement, not retriever similarity),
plus paired-bootstrap helpers. R1 definitions: Jaccard@K is the mean Jaccard of the top-K reports vs the study's true
finding set; nDCG@K uses Jaccard as graded gain with the ideal ranking over the whole corpus; Hit@K / RR use an exact
finding-set match as the only binary criterion; union coverage = share of true findings present in the union of the
top-K reports; duplicate-text rate = 1 - (distinct normalised texts / K).
"""

from __future__ import annotations

import numpy as np

from src.retrieval.metrics import KS, hit_and_rr, jaccard, ndcg_at_k


def evaluate_ranking(ranked: list[str], truth: frozenset, qset: frozenset, doc_set: dict, doc_norm: dict, doc_words: dict,
                     emb: np.ndarray, row_of: dict, all_gain: np.ndarray, exact_possible: bool) -> dict:
    ranked = list(ranked)[:max(KS)]
    if len(ranked) < max(KS):
        raise ValueError("need at least 10 ranked documents")
    g_truth = np.array([jaccard(truth, doc_set[d]) if doc_set[d] else 0.0 for d in ranked])
    g_query = np.array([jaccard(qset, doc_set[d]) if doc_set[d] else 0.0 for d in ranked]) if qset else np.full(len(ranked), np.nan)
    exact = np.array([doc_set[d] == truth for d in ranked])
    m = {}
    for k in KS:
        h, rr = hit_and_rr(exact, k)
        sets = [doc_set[d] for d in ranked[:k]]
        union = frozenset().union(*sets)
        norms = [doc_norm[d] for d in ranked[:k]]
        pair = np.nan
        if k > 1:
            e = emb[[row_of[d] for d in ranked[:k]]]
            s = e @ e.T
            pair = float((s.sum() - np.trace(s)) / (k * (k - 1)))
        m.update({f"jaccard_truth@{k}": float(g_truth[:k].mean()), f"jaccard_query@{k}": float(g_query[:k].mean()),
                  f"ndcg@{k}": ndcg_at_k(g_truth, all_gain, k), f"hit@{k}": h if exact_possible else np.nan,
                  f"rr@{k}": rr if exact_possible else np.nan, f"union_coverage@{k}": float(len(truth & union) / len(truth)),
                  f"zero_overlap_rate@{k}": float((g_truth[:k] == 0).mean()), f"duplicate_text_rate@{k}": float(1 - len(set(norms)) / k),
                  f"unique_templates@{k}": float(len(set(norms))), f"mean_pairwise_cosine@{k}": pair,
                  f"context_words@{k}": float(sum(doc_words[d] for d in ranked[:k]))})
    return m


def bootstrap_draws(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) resampling indices over queries (the unit; IU has no patient id). Shared by all comparisons."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, n) for _ in range(n_boot)])


def paired_ci(a: np.ndarray, b: np.ndarray, draws: np.ndarray) -> dict:
    """Mean paired difference a - b with a 95% percentile interval (NaNs ignored consistently by nanmean)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boots = np.array([np.nanmean(d[i]) for i in draws])
    return {"mean_difference": float(np.nanmean(d)), "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5))}


def mean_ci(a: np.ndarray, draws: np.ndarray) -> dict:
    boots = np.array([np.nanmean(np.asarray(a, dtype=float)[i]) for i in draws])
    return {"mean": float(np.nanmean(a)), "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5))}
