"""Deterministic evidence-support table: for every classifier-positive finding, how many of the Top-K retrieved reports
carry that finding in their mapped IU findings. This is metadata for the generator, NOT another classifier; it never
alters the stored classifier prediction."""

from __future__ import annotations

from src.generation.extraction import ABNORMAL


def support_table(positives: list[str], probs: dict[str, float], retrieved_sets: list[frozenset]) -> list[dict]:
    n = len(retrieved_sets)
    rows = []
    for l in ABNORMAL:
        if l in set(positives):
            k = sum(l in s for s in retrieved_sets)
            rows.append({"finding": l, "classifier_probability": float(probs[l]), "n_supporting": k, "n_retrieved": n, "support_fraction": (k / n) if n else 0.0})
    return rows


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no classifier-positive finding)"
    out = ["| Finding | Classifier probability | Retrieved support |", "|---|---:|---:|"]
    out += [f"| {r['finding']} | {r['classifier_probability']:.2f} | {r['n_supporting']}/{r['n_retrieved']} |" for r in rows]
    return "\n".join(out)
