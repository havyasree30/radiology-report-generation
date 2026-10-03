"""Report-generation metrics (all deterministic, implemented locally; no downloads).

Text overlap (secondary; lexical overlap is not clinical correctness):
  tokenisation = lowercase alphanumeric runs.
  BLEU-1 / BLEU-4: sentence-level, brevity penalty, add-epsilon smoothing (eps = 0.1) of zero n-gram counts (Chen & Cherry method 1).
  ROUGE-L: F1 of the longest common subsequence.
  METEOR: a LIMITED variant (exact unigram matching only; no stemming, no WordNet synonyms), Banerjee & Lavie parameters
          (alpha 0.9, beta 3, gamma 0.5). It is not comparable with published METEOR values.
Clinical finding agreement (primary): sets of the 13 abnormal classes; No Finding is evaluated separately.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.;!?])\s+|\n+")


def tokens(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


def _ngrams(t: list[str], n: int) -> Counter:
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def bleu(hyp: str, ref: str, max_n: int = 4, eps: float = 0.1) -> float:
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return 0.0
    logp = 0.0
    for n in range(1, max_n + 1):
        hn, rn = _ngrams(h, n), _ngrams(r, n)
        total = max(sum(hn.values()), 0)
        match = sum(min(c, rn[g]) for g, c in hn.items())
        if total == 0:
            return 0.0
        p = (match if match > 0 else eps) / total
        logp += math.log(p) / max_n
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return float(bp * math.exp(logp))


def rouge_l(hyp: str, ref: str) -> float:
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in h:
        cur = [0]
        for j, b in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return float(2 * p * rc / (p + rc))


def meteor_exact(hyp: str, ref: str, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    h, r = tokens(hyp), tokens(ref)
    if not h or not r:
        return 0.0
    used, align = set(), []
    for i, w in enumerate(h):
        for j, v in enumerate(r):
            if j not in used and v == w:
                used.add(j)
                align.append((i, j))
                break
    m = len(align)
    if m == 0:
        return 0.0
    p, rc = m / len(h), m / len(r)
    fmean = p * rc / (alpha * p + (1 - alpha) * rc)
    chunks = 1 + sum(1 for (i1, j1), (i2, j2) in zip(align, align[1:]) if not (i2 == i1 + 1 and j2 == j1 + 1))
    return float(fmean * (1 - gamma * (chunks / m) ** beta))


def text_metrics(hyp: str, ref: str) -> dict:
    return {"bleu1": bleu(hyp, ref, 1), "bleu4": bleu(hyp, ref, 4), "rouge_l": rouge_l(hyp, ref), "meteor_exact": meteor_exact(hyp, ref)}


# ---------------------------------------------------------------- length / redundancy / copying
def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s.strip()]


def norm_sentence(s: str) -> str:
    return " ".join(tokens(s))


def repeated_sentence_rate(text: str) -> float:
    ss = [norm_sentence(s) for s in sentences(text)]
    ss = [s for s in ss if s]
    return float(1 - len(set(ss)) / len(ss)) if ss else 0.0


def copy_stats(generated: str, retrieved_texts: list[str], min_words: int = 6, ngram: int = 8) -> dict:
    """Sentence-level exact copy against any retrieved report (sentences of >= min_words only, so generic short phrases do not count)
    and the share of generated 8-grams that occur in the retrieved texts."""
    ret_sent = {norm_sentence(s) for t in retrieved_texts for s in sentences(t)}
    gs = [norm_sentence(s) for s in sentences(generated)]
    long_ = [s for s in gs if len(s.split()) >= min_words]
    copied = [s for s in long_ if s in ret_sent]
    ret_ng = set()
    for t in retrieved_texts:
        ret_ng |= set(_ngrams(tokens(t), ngram))
    g_ng = list(_ngrams(tokens(generated), ngram).elements())
    return {"n_sentences": len(gs), "n_long_sentences": len(long_), "n_copied_sentences": len(copied),
            "copied_sentence_rate": (len(copied) / len(long_)) if long_ else 0.0,
            "ngram_overlap_rate": (sum(g in ret_ng for g in g_ng) / len(g_ng)) if g_ng else 0.0,
            "whole_report_copy": bool(long_) and len(copied) == len(long_)}


# ---------------------------------------------------------------- finding agreement (abnormal classes)
def study_counts(gen: frozenset, truth: frozenset) -> dict:
    return {"tp": len(gen & truth), "fp": len(gen - truth), "fn": len(truth - gen)}


def prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"precision": p, "recall": r, "f1": f}


def per_class_prf(gen_sets: list[frozenset], truth_sets: list[frozenset], classes: list[str]) -> list[dict]:
    rows = []
    for c in classes:
        tp = sum(c in g and c in t for g, t in zip(gen_sets, truth_sets))
        fp = sum(c in g and c not in t for g, t in zip(gen_sets, truth_sets))
        fn = sum(c not in g and c in t for g, t in zip(gen_sets, truth_sets))
        rows.append({"finding": c, "n_truth": tp + fn, "n_generated": tp + fp, "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)})
    return rows


def classifier_propagation(clf_pos: frozenset, truth: frozenset, gen: frozenset) -> dict:
    """Classifier false positives (positive, not in truth) mentioned in the report; classifier true positives retained in the report."""
    fp, tp = clf_pos - truth, clf_pos & truth
    return {"clf_fp": len(fp), "clf_fp_mentioned": len(fp & gen), "clf_tp": len(tp), "clf_tp_retained": len(tp & gen)}


def provenance(gen: frozenset, clf_pos: frozenset, retrieval_supported: frozenset) -> dict:
    """Source of each stated finding: A classifier only, B retrieval only, C both, D neither."""
    out = {"A": 0, "B": 0, "C": 0, "D": 0}
    for f in gen:
        c, r = f in clf_pos, f in retrieval_supported
        out["C" if c and r else "A" if c else "B" if r else "D"] += 1
    return out


def pooled_rate(num: np.ndarray, den: np.ndarray, idx: np.ndarray | None = None) -> float:
    n, d = (num[idx], den[idx]) if idx is not None else (num, den)
    return float(n.sum() / d.sum()) if d.sum() else float("nan")
