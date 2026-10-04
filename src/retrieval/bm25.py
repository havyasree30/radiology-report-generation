"""Okapi BM25 (k1=1.5, b=0.75, non-negative Lucene-style IDF), implemented in-house to avoid a dependency.
No stop-word removal and no stemming (kept deliberately simple)."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self.ids, self.k1, self.b = list(ids), k1, b
        toks = [tokenize(t) for t in texts]
        self.dl = np.array([len(t) for t in toks], dtype=float)
        self.avgdl = float(self.dl.mean())
        tf = [Counter(t) for t in toks]
        df = Counter()
        for c in tf:
            df.update(c.keys())
        n = len(toks)
        self.idf = {w: float(np.log(1 + (n - d + 0.5) / (d + 0.5))) for w, d in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(tf):
            for w, f in c.items():
                self.postings.setdefault(w, []).append((i, f))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.ids))
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            for i, f in self.postings[w]:
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (id, score), descending score; ties broken by corpus order (deterministic)."""
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in order]
