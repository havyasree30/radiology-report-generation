"""Shared loader for R2: the R1 corpus, indexes, primary validation set and frozen-classifier outputs, plus deterministic
ranking helpers. Read-only with respect to R1 / classifier artifacts; the locked retrieval-test studies are never loaded
(only their ids are used to assert exclusion)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.classification.labels import LABELS, NO_FINDING
from src.retrieval.bm25 import BM25
from src.retrieval.dense import embed, load_encoder
from src.retrieval.metrics import jaccard
from src.retrieval.queries import ABNORMAL
from src.utils.config import PROJECT_ROOT

R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
EXP = PROJECT_ROOT / "results/classification/experiments"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FS = lambda s: frozenset(x for x in str(s).split(";") if x) if isinstance(s, str) and s else frozenset()  # noqa: E731


def norm_text(t: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()) if isinstance(t, str) else ""


class R2Context:
    def __init__(self, device: str = "cpu", cache_name: str = "query_embedding_cache.npz"):
        split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
        self.test_ids = set(split["study_ids"]["locked_test"])
        corpus = pd.read_csv(R1 / "retrieval_corpus.csv", dtype={"uid": str})
        self.ids = json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8"))
        assert self.ids == corpus["uid"].tolist()
        self.corpus_index = {u: i for i, u in enumerate(self.ids)}
        self.doc_set = {u: FS(e) for u, e in zip(corpus.uid, corpus.eval_set)}
        self.doc_text = dict(zip(corpus.uid, corpus.retrieval_text))
        self.doc_norm = {u: norm_text(t) for u, t in self.doc_text.items()}
        self.doc_words = {u: len(t.split()) for u, t in self.doc_text.items()}
        self.corpus_norms = set(self.doc_norm.values()) - {""}
        self.emb = np.load(R1 / "corpus_embeddings.npy")
        self.index = faiss.read_index(str(R1 / "dense_minilm_flatip.faiss"))
        self.bm25 = BM25(self.ids, corpus.retrieval_text.tolist())
        self.model_device = device
        self._model = None
        self._cache_path = R2 / cache_name
        self._cache: dict[str, np.ndarray] = {}
        if self._cache_path.exists():
            z = np.load(self._cache_path, allow_pickle=False)
            self._cache = dict(zip(z["texts"].tolist(), z["vecs"]))
        sf = pd.read_csv(R1 / "iu_study_findings.csv", dtype={"uid": str}).fillna({"eval_set": "", "uncertain_terms": "", "unmapped_terms": "", "mapped_findings": ""})
        self.val = sf[sf.partition == "validation"].set_index("uid")
        self.cls = pd.read_csv(R1 / "iu_validation_classifier_outputs.csv", dtype={"uid": str}).fillna("").set_index("uid")
        vt = pd.read_csv(R1 / "validation_report_text_for_error_analysis.csv", dtype={"uid": str}).set_index("uid")["retrieval_text"]
        self.val_text = vt
        self.eligible = [u for u in self.val.index if self.val.loc[u, "has_eval_set"]]
        self.truth = {u: FS(self.val.loc[u, "eval_set"]) for u in self.eligible}
        with_cls = [u for u in self.eligible if u in self.cls.index]
        self.with_cls = with_cls
        self.primary = [u for u in with_cls if self.cls.loc[u, "query_all_positive_status"] != "empty"]
        self.all_gain = {u: np.array([jaccard(self.truth[u], self.doc_set[d]) if self.doc_set[d] else 0.0 for d in self.ids]) for u in self.eligible}
        self.exact_possible = {u: any(self.doc_set[d] == self.truth[u] for d in self.ids) for u in self.eligible}
        self.dupset = {u for u in self.primary if norm_text(vt.get(u)) in self.corpus_norms}
        cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
        self.cal_thr = {l: float(cal["classes"][l]["calibrated_equivalent_threshold"]) for l in LABELS}
        assert not (set(self.eligible) & self.test_ids) and not (set(self.ids) & self.test_ids)

    # ------------------------------------------------------------------ classifier information per study
    def positives(self, uid: str) -> list[str]:
        return [l for l in ABNORMAL if int(self.cls.loc[uid, f"{l}__pred_final"]) == 1]

    def no_finding_positive(self, uid: str) -> bool:
        return int(self.cls.loc[uid, f"{NO_FINDING}__pred_final"]) == 1

    def probs(self, uid: str) -> dict[str, float]:
        return {l: float(self.cls.loc[uid, f"{l}__prob"]) for l in LABELS}

    # ------------------------------------------------------------------ embeddings (cached on disk so every script uses identical vectors)
    @property
    def model(self):
        if self._model is None:
            self._model = load_encoder(MODEL_NAME, device=self.model_device)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = embed(self.model, [text])[0]
        return self._cache[text]

    def save_cache(self) -> None:
        texts = sorted(self._cache)
        np.savez_compressed(self._cache_path, texts=np.array(texts, dtype=str), vecs=np.stack([self._cache[t] for t in texts]))

    # ------------------------------------------------------------------ deterministic rankings (ties -> corpus order, independent of FAISS k)
    def dense_ranking(self, qvec: np.ndarray, depth: int | None = None) -> tuple[list[str], np.ndarray]:
        sims, rows = self.index.search(np.ascontiguousarray(qvec[None, :], dtype=np.float32), len(self.ids))
        s, r = sims[0], rows[0]
        order = np.lexsort((r, -s))                       # primary: score descending; secondary: corpus row ascending
        r, s = r[order], s[order]
        n = depth or len(r)
        return [self.ids[i] for i in r[:n]], s[:n]

    def bm25_ranking(self, text: str, depth: int | None = None) -> tuple[list[str], np.ndarray]:
        sc = self.bm25.scores(text)
        order = np.argsort(-sc, kind="stable")[:depth or len(sc)]
        return [self.ids[i] for i in order], sc[order]
