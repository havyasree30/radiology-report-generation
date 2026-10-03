"""R1 step 6-7: dense (all-MiniLM-L6-v2 + FAISS IndexFlatIP) and lexical (BM25) indexes over the reference corpus.

    .venv\\Scripts\\python.exe -m scripts.r1_02_build_indexes
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers

from src.classification.final_test import sha256_file
from src.data.retrieval_corpus import assert_train_only
from src.retrieval.bm25 import BM25, tokenize
from src.retrieval.dense import build_index, embed, embedding_dimension, load_encoder
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    corpus = pd.read_csv(OUT / "retrieval_corpus.csv", dtype={"uid": str})
    splits = pd.read_csv(PROJECT_ROOT / "data/splits/iu_xray/iu_report_splits.csv")
    ids = corpus["uid"].tolist()
    assert_train_only(ids, splits)
    assert len(set(ids)) == len(ids)
    texts = corpus["retrieval_text"].tolist()

    model = load_encoder(MODEL, device="cpu")
    dim = embedding_dimension(model)
    emb = embed(model, texts, batch_size=64)
    assert emb.shape == (len(ids), dim), (emb.shape, dim)
    norms = np.linalg.norm(emb, axis=1)
    index = build_index(emb)
    faiss.write_index(index, str(OUT / "dense_minilm_flatip.faiss"))
    np.save(OUT / "corpus_embeddings.npy", emb)
    (OUT / "corpus_study_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    # self-consistency: every corpus vector retrieves an identical vector (itself or an exact-duplicate text) first
    sims, rows = index.search(emb[:200], 1)
    id_roundtrip = bool(np.allclose(sims[:, 0], 1.0, atol=1e-5) and np.allclose(emb[rows[:, 0]], emb[:200], atol=1e-6))   # duplicates may tie, so compare vectors

    tok = model.tokenizer
    n_tok = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])
    max_len = int(model.max_seq_length)
    snap = next(iter(sorted((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots").glob("*"))), None)
    weights = next(iter(sorted(snap.glob("model.safetensors"))), None) if snap else None
    meta = {"model_identifier": MODEL, "model_snapshot_revision": snap.name if snap else None,
            "model_weights_sha256": sha256_file(weights) if weights else None, "embedding_dimension_verified": dim,
            "embedding_dimension_expected_not_assumed": 384, "n_corpus_studies": len(ids), "embeddings_l2_normalised": True,
            "norm_min": float(norms.min()), "norm_max": float(norms.max()), "faiss_index_type": type(index).__name__,
            "faiss_metric": "inner product over unit vectors == cosine similarity", "faiss_ntotal": int(index.ntotal),
            "faiss_row_to_study_id_roundtrip_first_200": id_roundtrip, "max_seq_length_tokens": max_len,
            "corpus_texts_truncated_by_encoder": int((n_tok > max_len).sum()), "corpus_token_length_median": float(np.median(n_tok)),
            "corpus_token_length_max": int(n_tok.max()), "fine_tuned": False, "device": "cpu",
            "versions": {"sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__, "faiss": faiss.__version__,
                         "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "embedding_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    bm = BM25(ids, texts)
    lex = {"method": "Okapi BM25 (in-house implementation)", "k1": bm.k1, "b": bm.b, "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
           "tokenizer": "lowercase alphanumeric runs; no stop-word removal, no stemming", "n_documents": len(ids), "avg_doc_length_tokens": bm.avgdl,
           "vocabulary_size": len(bm.idf), "tie_break": "corpus order (stable sort)", "hyperparameters_tuned": False}
    (OUT / "lexical_index_metadata.json").write_text(json.dumps(lex, indent=2), encoding="utf-8")
    assert tokenize("No XXXX acute") == ["no", "xxxx", "acute"]
    print(json.dumps({"dim": dim, "n": len(ids), "truncated": meta["corpus_texts_truncated_by_encoder"], "revision": meta["model_snapshot_revision"],
                      "norm_range": [meta["norm_min"], meta["norm_max"]], "faiss_ok": id_roundtrip, "bm25_vocab": len(bm.idf)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
