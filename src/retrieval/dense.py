"""Dense retrieval helpers: sentence-transformer embeddings (L2-normalised) + FAISS IndexFlatIP.

With unit-norm vectors the inner product equals cosine similarity.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm embedding")
    return x / n


def load_encoder(name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device)


def embedding_dimension(model) -> int:
    d = (getattr(model, 'get_embedding_dimension', None) or model.get_sentence_embedding_dimension)()
    probe = model.encode(["dimension probe"], convert_to_numpy=True)
    if probe.shape[1] != d:
        raise RuntimeError(f"reported dimension {d} != actual {probe.shape[1]}")
    return int(d)


def embed(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return l2_normalize(model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False,
                                     show_progress_bar=False))


def build_index(emb: np.ndarray):
    import faiss
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if not np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4):
        raise ValueError("embeddings must be L2-normalised for inner product == cosine")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def search(index, ids: list[str], query_emb: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
    """Per query: [(study id, cosine), ...] top-k, descending. FAISS row i <-> ids[i]."""
    if index.ntotal != len(ids):
        raise ValueError("index size differs from id mapping")
    sims, rows = index.search(np.ascontiguousarray(query_emb, dtype=np.float32), k)
    return [[(ids[r], float(s)) for r, s in zip(rr, ss) if r >= 0] for rr, ss in zip(rows, sims)]
