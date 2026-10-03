"""Leakage checks for the study-level retrieval split (the retrieval unit is the report/study, never a single image)."""

from __future__ import annotations

import pandas as pd


def studies_spanning_splits(image_manifest: pd.DataFrame, study_col: str = "uid", split_col: str = "split") -> list[str]:
    """Study ids whose images are spread over more than one split (e.g. frontal in train, lateral in test)."""
    n = image_manifest.groupby(study_col)[split_col].nunique()
    return [str(u) for u in n.index[n > 1]]


def images_in_multiple_splits(image_manifest: pd.DataFrame, file_col: str = "filename", split_col: str = "split") -> list[str]:
    n = image_manifest.groupby(file_col)[split_col].nunique()
    return [str(f) for f in n.index[n > 1]]


def pairwise_overlaps(corpus, validation, test) -> dict[str, int]:
    c, v, t = set(map(str, corpus)), set(map(str, validation)), set(map(str, test))
    return {"corpus_and_validation": len(c & v), "corpus_and_test": len(c & t), "validation_and_test": len(v & t)}
