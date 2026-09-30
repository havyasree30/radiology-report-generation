"""Retrieval-corpus eligibility (permanent invariant for System B).

ONLY IU X-Ray TRAINING reports may enter the retrieval index. Validation and
test reports are evaluation references and must never be retrievable.
"""

from __future__ import annotations

import pandas as pd

ELIGIBLE_SPLIT = "train"


class RetrievalLeakageError(AssertionError):
    pass


def eligible_report_ids(report_splits: pd.DataFrame, id_col: str = "uid", split_col: str = "split",
                        require_text: bool = True) -> list[str]:
    """Report ids allowed in the retrieval corpus: training split (and non-empty text)."""
    sel = report_splits[split_col] == ELIGIBLE_SPLIT
    if require_text and "has_any_text" in report_splits.columns:
        sel &= report_splits["has_any_text"].astype(bool)
    return sorted(report_splits.loc[sel, id_col].astype(str))


def assert_train_only(corpus_ids, report_splits: pd.DataFrame, id_col: str = "uid", split_col: str = "split") -> None:
    """Fail if any corpus id is unknown or belongs to a non-training split."""
    split_of = dict(zip(report_splits[id_col].astype(str), report_splits[split_col]))
    ids = [str(i) for i in corpus_ids]
    unknown = [i for i in ids if i not in split_of]
    if unknown:
        raise RetrievalLeakageError(f"{len(unknown)} corpus ids not in the split manifest, e.g. {unknown[:5]}")
    bad = [i for i in ids if split_of[i] != ELIGIBLE_SPLIT]
    if bad:
        raise RetrievalLeakageError(f"{len(bad)} non-training reports in retrieval corpus, e.g. {bad[:5]}")
