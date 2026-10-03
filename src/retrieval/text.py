"""Report text preparation for retrieval (report text is never rewritten or generated, only cleaned)."""

from __future__ import annotations

import re

import numpy as np

_WS = re.compile(r"\s+")
_ANON = re.compile(r"\bX{2,}\b")                                  # de-identification placeholder (XXXX)
_HEADING = re.compile(r"^\s*(?:findings?|impressions?|comparison|indication)\s*[:\-]\s*", re.I)
_TAGS = re.compile(r"<[^>]+>|&(?:amp|lt|gt|quot|nbsp);")          # XML / HTML residue
_PLACEHOLDER = {"", "none", "none.", "n/a", "na", "nan", "null", "-", "."}


def clean_section(value: object) -> str:
    """One report section -> cleaned string ('' for missing/placeholder)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = _TAGS.sub(" ", str(value))
    s = _HEADING.sub("", s)
    s = _ANON.sub(" ", s)
    s = re.sub(r"\s+([.,;:])", r"\1", _WS.sub(" ", s)).strip()
    s = re.sub(r"^[.,;:\s]+", "", s)
    return "" if s.lower() in _PLACEHOLDER or not re.search(r"[A-Za-z]{2,}", s) else s


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def build_retrieval_text(findings: object, impression: object) -> tuple[str, str]:
    """(retrieval text, source). Findings + Impression when both exist and differ; otherwise the available section."""
    f, i = clean_section(findings), clean_section(impression)
    if f and i:
        same = _norm(f) == _norm(i)
        return (f if same else f"{f} {i}"), ("both_identical" if same else "findings+impression")
    if f:
        return f, "findings_only"
    if i:
        return i, "impression_only"
    return "", "none"
