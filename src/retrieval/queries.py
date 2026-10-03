"""Deterministic retrieval-query construction. Phrases are fixed here BEFORE any evaluation and are never tuned."""

from __future__ import annotations

from src.classification.labels import LABELS, NO_FINDING

NORMAL_QUERY = "no acute abnormality"
PHRASE = {"Enlarged Cardiomediastinum": "enlarged cardiomediastinum", "Cardiomegaly": "cardiomegaly", "Lung Opacity": "lung opacity",
          "Lung Lesion": "lung lesion", "Edema": "pulmonary edema", "Consolidation": "consolidation", "Pneumonia": "pneumonia",
          "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural effusion",
          "Pleural Other": "pleural abnormality", "Fracture": "fracture", "Support Devices": "support device"}
ABNORMAL = [l for l in LABELS if l != NO_FINDING]
assert set(PHRASE) == set(ABNORMAL)


def phrase_query(findings) -> str:
    """Query text for a finding set; findings are emitted in the fixed class order (deterministic)."""
    fs = set(findings)
    if not fs:
        return ""
    if fs == {NO_FINDING}:
        return NORMAL_QUERY
    return " ".join(PHRASE[l] for l in ABNORMAL if l in fs)


def oracle_query(eval_findings: frozenset) -> tuple[str, list[str]]:
    return phrase_query(eval_findings), sorted(eval_findings, key=LABELS.index)


def classifier_query(pred_final: dict[str, bool]) -> dict:
    """All-frozen-positive policy. pred_final: label -> final decision (after the No Finding rule)."""
    pos = [l for l in ABNORMAL if pred_final[l]]
    if pos:
        return {"query": phrase_query(pos), "findings": pos, "status": "findings"}
    if pred_final[NO_FINDING]:
        return {"query": NORMAL_QUERY, "findings": [NO_FINDING], "status": "no_finding"}
    return {"query": "", "findings": [], "status": "empty"}


def gated_query(pred_final: dict[str, bool], prob: dict[str, float], gates: dict[str, float | None]) -> dict:
    """Precision-aware policy. gates[label] = calibrated-probability gate, or None when no defensible gate exists.

    States per finding: high_confidence (positive and passes the gate, or no gate), low_confidence (positive, fails the
    gate), negative. Fallback when no abnormal finding survives: the normal query if No Finding is positive, otherwise
    the strongest classifier-positive finding (highest calibrated probability), marked as a fallback; never an invented
    finding.
    """
    states = {}
    for l in ABNORMAL:
        if not pred_final[l]:
            states[l] = "negative"
        else:
            g = gates.get(l)
            states[l] = "high_confidence" if (g is None or prob[l] >= g) else "low_confidence"
    high = [l for l in ABNORMAL if states[l] == "high_confidence"]
    low = [l for l in ABNORMAL if states[l] == "low_confidence"]
    if high:
        return {"query": phrase_query(high), "findings": high, "status": "findings", "states": states, "low_confidence": low}
    if pred_final[NO_FINDING]:
        return {"query": NORMAL_QUERY, "findings": [NO_FINDING], "status": "no_finding", "states": states, "low_confidence": low}
    if low:
        best = max(low, key=lambda l: (prob[l], -LABELS.index(l)))
        return {"query": phrase_query([best]), "findings": [best], "status": "fallback_strongest_positive", "states": states,
                "low_confidence": [l for l in low if l != best]}
    return {"query": "", "findings": [], "status": "empty", "states": states, "low_confidence": []}
