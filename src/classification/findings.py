"""Deterministic post-processing: per-label predictions -> structured findings -> query-ready output.

Only what the classifier outputs is represented: observation name, probability,
threshold, status. No location, severity, laterality, or disease relationships
are inferred. Nothing is rounded here (rounding is a presentation concern).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS, SUPPORT_DEVICES


def label_predictions(probabilities: Sequence[float], thresholds: Sequence[float]) -> list[dict]:
    """All 14 observations, in LABELS order, each with probability, threshold and status."""
    p = np.asarray(probabilities, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    if p.shape != (len(LABELS),) or t.shape != (len(LABELS),):
        raise ValueError(f"expected {len(LABELS)} probabilities and thresholds, got {p.shape} and {t.shape}")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return [{"observation": lab, "probability": float(p[i]), "threshold": float(t[i]),
             "status": "positive" if p[i] >= t[i] else "negative"} for i, lab in enumerate(LABELS)]


def consistency_warnings(predictions: list[dict]) -> list[dict]:
    """Flag logically suspicious combinations WITHOUT altering any prediction."""
    status = {d["observation"]: d["status"] for d in predictions}
    warnings = []
    if status.get(NO_FINDING) == "positive":
        conflicting = [l for l in PATHOLOGY_LABELS if status.get(l) == "positive"]
        if conflicting:
            warnings.append({"code": "no_finding_with_pathology",
                             "message": "No Finding is positive while pathology observations are also positive",
                             "observations": conflicting})
    return warnings


def build_structured_findings(predictions: list[dict], borderline_margin: float) -> dict:
    """positive / negative lists partition all 14 labels; borderline is an additional, overlapping view."""
    if borderline_margin < 0:
        raise ValueError("borderline_margin must be >= 0")
    def item(d):
        return {"observation": d["observation"], "probability": d["probability"], "threshold": d["threshold"]}
    pos = [item(d) for d in predictions if d["status"] == "positive"]
    neg = [item(d) for d in predictions if d["status"] == "negative"]
    border = [dict(item(d), status=d["status"], margin=d["probability"] - d["threshold"])
              for d in predictions if abs(d["probability"] - d["threshold"]) <= borderline_margin]
    # Positives ordered by how far they clear their own threshold (deterministic tie-break: label order).
    order = {l: i for i, l in enumerate(LABELS)}
    pos.sort(key=lambda d: (-(d["probability"] - d["threshold"]), order[d["observation"]]))
    return {"positive_findings": pos, "negative_findings": neg, "borderline_findings": border,
            "warnings": consistency_warnings(predictions), "borderline_margin": borderline_margin,
            "label_order": list(LABELS)}


def _join(names: list[str]) -> str:
    names = [n.lower() for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def build_query(structured: dict) -> dict:
    """Retrieval-ready representation for Phase 3 (text is NOT embedded here).

    query_text: positives only (observation names, no probabilities, no invented detail).
    Borderline and warnings stay in structured metadata so the retriever/agents can decide.
    """
    pos = [d["observation"] for d in structured["positive_findings"]]
    border_only = [d["observation"] for d in structured["borderline_findings"] if d["status"] == "negative"]
    pathology_pos = [o for o in pos if o not in (NO_FINDING, SUPPORT_DEVICES)]
    nf_positive = NO_FINDING in pos
    if pathology_pos:
        extras = [o for o in pos if o == SUPPORT_DEVICES]
        text = "Chest radiograph findings: " + _join(pathology_pos + extras) + "."
        mode = "positive_findings"
    elif nf_positive:
        text = "Chest radiograph: no finding" + (" with support devices." if SUPPORT_DEVICES in pos else ".")
        mode = "no_finding_positive"
    elif pos:  # only Support Devices positive
        text = "Chest radiograph findings: support devices."
        mode = "support_devices_only"
    else:
        # No observation cleared its threshold. This is NOT asserted as a normal study:
        # No Finding itself did not reach its threshold either.
        text = "Chest radiograph without any classifier finding above its operating threshold."
        mode = "no_positive_findings"
    return {
        "query_text": text,
        "query_mode": mode,
        "positive_observations": pos,
        "borderline_negative_observations": border_only,
        "confidence": {d["observation"]: {"probability": d["probability"], "threshold": d["threshold"]}
                       for d in structured["positive_findings"] + structured["borderline_findings"]},
        "warnings": structured["warnings"],
    }
