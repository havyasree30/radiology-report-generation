"""Output schema of the report-generation stage (feeds the future application). Probabilities are per class and are never
normalised across classes (multi-label task)."""

from __future__ import annotations

from src.classification.labels import LABELS
from src.generation.extraction import ABNORMAL

DISCLAIMER = "Preliminary AI-generated draft for research use only; not a radiologist report and not a clinical diagnosis."


def build_record(case: dict, report: dict, model: str, prompt_sha256: str, temperature: float | None) -> dict:
    """case: a g1_cases.jsonl entry; report: {'findings','impression'}. The reference report / truth are never part of the record."""
    return {"image_id": case["image_id"],
            "classifier_findings": {"positive_findings": [{"finding": l, "calibrated_probability": float(case["classifier_probabilities"][l])} for l in ABNORMAL if l in case["classifier_positive_findings"]],
                                    "no_finding_positive": bool(case["no_finding_positive"]), "calibrated_probabilities_all_classes": {l: float(case["classifier_probabilities"][l]) for l in LABELS}},
            "retrieved_evidence": [{k: r[k] for k in ("rank", "study_id", "dense_rank", "bm25_rank", "rrf_score", "findings", "impression", "mapped_findings")} for r in case["retrieved"]],
            "evidence_summary": [{k: e[k] for k in ("finding", "classifier_probability", "n_supporting", "n_retrieved", "support_fraction")} for e in case["evidence_rows"]],
            "preliminary_report": {"findings": report["findings"], "impression": report["impression"]},
            "generation": {"model": model, "prompt_sha256": prompt_sha256, "temperature": temperature, "retrieval": "hybrid RRF (dense MiniLM + BM25), top-5"},
            "disclaimer": DISCLAIMER}


def output_schema() -> dict:
    num01 = {"type": "number", "minimum": 0, "maximum": 1}
    return {"$schema": "http://json-schema.org/draft-07/schema#", "title": "Preliminary report generation output (one image)", "type": "object", "additionalProperties": False,
            "required": ["image_id", "classifier_findings", "retrieved_evidence", "evidence_summary", "preliminary_report", "generation", "disclaimer"],
            "properties": {
                "image_id": {"type": "string"},
                "classifier_findings": {"type": "object", "required": ["positive_findings", "no_finding_positive", "calibrated_probabilities_all_classes"],
                                        "properties": {"positive_findings": {"type": "array", "items": {"type": "object", "required": ["finding", "calibrated_probability"],
                                                                                                         "properties": {"finding": {"enum": ABNORMAL}, "calibrated_probability": num01}}},
                                                       "no_finding_positive": {"type": "boolean"},
                                                       "calibrated_probabilities_all_classes": {"type": "object", "required": list(LABELS), "additionalProperties": num01,
                                                                                                "description": "independent per-class probabilities; never normalised across classes"}}},
                "retrieved_evidence": {"type": "array", "maxItems": 5, "items": {"type": "object", "required": ["rank", "study_id", "dense_rank", "bm25_rank", "rrf_score", "findings", "impression", "mapped_findings"],
                                                                                 "properties": {"rank": {"type": "integer", "minimum": 1, "maximum": 5}, "study_id": {"type": "string"},
                                                                                                "dense_rank": {"type": "integer"}, "bm25_rank": {"type": "integer"}, "rrf_score": {"type": "number"},
                                                                                                "findings": {"type": "string"}, "impression": {"type": "string"}, "mapped_findings": {"type": "array", "items": {"type": "string"}}}}},
                "evidence_summary": {"type": "array", "items": {"type": "object", "required": ["finding", "classifier_probability", "n_supporting", "n_retrieved", "support_fraction"]}},
                "preliminary_report": {"type": "object", "required": ["findings", "impression"], "properties": {"findings": {"type": "string"}, "impression": {"type": "string"}}},
                "generation": {"type": "object", "required": ["model", "prompt_sha256", "temperature", "retrieval"]},
                "disclaimer": {"type": "string"}}}


def validate_record(rec: dict) -> None:
    need = {"image_id", "classifier_findings", "retrieved_evidence", "evidence_summary", "preliminary_report", "generation", "disclaimer"}
    if set(rec) != need:
        raise ValueError(f"keys {sorted(rec)} differ from the schema")
    cf = rec["classifier_findings"]
    if list(cf["calibrated_probabilities_all_classes"]) != list(LABELS):
        raise ValueError("probabilities must cover the 14 classes in the fixed order")
    for p in list(cf["calibrated_probabilities_all_classes"].values()) + [x["calibrated_probability"] for x in cf["positive_findings"]]:
        if not (isinstance(p, float) and 0.0 <= p <= 1.0):
            raise ValueError("probability outside [0, 1]")
    if len(rec["retrieved_evidence"]) > 5 or [r["rank"] for r in rec["retrieved_evidence"]] != list(range(1, len(rec["retrieved_evidence"]) + 1)):
        raise ValueError("retrieved evidence must hold at most 5 reports with contiguous ranks")
    if set(rec["preliminary_report"]) != {"findings", "impression"}:
        raise ValueError("preliminary_report must hold findings and impression only")
    if "reference" in rec or "truth" in rec:
        raise ValueError("reference / truth must not appear in the output record")
