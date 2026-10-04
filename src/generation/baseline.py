"""B0_rule_based: a deterministic non-RAG report built only from the frozen classifier output (no retrieved context).

One short sentence per positive finding in the fixed class order using canonical terms; No Finding gives a normal report;
an output with neither a pathology nor No Finding gives an explicit 'indeterminate' report (it does not assert normal).
"""

from __future__ import annotations

from src.classification.labels import NO_FINDING
from src.generation.extraction import ABNORMAL

TERM = {"Enlarged Cardiomediastinum": "enlarged cardiomediastinal silhouette", "Cardiomegaly": "cardiomegaly", "Lung Opacity": "pulmonary opacity",
        "Lung Lesion": "lung nodule or mass", "Edema": "pulmonary edema", "Consolidation": "consolidation", "Pneumonia": "pneumonia",
        "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural effusion", "Pleural Other": "pleural thickening",
        "Fracture": "fracture", "Support Devices": "support device"}
NORMAL_FINDINGS = "No acute cardiopulmonary abnormality is detected."
NORMAL_IMPRESSION = "No acute cardiopulmonary abnormality."
INDETERMINATE_FINDINGS = "No finding was identified by the automated classifier."
INDETERMINATE_IMPRESSION = "Classifier output is indeterminate."


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def b0_report(positives: list[str], no_finding: bool) -> dict:
    pos = [l for l in ABNORMAL if l in set(positives)]
    patho = [l for l in pos if l != "Support Devices"]
    if not pos and no_finding:
        return {"findings": NORMAL_FINDINGS, "impression": NORMAL_IMPRESSION, "state": "normal"}
    if not pos:
        return {"findings": INDETERMINATE_FINDINGS, "impression": INDETERMINATE_IMPRESSION, "state": "indeterminate"}
    sents = [f"{_cap(TERM[l])} is detected." for l in pos]
    if no_finding and not patho:                                   # No Finding together with a support device only
        sents = [NORMAL_FINDINGS] + sents
        imp = f"{NORMAL_IMPRESSION} {_cap(TERM['Support Devices'])} in place."
    else:
        names = [TERM[l] for l in pos]
        imp = _cap(" and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)) + "."
    return {"findings": " ".join(sents), "impression": imp, "state": "abnormal"}
