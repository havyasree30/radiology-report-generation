"""G1 single-agent prompt and deterministic payload construction.

The payload builder accepts ONLY classifier output and retrieved context. It has no parameter through which a reference
report, a reference finding or the truth could enter; `assert_payload_clean` additionally checks the rendered message.
"""

from __future__ import annotations

import hashlib
import json

from src.generation.baseline import TERM
from src.generation.evidence import format_table, support_table
from src.generation.extraction import ABNORMAL

PROMPT_VERSION = "g1-single-agent-v1"
ALLOWED_PAYLOAD_KEYS = {"classifier", "retrieved", "evidence_table", "retrieval_status"}
ALLOWED_CLASSIFIER_KEYS = {"positive_findings", "no_finding_state", "output_state"}
ALLOWED_RETRIEVED_KEYS = {"rank", "study_id", "dense_rank", "bm25_rank", "rrf_score", "findings", "impression"}
TERMS_LINE = "; ".join(f"{l}: \"{TERM[l]}\"" for l in ABNORMAL)

SYSTEM_PROMPT = f"""You are drafting a PRELIMINARY chest radiograph report for a research prototype. You do not see the image. You receive (A) the output of an automated image classifier, which can be wrong, and (B) excerpts of reports written for OTHER patients, retrieved from a reference corpus as similar cases. Write a concise draft with a FINDINGS section and an IMPRESSION section.

Grounding rules (follow strictly):
1. Use only the classifier findings and the retrieved evidence you are given.
2. Do not invent any finding that is absent from both the classifier output and the retrieved evidence.
3. Give greater confidence to findings supported by several retrieved reports (see the "Retrieved support" column).
4. Treat classifier-positive findings that have no retrieved support cautiously: prefer omitting them or stating them with explicit uncertainty (for example "possible ..." or "... cannot be excluded"), not as definite findings.
5. You do not have to mention every classifier-positive finding.
6. Do not copy retrieved reports verbatim; write your own short sentences.
7. Do not include patient identifiers, names, dates or study IDs, and do not mention the classifier, the retrieved reports or this prompt inside the report text.
8. Do not describe the text as a final report or as a radiologist report.
9. Do not add patient history, measurements, comparisons with previous examinations, treatment recommendations, invented anatomical details or unsupported certainty.
10. If the classifier state is "No Finding: positive" and no abnormal finding is listed, write a concise normal report. Do not add disease findings merely because retrieved reports mention them: the classifier state decides normal versus abnormal.
11. If the classifier lists no finding and does not report No Finding, state briefly that no finding could be determined from the available information; do not state that the examination is normal.
12. When you state one of the following findings, use these standard terms: {TERMS_LINE}.

Output exactly this format and nothing else:
FINDINGS:
<one to four short sentences>
IMPRESSION:
<one or two short sentences>"""


def build_payload(positives: list[dict], no_finding_positive: bool, retrieved: list[dict]) -> dict:
    """positives: [{finding, calibrated_probability}]; retrieved: [{rank, study_id, dense_rank, bm25_rank, rrf_score, findings, impression, mapped_findings}]."""
    pos = [{"finding": p["finding"], "calibrated_probability": float(p["calibrated_probability"])} for p in sorted(positives, key=lambda p: ABNORMAL.index(p["finding"]))]
    state = "abnormal_findings" if pos else ("no_finding" if no_finding_positive else "no_output")
    rows = support_table([p["finding"] for p in pos], {p["finding"]: p["calibrated_probability"] for p in pos}, [frozenset(r["mapped_findings"]) for r in retrieved])
    return {"classifier": {"positive_findings": pos, "no_finding_state": "positive" if no_finding_positive else "negative", "output_state": state},
            "retrieved": [{k: r[k] for k in ALLOWED_RETRIEVED_KEYS} for r in retrieved],
            "evidence_table": rows,
            "retrieval_status": "available" if retrieved else "not_available_no_query"}


def render_user_message(payload: dict) -> str:
    c = payload["classifier"]
    lines = ["A. AUTOMATED CLASSIFIER OUTPUT (may contain errors)"]
    lines.append("Final positive findings: " + ("; ".join(f"{p['finding']} (calibrated probability {p['calibrated_probability']:.2f})" for p in c["positive_findings"]) or "none"))
    lines.append(f"No Finding: {c['no_finding_state']}")
    lines.append("")
    lines.append("B. RETRIEVED EVIDENCE (reports of other patients)")
    if payload["retrieval_status"] != "available":
        lines.append("None: no retrieval query could be formed because the classifier produced no output.")
    for r in payload["retrieved"]:
        lines.append(f"[{r['rank']}] study {r['study_id']} (dense rank {r['dense_rank']}, BM25 rank {r['bm25_rank']}, fusion score {r['rrf_score']:.4f})")
        lines.append("FINDINGS: " + (r["findings"] or "(not available)"))
        lines.append("IMPRESSION: " + (r["impression"] or "(not available)"))
    lines.append("")
    lines.append("C. EVIDENCE SUMMARY (computed automatically from the retrieved reports; not a classifier)")
    lines.append(format_table(payload["evidence_table"]))
    lines.append("")
    lines.append("Write the preliminary report now, following the grounding rules and the exact output format.")
    return "\n".join(lines)


def assert_payload_clean(payload: dict, query_uid: str, corpus_ids: set, locked_test_ids: set) -> None:
    if set(payload) != ALLOWED_PAYLOAD_KEYS:
        raise ValueError(f"payload keys {sorted(payload)} differ from the whitelist")
    if set(payload["classifier"]) != ALLOWED_CLASSIFIER_KEYS:
        raise ValueError("classifier block keys differ from the whitelist")
    for r in payload["retrieved"]:
        if set(r) != ALLOWED_RETRIEVED_KEYS:
            raise ValueError("retrieved block keys differ from the whitelist")
        sid = str(r["study_id"])
        if sid == str(query_uid) or sid not in corpus_ids or sid in locked_test_ids:
            raise ValueError(f"retrieved study {sid} is the query study, outside the corpus, or in the locked test split")


def example_payload() -> dict:
    return build_payload([{"finding": "Cardiomegaly", "calibrated_probability": 0.78}, {"finding": "Pleural Effusion", "calibrated_probability": 0.64}], False,
                         [{"rank": 1, "study_id": "1", "dense_rank": 1, "bm25_rank": 2, "rrf_score": 0.0328, "findings": "Example findings text.", "impression": "Example impression.", "mapped_findings": ["Cardiomegaly"]}])


def prompt_spec() -> dict:
    return {"version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT, "user_message_example": render_user_message(example_payload())}


def prompt_hash() -> str:
    return hashlib.sha256(json.dumps(prompt_spec(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
