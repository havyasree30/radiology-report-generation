"""G1A (no-retrieval ablation) prompt and payload.

The ONLY experimental difference between G1 and G1A is that G1A receives no retrieved reports. The G1A system prompt is the G1
prompt with exactly the retrieval-dependent wording removed or reduced (the intro clause about retrieved excerpts; the two rules that
refer to retrieved support; the retrieval mentions inside rules 1, 2, 6, 7 and 10); every other instruction, including the output
format, the standard-term list and the normal/empty-classifier rules, is unchanged. The user message keeps the classifier block
(section A) byte-identical to G1 and drops sections B (retrieved evidence) and C (evidence summary).
"""

from __future__ import annotations

import difflib
import hashlib
import json

from src.generation.baseline import TERM
from src.generation.extraction import ABNORMAL
from src.generation.prompts import SYSTEM_PROMPT as G1_SYSTEM_PROMPT, build_payload, render_user_message

PROMPT_VERSION = "g1a-no-retrieval-v1"
ALLOWED_PAYLOAD_KEYS = {"classifier"}
TERMS_LINE = "; ".join(f"{l}: \"{TERM[l]}\"" for l in ABNORMAL)

SYSTEM_PROMPT = f"""You are drafting a PRELIMINARY chest radiograph report for a research prototype. You do not see the image. You receive the output of an automated image classifier, which can be wrong. Write a concise draft with a FINDINGS section and an IMPRESSION section.

Grounding rules (follow strictly):
1. Use only the classifier findings you are given.
2. Do not invent any finding that is absent from the classifier output.
3. You do not have to mention every classifier-positive finding.
4. Write your own short sentences.
5. Do not include patient identifiers, names, dates or study IDs, and do not mention the classifier or this prompt inside the report text.
6. Do not describe the text as a final report or as a radiologist report.
7. Do not add patient history, measurements, comparisons with previous examinations, treatment recommendations, invented anatomical details or unsupported certainty.
8. If the classifier state is "No Finding: positive" and no abnormal finding is listed, write a concise normal report. Do not add disease findings that are not listed: the classifier state decides normal versus abnormal.
9. If the classifier lists no finding and does not report No Finding, state briefly that no finding could be determined from the available information; do not state that the examination is normal.
10. When you state one of the following findings, use these standard terms: {TERMS_LINE}.

Output exactly this format and nothing else:
FINDINGS:
<one to four short sentences>
IMPRESSION:
<one or two short sentences>"""


def build_payload_g1a(positives: list[dict], no_finding_positive: bool) -> dict:
    """Classifier output only. Built through the G1 builder with no retrieved reports so that the classifier block is identical."""
    p = build_payload(positives, no_finding_positive, [])
    return {"classifier": p["classifier"]}


def render_user_message_g1a(payload: dict) -> str:
    c = payload["classifier"]
    lines = ["A. AUTOMATED CLASSIFIER OUTPUT (may contain errors)"]
    lines.append("Final positive findings: " + ("; ".join(f"{p['finding']} (calibrated probability {p['calibrated_probability']:.2f})" for p in c["positive_findings"]) or "none"))
    lines.append(f"No Finding: {c['no_finding_state']}")
    lines.append("")
    lines.append("Write the preliminary report now, following the grounding rules and the exact output format.")
    return "\n".join(lines)


def assert_payload_clean_g1a(payload: dict) -> None:
    if set(payload) != ALLOWED_PAYLOAD_KEYS:
        raise ValueError(f"G1A payload keys {sorted(payload)} differ from the whitelist {sorted(ALLOWED_PAYLOAD_KEYS)}")
    if set(payload["classifier"]) != {"positive_findings", "no_finding_state", "output_state"}:
        raise ValueError("classifier block keys differ from the whitelist")


def classifier_block(message: str) -> str:
    """Lines of section A of a user message (G1 or G1A), for the byte-identity check."""
    out = []
    for line in message.split("\n"):
        if line.startswith(("B. RETRIEVED EVIDENCE", "C. EVIDENCE SUMMARY", "Write the preliminary report now")):
            break
        out.append(line)
    return "\n".join(out).rstrip()


def example_payload() -> dict:
    return build_payload_g1a([{"finding": "Cardiomegaly", "calibrated_probability": 0.78}, {"finding": "Pleural Effusion", "calibrated_probability": 0.64}], False)


def prompt_spec() -> dict:
    return {"version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT, "user_message_example": render_user_message_g1a(example_payload())}


def prompt_hash() -> str:
    return hashlib.sha256(json.dumps(prompt_spec(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prompt_diff_vs_g1() -> str:
    """Unified diff of the system prompts (G1 -> G1A) for the record."""
    return "\n".join(difflib.unified_diff(G1_SYSTEM_PROMPT.splitlines(), SYSTEM_PROMPT.splitlines(), "G1_system_prompt", "G1A_system_prompt", lineterm="", n=0))


def user_message_diff_vs_g1(payload_g1: dict) -> str:
    return "\n".join(difflib.unified_diff(render_user_message(payload_g1).splitlines(), render_user_message_g1a({"classifier": payload_g1["classifier"]}).splitlines(),
                                          "G1_user_message", "G1A_user_message", lineterm="", n=0))
