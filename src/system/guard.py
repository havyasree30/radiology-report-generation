"""Final-system deterministic routing guard (G1F).

The guard sits between the frozen classifier output (C4 binary decisions with the No Finding consistency rule) and the frozen G1 Single-Agent RAG
generator. It is pure logic: no model, no retrieval, no randomness, and it never reads generated prose.

Routing (exactly three states):

* Path A, ``normal``: No Finding is positive and no pathology label is positive. Support Devices is not a pathology label, in agreement with the frozen C4 rule
  (``support_devices_suppresses_no_finding = False``). The frozen G1 normal-report behaviour is used (retrieval query = the No Finding phrase, G1 prompt rule 10).
* Path B, ``abnormal``: at least one other label is positive, i.e. any positive label that does not make the study an explicit normal one. The frozen G1
  Single-Agent RAG output is used.
* Path C, ``indeterminate``: no label at all is positive and No Finding is not positive (an empty classifier output). Retrieval is not called, the language
  model is not called, the study is not declared normal, and the fixed message below is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.classification.labels import LABELS, NO_FINDING

SUPPORT_DEVICES = "Support Devices"
ABNORMAL_LABELS = tuple(l for l in LABELS if l != NO_FINDING)
PATHOLOGY_LABELS = tuple(l for l in ABNORMAL_LABELS if l != SUPPORT_DEVICES)      # the labels that suppress No Finding in the frozen C4 rule

STATE_NORMAL, STATE_ABNORMAL, STATE_INDETERMINATE = "normal", "abnormal", "indeterminate"
STATES = (STATE_NORMAL, STATE_ABNORMAL, STATE_INDETERMINATE)
PATH_OF_STATE = {STATE_NORMAL: "A", STATE_ABNORMAL: "B", STATE_INDETERMINATE: "C"}

INDETERMINATE_FINDINGS = "Model output is indeterminate for this study."
INDETERMINATE_IMPRESSION = "Automated preliminary interpretation could not be established. Radiologist review is required."
INDETERMINATE_REPORT_TEXT = f"FINDINGS: {INDETERMINATE_FINDINGS}\nIMPRESSION: {INDETERMINATE_IMPRESSION}"
UI_MESSAGE_INDETERMINATE = "Automated interpretation unavailable — radiologist review required."
UI_FORBIDDEN_FOR_INDETERMINATE = ("No abnormality detected", "Normal X-ray")


def route(positive_findings: list[str] | tuple[str, ...], no_finding_positive: bool) -> str:
    """Deterministic routing state from the frozen classifier output only."""
    pos = set(positive_findings)
    unknown = pos - set(ABNORMAL_LABELS)
    if unknown:
        raise ValueError(f"unknown or non-pathology-vocabulary label(s) in the classifier output: {sorted(unknown)}")
    if not pos and not no_finding_positive:
        return STATE_INDETERMINATE
    if no_finding_positive and not (pos & set(PATHOLOGY_LABELS)):
        return STATE_NORMAL
    return STATE_ABNORMAL


@dataclass(frozen=True)
class GuardResult:
    system_interpretation_state: str
    routing_path: str
    retrieval_invoked: bool
    llm_invoked: bool
    findings: str
    impression: str

    @property
    def report_text(self) -> str:
        return f"FINDINGS: {self.findings}\nIMPRESSION: {self.impression}"


def run_guarded(positive_findings: list[str], no_finding_positive: bool, retrieve: Callable[[], object], generate: Callable[[object], dict]) -> GuardResult:
    """Run the final system for one study. ``retrieve()`` and ``generate(retrieved)`` are the frozen R2 retrieval and the frozen G1 generator; the guard
    calls neither for an indeterminate study. ``generate`` returns {'findings': str, 'impression': str}."""
    state = route(positive_findings, no_finding_positive)
    if state == STATE_INDETERMINATE:
        return GuardResult(state, PATH_OF_STATE[state], False, False, INDETERMINATE_FINDINGS, INDETERMINATE_IMPRESSION)
    retrieved = retrieve()
    out = generate(retrieved)
    return GuardResult(state, PATH_OF_STATE[state], True, True, out["findings"], out["impression"])
