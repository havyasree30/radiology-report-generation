"""Finding extraction from report text (generated or reference) and report-level normal/abnormal state.

Uses the Phase 1 assertion-aware lexicon with the documented G1 change set (configs/generation/finding_extraction_changes.yaml).
It is a transparent rule-based instrument, NOT a validated clinical labeler; the same extractor is applied to B0, G1 and
(as a sensitivity analysis) the reference report text, and its agreement with the MeSH truth is reported.

stated findings   = concepts with a POSITIVE or UNCERTAIN (hedged) affirmative mention   -> primary ("any mention")
definite findings = concepts with a POSITIVE mention only                                 -> sensitivity
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

from src.analysis.text_concepts import NEGATED, NOT_MENTIONED, POSITIVE, UNCERTAIN, ConceptExtractor
from src.classification.labels import LABELS, NO_FINDING
from src.utils.config import PROJECT_ROOT

ABNORMAL = [l for l in LABELS if l != NO_FINDING]
LEXICON = PROJECT_ROOT / "configs/iu_concept_lexicon.yaml"
CHANGES = PROJECT_ROOT / "configs/generation/finding_extraction_changes.yaml"


def load_extraction_config(lexicon: Path = LEXICON, changes: Path = CHANGES) -> tuple[dict, dict]:
    cfg = yaml.safe_load(lexicon.read_text(encoding="utf-8"))
    ch = yaml.safe_load(changes.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    for cls, pats in ch.get("remove_patterns", {}).items():
        cfg["concepts"][cls]["patterns"] = [p for p in cfg["concepts"][cls]["patterns"] if p not in pats]
    for cls, pats in ch.get("add_patterns", {}).items():
        cfg["concepts"][cls]["patterns"] = list(cfg["concepts"][cls]["patterns"]) + [p for p in pats if p not in cfg["concepts"][cls]["patterns"]]
    for key, pats in ch.get("add_assertion_patterns", {}).items():
        cfg["assertion"][key] = list(cfg["assertion"][key]) + [p for p in pats if p not in cfg["assertion"][key]]
    return cfg, ch


class ReportExtractor:
    def __init__(self):
        cfg, ch = load_extraction_config()
        self.ext = ConceptExtractor.from_config(cfg)
        self.normal = re.compile("|".join(f"(?:{p})" for p in ch["normal_patterns"]), re.IGNORECASE)
        missing = set(ABNORMAL) - set(self.ext.concepts)
        if missing:
            raise ValueError(f"lexicon lacks concepts: {missing}")

    def status(self, text: str) -> dict[str, str]:
        st = self.ext.report_status(text or "")
        return {l: st.get(l, NOT_MENTIONED) for l in ABNORMAL}

    def stated(self, text: str) -> frozenset:
        return frozenset(l for l, s in self.status(text).items() if s in (POSITIVE, UNCERTAIN))

    def definite(self, text: str) -> frozenset:
        return frozenset(l for l, s in self.status(text).items() if s == POSITIVE)

    def report_state(self, text: str) -> str:
        """'abnormal' if any abnormal finding is stated; else 'normal' if a normal statement is present; else 'indeterminate'."""
        if self.stated(text):
            return "abnormal"
        return "normal" if self.normal.search(text or "") else "indeterminate"

    def unmappable_statements(self, text: str) -> list[str]:
        """Sentences that are neither mapped to a concept nor a normal statement (documentation of what the extractor cannot map)."""
        out = []
        for s in re.split(r"(?<=[.;!?])\s+|\n+", text or ""):
            s = s.strip()
            if s and not self.ext.mentions(s) and not self.normal.search(s):
                out.append(s)
        return out
