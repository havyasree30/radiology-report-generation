"""Rule-based, assertion-aware concept detection for EDA of radiology reports.

A simplified NegEx-style method (Chapman et al., 2001): each concept mention is
labelled positive / negated / uncertain by trigger phrases within its clause.
It is transparent and deliberately approximate. It is NOT a validated clinical
labeler and its outputs must not be treated as ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

POSITIVE, NEGATED, UNCERTAIN, NOT_MENTIONED = "positive", "negated", "uncertain", "not_mentioned"
_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _alt(patterns: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(f"(?:{p})" for p in patterns) + r")\b", re.IGNORECASE)


@dataclass
class ConceptExtractor:
    concepts: dict[str, re.Pattern]
    neg_pre: re.Pattern
    neg_post: re.Pattern
    unc_pre: re.Pattern
    unc_post: re.Pattern
    pseudo: re.Pattern
    terminators: re.Pattern
    scope_tokens: int

    @classmethod
    def from_config(cls, cfg: dict) -> "ConceptExtractor":
        a = cfg["assertion"]
        return cls(
            concepts={name: _alt(spec["patterns"]) for name, spec in cfg["concepts"].items()},
            neg_pre=_alt(a["negation_pre"]), neg_post=_alt(a["negation_post"]),
            unc_pre=_alt(a["uncertainty_pre"]), unc_post=_alt(a["uncertainty_post"]),
            pseudo=_alt(a["pseudo_negation"]), terminators=_alt(a["scope_terminators"]),
            scope_tokens=int(a["scope_tokens"]),
        )

    def _clause_window(self, sentence: str, start: int, end: int) -> tuple[str, str]:
        """Text before/after a mention, truncated at clause terminators and to scope_tokens tokens."""
        before = sentence[:start]
        term = list(self.terminators.finditer(before))
        if term:
            before = before[term[-1].end():]
        before_toks = _TOKEN.findall(before.lower())[-self.scope_tokens:]
        after = sentence[end:]
        t = self.terminators.search(after)
        if t:
            after = after[:t.start()]
        after_toks = _TOKEN.findall(after.lower())[: self.scope_tokens // 2]
        return " ".join(before_toks), " ".join(after_toks)

    def assert_mention(self, sentence: str, start: int, end: int) -> str:
        before, after = self._clause_window(sentence, start, end)
        before_clean = self.pseudo.sub(" ", before)
        if self.neg_pre.search(before_clean) or self.neg_post.match(after.strip()) or self.neg_post.search(after[:60]):
            return NEGATED
        if self.unc_pre.search(before_clean) or self.unc_post.search(after):
            return UNCERTAIN
        return POSITIVE

    def mentions(self, text: str) -> list[dict]:
        """All concept mentions with their assertion status."""
        out = []
        if not text:
            return out
        text = text.replace("XXXX", "xxxx")
        for sent in _SENT_SPLIT.split(text):
            for name, pat in self.concepts.items():
                for m in pat.finditer(sent):
                    out.append({"concept": name, "match": m.group(0),
                                "status": self.assert_mention(sent, m.start(), m.end()), "sentence": sent})
        return out

    def report_status(self, text: str) -> dict[str, str]:
        """Report-level status per concept: positive > uncertain > negated > not_mentioned."""
        rank = {POSITIVE: 3, UNCERTAIN: 2, NEGATED: 1}
        best = dict.fromkeys(self.concepts, NOT_MENTIONED)
        for m in self.mentions(text):
            cur = best[m["concept"]]
            if cur == NOT_MENTIONED or rank[m["status"]] > rank[cur]:
                best[m["concept"]] = m["status"]
        return best
