"""Parse a generated report into FINDINGS and IMPRESSION. Never fabricates content: a response that deviates from the
requested format is kept (flagged), with the whole text treated as Findings when no headings are present.

Robustness against a known small-model behaviour: after the IMPRESSION the model may echo the INPUT template. Text from the first
echo marker onward is discarded, and a repeated FINDINGS / IMPRESSION heading after an IMPRESSION has started is not parsed as a new
section (both are flagged)."""

from __future__ import annotations

import re

_H = re.compile(r"^[\s>*#_`-]*\**\s*(FINDINGS?|IMPRESSIONS?)\s*\**\s*[:\-]?\s*\**\s*(.*)$", re.IGNORECASE)
ECHO_MARKERS = ("A. AUTOMATED CLASSIFIER OUTPUT", "B. RETRIEVED EVIDENCE", "C. EVIDENCE SUMMARY")


def parse_report(text: str) -> dict:
    text = (text or "").replace("\r", "")
    flags = []
    cut = [text.find(m) for m in ECHO_MARKERS if text.find(m) >= 0]
    if cut:
        text = text[:min(cut)]
        flags.append("echoed_input_removed")
    sections: dict[str, list[str]] = {"findings": [], "impression": []}
    current, preamble, done = None, [], False
    for line in text.split("\n"):
        m = _H.match(line)
        if m and len(line.strip()) < 400:
            sec = "findings" if m.group(1).upper().startswith("FINDING") else "impression"
            if done or (sec == "findings" and current == "impression"):
                flags.append("repeated_section_ignored")
                done = True
                continue
            current = sec
            if sec == "impression":
                pass
            if m.group(2).strip(" *_"):
                sections[current].append(m.group(2).strip())
        elif done:
            continue
        elif current is None:
            preamble.append(line)
        else:
            sections[current].append(line.strip())
    f, i = " ".join(s for s in sections["findings"] if s).strip(), " ".join(s for s in sections["impression"] if s).strip()
    if current is None:
        f = " ".join(s.strip() for s in preamble if s.strip())
        flags.append("no_headings_whole_text_as_findings")
    elif preamble and any(s.strip() for s in preamble):
        flags.append("text_before_first_heading")
    if not f:
        flags.append("empty_findings")
    if not i:
        flags.append("empty_impression")
    hard = [x for x in flags if x not in ("echoed_input_removed", "repeated_section_ignored")]
    return {"findings": f, "impression": i, "format_ok": not hard, "format_flags": flags}


def combined(parsed: dict) -> str:
    return " ".join(x for x in (parsed["findings"], parsed["impression"]) if x).strip()
