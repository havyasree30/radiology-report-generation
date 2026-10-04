"""G2 structured multi-agent RAG: three sequential roles run by the same frozen local model (no loop, one critic pass).

* Agent 1 - Evidence Verifier: sees the frozen classifier output and the frozen R2 Top-5 retrieved reports; returns structured JSON only
  (never a report). The raw JSON is VALIDATED deterministically (see `validate_agent1`) before it is passed on.
* Agent 2 - Grounded Report Writer: sees ONLY the classifier state, calibrated probabilities and the validated Agent 1 evidence JSON.
  It never receives raw retrieved report text.
* Agent 3 - Grounding Critic: sees ONLY the classifier state, the validated Agent 1 evidence and the Agent 2 draft. It never receives raw
  retrieved report text. It returns APPROVE (the draft is kept unchanged by code) or one corrected final report.

No payload builder has a parameter through which a reference report, reference finding or truth could enter.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import replace

from src.generation.baseline import TERM
from src.generation.client import GenerationConfig, OllamaRequestError, OllamaTransientError, _record, request_hash, request_params
from src.generation.extraction import ABNORMAL
from src.generation.metrics import norm_sentence
from src.generation.parse import combined, parse_report

PROMPT_VERSION = "g2-multi-agent-v1"
STATUS = ("supported", "partially_supported", "unsupported")
ISSUES = ("unsupported_finding_stated", "missed_strongly_supported_finding", "no_finding_contradiction", "abnormal_to_normal_collapse", "invented_measurement_history_or_comparison",
          "excessive_certainty", "duplication_or_repetition", "malformed_structure", "none")
DECISIONS = ("APPROVE", "REVISE")
MAX_RANK = 5
SUMMARY_MAX_WORDS = 20
PROMOTE_MIN_RANKS = 3                  # retrieval-only candidates may be written only with at least this many supporting reports (stated in the Agent 2 prompt)
TERMS_LINE = "; ".join(f"{l}: \"{TERM[l]}\"" for l in ABNORMAL)
NAMES_LINE = "; ".join(ABNORMAL)

# ------------------------------------------------------------------------------------------------------------------ prompts
AGENT1_SYSTEM = f"""You are the Evidence Verifier in a research prototype that helps draft PRELIMINARY chest radiograph reports. You do not see the image and you must NOT write a radiology report. You receive (A) the output of an automated image classifier, which can be wrong, and (B) up to five reports written for OTHER patients, retrieved from a reference corpus as similar cases, numbered by rank. Your only task is to check, report by report, which classifier findings the retrieved reports support, and to list other abnormal findings that appear only in the retrieved reports.

Rules:
1. Return JSON only, exactly in the requested structure. Write no text outside the JSON.
2. For every classifier-positive finding, list in supporting_ranks the ranks of the retrieved reports that state that finding as present (hedged mentions such as "possible" or "suspected" count). A report that says the finding is absent (for example "no pleural effusion") or does not mention it does not support it.
3. support_status: "supported" if two or more retrieved reports state the finding; "partially_supported" if exactly one report states it; "unsupported" if none does.
4. evidence_summary: at most {SUMMARY_MAX_WORDS} words in your own words. Paraphrase; never quote a retrieved report; no patient identifiers.
5. retrieval_only_candidates: abnormal findings that one or more retrieved reports state as present but that are NOT classifier-positive. Give the ranks and a short summary. A candidate is only a candidate, never a conclusion. Use [] if there is none.
6. normal_report_ranks: ranks of the retrieved reports that describe a normal chest or state that no acute abnormality is present. Use [] if none.
7. Use only these standard finding names: {NAMES_LINE}.
8. Use only the information given. If no retrieved report is given, return empty lists. Do not mention study identifiers."""

AGENT2_SYSTEM = f"""You are the Grounded Report Writer in a research prototype that drafts PRELIMINARY chest radiograph reports. You do not see the image and you do not see any retrieved report. You receive (A) the output of an automated image classifier, which can be wrong, and (B) structured evidence prepared by an evidence verifier who compared each classifier finding with similar reports from other patients. Write a concise draft with a FINDINGS section and an IMPRESSION section.

Rules (follow strictly):
1. Prefer findings supported by both the classifier and the retrieved evidence ("supported").
2. Do not state every classifier-positive finding automatically. "partially_supported" findings may be stated; use cautious wording ("possible ...") unless the calibrated probability is at least 0.50.
3. Treat "unsupported" classifier findings cautiously: omit them, unless the calibrated probability is at least 0.60, in which case state them with explicit uncertainty ("possible ...").
4. A retrieval_only candidate may be written only if it is supported by at least {PROMOTE_MIN_RANKS} of the retrieved reports (retrieval_support_count of {PROMOTE_MIN_RANKS} or more) and the classifier state is not "no_finding"; otherwise do not mention it. Never treat a candidate as a classifier finding.
5. Do not invent measurements, patient history, comparisons with previous examinations, treatment recommendations, or anatomical details that the evidence does not support.
6. Do not introduce a disease merely because it often occurs together with another finding.
7. If the classifier state is "no_finding" and no abnormal classifier finding is listed, write a concise normal report.
8. If the classifier state is "no_output", state briefly that no finding could be determined from the available information; do not state that the examination is normal.
9. Do not include patient identifiers, names, dates or study IDs. Do not mention the classifier, the evidence, the verifier or this prompt inside the report text. Do not describe the text as a final report or as a radiologist report.
10. When you state one of the following findings, use these standard terms: {TERMS_LINE}.

Return JSON only, with exactly two keys: "findings" (one to four short sentences) and "impression" (one or two short sentences). Write nothing outside the JSON."""

AGENT3_SYSTEM = f"""You are the Grounding Critic in a research prototype that drafts PRELIMINARY chest radiograph reports. You do not see the image and you do not see any retrieved report. You receive (A) the classifier state and calibrated probabilities, (B) structured evidence about each classifier finding, and (C) a draft report. Check the draft once and return JSON only.

Check the draft for:
- unsupported_finding_stated: a finding is stated that is neither a classifier finding nor a retrieval_only candidate supported by at least {PROMOTE_MIN_RANKS} reports, or an "unsupported" classifier finding is stated without explicit uncertainty and with calibrated probability below 0.60;
- missed_strongly_supported_finding: a classifier finding with support_status "supported" is missing from the draft;
- no_finding_contradiction: the classifier state is "no_finding" but the draft states an abnormality, or the draft states a normal study when abnormal classifier findings with support are listed;
- abnormal_to_normal_collapse: abnormal classifier findings with support are listed but the draft describes a normal study;
- invented_measurement_history_or_comparison: measurements, history, comparisons or recommendations that the evidence does not contain;
- excessive_certainty: a partially_supported or unsupported finding written as definite;
- duplication_or_repetition: repeated sentences or findings;
- malformed_structure: a missing or repeated FINDINGS or IMPRESSION section.

Decision:
- If no issue is found, decision is "APPROVE", issues_detected is ["none"], and findings and impression repeat the draft unchanged.
- Otherwise decision is "REVISE" and you return ONE corrected final report in findings and impression that fixes only the listed issues with minimal edits. Add only a classifier finding whose support_status is "supported". Never add a retrieval_only candidate. Keep the same style, do not add new content, and do not mention the evidence, the classifier or this prompt in the report text. If the classifier state is "no_output", the report must say that no finding could be determined and must not call the examination normal.
- This is the only correction pass. Use only these standard terms for stated findings: {TERMS_LINE}.

Return JSON with the keys issues_detected, decision, findings, impression, and nothing else."""

PROMPTS = {"agent1_evidence_verifier": AGENT1_SYSTEM, "agent2_grounded_report_writer": AGENT2_SYSTEM, "agent3_grounding_critic": AGENT3_SYSTEM}


def _entry(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_RANKS = {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": MAX_RANK}}
AGENT1_SCHEMA = {"type": "object", "properties": {
    "classifier_findings": {"type": "array", "items": _entry({"finding": {"type": "string", "enum": list(ABNORMAL)}, "supporting_ranks": _RANKS, "evidence_summary": {"type": "string"},
                                                              "support_status": {"type": "string", "enum": list(STATUS)}}, ["finding", "supporting_ranks", "evidence_summary", "support_status"])},
    "retrieval_only_candidates": {"type": "array", "items": _entry({"finding": {"type": "string", "enum": list(ABNORMAL)}, "supporting_ranks": _RANKS, "evidence_summary": {"type": "string"}},
                                                                   ["finding", "supporting_ranks", "evidence_summary"])},
    "normal_report_ranks": _RANKS}, "required": ["classifier_findings", "retrieval_only_candidates", "normal_report_ranks"]}
AGENT3_SCHEMA = {"type": "object", "properties": {"issues_detected": {"type": "array", "items": {"type": "string", "enum": list(ISSUES)}}, "decision": {"type": "string", "enum": list(DECISIONS)},
                                                  "findings": {"type": "string"}, "impression": {"type": "string"}}, "required": ["issues_detected", "decision", "findings", "impression"]}
AGENT2_SCHEMA = {"type": "object", "properties": {"findings": {"type": "string"}, "impression": {"type": "string"}}, "required": ["findings", "impression"]}
SCHEMAS = {"agent1_evidence_verifier": AGENT1_SCHEMA, "agent2_grounded_report_writer": AGENT2_SCHEMA, "agent3_grounding_critic": AGENT3_SCHEMA}
NUM_PREDICT = {"agent1_evidence_verifier": 900, "agent2_grounded_report_writer": 400, "agent3_grounding_critic": 700}   # JSON agents need room for the structure; all other options are the frozen G1 options
STOPS = {"agent1_evidence_verifier": (), "agent2_grounded_report_writer": (), "agent3_grounding_critic": ()}


def agent_config(base: GenerationConfig, agent: str) -> GenerationConfig:
    """Same frozen model and sampling options as G1; only the output budget and the termination markers are agent-specific."""
    return replace(base, num_predict=NUM_PREDICT[agent], stop=STOPS[agent])


def prompt_spec() -> dict:
    return {"version": PROMPT_VERSION, "prompts": PROMPTS, "schemas": {k: v for k, v in SCHEMAS.items()}, "num_predict": NUM_PREDICT, "stop": {k: list(v) for k, v in STOPS.items()}}


def prompt_hashes() -> dict:
    h = {k: hashlib.sha256(json.dumps({"version": PROMPT_VERSION, "prompt": v, "schema": SCHEMAS[k], "num_predict": NUM_PREDICT[k], "stop": list(STOPS[k])}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
         for k, v in PROMPTS.items()}
    h["combined"] = hashlib.sha256(json.dumps(h, sort_keys=True).encode("utf-8")).hexdigest()
    return h


# ------------------------------------------------------------------------------------------------------------------ inputs
def classifier_state_name(positives: list[str], no_finding_positive: bool) -> str:
    return "abnormal_findings" if positives else ("no_finding" if no_finding_positive else "no_output")


def _classifier_lines(positives: list[dict], no_finding_positive: bool) -> list[str]:
    pos = sorted(positives, key=lambda p: ABNORMAL.index(p["finding"]))
    return ["Final positive findings: " + ("; ".join(f"{p['finding']} (calibrated probability {p['calibrated_probability']:.2f})" for p in pos) or "none"), f"No Finding: {'positive' if no_finding_positive else 'negative'}"]


def build_agent1_message(positives: list[dict], no_finding_positive: bool, retrieved: list[dict]) -> str:
    """retrieved: [{rank, findings, impression}] - the only fields of a retrieved report that Agent 1 receives (no study id, no scores)."""
    lines = ["A. AUTOMATED CLASSIFIER OUTPUT (may contain errors)"] + _classifier_lines(positives, no_finding_positive) + ["", "B. RETRIEVED REPORTS (reports of other patients)"]
    if not retrieved:
        lines.append("None.")
    for r in retrieved:
        lines += [f"[{r['rank']}]", "FINDINGS: " + (r["findings"] or "(not available)"), "IMPRESSION: " + (r["impression"] or "(not available)")]
    lines += ["", "Return the JSON now."]
    return "\n".join(lines)


def _ngram_set(text: str, n: int) -> set:
    t = norm_sentence(text).split()
    return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def shares_ngram(text: str, sources: list[str], n: int = 6) -> bool:
    g = _ngram_set(text, n)
    return bool(g) and any(g & _ngram_set(s, n) for s in sources)


def _ranks(v, k: int) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else []:
        if isinstance(x, bool) or not isinstance(x, int) and not (isinstance(x, str) and x.isdigit()):
            continue
        x = int(x)
        if 1 <= x <= k and x not in out:
            out.append(x)
    return sorted(out)


def status_from_count(n: int) -> str:
    return "supported" if n >= 2 else ("partially_supported" if n == 1 else "unsupported")


def _summary(text, ranks: list[int], k: int, texts: list[str], flags: list[str], tag: str) -> str:
    s = " ".join(str(text or "").split())
    words = s.split()
    if len(words) > SUMMARY_MAX_WORDS:
        s = " ".join(words[:SUMMARY_MAX_WORDS])
        flags.append(f"{tag}_summary_truncated")
    if not s or shares_ngram(s, texts):
        flags.append(f"{tag}_summary_replaced")           # empty, or it reproduces retrieved prose: replaced by a deterministic sentence
        s = f"Stated in {len(ranks)} of {k} retrieved reports." if ranks else "Not stated in any retrieved report."
    return s


def validate_agent1(raw: dict | None, positives: list[dict], no_finding_positive: bool, retrieved: list[dict]) -> tuple[dict, list[str]]:
    """Deterministic validation of the Agent 1 JSON (no reference information): ranks restricted to the retrieved reports, support count recomputed,
    status recomputed from the count by the predeclared rule (the model's own status is kept as `agent_stated_status`), exactly one entry per classifier
    positive, candidate list restricted to non-classifier findings with at least one valid rank, and summaries that reproduce retrieved prose replaced."""
    flags: list[str] = []
    k = len(retrieved)
    texts = [r["findings"] + " " + r["impression"] for r in retrieved]
    if not isinstance(raw, dict):
        raw, _ = {}, flags.append("agent1_unparseable")
    pos_names = [p["finding"] for p in sorted(positives, key=lambda p: ABNORMAL.index(p["finding"]))]
    prob = {p["finding"]: float(p["calibrated_probability"]) for p in positives}
    by_name: dict[str, dict] = {}
    for e in raw.get("classifier_findings") or []:
        if isinstance(e, dict) and e.get("finding") in pos_names and e["finding"] not in by_name:
            by_name[e["finding"]] = e
        elif isinstance(e, dict):
            flags.append("agent1_extra_or_duplicate_classifier_entry")
    items = []
    for f in pos_names:
        e = by_name.get(f)
        if e is None:
            flags.append("agent1_omitted_classifier_finding")
        e = e or {}
        rk = _ranks(e.get("supporting_ranks"), k)
        stated = e.get("support_status") if e.get("support_status") in STATUS else None
        val = status_from_count(len(rk))
        if stated != val:
            flags.append("agent1_status_inconsistent_with_count")
        items.append({"finding": f, "classifier_probability": round(prob[f], 4), "retrieval_support_count": len(rk), "supporting_retrieved_report_ranks": rk,
                      "evidence_summary": _summary(e.get("evidence_summary"), rk, k, texts, flags, "agent1"), "support_status": val, "agent_stated_status": stated})
    cands, seen = [], set()
    for e in raw.get("retrieval_only_candidates") or []:
        if not isinstance(e, dict) or e.get("finding") not in ABNORMAL or e["finding"] in pos_names or e["finding"] in seen:
            continue
        rk = _ranks(e.get("supporting_ranks"), k)
        if not rk:
            continue
        seen.add(e["finding"])
        cands.append({"finding": e["finding"], "retrieval_support_count": len(rk), "supporting_retrieved_report_ranks": rk, "evidence_summary": _summary(e.get("evidence_summary"), rk, k, texts, flags, "agent1"), "status": "retrieval_only"})
    cands.sort(key=lambda c: (-c["retrieval_support_count"], ABNORMAL.index(c["finding"])))
    evidence = {"classifier_output_state": classifier_state_name(pos_names, no_finding_positive), "n_retrieved_reports": k, "classifier_findings": items, "retrieval_only_candidates": cands,
                "retrieved_report_ranks_describing_a_normal_study": _ranks(raw.get("normal_report_ranks"), k)}
    return evidence, sorted(set(flags))


def evidence_for_downstream(evidence: dict) -> dict:
    """The validated evidence as passed to Agents 2 and 3: the model's own (pre-validation) status is not forwarded."""
    ev = json.loads(json.dumps(evidence))
    for it in ev["classifier_findings"]:
        it.pop("agent_stated_status", None)
    return ev


def empty_evidence(no_finding_positive: bool) -> dict:
    return {"classifier_output_state": classifier_state_name([], no_finding_positive), "n_retrieved_reports": 0, "classifier_findings": [], "retrieval_only_candidates": [], "retrieved_report_ranks_describing_a_normal_study": []}


def _evidence_block(evidence: dict) -> str:
    return json.dumps(evidence_for_downstream(evidence), indent=1, ensure_ascii=False)


def build_agent2_message(positives: list[dict], no_finding_positive: bool, evidence: dict) -> str:
    lines = ["A. CLASSIFIER STATE (may contain errors)", f"State: {classifier_state_name([p['finding'] for p in positives], no_finding_positive)}"] + _classifier_lines(positives, no_finding_positive)
    lines += ["", "B. STRUCTURED EVIDENCE (JSON from the evidence verifier)", _evidence_block(evidence), "", "Write the preliminary report now, following the rules and the exact output format."]
    return "\n".join(lines)


def build_agent3_message(positives: list[dict], no_finding_positive: bool, evidence: dict, draft: dict) -> str:
    lines = ["A. CLASSIFIER STATE (may contain errors)", f"State: {classifier_state_name([p['finding'] for p in positives], no_finding_positive)}"] + _classifier_lines(positives, no_finding_positive)
    lines += ["", "B. STRUCTURED EVIDENCE (JSON from the evidence verifier)", _evidence_block(evidence), "", "C. DRAFT REPORT",
              "FINDINGS: " + (draft["findings"] or "(empty)"), "IMPRESSION: " + (draft["impression"] or "(empty)"), "", "Return the JSON now."]
    return "\n".join(lines)


def message_sha256(system: str, user: str) -> str:
    return hashlib.sha256(json.dumps({"system": system, "user": user}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ parsing
def parse_json_text(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def parse_critic(text: str, draft: dict) -> dict:
    """Critic action parsing. APPROVE keeps the Agent 2 draft EXACTLY (the model's copy is only compared with it); REVISE returns the corrected report;
    an unparseable or empty answer keeps the draft and is flagged (nothing is fabricated)."""
    obj = parse_json_text(text)
    flags: list[str] = []
    if obj is None:
        return {"action": "unparseable_kept_draft", "issues": [], "report": dict(draft), "model_copy_identical_to_draft": None, "flags": ["critic_unparseable"]}
    dec = obj.get("decision")
    issues = [i for i in (obj.get("issues_detected") or []) if i in ISSUES]
    f, i = " ".join(str(obj.get("findings") or "").split()), " ".join(str(obj.get("impression") or "").split())
    same = (norm_sentence(f) == norm_sentence(draft["findings"])) and (norm_sentence(i) == norm_sentence(draft["impression"]))
    if dec == "APPROVE":
        if not same:
            flags.append("approve_with_changed_copy_ignored")
        if issues and issues != ["none"]:
            flags.append("approve_with_issues")
        return {"action": "approve", "issues": issues, "report": dict(draft), "model_copy_identical_to_draft": same, "flags": flags}
    if dec == "REVISE":
        if not f or not i:
            return {"action": "revise_unusable_kept_draft", "issues": issues, "report": dict(draft), "model_copy_identical_to_draft": same, "flags": ["revise_without_complete_report"]}
        if same:
            flags.append("revise_but_identical")
            return {"action": "approve", "issues": issues, "report": dict(draft), "model_copy_identical_to_draft": True, "flags": flags}
        return {"action": "revise", "issues": issues, "report": {"findings": f, "impression": i, "format_ok": True, "format_flags": []}, "model_copy_identical_to_draft": False, "flags": flags}
    return {"action": "unparseable_kept_draft", "issues": issues, "report": dict(draft), "model_copy_identical_to_draft": same, "flags": ["critic_missing_decision"]}


# ------------------------------------------------------------------------------------------------------------------ client call
def generate_agent(client, cfg: GenerationConfig, agent: str, user: str, model_digest: str | None, max_attempts: int = 5, sleep=time.sleep) -> dict:
    """One chat call for an agent: same retry/caching conventions as G1 plus the structured-output `format` schema (a local Ollama request field)."""
    c = agent_config(cfg, agent)
    params = request_params(c, PROMPTS[agent], user)
    if SCHEMAS[agent] is not None:
        params["format"] = SCHEMAS[agent]
    rh = request_hash(params, model_digest)
    last: Exception | None = None
    attempt = 0
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            rec = _record(client.chat(params), params, rh, attempt, time.time() - t0)
            rec["agent"] = agent
            return rec
        except OllamaTransientError as e:
            last = e
            if attempt < max_attempts:
                sleep(min(60.0, 2.0 ** attempt))
        except OllamaRequestError as e:
            last = e
            break
    return {"status": "failed", "agent": agent, "error_class": type(last).__name__, "error": str(last)[:200], "attempts": attempt}


def request_hash_for(cfg: GenerationConfig, agent: str, user: str, model_digest: str | None) -> str:
    params = request_params(agent_config(cfg, agent), PROMPTS[agent], user)
    if SCHEMAS[agent] is not None:
        params["format"] = SCHEMAS[agent]
    return request_hash(params, model_digest)


def final_text(report: dict) -> str:
    return combined(report)


def parse_draft(text: str) -> dict:
    """Agent 2 returns JSON {findings, impression} (constrained decoding; the plain-text format let the model enter a hidden reasoning mode in the smoke test)."""
    obj = parse_json_text(text)
    if obj is not None and isinstance(obj.get("findings"), str) and isinstance(obj.get("impression"), str):
        f, i = " ".join(obj["findings"].split()), " ".join(obj["impression"].split())
        flags = [x for x, bad in (("empty_findings", not f), ("empty_impression", not i)) if bad]
        return {"findings": f, "impression": i, "format_ok": not flags, "format_flags": flags}
    r = parse_report(text)
    r["format_flags"] = r["format_flags"] + ["agent2_json_unparseable"]
    r["format_ok"] = False
    return r
