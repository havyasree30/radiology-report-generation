"""G2 pipeline for one study: Agent 1 -> deterministic validation -> Agent 2 -> Agent 3 (single pass). Every agent response is cached per study id
(request hash includes the full message and the model digest), so an interrupted run resumes without regenerating completed calls. A failed call is
recorded as a failure; nothing is fabricated or replaced."""

from __future__ import annotations

from src.generation.client import GenerationConfig, ResponseCache
from src.generation.g2_agents import (PROMPTS, build_agent1_message, build_agent2_message, build_agent3_message, empty_evidence, generate_agent, message_sha256, parse_critic, parse_draft, parse_json_text,
                                      request_hash_for, validate_agent1)

AGENTS = ("agent1_evidence_verifier", "agent2_grounded_report_writer", "agent3_grounding_critic")


def case_inputs(case: dict) -> tuple[list[dict], bool, list[dict]]:
    positives = [{"finding": f, "calibrated_probability": case["classifier_probabilities"][f]} for f in case["classifier_positive_findings"]]
    return positives, bool(case["no_finding_positive"]), case["retrieved"]


def _call(client, cfg, agent, user, digest, cache: ResponseCache | None, uid: str) -> dict:
    rh = request_hash_for(cfg, agent, user, digest)
    rec = cache.get(uid, rh) if cache is not None else None
    if rec is None:
        rec = generate_agent(client, cfg, agent, user, digest)
        if rec["status"] == "success" and cache is not None:
            cache.put(uid, rec)
    rec["message_sha256"] = message_sha256(PROMPTS[agent], user)
    return rec


def run_case(case: dict, client, cfg: GenerationConfig, digest: str, caches: dict[str, ResponseCache] | None = None) -> dict:
    uid = case["uid"]
    caches = caches or {}
    positives, nf, retrieved = case_inputs(case)
    out: dict = {"uid": uid, "failed": None, "messages": {}, "records": {}}
    if retrieved:
        m1 = build_agent1_message(positives, nf, retrieved)
        r1 = _call(client, cfg, AGENTS[0], m1, digest, caches.get(AGENTS[0]), uid)
        out["messages"][AGENTS[0]], out["records"][AGENTS[0]] = m1, r1
        if r1["status"] != "success":
            out["failed"] = AGENTS[0]
            return out
        raw = parse_json_text(r1["text"])
        evidence, flags = validate_agent1(raw, positives, nf, retrieved)
    else:                                                          # empty-query study: nothing to verify; explicit no_output / indeterminate handling as in G1
        evidence, flags = empty_evidence(nf), ["agent1_skipped_no_query"]
    out["evidence"], out["agent1_flags"] = evidence, flags
    m2 = build_agent2_message(positives, nf, evidence)
    r2 = _call(client, cfg, AGENTS[1], m2, digest, caches.get(AGENTS[1]), uid)
    out["messages"][AGENTS[1]], out["records"][AGENTS[1]] = m2, r2
    if r2["status"] != "success":
        out["failed"] = AGENTS[1]
        return out
    draft = parse_draft(r2["text"])
    out["draft"] = draft
    m3 = build_agent3_message(positives, nf, evidence, draft)
    r3 = _call(client, cfg, AGENTS[2], m3, digest, caches.get(AGENTS[2]), uid)
    out["messages"][AGENTS[2]], out["records"][AGENTS[2]] = m3, r3
    if r3["status"] != "success":
        out["failed"] = AGENTS[2]
        return out
    crit = parse_critic(r3["text"], draft)
    out["critic"] = crit
    out["final"] = crit["report"]
    return out
