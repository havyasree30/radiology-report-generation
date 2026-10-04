"""G2 step 2: verify that the generator is EXACTLY the G1 generator, run the 4-study implementation smoke test (normal, one finding, multiple
findings, empty classifier output) through the full three-agent pipeline, and freeze the three agent prompts (with hashes) before bulk generation.
The smoke test checks JSON parsing, output structure, local connectivity, input leakage and cache correctness only; no prompt is revised according
to the clinical content of the answers.

    .venv\\Scripts\\python.exe -m scripts.g2_02_smoke_test_and_freeze
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from scripts.g1_02_smoke_test_and_freeze import identifier_scan, model_record, pick_smoke_cases
from src.generation.client import GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, hardware_record, installed_model
from src.generation.g2_agents import PROMPT_VERSION, PROMPTS, SCHEMAS, agent_config, parse_json_text, prompt_hashes, shares_ngram
from src.generation.g2_pipeline import AGENTS, run_case
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G2 = PROJECT_ROOT / "results/report_generation/experiments/g2_multi_agent"


def main() -> int:
    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    g1_prompt = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in g1_prompt["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    assert cfg.options() == g1_gen["generation_options"], "G1 generation options changed"
    client = OllamaClient()
    try:
        version = client.version()
    except OllamaTransientError:
        print("STOP: Ollama is not answering on http://localhost:11434", file=sys.stderr)
        return 2
    m = installed_model(client, cfg.model)
    mrec = model_record(client, cfg.model) if m else None
    same = {"ollama_version": version == g1_gen["ollama_version"], "model_tag": cfg.model == g1_gen["model"]["model_tag"], "model_digest": bool(m) and m["digest"] == g1_gen["model"]["digest"],
            "quantization": bool(mrec) and mrec["quantization_level"] == g1_gen["model"]["quantization_level"],
            "sampling_options": all(cfg.options()[k] == g1_gen["generation_options"][k] for k in ("temperature", "top_k", "top_p", "seed", "num_ctx")), "num_ctx": cfg.num_ctx == g1_gen["context_size_num_ctx"]}
    if not all(same.values()):
        print(f"STOP: the generator differs from the G1 generator: {same}", file=sys.stderr)
        return 3
    cases = [json.loads(l) for l in open(G2 / "g2_cases.jsonl", encoding="utf-8")]
    smoke = pick_smoke_cases(cases)
    caches = {a: ResponseCache(G2 / "smoke_test_cache" / a) for a in AGENTS}
    rows = []
    for c in smoke:
        res = run_case(c, client, cfg, m["digest"], caches)
        texts = [r["findings"] + " " + r["impression"] for r in c["retrieved"]]
        row = {"uid": c["uid"], "anon_id": c["anon_id"], "case_type": c["_smoke_type"], "pipeline_failed_at": res["failed"]}
        if not res["failed"]:
            recs = res["records"]
            a1 = recs.get(AGENTS[0])
            row.update({"agent1_json_parsed": (parse_json_text(a1["text"]) is not None) if a1 else None, "agent1_skipped_no_query": a1 is None, "agent1_validation_flags": res["agent1_flags"],
                        "agent2_format_ok": res["draft"]["format_ok"], "agent3_action": res["critic"]["action"], "agent3_flags": res["critic"]["flags"], "final_format_ok": bool(res["final"]["findings"] and res["final"]["impression"]),
                        "done_reasons": {a: r["done_reason"] for a, r in recs.items()}, "prompt_truncated": {a: r["prompt_may_be_truncated"] for a, r in recs.items()},
                        "tokens": {a: [r["prompt_tokens"], r["output_tokens"]] for a, r in recs.items()}, "seconds": {a: r["wall_seconds"] for a, r in recs.items()},
                        "agent2_message_contains_retrieved_prose": shares_ngram(res["messages"][AGENTS[1]], texts, 6), "agent3_message_contains_retrieved_prose": shares_ngram(res["messages"][AGENTS[2]], texts, 6),
                        "identifier_flags": identifier_scan(res["final"]["findings"] + " " + res["final"]["impression"]), "cache_reloadable": all(caches[a].get(c["uid"], r["request_hash"]) is not None for a, r in recs.items())})
        rows.append(row)
    ok = (len(rows) == 4 and all(not r["pipeline_failed_at"] for r in rows)
          and all((r["agent1_json_parsed"] or r["agent1_skipped_no_query"]) for r in rows if not r["pipeline_failed_at"])
          and all(r.get("agent2_format_ok") and r.get("final_format_ok") and r.get("cache_reloadable") and not r.get("agent2_message_contains_retrieved_prose") and not r.get("agent3_message_contains_retrieved_prose")
                  and not r.get("identifier_flags") and not any(r["prompt_truncated"].values()) and all(v == "stop" for v in r["done_reasons"].values()) and r["agent3_action"] in ("approve", "revise") for r in rows))
    report = {"purpose": "implementation validation only (JSON parsing, output structure, connectivity, leakage, cache); prompts are not modified according to clinical answer quality", "cases": rows,
              "all_implementation_checks_passed": ok, "generator_identical_to_g1": same, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (G2 / "smoke_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_implementation_checks_passed",)}), same)
    for r in rows:
        print({k: r.get(k) for k in ("anon_id", "case_type", "pipeline_failed_at", "agent1_json_parsed", "agent1_validation_flags", "agent2_format_ok", "agent3_action", "agent3_flags", "done_reasons", "seconds")})
    if not ok:
        print("Implementation checks did not pass: prompts NOT frozen.", file=sys.stderr)
        return 1
    hashes = prompt_hashes()
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prompt_version": PROMPT_VERSION, "prompt_sha256": hashes, "prompts": PROMPTS, "output_schemas": SCHEMAS,
              "agent_generation_options": {a: agent_config(cfg, a).options() for a in AGENTS}, "generation_config_base": cfg.as_dict(), "frozen_before_bulk_generation": True, "external_api_used": False,
              "validation_rules": "Agent 1 JSON is validated deterministically (valid ranks only; support count recomputed from ranks; status recomputed by the count rule >=2 supported / 1 partially_supported / 0 unsupported; "
                                  "summaries that reproduce retrieved prose are replaced; candidates need >=1 rank); the model's own status is stored as agent_stated_status; APPROVE keeps the Agent 2 draft unchanged by code"}
    (G2 / "g2_prompts_frozen.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(G2 / "g2_prompts_frozen.txt", "w", encoding="utf-8") as f:
        for a, p in PROMPTS.items():
            f.write(f"==== {a}  sha256 {hashes[a]} ====\n{p}\n\n")
    meta = {"frozen_utc": frozen["frozen_utc"], "same_generator_as_g1": same, "ollama_version": version, "model": mrec, "model_tag": cfg.model, "model_digest": m["digest"], "generation_options": cfg.options(),
            "agent_specific": {a: {"num_predict": agent_config(cfg, a).num_predict, "stop": list(agent_config(cfg, a).stop), "structured_output_format": SCHEMAS[a] is not None} for a in AGENTS},
            "endpoint": "http://localhost:11434/api/chat", "tools": "none", "web_access": False, "external_api": False, "hardware_backend": hardware_record(client)}
    (G2 / "g2_generator_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("G2 prompts frozen:", json.dumps(hashes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
