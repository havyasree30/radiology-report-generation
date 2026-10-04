"""G1A step 2: verify that the generator is EXACTLY the G1 generator, run the 4-study implementation smoke test, and freeze the G1A
prompt before bulk generation. The smoke test only checks connectivity, parsing, format, input leakage and caching; the prompt is not
revised according to the clinical content of the answers.

    .venv\\Scripts\\python.exe -m scripts.g1a_02_smoke_test_and_freeze
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from scripts.g1_02_smoke_test_and_freeze import identifier_scan, model_record, pick_smoke_cases
from src.generation.client import GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, generate_single, hardware_record, installed_model
from src.generation.parse import parse_report
from src.generation.prompts_g1a import SYSTEM_PROMPT, prompt_hash, prompt_spec
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"


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
            "quantization": bool(mrec) and mrec["quantization_level"] == g1_gen["model"]["quantization_level"], "generation_options": cfg.options() == g1_gen["generation_options"],
            "num_ctx_and_num_predict": cfg.num_ctx == g1_gen["context_size_num_ctx"] and cfg.num_predict == g1_gen["max_new_tokens"]}
    if not all(same.values()):
        print(f"STOP: the generator differs from the G1 generator: {same}", file=sys.stderr)
        return 3
    cases = [json.loads(l) for l in open(G1A / "g1a_cases.jsonl", encoding="utf-8")]
    smoke = pick_smoke_cases(cases)
    cache = ResponseCache(G1A / "smoke_test_cache")
    rows = []
    for c in smoke:
        rec = generate_single(client, cfg, SYSTEM_PROMPT, c["user_message"], m["digest"])
        row = {"uid": c["uid"], "anon_id": c["anon_id"], "case_type": c["_smoke_type"], "api_ok": rec["status"] == "success"}
        if row["api_ok"]:
            parsed = parse_report(rec["text"])
            cache.put(c["uid"], rec)
            row.update({"parsed_findings_nonempty": bool(parsed["findings"]), "format_ok": parsed["format_ok"], "format_flags": parsed["format_flags"], "words": len(rec["text"].split()), "done_reason": rec["done_reason"],
                        "prompt_tokens": rec["prompt_tokens"], "output_tokens": rec["output_tokens"], "prompt_may_be_truncated": rec["prompt_may_be_truncated"], "identifier_flags": identifier_scan(rec["text"]),
                        "saved_and_reloadable": cache.get(c["uid"], rec["request_hash"]) is not None, "wall_seconds": rec["wall_seconds"]})
        else:
            row.update({k: rec.get(k) for k in ("error_class", "error")})
        rows.append(row)
    ok = len(rows) == 4 and all(r["api_ok"] and r["parsed_findings_nonempty"] and r["saved_and_reloadable"] and not r["prompt_may_be_truncated"] and not r["identifier_flags"] and r["done_reason"] == "stop" for r in rows)
    report = {"purpose": "implementation validation only; the prompt is not modified according to clinical answer quality", "cases": rows, "all_implementation_checks_passed": ok,
              "generator_identical_to_g1": same, "format_deviations": int(sum(not r.get("format_ok", False) for r in rows)), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (G1A / "smoke_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_implementation_checks_passed", "format_deviations")}), same)
    for r in rows:
        print({k: r.get(k) for k in ("anon_id", "case_type", "api_ok", "format_ok", "format_flags", "words", "done_reason", "prompt_tokens", "identifier_flags", "wall_seconds")})
    if not ok:
        print("Implementation checks did not pass: prompt NOT frozen.", file=sys.stderr)
        return 1
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prompt_version": prompt_spec()["version"], "prompt_sha256": prompt_hash(), "system_prompt": SYSTEM_PROMPT,
              "user_message_template_example": prompt_spec()["user_message_example"], "generation_config": cfg.as_dict(), "frozen_before_bulk_generation": True, "external_api_used": False,
              "differs_from_g1_only_by": "no retrieved reports / retrieval scores / evidence-support counts; retrieval-dependent wording removed (see g1a_prompt_diff_vs_g1.txt)"}
    (G1A / "g1a_prompt_frozen.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    (G1A / "g1a_prompt_frozen.txt").write_text(SYSTEM_PROMPT + "\n\n---- USER MESSAGE TEMPLATE (example) ----\n" + prompt_spec()["user_message_example"] + "\n", encoding="utf-8")
    meta = {"frozen_utc": frozen["frozen_utc"], "same_generator_as_g1": same, "ollama_version": version, "model": mrec, "generation_options": cfg.options(), "endpoint": "http://localhost:11434/api/chat",
            "tools": "none", "web_access": False, "external_api": False, "hardware_backend": hardware_record(client), "g1_generator_freeze_sha256_reference": g1_gen["frozen_utc"]}
    (G1A / "g1a_generator_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("G1A prompt frozen:", frozen["prompt_sha256"], "| digest", m["digest"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
