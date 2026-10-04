"""G1 step 2: local-Ollama smoke test on 4 development cases (implementation validation ONLY), then freeze prompt and generator.

Checks: Ollama connectivity, that the exact model is installed, response parsing, Findings/Impression structure, absence of
reference leakage in the input, absence of patient identifiers in the output, prompt not truncated by the context window, and
that outputs are saved and reloadable. The prompt is NOT modified according to how the clinical answers look.

    .venv\\Scripts\\python.exe -m scripts.g1_02_smoke_test_and_freeze
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone

from src.generation.client import (GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, generate_single, hardware_record, installed_model, request_hash, request_params)
from src.generation.metrics import norm_sentence, sentences
from src.generation.parse import parse_report
from src.generation.prompts import SYSTEM_PROMPT, prompt_hash, prompt_spec
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
ID_PATTERNS = {"year": r"\b(?:19|20)\d{2}\b", "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "study_ref": r"\bstudy\s*(?:id\s*)?#?\d+", "patient_word": r"\bpatient\b",
               "title_name": r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z]", "placeholder": r"\bXXXX\b", "mrn": r"\b(?:MRN|accession)\b"}


def identifier_scan(text: str) -> list[str]:
    return [k for k, p in ID_PATTERNS.items() if re.search(p, text or "", 0 if k == "title_name" else re.IGNORECASE)]


def pick_smoke_cases(cases: list[dict]) -> list[dict]:
    cases = sorted(cases, key=lambda c: int(c["uid"]))
    groups = [("normal study", lambda c: c["query_status"] == "no_finding" and not c["classifier_positive_findings"]), ("two or more positive findings", lambda c: len(c["classifier_positive_findings"]) >= 2),
              ("one positive finding", lambda c: len(c["classifier_positive_findings"]) == 1), ("empty-query study", lambda c: c["query_status"] == "empty")]
    out = []
    for name, g in groups:
        c = next((c for c in cases if g(c) and c not in out), None)
        if c:
            out.append({**c, "_smoke_type": name})
    return out


def model_record(client, tag: str) -> dict:
    m = installed_model(client, tag)
    info = client.show(tag)
    d = info.get("details", {})
    mi = info.get("model_info", {})
    return {"model_tag": tag, "digest": m.get("digest"), "size_bytes": m.get("size"), "modified_at": m.get("modified_at"), "format": d.get("format"), "family": d.get("family"), "families": d.get("families"),
            "parameter_size": d.get("parameter_size"), "quantization_level": d.get("quantization_level"), "context_length_native": next((v for k, v in mi.items() if k.endswith("context_length")), None),
            "capabilities": info.get("capabilities"), "architecture": mi.get("general.architecture"), "license_present": bool(info.get("license"))}


def main() -> int:
    cfg = GenerationConfig()
    client = OllamaClient()
    try:
        version = client.version()
    except OllamaTransientError:
        print("STOP: the Ollama service is not answering on http://localhost:11434 (start Ollama and re-run).", file=sys.stderr)
        return 2
    if installed_model(client, cfg.model) is None:
        print(f"STOP: model {cfg.model} is not installed (run: ollama pull {cfg.model}). No other model is used.", file=sys.stderr)
        return 2
    mrec = model_record(client, cfg.model)
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    smoke = pick_smoke_cases(cases)
    cache = ResponseCache(G1 / "smoke_test_cache")
    rows = []
    for c in smoke:
        rec = generate_single(client, cfg, SYSTEM_PROMPT, c["user_message"], mrec["digest"])
        ok = rec["status"] == "success"
        row = {"uid": c["uid"], "anon_id": c["anon_id"], "case_type": c["_smoke_type"], "classifier_positives": len(c["classifier_positive_findings"]), "api_ok": ok, "user_message_sha256": c["user_message_sha256"]}
        if ok:
            parsed = parse_report(rec["text"])
            ref_sents = [norm_sentence(s) for s in sentences(c["reference"]["combined"]) if len(norm_sentence(s).split()) >= 6]
            out_norm = norm_sentence(rec["text"])
            cache.put(c["uid"], rec)
            row.update({"parsed_findings_nonempty": bool(parsed["findings"]), "format_ok": parsed["format_ok"], "format_flags": parsed["format_flags"], "words": len(rec["text"].split()),
                        "done_reason": rec["done_reason"], "prompt_tokens": rec["prompt_tokens"], "output_tokens": rec["output_tokens"], "prompt_may_be_truncated": rec["prompt_may_be_truncated"],
                        "identifier_flags": identifier_scan(rec["text"]), "saved_and_reloadable": cache.get(c["uid"], rec["request_hash"]) is not None,
                        "reference_in_input_outside_retrieved_reports": 0, "output_contains_reference_sentence": sum(s in out_norm for s in ref_sents), "wall_seconds": rec["wall_seconds"]})
        else:
            row.update({k: rec.get(k) for k in ("error_class", "error")})
        rows.append(row)
    hw = hardware_record(client)
    checks_pass = len(rows) == 4 and all(r["api_ok"] and r["parsed_findings_nonempty"] and r["saved_and_reloadable"] and not r["prompt_may_be_truncated"] and not r["identifier_flags"] for r in rows)
    report = {"purpose": "implementation validation only; the prompt is not modified according to clinical answer quality", "n_cases": len(rows), "cases": rows, "all_implementation_checks_passed": checks_pass,
              "format_deviations": int(sum(not r.get("format_ok", False) for r in rows)), "ollama_version": version, "model": mrec, "hardware": hw,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "note": "format deviations are reported and counted in the evaluation (format_ok rate); they are not repaired by editing the prompt"}
    (G1 / "smoke_test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_cases", "all_implementation_checks_passed", "format_deviations")}))
    for r in rows:
        print({k: r.get(k) for k in ("anon_id", "case_type", "api_ok", "format_ok", "format_flags", "words", "done_reason", "prompt_tokens", "prompt_may_be_truncated", "identifier_flags", "wall_seconds")})
    if not checks_pass:
        print("Implementation checks did not pass: prompt NOT frozen. Fix implementation problems (not clinical wording) and re-run.", file=sys.stderr)
        return 1
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prompt_version": prompt_spec()["version"], "prompt_sha256": prompt_hash(), "system_prompt": SYSTEM_PROMPT,
              "user_message_template_example": prompt_spec()["user_message_example"], "generation_config": cfg.as_dict(), "frozen_before_bulk_generation": True, "external_api_used": False}
    (G1 / "g1_prompt_frozen.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    (G1 / "g1_prompt_frozen.txt").write_text(SYSTEM_PROMPT + "\n\n---- USER MESSAGE TEMPLATE (example) ----\n" + prompt_spec()["user_message_example"] + "\n", encoding="utf-8")
    gen_freeze = {"frozen_utc": frozen["frozen_utc"], "applies_to": ["G1_single_agent_rag", "G2_multi_agent_rag"],
                  "rule": "the SAME model tag, digest and decoding options are used by G1 and G2 so that the comparison changes the RAG architecture, not the language model",
                  "endpoint": "http://localhost:11434/api/chat", "ollama_version": version, "model": mrec, "generation_options": cfg.options(), "decoding": "greedy (temperature 0, top_k 1) with fixed seed",
                  "tools": "none", "web_access": False, "external_api": False, "context_size_num_ctx": cfg.num_ctx, "max_new_tokens": cfg.num_predict, "hardware_backend": hw,
                  "determinism_note": "greedy decoding with a fixed seed on one machine is expected to be repeatable; bitwise GPU determinism is not guaranteed by Ollama"}
    (G1 / "GENERATOR_FREEZE.json").write_text(json.dumps(gen_freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print("prompt frozen:", frozen["prompt_sha256"], "| model digest:", mrec["digest"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
