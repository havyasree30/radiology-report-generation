"""G1B step 2: pre-flight (generator and prompt are EXACTLY those of G1; sham mapping hash equals the frozen hash) and bulk generation of the
G1B_llm_sham_retrieval reports with the unchanged G1 prompt. Every response is cached by study id (resumable); failures stay failures.

    .venv\Scripts\python.exe -m scripts.g1b_02_generate
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from src.classification.final_test import sha256_file
from src.generation.client import GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, generate_single, hardware_record, installed_model, request_hash, request_params
from src.generation.prompts import SYSTEM_PROMPT, prompt_hash
from src.utils.config import PROJECT_ROOT

G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"


def main() -> int:
    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    g1_prompt = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    frozen_map = json.loads((G1B / "g1b_sham_mapping_frozen.json").read_text(encoding="utf-8"))
    if frozen_map["mapping_sha256"] != sha256_file(G1B / "g1b_sham_mapping.csv"):
        raise SystemExit("STOP: the sham mapping differs from the frozen mapping")
    if g1_prompt["prompt_sha256"] != prompt_hash() or g1_prompt["system_prompt"] != SYSTEM_PROMPT:
        raise SystemExit("STOP: the prompt differs from the frozen G1 prompt")
    cfg = GenerationConfig(**{k: v for k, v in g1_prompt["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    try:
        version = client.version()
    except OllamaTransientError:
        print("STOP: Ollama is not answering", file=sys.stderr)
        return 2
    m = installed_model(client, cfg.model)
    same = {"ollama_version": version == g1_gen["ollama_version"], "model_digest": bool(m) and m["digest"] == g1_gen["model"]["digest"], "options": cfg.options() == g1_gen["generation_options"],
            "num_ctx_and_num_predict": cfg.num_ctx == g1_gen["context_size_num_ctx"] and cfg.num_predict == g1_gen["max_new_tokens"], "model_tag": cfg.model == g1_gen["model"]["model_tag"]}
    if not all(same.values()):
        raise SystemExit(f"STOP: the generator differs from the G1 generator: {same}")
    digest = m["digest"]
    meta = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "same_generator_as_g1": same, "ollama_version": version, "model_tag": cfg.model, "model_digest": digest,
            "generation_options": cfg.options(), "prompt_sha256": prompt_hash(), "prompt_is_g1_prompt": True, "sham_mapping_sha256": frozen_map["mapping_sha256"], "endpoint": "http://localhost:11434/api/chat",
            "tools": "none", "web_access": False, "external_api": False, "hardware_backend": hardware_record(client)}
    (G1B / "g1b_generator_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cases = [json.loads(l) for l in open(G1B / "g1b_cases.jsonl", encoding="utf-8")]
    cache = ResponseCache(G1B / "generation_cache")
    hashes = {c["uid"]: request_hash(request_params(cfg, SYSTEM_PROMPT, c["user_message"]), digest) for c in cases}
    pending = [c for c in cases if cache.get(c["uid"], hashes[c["uid"]]) is None]
    log = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_cases": len(cases), "already_cached": len(cases) - len(pending), "prompt_sha256": prompt_hash(), "model_tag": cfg.model,
           "model_digest": digest, "ollama_version": version}
    print(json.dumps(log), flush=True)
    failed, t0 = {}, time.time()
    for i, c in enumerate(pending, 1):
        rec = generate_single(client, cfg, SYSTEM_PROMPT, c["user_message"], digest)
        if rec["status"] == "success":
            cache.put(c["uid"], rec)
        else:
            failed[c["uid"]] = rec
        if i % 25 == 0 or i == len(pending):
            print(f"{i}/{len(pending)} done, failed={len(failed)}, elapsed={time.time() - t0:.0f}s", flush=True)
    still = [c["uid"] for c in cases if cache.get(c["uid"], hashes[c["uid"]]) is None]
    log.update({"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_cached_success": len(cases) - len(still), "n_failed": len(still), "failed_ids": still,
                "failures": {u: failed.get(u) for u in still}, "hardware_at_end": hardware_record(client), "wall_seconds_this_run": round(time.time() - t0, 1)})
    (G1B / "generation_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    with open(G1B / "g1b_reports_raw.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            rec = cache.get(c["uid"], hashes[c["uid"]])
            if rec:
                f.write(json.dumps({"uid": c["uid"], **{k: rec[k] for k in ("text", "done_reason", "model_returned", "prompt_tokens", "output_tokens", "prompt_may_be_truncated", "wall_seconds", "created_utc")}}, ensure_ascii=False) + "\n")
    print(json.dumps({k: log[k] for k in ("n_cached_success", "n_failed")}))
    return 0 if not still else 1


if __name__ == "__main__":
    raise SystemExit(main())
