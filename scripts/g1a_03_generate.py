"""G1A step 3: bulk generation of the G1A_llm_no_retrieval reports (same 547 studies, same model, same options as G1) with the FROZEN
G1A prompt. Every response is cached by study id (resumable); failures stay failures (nothing is fabricated).

    .venv\\Scripts\\python.exe -m scripts.g1a_03_generate
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from src.generation.client import GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, generate_single, hardware_record, installed_model, request_hash, request_params
from src.generation.prompts_g1a import SYSTEM_PROMPT, prompt_hash
from src.utils.config import PROJECT_ROOT

G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"


def main() -> int:
    frozen = json.loads((G1A / "g1a_prompt_frozen.json").read_text(encoding="utf-8"))
    meta = json.loads((G1A / "g1a_generator_metadata.json").read_text(encoding="utf-8"))
    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    if frozen["prompt_sha256"] != prompt_hash() or frozen["system_prompt"] != SYSTEM_PROMPT:
        raise SystemExit("STOP: the prompt differs from the frozen G1A prompt")
    cfg = GenerationConfig(**{k: v for k, v in frozen["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    try:
        version = client.version()
    except OllamaTransientError:
        print("STOP: Ollama is not answering", file=sys.stderr)
        return 2
    m = installed_model(client, cfg.model)
    if m is None or m["digest"] != g1_gen["model"]["digest"] or version != g1_gen["ollama_version"] or cfg.options() != g1_gen["generation_options"]:
        raise SystemExit("STOP: the generator differs from the G1 generator")
    digest = m["digest"]
    cases = [json.loads(l) for l in open(G1A / "g1a_cases.jsonl", encoding="utf-8")]
    cache = ResponseCache(G1A / "generation_cache")
    hashes = {c["uid"]: request_hash(request_params(cfg, SYSTEM_PROMPT, c["user_message"]), digest) for c in cases}
    pending = [c for c in cases if cache.get(c["uid"], hashes[c["uid"]]) is None]
    log = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_cases": len(cases), "already_cached": len(cases) - len(pending), "prompt_sha256": frozen["prompt_sha256"], "model_tag": cfg.model,
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
    (G1A / "generation_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    with open(G1A / "g1a_reports_raw.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            rec = cache.get(c["uid"], hashes[c["uid"]])
            if rec:
                f.write(json.dumps({"uid": c["uid"], **{k: rec[k] for k in ("text", "done_reason", "model_returned", "prompt_tokens", "output_tokens", "prompt_may_be_truncated", "wall_seconds", "created_utc")}}, ensure_ascii=False) + "\n")
    print(json.dumps({k: log[k] for k in ("n_cached_success", "n_failed")}))
    return 0 if not still else 1


if __name__ == "__main__":
    raise SystemExit(main())
