"""G1 step 3: bulk generation of the G1_single_agent_rag reports with the FROZEN prompt and the frozen local model.

    .venv\\Scripts\\python.exe -m scripts.g1_03_generate

* Refuses to run unless the prompt hash equals the frozen hash and the installed model digest equals the frozen digest.
* Every completed response is cached by study id (resumable; completed cases are never regenerated). The four smoke-test studies are
  regenerated here like every other study (the smoke-test cache is separate and is not reused).
* Sequential calls to the local Ollama server; transient failures are retried; failures stay failures (nothing is fabricated).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from src.generation.client import GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, generate_single, hardware_record, installed_model, request_hash, request_params
from src.generation.prompts import SYSTEM_PROMPT, prompt_hash
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"


def main() -> int:
    frozen = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    if frozen["prompt_sha256"] != prompt_hash() or frozen["system_prompt"] != SYSTEM_PROMPT:
        raise SystemExit("STOP: the prompt differs from the frozen prompt; bulk generation is not allowed after a prompt change")
    cfg = GenerationConfig(**{k: v for k, v in frozen["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    try:
        client.version()
    except OllamaTransientError:
        print("STOP: Ollama is not answering on http://localhost:11434", file=sys.stderr)
        return 2
    m = installed_model(client, cfg.model)
    if m is None or m.get("digest") != gen["model"]["digest"]:
        raise SystemExit("STOP: the installed model is missing or its digest differs from the frozen generator")
    digest = m["digest"]
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    cache = ResponseCache(G1 / "generation_cache")
    hashes = {c["uid"]: request_hash(request_params(cfg, SYSTEM_PROMPT, c["user_message"]), digest) for c in cases}
    pending = [c for c in cases if cache.get(c["uid"], hashes[c["uid"]]) is None]
    log = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_cases": len(cases), "already_cached": len(cases) - len(pending), "prompt_sha256": frozen["prompt_sha256"],
           "model_tag": cfg.model, "model_digest": digest}
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
    (G1 / "generation_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    with open(G1 / "g1_reports_raw.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            rec = cache.get(c["uid"], hashes[c["uid"]])
            if rec:
                f.write(json.dumps({"uid": c["uid"], **{k: rec[k] for k in ("text", "done_reason", "model_returned", "prompt_tokens", "output_tokens", "prompt_may_be_truncated", "wall_seconds", "created_utc")}}, ensure_ascii=False) + "\n")
    print(json.dumps({k: log[k] for k in ("n_cached_success", "n_failed")}))
    return 0 if not still else 1


if __name__ == "__main__":
    raise SystemExit(main())
