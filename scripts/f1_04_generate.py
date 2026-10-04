"""F1 step 4: frozen G1 report generation for the locked test (Paths A and B, each study exactly once; Path C needs no call). Exactly the G1 bulk procedure:
pre-flight (prompt hash, model digest, Ollama version and options equal the frozen ones), cache per study with request hash, retry only per the established policy
(generate_single), failures stay failures (nothing is fabricated or edited). GPU memory is sampled with nvidia-smi.

    .venv\\Scripts\\python.exe -m scripts.f1_04_generate
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.f1.pipeline import out_dir, require_open
from src.generation.client import GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, generate_single, hardware_record, installed_model, request_hash, request_params
from src.generation.prompts import SYSTEM_PROMPT, prompt_hash
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1F = PROJECT_ROOT / "results/report_generation/experiments/g1f_final_system"


def gpu_used_mib() -> float | None:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return float(o[0]) if o else None
    except Exception:                                                                       # noqa: BLE001
        return None


def main(partition: str = "test") -> int:
    require_open(partition)
    od = out_dir(partition)
    dry = partition != "test"                        # dry run: 3 validation studies into a separate smoke cache, compared with the stored G1 reports
    cfgf = json.loads((G1F / "FINAL_SYSTEM_CONFIG.json").read_text(encoding="utf-8"))["settings"]["generation"]
    frozen = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    if frozen["prompt_sha256"] != prompt_hash() or frozen["system_prompt"] != SYSTEM_PROMPT or cfgf["g1_prompt_sha256"] != prompt_hash():
        raise SystemExit("STOP: the prompt differs from the frozen G1 prompt")
    cfg = GenerationConfig(**{k: v for k, v in frozen["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    try:
        version = client.version()
    except OllamaTransientError:
        print("STOP: Ollama is not answering on http://localhost:11434", file=sys.stderr)
        return 2
    m = installed_model(client, cfg.model)
    if m is None or m.get("digest") != gen["model"]["digest"] or m.get("digest") != cfgf["model_digest"] or version != gen["ollama_version"] or cfg.options() != gen["generation_options"]:
        raise SystemExit("STOP: the installed model, Ollama version or options differ from the frozen generator")
    digest = m["digest"]
    cases = [c for c in (json.loads(l) for l in open(od / "cases.jsonl", encoding="utf-8")) if c["system_interpretation_state"] != "indeterminate"]
    assert all(c["retrieved"] for c in cases)
    if dry:
        cases = cases[:3]
    cache = ResponseCache(od / ("generation_smoke_cache" if dry else "generation_cache"))
    hashes = {c["uid"]: request_hash(request_params(cfg, SYSTEM_PROMPT, c["user_message"]), digest) for c in cases}
    pending = [c for c in cases if cache.get(c["uid"], hashes[c["uid"]]) is None]
    log = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_cases_requiring_generation": len(cases), "already_cached": len(cases) - len(pending), "prompt_sha256": frozen["prompt_sha256"], "model_tag": cfg.model, "model_digest": digest,
           "ollama_version": version, "path_C_studies_not_called": int(sum(1 for _ in open(od / "cases.jsonl", encoding="utf-8")) - len(cases))}
    print(json.dumps(log), flush=True)
    failed, t0, peak = {}, time.time(), [gpu_used_mib() or 0.0]
    for i, c in enumerate(pending, 1):
        rec = generate_single(client, cfg, SYSTEM_PROMPT, c["user_message"], digest)
        if rec["status"] == "success":
            cache.put(c["uid"], rec)
        else:
            failed[c["uid"]] = rec
        if i % 5 == 0:
            peak.append(gpu_used_mib() or 0.0)
        if i % 25 == 0 or i == len(pending):
            print(f"{i}/{len(pending)} done, failed={len(failed)}, elapsed={time.time() - t0:.0f}s", flush=True)
    still = [c["uid"] for c in cases if cache.get(c["uid"], hashes[c["uid"]]) is None]
    log.update({"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_cached_success": len(cases) - len(still), "n_failed": len(still), "failed_ids": still, "failures": {u: failed.get(u) for u in still},
                "wall_seconds_this_run": round(time.time() - t0, 1), "gpu_memory_used_mib_peak_sampled": max(peak), "gpu_memory_samples": len(peak), "hardware_at_end": hardware_record(client)})
    if dry:
        stored = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")}
        same = {c["uid"]: (cache.get(c["uid"], hashes[c["uid"]]) or {}).get("text") == stored[c["uid"]]["text"] for c in cases}
        (od / "generation_dry_run_check.json").write_text(json.dumps({"n": len(cases), "text_identical_to_stored_g1": same, "log": log}, indent=2), encoding="utf-8")
        print(json.dumps({"dry_run_identical_to_stored_g1": same, "failed": len(still)}))
        return 0 if all(same.values()) and not still else 1
    (od / "generation_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    with open(od / "g1_reports_raw.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            rec = cache.get(c["uid"], hashes[c["uid"]])
            if rec:
                f.write(json.dumps({"uid": c["uid"], **{k: rec[k] for k in ("text", "done_reason", "model_returned", "prompt_tokens", "output_tokens", "prompt_may_be_truncated", "wall_seconds", "created_utc")}}, ensure_ascii=False) + "\n")
    print(json.dumps({k: log[k] for k in ("n_cached_success", "n_failed", "wall_seconds_this_run")}))
    return 0 if not still else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
