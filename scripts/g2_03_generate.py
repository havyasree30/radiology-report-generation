"""G2 step 3: bulk three-agent generation for the same 547 studies with the FROZEN prompts. Pre-flight: the generator equals the G1 generator and the
three prompt hashes equal the frozen hashes. Every agent response is cached per study (resumable); failures stay failures; nothing is fabricated.
GPU memory is sampled with nvidia-smi (read-only) to report peak usage.

    .venv\\Scripts\\python.exe -m scripts.g2_03_generate
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.generation.client import GenerationConfig, OllamaClient, OllamaTransientError, ResponseCache, hardware_record, installed_model
from src.generation.g2_agents import prompt_hashes
from src.generation.g2_pipeline import AGENTS, run_case
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G2 = PROJECT_ROOT / "results/report_generation/experiments/g2_multi_agent"


def gpu_used_mib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:                                              # noqa: BLE001
        return None


def main() -> int:
    frozen = json.loads((G2 / "g2_prompts_frozen.json").read_text(encoding="utf-8"))
    if frozen["prompt_sha256"] != prompt_hashes():
        raise SystemExit("STOP: the agent prompts differ from the frozen prompts")
    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    g1_prompt = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in g1_prompt["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
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
    cases = [json.loads(l) for l in open(G2 / "g2_cases.jsonl", encoding="utf-8")]
    caches = {a: ResponseCache(G2 / "generation_cache" / a) for a in AGENTS}
    outdir = G2 / "agent_outputs"
    outdir.mkdir(exist_ok=True)
    log = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_cases": len(cases), "prompt_sha256": frozen["prompt_sha256"], "model_tag": cfg.model, "model_digest": digest, "ollama_version": version}
    print(json.dumps(log), flush=True)
    failed, t0, peak = {}, time.time(), [gpu_used_mib() or 0.0]
    for i, c in enumerate(cases, 1):
        res = run_case(c, client, cfg, digest, caches)
        if res["failed"]:
            failed[c["uid"]] = {"agent": res["failed"], "record": res["records"].get(res["failed"])}
        else:
            rec = {"uid": c["uid"], "evidence": res["evidence"], "agent1_validation_flags": res["agent1_flags"], "agent2_draft": res["draft"], "agent3": {k: res["critic"][k] for k in ("action", "issues", "model_copy_identical_to_draft", "flags")},
                   "final_report": res["final"], "agent1_raw_text": res["records"].get(AGENTS[0], {}).get("text"), "agent2_raw_text": res["records"][AGENTS[1]]["text"], "agent3_raw_text": res["records"][AGENTS[2]]["text"],
                   "messages_sha256": {a: r["message_sha256"] for a, r in res["records"].items()}, "request_hashes": {a: r["request_hash"] for a, r in res["records"].items()},
                   "seconds": {a: r["wall_seconds"] for a, r in res["records"].items()}, "output_tokens": {a: r["output_tokens"] for a, r in res["records"].items()}, "prompt_tokens": {a: r["prompt_tokens"] for a, r in res["records"].items()},
                   "done_reasons": {a: r["done_reason"] for a, r in res["records"].items()}, "prompt_may_be_truncated": {a: r["prompt_may_be_truncated"] for a, r in res["records"].items()},
                   "messages": res["messages"]}
            (outdir / f"{c['uid']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        if i % 5 == 0:
            peak.append(gpu_used_mib() or 0.0)
        if i % 10 == 0 or i == len(cases):
            print(f"{i}/{len(cases)} done, failed={len(failed)}, elapsed={time.time() - t0:.0f}s", flush=True)
    done = [c["uid"] for c in cases if (outdir / f"{c['uid']}.json").exists()]
    log.update({"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_completed": len(done), "n_failed": len(cases) - len(done), "failed": failed, "wall_seconds_this_run": round(time.time() - t0, 1),
                "gpu_memory_used_mib_peak_sampled": max(peak), "gpu_memory_samples": len(peak), "hardware_at_end": hardware_record(client)})
    (G2 / "generation_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({k: log[k] for k in ("n_completed", "n_failed", "wall_seconds_this_run", "gpu_memory_used_mib_peak_sampled")}))
    return 0 if len(done) == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
