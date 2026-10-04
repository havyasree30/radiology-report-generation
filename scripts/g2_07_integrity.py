"""G2 integrity check -> g2_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.g2_07_integrity <snapshot.sha256>
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS
from src.generation.client import GenerationConfig, OllamaClient, installed_model
from src.generation.g2_agents import PROMPTS, agent_config, prompt_hashes, request_hash_for
from src.generation.g2_pipeline import AGENTS
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
G2 = PROJECT_ROOT / "results/report_generation/experiments/g2_multi_agent"
COMMITS = {"G1": "9714f50d3f72587e91bf30cef238d4e5240f55e6", "G1A": "d4f78fca8601bb8a5146b307d4dcc74cd360d1f6", "G1B": "26dc0b34514d825710cf0d5a22bb59b33ae6cf43"}
G1_CODE = ["scripts/g1_01_build_cases.py", "scripts/g1_02_smoke_test_and_freeze.py", "scripts/g1_03_generate.py", "scripts/g1_04_evaluate.py", "scripts/g1_05_examples_and_schema.py", "scripts/g1_06_figures_tables.py",
           "scripts/g1_07_write_docs.py", "scripts/g1_08_integrity.py", "src/generation/baseline.py", "src/generation/client.py", "src/generation/evidence.py", "src/generation/extraction.py", "src/generation/metrics.py",
           "src/generation/parse.py", "src/generation/prompts.py", "src/generation/schema.py", "configs/generation/finding_extraction_changes.yaml", "tests/test_generation_g1.py"]
G1A_CODE = [f"scripts/g1a_0{i}_{n}.py" for i, n in ((1, "build_cases"), (2, "smoke_test_and_freeze"), (3, "generate"), (4, "evaluate"), (5, "figures_tables"), (6, "write_docs"), (7, "integrity"))] + [
    "src/generation/prompts_g1a.py", "src/generation/study_eval.py", "tests/test_g1a.py"]
G1B_CODE = [f"scripts/g1b_0{i}_{n}.py" for i, n in ((1, "build_cases"), (2, "generate"), (3, "evaluate"), (4, "figures_tables"), (5, "write_docs"), (6, "integrity"))] + ["tests/test_g1b.py"]
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table", "patient_bootstrap_youden", "save_calibrators",
          "save_threshold_file", "save_thresholds", "save_final_policy", "load_checkpoint", "backward"}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main(snapshot: str) -> int:
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    roots = [PROJECT_ROOT / "results/classification", PROJECT_ROOT / "results/retrieval", G1, G1A, G1B]
    now = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for r in roots for p in r.rglob("*") if p.is_file()}
    changed, missing = sorted(k for k in before if k in now and now[k] != before[k]), sorted(k for k in before if k not in now)
    frozen_files = verify_freeze(man_path, PROJECT_ROOT)
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    thr_ok = all(pol["classes"][l]["threshold"] == man["f1_thresholds_raw_score"][l] for l in LABELS)
    cal_ok = all((cal["classes"][l]["a"], cal["classes"][l]["b"]) == (man["calibration"]["parameters"][l]["a"], man["calibration"]["parameters"][l]["b"]) for l in LABELS)
    g1_integ = json.loads((G1 / "retrieval_top5_integrity.json").read_text(encoding="utf-8"))
    r2_cfg_ok = g1_integ["retrieval_config"]["sha256_of_r2_candidate_config"] == sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json")
    unchanged = lambda commit, files: {p: subprocess.run(["git", "diff", "--quiet", commit, "--", p], cwd=PROJECT_ROOT).returncode == 0 for p in files}  # noqa: E731
    code_ok = {"G1": unchanged(COMMITS["G1"], G1_CODE), "G1A": unchanged(COMMITS["G1A"], G1A_CODE), "G1B": unchanged(COMMITS["G1B"], G1B_CODE)}

    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    g1_prompt = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    fz = json.loads((G2 / "g2_prompts_frozen.json").read_text(encoding="utf-8"))
    meta = json.loads((G2 / "g2_generator_metadata.json").read_text(encoding="utf-8"))
    run = json.loads((G2 / "generation_run_log.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in g1_prompt["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    m = installed_model(client, cfg.model)
    same_model = {"ollama_version": client.version() == g1_gen["ollama_version"], "model_digest": bool(m) and m["digest"] == g1_gen["model"]["digest"], "model_tag": cfg.model == g1_gen["model"]["model_tag"],
                  "sampling_options": all(cfg.options()[k] == g1_gen["generation_options"][k] for k in ("temperature", "top_k", "top_p", "seed", "num_ctx")), "recorded_in_g2_metadata": all(meta["same_generator_as_g1"].values()),
                  "run_log_digest": run["model_digest"] == g1_gen["model"]["digest"], "run_log_version": run["ollama_version"] == g1_gen["ollama_version"]}
    prompts_ok = {"hashes_equal_frozen": prompt_hashes() == fz["prompt_sha256"], "run_log_hashes_equal_frozen": run["prompt_sha256"] == fz["prompt_sha256"], "frozen_before_bulk_generation": bool(fz["frozen_before_bulk_generation"]),
                  "frozen_timestamp_before_generation_start": fz["frozen_utc"] < run["started_utc"]}

    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    test_ids = set(split["study_ids"]["locked_test"])
    g1_cases = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}
    cases = [json.loads(l) for l in open(G2 / "g2_cases.jsonl", encoding="utf-8")]
    n_outputs = req_ok = opt_bad = top5_diff = clf_diff = ref_keys = 0
    calls = {a: 0 for a in AGENTS}
    for c in cases:
        g = g1_cases[c["uid"]]
        top5_diff += int(c["retrieved_study_ids"] != [r["study_id"] for r in g["retrieved"]])
        clf_diff += int(c["classifier_positive_findings"] != g["classifier_positive_findings"] or c["classifier_probabilities"] != g["classifier_probabilities"])
        ref_keys += int(bool({"reference", "truth", "truth_findings"} & set(c)))
        f = G2 / "agent_outputs" / f"{c['uid']}.json"
        if not f.exists():
            continue
        a = json.loads(f.read_text(encoding="utf-8"))
        n_outputs += 1
        for ag_name, msg in a["messages"].items():
            calls[ag_name] += 1
            req_ok += int(request_hash_for(cfg, ag_name, msg, g1_gen["model"]["digest"]) == a["request_hashes"][ag_name])
            opt_bad += int(agent_config(cfg, ag_name).options() != json.loads((G2 / "generation_cache" / ag_name / f"{c['uid']}.json").read_text(encoding="utf-8"))["options"])
    n_calls = sum(calls.values())
    leak = json.loads((G2 / "g2_leakage_checks.json").read_text(encoding="utf-8"))
    leaks = {"g2_cases.jsonl": sum(c["uid"] in test_ids for c in cases) + sum(i in test_ids for c in cases for i in c["retrieved_study_ids"]), "g2_per_study_results.csv": int(pd.read_csv(G2 / "g2_per_study_results.csv", dtype={"uid": str}).uid.isin(test_ids).sum())}
    code = sorted((PROJECT_ROOT / "scripts").glob("g2_*.py")) + [PROJECT_ROOT / "src/generation/g2_agents.py", PROJECT_ROOT / "src/generation/g2_pipeline.py"]
    hits, unlock, ext_api, loops = {}, {}, {}, {}
    for p in code:
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text)
        called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)} - {None}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        if BANNED & (called | imported):
            hits[p.name] = sorted(BANNED & (called | imported))
        if sum(isinstance(k.value, ast.Constant) and k.value.value is True for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords if k.arg == "allow_locked"):
            unlock[p.name] = True
        if p.name != "g2_07_integrity.py" and re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai", text):
            ext_api[p.name] = True
        if p.name == "g2_pipeline.py":
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_case")
            if any(isinstance(n, (ast.While, ast.For)) for n in ast.walk(fn)):
                loops[p.name] = True
    max_calls_per_study = max(len(json.loads(f.read_text(encoding="utf-8"))["messages"]) for f in (G2 / "agent_outputs").glob("*.json"))
    ok = (not changed and not missing and all(frozen_files.values()) and thr_ok and cal_ok and r2_cfg_ok and all(all(v.values()) for v in code_ok.values()) and all(same_model.values()) and all(prompts_ok.values())
          and n_outputs == len(cases) and req_ok == n_calls and opt_bad == 0 and top5_diff == clf_diff == ref_keys == 0 and leak["pass"] and not any(leaks.values()) and run["n_failed"] == 0 and max_calls_per_study <= 3
          and not hits and not unlock and not ext_api and not loops)
    out = {"status": "PASS" if ok else "FAIL", "earlier_artifacts_checked": len(before), "files_changed": changed, "files_missing": missing, "c6_freeze_manifest_files_all_match": all(frozen_files.values()),
           "checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"], "c4_thresholds_unchanged": thr_ok, "c5_platt_parameters_unchanged": cal_ok, "r2_candidate_config_unchanged": r2_cfg_ok,
           "retrieved_top5_ids_equal_frozen_g1_top5_mismatches": top5_diff, "classifier_outputs_different_from_g1": clf_diff,
           "g1_g1a_g1b_artifacts_unchanged": not any(k.startswith(("results/report_generation/experiments/g1_single_agent", "results/report_generation/experiments/g1a_no_retrieval", "results/report_generation/experiments/g1b_sham_retrieval")) for k in changed + missing),
           "code_unchanged_since_commit": {k: {"commit": COMMITS[k], "all_unchanged": all(v.values()), "changed": [p for p, x in v.items() if not x]} for k, v in code_ok.items()},
           "same_generator_as_g1": same_model, "model_tag": cfg.model, "model_digest": g1_gen["model"]["digest"], "prompts": prompts_ok, "prompt_sha256": fz["prompt_sha256"], "n_cases": len(cases), "n_completed": n_outputs,
           "generation_failures": run["n_failed"], "agent_calls_by_agent": calls, "max_agent_calls_per_study": max_calls_per_study, "agent_loops_in_pipeline": loops, "requests_equal_stored_messages": f"{req_ok} of {n_calls}",
           "responses_with_options_different_from_expected": opt_bad, "reference_keys_in_case_file": ref_keys, "leakage_checks": leak, "locked_retrieval_test_studies_in_g2_outputs": leaks, "external_api_used_in_g2_code": ext_api,
           "fit_or_tune_calls_in_g2_code": hits, "allow_locked_true_in_g2_code": unlock, "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}
    (G2 / "g2_integrity_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "earlier_artifacts_checked", "files_changed", "n_completed", "generation_failures", "agent_calls_by_agent", "requests_equal_stored_messages", "same_generator_as_g1", "prompts",
                                          "locked_retrieval_test_studies_in_g2_outputs", "code_unchanged_since_commit")}, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
