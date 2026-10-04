"""G1A integrity check -> g1a_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.g1a_07_integrity <snapshot.sha256>
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
from src.generation.client import GenerationConfig, OllamaClient, installed_model, request_hash, request_params
from src.generation.metrics import norm_sentence, sentences
from src.generation.prompts_g1a import SYSTEM_PROMPT, assert_payload_clean_g1a, classifier_block, prompt_hash, render_user_message_g1a
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
G1_COMMIT = "9714f50d3f72587e91bf30cef238d4e5240f55e6"
G1_CODE = ["scripts/g1_01_build_cases.py", "scripts/g1_02_smoke_test_and_freeze.py", "scripts/g1_03_generate.py", "scripts/g1_04_evaluate.py", "scripts/g1_05_examples_and_schema.py", "scripts/g1_06_figures_tables.py",
           "scripts/g1_07_write_docs.py", "scripts/g1_08_integrity.py", "src/generation/baseline.py", "src/generation/client.py", "src/generation/evidence.py", "src/generation/extraction.py", "src/generation/metrics.py",
           "src/generation/parse.py", "src/generation/prompts.py", "src/generation/schema.py", "configs/generation/finding_extraction_changes.yaml", "tests/test_generation_g1.py"]
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table", "patient_bootstrap_youden", "save_calibrators",
          "save_threshold_file", "save_thresholds", "save_final_policy", "load_checkpoint", "backward"}
RETRIEVAL_WORDS = r"retriev|dense rank|BM25|fusion score|EVIDENCE SUMMARY|Retrieved support|study \d+"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main(snapshot: str) -> int:
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    roots = [PROJECT_ROOT / "results/classification", PROJECT_ROOT / "results/retrieval", G1]
    now = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for r in roots for p in r.rglob("*") if p.is_file()}
    changed, missing = sorted(k for k in before if k in now and now[k] != before[k]), sorted(k for k in before if k not in now)
    frozen_files = verify_freeze(man_path, PROJECT_ROOT)
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    thr_ok = all(pol["classes"][l]["threshold"] == man["f1_thresholds_raw_score"][l] for l in LABELS)
    cal_ok = all((cal["classes"][l]["a"], cal["classes"][l]["b"]) == (man["calibration"]["parameters"][l]["a"], man["calibration"]["parameters"][l]["b"]) for l in LABELS)
    g1_integ = json.loads((G1 / "retrieval_top5_integrity.json").read_text(encoding="utf-8"))
    r2_cfg_ok = g1_integ["retrieval_config"]["sha256_of_r2_candidate_config"] == sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json")
    g1_code_unchanged = {p: subprocess.run(["git", "diff", "--quiet", G1_COMMIT, "--", p], cwd=PROJECT_ROOT).returncode == 0 for p in G1_CODE}

    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    frozen = json.loads((G1A / "g1a_prompt_frozen.json").read_text(encoding="utf-8"))
    meta = json.loads((G1A / "g1a_generator_metadata.json").read_text(encoding="utf-8"))
    run = json.loads((G1A / "generation_run_log.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in frozen["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    m = installed_model(client, cfg.model)
    same_model_now = {"ollama_version": client.version() == g1_gen["ollama_version"], "model_digest": bool(m) and m["digest"] == g1_gen["model"]["digest"], "options": cfg.options() == g1_gen["generation_options"],
                      "recorded_in_g1a_metadata": all(meta["same_generator_as_g1"].values()), "run_log_digest": run["model_digest"] == g1_gen["model"]["digest"], "run_log_version": run["ollama_version"] == g1_gen["ollama_version"]}

    g1_cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    cases = [json.loads(l) for l in open(G1A / "g1a_cases.jsonl", encoding="utf-8")]
    gcase = {c["uid"]: c for c in g1_cases}
    test_ids = set(json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))["study_ids"]["locked_test"])
    n_cached = sent_ok = bad_opts = ref_leaks = retr_leaks = block_ok = 0
    cache_dir = G1A / "generation_cache"
    for c in cases:
        assert_payload_clean_g1a(c["payload"])
        msg = render_user_message_g1a(c["payload"])
        assert msg == c["user_message"]
        retr_leaks += int(bool(re.search(RETRIEVAL_WORDS, msg, re.IGNORECASE)))
        block_ok += int(classifier_block(msg) == classifier_block(gcase[c["uid"]]["user_message"]))
        nm = norm_sentence(msg)
        ref_leaks += sum(norm_sentence(s) in nm for s in sentences(gcase[c["uid"]]["reference"]["combined"]) if len(norm_sentence(s).split()) >= 6)
        f = cache_dir / f"{c['uid']}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            n_cached += 1
            sent_ok += int(rec["request_hash"] == request_hash(request_params(cfg, SYSTEM_PROMPT, msg), g1_gen["model"]["digest"]))
            bad_opts += int(rec["options"] != cfg.options())
    leaks = {"g1a_cases.jsonl": sum(c["uid"] in test_ids for c in cases)}
    for name in ("g1a_per_study_results.csv",):
        leaks[name] = int(pd.read_csv(G1A / name, dtype={"uid": str}).uid.isin(test_ids).sum())
    code = sorted((PROJECT_ROOT / "scripts").glob("g1a_*.py")) + [PROJECT_ROOT / "src/generation/prompts_g1a.py", PROJECT_ROOT / "src/generation/study_eval.py"]
    hits, unlock, ext_api = {}, {}, {}
    for p in code:
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text)
        called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)} - {None}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        if BANNED & (called | imported):
            hits[p.name] = sorted(BANNED & (called | imported))
        if sum(isinstance(k.value, ast.Constant) and k.value.value is True for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords if k.arg == "allow_locked"):
            unlock[p.name] = True
        if p.name != "g1a_07_integrity.py" and re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai", text):
            ext_api[p.name] = True
    ok = (not changed and not missing and all(frozen_files.values()) and thr_ok and cal_ok and r2_cfg_ok and all(g1_code_unchanged.values()) and all(same_model_now.values())
          and block_ok == len(cases) and retr_leaks == 0 and ref_leaks == 0 and n_cached == len(cases) == sent_ok and bad_opts == 0 and not any(leaks.values()) and run["n_failed"] == 0
          and frozen["prompt_sha256"] == prompt_hash() and not hits and not unlock and not ext_api)
    out = {"status": "PASS" if ok else "FAIL", "earlier_artifacts_checked": len(before), "files_changed": changed, "files_missing": missing, "c6_freeze_manifest_files_all_match": all(frozen_files.values()),
           "checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"], "c4_thresholds_unchanged": thr_ok, "c5_platt_parameters_unchanged": cal_ok,
           "r2_candidate_config_unchanged": r2_cfg_ok, "g1_artifacts_unchanged_incl_generation_cache": not any(k.startswith("results/report_generation/experiments/g1_single_agent") for k in changed + missing),
           "g1_code_unchanged_since_commit": {"commit": G1_COMMIT, "all_unchanged": all(g1_code_unchanged.values()), "changed": [p for p, v in g1_code_unchanged.items() if not v]},
           "same_generator_as_g1": same_model_now, "model_tag": cfg.model, "model_digest": g1_gen["model"]["digest"], "n_cases": len(cases), "n_responses_cached": n_cached, "generation_failures": run["n_failed"],
           "responses_whose_request_equals_the_checked_message": sent_ok, "responses_with_options_different_from_g1": bad_opts, "classifier_block_identical_to_g1": block_ok,
           "messages_containing_retrieval_wording": retr_leaks, "reference_sentences_in_any_generator_input": ref_leaks, "prompt_sha256_equals_frozen": frozen["prompt_sha256"] == prompt_hash(),
           "locked_retrieval_test_studies_in_g1a_outputs": leaks, "external_api_used_in_g1a_code": ext_api, "fit_or_tune_calls_in_g1a_code": hits, "allow_locked_true_in_g1a_code": unlock,
           "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}
    (G1A / "g1a_integrity_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "earlier_artifacts_checked", "files_changed", "n_responses_cached", "generation_failures", "responses_whose_request_equals_the_checked_message", "messages_containing_retrieval_wording",
                                          "reference_sentences_in_any_generator_input", "same_generator_as_g1", "locked_retrieval_test_studies_in_g1a_outputs")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
