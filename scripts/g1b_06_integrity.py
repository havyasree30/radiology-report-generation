"""G1B integrity check -> g1b_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.g1b_06_integrity <snapshot.sha256>
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
from src.generation.prompts import SYSTEM_PROMPT, assert_payload_clean, prompt_hash, render_user_message
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
G1_COMMIT = "9714f50d3f72587e91bf30cef238d4e5240f55e6"
G1A_COMMIT = "d4f78fca8601bb8a5146b307d4dcc74cd360d1f6"
G1_CODE = ["scripts/g1_01_build_cases.py", "scripts/g1_02_smoke_test_and_freeze.py", "scripts/g1_03_generate.py", "scripts/g1_04_evaluate.py", "scripts/g1_05_examples_and_schema.py", "scripts/g1_06_figures_tables.py",
           "scripts/g1_07_write_docs.py", "scripts/g1_08_integrity.py", "src/generation/baseline.py", "src/generation/client.py", "src/generation/evidence.py", "src/generation/extraction.py", "src/generation/metrics.py",
           "src/generation/parse.py", "src/generation/prompts.py", "src/generation/schema.py", "configs/generation/finding_extraction_changes.yaml", "tests/test_generation_g1.py"]
G1A_CODE = [f"scripts/g1a_0{i}_{n}.py" for i, n in ((1, "build_cases"), (2, "smoke_test_and_freeze"), (3, "generate"), (4, "evaluate"), (5, "figures_tables"), (6, "write_docs"), (7, "integrity"))] + [
    "src/generation/prompts_g1a.py", "src/generation/study_eval.py", "tests/test_g1a.py"]
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table", "patient_bootstrap_youden", "save_calibrators",
          "save_threshold_file", "save_thresholds", "save_final_policy", "load_checkpoint", "backward"}
SKELETON = re.compile(r"^(A\. |B\. |C\. |Final positive|No Finding:|Write the|\[\d\] study |None: no retrieval|\| |\(no classifier)")


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main(snapshot: str) -> int:
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    roots = [PROJECT_ROOT / "results/classification", PROJECT_ROOT / "results/retrieval", G1, G1A]
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
    g1_code, g1a_code = unchanged(G1_COMMIT, G1_CODE), unchanged(G1A_COMMIT, G1A_CODE)

    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    g1_prompt = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    fz = json.loads((G1B / "g1b_sham_mapping_frozen.json").read_text(encoding="utf-8"))
    meta = json.loads((G1B / "g1b_generator_metadata.json").read_text(encoding="utf-8"))
    run = json.loads((G1B / "generation_run_log.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in g1_prompt["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    m = installed_model(client, cfg.model)
    same_model = {"ollama_version": client.version() == g1_gen["ollama_version"], "model_digest": bool(m) and m["digest"] == g1_gen["model"]["digest"], "options": cfg.options() == g1_gen["generation_options"],
                  "recorded_in_g1b_metadata": all(meta["same_generator_as_g1"].values()), "run_log_digest": run["model_digest"] == g1_gen["model"]["digest"], "run_log_version": run["ollama_version"] == g1_gen["ollama_version"],
                  "prompt_sha256_equals_g1": prompt_hash() == g1_prompt["prompt_sha256"] == meta["prompt_sha256"] == run["prompt_sha256"]}

    g1_cases = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}
    cases = [json.loads(l) for l in open(G1B / "g1b_cases.jsonl", encoding="utf-8")]
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    test_ids = set(split["study_ids"]["locked_test"])
    pool = set(json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8")))
    cache_dir = G1B / "generation_cache"
    n_cached = sent_ok = bad_opts = own_study = true5 = out_of_corpus = in_test = revealed = clf_diff = empty_ok = empty_n = 0
    for c in cases:
        g = g1_cases[c["uid"]]
        assert_payload_clean(c["payload"], c["uid"], pool, test_ids)
        msg = render_user_message(c["payload"])
        assert msg == c["user_message"]
        ids = [r["study_id"] for r in c["retrieved"]]
        own_study += int(c["uid"] in ids)
        true5 += len(set(ids) & {r["study_id"] for r in g["retrieved"]})
        out_of_corpus += sum(i not in pool for i in ids)
        in_test += sum(i in test_ids for i in ids)
        revealed += sum(bool(re.search(r"sham|random|unrelated|control", ln, re.IGNORECASE)) for ln in msg.splitlines() if SKELETON.match(ln))
        clf_diff += int(c["classifier_positive_findings"] != g["classifier_positive_findings"] or c["classifier_probabilities"] != g["classifier_probabilities"])
        if not g["retrieved"]:
            empty_n += 1
            empty_ok += int(msg == g["user_message"])
        f = cache_dir / f"{c['uid']}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            n_cached += 1
            sent_ok += int(rec["request_hash"] == request_hash(request_params(cfg, SYSTEM_PROMPT, msg), g1_gen["model"]["digest"]))
            bad_opts += int(rec["options"] != cfg.options())
    mapping_ok = fz["mapping_sha256"] == sha256_file(G1B / "g1b_sham_mapping.csv")
    leaks = {"g1b_cases.jsonl": sum(c["uid"] in test_ids for c in cases), "g1b_per_study_results.csv": int(pd.read_csv(G1B / "g1b_per_study_results.csv", dtype={"uid": str}).uid.isin(test_ids).sum())}
    code = sorted((PROJECT_ROOT / "scripts").glob("g1b_*.py"))
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
        if p.name != "g1b_06_integrity.py" and re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai", text):
            ext_api[p.name] = True
    ok = (not changed and not missing and all(frozen_files.values()) and thr_ok and cal_ok and r2_cfg_ok and all(g1_code.values()) and all(g1a_code.values()) and all(same_model.values()) and mapping_ok
          and own_study == true5 == out_of_corpus == in_test == revealed == clf_diff == 0 and empty_ok == empty_n and n_cached == len(cases) == sent_ok and bad_opts == 0 and not any(leaks.values())
          and run["n_failed"] == 0 and fz["frozen_before_generation"] and not hits and not unlock and not ext_api)
    out = {"status": "PASS" if ok else "FAIL", "earlier_artifacts_checked": len(before), "files_changed": changed, "files_missing": missing, "c6_freeze_manifest_files_all_match": all(frozen_files.values()),
           "checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"], "c4_thresholds_unchanged": thr_ok, "c5_platt_parameters_unchanged": cal_ok,
           "r2_candidate_config_unchanged": r2_cfg_ok,
           "g1_and_g1a_artifacts_unchanged": not any(k.startswith(("results/report_generation/experiments/g1_single_agent", "results/report_generation/experiments/g1a_no_retrieval")) for k in changed + missing),
           "g1_code_unchanged_since_commit": {"commit": G1_COMMIT, "all_unchanged": all(g1_code.values()), "changed": [p for p, v in g1_code.items() if not v]},
           "g1a_code_unchanged_since_commit": {"commit": G1A_COMMIT, "all_unchanged": all(g1a_code.values()), "changed": [p for p, v in g1a_code.items() if not v]},
           "same_generator_and_prompt_as_g1": same_model, "model_tag": cfg.model, "model_digest": g1_gen["model"]["digest"], "n_cases": len(cases), "n_responses_cached": n_cached, "generation_failures": run["n_failed"],
           "responses_whose_request_equals_the_checked_message": sent_ok, "responses_with_options_different_from_g1": bad_opts, "sham_mapping_sha256": fz["mapping_sha256"], "sham_mapping_hash_matches_frozen": mapping_ok,
           "sham_selection_seed": fz["seed"], "sham_reports_equal_to_query_study": own_study, "sham_reports_overlapping_the_true_top5": true5, "sham_reports_outside_the_training_corpus": out_of_corpus,
           "sham_reports_in_locked_test": in_test, "message_lines_revealing_the_control": revealed, "classifier_outputs_different_from_g1": clf_diff, "empty_query_messages_identical_to_g1": f"{empty_ok} of {empty_n}",
           "locked_retrieval_test_studies_in_g1b_outputs": leaks, "external_api_used_in_g1b_code": ext_api, "fit_or_tune_calls_in_g1b_code": hits, "allow_locked_true_in_g1b_code": unlock,
           "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}
    (G1B / "g1b_integrity_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "earlier_artifacts_checked", "files_changed", "n_responses_cached", "generation_failures", "responses_whose_request_equals_the_checked_message", "sham_reports_overlapping_the_true_top5",
                                          "message_lines_revealing_the_control", "same_generator_and_prompt_as_g1", "locked_retrieval_test_studies_in_g1b_outputs", "g1_code_unchanged_since_commit", "g1a_code_unchanged_since_commit")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
