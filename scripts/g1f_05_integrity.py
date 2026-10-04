"""G1F integrity check -> g1f_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.g1f_05_integrity <snapshot.sha256>

`run_checks(snapshot, require_config)` is also called by g1f_03_freeze (require_config=False) before the final configuration is written.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS
from src.generation.client import GenerationConfig, OllamaClient, installed_model
from src.generation.parse import parse_report
from src.generation.prompts import prompt_hash
from src.system import guard
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
RG = PROJECT_ROOT / "results/report_generation/experiments"
G1, G1A, G1B, G2, G1F = RG / "g1_single_agent", RG / "g1a_no_retrieval", RG / "g1b_sham_retrieval", RG / "g2_multi_agent", RG / "g1f_final_system"
COMMITS = {"G1": "9714f50d3f72587e91bf30cef238d4e5240f55e6", "G1A": "d4f78fca8601bb8a5146b307d4dcc74cd360d1f6", "G1B": "26dc0b34514d825710cf0d5a22bb59b33ae6cf43", "G2": "0f583d58466d39825060d781d41d97eedec0c419"}
G1_CODE = ["scripts/g1_01_build_cases.py", "scripts/g1_02_smoke_test_and_freeze.py", "scripts/g1_03_generate.py", "scripts/g1_04_evaluate.py", "scripts/g1_05_examples_and_schema.py", "scripts/g1_06_figures_tables.py",
           "scripts/g1_07_write_docs.py", "scripts/g1_08_integrity.py", "src/generation/baseline.py", "src/generation/client.py", "src/generation/evidence.py", "src/generation/extraction.py", "src/generation/metrics.py",
           "src/generation/parse.py", "src/generation/prompts.py", "src/generation/schema.py", "configs/generation/finding_extraction_changes.yaml", "tests/test_generation_g1.py"]
G1A_CODE = [f"scripts/g1a_0{i}_{n}.py" for i, n in ((1, "build_cases"), (2, "smoke_test_and_freeze"), (3, "generate"), (4, "evaluate"), (5, "figures_tables"), (6, "write_docs"), (7, "integrity"))] + [
    "src/generation/prompts_g1a.py", "src/generation/study_eval.py", "tests/test_g1a.py"]
G1B_CODE = [f"scripts/g1b_0{i}_{n}.py" for i, n in ((1, "build_cases"), (2, "generate"), (3, "evaluate"), (4, "figures_tables"), (5, "write_docs"), (6, "integrity"))] + ["tests/test_g1b.py"]
G2_CODE = [f"scripts/g2_0{i}_{n}.py" for i, n in ((1, "build_cases"), (2, "smoke_test_and_freeze"), (3, "generate"), (4, "evaluate"), (5, "figures_tables"), (6, "write_docs"), (7, "integrity"))] + [
    "src/generation/g2_agents.py", "src/generation/g2_pipeline.py", "tests/test_g2.py"]
BANNED_CALLS = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table", "patient_bootstrap_youden", "save_calibrators",
                "save_threshold_file", "save_thresholds", "save_final_policy", "load_checkpoint", "backward"}
NO_MODEL_OR_RETRIEVAL = r"OllamaClient|generate_single|generate_agent|\.chat\(|faiss|SentenceTransformer|\bEngine\b|dense_ranking|bm25_ranking|embed_text"
LOCKED_TEST_ACCESS = r"locked_test|retrieval_split|official_test|test_predictions|c6_final_test/.*(pred|test)"
SPEC_FINDINGS = "Model output is indeterminate for this study."
SPEC_IMPRESSION = "Automated preliminary interpretation could not be established. Radiologist review is required."
CONFIG_PATH = G1F / "FINAL_SYSTEM_CONFIG.json"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def config_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_checks(snapshot: str, require_config: bool) -> dict:
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    roots = [PROJECT_ROOT / "results/classification", PROJECT_ROOT / "results/retrieval", G1, G1A, G1B, G2]
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
    code_ok = {"G1": unchanged(COMMITS["G1"], G1_CODE), "G1A": unchanged(COMMITS["G1A"], G1A_CODE), "G1B": unchanged(COMMITS["G1B"], G1B_CODE), "G2": unchanged(COMMITS["G2"], G2_CODE)}

    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    g1_prompt = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in g1_prompt["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    m = installed_model(client, cfg.model)
    generator = {"ollama_version": client.version() == g1_gen["ollama_version"], "model_digest": bool(m) and m["digest"] == g1_gen["model"]["digest"], "options": cfg.options() == g1_gen["generation_options"],
                 "g1_prompt_hash_equals_current_prompt_module": prompt_hash() == g1_prompt["prompt_sha256"]}

    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    raw = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")}
    out = pd.read_csv(G1F / "g1f_final_validation_outputs.csv", dtype={"study_id": str}).fillna("")
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    test_ids = set(split["study_ids"]["locked_test"])
    empty_ids = {c["uid"] for c in cases if not c["retrieved"]}
    guard_ok = (len(out) == len(cases) == 547 and [str(x) for x in out.study_id] == [c["uid"] for c in cases])
    n_routes_ok = n_ab_identical = n_c_wording = 0
    for r, c in zip(out.itertuples(), cases):
        n_routes_ok += int(r.system_interpretation_state == guard.route(c["classifier_positive_findings"], c["no_finding_positive"]))
        if r.routing_path in ("A", "B"):
            p = parse_report(raw[c["uid"]]["text"])
            n_ab_identical += int(r.final_findings == p["findings"] and r.final_impression == p["impression"] and r.source_g1_raw_report_sha256 == hashlib.sha256(raw[c["uid"]]["text"].encode()).hexdigest())
        else:
            n_c_wording += int(r.final_findings == SPEC_FINDINGS and r.final_impression == SPEC_IMPRESSION and not r.retrieval_invoked_in_final_pipeline and not r.llm_invoked_in_final_pipeline)
    paths = out.routing_path.value_counts().sort_index().to_dict()
    c_ids = set(out[out.routing_path == "C"].study_id.astype(str))
    wording = {"module_constants_equal_specification": guard.INDETERMINATE_FINDINGS == SPEC_FINDINGS and guard.INDETERMINATE_IMPRESSION == SPEC_IMPRESSION,
               "rendered_text_equals_specification": guard.INDETERMINATE_REPORT_TEXT == f"FINDINGS: {SPEC_FINDINGS}\nIMPRESSION: {SPEC_IMPRESSION}"}

    code = sorted((PROJECT_ROOT / "scripts").glob("g1f_*.py")) + [PROJECT_ROOT / "src/system/guard.py", PROJECT_ROOT / "tests/test_g1f.py"]
    hits, unlock, ext_api, model_calls, locked_access = {}, {}, {}, {}, {}
    for p in code:
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".py":
            tree = ast.parse(text)
            called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)} - {None}
            imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            if BANNED_CALLS & (called | imported):
                hits[p.name] = sorted(BANNED_CALLS & (called | imported))
            if sum(isinstance(k.value, ast.Constant) and k.value.value is True for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords if k.arg == "allow_locked"):
                unlock[p.name] = True
        if p.name not in ("g1f_05_integrity.py", "test_g1f.py"):
            if re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai", text):
                ext_api[p.name] = True
            if re.search(NO_MODEL_OR_RETRIEVAL, text):
                model_calls[p.name] = True
            if p.name != "g1f_03_freeze.py" and re.search(LOCKED_TEST_ACCESS, text):
                locked_access[p.name] = True
    leaks = {"g1f_final_validation_outputs.csv": int(out.study_id.astype(str).isin(test_ids).sum()), "g1f_per_study_results.csv": int(pd.read_csv(G1F / "g1f_per_study_results.csv", dtype={"uid": str}).uid.isin(test_ids).sum())}

    config_checks: dict = {}
    if require_config:
        cf = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        s = cf["settings"]
        config_checks = {"hash_recomputed_equals_stored": config_hash(s) == cf["config_sha256"], "checkpoint_sha256": s["classification"]["checkpoint_sha256"] == man["checkpoint_sha256"] == sha256_file(PROJECT_ROOT / man["checkpoint"]),
                         "thresholds": s["classification"]["f1_thresholds_raw_score"] == man["f1_thresholds_raw_score"], "calibration_file_sha256": s["classification"]["calibration"]["file_sha256"] == sha256_file(EXP / "c5_calibration/final_calibrators.json"),
                         "r2_config_sha256": s["retrieval"]["r2_candidate_config_sha256"] == sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json"), "top_k": s["retrieval"]["top_k"] == 5,
                         "g1_prompt_hash": s["generation"]["g1_prompt_sha256"] == g1_prompt["prompt_sha256"], "model_digest": s["generation"]["model_digest"] == g1_gen["model"]["digest"],
                         "guard_hash": s["guard"]["guard_module_sha256"] == sha256_file(PROJECT_ROOT / "src/system/guard.py"), "selected_architecture": s["generation"]["selected_architecture"] == "G1 Single-Agent RAG",
                         "no_locked_test_results_in_config": cf["locked_test_results_included"] is False and not re.search(r"test_metric|test_result|auroc_test", json.dumps(cf["settings"]), re.IGNORECASE)}
    ok = (not changed and not missing and all(frozen_files.values()) and thr_ok and cal_ok and r2_cfg_ok and all(all(v.values()) for v in code_ok.values()) and all(generator.values()) and guard_ok
          and n_routes_ok == len(cases) and paths == {"A": 269, "B": 229, "C": 49} and c_ids == empty_ids and n_ab_identical == paths["A"] + paths["B"] and n_c_wording == paths["C"] and all(wording.values())
          and int(out.new_llm_calls_in_g1f_run.sum()) == 0 and int(out.new_retrieval_calls_in_g1f_run.sum()) == 0 and not any(leaks.values()) and not hits and not unlock and not ext_api and not model_calls and not locked_access
          and all(config_checks.values()))
    return {"status": "PASS" if ok else "FAIL", "earlier_artifacts_checked": len(before), "files_changed": changed, "files_missing": missing, "c6_freeze_manifest_files_all_match": all(frozen_files.values()),
            "checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"], "c4_thresholds_unchanged": thr_ok, "c5_platt_parameters_unchanged": cal_ok, "r2_candidate_config_unchanged": r2_cfg_ok,
            "c1_c6_r1_r2_g1_g1a_g1b_g2_artifacts_unchanged": not changed and not missing, "code_unchanged_since_commit": {k: {"commit": COMMITS[k], "all_unchanged": all(v.values()), "changed": [p for p, x in v.items() if not x]} for k, v in code_ok.items()},
            "g1_generator_and_prompt_unchanged": generator, "model_tag": cfg.model, "model_digest": g1_gen["model"]["digest"], "g1_prompt_sha256": g1_prompt["prompt_sha256"], "n_studies": len(out), "routing_path_counts": paths, "routing_matches_guard_function": f"{n_routes_ok} of {len(cases)}",
            "path_C_equals_the_49_empty_classifier_outputs": c_ids == empty_ids, "path_A_B_reports_byte_identical_to_stored_g1": f"{n_ab_identical} of {paths['A'] + paths['B']}", "path_C_exact_wording_and_no_calls": f"{n_c_wording} of {paths['C']}",
            "indeterminate_wording": wording, "new_llm_calls_in_g1f_run": int(out.new_llm_calls_in_g1f_run.sum()), "new_retrieval_calls_in_g1f_run": int(out.new_retrieval_calls_in_g1f_run.sum()),
            "locked_retrieval_test_studies_in_g1f_outputs": leaks, "g1f_code_references_to_locked_test_files_or_ids_outside_the_freeze_script": locked_access, "g1f_code_with_model_or_retrieval_calls": model_calls, "external_api_used_in_g1f_code": ext_api,
            "fit_or_tune_calls_in_g1f_code": hits, "allow_locked_true_in_g1f_code": unlock, "final_system_config_checks": config_checks if require_config else "not yet written", "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}


def main(snapshot: str) -> int:
    out = run_checks(snapshot, require_config=CONFIG_PATH.exists())
    (G1F / "g1f_integrity_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "earlier_artifacts_checked", "files_changed", "routing_path_counts", "path_C_equals_the_49_empty_classifier_outputs", "path_A_B_reports_byte_identical_to_stored_g1", "new_llm_calls_in_g1f_run",
                                          "g1_generator_and_prompt_unchanged", "locked_retrieval_test_studies_in_g1f_outputs", "final_system_config_checks", "code_unchanged_since_commit")}, default=str))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
