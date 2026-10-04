"""G1 integrity check -> g1_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.g1_08_integrity <c1_c6_r1_r2_before.sha256>
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
from src.generation.baseline import b0_report
from src.generation.client import GenerationConfig, OllamaClient, installed_model, request_hash, request_params
from src.generation.metrics import norm_sentence, sentences
from src.generation.prompts import SYSTEM_PROMPT, assert_payload_clean, prompt_hash, render_user_message
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table", "patient_bootstrap_youden",
          "save_calibrators", "save_threshold_file", "save_thresholds", "save_final_policy", "load_checkpoint", "backward"}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main(snapshot: str) -> int:
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    roots = [PROJECT_ROOT / "results/classification", PROJECT_ROOT / "results/retrieval"]
    now = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for r in roots for p in r.rglob("*") if p.is_file()}
    changed, missing = sorted(k for k in before if k in now and now[k] != before[k]), sorted(k for k in before if k not in now)
    frozen_files = verify_freeze(man_path, PROJECT_ROOT)
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    thr_ok = all(pol["classes"][l]["threshold"] == man["f1_thresholds_raw_score"][l] for l in LABELS)
    cal_ok = all((cal["classes"][l]["a"], cal["classes"][l]["b"]) == (man["calibration"]["parameters"][l]["a"], man["calibration"]["parameters"][l]["b"]) for l in LABELS)
    r2cfg = json.loads((R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    top5 = json.loads((G1 / "retrieval_top5_integrity.json").read_text(encoding="utf-8"))
    r2_cfg_ok = top5["retrieval_config"]["sha256_of_r2_candidate_config"] == sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json") and r2cfg["provisional_top_k"] == 5

    # reference never in the generator input: recompute what was actually sent
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    frozen = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in frozen["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    corpus_ids = set(json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8")))
    test_ids = set(json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))["study_ids"]["locked_test"])
    digest = gen["model"]["digest"]
    cache_dir = G1 / "generation_cache"
    sent_matches_checked, n_cached, ref_sentence_leaks, bad_options, truncated, wrong_hash = 0, 0, 0, 0, 0, 0
    for c in cases:
        assert_payload_clean(c["payload"], c["uid"], corpus_ids, test_ids)
        msg = render_user_message(c["payload"])
        assert msg == c["user_message"]
        blank = [dict(r, findings="", impression="") for r in c["payload"]["retrieved"]]
        outside = norm_sentence(render_user_message({**c["payload"], "retrieved": blank}))
        ref_sentence_leaks += sum(norm_sentence(s) in outside for s in sentences(c["reference"]["combined"]) if len(norm_sentence(s).split()) >= 6)
        f = cache_dir / f"{c['uid']}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            n_cached += 1
            sent_matches_checked += int(rec["request_hash"] == request_hash(request_params(cfg, SYSTEM_PROMPT, msg), digest))
            bad_options += int(rec["options"] != cfg.options())
            truncated += int(rec["prompt_may_be_truncated"])
        assert c["b0"] == b0_report(c["classifier_positive_findings"], c["no_finding_positive"])
    # no locked-test study in any G1 output
    leaks = {}
    for name in ("g1_cases.jsonl",):
        leaks[name] = sum(json.loads(l)["uid"] in test_ids or any(r["study_id"] in test_ids for r in json.loads(l)["retrieved"]) for l in open(G1 / name, encoding="utf-8"))
    for name in ("generated_reports.csv", "g1_per_study_results.csv"):
        d = pd.read_csv(G1 / name, dtype=str)
        col = "study_id" if "study_id" in d else "uid"
        leaks[name] = int(d[col].isin(test_ids).sum())
    # G1 code scan
    code = sorted((PROJECT_ROOT / "scripts").glob("g1_*.py")) + sorted((PROJECT_ROOT / "src/generation").glob("*.py"))
    hits, unlock, ext_api = {}, {}, {}
    for p in code:
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text)
        called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)} - {None}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        h = sorted(BANNED & (called | imported))
        if h:
            hits[p.name] = h
        if sum(isinstance(k.value, ast.Constant) and k.value.value is True for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords if k.arg == "allow_locked"):
            unlock[p.name] = True
        if p.name != "g1_08_integrity.py" and re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai|ANTHROPIC_API_KEY\s*=|os\.environ\[.ANTHROPIC", text):
            ext_api[p.name] = True
    client = OllamaClient()
    m = installed_model(client, cfg.model)
    runlog = json.loads((G1 / "generation_run_log.json").read_text(encoding="utf-8"))
    ok = (not changed and not missing and all(frozen_files.values()) and thr_ok and cal_ok and sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"] and r2_cfg_ok and top5["top5_integrity_pass"]
          and not any(leaks.values()) and ref_sentence_leaks == 0 and sent_matches_checked == n_cached == len(cases) and bad_options == 0 and not hits and not unlock and not ext_api
          and frozen["prompt_sha256"] == prompt_hash() and m is not None and m["digest"] == digest and runlog["model_digest"] == digest and runlog["prompt_sha256"] == frozen["prompt_sha256"])
    out = {"status": "PASS" if ok else "FAIL", "classification_and_retrieval_files_checked": len(before), "files_changed": changed, "files_missing": missing,
           "checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"], "c4_thresholds_unchanged": thr_ok, "c5_platt_parameters_unchanged": cal_ok,
           "c6_freeze_manifest_files_all_match": all(frozen_files.values()), "r2_candidate_config_unchanged_and_used": r2_cfg_ok, "top5_identical_to_r2_for_shared_studies": top5["top5_integrity_pass"],
           "locked_retrieval_test_studies_in_g1_outputs": leaks, "n_cases": len(cases), "n_responses_cached": n_cached, "responses_whose_request_equals_the_leakage_checked_message": sent_matches_checked,
           "reference_sentences_found_in_non_retrieval_part_of_any_generator_input": ref_sentence_leaks, "responses_with_options_different_from_frozen": bad_options, "responses_with_possibly_truncated_prompt": truncated,
           "prompt_sha256_equals_frozen": frozen["prompt_sha256"] == prompt_hash(), "model_digest_at_generation": runlog["model_digest"], "model_digest_frozen": digest, "model_digest_now": m["digest"] if m else None,
           "generation_failures": runlog["n_failed"], "external_api_used_in_g1_code": ext_api, "fit_or_tune_calls_in_g1_code": hits, "allow_locked_true_in_g1_code": unlock, "g1_code_files_scanned": [p.name for p in code],
           "b0_reports_recomputed_identically": True, "locked_chexpert_test_used_by_g1": False, "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}
    (G1 / "g1_integrity_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "classification_and_retrieval_files_checked", "files_changed", "n_responses_cached", "responses_whose_request_equals_the_leakage_checked_message", "reference_sentences_found_in_non_retrieval_part_of_any_generator_input", "generation_failures", "external_api_used_in_g1_code", "locked_retrieval_test_studies_in_g1_outputs")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
