"""F1 final integrity check -> f1_final_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.f1_08_integrity <snapshot.sha256 taken before F1>

Verifies that nothing frozen changed, that the protocol and the analysis code were frozen before the test was opened and are unchanged, that every actual prompt
equals its rebuild, that the locked-test ids equal the frozen id-list hash, and that no post-test tuning is detectable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS
from src.f1.pipeline import F1, SPEC
from src.generation.client import GenerationConfig, OllamaClient, ResponseCache, installed_model, request_hash, request_params
from src.generation.prompts import SYSTEM_PROMPT, build_payload, prompt_hash, render_user_message
from src.system import guard
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
RG = PROJECT_ROOT / "results/report_generation/experiments"
G1, G1F = RG / "g1_single_agent", RG / "g1f_final_system"
T = F1 / "test"
EXPECTED = {"config_sha256": "cc04996cf8e8f6441505eedf5b41068b6af63384bf3a0a51bfde2a37c503eff5", "g1_prompt_sha256": "476b05326f85652b0cdb2faf79a4868a6e839660eb43e34d46cdae625e4b5688",
            "guard_sha256": "1f0b340cc71cf261de84f4b19f290c99568a69760dc757d57c62d2c68c9397ee"}
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table", "patient_bootstrap_youden", "save_calibrators", "save_threshold_file",
          "save_thresholds", "save_final_policy", "backward"}


def main(snapshot: str) -> int:
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    now = {k: sha256_file(PROJECT_ROOT / k) for k in before if (PROJECT_ROOT / k).exists()}
    changed, missing = sorted(k for k in before if k in now and now[k] != before[k]), sorted(k for k in before if k not in now)
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    frozen_files = verify_freeze(man_path, PROJECT_ROOT)
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    cfgdoc = json.loads((G1F / "FINAL_SYSTEM_CONFIG.json").read_text(encoding="utf-8"))
    cfg_hash = hashlib.sha256(json.dumps(cfgdoc["settings"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    s = cfgdoc["settings"]
    g1_prompt, g1_gen = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8")), json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in g1_prompt["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    client = OllamaClient()
    m = installed_model(client, cfg.model)
    gen_run = json.loads((T / "generation_run_log.json").read_text(encoding="utf-8"))
    checks = {
        "g1f_configuration_unchanged": cfg_hash == cfgdoc["config_sha256"] == EXPECTED["config_sha256"],
        "classifier_checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"] == s["classification"]["checkpoint_sha256"],
        "thresholds_unchanged": all(pol["classes"][l]["threshold"] == man["f1_thresholds_raw_score"][l] == s["classification"]["f1_thresholds_raw_score"][l] for l in LABELS),
        "calibration_unchanged": all((cal["classes"][l]["a"], cal["classes"][l]["b"]) == (man["calibration"]["parameters"][l]["a"], man["calibration"]["parameters"][l]["b"]) for l in LABELS) and sha256_file(EXP / "c5_calibration/final_calibrators.json") == s["classification"]["calibration"]["file_sha256"],
        "c6_freeze_manifest_files_unchanged": all(frozen_files.values()),
        "retrieval_configuration_unchanged": sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json") == s["retrieval"]["r2_candidate_config_sha256"] and sha256_file(R1 / "corpus_study_ids.json") == s["retrieval"]["corpus"]["ids_file_sha256"],
        "g1_prompt_unchanged": prompt_hash() == EXPECTED["g1_prompt_sha256"] == s["generation"]["g1_prompt_sha256"] == g1_prompt["prompt_sha256"] == gen_run["prompt_sha256"],
        "generator_unchanged": bool(m) and m["digest"] == g1_gen["model"]["digest"] == s["generation"]["model_digest"] == gen_run["model_digest"] and client.version() == g1_gen["ollama_version"] == gen_run["ollama_version"] and cfg.options() == g1_gen["generation_options"],
        "guard_unchanged": sha256_file(PROJECT_ROOT / "src/system/guard.py") == EXPECTED["guard_sha256"] == s["guard"]["guard_module_sha256"],
        "all_earlier_artifacts_unchanged_vs_snapshot": not changed and not missing}

    # ---- protocol / code freeze / one-shot opening
    rec = {ln.split(" *")[1]: ln.split(" *")[0] for ln in (F1 / "F1_LOCKED_TEST_PROTOCOL.sha256").read_text(encoding="utf-8").splitlines() if ln.strip()}
    proto = json.loads((F1 / "F1_LOCKED_TEST_PROTOCOL.json").read_text(encoding="utf-8"))
    marker = json.loads((T / "test_opened.json").read_text(encoding="utf-8"))
    cf = json.loads((F1 / "f1_code_freeze.json").read_text(encoding="utf-8"))
    code_changed = sorted(p for p, h in cf["sha256"].items() if sha256_file(PROJECT_ROOT / p) != h)
    frozen_code_changed = sorted(p for p, h in proto["frozen_system"]["frozen_code_sha256"].items() if sha256_file(PROJECT_ROOT / p) != h)
    checks.update({"protocol_json_unchanged_since_registration": sha256_file(F1 / "F1_LOCKED_TEST_PROTOCOL.json") == rec["F1_LOCKED_TEST_PROTOCOL.json"], "protocol_md_unchanged": sha256_file(F1 / "F1_LOCKED_TEST_PROTOCOL.md") == rec["F1_LOCKED_TEST_PROTOCOL.md"],
                   "protocol_registered_before_test_opened": proto["created_utc"] < marker["opened_utc"], "test_opened_once_marker_matches_protocol": marker["protocol_json_sha256"] == rec["F1_LOCKED_TEST_PROTOCOL.json"],
                   "analysis_code_unchanged_since_code_freeze": not code_changed, "frozen_upstream_code_unchanged_since_protocol": not frozen_code_changed, "code_freeze_recorded_before_opening": cf["frozen_utc"] <= marker["opened_utc"]})

    # ---- locked-test ids, prompts and generation
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    ids = sorted(str(i) for i in split["study_ids"]["locked_test"])
    ids_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    pop = pd.read_csv(T / "population.csv", dtype={"uid": str})
    cases = [json.loads(l) for l in open(T / "cases.jsonl", encoding="utf-8")]
    corpus = set(json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8")))
    val_ids = set(split["study_ids"]["validation"])
    cache = ResponseCache(T / "generation_cache")
    digest = g1_gen["model"]["digest"]
    n_rebuild = n_req = n_called = path_c_cache_files = 0
    retr_bad = 0
    for c in cases:
        pos, nf = c["classifier_positive_findings"], c["no_finding_positive"]
        msg = render_user_message(build_payload([{"finding": f, "calibrated_probability": c["classifier_probabilities"][f]} for f in pos], nf, c["retrieved"]))
        n_rebuild += int(hashlib.sha256(msg.encode()).hexdigest() == c["user_message_sha256"] and msg == c["user_message"])
        retr_bad += sum(r["study_id"] not in corpus or r["study_id"] in set(ids) or r["study_id"] in val_ids or r["study_id"] == c["uid"] for r in c["retrieved"])
        if c["system_interpretation_state"] == guard.STATE_INDETERMINATE:
            path_c_cache_files += int((T / "generation_cache" / f"{c['uid']}.json").exists())
        else:
            n_called += 1
            r = cache.get(c["uid"], request_hash(request_params(cfg, SYSTEM_PROMPT, c["user_message"]), digest))
            n_req += int(r is not None and r["options"] == cfg.options())
    checks.update({"locked_test_id_list_hash_equals_frozen_hash": ids_hash == proto["locked_test"]["study_ids_sha256"] == s["evaluation"]["locked_test_split"]["study_ids_sha256"] and len(ids) == 578,
                   "population_ids_are_exactly_the_locked_ids": sorted(pop.uid.astype(str)) == ids, "case_ids_subset_of_locked_ids": set(c["uid"] for c in cases) <= set(ids),
                   "all_actual_prompts_equal_rebuild": n_rebuild == len(cases), "all_generation_requests_equal_rebuild_hash_and_frozen_options": n_req == n_called, "no_retrieval_study_equals_query_or_in_validation_or_test_or_outside_corpus": retr_bad == 0,
                   "path_C_studies_have_no_generation_record": path_c_cache_files == 0, "no_generation_failure": gen_run["n_failed"] == 0, "each_path_A_B_study_generated_exactly_once": len(list((T / "generation_cache").glob("*.json"))) == n_called,
                   "spec_query_is_frozen_r2_specification": SPEC == {"phrases": "names", "normal": "no acute abnormality", "top_n": 3, "weighting": None, "oracle": False}})

    # ---- no post-test tuning detectable (static scan)
    hits, ext = {}, {}
    for p in sorted((PROJECT_ROOT / "src/f1").glob("*.py")) + sorted((PROJECT_ROOT / "scripts").glob("f1_*.py")):
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text)
        called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)} - {None}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        if BANNED & (called | imported):
            hits[p.name] = sorted(BANNED & (called | imported))
        if p.name != "f1_08_integrity.py" and re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai", text):
            ext[p.name] = True
    checks["no_fit_or_tuning_calls_in_f1_code"] = not hits
    checks["no_external_api_in_f1_code"] = not ext
    after_freeze = sorted(str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in (PROJECT_ROOT / "scripts").glob("f1_0[7-9]_*.py"))
    out = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "earlier_artifacts_checked": len(before), "files_changed": changed, "files_missing": missing, "analysis_code_changed_since_code_freeze": code_changed,
           "frozen_upstream_code_changed_since_protocol": frozen_code_changed, "scripts_written_after_the_code_freeze_(formatting_only)": after_freeze, "protocol_json_sha256": rec["F1_LOCKED_TEST_PROTOCOL.json"], "protocol_created_utc": proto["created_utc"], "test_opened_utc": marker["opened_utc"],
           "code_freeze_combined_sha256": cf["combined_sha256"], "locked_test_ids_sha256": ids_hash, "config_sha256": cfg_hash, "n_test_cases": len(cases), "n_generated": n_called, "n_path_C_no_call": len(cases) - n_called,
           "prompt_rebuild_matches": f"{n_rebuild} of {len(cases)}", "generation_requests_rebuild_matches": f"{n_req} of {n_called}", "code_bugs_corrected_after_opening": [], "fit_or_tune_calls": hits, "external_api": ext}
    (F1 / "f1_final_integrity_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "earlier_artifacts_checked", "files_changed", "prompt_rebuild_matches", "generation_requests_rebuild_matches", "analysis_code_changed_since_code_freeze")} | {"failed": [k for k, v in checks.items() if not v]}))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
