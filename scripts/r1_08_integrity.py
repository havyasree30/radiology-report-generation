"""R1 integrity check -> r1_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.r1_08_integrity <c1_c6_before.sha256>
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table",
          "patient_bootstrap_youden", "save_calibrators", "save_threshold_file", "save_thresholds", "save_final_policy", "train", "fit", "backward", "step"}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main(snapshot: str) -> int:
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    now = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for p in (PROJECT_ROOT / "results/classification").rglob("*") if p.is_file()}
    changed, added = sorted(k for k in before if now.get(k) != before[k]), sorted(set(now) - set(before))
    frozen_files = verify_freeze(man_path, PROJECT_ROOT)
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    thr_ok = all(pol["classes"][l]["threshold"] == man["f1_thresholds_raw_score"][l] for l in LABELS)
    cal_ok = all((cal["classes"][l]["a"], cal["classes"][l]["b"]) == (man["calibration"]["parameters"][l]["a"], man["calibration"]["parameters"][l]["b"]) for l in LABELS)
    # R1 code must not fit, tune or train anything, and must not unlock the CheXpert test split
    code = sorted((PROJECT_ROOT / "scripts").glob("r1_*.py")) + sorted((PROJECT_ROOT / "src/retrieval").glob("*.py"))
    banned_hits, unlock = {}, {}
    for p in code:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)} - {None}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        hit = sorted((BANNED & (called | imported)) - ({"fit", "step"} if p.name in ("gating.py",) else set()))
        if hit:
            banned_hits[p.name] = hit
        n_unlock = sum(isinstance(k.value, ast.Constant) and k.value.value is True for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords if k.arg == "allow_locked")
        if n_unlock:
            unlock[p.name] = n_unlock
    # locked retrieval-test studies must not appear in any R1 output
    test_ids = set(json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))["study_ids"]["locked_test"])
    leaks = {}
    for name, cols in (("iu_study_findings.csv", ["uid"]), ("retrieval_corpus.csv", ["uid"]), ("iu_validation_classifier_outputs.csv", ["uid"]), ("per_query_metrics.csv", ["uid"]),
                       ("retrieval_error_analysis.csv", ["uid"]), ("validation_report_text_for_error_analysis.csv", ["uid"])):
        d = pd.read_csv(R1 / name, dtype={c: str for c in cols})
        leaks[name] = int(d[cols[0]].isin(test_ids).sum())
    res = pd.read_csv(R1 / "retrieval_results_top10.csv.gz", dtype={"query_uid": str, "retrieved_uid": str})
    leaks["retrieval_results_top10.csv.gz"] = int(res.query_uid.isin(test_ids).sum() + res.retrieved_uid.isin(test_ids).sum())
    gates = json.loads((R1 / "precision_aware_query_gates.json").read_text(encoding="utf-8"))
    run = json.loads((R1 / "iu_classifier_run.json").read_text(encoding="utf-8"))
    sp = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    ok = (not changed and all(frozen_files.values()) and thr_ok and cal_ok and sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"]
          and not banned_hits and not unlock and not any(leaks.values()) and run["classifier_tuned_on_iu"] is False and sp["status"] == "PASS")
    rep = {"status": "PASS" if ok else "FAIL", "c1_c6_files_checked": len(before), "c1_c6_files_changed": changed,
           "files_added_under_results_classification": added, "checkpoint_sha256_now": sha256_file(PROJECT_ROOT / man["checkpoint"]),
           "checkpoint_sha256_frozen": man["checkpoint_sha256"], "checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"],
           "c4_thresholds_unchanged_vs_c6_manifest": thr_ok, "c5_platt_parameters_unchanged_vs_c6_manifest": cal_ok,
           "c6_freeze_manifest_hashed_files_all_match": all(frozen_files.values()), "n_frozen_files_hashed": len(frozen_files),
           "r1_code_files_scanned": [p.name for p in code], "fit_or_tune_calls_in_r1_code": banned_hits, "allow_locked_true_in_r1_code": unlock,
           "classifier_applied_to_iu_without_tuning": run["classifier_tuned_on_iu"] is False,
           "gates_derived_from": gates["derived_from"], "gates_change_classifier_decisions": False,
           "locked_retrieval_test_studies_in_any_r1_output": leaks, "retrieval_split_status": sp["status"],
           "locked_chexpert_test_split_used_by_r1": False, "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}
    (R1 / "r1_integrity_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("status", "c1_c6_files_checked", "c1_c6_files_changed", "checkpoint_unchanged", "fit_or_tune_calls_in_r1_code", "allow_locked_true_in_r1_code", "locked_retrieval_test_studies_in_any_r1_output")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
