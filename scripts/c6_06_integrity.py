"""C6 final integrity check -> c6_integrity_report.json

    .venv\Scripts\python.exe -m scripts.c6_06_integrity <c1_c5_before.sha256>
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
OUT = EXP / "c6_final_test"
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds",
          "candidate_table", "patient_bootstrap_youden", "save_calibrators", "save_threshold_file", "save_thresholds", "save_final_policy"}


def main(snapshot: str) -> int:
    man = json.loads((OUT / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "test_inference_run.json").read_text(encoding="utf-8"))
    # 1. C1-C5 files byte-identical to the pre-C6 snapshot
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    now = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p)
           for p in (PROJECT_ROOT / "results/classification").rglob("*") if p.is_file() and "c6_final_test" not in p.parts}
    changed = sorted(k for k in before if now.get(k) != before[k])
    added = sorted(set(now) - set(before))
    # 2. frozen values
    ck = PROJECT_ROOT / man["checkpoint"]
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    thr_same_policy = all(pol["classes"][l]["threshold"] == man["f1_thresholds_raw_score"][l] for l in LABELS)
    pp = pd.read_csv(OUT / "test_predictions_final_policy.csv.gz", nrows=2, float_precision="round_trip")
    thr_in_output = all(pp[f"{l}_threshold_raw"].iloc[0] == man["f1_thresholds_raw_score"][l] for l in LABELS)
    platt_same = all((cal["classes"][l]["a"], cal["classes"][l]["b"], cal["classes"][l]["calibrated_equivalent_threshold"])
                     == (man["calibration"]["parameters"][l]["a"], man["calibration"]["parameters"][l]["b"],
                         man["calibration"]["parameters"][l]["calibrated_equivalent_threshold"]) for l in LABELS)
    frozen_files = verify_freeze(OUT / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json", PROJECT_ROOT)
    # 3. no fitting / tuning code in C6, and the test split is unlocked in exactly one place
    fit_calls, unlock = {}, {}
    for p in sorted((PROJECT_ROOT / "scripts").glob("c6_*.py")) + [PROJECT_ROOT / "src/classification/final_test.py"]:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        called = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        fit_calls[p.name] = sorted(BANNED & (called | imported))
    for d in ("src", "scripts"):
        for p in (PROJECT_ROOT / d).rglob("*.py"):
            n = sum(isinstance(k.value, ast.Constant) and k.value.value is True for c in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
                    if isinstance(c, ast.Call) for k in c.keywords if k.arg == "allow_locked")
            if n:
                unlock[str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")] = n
    # 4. ordering / one-shot
    created, finished = man["created_utc"], run["finished_utc"]
    order = {"freeze_manifest_created_utc": created, "test_inference_finished_utc": finished, "manifest_before_inference": created < finished,
             "manifest_sha256_recorded_by_inference_matches": run["freeze_manifest_sha256"] == sha256_file(OUT / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json"),
             "n_inference_passes": run["n_inference_passes"],
             "manifest_mtime_before_prediction_files": (OUT / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json").stat().st_mtime < (OUT / "test_raw_predictions.npz").stat().st_mtime}
    rep = json.loads((OUT / "validation_reproduction.json").read_text(encoding="utf-8"))
    checks = {
        "c1_c5_files_checked": len(before), "c1_c5_files_changed": changed, "files_added_under_results_classification_outside_c6": added,
        "checkpoint_sha256_now": sha256_file(ck), "checkpoint_sha256_frozen": man["checkpoint_sha256"],
        "checkpoint_unchanged": sha256_file(ck) == man["checkpoint_sha256"] == run["checkpoint_sha256_at_inference"],
        "c4_thresholds_unchanged_vs_manifest": thr_same_policy, "thresholds_used_in_test_output_equal_manifest": thr_in_output,
        "c5_platt_parameters_unchanged_vs_manifest": platt_same, "all_frozen_files_match_manifest_hashes": all(frozen_files.values()),
        "n_frozen_files_hashed": len(frozen_files), "banned_fit_or_tune_calls_in_c6_code": fit_calls,
        "allow_locked_true_call_sites_in_repository": unlock, "ordering": order,
        "validation_reference_reproduced_stored_c2_c5": rep["reproduced"],
        "test_labels_used_for": ["metric computation (c6_03_evaluate)", "bootstrap resampling of metrics", "descriptive error listing"],
        "test_labels_not_used_for": ["thresholds", "calibration", "checkpoint choice", "loss choice", "rule changes", "any parameter"],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()}
    ok = (not changed and checks["checkpoint_unchanged"] and thr_same_policy and thr_in_output and platt_same and all(frozen_files.values())
          and not any(fit_calls.values()) and list(unlock) == ["scripts/c6_02_test_inference.py"] and order["manifest_before_inference"]
          and order["manifest_sha256_recorded_by_inference_matches"] and order["manifest_mtime_before_prediction_files"]
          and order["n_inference_passes"] == 1 and rep["reproduced"])
    checks["status"] = "PASS" if ok else "FAIL"
    (OUT / "c6_integrity_report.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({k: checks[k] for k in ("status", "c1_c5_files_checked", "c1_c5_files_changed", "checkpoint_unchanged", "allow_locked_true_call_sites_in_repository")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
