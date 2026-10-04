"""C4 finalization: write the selected operating policy and the downstream validation predictions.

Selected AFTER inspecting scripts.c4_operating_policy output (decision rationale:
OPERATING_POLICY_SELECTION.md): F1-optimal per-class thresholds + the No Finding consistency rule.

    .venv\\Scripts\\python.exe -m scripts.c4_finalize
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.classification.labels import LABELS
from src.classification.operating_points import apply_thresholds, load_threshold_file
from src.classification.operating_policy import RULE_NAME, apply_final_policy, apply_no_finding_rule, load_final_policy, save_final_policy
from src.classification.serialization import load_predictions_npz
from src.utils.config import PROJECT_ROOT

EXPD = PROJECT_ROOT / "results/classification/experiments"
C3, OUT = EXPD / "c3_threshold_optimization", EXPD / "c4_operating_policy"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main() -> int:
    info = json.loads((C3 / "input_verification.json").read_text(encoding="utf-8"))
    ck = info["checkpoint"]
    f1_path = C3 / "thresholds_f1.json"
    thr = load_threshold_file(f1_path)
    prov = {
        "policy_name": "f1_optimal_plus_no_finding_consistency",
        "model": "DenseNet121, 14 independent sigmoid outputs (multi-label)",
        "checkpoint": ck["path"], "checkpoint_sha256": ck["sha256_recomputed"], "checkpoint_epoch": ck["selected_epoch"],
        "loss": "sqrt_weighted_bce", "loss_config": ck["training_loss"],
        "threshold_method": "f1_optimal",
        "threshold_tie_rule": "exact F1 ties: higher recall, then higher threshold",
        "decision_rule": "positive if sigmoid score >= class threshold; scores are model scores, not calibrated probabilities",
        "dataset_split_for_selection": "validation", "n_validation_images": info["validation"]["n_images"],
        "n_validation_patients": info["validation"]["n_patients"], "evaluation_precision_policy": info["evaluation_precision_policy"],
        "label_policy": info["label_policy"], "locked_test_split_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "thresholds_source_file": f1_path.relative_to(PROJECT_ROOT).as_posix(), "thresholds_source_sha256": sha256(f1_path),
            "thresholds_youden_file": "results/classification/experiments/c3_threshold_optimization/thresholds_youden.json",
            "c2_study": "results/classification/experiments/LOSS_COMPARISON.md (C2-B selected provisionally)",
            "c2_results_commit": "a2de417", "c2_infrastructure_commit": "c1e4800",
            "c3_commit": "031edd6", "c3_report": "results/classification/experiments/c3_threshold_optimization/THRESHOLD_OPTIMIZATION.md",
            "c4_report": "results/classification/experiments/c4_operating_policy/OPERATING_POLICY_SELECTION.md",
            "code_commit_at_creation": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "working_tree_dirty_at_creation": bool(git("status", "--porcelain"))},
    }
    save_final_policy(OUT / "final_operating_policy.json", thr, prov)
    pol = load_final_policy(OUT / "final_operating_policy.json")
    assert np.array_equal(pol["thresholds"], thr)                               # full precision preserved

    pred = load_predictions_npz(EXPD / "c2b_sqrt_weighted_bce/evaluation/fp32_strict/validation_predictions.npz")
    p, raw = pred["probs"], pred["raw_labels"]
    raw_pred, final_pred = apply_final_policy(p, pol)
    cols = {"image_id": pred["paths"], "patient_id": pred["patient_ids"]}
    for j, lab in enumerate(LABELS):
        cols[f"{lab}_label"] = raw[:, j]                     # 1 / 0 / -1 (uncertain) / blank (NaN)
        cols[f"{lab}_score"] = p[:, j]
        cols[f"{lab}_threshold"] = np.full(len(p), thr[j])
        cols[f"{lab}_pred_raw"] = raw_pred[:, j].astype(int)
        cols[f"{lab}_pred_final"] = final_pred[:, j].astype(int)
    path = OUT / "validation_predictions_final_policy.csv.gz"
    pd.DataFrame(cols).to_csv(path, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})

    back = pd.read_csv(path, float_precision="round_trip")
    sc = back[[f"{l}_score" for l in LABELS]].to_numpy()
    rawb = back[[f"{l}_pred_raw" for l in LABELS]].to_numpy().astype(bool)
    finb = back[[f"{l}_pred_final" for l in LABELS]].to_numpy().astype(bool)
    checks = {"scores_bit_identical_to_c2b": bool(np.array_equal(sc, p)),
              "raw_preds_equal_threshold_rule": bool(np.array_equal(rawb, apply_thresholds(sc, thr))),
              "final_preds_equal_rule_applied_to_raw": bool(np.array_equal(finb, apply_no_finding_rule(rawb))),
              "only_no_finding_differs_between_raw_and_final": bool(np.array_equal(np.delete(rawb, 0, axis=1), np.delete(finb, 0, axis=1))),
              "no_finding_columns_present": all(c in back.columns for c in ("No Finding_pred_raw", "No Finding_pred_final")),
              "image_ids_identical": back["image_id"].tolist() == pred["paths"], "n_rows": len(back),
              "scores_not_normalised": bool(not np.isclose(sc.sum(axis=1), 1.0).all()), "rule": RULE_NAME}
    assert all(v for k, v in checks.items() if k not in ("n_rows", "rule")), checks
    (OUT / "final_policy_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
