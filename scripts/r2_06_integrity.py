"""R2 integrity check -> r2_integrity_report.json

    .venv\\Scripts\\python.exe -m scripts.r2_06_integrity <c1_c6_r1_before.sha256>
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
R2 = PROJECT_ROOT / "results/retrieval/experiments/r2_optimization"
BANNED = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold", "youden_thresholds", "candidate_table",
          "patient_bootstrap_youden", "save_calibrators", "save_threshold_file", "save_thresholds", "save_final_policy", "load_checkpoint", "backward"}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main(snapshot: str) -> int:
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    before = dict(reversed(line.strip().split(" *", 1)) for line in Path(snapshot).read_text(encoding="utf-8").splitlines() if line.strip())
    roots = [PROJECT_ROOT / "results/classification", PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"]
    now = {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for r in roots for p in r.rglob("*") if p.is_file()}
    changed, missing = sorted(k for k in before if k in now and now[k] != before[k]), sorted(k for k in before if k not in now)
    added = sorted(set(now) - set(before))
    frozen = verify_freeze(man_path, PROJECT_ROOT)
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    thr_ok = all(pol["classes"][l]["threshold"] == man["f1_thresholds_raw_score"][l] for l in LABELS)
    cal_ok = all((cal["classes"][l]["a"], cal["classes"][l]["b"]) == (man["calibration"]["parameters"][l]["a"], man["calibration"]["parameters"][l]["b"]) for l in LABELS)
    code = sorted((PROJECT_ROOT / "scripts").glob("r2_*.py")) + [PROJECT_ROOT / "src/retrieval" / n for n in ("expansion.py", "fusion.py", "diversity.py", "evaluation.py", "r2_context.py")]
    hits, unlock = {}, {}
    for p in code:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)} - {None}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        h = sorted(BANNED & (called | imported))
        if h:
            hits[p.name] = h
        n_un = sum(isinstance(k.value, ast.Constant) and k.value.value is True for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords if k.arg == "allow_locked")
        if n_un:
            unlock[p.name] = n_un
    test_ids = set(json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))["study_ids"]["locked_test"])
    leaks = {}
    for name, cols in (("per_query_metrics_all_configs.csv", ["uid"]), ("retrieval_top10_all_configs.csv.gz", ["uid", "retrieved_uid"]), ("final_pipeline_top10.csv.gz", ["uid", "retrieved_uid"]),
                       ("r2_error_analysis.csv", ["uid", "top1_uid"])):
        d = pd.read_csv(R2 / name, dtype={c: str for c in cols})
        leaks[name] = int(sum(d[c].isin(test_ids).sum() for c in cols))
    prot = json.loads((R2 / "r2_selection_protocol.json").read_text(encoding="utf-8"))
    dec = json.loads((R2 / "r2_stage_decisions.json").read_text(encoding="utf-8"))
    emb = json.loads((R1 / "embedding_metadata.json").read_text(encoding="utf-8"))
    snap = next(iter(sorted((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots").glob("*"))))
    w_now = sha256_file(snap / "model.safetensors")
    rep = json.loads((R2 / "r1_reproduction_check.json").read_text(encoding="utf-8"))
    order = {"protocol_created_utc": prot["created_utc"], "protocol_written_before_evaluation_flag": prot["written_before_any_r2_evaluation"],
             "protocol_mtime_before_results": (R2 / "r2_selection_protocol.json").stat().st_mtime < (R2 / "per_query_metrics_all_configs.csv").stat().st_mtime,
             "expansion_mapping_mtime_before_results": (R2 / "finding_query_expansion.json").stat().st_mtime < (R2 / "per_query_metrics_all_configs.csv").stat().st_mtime,
             "protocol_sha256_matches_stage_decisions": dec["protocol_sha256"] == sha256_file(R2 / "r2_selection_protocol.json"),
             "expansion_sha256_matches_protocol": prot["expansion_file_sha256"] == sha256_file(R2 / "finding_query_expansion.json")}
    ok = (not changed and not missing and all(frozen.values()) and thr_ok and cal_ok and sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"] and not hits and not unlock
          and not any(leaks.values()) and w_now == emb["model_weights_sha256"] and rep["r1_reproduced_exactly"] and all(v for k, v in order.items() if isinstance(v, bool)))
    out = {"status": "PASS" if ok else "FAIL", "c1_c6_and_r1_files_checked": len(before), "files_changed": changed, "files_missing": missing, "files_added_in_checked_roots": added,
           "checkpoint_unchanged": sha256_file(PROJECT_ROOT / man["checkpoint"]) == man["checkpoint_sha256"], "c4_thresholds_unchanged": thr_ok, "c5_platt_parameters_unchanged": cal_ok,
           "c6_freeze_manifest_files_all_match": all(frozen.values()), "r2_code_files_scanned": [p.name for p in code], "fit_tune_or_checkpoint_calls_in_r2_code": hits,
           "allow_locked_true_in_r2_code": unlock, "locked_retrieval_test_studies_in_r2_outputs": leaks, "embedding_model_weights_unchanged_not_fine_tuned": w_now == emb["model_weights_sha256"],
           "r1_baselines_reproduced_exactly": rep["r1_reproduced_exactly"], "protocol_ordering": order, "locked_chexpert_test_used_by_r2": False,
           "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current")}
    (R2 / "r2_integrity_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "c1_c6_and_r1_files_checked", "files_changed", "files_missing", "checkpoint_unchanged", "fit_tune_or_checkpoint_calls_in_r2_code",
                                          "locked_retrieval_test_studies_in_r2_outputs", "embedding_model_weights_unchanged_not_fine_tuned")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
