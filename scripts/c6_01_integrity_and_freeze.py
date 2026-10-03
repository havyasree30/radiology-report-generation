"""C6 step 1+2: test-split independence check and the classifier freeze manifest.

    .venv\\Scripts\\python.exe -m scripts.c6_01_integrity_and_freeze

Reads the split MANIFEST only (paths / patient ids / split names). No test label, image or prediction is
loaded. Stops (non-zero exit) on any patient overlap. The freeze manifest is written here, BEFORE any
test inference or metric exists.
"""

from __future__ import annotations

import ast
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from src.classification.final_test import sha256_file
from src.classification.labels import LABELS
from src.classification.trainer import load_config
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/classification/experiments/c6_final_test"
EXP = PROJECT_ROOT / "results/classification/experiments"
C2B = EXP / "c2b_sqrt_weighted_bce"
COMMITS = {"C1": "a62073273cef155d6177d929a9d3a8e54279200b", "C2_infrastructure": "c1e4800983f8e050da63ac85e799ba443e4858fe",
           "C2_results": "a2de4174b386ca8325c3481d626227dc205cd952", "C3": "031edd6306fd2974d9ae6107921ed92860416bf4",
           "C4": "77dee289fda740a1b261e31cc575343a49baea68", "C5": "55ba4bb87453a1dbb2a3452f5e6aebbcb57120b8"}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def split_rows(cfg: dict) -> dict[str, pd.DataFrame]:
    """Rows per split after the SAME frontal-view and exclusion filters used for training/validation (labels not needed)."""
    m = pd.read_csv(PROJECT_ROOT / cfg["data"]["manifest"])
    drop = set(pd.read_csv(PROJECT_ROOT / cfg["data"]["exclusions"]).query("recommendation == 'exclude'")["Path"])
    out = {}
    for s in ("train", "val", "test", "official_valid"):
        r = m[m["split"] == s]
        n_all = len(r)
        r = r[r["Frontal/Lateral"] == cfg["data"]["view"]]
        n_frontal = len(r)
        r = r[~r["Path"].isin(drop)]
        r.attrs.update(n_manifest=n_all, n_frontal=n_frontal)
        out[s] = r
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(C2B / "config.yaml")
    rows = split_rows(cfg)
    pats = {s: set(r["patient_id"]) for s, r in rows.items()}
    grp = {s: set(r["split_group_id"]) for s, r in rows.items()}
    paths = {s: set(r["Path"]) for s, r in rows.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlap = {f"{a}_and_{b}": {"patients": len(pats[a] & pats[b]), "split_groups": len(grp[a] & grp[b]),
                                "image_paths": len(paths[a] & paths[b])} for a, b in pairs}
    spl = pd.read_csv(PROJECT_ROOT / "data/splits/chexpert/chexpert_patient_splits.csv")
    per_patient_unique = bool(spl["patient_id"].is_unique)
    full = pd.read_csv(PROJECT_ROOT / cfg["data"]["manifest"])
    one_split_per_patient = bool((full.groupby("patient_id")["split"].nunique() == 1).all())   # incl. official_valid

    # ---- evidence that the test split was never used (everything below is read from the repository)
    flags, bad_flags = {}, []
    for p in sorted((PROJECT_ROOT / "results/classification").rglob("*.json")):
        if "c6_final_test" in p.parts:
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in re.finditer(r'"locked_test_split_used"\s*:\s*(true|false|True|False)', txt):
            flags[str(p.relative_to(PROJECT_ROOT))] = m.group(1).lower()
            if m.group(1).lower() == "true":
                bad_flags.append(str(p.relative_to(PROJECT_ROOT)))
    def n_unlock_calls(path) -> int:   # real keyword arguments only (string literals / comments do not count)
        return sum(isinstance(k.value, ast.Constant) and k.value.value is True
                   for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Call)
                   for k in n.keywords if k.arg == "allow_locked")

    unlock_calls = {str(p.relative_to(PROJECT_ROOT)): n_unlock_calls(p)
                    for d in ("src", "scripts") for p in (PROJECT_ROOT / d).rglob("*.py")}
    unlock_calls = {k: v for k, v in unlock_calls.items() if v and "c6_" not in k and "final_test" not in k}
    test_named = [str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "results/classification").rglob("*")
                  if p.is_file() and re.search(r"(^|[_\-.])test([_\-.]|$)", p.name) and "c6_final_test" not in p.parts]
    ckpt_meta = json.loads((C2B / "checkpoints/best.meta.json").read_text(encoding="utf-8"))
    cal = json.loads((EXP / "c5_calibration/final_calibrators.json").read_text(encoding="utf-8"))
    pol = json.loads((EXP / "c4_operating_policy/final_operating_policy.json").read_text(encoding="utf-8"))
    summary = json.loads((C2B / "training_summary.json").read_text(encoding="utf-8"))
    never_used = {
        "model_training": {"train_split": cfg["data"]["train_split"], "locked_splits_config": cfg["data"]["locked_splits"],
                           "evidence": "training_summary / config use train+val only; loader raises LockedSplitError on 'test'"},
        "loss_selection": {"selection_metric": "val_macro_auprc on the validation split (C2)", "c2_reports_flag_locked_test": "see flags"},
        "threshold_selection": {"thresholds_split": pol["dataset_split_for_selection"], "locked_test_split_used": pol["locked_test_split_used"]},
        "calibration_fitting": {"fitting_split": cal["fitting_split"], "locked_test_split_used": cal["locked_test_split_used"]},
        "checkpoint_selection": {"selection_metric": summary.get("selection_metric", "val_macro_auprc"),
                                 "checkpoint_epoch": ckpt_meta.get("epoch"), "split_for_selection": "validation"},
        "locked_test_split_used_flags_in_C1_to_C5_json": flags, "flags_that_are_true": bad_flags,
        "allow_locked_True_occurrences_in_code_before_C6": unlock_calls,
        "result_files_with_'test'_in_the_name_before_C6": test_named}
    ok = (all(v["patients"] == 0 and v["split_groups"] == 0 and v["image_paths"] == 0 for v in overlap.values())
          and not bad_flags and not unlock_calls and not test_named and per_patient_unique and one_split_per_patient)
    meta = json.loads((PROJECT_ROOT / "data/splits/chexpert/split_metadata.json").read_text(encoding="utf-8"))
    integrity = {
        "status": "PASS" if ok else "FAIL",
        "image_counts_after_frontal_and_exclusion_filters": {s: int(len(r)) for s, r in rows.items()},
        "image_counts_in_manifest_before_filters": {s: int(r.attrs["n_manifest"]) for s, r in rows.items()},
        "frontal_image_counts_before_exclusions": {s: int(r.attrs["n_frontal"]) for s, r in rows.items()},
        "patient_counts": {s: len(p) for s, p in pats.items()},
        "split_group_counts": {s: len(g) for s, g in grp.items()},
        "overlap_counts": overlap,
        "patient_unique_in_patient_split_table": per_patient_unique,
        "every_patient_in_exactly_one_split": one_split_per_patient,
        "split_provenance": {"created_by": meta["created_by"], "seed": meta["seed"], "ratios": meta["ratios"], "unit": meta["unit"],
                             "method": meta["decision"]["chosen"], "train_csv_sha256": meta["train_csv_sha256"],
                             "manifest_sha256": sha256_file(PROJECT_ROOT / cfg["data"]["manifest"]),
                             "patient_splits_sha256": sha256_file(PROJECT_ROOT / "data/splits/chexpert/chexpert_patient_splits.csv"),
                             "official_valid": "separate partition, not part of train/val/test"},
        "test_never_used_for": never_used,
        "labels_or_images_of_test_opened_by_this_script": False}
    (OUT / "test_split_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    if not ok:
        print(json.dumps(integrity, indent=1)[:3000])
        print("STOP: split independence check FAILED", file=sys.stderr)
        return 1

    # ---- freeze manifest
    hashed = {}
    for rel in ["results/classification/experiments/c2b_sqrt_weighted_bce/checkpoints/best.pt",
                "results/classification/experiments/c2b_sqrt_weighted_bce/config.yaml",
                "results/classification/experiments/c3_threshold_optimization/thresholds_f1.json",
                "results/classification/experiments/c4_operating_policy/final_operating_policy.json",
                "results/classification/experiments/c5_calibration/final_calibrators.json",
                "data/splits/chexpert/chexpert_image_manifest.csv.gz",
                "results/eda/tables/chexpert_exclusion_candidates.csv",
                "src/classification/model.py", "src/classification/dataset.py", "src/classification/evaluator.py",
                "src/classification/inference_policy.py", "src/classification/operating_policy.py",
                "src/classification/calibration.py", "src/classification/final_test.py", "src/preprocessing/transforms.py",
                "src/data/label_policies.py", "src/data/chexpert.py"]:
        hashed[rel] = sha256_file(PROJECT_ROOT / rel)
    if hashed["results/classification/experiments/c2b_sqrt_weighted_bce/checkpoints/best.pt"] != pol["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash differs from the C4 policy record")
    thr = {l: pol["classes"][l]["threshold"] for l in LABELS}
    platt = {l: {"a": cal["classes"][l]["a"], "b": cal["classes"][l]["b"],
                 "calibrated_equivalent_threshold": cal["classes"][l]["calibrated_equivalent_threshold"]} for l in LABELS}
    manifest = {
        "purpose": "Frozen classification pipeline. Written BEFORE any locked-test inference or metric. No entry may be changed after the test set is opened.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_architecture": "densenet121 (ImageNet-pretrained init), Linear(1024, 14) logits, 14 independent sigmoids (multi-label)",
        "checkpoint": "results/classification/experiments/c2b_sqrt_weighted_bce/checkpoints/best.pt",
        "checkpoint_sha256": pol["checkpoint_sha256"], "checkpoint_epoch": pol["checkpoint_epoch"],
        "loss": "sqrt_weighted_bce", "loss_config": pol["loss_config"],
        "preprocessing": cfg["preprocessing"], "view": cfg["data"]["view"],
        "exclusions_file": cfg["data"]["exclusions"], "inference_policy": "fp32_strict",
        "test_time_augmentation": False, "ensembling": False,
        "label_order": list(LABELS),
        "uncertain_label_policy": "ignore (uncertain -1 masked: excluded from every metric)",
        "missing_label_policy": "blank/unmentioned scored as negative (0)",
        "f1_thresholds_raw_score": thr,
        "binary_rule": "positive = raw_score >= frozen_F1_threshold",
        "no_finding_rule": pol["no_finding_rule"], "no_finding_rule_abnormal_labels": pol["no_finding_rule_abnormal_labels"],
        "support_devices_suppresses_no_finding": False,
        "calibration": {"method": "platt (all 14 classes)", "version": cal["calibration_version"], "fitted_on": "validation (full)",
                        "input": "logit(raw sigmoid score)", "parameters": platt},
        "commits": {**COMMITS, "c6_branch_head_at_freeze": git("rev-parse", "HEAD"), "c6_branch": git("branch", "--show-current"),
                    "working_tree_clean_at_freeze": git("status", "--porcelain") == ""},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "bootstrap": {"n_resamples": 1000, "seed": 42, "unit": "patient"},
        "hashed_files": hashed,
        "test_inference_started": False}
    (OUT / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"integrity": integrity["status"], "test_images": integrity["image_counts_after_frontal_and_exclusion_filters"]["test"],
                      "test_patients": integrity["patient_counts"]["test"], "overlaps": overlap,
                      "dirty_tree_at_freeze": not manifest["commits"]["working_tree_clean_at_freeze"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
