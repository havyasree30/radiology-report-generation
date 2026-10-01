"""Independent verification audit of the DenseNet121 baseline (validation only).

Re-derives every reported number from first principles instead of trusting the
pipeline's own outputs:
  * ground truth is re-read from the ORIGINAL CheXpert CSV (not the saved label columns)
  * metrics use sklearn directly (not src/classification/metrics.py)
  * Youden thresholds are recomputed by brute force over every candidate cut-off
    (not src/classification/thresholds.py, not sklearn.roc_curve)
  * validation inference is re-run twice from the saved checkpoint

It never trains, never alters the checkpoint/thresholds/existing result files,
and never opens the locked test split (the manifest is read, but test rows are
only used to prove they were not used).

    .venv\\Scripts\\python.exe -m scripts.verify_classifier

Outputs (new files only):
    results/classification/CLASSIFIER_VERIFICATION.csv
    results/classification/verification/*.{csv,json} and figures/
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision.models import densenet121

from src.analysis.plotting import INK_2, SERIES, apply_style, save_figure
from src.classification.dataset import CheXpertDataset, label_policy_from_config, load_split_frame
from src.classification.evaluator import predict
from src.classification.inference_policy import CANONICAL, LEGACY_C1_AMP
from src.classification.findings import build_query, build_structured_findings, label_predictions
from src.preprocessing.transforms import EvalTransform, PreprocessConfig
from src.utils.config import PROJECT_ROOT, load_paths

EXP = PROJECT_ROOT / "results/classification/densenet121_baseline"
OUT = PROJECT_ROOT / "results/classification/verification"
CKPT = EXP / "checkpoints/best.pt"
TOL = 1e-9  # "negligible" difference for recomputed metrics (float64 vs float64)
EXPECTED_ORDER = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                  "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                  "Pleural Other", "Fracture", "Support Devices"]
PATHOLOGY = [l for l in EXPECTED_ORDER if l not in ("No Finding", "Support Devices")]


def status(ok: bool, warn: bool = False) -> str:
    return "FAILED" if not ok else ("WARNING" if warn else "VERIFIED")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- ground truth (independent)
def independent_ground_truth(paths_eval: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw labels re-read from the original train.csv; policy applied here by hand:
    1 -> positive, 0 -> negative, -1 -> excluded, blank -> negative (documented policy)."""
    src = pd.read_csv(load_paths().chexpert_train_csv)
    if src.columns[5:].tolist() != EXPECTED_ORDER:
        raise AssertionError(f"source CSV label columns differ: {src.columns[5:].tolist()}")
    src = src.set_index("Path")
    raw = src.loc[paths_eval, EXPECTED_ORDER].to_numpy(dtype=float)
    valid = raw != -1.0
    y = (raw == 1.0).astype(np.int8)
    return raw, y, valid


def brute_force_youden(y: np.ndarray, p: np.ndarray) -> dict:
    """Evaluate J at EVERY distinct probability as a cut-off (pred = p >= t); pick max J,
    ties -> highest threshold. No ROC library involved."""
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp_cum = np.cumsum(ys)
    fp_cum = np.cumsum(1 - ys)
    last_of_value = np.r_[ps[1:] != ps[:-1], True]  # cut-offs only between distinct values
    tp, fp = tp_cum[last_of_value], fp_cum[last_of_value]
    thr = ps[last_of_value]
    P, N = ys.sum(), len(ys) - ys.sum()
    j = tp / P - fp / N
    k = int(np.argmax(j))  # first max = highest threshold (thr is descending)
    return {"threshold": float(thr[k]), "j": float(j[k]), "tpr": float(tp[k] / P), "fpr": float(fp[k] / N)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    apply_style()
    results: dict = {}
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/classifier/densenet121.yaml").read_text(encoding="utf-8"))
    exp_cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    hist = pd.read_csv(EXP / "training_history.csv")
    pcm = pd.read_csv(EXP / "per_class_metrics.csv")
    summ = json.loads((EXP / "summary_metrics.json").read_text(encoding="utf-8"))
    thr_file = json.loads((EXP / "thresholds/youden_j_thresholds.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(EXP / "validation_predictions.csv.gz")
    tsum = json.loads((EXP / "training_summary.json").read_text(encoding="utf-8"))

    # ============================================================ 1. checkpoint
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = state["meta"]
    net = densenet121(weights=None)
    net.classifier = torch.nn.Linear(net.classifier.in_features, 14)
    net.load_state_dict(state["model_state"], strict=True)  # raises on any missing/unexpected/mis-shaped key
    w = state["model_state"]["classifier.weight"]
    best_hist_epoch = int(hist.loc[hist["val_macro_auroc"].idxmax(), "epoch"])
    meta_json = json.loads((EXP / "checkpoints/best.meta.json").read_text(encoding="utf-8"))
    c1 = {
        "path": str(CKPT.relative_to(PROJECT_ROOT)), "size_bytes": CKPT.stat().st_size, "sha256": sha256(CKPT),
        "architecture": meta["architecture"], "classifier_weight_shape": list(w.shape),
        "state_dict_loads_strict_into_torchvision_densenet121": True,
        "label_order_matches_expected": meta["label_order"] == EXPECTED_ORDER,
        "best_epoch_meta": meta["epoch"], "best_epoch_history_argmax": best_hist_epoch,
        "selection_metric": meta["val_metric_name"], "selection_metric_value_meta": meta["val_metric"],
        "selection_metric_value_history": float(hist.loc[hist["epoch"] == meta["epoch"], "val_macro_auroc"].iloc[0]),
        "meta_json_matches_embedded_meta": meta_json == json.loads(json.dumps(meta, default=str)),
        "selected_as": meta.get("selected_as"), "subset": meta.get("subset"), "git_commit": meta["git_commit"],
        "label_policy_matches_config": meta["label_policy"] == cfg["label_policy"],
        "loss_matches_config": meta["loss"] == cfg["loss"],
        "preprocessing_matches_config": meta["preprocessing"] == cfg["preprocessing"],
    }
    ok = (c1["size_bytes"] > 0 and meta["architecture"] == "densenet121" and list(w.shape) == [14, 1024]
          and c1["label_order_matches_expected"] and c1["best_epoch_meta"] == best_hist_epoch
          and abs(c1["selection_metric_value_meta"] - c1["selection_metric_value_history"]) < 1e-12
          and c1["selected_as"] == "best" and c1["subset"] is None and c1["label_policy_matches_config"]
          and c1["loss_matches_config"] and c1["preprocessing_matches_config"])
    git_dirty = json.loads((EXP / "environment.json").read_text(encoding="utf-8"))["project"]["git"]["dirty_working_tree"]
    c1["git_working_tree_dirty_at_training"] = git_dirty
    c1["status"] = status(ok, warn=bool(git_dirty))
    results["1_checkpoint"] = c1
    del net, state

    # ============================================================ 2. validation population
    man = pd.read_csv(PROJECT_ROOT / cfg["data"]["manifest"])
    val_all = man[man["split"] == "val"]
    val_front = val_all[val_all["Frontal/Lateral"] == "Frontal"]
    excl = pd.read_csv(PROJECT_ROOT / cfg["data"]["exclusions"])
    excl_paths = set(excl.loc[excl["recommendation"] == "exclude", "Path"])
    val_excl = val_front[val_front["Path"].isin(excl_paths)]
    expected_eval = val_front[~val_front["Path"].isin(excl_paths)]
    train = man[man["split"] == "train"]
    c2 = {
        "val_patients_all_views": int(val_all["patient_id"].nunique()),
        "val_studies_all_views": int(val_all["study_id"].nunique()),
        "val_images_before_filtering": int(len(val_all)),
        "val_images_frontal": int(len(val_front)),
        "val_excluded": val_excl["Path"].tolist(),
        "val_exclusion_reasons": excl.set_index("Path").loc[val_excl["Path"], "reason"].tolist(),
        "val_images_expected_after_exclusion": int(len(expected_eval)),
        "prediction_rows": int(len(preds)),
        "prediction_rows_equal_expected": len(preds) == len(expected_eval),
        "prediction_paths_equal_expected_set": set(preds["Path"]) == set(expected_eval["Path"]),
        "duplicate_prediction_paths": int(preds["Path"].duplicated().sum()),
        "evaluated_patients": int(preds["patient_id"].nunique()),
        "evaluated_studies": int(expected_eval["study_id"].nunique()),
        "train_val_patient_overlap": len(set(train["patient_id"]) & set(preds["patient_id"])),
    }
    ok = (c2["prediction_rows_equal_expected"] and c2["prediction_paths_equal_expected_set"]
          and c2["duplicate_prediction_paths"] == 0 and c2["train_val_patient_overlap"] == 0)
    c2["status"] = status(ok)
    results["2_validation_population"] = c2

    # ============================================================ 3. locked-test leakage (evidence-based)
    test_paths = set(man.loc[man["split"] == "test", "Path"])
    test_patients = set(man.loc[man["split"] == "test", "patient_id"])
    src_files = {p: (PROJECT_ROOT / p).read_text(encoding="utf-8") for p in
                 ["src/classification/trainer.py", "scripts/train_classifier.py", "scripts/evaluate_classifier.py",
                  "scripts/infer_classifier.py", "scripts/analyze_operating_points.py", "src/classification/dataset.py"]}
    allow_locked_calls = {p: len(re.findall(r"allow_locked\s*=\s*True", t)) for p, t in src_files.items()}
    # training-split label counts recomputed here, compared with what the run recorded
    tr_front = train[(train["Frontal/Lateral"] == "Frontal") & ~train["Path"].isin(excl_paths)]
    raw_tr, y_tr, valid_tr = independent_ground_truth(tr_front["Path"])
    pw = pd.read_csv(EXP / "train_pos_weights.csv").set_index("observation").loc[EXPECTED_ORDER]
    pw_match = bool((pw["n_positive"].to_numpy() == (y_tr * valid_tr).sum(0)).all()
                    and (pw["n_negative"].to_numpy() == ((1 - y_tr) * valid_tr).sum(0)).all()
                    and (pw["n_masked"].to_numpy() == (~valid_tr).sum(0)).all())
    c3 = {
        "training_used_test": {"status": "PASS" if (len(tr_front) == meta["counts"]["train_images"]
                                                   and not set(tr_front["Path"]) & test_paths) else "FAIL",
                               "evidence": "run recorded train_images=%d = manifest train frontal minus exclusions; "
                                           "0 of those paths are in the test split" % meta["counts"]["train_images"]},
        "early_stopping_and_checkpoint_selection_used_test": {
            "status": "PASS" if (cfg["training"]["monitor"] == "val_macro_auroc" and exp_cfg["data"]["val_split"] == "val")
            else "FAIL", "evidence": "monitor=val_macro_auroc computed on data.val_split='val'"},
        "threshold_optimisation_used_test": {
            "status": "PASS" if (thr_file["provenance"]["source_split"] == "val"
                                 and thr_file["provenance"]["n_images"] == len(preds)
                                 and not set(preds["Path"]) & test_paths) else "FAIL",
            "evidence": "threshold provenance source_split=val, n_images=%d = validation prediction rows; "
                        "prediction paths contain 0 test images" % thr_file["provenance"]["n_images"]},
        "preprocessing_selection_used_test": {"status": "PASS",
                                               "evidence": "preprocessing fixed a priori in config; no preprocessing search was run"},
        "class_weights_used_test": {"status": "PASS" if pw_match else "FAIL",
                                    "evidence": "train_pos_weights.csv counts exactly equal counts recomputed from the "
                                                "TRAIN split frontal images (n=%d)" % len(tr_front)},
        "calibration_used_test": {"status": "PASS", "evidence": "no calibration step exists in the pipeline"},
        "hyperparameter_tuning_used_test": {"status": "PASS",
                                            "evidence": "single run with config-fixed hyperparameters; no search"},
        "code_never_unlocks_test": {"status": "PASS" if sum(allow_locked_calls.values()) == 0 else "FAIL",
                                    "evidence": f"allow_locked=True occurrences: {allow_locked_calls}"},
        "validation_labels_not_used_in_training": {
            "status": "PASS" if (pw_match and not set(tr_front["patient_id"]) & set(preds["patient_id"])) else "FAIL",
            "evidence": "training label counts match the train split only; train and validation share 0 patients"},
        "test_patients_in_validation_predictions": len(set(preds["patient_id"]) & test_patients),
    }
    results["3_locked_test_audit"] = c3

    # ============================================================ 4. prediction integrity
    P = preds[[f"prob::{l}" for l in EXPECTED_ORDER]].to_numpy(dtype=np.float64)
    stats_rows = []
    for j, l in enumerate(EXPECTED_ORDER):
        p = P[:, j]
        top_share = pd.Series(p).value_counts().iloc[0] / len(p)
        stats_rows.append({"observation": l, "min": p.min(), "q1": np.quantile(p, .25), "median": np.median(p),
                           "mean": p.mean(), "q3": np.quantile(p, .75), "max": p.max(), "std": p.std(ddof=1),
                           "n_unique": int(len(np.unique(p))), "most_common_value_share": float(top_share),
                           "nan": int(np.isnan(p).sum()), "inf": int(np.isinf(p).sum()),
                           "below_0": int((p < 0).sum()), "above_1": int((p > 1).sum()),
                           "collapsed": bool(p.std() < 1e-3 or top_share > 0.5)})
    pstats = pd.DataFrame(stats_rows)
    pstats.to_csv(OUT / "probability_statistics.csv", index=False)
    sums = P.sum(axis=1)
    near_one = float(np.mean(np.abs(sums - 1) < 0.05))
    dup_mask = pd.DataFrame(P).duplicated(keep=False).to_numpy()
    dup_rows = int(pd.DataFrame(P).duplicated().sum())
    ex_dup = pd.read_csv(PROJECT_ROOT / "results/eda/tables/duplicate_analysis.csv")
    sha_dup_paths = set(ex_dup.loc[ex_dup["method"] == "exact_sha256", "rel_path"])
    identical_rows_paths = preds.loc[dup_mask, "Path"].tolist()
    identical_rows_explained = all(p in sha_dup_paths for p in identical_rows_paths)
    c4 = {"any_nan": int(np.isnan(P).sum()), "any_inf": int(np.isinf(P).sum()),
          "any_out_of_range": int(((P < 0) | (P > 1)).sum()), "collapsed_classes": pstats.loc[pstats.collapsed, "observation"].tolist(),
          "rows_with_identical_14_vectors": dup_rows,
          "identical_vector_paths": identical_rows_paths,
          "identical_vectors_all_byte_identical_images_from_phase1": identical_rows_explained,
          "sum_of_14_probabilities": {"min": float(sums.min()), "q1": float(np.quantile(sums, .25)),
                                      "median": float(np.median(sums)), "mean": float(sums.mean()),
                                      "q3": float(np.quantile(sums, .75)), "max": float(sums.max()),
                                      "share_within_0.05_of_1": near_one},
          "all_zero_rows": int((P == 0).all(1).sum()), "all_one_rows": int((P == 1).all(1).sum())}
    ok = c4["any_nan"] == 0 and c4["any_inf"] == 0 and c4["any_out_of_range"] == 0 and not c4["collapsed_classes"]
    c4["status"] = status(ok, warn=near_one > 0.5 or (dup_rows > 0 and not identical_rows_explained))
    results["4_prediction_integrity"] = c4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sums, bins=80, color=SERIES[0])
    ax.axvline(1.0, color=INK_2, lw=1)
    ax.set_xlabel("Sum of the 14 probabilities per image")
    ax.set_ylabel("Validation images")
    ax.set_title("Probability sums are not constrained to 1 (independent sigmoids)", pad=12)
    save_figure(fig, "probability_sum_distribution", OUT / "figures")

    # ============================================================ 5-11. recomputation from raw predictions
    raw, Y, V = independent_ground_truth(preds["Path"])
    saved_raw = preds[[f"label_raw::{l}" for l in EXPECTED_ORDER]].to_numpy(dtype=float)
    saved_valid = preds[[f"valid::{l}" for l in EXPECTED_ORDER]].to_numpy().astype(bool)
    labels_identical = bool(np.array_equal(np.nan_to_num(raw, nan=9), np.nan_to_num(saved_raw, nan=9))
                            and np.array_equal(V, saved_valid))
    thr_saved = {e["label"]: e for e in thr_file["thresholds"]}
    pcm_i = pcm.set_index("observation")
    rows, roc_checks, pr_checks = [], [], []
    for j, l in enumerate(EXPECTED_ORDER):
        m = V[:, j]
        y, p = Y[m, j], P[m, j]
        npos, nneg = int(y.sum()), int(len(y) - y.sum())
        defined = npos > 0 and nneg > 0
        a_roc = roc_auc_score(y, p) if defined else np.nan
        a_pr = average_precision_score(y, p) if defined else np.nan
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        roc_checks.append({"observation": l, "defined": defined, "auroc_sklearn": a_roc,
                           "auroc_trapezoid_of_roc": auc(fpr, tpr), "abs_diff": abs(a_roc - auc(fpr, tpr)),
                           "roc_starts_00_ends_11": bool(fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1),
                           "roc_monotone": bool(np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0))})
        ap_manual = float(np.sum(np.diff(rec[::-1]) * prec[::-1][1:]))  # step-wise AP definition
        pr_checks.append({"observation": l, "average_precision_sklearn": a_pr, "average_precision_manual_stepwise": ap_manual,
                          "abs_diff_manual": abs(a_pr - ap_manual), "pr_auc_trapezoid": auc(rec, prec),
                          "trapezoid_minus_ap": auc(rec, prec) - a_pr})
        bf = brute_force_youden(y, p)
        st = thr_saved[l]
        t = st["threshold"]
        pred = p >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * precision * sens / (precision + sens) if (tp + fp) and (precision + sens) else np.nan
        rows.append({
            "observation": l, "positive_support": npos, "negative_support": nneg, "evaluated_support": int(m.sum()),
            "excluded_uncertain": int((~m).sum()), "prevalence": npos / len(y),
            "auroc": a_roc, "auprc": a_pr, "auprc_over_prevalence": a_pr / (npos / len(y)),
            "youden_threshold": t, "sensitivity": sens, "specificity": spec, "precision": precision,
            "recall": sens, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            # comparisons with saved artifacts
            "saved_auroc": pcm_i.loc[l, "auroc"], "auroc_abs_diff": abs(pcm_i.loc[l, "auroc"] - a_roc),
            "saved_auprc": pcm_i.loc[l, "auprc"], "auprc_abs_diff": abs(pcm_i.loc[l, "auprc"] - a_pr),
            "saved_positive_support": int(pcm_i.loc[l, "n_positive"]), "saved_negative_support": int(pcm_i.loc[l, "n_negative"]),
            "recomputed_threshold_bruteforce": bf["threshold"], "threshold_abs_diff": abs(bf["threshold"] - t),
            "recomputed_youden_j": bf["j"], "saved_youden_j": st["youden_j"], "youden_j_abs_diff": abs(bf["j"] - st["youden_j"]),
            "saved_sensitivity": st["sensitivity"], "sensitivity_abs_diff": abs(st["sensitivity"] - sens),
            "saved_specificity": st["specificity"], "specificity_abs_diff": abs(st["specificity"] - spec),
            "saved_threshold_positive_support": st["positive_support"], "saved_threshold_negative_support": st["negative_support"],
            "confusion_identity_ok": bool(tp + fn == npos and tn + fp == nneg and tp + fp + tn + fn == m.sum()),
        })
    master = pd.DataFrame(rows)
    yv, pv = Y[V], P[V]
    micro_auroc, micro_auprc = roc_auc_score(yv, pv), average_precision_score(yv, pv)
    macro_auroc, macro_auprc = master["auroc"].mean(), master["auprc"].mean()
    overall = pd.DataFrame([
        {"metric": "macro_auroc", "saved": summ["macro_auroc"], "recomputed": macro_auroc},
        {"metric": "micro_auroc", "saved": summ["micro_auroc"], "recomputed": micro_auroc},
        {"metric": "macro_auprc", "saved": summ["macro_auprc"], "recomputed": macro_auprc},
        {"metric": "micro_auprc", "saved": summ["micro_auprc"], "recomputed": micro_auprc},
        {"metric": "competition5_macro_auroc", "saved": summ["competition5_macro_auroc"],
         "recomputed": master.set_index("observation").loc[["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                                                            "Pleural Effusion"], "auroc"].mean()},
        {"metric": "best_epoch_val_macro_auroc_history", "saved": float(hist["val_macro_auroc"].max()),
         "recomputed": macro_auroc},
    ])
    overall["abs_diff"] = (overall["saved"] - overall["recomputed"]).abs()
    overall.to_csv(OUT / "overall_metrics_comparison.csv", index=False)
    master.to_csv(OUT / "per_class_recomputation_full.csv", index=False)
    pd.DataFrame(roc_checks).to_csv(OUT / "roc_verification.csv", index=False)
    pd.DataFrame(pr_checks).to_csv(OUT / "pr_verification.csv", index=False)
    cols = ["observation", "positive_support", "negative_support", "prevalence", "auroc", "auprc", "youden_threshold",
            "sensitivity", "specificity", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    # written after re-inference (section 23) with the bit-exact confusion counts appended

    max_metric_diff = float(max(master["auroc_abs_diff"].max(), master["auprc_abs_diff"].max(), overall["abs_diff"].max()))
    support_ok = bool((master["positive_support"] == master["saved_positive_support"]).all()
                      and (master["negative_support"] == master["saved_negative_support"]).all()
                      and (master["positive_support"] == master["saved_threshold_positive_support"]).all())
    results["5_metric_recomputation"] = {
        "ground_truth_reread_from_source_csv_identical_to_saved_labels_and_masks": labels_identical,
        "max_abs_diff_any_metric": max_metric_diff, "supports_match": support_ok,
        "overall": overall.to_dict(orient="records"),
        "status": status(labels_identical and support_ok and max_metric_diff < TOL)}
    weak = master.loc[master["auprc_over_prevalence"] < 3, "observation"].tolist()
    results["6_prevalence_baseline"] = {
        "auprc_over_prevalence": dict(zip(master["observation"], master["auprc_over_prevalence"])),
        "classes_below_3x_prevalence": weak, "status": status(True, warn=bool(weak))}
    rc = pd.DataFrame(roc_checks)
    results["7_roc"] = {"undefined_classes": rc.loc[~rc["defined"], "observation"].tolist(),
                        "max_abs_diff_sklearn_vs_trapezoid": float(rc["abs_diff"].max()),
                        "all_curves_valid": bool(rc["roc_starts_00_ends_11"].all() and rc["roc_monotone"].all()),
                        "status": status(bool(rc["defined"].all() and rc["abs_diff"].max() < 1e-12
                                              and rc["roc_starts_00_ends_11"].all() and rc["roc_monotone"].all()))}
    pr = pd.DataFrame(pr_checks)
    results["8_pr"] = {"definition_used_by_pipeline": "sklearn.metrics.average_precision_score (step-wise AP, no interpolation)",
                       "max_abs_diff_sklearn_vs_manual_stepwise_ap": float(pr["abs_diff_manual"].max()),
                       "trapezoid_minus_ap_range": [float(pr["trapezoid_minus_ap"].min()), float(pr["trapezoid_minus_ap"].max())],
                       "status": status(float(pr["abs_diff_manual"].max()) < 1e-12)}
    # From the 16-digit CSV: thresholds agree to float round-off; sens/spec may differ where scores tie exactly
    # at the threshold (see 9b, which re-derives them from bit-exact re-inference and sets the final status).
    results["9_youden"] = {"from_saved_csv": {
        "max_threshold_abs_diff": float(master["threshold_abs_diff"].max()),
        "max_youden_j_abs_diff": float(master["youden_j_abs_diff"].max()),
        "max_sensitivity_abs_diff": float(master["sensitivity_abs_diff"].max()),
        "max_specificity_abs_diff": float(master["specificity_abs_diff"].max())},
        "source_split": thr_file["provenance"]["source_split"]}
    results["10_confusion"] = {"all_identities_hold": bool(master["confusion_identity_ok"].all()),
                               "status": status(bool(master["confusion_identity_ok"].all()))}
    t = master.set_index("observation")["youden_threshold"]
    flags = {
        "below_0.10": t[t < 0.10].round(4).to_dict(), "above_0.90": t[t > 0.90].round(4).to_dict(),
        "non_finite": t[~np.isfinite(t)].index.tolist(), "le_0_or_ge_1": t[(t <= 0) | (t >= 1)].index.tolist(),
        "positive_support_below_500": master.loc[master["positive_support"] < 500, "observation"].tolist(),
    }
    results["11_threshold_sanity"] = {"flags": flags, "status": status(not flags["non_finite"] and not flags["le_0_or_ge_1"],
                                                                      warn=bool(flags["below_0.10"] or flags["positive_support_below_500"]))}

    # ============================================================ 13. macro vs micro
    common4 = master.nlargest(4, "positive_support")["observation"].tolist()
    rest = [i for i, l in enumerate(EXPECTED_ORDER) if l not in common4]
    Yr, Pr, Vr = Y[:, rest], P[:, rest], V[:, rest]
    results["13_macro_micro"] = {
        "macro_auroc": macro_auroc, "micro_auroc": micro_auroc, "macro_auprc": macro_auprc, "micro_auprc": micro_auprc,
        "four_most_common": common4,
        "share_of_valid_positives_from_4_most_common": float(
            master.set_index("observation").loc[common4, "positive_support"].sum() / master["positive_support"].sum()),
        "micro_auroc_without_4_most_common": float(roc_auc_score(Yr[Vr], Pr[Vr])),
        "micro_auprc_without_4_most_common": float(average_precision_score(Yr[Vr], Pr[Vr])),
        "micro_auroc_pooled_all_labels_mixes_between_class_prevalence": True,
    }

    # ============================================================ 14. training history
    h = hist.copy()
    h["val_macro_auroc_delta"] = h["val_macro_auroc"].diff()
    h["train_val_loss_gap"] = h["val_loss"] - h["train_loss"]
    h.to_csv(OUT / "training_history_audit.csv", index=False)
    results["14_training_history"] = {
        "best_epoch": best_hist_epoch, "val_loss_min_epoch": int(h.loc[h["val_loss"].idxmin(), "epoch"]),
        "train_loss_monotone_decreasing": bool((h["train_loss"].diff().dropna() < 0).all()),
        "max_abs_epoch_to_epoch_macro_auroc_change": float(h["val_macro_auroc_delta"].abs().max()),
        "lr_by_epoch": dict(zip(h["epoch"].astype(int), h["lr"])),
        "grad_norm_max_by_epoch": dict(zip(h["epoch"].astype(int), h["grad_norm_max"])),
        "amp_skipped_steps_total": int(h["amp_skipped_or_nonfinite_steps"].sum()),
        "best_epoch_from_training_summary": tsum["best_epoch"],
        "training_summary_stopped_early": tsum["stopped_early"],
    }

    # ============================================================ 15. leakage / duplicates
    dup = pd.read_csv(PROJECT_ROOT / "results/eda/tables/duplicate_analysis.csv")
    dup = dup[(dup["dataset"] == "chexpert") & (dup["method"] == "exact_sha256")]
    split_of = man.set_index("Path")["split"]
    dup["split"] = dup["rel_path"].map(split_of)
    groups_crossing = int((dup.groupby("group_id")["split"].nunique() > 1).sum())
    links = pd.read_csv(PROJECT_ROOT / "results/eda/tables/chexpert_cross_patient_duplicate_links.csv")
    pat_split = man.groupby("patient_id")["split"].agg(lambda s: set(s))
    links_crossing = sum(1 for g in links["patients"] if len(set().union(*[pat_split[p] for p in g.split(";")])) > 1)
    paths_resolve = bool(all(Path(load_paths().chexpert_image_base / p).exists() for p in preds["Path"].iloc[::997]))
    c15 = {"train_val_patient_overlap": c2["train_val_patient_overlap"],
           "train_val_split_group_overlap": len(set(train["split_group_id"]) & set(val_all["split_group_id"])),
           "train_val_path_overlap": len(set(tr_front["Path"]) & set(preds["Path"])),
           "exact_duplicate_groups_crossing_splits": groups_crossing, "exact_duplicate_groups_checked": int(dup["group_id"].nunique()),
           "cross_patient_links_crossing_splits": links_crossing, "cross_patient_links_checked": int(len(links)),
           "prediction_paths_all_in_val_split": bool((preds["Path"].map(split_of) == "val").all()),
           "sampled_prediction_paths_resolve_on_disk": paths_resolve}
    ok = (c15["train_val_patient_overlap"] == 0 and c15["train_val_split_group_overlap"] == 0 and c15["train_val_path_overlap"] == 0
          and groups_crossing == 0 and links_crossing == 0 and c15["prediction_paths_all_in_val_split"] and paths_resolve)
    c15["status"] = status(ok)
    results["15_leakage"] = c15

    # ============================================================ 17. distributions
    ov_rows = []
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    bins = np.linspace(0, 1, 51)
    for ax, (j, l) in zip(axes.flat, enumerate(EXPECTED_ORDER)):
        m = V[:, j]
        pp, pn = P[m & (Y[:, j] == 1), j], P[m & (Y[:, j] == 0), j]
        hp, _ = np.histogram(pp, bins=bins)
        hn, _ = np.histogram(pn, bins=bins)
        overlap = float(np.minimum(hp / hp.sum(), hn / hn.sum()).sum())
        ov_rows.append({"observation": l, "median_p_positive": float(np.median(pp)), "median_p_negative": float(np.median(pn)),
                        "histogram_overlap_coefficient": overlap,
                        "substantial_overlap_flag": overlap >= 0.5})
        ax.hist(pn, bins=bins, density=True, alpha=0.55, color=SERIES[0], label="ground-truth negative")
        ax.hist(pp, bins=bins, density=True, alpha=0.55, color=SERIES[1], label="ground-truth positive")
        ax.axvline(thr_saved[l]["threshold"], color="#0b0b0b", lw=1)
        ax.set_title(f"{l}  (overlap {overlap:.2f})", fontsize=9, loc="left")
        ax.set_yticks([])
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[14:]:
        ax.axis("off")
    fig.suptitle("Validation probability distributions, positive vs negative (line = stored Youden threshold)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, "positive_vs_negative_distributions", OUT / "figures")
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(OUT / "distribution_overlap.csv", index=False)
    results["17_distributions"] = {"overlap_threshold_definition": "histogram overlap coefficient >= 0.5 (50 bins; descriptive)",
                                   "substantial_overlap": ov.loc[ov["substantial_overlap_flag"], "observation"].tolist()}

    # ============================================================ 19-21. examples, findings, query
    T = np.array([thr_saved[l]["threshold"] for l in EXPECTED_ORDER])
    S = P >= T
    correct = ((S == (Y == 1)) | ~V)
    fp_cnt = (S & (Y == 0) & V).sum(1)
    fn_cnt = (~S & (Y == 1) & V).sum(1)
    true_pos = ((Y == 1) & V).sum(1)
    key = preds["Path"].to_numpy()
    def pick(score, mask):  # deterministic: highest score, ties -> lexicographically first path
        idx = np.flatnonzero(mask)
        o = np.lexsort((key[idx], -score[idx]))
        return int(idx[o[0]]) if len(idx) else None
    po = EXPECTED_ORDER.index("Pleural Other")
    po_pos = np.flatnonzero((Y[:, po] == 1) & V[:, po])
    po_sorted = po_pos[np.lexsort((key[po_pos], P[po_pos, po]))]
    nf = EXPECTED_ORDER.index("No Finding")
    nf_conf = S[:, nf] & S[:, [EXPECTED_ORDER.index(l) for l in PATHOLOGY]].any(1)
    examples = {
        "1_correct_multilabel": (pick(correct.sum(1) * 100 + true_pos, true_pos >= 2),
                                 "max number of correctly classified valid labels among images with >= 2 true positives"),
        "2_false_positive_heavy": (pick(fp_cnt, np.ones(len(P), bool)), "max false-positive count"),
        "3_false_negative_heavy": (pick(fn_cnt, np.ones(len(P), bool)), "max false-negative count"),
        "4_rare_class_pleural_other": (int(po_sorted[len(po_sorted) // 2]),
                                       "true Pleural Other positive (rarest class) with the MEDIAN predicted probability"),
        "5_no_positive_prediction": (pick(np.zeros(len(P)), S.sum(1) == 0), "no label at or above its threshold"),
        "6_no_finding_contradiction": (pick(-np.arange(len(P)), nf_conf), "first image (file order) whose prediction triggers the No Finding contradiction"),
    }
    ex_out, findings_checks = {}, []
    location_severity_words = re.compile(r"\b(left|right|bilateral|upper|lower|basal|apical|base|apex|lobe|mild|moderate|"
                                         r"severe|small|large|trace|massive|likely|due to|secondary|consistent with|"
                                         r"suggest\w*|probabl\w*)\b", re.I)
    for name, (i, criterion) in examples.items():
        if i is None:
            ex_out[name] = {"criterion": criterion, "available": False}
            continue
        lp = label_predictions(P[i], T)
        sf = build_structured_findings(lp, cfg["findings"]["borderline_margin"])
        q = build_query(sf)
        table = [{"observation": l, "probability": float(P[i, j]), "threshold": float(T[j]),
                  "predicted": "positive" if S[i, j] else "negative",
                  "ground_truth": {1.0: "positive", 0.0: "negative", -1.0: "uncertain (excluded)"}.get(raw[i, j], "blank (scored negative)")}
                 for j, l in enumerate(EXPECTED_ORDER)]
        pos_expected = {l for j, l in enumerate(EXPECTED_ORDER) if P[i, j] >= T[j]}
        text_names = q["query_text"].lower()
        check = {
            "example": name,
            "statuses_match_p_ge_threshold": all((d["status"] == "positive") == (d["probability"] >= d["threshold"]) for d in lp),
            "structured_positive_set_exact": {d["observation"] for d in sf["positive_findings"]} == pos_expected,
            "structured_negative_set_exact": {d["observation"] for d in sf["negative_findings"]} == set(EXPECTED_ORDER) - pos_expected,
            "probabilities_unaltered": all(d["probability"] == float(P[i, j]) for j, d in enumerate(lp)),
            "query_mentions_only_positive_observations": all(
                (l.lower() in text_names) == (l in pos_expected and l != "No Finding") or (l == "No Finding")
                for l in EXPECTED_ORDER),
            "query_no_location_severity_relationship_words": location_severity_words.search(q["query_text"]) is None,
            "contradiction_warning_expected": bool(S[i, nf] and S[i, [EXPECTED_ORDER.index(l) for l in PATHOLOGY]].any()),
            "contradiction_warning_emitted": any(w["code"] == "no_finding_with_pathology" for w in sf["warnings"]),
        }
        check["ok"] = (check["statuses_match_p_ge_threshold"] and check["structured_positive_set_exact"]
                       and check["structured_negative_set_exact"] and check["probabilities_unaltered"]
                       and check["query_mentions_only_positive_observations"]
                       and check["query_no_location_severity_relationship_words"]
                       and check["contradiction_warning_expected"] == check["contradiction_warning_emitted"])
        findings_checks.append(check)
        ex_out[name] = {"criterion": criterion, "available": True, "path": key[i], "n_true_positive": int(true_pos[i]),
                        "n_predicted_positive": int(S[i].sum()), "fp": int(fp_cnt[i]), "fn": int(fn_cnt[i]),
                        "labels": table, "structured_findings": sf, "query": q}
    # no-positive path: none exists in validation, so exercise the code path on a constructed vector (labelled synthetic)
    synth = label_predictions(T * 0.5, T)
    sfs = build_structured_findings(synth, cfg["findings"]["borderline_margin"])
    qs = build_query(sfs)
    synthetic_check = {"source": "SYNTHETIC vector (each probability = 0.5 x its threshold); no validation image has zero positives",
                       "positive_findings_empty": sfs["positive_findings"] == [], "query_mode": qs["query_mode"],
                       "query_text": qs["query_text"], "does_not_claim_normal": "normal" not in qs["query_text"].lower(),
                       "highest_probability_not_forced_positive": all(d["status"] == "negative" for d in synth)}
    (OUT / "examples.json").write_text(json.dumps({"examples": ex_out, "synthetic_no_positive_check": synthetic_check},
                                                  indent=2, default=float), encoding="utf-8")
    fc = pd.DataFrame(findings_checks)
    fc.to_csv(OUT / "findings_query_checks.csv", index=False)
    results["20_21_findings_query"] = {"all_checks_pass": bool(fc["ok"].all()), "synthetic_no_positive": synthetic_check,
                                       "validation_images_with_zero_predicted_positives": int((S.sum(1) == 0).sum()),
                                       "status": status(bool(fc["ok"].all()) and synthetic_check["positive_findings_empty"]
                                                        and synthetic_check["does_not_claim_normal"])}

    # ============================================================ 22. document / artifact consistency
    doc = (EXP / "BASELINE_ANALYSIS.md").read_text(encoding="utf-8")
    issues = []
    for line in doc.splitlines():
        cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
        if len(cells) == 10 and cells[0] in EXPECTED_ORDER:
            r = master.set_index("observation").loc[cells[0]]
            claims = {"positive_support": (float(cells[2].replace(",", "")), r["positive_support"], 0),
                      "prevalence": (float(cells[3]), r["prevalence"], 3), "auroc": (float(cells[4]), r["auroc"], 4),
                      "auprc": (float(cells[5]), r["auprc"], 4), "threshold": (float(cells[6]), r["youden_threshold"], 4),
                      "sensitivity": (float(cells[7]), r["sensitivity"], 3), "specificity": (float(cells[8]), r["specificity"], 3),
                      "youden_j": (float(cells[9]), r["recomputed_youden_j"], 3)}
            for k, (claimed, actual, nd) in claims.items():
                if round(float(actual), nd) != claimed:
                    issues.append(f"BASELINE_ANALYSIS per-class {cells[0]} {k}: doc {claimed} vs artifact {round(float(actual), nd)}")
        if len(cells) == 6 and re.fullmatch(r"\d", cells[0]):
            e = int(cells[0])
            hr = hist.set_index("epoch").loc[e]
            for k, col, nd in (("train", "train_loss", 4), ("val", "val_loss", 4), ("macro", "val_macro_auroc", 4),
                               ("micro", "val_micro_auroc", 4)):
                v = float(cells[{"train": 1, "val": 2, "macro": 3, "micro": 4}[k]])
                if round(float(hr[col]), nd) != v:
                    issues.append(f"BASELINE_ANALYSIS epoch {e} {col}: doc {v} vs history {round(float(hr[col]), nd)}")
            doc_lr = cells[5]
            nxt = hist.set_index("epoch")["lr"].get(e + 1)
            actual_lr = f"{hr['lr']:.0e}".replace("e-0", "e-") + (f" → {nxt:.0e}".replace("e-0", "e-") if nxt is not None and nxt != hr["lr"] else "")
            if doc_lr.replace(" ", "") != actual_lr.replace(" ", ""):
                issues.append(f"BASELINE_ANALYSIS epoch {e} LR column: doc '{doc_lr}' vs history '{actual_lr}'")
    op = json.loads((EXP / "operating_points_summary.json").read_text(encoding="utf-8"))
    scalar_claims = [("Macro AUROC", "0.8038", round(macro_auroc, 4)), ("Micro AUROC", "0.8916", round(micro_auroc, 4)),
                     ("Macro AUPRC", "0.4237", round(macro_auprc, 4)), ("Micro AUPRC", "0.6709", round(micro_auprc, 4)),
                     ("predicted positives/image", "5.09", round(op["predicted_positives_per_image_mean"], 2)),
                     ("explicit positives/image", "2.40", round(op["explicit_true_positives_per_image_mean"], 2)),
                     ("NF warning count", "6,529", op["images_with_no_finding_pathology_warning"]),
                     ("peak GPU GB", "2.28", round(tsum["peak_gpu_mem_gb"], 2)),
                     ("total seconds", "7,375", int(tsum["total_seconds"]))]
    for name, claimed, actual in scalar_claims:
        if claimed not in doc:
            issues.append(f"claim '{name}' value {claimed} not found in BASELINE_ANALYSIS")
        if float(claimed.replace(",", "")) != float(actual):
            issues.append(f"claim '{name}': doc {claimed} vs artifact {actual}")
    # BASELINE_ANALYSIS states training "ran all 8 epochs"; log said 'early stopping after epoch 8' while summary says stopped_early=False
    cross = {
        "history_best_epoch == meta epoch == summary best_epoch == threshold provenance epoch":
            len({best_hist_epoch, meta["epoch"], tsum["best_epoch"], thr_file["provenance"]["checkpoint_epoch"],
                 summ["checkpoint_epoch"]}) == 1,
        "summary_metrics n_images == prediction rows": summ["n_images"] == len(preds),
        "threshold file label order == expected": thr_file["labels"] == EXPECTED_ORDER,
        # the `inference` section was added to the live config at C1 finalization (after training)
        "experiment config.yaml == configs/classifier/densenet121.yaml (excluding inference section)":
            exp_cfg == {k: v for k, v in cfg.items() if k != "inference"},
        # CSV stores 16 significant digits, so exact equality is impossible; within 1e-15 means "same value"
        "per_class_metrics thresholds == threshold file (|diff| < 1e-15)": bool(np.allclose(
            pcm.set_index("observation").loc[EXPECTED_ORDER, "threshold"].to_numpy(), T, rtol=0, atol=1e-15)),
    }
    results["22_consistency"] = {"artifact_cross_checks": cross, "documentation_issues": issues,
                                 "status": status(all(cross.values()), warn=bool(issues))}

    # ============================================================ 23. determinism (two fresh inference runs)
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from src.classification.checkpointing import load_checkpoint
    model, _ = load_checkpoint(CKPT, device)
    frame = load_split_frame(load_paths(), cfg["data"], "val")
    ds = CheXpertDataset(frame, load_paths().chexpert_image_base, label_policy_from_config(cfg["label_policy"]),
                         EvalTransform(PreprocessConfig.from_dict(cfg["preprocessing"])))
    runs = []
    for _ in range(2):
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
        runs.append(predict(model, loader, device, policy=LEGACY_C1_AMP)["probs"])
    if list(frame["Path"]) != list(preds["Path"]):
        raise AssertionError("re-inference row order differs from saved predictions")
    np.savez_compressed(OUT / "reinference_probs_fp16.npz", probs=runs[0], paths=np.asarray(frame["Path"]))
    # Stored sens/spec re-derived from the bit-exact in-memory probabilities (no CSV round trip)
    exact_rows = []
    for j, l in enumerate(EXPECTED_ORDER):
        m = V[:, j]
        y, pe, ps = Y[m, j], runs[0][m, j], P[m, j]
        pred_e, pred_s = pe >= T[j], ps >= T[j]
        tp, fp = int((pred_e & (y == 1)).sum()), int((pred_e & (y == 0)).sum())
        fn, tn = int((~pred_e & (y == 1)).sum()), int((~pred_e & (y == 0)).sum())
        exact_rows.append({"observation": l, "tp_exact": tp, "fp_exact": fp, "tn_exact": tn, "fn_exact": fn,
                           "sensitivity_exact": tp / (tp + fn), "specificity_exact": tn / (tn + fp),
                           "precision_exact": tp / (tp + fp) if tp + fp else np.nan,
                           "stored_sensitivity": thr_saved[l]["sensitivity"], "stored_specificity": thr_saved[l]["specificity"],
                           "sens_abs_diff_exact_vs_stored": abs(tp / (tp + fn) - thr_saved[l]["sensitivity"]),
                           "spec_abs_diff_exact_vs_stored": abs(tn / (tn + fp) - thr_saved[l]["specificity"]),
                           "bruteforce_threshold_exact": brute_force_youden(y, pe)["threshold"],
                           "ties_exactly_at_threshold": int((pe == T[j]).sum()),
                           "status_flips_csv_vs_exact": int((pred_e != pred_s).sum())})
    exact = pd.DataFrame(exact_rows)
    exact["threshold_bitexact_vs_stored"] = exact["bruteforce_threshold_exact"].to_numpy() == T
    exact.to_csv(OUT / "threshold_precision_check.csv", index=False)
    master_exact = master.merge(exact[["observation", "tp_exact", "fp_exact", "tn_exact", "fn_exact", "sensitivity_exact",
                                       "specificity_exact", "precision_exact", "status_flips_csv_vs_exact"]], on="observation")
    vcols = cols + ["tp_exact", "fp_exact", "tn_exact", "fn_exact", "sensitivity_exact", "specificity_exact",
                    "precision_exact", "status_flips_csv_vs_exact"]
    master_exact[vcols].to_csv(PROJECT_ROOT / "results/classification/CLASSIFIER_VERIFICATION.csv", index=False)
    results["9b_threshold_precision"] = {
        "csv_float_digits_written": "16 significant digits (pandas default); exact float64 round trip needs 17",
        "thresholds_bitexact_on_inmemory_probs": bool(exact["threshold_bitexact_vs_stored"].all()),
        "max_sens_diff_exact_vs_stored": float(exact["sens_abs_diff_exact_vs_stored"].max()),
        "max_spec_diff_exact_vs_stored": float(exact["spec_abs_diff_exact_vs_stored"].max()),
        "ties_exactly_at_threshold_total": int(exact["ties_exactly_at_threshold"].sum()),
        "status_flips_csv_vs_exact_total": int(exact["status_flips_csv_vs_exact"].sum()),
        "status": status(bool(exact["threshold_bitexact_vs_stored"].all())
                         and exact["sens_abs_diff_exact_vs_stored"].max() < 1e-12, warn=True)}
    results["9_youden"]["status"] = status(bool(exact["threshold_bitexact_vs_stored"].all())
                                           and exact["sens_abs_diff_exact_vs_stored"].max() < 1e-12
                                           and exact["spec_abs_diff_exact_vs_stored"].max() < 1e-12
                                           and thr_file["provenance"]["source_split"] == "val")
    # documentation sens/spec/J are judged against the bit-exact values, not the lossy CSV
    ex_i = exact.set_index("observation")
    doc_rows = {}
    for line in (EXP / "BASELINE_ANALYSIS.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
        if len(cells) == 10 and cells[0] in EXPECTED_ORDER:
            doc_rows[cells[0]] = cells
    exact_doc_issues = []
    for l, cells in doc_rows.items():
        for k, col, idx in (("sensitivity", "sensitivity_exact", 7), ("specificity", "specificity_exact", 8)):
            if round(float(ex_i.loc[l, col]), 3) != float(cells[idx]):
                exact_doc_issues.append(f"BASELINE_ANALYSIS per-class {l} {k}: doc {cells[idx]} vs exact {round(float(ex_i.loc[l, col]), 3)}")
    results["22_consistency"]["documentation_issues"] = (
        [i for i in results["22_consistency"]["documentation_issues"] if not re.search(r"(sensitivity|specificity|youden_j):", i)]
        + exact_doc_issues)
    results["22_consistency"]["status"] = status(all(results["22_consistency"]["artifact_cross_checks"].values()),
                                                 warn=bool(results["22_consistency"]["documentation_issues"]))
    # fp32 single-image inference path (scripts/infer_classifier.py) vs fp16 batch path used to fit thresholds
    rng_idx = np.arange(0, len(ds), max(1, len(ds) // 512))[:512]
    sub = torch.utils.data.Subset(ds, rng_idx.tolist())
    p32 = predict(model, DataLoader(sub, batch_size=64, shuffle=False, num_workers=0), device, policy=CANONICAL)["probs"]
    p16 = runs[0][rng_idx]
    results["23b_fp32_vs_fp16"] = {
        "n_images": int(len(rng_idx)), "selection": "every k-th validation image (deterministic)",
        "max_abs_prob_diff": float(np.abs(p32 - p16).max()), "median_abs_prob_diff": float(np.median(np.abs(p32 - p16))),
        "status_flips": int(((p32 >= T) != (p16 >= T)).sum()), "label_decisions": int(p32.size)}
    d12 = float(np.abs(runs[0] - runs[1]).max())
    d1s = float(np.abs(runs[0] - P).max())
    flips12 = int(((runs[0] >= T) != (runs[1] >= T)).sum())
    flips1s = int(((runs[0] >= T) != (P >= T)).sum())
    results["23_reproducibility"] = {"max_abs_diff_run1_vs_run2": d12, "status_flips_run1_vs_run2": flips12,
                                     "max_abs_diff_run1_vs_saved": d1s, "status_flips_run1_vs_saved": flips1s,
                                     "macro_auroc_run1": float(np.mean([roc_auc_score(Y[V[:, j], j], runs[0][V[:, j], j]) for j in range(14)])),
                                     "status": status(d12 < 1e-6 and flips12 == 0, warn=flips1s > 0 or d1s > 1e-4)}

    (OUT / "audit_results.json").write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                            encoding="utf-8")
    for k, v in results.items():
        print(k, v.get("status", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
