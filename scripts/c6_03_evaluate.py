"""C6 steps 4-14: apply the frozen calibrators and the frozen C4 policy to the stored test scores and compute
every test metric. Nothing is fitted or tuned; the validation numbers are recomputed with the SAME functions
and checked against the stored C2-C5 results.

    .venv\\Scripts\\python.exe -m scripts.c6_03_evaluate
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pandas as pd

from src.classification.calibration import adaptive_bins, class_metrics
from src.classification.final_test import (IDX12, IDX13, NF, RARE, apply_frozen_pipeline, binary_table, bootstrap_weights,
                                           count_stats, load_frozen, macro_binary, micro_binary, output_record, sha256_file,
                                           validate_record, verify_freeze, weighted_macro_metrics)
from src.classification.labels import LABELS
from src.classification.metrics import per_class_metrics, summary_metrics
from src.classification.serialization import load_predictions_csv, load_predictions_npz
from src.utils.audit import setup_logging
from src.utils.config import PROJECT_ROOT

log = logging.getLogger("c6_evaluate")
EXP = PROJECT_ROOT / "results/classification/experiments"
OUT = EXP / "c6_final_test"
N_BOOT, SEED, N_BINS = 1000, 42, 10
REPRESENTATIVE = {"rare": "Pleural Other", "medium": "Pneumothorax", "common": "Support Devices"}   # as fixed in C5 (validation prevalence)
FMT = "%.17g"


def anon(prefix: str, x: str) -> str:
    return prefix + hashlib.sha256(f"c6-anon-v1|{x}".encode()).hexdigest()[:12]


def evaluate(split: str, scores, raw, valid, patient_ids, policy, cal) -> dict:
    """Everything that is computed identically for validation and test."""
    out = apply_frozen_pipeline(scores, policy, cal)
    y = (raw == 1).astype(float)
    pc = per_class_metrics(y, scores, valid)
    sm = summary_metrics(y, scores, valid)
    bt = binary_table(out["pred_final"], raw, valid, out["thresholds"])
    bt_raw = binary_table(out["pred_raw"], raw, valid, out["thresholds"])
    cal_rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        r = class_metrics(scores[v, j], y[v, j], N_BINS)
        c = class_metrics(out["prob"][v, j], y[v, j], N_BINS)
        cal_rows.append({"observation": lab, "prevalence": c["prevalence"], "mean_calibrated_probability": c["mean_predicted"],
                         "brier": c["brier"], "log_loss": c["log_loss"], "ece": c["ece"],
                         "cal_slope": c["cal_slope"], "cal_intercept_slope1": c["cal_intercept_slope1"],
                         "raw_brier": r["brier"], "raw_log_loss": r["log_loss"], "raw_ece": r["ece"],
                         "raw_mean_score": r["mean_predicted"], "raw_cal_slope": r["cal_slope"]})
    cal_t = pd.DataFrame(cal_rows)
    gt = raw == 1
    # No Finding analysis
    pr, pf = out["pred_raw"], out["pred_final"]
    contra_before = pr[:, NF] & pr[:, IDX12].any(1)
    contra_after = pf[:, NF] & pf[:, IDX12].any(1)
    gt_contra = gt[:, NF] & gt[:, IDX12].any(1)
    nf = {"n_images": int(len(scores)), "no_finding_prevalence": float(bt.iloc[NF]["prevalence"]),
          "precision": float(bt.iloc[NF].precision), "recall": float(bt.iloc[NF].recall),
          "specificity": float(bt.iloc[NF].specificity), "f1": float(bt.iloc[NF].f1),
          "balanced_accuracy": float(bt.iloc[NF].balanced_accuracy),
          "pre_rule_precision": float(bt_raw.iloc[NF].precision), "pre_rule_recall": float(bt_raw.iloc[NF].recall),
          "pre_rule_specificity": float(bt_raw.iloc[NF].specificity), "pre_rule_f1": float(bt_raw.iloc[NF].f1),
          "pct_predicted_no_finding_before_rule": float(100 * pr[:, NF].mean()),
          "pct_predicted_no_finding_after_rule": float(100 * pf[:, NF].mean()),
          "contradiction_n_before_rule": int(contra_before.sum()), "contradiction_pct_of_images_before_rule": float(100 * contra_before.mean()),
          "contradiction_pct_of_predicted_no_finding_before_rule": float(100 * contra_before.sum() / max(pr[:, NF].sum(), 1)),
          "contradiction_n_after_rule": int(contra_after.sum()), "contradiction_pct_of_images_after_rule": float(100 * contra_after.mean()),
          "images_modified_by_rule": int((pr[:, NF] != pf[:, NF]).sum()),
          "ground_truth_no_finding_positive": int(gt[:, NF].sum()),
          "ground_truth_contradiction_with_12_pathology_n": int(gt_contra.sum()),
          "ground_truth_no_finding_with_support_devices_n": int((gt[:, NF] & gt[:, LABELS.index("Support Devices")]).sum()),
          "ground_truth_no_finding_with_uncertain_pathology_n": int((gt[:, NF] & (raw[:, IDX12] == -1).any(1)).sum())}
    counts = {}
    for name, idx in (("13_non_no_finding", IDX13), ("12_pathology", IDX12)):
        counts[name] = {"ground_truth": count_stats(gt[:, idx].sum(1)), "predicted": count_stats(pf[:, idx].sum(1))}
    return {"pipeline": out, "ranking_pc": pc, "ranking_sum": sm, "binary": bt, "binary_pre_rule": bt_raw, "calibration": cal_t,
            "no_finding": nf, "counts": counts}


def macro_cal(t: pd.DataFrame, prefix: str = "") -> dict:
    return {k: float(t[prefix + k].mean()) for k in ("brier", "log_loss", "ece")}


def main() -> int:
    setup_logging()
    mp = OUT / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    verify_freeze(mp, PROJECT_ROOT)
    manifest_sha = sha256_file(mp)
    policy, cal = load_frozen(EXP / "c4_operating_policy/final_operating_policy.json", EXP / "c5_calibration/final_calibrators.json")
    # ---------------------------------------------------------------- load stored scores (no inference here)
    te = load_predictions_npz(OUT / "test_raw_predictions.npz")
    te_csv = load_predictions_csv(OUT / "test_raw_predictions.csv.gz")
    assert np.array_equal(te["probs"], te_csv["probs"]) and te["paths"] == te_csv["paths"]
    va = load_predictions_npz(EXP / "c2b_sqrt_weighted_bce/evaluation/fp32_strict/validation_predictions.npz")
    for d in (te, va):
        assert np.array_equal(d["valid"], d["raw_labels"] != -1), "valid mask must be exactly 'not uncertain'"
    R_te = evaluate("test", te["probs"], te["raw_labels"], te["valid"], te["patient_ids"], policy, cal)
    R_va = evaluate("validation", va["probs"], va["raw_labels"], va["valid"], va["patient_ids"], policy, cal)
    n_te, n_pat = len(te["paths"]), len(set(te["patient_ids"]))

    # ---------------------------------------------------------------- validation reproduction (same functions, stored C2-C5 results)
    c4 = EXP / "c4_operating_policy"
    c5 = EXP / "c5_calibration"
    long = pd.read_csv(c4 / "policy_per_class_long.csv", float_precision="round_trip")
    ref = long[long.policy == "f1_nf"].reset_index(drop=True)
    bt = R_va["binary"]
    d_counts = int(sum((bt[k].to_numpy() != ref[k].to_numpy()).sum() for k in ("tp", "fp", "tn", "fn")))
    d_metrics = float(max(np.nanmax(np.abs(bt[k].to_numpy() - ref[k].to_numpy())) for k in ("precision", "recall", "specificity", "f1", "balanced_accuracy")))
    pcmp = pd.read_csv(c4 / "policy_comparison.csv", float_precision="round_trip").query("policy == 'f1_nf' and scope == '14 classes'").iloc[0]
    mac_va = macro_binary(bt)
    d_macro = float(max(abs(mac_va[f"macro_{k}"] - pcmp[f"macro_{k}"]) for k in ("precision", "recall", "specificity", "f1", "balanced_accuracy")))
    c2b_sum = json.loads((EXP / "c2b_sqrt_weighted_bce/evaluation/fp32_strict/summary_metrics.json").read_text(encoding="utf-8"))
    d_rank = float(max(abs(R_va["ranking_sum"][k] - c2b_sum[k]) for k in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")))
    c4fc = pd.read_csv(c4 / "finding_counts_per_image.csv", float_precision="round_trip")
    fc_ref = c4fc[c4fc.definition.str.startswith("13") & (c4fc.policy == "ground_truth")].iloc[0]
    fc_f1 = c4fc[c4fc.definition.str.startswith("13") & (c4fc.policy == "f1")].iloc[0]
    d_counts_stats = float(max(abs(R_va["counts"]["13_non_no_finding"]["ground_truth"][k] - fc_ref[k]) for k in ("mean", "median", "p25", "p75", "p95", "max", "pct_0", "pct_gt5"))
                           + max(abs(R_va["counts"]["13_non_no_finding"]["predicted"][k] - fc_f1[k]) for k in ("mean", "median", "p25", "p75", "p95", "max", "pct_0", "pct_gt5")))
    c4r = json.loads((c4 / "c4_results.json").read_text(encoding="utf-8"))
    r5 = pd.read_csv(c5 / "raw_vs_selected_per_class.csv", float_precision="round_trip")
    vc = pd.read_csv(c5 / "validation_predictions_calibrated.csv.gz", float_precision="round_trip")
    oof = np.column_stack([vc[f"{l}_prob_calibrated_oof"].to_numpy() for l in LABELS])
    full = np.column_stack([vc[f"{l}_prob_calibrated"].to_numpy() for l in LABELS])
    d_cal_full = float(np.abs(full - R_va["pipeline"]["prob"]).max())
    oof_rows = []
    for j, lab in enumerate(LABELS):
        v = va["valid"][:, j]
        oof_rows.append(class_metrics(oof[v, j], (va["raw_labels"][v, j] == 1).astype(float), N_BINS))
    oof_df = pd.DataFrame(oof_rows)
    d_oof = float(np.abs(oof_df["brier"].to_numpy() - r5["calibrated_brier"].to_numpy()).max())
    d_raw_cal = float(np.abs(R_va["calibration"]["raw_brier"].to_numpy() - r5["raw_brier"].to_numpy()).max())
    repro = {"binary_count_differences": d_counts, "binary_metric_max_abs_diff": d_metrics, "macro_binary_max_abs_diff": d_macro,
             "ranking_macro_micro_max_abs_diff_vs_c2b": d_rank, "finding_count_stat_abs_diff": d_counts_stats,
             "no_finding_rule_images_changed": R_va["no_finding"]["images_modified_by_rule"], "c4_images_changed": c4r["rule"]["images_changed"],
             "calibrated_probs_vs_c5_stored_max_abs_diff": d_cal_full, "c5_oof_brier_max_abs_diff": d_oof, "c5_raw_brier_max_abs_diff": d_raw_cal}
    repro["reproduced"] = bool(d_counts == 0 and d_metrics < 1e-12 and d_macro < 1e-12 and d_rank < 1e-9 and d_counts_stats < 1e-9
                               and repro["no_finding_rule_images_changed"] == repro["c4_images_changed"] and d_cal_full < 1e-12
                               and d_oof < 1e-12 and d_raw_cal < 1e-12)
    (OUT / "validation_reproduction.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    if not repro["reproduced"]:
        raise AssertionError(f"validation reference did not reproduce stored C2-C5 results: {repro}")
    log.info("validation reference reproduced: %s", repro)

    # ---------------------------------------------------------------- step 4/5: prediction files
    pip = R_te["pipeline"]
    y_te = (te["raw_labels"] == 1)
    cols_cal, cols_pol = {"image_id": te["paths"], "patient_id": te["patient_ids"]}, {"image_id": te["paths"], "patient_id": te["patient_ids"]}
    for j, lab in enumerate(LABELS):
        lab_gt = np.where(te["valid"][:, j], y_te[:, j].astype(float), np.nan)
        cols_cal[f"{lab}_label"] = lab_gt
        cols_cal[f"{lab}_label_raw"] = te["raw_labels"][:, j]
        cols_cal[f"{lab}_score_raw"] = pip["scores"][:, j]
        cols_cal[f"{lab}_prob_calibrated"] = pip["prob"][:, j]
        cols_pol[f"{lab}_label"] = lab_gt
        cols_pol[f"{lab}_score_raw"] = pip["scores"][:, j]
        cols_pol[f"{lab}_prob_calibrated"] = pip["prob"][:, j]
        cols_pol[f"{lab}_threshold_raw"] = pip["thresholds"][j]
        cols_pol[f"{lab}_pred_raw"] = pip["pred_raw"][:, j].astype(int)
        cols_pol[f"{lab}_pred_final"] = pip["pred_final"][:, j].astype(int)
    comp = {"method": "gzip", "mtime": 0}
    pd.DataFrame(cols_cal).to_csv(OUT / "test_predictions_calibrated.csv.gz", index=False, float_format=FMT, compression=comp)
    pd.DataFrame(cols_pol).to_csv(OUT / "test_predictions_final_policy.csv.gz", index=False, float_format=FMT, compression=comp)
    chk = pd.read_csv(OUT / "test_predictions_final_policy.csv.gz", float_precision="round_trip")
    back_s = np.column_stack([chk[f"{l}_score_raw"] for l in LABELS])
    back_f = np.column_stack([chk[f"{l}_pred_final"] for l in LABELS]).astype(bool)
    file_checks = {"raw_scores_bit_identical_after_reload": bool(np.array_equal(back_s, te["probs"])),
                   "final_preds_equal_rule_on_thresholded_scores": bool(np.array_equal(back_f, pip["pred_final"])),
                   "only_no_finding_differs_between_raw_and_final": bool(np.array_equal(np.delete(pip["pred_raw"], NF, 1), np.delete(pip["pred_final"], NF, 1))),
                   "calibrated_probability_range": [float(pip["prob"].min()), float(pip["prob"].max())],
                   "calibrated_decisions_equal_raw_decisions_mismatches": int(((pip["prob"] >= pip["calibrated_thresholds"]) != (pip["scores"] >= pip["thresholds"])).sum()),
                   "rows": int(len(chk))}
    assert file_checks["raw_scores_bit_identical_after_reload"] and file_checks["final_preds_equal_rule_on_thresholded_scores"]
    assert file_checks["only_no_finding_differs_between_raw_and_final"]

    # ---------------------------------------------------------------- step 6-8 tables (test)
    pc, sm = R_te["ranking_pc"], R_te["ranking_sum"]
    pc = pc.rename(columns={"n_positive": "positive_count", "n_negative": "negative_count"})
    pc.to_csv(OUT / "test_ranking_metrics.csv", index=False, float_format=FMT)
    (OUT / "test_ranking_summary.json").write_text(json.dumps({"split": "test", "n_images": n_te, "n_patients": n_pat, **sm}, indent=2), encoding="utf-8")
    bt = R_te["binary"].copy()
    bt["positives"], bt["negatives"] = bt.tp + bt.fn, bt.tn + bt.fp
    bt.to_csv(OUT / "test_binary_metrics_per_class.csv", index=False, float_format=FMT)
    macro14, macro13 = macro_binary(R_te["binary"]), macro_binary(R_te["binary"], IDX13)
    bsum = {"split": "test", "n_images": n_te, "policy": "f1_optimal_plus_no_finding_consistency",
            "macro_over_14_classes_primary": macro14, "macro_over_13_non_no_finding_classes": macro13,
            "micro_pooled_over_14_classes_secondary": micro_binary(R_te["binary"]),
            "n_classes_defined": {m: int(R_te["binary"][m].notna().sum()) for m in ("precision", "recall", "specificity", "f1", "balanced_accuracy")},
            "images_with_no_predicted_positive": int((~pip["pred_final"].any(1)).sum())}
    (OUT / "test_binary_summary.json").write_text(json.dumps(bsum, indent=2), encoding="utf-8")
    ct = R_te["calibration"]
    ct.to_csv(OUT / "test_calibration_per_class.csv", index=False, float_format=FMT)
    csum = {"split": "test", "n_images": n_te, "ece_definition": f"{N_BINS} adaptive equal-frequency bins (identical to C5)",
            "macro_calibrated": macro_cal(ct), "macro_raw_score_for_reference": macro_cal(ct, "raw_"),
            "fraction_of_classes_with_lower_brier_after_calibration": float((ct.brier < ct.raw_brier).mean()),
            "fraction_of_classes_with_lower_log_loss_after_calibration": float((ct.log_loss < ct.raw_log_loss).mean())}
    (OUT / "test_calibration_summary.json").write_text(json.dumps(csum, indent=2), encoding="utf-8")
    bins = []
    for series, P in (("raw", pip["scores"]), ("calibrated", pip["prob"])):
        for j, lab in enumerate(LABELS):
            v = te["valid"][:, j]
            for i, b in enumerate(adaptive_bins(P[v, j], y_te[v, j], N_BINS)):
                bins.append({"observation": lab, "series": series, "bin": i, **b})
    pd.DataFrame(bins).to_csv(OUT / "test_reliability_bins.csv", index=False, float_format=FMT)

    # ---------------------------------------------------------------- step 9: validation vs test
    vb, tb = R_va["binary"], R_te["binary"]
    vsum, mv, mt = R_va["ranking_sum"], macro_binary(vb), macro14
    vcal = macro_cal(R_va["calibration"])
    vcal_oof = {k: float(oof_df[k].mean()) for k in ("brier", "log_loss", "ece")}
    rows = []
    for k in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc"):
        rows.append({"group": "ranking", "metric": k, "validation": vsum[k], "test": sm[k]})
    for m in ("precision", "recall", "specificity", "f1", "balanced_accuracy"):
        rows.append({"group": "binary (macro, 14 classes)", "metric": f"macro_{m}", "validation": mv[f"macro_{m}"], "test": macro14[f"macro_{m}"]})
    for k in ("brier", "log_loss", "ece"):
        rows.append({"group": "calibration (frozen Platt; macro)", "metric": f"macro_{k}", "validation": vcal[k], "test": macro_cal(ct)[k],
                     "validation_out_of_fold": vcal_oof[k]})
    cmp_df = pd.DataFrame(rows)
    cmp_df["difference_test_minus_validation"] = cmp_df["test"] - cmp_df["validation"]
    cmp_df.to_csv(OUT / "validation_vs_test_summary.csv", index=False, float_format=FMT)
    pcv = R_va["ranking_pc"].set_index("observation")
    per_cls = pd.DataFrame({"observation": list(LABELS)})
    for nm, V, T in (("auroc", pcv["auroc"], pc.set_index("observation")["auroc"]), ("auprc", pcv["auprc"], pc.set_index("observation")["auprc"]),
                     ("prevalence", pcv["prevalence"], pc.set_index("observation")["prevalence"])):
        per_cls[f"validation_{nm}"] = per_cls.observation.map(V)
        per_cls[f"test_{nm}"] = per_cls.observation.map(T)
    for m in ("precision", "recall", "specificity", "f1", "balanced_accuracy"):
        per_cls[f"validation_{m}"] = vb[m].to_numpy()
        per_cls[f"test_{m}"] = tb[m].to_numpy()
    for k in ("brier", "log_loss", "ece"):
        per_cls[f"validation_{k}"] = R_va["calibration"][k].to_numpy()
        per_cls[f"test_{k}"] = ct[k].to_numpy()
    per_cls.to_csv(OUT / "validation_vs_test_per_class.csv", index=False, float_format=FMT)

    # ---------------------------------------------------------------- step 10: rare classes (C2 definition, not test prevalence)
    rr = []
    for lab in RARE:
        j = LABELS.index(lab)
        for sp, R in (("validation", R_va), ("test", R_te)):
            b, p_ = R["binary"].iloc[j], R["ranking_pc"].iloc[j]
            rr.append({"observation": lab, "split": sp, "prevalence": float(p_.prevalence), "auroc": float(p_.auroc), "auprc": float(p_.auprc),
                       "precision": float(b.precision), "recall": float(b.recall), "specificity": float(b.specificity), "f1": float(b.f1),
                       "tp": int(b.tp), "fp": int(b.fp), "fp_per_tp": float(b.fp_per_tp), "threshold": float(b.threshold)})
    rare_df = pd.DataFrame(rr)
    rare_df.to_csv(OUT / "test_rare_class_analysis.csv", index=False, float_format=FMT)

    # ---------------------------------------------------------------- steps 11/12: No Finding and finding counts
    nf_df = pd.DataFrame([{"split": "validation", **R_va["no_finding"]}, {"split": "test", **R_te["no_finding"]}])
    nf_df.to_csv(OUT / "test_no_finding_analysis.csv", index=False, float_format=FMT)
    crow = []
    for sp, R in (("validation", R_va), ("test", R_te)):
        for dfn, d in R["counts"].items():
            for who, s in d.items():
                crow.append({"split": sp, "definition": dfn, "source": who, **s})
    counts_df = pd.DataFrame(crow)
    counts_df.to_csv(OUT / "test_finding_counts.csv", index=False, float_format=FMT)
    cnt_dist = []
    for sp, (P, G) in (("test", (pip["pred_final"], y_te)),):
        for who, M in (("ground_truth", G), ("predicted", P)):
            n = M[:, IDX13].sum(1)
            cnt_dist.append({"split": sp, "source": who, **{str(i) if i < 10 else "10+": float(100 * (np.minimum(n, 10) == i).mean()) for i in range(11)}})
    pd.DataFrame(cnt_dist).to_csv(OUT / "test_finding_count_distribution.csv", index=False, float_format=FMT)
    pairs = pd.DataFrame({"ground_truth": y_te[:, IDX13].sum(1), "predicted": pip["pred_final"][:, IDX13].sum(1)})
    pairs.value_counts().rename("n_images").reset_index().to_csv(OUT / "test_finding_count_pairs.csv", index=False)

    # ---------------------------------------------------------------- step 13: patient-level bootstrap
    s, pr_, pf_, rl, vd = pip["scores"], pip["prob"], pip["pred_final"], te["raw_labels"], te["valid"]
    point = weighted_macro_metrics(s, pr_, pf_, rl, vd, np.ones(n_te, dtype=np.int64))
    assert abs(point["macro_auroc"] - sm["macro_auroc"]) < 1e-12 and abs(point["macro_f1"] - macro14["macro_f1"]) < 1e-12
    draws = {k: [] for k in point}
    for w in bootstrap_weights(te["patient_ids"], N_BOOT, SEED):
        for k, v in weighted_macro_metrics(s, pr_, pf_, rl, vd, w).items():
            draws[k].append(v)
    D = {k: np.asarray(v) for k, v in draws.items()}
    np.savez_compressed(OUT / "test_bootstrap_draws.npz", seed=SEED, n_boot=N_BOOT, **D)
    brow = [{"metric": k, "point_estimate": point[k], "ci95_low": float(np.nanpercentile(v, 2.5)), "ci95_high": float(np.nanpercentile(v, 97.5)),
             "bootstrap_mean": float(np.nanmean(v)), "bootstrap_sd": float(np.nanstd(v, ddof=1)), "n_resamples": N_BOOT, "seed": SEED,
             "unit": "patient", "n_nan_draws": int(np.isnan(v).sum())} for k, v in D.items()]
    boot_df = pd.DataFrame(brow)
    boot_df.to_csv(OUT / "test_bootstrap_summary.csv", index=False, float_format=FMT)

    # ---------------------------------------------------------------- step 14: error examples (descriptive)
    ex = []
    for j, lab in enumerate(LABELS):
        v = np.where(vd[:, j])[0]
        fp_idx = v[(pf_[v, j]) & ~y_te[v, j]]
        fn_idx = v[(~pf_[v, j]) & y_te[v, j]]
        for kind, idxs, order in (("high_confidence_false_positive", fp_idx, -s[fp_idx, j]), ("high_confidence_false_negative", fn_idx, s[fn_idx, j])):
            for rank, i in enumerate(idxs[np.argsort(order, kind="mergesort")][:5], 1):
                ex.append({"observation": lab, "error_type": kind, "rank": rank, "anonymized_image_id": anon("img_", te["paths"][i]),
                           "anonymized_patient_id": anon("pt_", te["patient_ids"][i]), "raw_score": float(s[i, j]),
                           "calibrated_probability": float(pr_[i, j]), "threshold": float(pip["thresholds"][j]),
                           "n_true_positive_labels_in_image": int(y_te[i, IDX13].sum()), "n_predicted_positive_findings_in_image": int(pf_[i, IDX13].sum()),
                           "suppressed_by_no_finding_rule": bool(j == NF and pip["pred_raw"][i, NF] and not pf_[i, NF])})
    pd.DataFrame(ex).to_csv(OUT / "test_error_examples.csv", index=False, float_format=FMT)

    # ---------------------------------------------------------------- schema validation of every downstream record
    for i in range(n_te):
        validate_record(output_record(anon("img_", te["paths"][i]), pip["scores"][i], pip["prob"][i], pip["thresholds"],
                                      pip["calibrated_thresholds"], pip["pred_final"][i]))

    results = {"n_images": n_te, "n_patients": n_pat, "freeze_manifest_sha256": manifest_sha, "ranking": sm, "binary_macro14": macro14,
               "binary_macro13": macro13, "binary_micro": bsum["micro_pooled_over_14_classes_secondary"], "calibration": csum,
               "validation": {"ranking": {k: vsum[k] for k in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc")},
                              "binary_macro14": mv, "calibration_frozen_in_sample": vcal, "calibration_out_of_fold": vcal_oof},
               "no_finding_test": R_te["no_finding"], "no_finding_validation": R_va["no_finding"],
               "counts_test": R_te["counts"], "counts_validation": R_va["counts"], "file_checks": file_checks,
               "records_validated_against_schema": n_te, "representative_classes": REPRESENTATIVE, "rare_classes": RARE,
               "bootstrap": {"n_resamples": N_BOOT, "seed": SEED, "unit": "patient"}, "validation_reproduction": repro}
    (OUT / "c6_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"ranking": sm, "macro14": macro14, "cal": csum["macro_calibrated"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
