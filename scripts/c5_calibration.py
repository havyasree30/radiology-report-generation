"""C5: probability calibration analysis on the C2-B validation scores (validation only, no inference).

Steps: reproduce C4 -> raw calibration baseline -> 5-fold PATIENT-level cross-validation of raw /
temperature / Platt / isotonic -> per-class method selection (rule declared below, before any result) ->
final calibrators fitted on the full validation set -> calibrated-equivalent C4 thresholds ->
per-image output. The C4 binary policy is never changed. The locked test split is never opened.

    .venv\\Scripts\\python.exe -m scripts.c5_calibration
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.compare_losses import COMMON, RARE  # noqa: F401  (groups fixed in C2)
from src.classification.calibration import (COMPLEXITY, METHODS, adaptive_bins, apply_calibrator, calibrated_threshold,
                                            class_metrics, decision_mismatches, fit_calibrator, is_strictly_increasing,
                                            patient_folds, safe_logit, save_calibrators)
from src.classification.labels import LABELS
from src.classification.operating_points import apply_thresholds, load_threshold_file, metrics_from_counts
from src.classification.operating_policy import apply_no_finding_rule, load_final_policy
from src.classification.serialization import load_predictions_npz
from src.utils.audit import setup_logging
from src.utils.config import PROJECT_ROOT
from src.utils.reproducibility import environment_snapshot

log = logging.getLogger("c5")
EXPD = PROJECT_ROOT / "results/classification/experiments"
C3, C4, OUT = EXPD / "c3_threshold_optimization", EXPD / "c4_operating_policy", EXPD / "c5_calibration"
N_FOLDS, SEED, N_BINS = 5, 42, 10
# Declared BEFORE any C5 number was inspected:
SELECTION = {"tolerance_rel": 0.005,            # a simpler method is preferred if within 0.5% of the best on BOTH log loss and Brier
             "uniform_min_classes": 10,         # one method chosen for >= 10 classes ...
             "uniform_max_rel_loss": 0.01}      # ... and costing <= 1% (log loss and Brier) in every class -> use it everywhere
METRICS = ("brier", "log_loss", "ece")


def representative(prev: dict) -> dict:
    order = sorted(LABELS, key=lambda l: (prev[l], list(LABELS).index(l)))
    return {"rare": order[0], "medium": order[len(order) // 2], "common": order[-1]}


def main() -> int:
    setup_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    pred = load_predictions_npz(EXPD / "c2b_sqrt_weighted_bce/evaluation/fp32_strict/validation_predictions.npz")
    p, raw, valid = pred["probs"], pred["raw_labels"], pred["valid"]
    n, pats = len(p), pred["patient_ids"]
    y_all = raw == 1
    info = json.loads((C3 / "input_verification.json").read_text(encoding="utf-8"))
    ck = info["checkpoint"]

    # ================================================================ Step 1: reproduce C4
    pol = load_final_policy(C4 / "final_operating_policy.json")
    thr = pol["thresholds"]
    c4 = pd.read_csv(C4 / "validation_predictions_final_policy.csv.gz", float_precision="round_trip")
    raw_pred = apply_thresholds(p, thr)
    final_pred = apply_no_finding_rule(raw_pred)
    c4_final = c4[[f"{l}_pred_final" for l in LABELS]].to_numpy().astype(bool)
    rep = {"n_images": n, "n_patients": len(set(pats)), "class_order_ok": pol["label_order"] == list(LABELS),
           "image_ids_identical": c4["image_id"].tolist() == pred["paths"], "patient_ids_identical": c4["patient_id"].tolist() == pats,
           "scores_bit_identical": bool(np.array_equal(c4[[f"{l}_score" for l in LABELS]].to_numpy(), p)),
           "thresholds_equal_thresholds_f1_json": bool(np.array_equal(thr, load_threshold_file(C3 / "thresholds_f1.json"))),
           "raw_predictions_equal": bool(np.array_equal(c4[[f"{l}_pred_raw" for l in LABELS]].to_numpy().astype(bool), raw_pred)),
           "final_predictions_equal_rule": bool(np.array_equal(c4_final, final_pred)),
           "no_finding_rule": pol["no_finding_rule"], "rule_abnormal_labels": pol["no_finding_rule_abnormal_labels"]}
    rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        y, pr = y_all[v, j], final_pred[v, j]
        m = metrics_from_counts(int((pr & y).sum()), int((pr & ~y).sum()), int((~pr & ~y).sum()), int((~pr & y).sum()))
        rows.append({k: m[k] for k in ("precision", "recall", "specificity", "f1", "balanced_accuracy")})
    mine14 = pd.DataFrame(rows).mean()
    mine13 = pd.DataFrame(rows).iloc[1:].mean()
    cp = pd.read_csv(C4 / "policy_comparison.csv", float_precision="round_trip")
    ref14 = cp[(cp.policy == "f1_nf") & (cp.scope == "14 classes")].iloc[0]
    ref13 = cp[(cp.policy == "f1_nf") & cp.scope.str.startswith("13")].iloc[0]
    rep["max_abs_diff_macro_vs_c4_14"] = float(max(abs(mine14[m] - ref14[f"macro_{m}"]) for m in mine14.index))
    rep["max_abs_diff_macro_vs_c4_13"] = float(max(abs(mine13[m] - ref13[f"macro_{m}"]) for m in mine13.index))
    rep["reproduced"] = bool(all(v for k, v in rep.items() if isinstance(v, bool)) and rep["max_abs_diff_macro_vs_c4_14"] < 1e-12
                             and rep["max_abs_diff_macro_vs_c4_13"] < 1e-12)
    (OUT / "c4_reproduction.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["reproduced"]:
        raise AssertionError(f"C4 did NOT reproduce: {json.dumps(rep, indent=1)}")
    log.info("C4 reproduced (%d images, %d patients; macro diff %.1e)", n, rep["n_patients"], rep["max_abs_diff_macro_vs_c4_14"])

    # ================================================================ logits
    z = safe_logit(p)
    f32_exact = bool(np.array_equal(p, p.astype(np.float32).astype(np.float64)))
    ulp = np.spacing(p.astype(np.float32)).astype(np.float64)
    logit_err = 0.5 * ulp / (p * (1 - p))
    lg = {"scores_are_float32_values": f32_exact, "min_score": float(p.min()), "max_score": float(p.max()),
          "n_clipped": int(((p <= 1e-12) | (p >= 1 - 1e-12)).sum()),
          "max_logit_rounding_error_from_float32_storage": float(logit_err.max()),
          "median_logit_rounding_error": float(np.median(logit_err)), "logit_min": float(z.min()), "logit_max": float(z.max()),
          "note": "stored scores are float32 sigmoid outputs held as float64; logit(p) is recovered to the stated bound, no inference rerun"}
    (OUT / "logit_reconstruction.json").write_text(json.dumps(lg, indent=2), encoding="utf-8")
    assert lg["n_clipped"] == 0

    # ================================================================ Step 2: raw baseline
    raw_rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        raw_rows.append({"observation": lab, "n_valid": int(v.sum()), "n_positive": int(y_all[v, j].sum()),
                         **class_metrics(p[v, j], y_all[v, j], N_BINS)})
    rawdf = pd.DataFrame(raw_rows)
    rawdf.to_csv(OUT / "raw_calibration_per_class.csv", index=False, float_format="%.17g")
    macro_raw = rawdf[["brier", "log_loss", "ece"]].mean().to_dict()

    # ================================================================ Step 4: patient-level CV
    folds = patient_folds(pats, valid, raw, N_FOLDS, SEED)
    pa = np.asarray(pats)
    split_info = {"n_folds": N_FOLDS, "seed": SEED, "method": "multi-label iterative stratification over patients (valid positive images per class)",
                  "images_per_fold": [int((folds == k).sum()) for k in range(N_FOLDS)],
                  "patients_per_fold": [int(len(set(pa[folds == k]))) for k in range(N_FOLDS)],
                  "patient_overlap_between_folds": 0,
                  "min_positives_per_class_per_fold": {lab: int(min(((y_all[:, j]) & valid[:, j] & (folds == k)).sum() for k in range(N_FOLDS)))
                                                       for j, lab in enumerate(LABELS)}}
    assert all(len(set(pa[folds == a]) & set(pa[folds == b])) == 0 for a in range(N_FOLDS) for b in range(a + 1, N_FOLDS))
    assert min(split_info["min_positives_per_class_per_fold"].values()) >= 20, split_info["min_positives_per_class_per_fold"]
    (OUT / "cv_split_summary.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    oof = {m: np.full((n, len(LABELS)), np.nan) for m in METHODS}
    fitlog = []
    for j, lab in enumerate(LABELS):
        for k in range(N_FOLDS):
            tr = (folds != k) & valid[:, j]
            te = folds == k
            for m in METHODS:
                spec = fit_calibrator(m, z[tr, j], y_all[tr, j])
                oof[m][te, j] = apply_calibrator(spec, z[te, j])
                fitlog.append({"observation": lab, "fold": k, "method": m, "n_train": int(tr.sum()),
                               "n_train_positive": int(y_all[tr, j].sum()),
                               **{kk: spec.get(kk) for kk in ("T", "a", "b", "n_levels", "n_knots")},
                               "fit_flags": ";".join(spec["flags"]), "pathological": spec["pathological"]})
    fitdf = pd.DataFrame(fitlog)
    fitdf.to_csv(OUT / "cv_fit_log.csv", index=False, float_format="%.17g")
    log.info("cross-validation done: %d calibrator fits", len(fitdf))

    # ================================================================ Step 5: comparison
    cmp_rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        for m in METHODS:
            cmp_rows.append({"observation": lab, "method": m, **class_metrics(oof[m][v, j], y_all[v, j], N_BINS)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUT / "method_comparison_per_class.csv", index=False, float_format="%.17g")
    macro = cmp.groupby("method")[list(METRICS)].mean().loc[list(METHODS)]
    macro.to_csv(OUT / "method_comparison_macro.csv", float_format="%.17g")

    # ================================================================ Step 6-8: guardrails, full-data fits, equivalence, selection
    full_specs: dict[str, dict[str, dict]] = {}
    mism: dict[tuple, int] = {}
    guard = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        full_specs[lab] = {}
        for m in METHODS:
            spec = fit_calibrator(m, z[v, j], y_all[v, j])
            full_specs[lab][m] = spec
            mism[(lab, m)] = decision_mismatches(p[:, j], float(thr[j]), spec)
            cv_flags = sorted({f for s in fitdf[(fitdf.observation == lab) & (fitdf.method == m)]["fit_flags"].dropna() for f in s.split(";") if f})
            guard.append({"observation": lab, "method": m, "cv_fold_flags": ";".join(cv_flags), "cv_pathological_folds": int(
                fitdf[(fitdf.observation == lab) & (fitdf.method == m)].pathological.sum()), "full_fit_flags": ";".join(spec["flags"]),
                          "strictly_increasing": is_strictly_increasing(spec), "decision_mismatches_vs_c4": mism[(lab, m)]})
    gdf = pd.DataFrame(guard)
    gdf.to_csv(OUT / "calibrator_guardrails.csv", index=False)

    tol = SELECTION["tolerance_rel"]
    sel_rows, per_class_choice = [], {}
    for lab in LABELS:
        c = cmp[cmp.observation == lab].set_index("method")
        g = gdf[gdf.observation == lab].set_index("method")
        elig, why = [], {}
        for m in METHODS:
            bad = []
            if g.loc[m, "cv_pathological_folds"] > 0 or g.loc[m, "full_fit_flags"]:
                bad.append("pathological fit")
            if g.loc[m, "decision_mismatches_vs_c4"] > 0:
                bad.append("cannot reproduce C4 decisions")
            (why.__setitem__(m, "; ".join(bad)) if bad else elig.append(m))
        best_ll, best_br = min(c.loc[elig, "log_loss"]), min(c.loc[elig, "brier"])
        cands = [m for m in elig if c.loc[m, "log_loss"] <= best_ll * (1 + tol) and c.loc[m, "brier"] <= best_br * (1 + tol)]
        chosen = min(cands, key=lambda m: COMPLEXITY[m]) if cands else min(elig, key=lambda m: c.loc[m, "log_loss"])
        per_class_choice[lab] = {"chosen": chosen, "best_ll": best_ll, "best_brier": best_br, "elig": elig}
        sel_rows.append({"observation": lab, "eligible": ",".join(elig), "ineligible_reasons": "; ".join(f"{m}: {r}" for m, r in why.items()),
                         "best_log_loss_among_eligible": best_ll, "best_brier_among_eligible": best_br,
                         "per_class_selected": chosen})
    mode, cnt = Counter(v["chosen"] for v in per_class_choice.values()).most_common(1)[0]
    uniform = None
    if cnt >= SELECTION["uniform_min_classes"] and all(mode in per_class_choice[l]["elig"] for l in LABELS):
        c_all = cmp.set_index(["observation", "method"])
        worst = max(max(c_all.loc[(l, mode), "log_loss"] / per_class_choice[l]["best_ll"] - 1,
                        c_all.loc[(l, mode), "brier"] / per_class_choice[l]["best_brier"] - 1) for l in LABELS)
        if worst <= SELECTION["uniform_max_rel_loss"]:
            uniform = mode
        sel_uniform_info = {"dominant_method": mode, "n_classes_selected": cnt, "worst_relative_loss_if_uniform": float(worst)}
    else:
        sel_uniform_info = {"dominant_method": mode, "n_classes_selected": cnt, "worst_relative_loss_if_uniform": None}
    final_choice = {lab: (uniform or per_class_choice[lab]["chosen"]) for lab in LABELS}
    seldf = pd.DataFrame(sel_rows).assign(final_selected=[final_choice[l] for l in LABELS])
    seldf.to_csv(OUT / "method_selection.csv", index=False, float_format="%.17g")
    log.info("selection: uniform=%s; per-class counts=%s; final=%s", uniform, dict(Counter(final_choice.values())), sel_uniform_info)

    # ================================================================ Step 7: final calibrators (full validation fit)
    specs = {lab: dict(full_specs[lab][final_choice[lab]], class_name=lab, fitted_on="validation (full), after OOF method selection",
                       n_fit=int(valid[:, j].sum()), n_fit_positive=int(y_all[valid[:, j], j].sum()), raw_c4_threshold=float(thr[j]),
                       calibrated_equivalent_threshold=calibrated_threshold(full_specs[lab][final_choice[lab]], float(thr[j])))
             for j, lab in enumerate(LABELS)}
    meta = {"calibration_version": "c5-v1", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fitting_split": "validation", "checkpoint": ck["path"], "checkpoint_sha256": ck["sha256_recomputed"],
            "loss": "sqrt_weighted_bce", "score_logit_note": "calibrators take logit(raw sigmoid score)",
            "selection_rule": SELECTION, "uniform_method": uniform, "cv": {"folds": N_FOLDS, "seed": SEED},
            "c4_policy": "results/classification/experiments/c4_operating_policy/final_operating_policy.json",
            "no_finding_rule": "unchanged from C4 (suppress if any of 12 pathology findings positive; Support Devices does not suppress)",
            "locked_test_split_used": False, "code_commit": info["environment"]["git"]["commit"]}
    save_calibrators(OUT / "final_calibrators.json", specs, meta)

    # ================================================================ Step 8-9: equivalence + per-image output
    cal_full = np.zeros_like(p)
    cal_thr = np.zeros(len(LABELS))
    for j, lab in enumerate(LABELS):
        cal_full[:, j] = apply_calibrator(specs[lab], z[:, j])
        cal_thr[j] = specs[lab]["calibrated_equivalent_threshold"]
    cal_pred = cal_full >= cal_thr[None, :]
    mism_rows = [{"observation": lab, "raw_f1_threshold": float(thr[j]), "calibrated_equivalent_threshold": float(cal_thr[j]),
                  "method": final_choice[lab], "strictly_increasing": is_strictly_increasing(specs[lab]),
                  "decision_mismatches_all_images": int((cal_pred[:, j] != raw_pred[:, j]).sum()),
                  "decision_mismatches_valid_label_images": int(((cal_pred[:, j] != raw_pred[:, j]) & valid[:, j]).sum())}
                 for j, lab in enumerate(LABELS)]
    mmdf = pd.DataFrame(mism_rows)
    mmdf.to_csv(OUT / "threshold_mapping.csv", index=False, float_format="%.17g")
    assert mmdf["decision_mismatches_all_images"].sum() == 0, mmdf
    assert np.array_equal(apply_no_finding_rule(cal_pred), final_pred)         # final policy reproduced from calibrated side

    cols = {"image_id": pred["paths"], "patient_id": pats}
    for j, lab in enumerate(LABELS):
        cols[f"{lab}_label"] = raw[:, j]
        cols[f"{lab}_score_raw"] = p[:, j]
        cols[f"{lab}_prob_calibrated"] = cal_full[:, j]
        cols[f"{lab}_prob_calibrated_oof"] = oof[final_choice[lab]][:, j]
        cols[f"{lab}_threshold_raw"] = np.full(n, thr[j])
        cols[f"{lab}_threshold_calibrated"] = np.full(n, cal_thr[j])
        cols[f"{lab}_pred_raw"] = raw_pred[:, j].astype(int)
        cols[f"{lab}_pred_final"] = final_pred[:, j].astype(int)
    pi_path = OUT / "validation_predictions_calibrated.csv.gz"
    pd.DataFrame(cols).to_csv(pi_path, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    back = pd.read_csv(pi_path, float_precision="round_trip")
    assert np.array_equal(back[[f"{l}_score_raw" for l in LABELS]].to_numpy(), p)
    assert np.array_equal(back[[f"{l}_prob_calibrated" for l in LABELS]].to_numpy(), cal_full)
    assert np.array_equal(back[[f"{l}_pred_final" for l in LABELS]].to_numpy().astype(bool), final_pred)

    # ================================================================ selected-method OOF metrics, ranking sanity, reliability
    sel_rows2, rank_rows = [], []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        m = final_choice[lab]
        o = cmp[(cmp.observation == lab) & (cmp.method == m)].iloc[0]
        r = rawdf[rawdf.observation == lab].iloc[0]
        sel_rows2.append({"observation": lab, "prevalence": r.prevalence, "method": m,
                          **{f"raw_{k}": r[k] for k in METRICS}, **{f"calibrated_{k}": o[k] for k in METRICS},
                          "raw_mean_score": r.mean_predicted, "calibrated_oof_mean": o.mean_predicted,
                          "raw_cal_slope": r.cal_slope, "calibrated_cal_slope": o.cal_slope,
                          "raw_cal_intercept": r.cal_intercept_slope1, "calibrated_cal_intercept": o.cal_intercept_slope1})
        yv = y_all[v, j]
        rank_rows.append({"observation": lab, "method": m, "raw_auroc": roc_auc_score(yv, p[v, j]), "calibrated_final_auroc": roc_auc_score(yv, cal_full[v, j]),
                          "calibrated_oof_auroc": roc_auc_score(yv, oof[m][v, j]), "raw_auprc": average_precision_score(yv, p[v, j]),
                          "calibrated_final_auprc": average_precision_score(yv, cal_full[v, j]), "calibrated_oof_auprc": average_precision_score(yv, oof[m][v, j]),
                          "n_unique_raw": int(len(np.unique(p[v, j]))), "n_unique_calibrated_final": int(len(np.unique(cal_full[v, j])))})
    seldf2 = pd.DataFrame(sel_rows2)
    seldf2.to_csv(OUT / "raw_vs_selected_per_class.csv", index=False, float_format="%.17g")
    pd.DataFrame(rank_rows).to_csv(OUT / "ranking_sanity.csv", index=False, float_format="%.17g")
    prev = dict(zip(rawdf.observation, rawdf.prevalence))
    reps = representative(prev)
    rel_rows = []
    for role, lab in reps.items():
        j, v = list(LABELS).index(lab), valid[:, list(LABELS).index(lab)]
        for series, arr in (("raw", p[:, j]), ("calibrated_oof", oof[final_choice[lab]][:, j])):
            for i, b in enumerate(adaptive_bins(arr[v], y_all[v, j], N_BINS)):
                rel_rows.append({"role": role, "observation": lab, "series": series, "bin": i + 1, **b})
    pd.DataFrame(rel_rows).to_csv(OUT / "reliability_bins.csv", index=False, float_format="%.17g")

    macro_sel = seldf2[[f"raw_{k}" for k in METRICS]].mean().to_dict() | seldf2[[f"calibrated_{k}" for k in METRICS]].mean().to_dict()
    results = {"c4_reproduced": rep["reproduced"], "n_images": n, "n_patients": rep["n_patients"], "cv": split_info,
               "logit_reconstruction": lg, "macro_raw": macro_raw, "macro_by_method": macro.to_dict(orient="index"),
               "macro_selected": macro_sel, "final_choice": final_choice, "uniform_method": uniform, "uniform_check": sel_uniform_info,
               "selection_rule": SELECTION, "representative": reps, "total_threshold_decision_mismatches": int(mmdf["decision_mismatches_all_images"].sum()),
               "guardrail_flags": {f"{r.observation}/{r.method}": (r.cv_fold_flags, r.full_fit_flags) for r in gdf.itertuples() if r.cv_fold_flags or r.full_fit_flags},
               "locked_test_split_used": False, "checkpoint": ck}
    (OUT / "c5_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(macro.round(5).to_string()); print("raw macro:", {k: round(v, 5) for k, v in macro_raw.items()})
    print(seldf[["observation", "eligible", "per_class_selected", "final_selected"]].to_string(index=False))
    print(json.dumps(sel_uniform_info), "| uniform:", uniform, "| mismatches:", results["total_threshold_decision_mismatches"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
