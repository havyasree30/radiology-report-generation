"""C4 analysis: compare four operating policies on the C2-B validation scores (no inference, no bootstrap).

Policies: (1) 0.50  (2) Youden J  (3) F1-optimal  (4) F1-optimal + No Finding consistency rule.
Validation only; the locked test split is never opened. This script does NOT select the final
policy; scripts.c4_finalize does that after the numbers have been inspected.

    .venv\\Scripts\\python.exe -m scripts.c4_operating_policy
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from scripts.compare_losses import RARE
from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS
from src.classification.operating_points import apply_thresholds, load_threshold_file, metrics_from_counts
from src.classification.operating_policy import apply_no_finding_rule
from src.classification.serialization import load_predictions_npz
from src.utils.audit import setup_logging
from src.utils.config import PROJECT_ROOT

log = logging.getLogger("c4")
EXPD = PROJECT_ROOT / "results/classification/experiments"
C3 = EXPD / "c3_threshold_optimization"
OUT = EXPD / "c4_operating_policy"
NF = LABELS.index(NO_FINDING)
IDX12 = [LABELS.index(l) for l in PATHOLOGY_LABELS]          # pathology (rule + contradiction definition)
IDX13 = [i for i in range(len(LABELS)) if i != NF]            # all non-No-Finding observations
POLICIES = ("thr_050", "youden", "f1", "f1_nf")
POL_NAME = {"thr_050": "Threshold 0.50", "youden": "Youden J", "f1": "F1-optimal", "f1_nf": "F1-optimal + No Finding rule"}
C3_TABLE = {"thr_050": "baseline_050_table.csv", "youden": "youden_table.csv", "f1": "f1_optimal_table.csv"}
METRICS = ("precision", "recall", "specificity", "f1", "balanced_accuracy")


def class_table(pred: np.ndarray, raw: np.ndarray, valid: np.ndarray, thr: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        y, pr = raw[v, j] == 1, pred[v, j]
        c = {"tp": int((pr & y).sum()), "fp": int((pr & ~y).sum()), "tn": int((~pr & ~y).sum()), "fn": int((~pr & y).sum())}
        m = metrics_from_counts(**c)
        rows.append({"observation": lab, "threshold": float(thr[j]), "prevalence": float(y.mean()), **c,
                     **{k: m[k] for k in METRICS}, "fp_per_tp": c["fp"] / c["tp"] if c["tp"] else float("nan")})
    return pd.DataFrame(rows)


def macro(t: pd.DataFrame, idx: list[int]) -> dict:
    sub = t.iloc[idx]
    return {f"macro_{m}": float(sub[m].mean()) for m in METRICS}


def count_stats(n_pos: np.ndarray) -> dict:
    return {"mean": float(n_pos.mean()), "median": float(np.median(n_pos)), "p25": float(np.percentile(n_pos, 25)),
            "p75": float(np.percentile(n_pos, 75)), "p95": float(np.percentile(n_pos, 95)), "max": int(n_pos.max()),
            "pct_0": float(100 * (n_pos == 0).mean()), "pct_1": float(100 * (n_pos == 1).mean()),
            "pct_2_3": float(100 * ((n_pos >= 2) & (n_pos <= 3)).mean()),
            "pct_4_5": float(100 * ((n_pos >= 4) & (n_pos <= 5)).mean()), "pct_gt5": float(100 * (n_pos > 5).mean())}


def main() -> int:
    setup_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures" / "source_data").mkdir(parents=True, exist_ok=True)
    c3_info = json.loads((C3 / "input_verification.json").read_text(encoding="utf-8"))
    pred = load_predictions_npz(EXPD / "c2b_sqrt_weighted_bce/evaluation/fp32_strict/validation_predictions.npz")
    p, raw, valid = pred["probs"], pred["raw_labels"], pred["valid"]
    n = len(p)
    thr = {"thr_050": np.full(14, 0.5), "youden": load_threshold_file(C3 / "thresholds_youden.json"),
           "f1": load_threshold_file(C3 / "thresholds_f1.json")}
    thr["f1_nf"] = thr["f1"]
    raw_pred = {k: apply_thresholds(p, thr[k]) for k in ("thr_050", "youden", "f1")}
    pr = {**raw_pred, "f1_nf": apply_no_finding_rule(raw_pred["f1"])}
    pr_strict = apply_no_finding_rule(raw_pred["f1"], abnormal_labels=[l for l in LABELS if l != NO_FINDING])
    tabs = {k: class_table(pr[k], raw, valid, thr[k]) for k in POLICIES}

    # ---------------------------------------------------------------- Step 1: reproduce C3
    rep = {"policies": {}, "reproduced": True}
    for k, fname in C3_TABLE.items():
        c3t = pd.read_csv(C3 / fname, float_precision="round_trip").set_index("observation").loc[list(LABELS)]
        mine = tabs[k].set_index("observation")
        counts_equal = all((mine[c].to_numpy() == c3t[c].to_numpy()).all() for c in ("tp", "fp", "tn", "fn"))
        fl = max(float(np.abs(mine[m].to_numpy() - c3t[m].to_numpy()).max()) for m in METRICS)
        rep["policies"][k] = {"counts_identical": bool(counts_equal), "max_abs_diff_metrics": fl}
        rep["reproduced"] &= bool(counts_equal and fl < 1e-12)
    c3m = pd.read_csv(C3 / "macro_operating_summary.csv", float_precision="round_trip").set_index("policy")
    for k, c3k in (("thr_050", "baseline_050"), ("youden", "youden"), ("f1", "f1_optimal")):
        d = max(abs(macro(tabs[k], list(range(14)))[f"macro_{m}"] - c3m.loc[c3k, f"macro_{m}"]) for m in METRICS)
        rep["policies"][k]["macro_max_abs_diff"] = float(d)
        rep["reproduced"] &= bool(d < 1e-12)
    ds = pd.read_csv(C3 / "validation_predictions_youden.csv.gz", float_precision="round_trip")
    rep["c3_per_image_file"] = {"scores_bit_identical": bool(np.array_equal(ds[[f"{l}_score" for l in LABELS]].to_numpy(), p)),
                                "youden_preds_identical": bool(np.array_equal(ds[[f"{l}_pred" for l in LABELS]].to_numpy().astype(bool), pr["youden"])),
                                "image_ids_identical": ds["image_id"].tolist() == pred["paths"]}
    rep["reproduced"] &= all(rep["c3_per_image_file"].values())
    rep["n_images"], rep["n_patients"] = n, len(set(pred["patient_ids"]))
    (OUT / "c3_reproduction.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["reproduced"]:
        raise AssertionError(f"C3 results did NOT reproduce: {json.dumps(rep, indent=1)}")
    log.info("C3 reproduced for 0.50 / Youden / F1-optimal (counts identical, metrics < 1e-12)")

    # ---------------------------------------------------------------- Step 2: full F1-optimal table
    f1t = tabs["f1"][["observation", "threshold", "prevalence", "tp", "tn", "fp", "fn", "precision", "recall", "specificity", "f1", "balanced_accuracy"]]
    f1t.rename(columns={"threshold": "f1_optimal_threshold"}).to_csv(OUT / "f1_operating_table.csv", index=False, float_format="%.17g")
    long = pd.concat([t.assign(policy=k) for k, t in tabs.items()], ignore_index=True)
    long.to_csv(OUT / "policy_per_class_long.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- Step 3: finding counts per image
    gt_pos = raw == 1
    count_rows, dist = [], {}
    for definition, idx in (("13 non-No-Finding observations (primary; what downstream queries list)", IDX13),
                            ("12 pathology observations (excl. Support Devices)", IDX12)):
        series = {"ground_truth": gt_pos[:, idx].sum(1), **{k: raw_pred[k][:, idx].sum(1) for k in ("thr_050", "youden", "f1")}}
        for k, c in series.items():
            count_rows.append({"definition": definition, "policy": k, **count_stats(c)})
        if idx is IDX13:
            kmax = 10
            for k, c in series.items():
                dist[k] = [float((np.minimum(c, kmax) == i).mean() * 100) for i in range(kmax + 1)]
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT / "finding_counts_per_image.csv", index=False, float_format="%.17g")
    pd.DataFrame({"n_findings": [str(i) if i < 10 else "10+" for i in range(11)], **{f"pct_images_{k}": v for k, v in dist.items()}}
                 ).to_csv(OUT / "figures/source_data/fig3_finding_count_distribution.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- Step 4: rare-class false-alarm
    rare = long[long["observation"].isin(RARE)][["observation", "policy", "threshold", "prevalence", "tp", "fp", "fp_per_tp",
                                                  "precision", "recall", "specificity", "f1"]]
    rare = rare[rare["policy"].isin(["thr_050", "youden", "f1"])]
    rare.to_csv(OUT / "rare_class_false_alarm.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- Steps 5-6: No Finding rule
    def contradiction(pred_nf: np.ndarray, abnormal_pred: np.ndarray, idx: list[int]) -> tuple[int, float, float]:
        c = pred_nf & abnormal_pred[:, idx].any(axis=1)
        return int(c.sum()), float(100 * c.mean()), float(100 * c.sum() / max(pred_nf.sum(), 1))
    nf_rows = []
    cases = {"thr_050": (pr["thr_050"], pr["thr_050"]), "youden": (pr["youden"], pr["youden"]),
             "f1": (pr["f1"], pr["f1"]), "f1_nf": (pr["f1_nf"], pr["f1_nf"]),
             "f1_nf_strict": (pr_strict, pr_strict)}
    for k, (pnf, pab) in cases.items():
        t = class_table(pnf, raw, valid, thr["f1"])  # only the No Finding row is used
        r = t.iloc[NF]
        n12, p12, s12 = contradiction(pnf[:, NF], pab, IDX12)
        n13, p13, s13 = contradiction(pnf[:, NF], pab, IDX13)
        nf_rows.append({"policy": k, "no_finding_threshold": float(thr["f1"][NF] if k != "thr_050" else 0.5) if k not in ("youden",) else float(thr["youden"][NF]),
                        "tp": r.tp, "fp": r.fp, "tn": r.tn, "fn": r.fn, "precision": r.precision, "recall": r.recall,
                        "specificity": r.specificity, "f1": r.f1, "balanced_accuracy": r.balanced_accuracy,
                        "pct_predicted_no_finding": float(100 * pnf[:, NF].mean()),
                        "contradiction_n_pathology12": n12, "contradiction_pct_of_all_images_pathology12": p12,
                        "contradiction_pct_of_no_finding_positive_pathology12": s12,
                        "contradiction_n_incl_support_devices": n13, "contradiction_pct_of_all_images_incl_support_devices": p13})
    nft = pd.DataFrame(nf_rows)
    nft.to_csv(OUT / "no_finding_before_after.csv", index=False, float_format="%.17g")
    gt_nf = gt_pos[:, NF]
    gt_contra = {"gt_no_finding_positive": int(gt_nf.sum()), "gt_contradiction_pathology12": int((gt_nf & gt_pos[:, IDX12].any(1)).sum()),
                 "gt_no_finding_with_support_devices": int((gt_nf & gt_pos[:, LABELS.index("Support Devices")]).sum())}
    changed = int((pr["f1_nf"][:, NF] != pr["f1"][:, NF]).sum())
    other_cols_same = bool(np.array_equal(pr["f1_nf"][:, IDX13], pr["f1"][:, IDX13]))

    # ---------------------------------------------------------------- Step 7: policy comparison
    comp = []
    for k in POLICIES:
        for scope, idx in (("14 classes", list(range(14))), ("13 abnormal findings (excl. No Finding)", IDX13)):
            row = {"policy": k, "policy_name": POL_NAME[k], "scope": scope, **macro(tabs[k], idx)}
            ab = pr[k][:, IDX13].sum(1)
            row["mean_predicted_abnormal_findings_per_image"] = float(ab.mean())
            row["mean_predicted_pathology_findings_per_image"] = float(pr[k][:, IDX12].sum(1).mean())
            c = nft.set_index("policy").loc[k]
            row["no_finding_contradiction_pct_of_images"] = float(c.contradiction_pct_of_all_images_pathology12)
            row["no_finding_contradiction_pct_of_no_finding_positive"] = float(c.contradiction_pct_of_no_finding_positive_pathology12)
            row["no_finding_contradiction_pct_incl_support_devices"] = float(c.contradiction_pct_of_all_images_incl_support_devices)
            comp.append(row)
    compdf = pd.DataFrame(comp)
    compdf.to_csv(OUT / "policy_comparison.csv", index=False, float_format="%.17g")

    results = {"c3_reproduced": rep["reproduced"], "n_images": n, "n_patients": rep["n_patients"],
               "no_finding_ground_truth": gt_contra, "rule": {"images_changed": changed, "abnormal_columns_identical": other_cols_same},
               "checkpoint": c3_info["checkpoint"], "locked_test_split_used": False,
               "definitions": {"abnormal_for_rule_and_contradiction": list(PATHOLOGY_LABELS),
                               "abnormal_for_counts_and_macro13": [LABELS[i] for i in IDX13]}}
    (OUT / "c4_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(compdf[["policy", "scope", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "macro_balanced_accuracy",
                  "mean_predicted_abnormal_findings_per_image", "no_finding_contradiction_pct_of_images"]].round(4).to_string(index=False))
    print(nft.drop(columns=["balanced_accuracy"]).round(4).to_string(index=False))
    print(counts.round(2).to_string(index=False))
    print(rare.round(3).to_string(index=False))
    print(json.dumps(results["no_finding_ground_truth"]), results["rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
