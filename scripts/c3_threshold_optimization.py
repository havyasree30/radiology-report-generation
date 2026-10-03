"""C3: per-class threshold optimisation on the C2-B (sqrt-weighted BCE) validation scores.

No model is trained and no inference is run: the canonical C2-B validation predictions are
reused after verification. Thresholds are selected on the VALIDATION split only. The locked
test split is never opened.

    .venv\\Scripts\\python.exe -m scripts.c3_threshold_optimization

Outputs: results/classification/experiments/c3_threshold_optimization/
"""

from __future__ import annotations

import hashlib
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from scripts.compare_losses import COMMON, RARE  # groups fixed in C2, not redefined here
from src.analysis.plotting import INK_2, MUTED, SERIES, apply_style, save_figure, subtitle
from src.classification import thresholds as c1_thresholds
from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS, assert_matches
from src.classification.operating_points import (apply_thresholds, candidate_table, check_scores, f1_threshold,
                                                 load_threshold_file, macro_summary, operating_point,
                                                 patient_bootstrap_youden, per_class_table, save_threshold_file,
                                                 youden_threshold)
from src.classification.serialization import load_predictions_npz
from src.utils.audit import setup_logging
from src.utils.config import PROJECT_ROOT
from src.utils.reproducibility import environment_snapshot

log = logging.getLogger("c3")
EXPD = PROJECT_ROOT / "results/classification/experiments"
C2B = EXPD / "c2b_sqrt_weighted_bce"
OUT = EXPD / "c3_threshold_optimization"
FIG = OUT / "figures"
N_BOOT, BOOT_SEED = 1000, 42
# Declared BEFORE any C3 result was inspected:
SANITY = {"low": 0.10, "high": 0.70, "near_zero": 0.02, "near_one": 0.98}
UNSTABLE = {"max_interval_width": 0.20, "max_width_over_median": 1.0}
EXPECTED_C2B = {"macro_auroc": 0.8011, "micro_auroc": 0.8922, "macro_auprc": 0.4235, "micro_auprc": 0.6674}
POLICIES = ("baseline_050", "youden", "f1_optimal")
POL_LABEL = {"baseline_050": "0.50", "youden": "Youden J", "f1_optimal": "F1-optimal"}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ======================================================================= Step 0
def verify_inputs() -> dict:
    meta = json.loads((C2B / "checkpoints/best.meta.json").read_text(encoding="utf-8"))
    sha_rec = json.loads((C2B / "checkpoint_sha256.json").read_text(encoding="utf-8"))["best.pt"]
    ckpt = C2B / "checkpoints/best.pt"
    sha_now = sha256(ckpt)
    summ = json.loads((C2B / "evaluation/fp32_strict/summary_metrics.json").read_text(encoding="utf-8"))
    tsum = json.loads((C2B / "training_summary.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((C2B / "config.yaml").read_text(encoding="utf-8"))
    c2_cfg = yaml.safe_load((PROJECT_ROOT / "configs/classifier/c2/c2b_sqrt_weighted_bce.yaml").read_text(encoding="utf-8"))
    pred = load_predictions_npz(C2B / "evaluation/fp32_strict/validation_predictions.npz")
    assert_matches(list(np.load(C2B / "evaluation/fp32_strict/validation_predictions.npz")["labels"]))

    man = pd.read_csv(PROJECT_ROOT / cfg["data"]["manifest"])
    excl = pd.read_csv(PROJECT_ROOT / cfg["data"]["exclusions"])
    drop = set(excl.loc[excl["recommendation"] == "exclude", "Path"])
    val = man[(man["split"] == "val") & (man["Frontal/Lateral"] == "Frontal") & ~man["Path"].isin(drop)]
    paths_ok = pred["paths"] == val["Path"].tolist()
    pat_ok = pred["patient_ids"] == val["patient_id"].tolist()
    train_pat = set(man.loc[man["split"] == "train", "patient_id"])
    test_pat = set(man.loc[man["split"] == "test", "patient_id"])
    src = pd.read_csv(PROJECT_ROOT / "data/train.csv").set_index("Path")
    raw_src = src.loc[pred["paths"], list(LABELS)].to_numpy(dtype=float)
    labels_ok = bool(np.array_equal(np.nan_to_num(raw_src, nan=9), np.nan_to_num(pred["raw_labels"], nan=9)))
    # canonical policy: valid <=> label != -1 (uncertain masked); blank (NaN) is scored as negative
    valid_ok = bool(np.array_equal(pred["valid"], pred["raw_labels"] != -1.0))
    check_scores(pred["probs"], "C2-B validation scores")
    # same rows / masks as the other C2 arms
    other = load_predictions_npz(EXPD / "c2a_bce/evaluation/fp32_strict/validation_predictions.npz")
    same_rows = other["paths"] == pred["paths"] and np.array_equal(other["valid"], pred["valid"])

    info = {
        "checkpoint": {"path": ckpt.relative_to(PROJECT_ROOT).as_posix(), "sha256_recomputed": sha_now,
                       "sha256_recorded": sha_rec["sha256"], "sha256_match": sha_now == sha_rec["sha256"],
                       "size_bytes": ckpt.stat().st_size, "git_commit_in_checkpoint": meta["git_commit"],
                       "architecture": meta["architecture"], "num_labels": meta["num_labels"],
                       "training_loss": meta["loss"], "selected_epoch": meta["epoch"],
                       "selection_metric": meta["val_metric_name"], "selection_metric_value": meta["val_metric"],
                       "training_summary_best_epoch": tsum["best_epoch"]},
        "label_order": list(LABELS),
        "label_policy": meta["label_policy"],
        "preprocessing": meta["preprocessing"],
        "evaluation_precision_policy": summ["inference_policy"],
        "evaluation_determinism_bit_identical": summ["determinism"]["bit_identical"],
        "validation": {"n_images": len(pred["paths"]), "n_patients": len(set(pred["patient_ids"])),
                       "paths_equal_manifest_val_frontal_minus_exclusions": paths_ok,
                       "patient_ids_equal_manifest": pat_ok,
                       "train_val_patient_overlap": len(train_pat & set(pred["patient_ids"])),
                       "test_patients_in_validation": len(test_pat & set(pred["patient_ids"])),
                       "labels_reread_from_source_csv_identical": labels_ok,
                       "valid_mask_equals_label_not_uncertain": valid_ok,
                       "rows_and_masks_identical_to_c2a": same_rows,
                       "n_uncertain_excluded_entries": int((~pred["valid"]).sum())},
        "config_matches_c2_base_config": cfg == c2_cfg,
        "scores": {"min": float(pred["probs"].min()), "max": float(pred["probs"].max()), "all_finite_in_01": True},
        "locked_test_split_used": False,
    }
    ok = (info["checkpoint"]["sha256_match"] and info["checkpoint"]["selected_epoch"] == tsum["best_epoch"]
          and paths_ok and pat_ok and labels_ok and valid_ok and same_rows and info["validation"]["train_val_patient_overlap"] == 0
          and info["validation"]["test_patients_in_validation"] == 0 and info["config_matches_c2_base_config"]
          and info["evaluation_precision_policy"] == "fp32_strict")
    info["all_checks_passed"] = bool(ok)
    if not ok:
        raise AssertionError(f"input verification FAILED: {json.dumps(clean(info), indent=1)}")
    return info, pred


# ======================================================================= Step 1
def reproduce_metrics(pred: dict) -> dict:
    p, raw, v = pred["probs"], pred["raw_labels"], pred["valid"]
    au, ap = [], []
    for j in range(len(LABELS)):
        y = raw[v[:, j], j] == 1
        au.append(roc_auc_score(y, p[v[:, j], j]))
        ap.append(average_precision_score(y, p[v[:, j], j]))
    yf, pf = (raw == 1)[v], p[v]
    rec = {"macro_auroc": float(np.mean(au)), "micro_auroc": float(roc_auc_score(yf, pf)),
           "macro_auprc": float(np.mean(ap)), "micro_auprc": float(average_precision_score(yf, pf))}
    saved = json.loads((C2B / "evaluation/fp32_strict/summary_metrics.json").read_text(encoding="utf-8"))
    pcm = pd.read_csv(C2B / "evaluation/fp32_strict/per_class_metrics.csv", float_precision="round_trip").set_index("observation")
    out = {"recomputed": rec, "saved": {k: saved[k] for k in rec},
           "abs_diff": {k: abs(rec[k] - saved[k]) for k in rec},
           "per_class": {lab: {"auroc": au[j], "auprc": ap[j],
                               "saved_auroc": float(pcm.loc[lab, "auroc"]), "saved_auprc": float(pcm.loc[lab, "auprc"])}
                         for j, lab in enumerate(LABELS)},
           "task_expected_approx": EXPECTED_C2B,
           "abs_diff_vs_task_expected": {k: abs(rec[k] - EXPECTED_C2B[k]) for k in rec}}
    out["max_per_class_abs_diff"] = max(max(abs(d["auroc"] - d["saved_auroc"]), abs(d["auprc"] - d["saved_auprc"]))
                                        for d in out["per_class"].values())
    out["reproduced"] = bool(max(out["abs_diff"].values()) < 1e-12 and out["max_per_class_abs_diff"] < 1e-12
                             and max(out["abs_diff_vs_task_expected"].values()) < 5e-4)
    if not out["reproduced"]:
        raise AssertionError(f"C2-B metrics did NOT reproduce: {json.dumps(clean(out), indent=1)}")
    return out, au, ap


# ======================================================================= Steps 2-6
def build_tables(pred: dict) -> dict:
    p, raw, v = pred["probs"], pred["raw_labels"], pred["valid"]
    thr = {"baseline_050": [0.5] * len(LABELS), "youden": [], "f1_optimal": []}
    sel = {"youden": [], "f1_optimal": []}
    for j in range(len(LABELS)):
        y, s = raw[v[:, j], j] == 1, p[v[:, j], j]
        for pol, fn in (("youden", youden_threshold), ("f1_optimal", f1_threshold)):
            r = fn(y, s)
            sel[pol].append(r)
            thr[pol].append(r["threshold"])
    tables = {pol: per_class_table(p, raw, v, thr[pol]) for pol in POLICIES}
    return {"thr": thr, "sel": sel, "tables": tables, "macro": {pol: macro_summary(tables[pol]) for pol in POLICIES}}


def long_frame(tables: dict) -> pd.DataFrame:
    rows = []
    for pol, tab in tables.items():
        for r in tab:
            rows.append({"policy": pol, **r})
    return pd.DataFrame(rows)


def wide_comparison(tables: dict) -> pd.DataFrame:
    b = {r["observation"]: r for r in tables["baseline_050"]}
    y = {r["observation"]: r for r in tables["youden"]}
    f = {r["observation"]: r for r in tables["f1_optimal"]}
    rows = []
    for lab in LABELS:
        r = {"observation": lab, "prevalence": b[lab]["prevalence"], "positives": b[lab]["positives"],
             "negatives": b[lab]["negatives"], "thr_050": 0.5, "thr_youden": y[lab]["threshold"], "thr_f1": f[lab]["threshold"]}
        for m in ("recall", "precision", "specificity", "f1", "balanced_accuracy"):
            for tag, src in (("050", b), ("youden", y), ("f1opt", f)):
                r[f"{m}_{tag}"] = src[lab].get(m)
            r[f"delta_{m}_050_to_youden"] = (y[lab].get(m) or np.nan) - (b[lab].get(m) or np.nan) \
                if y[lab].get(m) is not None and b[lab].get(m) is not None else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ======================================================================= Step 8
def sanity_table(pred: dict, tables: dict, au: list) -> pd.DataFrame:
    p, raw, v = pred["probs"], pred["raw_labels"], pred["valid"]
    rows = []
    for j, lab in enumerate(LABELS):
        y, s = raw[v[:, j], j] == 1, p[v[:, j], j]
        t = tables["youden"][j]["threshold"]
        flags = [k for k, cond in (("low", t < SANITY["low"]), ("high", t > SANITY["high"]),
                                   ("near_zero", t < SANITY["near_zero"]), ("near_one", t > SANITY["near_one"])) if cond]
        pos, neg = s[y], s[~y]
        hp, hn = np.histogram(pos, bins=np.linspace(0, 1, 51))[0], np.histogram(neg, bins=np.linspace(0, 1, 51))[0]
        rows.append({"observation": lab, "youden_threshold": t, "flags": ";".join(flags) or "none",
                     "prevalence": float(y.mean()), "auroc": au[j], "median_score_positives": float(np.median(pos)),
                     "median_score_negatives": float(np.median(neg)), "score_p05": float(np.quantile(s, 0.05)),
                     "score_p95": float(np.quantile(s, 0.95)),
                     "threshold_percentile_of_negatives": float((neg < t).mean() * 100),
                     "histogram_overlap_coefficient": float(np.minimum(hp / hp.sum(), hn / hn.sum()).sum()),
                     "mean_score_positives_vs_prevalence": float(pos.mean() - y.mean())})
    return pd.DataFrame(rows)


# ======================================================================= Step 9
def no_finding_analysis(pred: dict, thr_y: np.ndarray) -> dict:
    p, raw = pred["probs"], pred["raw_labels"]
    n = len(p)
    nf = LABELS.index(NO_FINDING)
    ab = [LABELS.index(l) for l in PATHOLOGY_LABELS]
    ab_all = [i for i in range(len(LABELS)) if i != nf]
    out = {"n_images": n, "abnormal_definition": "any of the 12 pathology observations (all except No Finding and Support Devices)",
           "pathology_labels": list(PATHOLOGY_LABELS)}
    for tag, thr in (("youden", thr_y), ("baseline_050", np.full(len(LABELS), 0.5))):
        pr = apply_thresholds(p, thr)
        nf_pos, ab_pos = pr[:, nf], pr[:, ab].any(axis=1)
        both = nf_pos & ab_pos
        co = {LABELS[i]: int((pr[:, i] & nf_pos).sum()) for i in ab}
        out[tag] = {"no_finding_positive": int(nf_pos.sum()), "no_finding_positive_pct": float(100 * nf_pos.mean()),
                    "abnormal_positive": int(ab_pos.sum()), "abnormal_positive_pct": float(100 * ab_pos.mean()),
                    "both": int(both.sum()), "both_pct_of_all_images": float(100 * both.mean()),
                    "contradiction_rate_pct_of_no_finding_positive": float(100 * both.sum() / max(nf_pos.sum(), 1)),
                    "no_finding_positive_and_no_abnormal": int((nf_pos & ~ab_pos).sum()),
                    "neither_no_finding_nor_abnormal": int((~nf_pos & ~ab_pos).sum()),
                    "co_occurrence_with_predicted_no_finding": co,
                    "co_occurrence_pct_of_no_finding_positive": {k: float(100 * c / max(nf_pos.sum(), 1)) for k, c in co.items()},
                    "contradiction_if_support_devices_counted_as_abnormal": int((nf_pos & pr[:, ab_all].any(axis=1)).sum()),
                    "mean_number_of_abnormal_positives_in_contradictions": float(pr[both][:, ab].sum(axis=1).mean()) if both.any() else None}
    gt = raw[:, nf] == 1
    out["ground_truth"] = {"no_finding_positive": int(gt.sum()), "no_finding_positive_pct": float(100 * gt.mean()),
                           "gt_no_finding_with_gt_abnormal": int((gt & (raw[:, ab] == 1).any(axis=1)).sum())}
    return out


# ======================================================================= Step 11
def bootstrap_summary(pred: dict, thr_y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    draws = patient_bootstrap_youden(pred["probs"], pred["raw_labels"], pred["valid"], pred["patient_ids"],
                                     N_BOOT, BOOT_SEED)
    rows = []
    for j, lab in enumerate(LABELS):
        d = draws[:, j]
        ok = d[np.isfinite(d)]
        q = np.quantile(ok, [0.025, 0.25, 0.5, 0.75, 0.975])
        width = q[4] - q[0]
        unstable = bool(width > UNSTABLE["max_interval_width"] or width > UNSTABLE["max_width_over_median"] * q[2])
        rows.append({"observation": lab, "original_threshold": float(thr_y[j]), "bootstrap_median": float(q[2]),
                     "p2_5": float(q[0]), "p97_5": float(q[4]), "q1": float(q[1]), "q3": float(q[3]),
                     "ci95_width": float(width), "iqr": float(q[3] - q[1]), "ci95_width_over_median": float(width / q[2]),
                     "original_inside_ci95": bool(q[0] <= thr_y[j] <= q[4]), "n_resamples": N_BOOT,
                     "n_undefined_resamples": int(np.isnan(d).sum()), "unstable_flag": unstable})
    return pd.DataFrame(rows), draws


# ======================================================================= figures
def representative_classes(prev: dict) -> dict:
    order = sorted(LABELS, key=lambda l: (prev[l], list(LABELS).index(l)))
    return {"rare (lowest prevalence)": order[0], "medium (median-rank prevalence)": order[len(order) // 2],
            "common (highest prevalence)": order[-1]}


def paired_dot_figure(wide: pd.DataFrame, metric: str, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(LABELS))
    a, b = wide[f"{metric}_050"].to_numpy(float), wide[f"{metric}_youden"].to_numpy(float)
    for i in y:
        ax.plot([a[i], b[i]], [i, i], color="#c3c2b7", lw=2, zorder=1)
    ax.scatter(a, y, s=55, color=SERIES[0], label="threshold 0.50", zorder=3, edgecolor="white", linewidth=1.2)
    ax.scatter(b, y, s=55, color=SERIES[1], label="Youden J threshold", zorder=3, edgecolor="white", linewidth=1.2)
    ax.set_yticks(y, wide["observation"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title, pad=22)
    subtitle(ax, "C2-B validation scores, per-class thresholds; 28,671 images (uncertain labels excluded)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save_figure(fig, fname, FIG)


def main() -> int:
    setup_logging()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)

    info, pred = verify_inputs()
    print("CLASS ORDER:", " | ".join(f"{i + 1}:{l}" for i, l in enumerate(LABELS)))
    info["environment"] = environment_snapshot(BOOT_SEED)
    (OUT / "input_verification.json").write_text(json.dumps(clean(info), indent=2), encoding="utf-8")
    log.info("inputs verified: %s images, %s patients", info["validation"]["n_images"], info["validation"]["n_patients"])

    rep, au, ap = reproduce_metrics(pred)
    (OUT / "reproduced_c2b_metrics.json").write_text(json.dumps(clean(rep), indent=2), encoding="utf-8")
    log.info("C2-B metrics reproduced (max abs diff vs saved: %.2e)", max(rep["abs_diff"].values()))

    res = build_tables(pred)
    tables, thr = res["tables"], res["thr"]
    long = long_frame(tables)
    long.to_csv(OUT / "operating_points_long.csv", index=False, float_format="%.17g")
    for pol, name in (("baseline_050", "baseline_050_table.csv"), ("youden", "youden_table.csv"), ("f1_optimal", "f1_optimal_table.csv")):
        pd.DataFrame(tables[pol]).to_csv(OUT / name, index=False, float_format="%.17g")
    wide = wide_comparison(tables)
    wide.to_csv(OUT / "threshold_comparison.csv", index=False, float_format="%.17g")
    macro = pd.DataFrame([{"policy": pol, **res["macro"][pol]} for pol in POLICIES])
    macro.to_csv(OUT / "macro_operating_summary.csv", index=False, float_format="%.17g")

    # cross-check against the C1/C2-era Youden implementation (different tie rule only)
    p, raw, v = pred["probs"], pred["raw_labels"], pred["valid"]
    old = c1_thresholds.youden_thresholds((raw == 1).astype(np.float32), p, v, source_split="val")
    old_t = np.array([e["threshold"] for e in old], dtype=float)
    new_t = np.array(thr["youden"], dtype=float)
    xcheck = {"max_abs_diff_vs_src.classification.thresholds": float(np.abs(old_t - new_t).max()),
              "classes_differing": [LABELS[j] for j in range(len(LABELS)) if old_t[j] != new_t[j]],
              "note": "the earlier implementation breaks exact J ties by highest threshold; C3 breaks them by higher sensitivity"}
    log.info("cross-check vs earlier Youden implementation: %s", xcheck)

    prev = {r["observation"]: r["prevalence"] for r in tables["baseline_050"]}
    reps = representative_classes(prev)
    san = sanity_table(pred, tables, au)
    san.to_csv(OUT / "threshold_sanity.csv", index=False, float_format="%.17g")

    groups = {l: "rare" for l in RARE} | {l: "common" for l in COMMON}
    rc = wide.assign(group=wide["observation"].map(lambda l: groups.get(l, "other")))
    rc[["group", "observation", "prevalence", "thr_youden", "recall_050", "recall_youden", "delta_recall_050_to_youden",
        "precision_050", "precision_youden", "delta_precision_050_to_youden", "specificity_050", "specificity_youden",
        "f1_050", "f1_youden", "balanced_accuracy_050", "balanced_accuracy_youden"]].to_csv(
        OUT / "rare_common_analysis.csv", index=False, float_format="%.17g")

    thr_y = np.array(thr["youden"], dtype=float)
    nfa = no_finding_analysis(pred, thr_y)
    (OUT / "no_finding_consistency.json").write_text(json.dumps(clean(nfa), indent=2), encoding="utf-8")

    boot, draws = bootstrap_summary(pred, thr_y)
    boot.to_csv(OUT / "threshold_bootstrap.csv", index=False, float_format="%.17g")
    np.savez_compressed(OUT / "threshold_bootstrap_draws.npz", draws=draws, seed=BOOT_SEED, labels=np.asarray(LABELS))
    log.info("bootstrap done: %d resamples, unstable classes: %s", N_BOOT, boot.loc[boot.unstable_flag, "observation"].tolist())

    # ---------------- threshold files
    meta = info["checkpoint"]
    common = dict(split="validation", checkpoint=meta["path"], checkpoint_sha256=meta["sha256_recomputed"])
    n_img, n_pat = info["validation"]["n_images"], info["validation"]["n_patients"]
    for pol, fname, method, tie in (
            ("youden", "thresholds_youden.json", "youden_j", "exact J ties: higher sensitivity, then higher threshold"),
            ("f1_optimal", "thresholds_f1.json", "f1_optimal", "exact F1 ties: higher recall, then higher threshold")):
        classes = {}
        for r in tables[pol]:
            c = {k: r.get(k) for k in ("threshold", "youden_j", "sensitivity", "specificity", "precision", "f1",
                                       "balanced_accuracy", "tp", "fp", "tn", "fn", "positives", "negatives", "prevalence")}
            c["status"] = r["status"]
            classes[r["observation"]] = clean(c)
        save_threshold_file(OUT / fname, method, **common, classes=classes, extra={
            "tie_rule": tie, "decision_rule_note": "positive if score >= threshold",
            "evaluation_precision_policy": info["evaluation_precision_policy"], "n_images": n_img, "n_patients": n_pat,
            "uncertain_entries_excluded": info["validation"]["n_uncertain_excluded_entries"],
            "label_policy": info["label_policy"], "score_terminology": "sigmoid model scores; calibration not established",
            "created_by": "scripts/c3_threshold_optimization.py", "code_commit": info["environment"]["git"]["commit"]})
    # reload and confirm bit-exact
    assert np.array_equal(load_threshold_file(OUT / "thresholds_youden.json"), thr_y)
    assert np.array_equal(load_threshold_file(OUT / "thresholds_f1.json"), np.array(thr["f1_optimal"], dtype=float))

    # ---------------- Step 14 per-image downstream file
    pred_bin = apply_thresholds(p, thr_y)
    cols = {"image_id": pred["paths"], "patient_id": pred["patient_ids"]}
    for j, lab in enumerate(LABELS):
        cols[f"{lab}_label"] = raw[:, j]            # raw: 1 / 0 / -1 (uncertain) / blank (NaN)
        cols[f"{lab}_score"] = p[:, j]
        cols[f"{lab}_threshold"] = np.full(len(p), thr_y[j])
        cols[f"{lab}_pred"] = pred_bin[:, j].astype(int)
    per_image = pd.DataFrame(cols)
    pi_path = OUT / "validation_predictions_youden.csv.gz"
    per_image.to_csv(pi_path, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    back = pd.read_csv(pi_path, float_precision="round_trip")
    assert np.array_equal(back[[f"{l}_score" for l in LABELS]].to_numpy(), p)               # scores bit-exact
    assert np.array_equal(back[[f"{l}_pred" for l in LABELS]].to_numpy().astype(bool), apply_thresholds(
        back[[f"{l}_score" for l in LABELS]].to_numpy(), back[[f"{l}_threshold" for l in LABELS]].to_numpy()[0]))
    assert back["image_id"].tolist() == pred["paths"] and list(back.columns[2:6]) == [
        f"{LABELS[0]}_label", f"{LABELS[0]}_score", f"{LABELS[0]}_threshold", f"{LABELS[0]}_pred"]

    # ---------------- figures
    t_all = wide["thr_youden"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(LABELS))
    ax.barh(y, t_all, color=SERIES[1], height=0.6, label="Youden J threshold")
    ax.axvline(0.5, color="#0b0b0b", lw=1.2, label="default 0.50")
    for i, t in enumerate(t_all):
        ax.text(t + 0.01, i, f"{t:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selected threshold on the sigmoid score")
    ax.set_title("Per-class Youden J thresholds (C2-B)", pad=22)
    subtitle(ax, "Fitted on the validation split only; the same 0.50 would be applied to every class by default")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig1_youden_thresholds", FIG)
    paired_dot_figure(wide, "recall", "Recall: 0.50 vs Youden J threshold", "fig2_recall_050_vs_youden")
    paired_dot_figure(wide, "precision", "Precision: 0.50 vs Youden J threshold", "fig3_precision_050_vs_youden")
    paired_dot_figure(wide, "f1", "F1: 0.50 vs Youden J threshold", "fig4_f1_050_vs_youden")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    figp, axp = plt.subplots(1, 3, figsize=(15, 4.8))
    figs, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, (role, lab) in enumerate(reps.items()):
        j = list(LABELS).index(lab)
        yy, ss = raw[v[:, j], j] == 1, p[v[:, j], j]
        fpr, tpr, _ = roc_curve(yy, ss)
        prec, rec, _ = precision_recall_curve(yy, ss)
        ty = thr_y[j]
        oy, o5 = operating_point(yy, ss, ty), operating_point(yy, ss, 0.5)
        ax = axes[k]
        ax.plot(fpr, tpr, color=SERIES[0], lw=1.8)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
        ax.scatter([1 - oy["specificity"]], [oy["sensitivity"]], s=90, color=SERIES[1], zorder=3, label=f"Youden thr {ty:.3f}", edgecolor="white", linewidth=1.5)
        ax.scatter([1 - o5["specificity"]], [o5["sensitivity"]], s=90, color="#0b0b0b", marker="s", zorder=3, label="threshold 0.50", edgecolor="white", linewidth=1.5)
        ax.set_title(f"{lab}\n({role}; prevalence {yy.mean():.3f}; AUROC {au[j]:.3f})", fontsize=10, loc="left")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right", fontsize=8)
        ax = axp[k]
        ax.plot(rec, prec, color=SERIES[0], lw=1.8)
        ax.axhline(yy.mean(), color=MUTED, lw=0.8)
        ax.scatter([oy["recall"]], [oy["precision"]], s=90, color=SERIES[1], zorder=3, label=f"Youden thr {ty:.3f}", edgecolor="white", linewidth=1.5)
        if np.isfinite(o5["precision"]):
            ax.scatter([o5["recall"]], [o5["precision"]], s=90, color="#0b0b0b", marker="s", zorder=3, label="threshold 0.50", edgecolor="white", linewidth=1.5)
        ax.set_title(f"{lab}\n({role}; AUPRC {ap[j]:.3f}; line = prevalence)", fontsize=10, loc="left")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(loc="upper right", fontsize=8)
        ax = axs[k]
        c = candidate_table(yy, ss)
        sens, spec = c.tp / c.P, 1 - c.fp / c.N
        prc = np.divide(c.tp, c.tp + c.fp, out=np.full(len(c.tp), np.nan), where=(c.tp + c.fp) > 0)
        f1c = 2 * c.tp / (c.tp + c.fp + c.P)
        for arr, nm, colr in ((sens, "sensitivity", SERIES[0]), (spec, "specificity", SERIES[2]), (prc, "precision", SERIES[3]), (f1c, "F1", SERIES[4])):
            ax.plot(c.thresholds, arr, lw=1.6, color=colr, label=nm)
        ax.axvline(ty, color=SERIES[1], lw=1.5, label=f"Youden {ty:.3f}")
        ax.axvline(0.5, color="#0b0b0b", lw=1.0, ls="-", label="0.50")
        ax.set_xlim(0, 1); ax.set_xlabel("Threshold on score"); ax.set_title(f"{lab} ({role})", fontsize=10, loc="left")
        if k == 0:
            ax.legend(fontsize=7, loc="center right")
    fig.suptitle("Selected ROC curves with operating points (validation)", x=0.01, ha="left", fontweight="semibold"); fig.tight_layout()
    save_figure(fig, "fig5_roc_operating_points", FIG)
    figp.suptitle("Selected precision-recall curves with operating points (validation)", x=0.01, ha="left", fontweight="semibold"); figp.tight_layout()
    save_figure(figp, "fig6_pr_operating_points", FIG)
    figs.suptitle("Metrics as a function of the threshold (validation)", x=0.01, ha="left", fontweight="semibold"); figs.tight_layout()
    save_figure(figs, "fig7_threshold_sweeps", FIG)

    fig, axes = plt.subplots(4, 4, figsize=(15, 12.5))
    bins = np.linspace(0, 1, 51)
    for ax, (j, lab) in zip(axes.flat, enumerate(LABELS)):
        yy, ss = raw[v[:, j], j] == 1, p[v[:, j], j]
        ax.hist(ss[~yy], bins=bins, density=True, alpha=0.55, color=SERIES[0], label="negative")
        ax.hist(ss[yy], bins=bins, density=True, alpha=0.55, color=SERIES[1], label="positive")
        ax.axvline(thr_y[j], color="#0b0b0b", lw=1.2)
        ax.axvline(0.5, color=MUTED, lw=1.0, ls="--")
        fl = san.loc[j, "flags"]
        ax.set_title(f"{lab} (thr {thr_y[j]:.3f}{'; ' + fl if fl != 'none' else ''})", fontsize=9, loc="left")
        ax.set_yticks([])
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[14:]:
        ax.axis("off")
    fig.suptitle("Validation score distributions by true label (solid = Youden threshold, dashed = 0.50)", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, "fig8_score_distributions", FIG)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(LABELS))
    ax.hlines(y, boot["p2_5"], boot["p97_5"], color=SERIES[0], lw=3, alpha=0.7, label="bootstrap 95% interval")
    ax.hlines(y, boot["q1"], boot["q3"], color=SERIES[0], lw=7, alpha=0.9, label="IQR")
    ax.scatter(boot["bootstrap_median"], y, s=36, color="white", edgecolor=SERIES[0], zorder=3, label="bootstrap median")
    ax.scatter(boot["original_threshold"], y, s=60, color=SERIES[1], zorder=4, label="selected (original sample)", edgecolor="white", linewidth=1.2)
    ax.axvline(0.5, color="#0b0b0b", lw=1.0)
    for i, r in boot.iterrows():
        if r.unstable_flag:
            ax.text(min(r.p97_5 + 0.015, 0.97), i, "unstable", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, LABELS); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Youden threshold")
    ax.set_title("Threshold stability under patient-level bootstrap", pad=22)
    subtitle(ax, f"{N_BOOT:,} resamples of {n_pat:,} validation patients (seed {BOOT_SEED}); flagged if width > {UNSTABLE['max_interval_width']} or > median")
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", visible=False)
    save_figure(fig, "fig9_threshold_bootstrap", FIG)

    co = nfa["youden"]["co_occurrence_pct_of_no_finding_positive"]
    order = sorted(co, key=lambda k: -co[k])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(order, [co[k] for k in order], color=SERIES[1], height=0.6)
    ax.invert_yaxis(); ax.set_xlabel("% of images with predicted No Finding that also predict this finding")
    ax.set_title("Predicted No Finding vs predicted abnormal findings (Youden thresholds)", pad=22)
    subtitle(ax, f"{nfa['youden']['no_finding_positive']:,} images with No Finding positive; contradiction rate {nfa['youden']['contradiction_rate_pct_of_no_finding_positive']:.1f}%")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig10_no_finding_cooccurrence", FIG)

    summary = {"reproduction": {"reproduced": rep["reproduced"], "max_abs_diff_vs_saved": max(rep["abs_diff"].values())},
               "macro_operating_summary": res["macro"], "youden_thresholds": dict(zip(LABELS, thr["youden"])),
               "f1_thresholds": dict(zip(LABELS, thr["f1_optimal"])), "cross_check_vs_earlier_youden_impl": xcheck,
               "representative_classes": reps, "unstable_classes": boot.loc[boot.unstable_flag, "observation"].tolist(),
               "sanity_flagged": san.loc[san["flags"] != "none", ["observation", "flags"]].to_dict(orient="records"),
               "no_finding": nfa, "n_images": n_img, "n_patients": n_pat, "n_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED,
               "declared_rules": {"sanity_flags": SANITY, "unstable": UNSTABLE},
               "locked_test_split_used": False}
    (OUT / "c3_results.json").write_text(json.dumps(clean(summary), indent=2), encoding="utf-8")
    print(macro.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
