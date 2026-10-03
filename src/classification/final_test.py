"""C6 helpers: the frozen classification pipeline applied to held-out scores, plus the evaluation
functions that are shared by the C6 scripts and their tests.

Nothing here fits a threshold or a calibrator. The frozen objects are loaded from the C4 / C5 files,
whose hashes are recorded in the freeze manifest and re-verified before any test metric is computed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.classification.calibration import apply_calibrator, load_calibrators, safe_logit
from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS
from src.classification.operating_points import metrics_from_counts
from src.classification.operating_policy import apply_final_policy, load_final_policy

NF = LABELS.index(NO_FINDING)
IDX12 = [LABELS.index(l) for l in PATHOLOGY_LABELS]
IDX13 = [i for i in range(len(LABELS)) if i != NF]
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]  # C2 definition
METRICS = ("precision", "recall", "specificity", "f1", "balanced_accuracy")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ frozen pipeline
def load_frozen(policy_path: Path, calibrators_path: Path) -> tuple[dict, dict]:
    policy = load_final_policy(policy_path)
    cal = load_calibrators(calibrators_path)
    if cal.get("uniform_method") != "platt" or any(cal["classes"][l]["method"] != "platt" for l in LABELS):
        raise ValueError("C6 expects the frozen C5 Platt calibrators for all 14 classes")
    return policy, cal


def apply_frozen_pipeline(scores: np.ndarray, policy: dict, cal: dict) -> dict:
    """Raw scores (untouched) -> calibrated probabilities, raw-threshold decisions and final (post-rule) decisions."""
    scores = np.asarray(scores, dtype=np.float64)
    z = safe_logit(scores)
    prob = np.column_stack([apply_calibrator(cal["classes"][l], z[:, j]) for j, l in enumerate(LABELS)])
    pred_raw, pred_final = apply_final_policy(scores, policy)
    cal_thr = np.array([cal["classes"][l]["calibrated_equivalent_threshold"] for l in LABELS])
    return {"scores": scores, "prob": prob, "pred_raw": pred_raw, "pred_final": pred_final,
            "thresholds": policy["thresholds"], "calibrated_thresholds": cal_thr}


# ------------------------------------------------------------------ binary metrics
def binary_table(pred: np.ndarray, raw_labels: np.ndarray, valid: np.ndarray, thresholds: np.ndarray,
                 weights: np.ndarray | None = None) -> pd.DataFrame:
    """Per-class confusion counts and metrics on valid entries (optionally with integer weights)."""
    rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        y, pr = raw_labels[v, j] == 1, pred[v, j]
        w = np.ones(v.sum()) if weights is None else weights[v].astype(float)
        c = {"tp": float((w * (pr & y)).sum()), "fp": float((w * (pr & ~y)).sum()),
             "tn": float((w * (~pr & ~y)).sum()), "fn": float((w * (~pr & y)).sum())}
        if weights is None:
            c = {k: int(x) for k, x in c.items()}
        m = metrics_from_counts(**c)
        rows.append({"observation": lab, "threshold": float(thresholds[j]),
                     "prevalence": float((w * y).sum() / w.sum()), **{k: c[k] for k in ("tp", "fp", "tn", "fn")},
                     **{k: m[k] for k in METRICS}, "fp_per_tp": c["fp"] / c["tp"] if c["tp"] else float("nan"),
                     "predicted_positive_rate": float((w * pr).sum() / w.sum())})
    return pd.DataFrame(rows)


def macro_binary(table: pd.DataFrame, idx: list[int] | None = None) -> dict:
    sub = table if idx is None else table.iloc[idx]
    return {f"macro_{m}": float(np.nanmean(sub[m])) for m in METRICS}


def micro_binary(table: pd.DataFrame) -> dict:
    tp, fp, tn, fn = (int(table[k].sum()) for k in ("tp", "fp", "tn", "fn"))
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return {"micro_precision": prec, "micro_recall": rec, "micro_f1": 2 * prec * rec / (prec + rec)}


def count_stats(n: np.ndarray) -> dict:
    return {"mean": float(n.mean()), "median": float(np.median(n)), "p25": float(np.percentile(n, 25)),
            "p75": float(np.percentile(n, 75)), "p95": float(np.percentile(n, 95)), "max": int(n.max()),
            "pct_0": float(100 * (n == 0).mean()), "pct_1": float(100 * (n == 1).mean()),
            "pct_2_3": float(100 * ((n >= 2) & (n <= 3)).mean()), "pct_4_5": float(100 * ((n >= 4) & (n <= 5)).mean()),
            "pct_gt5": float(100 * (n > 5).mean())}


# ------------------------------------------------------------------ patient-level bootstrap
def bootstrap_weights(patient_ids, n_boot: int, seed: int):
    """Yield (n_images,) integer weights: every image of a resampled patient is repeated as often as the patient was drawn."""
    pids = np.asarray(patient_ids).astype(str)
    uniq, inv = np.unique(pids, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw = rng.integers(0, len(uniq), len(uniq))
        yield np.bincount(draw, minlength=len(uniq))[inv]


def weighted_macro_metrics(scores, prob, pred_final, raw_labels, valid, weights) -> dict:
    """Macro (14-class) metrics for one bootstrap draw; per-class values undefined in a draw are skipped (nan-mean)."""
    keep = weights > 0
    s, p, pr, rl, vd, w = scores[keep], prob[keep], pred_final[keep], raw_labels[keep], valid[keep], weights[keep]
    au, ap, br = [], [], []
    for j in range(len(LABELS)):
        v = vd[:, j]
        y = rl[v, j] == 1
        wv = w[v]
        if (wv * y).sum() == 0 or (wv * ~y).sum() == 0:
            au.append(np.nan), ap.append(np.nan), br.append(np.nan)
            continue
        au.append(roc_auc_score(y, s[v, j], sample_weight=wv))
        ap.append(average_precision_score(y, s[v, j], sample_weight=wv))
        br.append(float((wv * (p[v, j] - y) ** 2).sum() / wv.sum()))
    tab = binary_table(pr, rl, vd, np.zeros(len(LABELS)), weights=w)
    out = {"macro_auroc": float(np.nanmean(au)), "macro_auprc": float(np.nanmean(ap)), "macro_brier": float(np.nanmean(br)),
           **macro_binary(tab)}
    return out


# ------------------------------------------------------------------ downstream output schema
def output_record(image_id: str, raw: np.ndarray, prob: np.ndarray, thr: np.ndarray, cal_thr: np.ndarray,
                  final: np.ndarray) -> dict:
    """One image -> the downstream record. Probabilities are per class and are NOT normalised across classes."""
    findings = {l: {"raw_score": float(raw[j]), "calibrated_probability": float(prob[j]), "threshold": float(thr[j]),
                    "calibrated_equivalent_threshold": float(cal_thr[j]), "positive": bool(final[j])}
                for j, l in enumerate(LABELS)}
    return {"image_id": image_id, "findings": findings, "positive_findings": [l for j, l in enumerate(LABELS) if final[j]]}


def output_schema() -> dict:
    f = {"type": "object", "additionalProperties": False,
         "required": ["raw_score", "calibrated_probability", "threshold", "calibrated_equivalent_threshold", "positive"],
         "properties": {"raw_score": {"type": "number", "minimum": 0, "maximum": 1,
                                      "description": "DenseNet-121 sigmoid output (model score, not a probability)"},
                        "calibrated_probability": {"type": "number", "minimum": 0, "maximum": 1,
                                                   "description": "C5 Platt-calibrated estimate of P(CheXpert label positive | image) under the project label policy"},
                        "threshold": {"type": "number", "minimum": 0, "maximum": 1,
                                      "description": "frozen C4 F1-optimal threshold on the RAW score; it alone decides activation"},
                        "calibrated_equivalent_threshold": {"type": "number", "minimum": 0, "maximum": 1,
                                                            "description": "the same threshold expressed on the calibrated scale (for display only)"},
                        "positive": {"type": "boolean",
                                     "description": "final decision after the No Finding rule (only No Finding can differ from raw_score >= threshold)"}}}
    return {"$schema": "http://json-schema.org/draft-07/schema#", "title": "Final classifier output (one image)",
            "description": "Multi-label output: the 14 entries are independent and are never normalised or softmaxed across classes.",
            "type": "object", "additionalProperties": False, "required": ["image_id", "findings", "positive_findings"],
            "properties": {"image_id": {"type": "string"},
                           "findings": {"type": "object", "additionalProperties": False, "required": list(LABELS),
                                        "properties": {l: f for l in LABELS}},
                           "positive_findings": {"type": "array", "uniqueItems": True, "items": {"enum": list(LABELS)}}}}


def validate_record(rec: dict) -> None:
    """Dependency-free structural validation of one output record against the schema above."""
    if set(rec) != {"image_id", "findings", "positive_findings"}:
        raise ValueError("record keys mismatch")
    if list(rec["findings"]) != list(LABELS):
        raise ValueError("findings must contain the 14 labels in the fixed order")
    for lab, f in rec["findings"].items():
        if set(f) != {"raw_score", "calibrated_probability", "threshold", "calibrated_equivalent_threshold", "positive"}:
            raise ValueError(f"{lab}: field mismatch")
        for k in ("raw_score", "calibrated_probability", "threshold", "calibrated_equivalent_threshold"):
            if not (isinstance(f[k], float) and 0.0 <= f[k] <= 1.0):
                raise ValueError(f"{lab}.{k} outside [0,1]")
        if not isinstance(f["positive"], bool):
            raise ValueError(f"{lab}.positive must be boolean")
    if rec["positive_findings"] != [l for l, f in rec["findings"].items() if f["positive"]]:
        raise ValueError("positive_findings inconsistent with findings")


# ------------------------------------------------------------------ freeze manifest
def verify_freeze(manifest_path: Path, root: Path) -> dict:
    """Re-hash every file recorded in the manifest; returns {path: matches}. Raises if any differs."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    status = {rel: sha256_file(root / rel) == digest for rel, digest in m["hashed_files"].items()}
    bad = [k for k, ok in status.items() if not ok]
    if bad:
        raise RuntimeError(f"frozen files changed since the freeze manifest was written: {bad}")
    return status
