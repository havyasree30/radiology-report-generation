"""R1 additional experiment: validation-derived, precision-aware retrieval gates.

Uses ONLY the CheXpert classifier VALIDATION outputs (stored C2-B scores -> frozen C5 calibration -> frozen C4
decisions). The frozen classifier, thresholds and calibrators are not changed; the locked classifier test set and every
retrieval result are not used.

    .venv\\Scripts\\python.exe -m scripts.r1_03_derive_gates
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.classification.calibration import patient_folds
from src.classification.final_test import apply_frozen_pipeline, load_frozen, verify_freeze
from src.classification.labels import LABELS, NO_FINDING
from src.classification.serialization import load_predictions_npz
from src.retrieval.gating import BETA, QUANTILES, cross_fitted_gain, fbeta, gate_counts, precision_recall, select_gate
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
OUT = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
N_BOOT, SEED, N_FOLDS = 1000, 42, 5


def main() -> int:
    verify_freeze(EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json", PROJECT_ROOT)
    policy, cal = load_frozen(EXP / "c4_operating_policy/final_operating_policy.json", EXP / "c5_calibration/final_calibrators.json")
    va = load_predictions_npz(EXP / "c2b_sqrt_weighted_bce/evaluation/fp32_strict/validation_predictions.npz")
    o = apply_frozen_pipeline(va["probs"], policy, cal)
    prob, final, valid, raw = o["prob"], o["pred_final"], va["valid"], va["raw_labels"]
    uniq, pidx = np.unique(np.asarray(va["patient_ids"]).astype(str), return_inverse=True)
    folds = patient_folds(va["patient_ids"], valid, raw, N_FOLDS, SEED)
    rows, js = [], {}
    for j, lab in enumerate(LABELS):
        if lab == NO_FINDING:
            continue
        v = valid[:, j]
        p, y, pos = prob[v, j], raw[v, j] == 1, final[v, j]
        g_star, scores = select_gate(p, y, pos)
        op_gate = float(p[pos].min())
        cv = cross_fitted_gain(p, y, pos, folds[v], pidx[v], N_BOOT, SEED)
        n_pos, n_tp0 = int(y.sum()), int((pos & y).sum())
        pr0, rc0 = precision_recall(p, y, pos, op_gate)
        f0 = float(fbeta(n_tp0, int((pos & ~y).sum()), n_pos - n_tp0))
        stricter = g_star > op_gate
        adopted = bool(stricter and cv["ci95_low"] > 0)
        gate = g_star if adopted else None
        pr1, rc1 = precision_recall(p, y, pos, gate) if adopted else (pr0, rc0)
        tp1, fp1 = gate_counts(p, y, pos, gate) if adopted else (n_tp0, int((pos & ~y).sum()))
        f1 = float(fbeta(tp1, fp1, n_pos - tp1))
        c = cal["classes"][lab]
        raw_equiv = float(1 / (1 + np.exp(-((np.log(g_star / (1 - g_star)) - c["b"]) / c["a"])))) if adopted else None
        reason = ("adopted: cross-fitted F0.5 gain lower 95% bound > 0" if adopted else
                  "no gate: best candidate equals the operating point" if not stricter else
                  "no gate: cross-fitted F0.5 gain not reliably above zero (95% lower bound <= 0)")
        rows.append({"finding": lab, "frozen_raw_threshold": policy["thresholds"][j], "frozen_calibrated_equivalent_threshold": c["calibrated_equivalent_threshold"],
                     "retrieval_gate_calibrated_probability": gate, "gate_adopted": adopted, "candidate_gate_best_f05": g_star,
                     "precision_before": pr0, "precision_after": pr1, "recall_before": rc0, "recall_after": rc1,
                     "f05_before": f0, "f05_after": f1, "frozen_positives": int(pos.sum()), "positives_passing_gate": int(tp1 + fp1),
                     "fraction_of_frozen_positives_passing": (tp1 + fp1) / max(int(pos.sum()), 1),
                     "true_positives_before": n_tp0, "true_positives_after": tp1, "oof_f05_gain": cv["oof_gain_f05"],
                     "oof_gain_ci95_low": cv["ci95_low"], "oof_gain_ci95_high": cv["ci95_high"], "decision_reason": reason})
        js[lab] = {"gate_calibrated_probability": gate, "gate_adopted": adopted, "gate_as_raw_score_equivalent": raw_equiv,
                   "frozen_raw_threshold": float(policy["thresholds"][j]), "frozen_calibrated_equivalent_threshold": c["calibrated_equivalent_threshold"],
                   "reason": reason}
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "retrieval_gate_table.csv", index=False, float_format="%.17g")
    doc = {"purpose": "Downstream query-building gates. NOT classifier thresholds: the frozen C4 decisions are unchanged; a gate only decides whether a classifier-positive finding enters a retrieval query.",
           "derived_from": "classifier VALIDATION outputs only (C2-B scores, frozen C5 calibration, frozen C4 decisions); locked classifier test set not used; no retrieval result used",
           "n_validation_images": int(len(prob)), "n_validation_patients": int(len(uniq)),
           "principle": {"criterion": f"F_beta with beta={BETA} (precision counts twice as much as recall) to reduce false-positive propagation",
                         "candidates": f"calibrated-probability quantiles {list(QUANTILES)} of the frozen-positive validation images of each class (quantile 0 = operating point)",
                         "adoption_rule": "adopt the best candidate only if it is stricter than the operating point AND its cross-fitted (5-fold patient-level, seed 42) F_0.5 gain has a 95% patient-bootstrap lower bound > 0; otherwise no gate (finding passes unchanged)",
                         "bootstrap": {"n_resamples": N_BOOT, "seed": SEED}, "folds": {"n_folds": N_FOLDS, "seed": SEED, "same_folds_as_C5": True}},
           "no_finding": "not gated (it is not a pathology query term)", "gates": js,
           "classes_with_gate": [l for l, d in js.items() if d["gate_adopted"]], "classes_without_gate": [l for l, d in js.items() if not d["gate_adopted"]]}
    (OUT / "precision_aware_query_gates.json").write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tab[["finding", "retrieval_gate_calibrated_probability", "precision_before", "precision_after", "recall_before", "recall_after",
               "fraction_of_frozen_positives_passing", "oof_f05_gain", "oof_gain_ci95_low", "gate_adopted"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
