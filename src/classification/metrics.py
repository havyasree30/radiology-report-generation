"""Validation metrics over valid (unmasked) entries only.

A per-class metric is UNDEFINED (NaN, with a reason) when the class has no
valid positives or no valid negatives. Undefined values are never replaced by 0;
macro averages are taken over defined classes and report how many were used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.classification.labels import LABELS


def per_class_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(LABELS):
        m = mask[:, j].astype(bool)
        yj, pj = y[m, j], p[m, j]
        n_pos, n_neg = int((yj == 1).sum()), int((yj == 0).sum())
        row = {"observation": lab, "n_valid": int(m.sum()), "n_positive": n_pos, "n_negative": n_neg,
               "prevalence": n_pos / max(m.sum(), 1), "auroc": np.nan, "auprc": np.nan, "undefined_reason": ""}
        if n_pos == 0 or n_neg == 0:
            row["undefined_reason"] = "no valid positives" if n_pos == 0 else "no valid negatives"
        else:
            row["auroc"] = roc_auc_score(yj, pj)
            row["auprc"] = average_precision_score(yj, pj)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict:
    pc = per_class_metrics(y, p, mask)
    m = mask.astype(bool)
    yf, pf = y[m], p[m]
    micro_ok = 0 < yf.sum() < len(yf)
    return {
        "macro_auroc": float(np.nanmean(pc["auroc"])) if pc["auroc"].notna().any() else np.nan,
        "macro_auprc": float(np.nanmean(pc["auprc"])) if pc["auprc"].notna().any() else np.nan,
        "micro_auroc": float(roc_auc_score(yf, pf)) if micro_ok else np.nan,
        "micro_auprc": float(average_precision_score(yf, pf)) if micro_ok else np.nan,
        "n_classes_defined": int(pc["auroc"].notna().sum()),
        "undefined_classes": pc.loc[pc["auroc"].isna(), "observation"].tolist(),
    }
