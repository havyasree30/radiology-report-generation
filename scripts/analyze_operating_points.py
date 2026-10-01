"""Behaviour of the Youden operating points on the VALIDATION predictions (no test data).

    .venv\\Scripts\\python.exe -m scripts.analyze_operating_points
Outputs: <experiment>/operating_points.csv, operating_points_summary.json
Note: labels are scored under the training label policy (blank = 0), so a
"false positive" may be a finding that was present but unmentioned in the report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS
from src.classification.serialization import load_predictions_npz
from src.classification.thresholds import load_thresholds
from src.utils.config import PROJECT_ROOT


def main() -> int:
    out = PROJECT_ROOT / "results/classification/densenet121_baseline"
    # lossless artifact (the pandas default CSV parser is not exact; see C1_FINALIZATION_LOG.json)
    d = load_predictions_npz(out / "validation_predictions.npz")
    thr = load_thresholds(out / "thresholds/youden_j_thresholds.json")
    P, raw, V = d["probs"], d["raw_labels"], d["valid"]
    pos, Y = P >= thr, raw == 1
    rows = []
    for j, lab in enumerate(LABELS):
        m = V[:, j]
        tp, fp = int((pos[m, j] & Y[m, j]).sum()), int((pos[m, j] & ~Y[m, j]).sum())
        fn = int((~pos[m, j] & Y[m, j]).sum())
        rows.append({"observation": lab, "threshold": thr[j], "prevalence": Y[m, j].mean(),
                     "predicted_positive_rate": pos[m, j].mean(), "tp": tp, "fp": fp, "fn": fn,
                     "ppv": tp / (tp + fp) if tp + fp else np.nan, "sensitivity": tp / (tp + fn) if tp + fn else np.nan})
    pd.DataFrame(rows).to_csv(out / "operating_points.csv", index=False)
    nf = LABELS.index(NO_FINDING)
    patho = [LABELS.index(l) for l in PATHOLOGY_LABELS]
    conflict = pos[:, nf] & pos[:, patho].any(axis=1)
    summ = {"n_images": int(len(d)),
            "predicted_positives_per_image_mean": float(pos.sum(1).mean()),
            "predicted_positives_per_image_median": float(np.median(pos.sum(1))),
            "explicit_true_positives_per_image_mean": float(Y.sum(1).mean()),
            "images_with_zero_predicted_positives": int((pos.sum(1) == 0).sum()),
            "images_with_no_finding_pathology_warning": int(conflict.sum()),
            "pct_images_with_no_finding_pathology_warning": float(100 * conflict.mean())}
    (out / "operating_points_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
