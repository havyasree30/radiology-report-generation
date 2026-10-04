"""F1 step 6: descriptive validation -> test comparison (test minus validation, absolute differences). Validation = the dry run of the SAME code on the IU
validation partition. The differences are never used for tuning.

    .venv\\Scripts\\python.exe -m scripts.f1_06_compare
"""

from __future__ import annotations

import json

import pandas as pd

from src.f1.pipeline import F1
from src.utils.config import PROJECT_ROOT



def main(test_dir: str = "test") -> int:
    v = json.loads((F1 / "validation_dry_run/f1_summary.json").read_text(encoding="utf-8"))
    t = json.loads((F1 / test_dir / "f1_summary.json").read_text(encoding="utf-8"))
    vr = pd.read_csv(F1 / "validation_dry_run/retrieval_metrics_by_k.csv")
    tr = pd.read_csv(F1 / test_dir / "retrieval_metrics_by_k.csv")
    rows = []

    def add(group, name, vv, tv, vlo=None, vhi=None, tlo=None, thi=None):
        rows.append({"group": group, "metric": name, "validation": vv, "validation_ci95_low": vlo, "validation_ci95_high": vhi, "test": tv, "test_ci95_low": tlo, "test_ci95_high": thi, "test_minus_validation": tv - vv})

    for k, lab in (("macro_auroc", "macro AUROC"), ("macro_auprc", "macro AUPRC"), ("macro_f1", "macro F1"), ("macro_brier", "Brier score (macro)"), ("macro_ece", "ECE (macro)")):
        a, b = v["classification"]["ci95"].get(k), t["classification"]["ci95"].get(k)
        add("classification (IU, same pipeline)", lab, v["classification"]["point"][k], t["classification"]["point"][k], a and a["ci95_low"], a and a["ci95_high"], b and b["ci95_low"], b and b["ci95_high"])
    for m, lab in (("jaccard_truth", "Jaccard@5"), ("union_coverage", "finding coverage@5")):
        a, b = vr[(vr.K == 5) & (vr.metric == m)].iloc[0], tr[(tr.K == 5) & (tr.metric == m)].iloc[0]
        add("retrieval (classifier query)", lab, a.classifier_query, b.classifier_query, a.classifier_ci95_low, a.classifier_ci95_high, b.classifier_ci95_low, b.classifier_ci95_high)
    for k, lab in (("precision", "finding precision"), ("recall", "finding recall"), ("f1", "finding F1"), ("hallucination_rate", "hallucination rate"), ("omission_rate", "omission rate"), ("clf_fp_propagation", "classifier FP propagation"), ("tp_retention", "classifier TP retention")):
        a, b = v["report"]["metrics"][k], t["report"]["metrics"][k]
        add("report generation", lab, a["value"], b["value"], a["ci95_low"], a["ci95_high"], b["ci95_low"], b["ci95_high"])
    vd, td = v["report"]["three_state"]["decided"], t["report"]["three_state"]["decided"]
    for k, lab in (("decision_coverage", "decision coverage"), ("normal_recall_among_decided", "normal recall (decided)"), ("abnormal_recall_among_decided", "abnormal recall (decided)")):
        add("system", lab, vd[k]["value"], td[k]["value"], vd[k]["ci95_low"], vd[k]["ci95_high"], td[k]["ci95_low"], td[k]["ci95_high"])
    add("system", "indeterminate rate (P3)", v["report"]["three_state"]["state_counts_P3"]["indeterminate"] / v["report"]["n_primary_P3_with_final_report"], t["report"]["three_state"]["state_counts_P3"]["indeterminate"] / t["report"]["n_primary_P3_with_final_report"])
    add("system", "abnormal-routing -> normal-prose rate (%)", v["report"]["routing_prose_consistency"]["abnormal_routing_to_normal_prose_rate_P3"], t["report"]["routing_prose_consistency"]["abnormal_routing_to_normal_prose_rate_P3"])
    df = pd.DataFrame(rows)
    df.to_csv(F1 / ("validation_to_test_comparison.csv" if test_dir == "test" else "dry_run_comparison_validation_vs_validation.csv"), index=False, float_format="%.17g")
    print(df.round(4).to_string())
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
