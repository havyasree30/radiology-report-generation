# C6 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_final_overall_performance.*` | Macro AUROC 0.803, AUPRC 0.427, F1 0.450 with bootstrap CIs | Results |
| Table 2 `tables/table2_per_class_final_performance.*` | AUROC 0.682 (Enlarged Cardiomediastinum) to 0.888 (Pleural Effusion) | Results / Supplementary |
| Table 3 `tables/table3_validation_vs_test.*` | Macro AUROC difference +0.0014 (test minus validation) | Results |
| Table 4 `tables/table4_rare_class_results.*` | Rare-class precision 0.113–0.261 on test | Results |
| Table 5 `tables/table5_locked_test_calibration.*` | Macro ECE 0.0069, Brier 0.0837 on test | Results |
| Figure 1 `figures/fig1_validation_vs_test.*` | Validation and test summary metrics are close | Results |
| Figure 2 `figures/fig2_per_class_auroc_auprc.*` | Per-class discrimination; AUPRC above prevalence in all classes | Results |
| Figure 3 `figures/fig3_precision_recall_f1.*` | Recall exceeds precision in 14 of 14 classes at F1-optimal thresholds | Results |
| Figure 4 `figures/fig4_rare_class_performance.*` | Rare classes: low precision and several false positives per true positive | Results / Discussion |
| Figure 5 `figures/fig5_test_reliability.*` | Frozen Platt calibration on test vs raw scores | Results |
| Figure 6 `figures/fig6_finding_counts.*` | Policy predicts more findings per image than the labels contain (3.43 vs 2.32) | Results / Discussion |
| Figure 7 `figures/fig7_bootstrap_ci.*` | Narrow patient-level bootstrap intervals | Results / Supplementary |
| `FINAL_CLASSIFIER_TEST_REPORT.md` | Full locked-test report | Supplementary / internal |
| `MANUSCRIPT_C6_METHODS.md`, `MANUSCRIPT_C6_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `FINAL_CLASSIFIER_SPECIFICATION.md`, `FINAL_CLASSIFIER_FREEZE_MANIFEST.json` | Permanent frozen specification and hashes | Methods / Supplementary / code release |
| `FINAL_CLASSIFIER_OUTPUT_SCHEMA.json` | Downstream output record | Methods / downstream use |
| `test_*.csv`, `test_predictions_*.csv.gz`, `test_bootstrap_draws.npz` | Per-image predictions and all metric tables | Supplementary data |
| `c6_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
