# C4 journal asset index

| Asset | File | What it demonstrates | Suggested manuscript section |
|---|---|---|---|
| Figure 1 | `figures/fig1_policy_precision_recall_f1.{png,pdf,svg}` | Macro precision/recall/F1 trade-off across policies | Results |
| Figure 2 | `figures/fig2_policy_specificity_balanced_accuracy.{png,pdf,svg}` | Specificity and balanced accuracy by policy | Results |
| Figure 3 | `figures/fig3_predicted_findings_per_image.{png,pdf,svg}` | Over-prediction of findings per image by policy | Results / Discussion |
| Figure 4 | `figures/fig4_per_class_f1.{png,pdf,svg}` | Per-class F1 under three policies | Results / Supplementary |
| Figure 5 | `figures/fig5_per_class_precision_recall_tradeoff.{png,pdf,svg}` | Per-class movement along the precision-recall trade-off | Results / Supplementary |
| Figure 6 | `figures/fig6_rare_class_false_alarm_burden.{png,pdf,svg}` | Rare-class precision, recall and false-alarm burden | Results / Discussion |
| Figure 7 | `figures/fig7_no_finding_consistency.{png,pdf,svg}` | Effect of the No Finding rule | Results |
| Figure 8 | `figures/fig8_final_thresholds.{png,pdf,svg}` | Final per-class thresholds | Results / Supplementary |
| Table 1 | `tables/table1_policy_comparison.{csv,md,tex}` | Overall four-policy comparison (14 classes) | Results |
| Table 1b | `tables/table1b_policy_comparison_13_abnormal.{csv,md,tex}` | Same, 13 abnormal observations only | Results / Supplementary |
| Table 2 | `tables/table2_final_per_class_operating_points.{csv,md,tex}` | Final class-specific thresholds and metrics | Results / Supplementary |
| Table 3 | `tables/table3_rare_class_performance.{csv,md,tex}` | Rare-class false-alarm burden | Results / Discussion |
| Table 4 | `tables/table4_no_finding_consistency.{csv,md,tex}` | No Finding before/after the rule | Results |
| Captions | `FIGURE_CAPTIONS.md`, `TABLE_CAPTIONS.md` | Manuscript-ready captions | All |
| Prose | `MANUSCRIPT_C4_METHODS.md`, `MANUSCRIPT_C4_RESULTS.md` | Draft Methods and Results text | Methods / Results |
| Policy | `final_operating_policy.json` | Machine-readable selected policy | Supplementary / code release |
| Predictions | `validation_predictions_final_policy.csv.gz` | Per-image scores, thresholds, raw and final predictions | Supplementary / reproducibility |
| Figure data | `figures/source_data/*.csv` | Exact numbers plotted in each figure | Supplementary |
