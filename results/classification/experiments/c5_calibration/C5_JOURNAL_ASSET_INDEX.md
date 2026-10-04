# C5 journal asset index

| Asset | File | Purpose | Suggested manuscript section |
|---|---|---|---|
| Figure 1 | `figures/fig1_brier_raw_vs_calibrated.{png,pdf,svg}` | Per-class Brier score, raw vs calibrated | Results |
| Figure 2 | `figures/fig2_logloss_raw_vs_calibrated.{png,pdf,svg}` | Per-class log loss, raw vs calibrated | Results / Supplementary |
| Figure 3 | `figures/fig3_reliability_curves.{png,pdf,svg}` | Reliability curves for rare, medium and common classes | Results |
| Figure 4 | `figures/fig4_calibration_change_per_class.{png,pdf,svg}` | Per-class relative change in calibration metrics | Results |
| Figure 5 | `figures/fig5_mean_probability_vs_prevalence.{png,pdf,svg}` | Over-/under-confidence pattern across all classes | Results / Discussion |
| Table 1 | `tables/table1_calibration_method_comparison.{csv,md,tex}` | Macro comparison of calibration methods | Results |
| Table 2 | `tables/table2_per_class_raw_vs_calibrated.{csv,md,tex}` | Per-class raw vs selected calibration | Results / Supplementary |
| Table 3 | `tables/table3_final_calibration_parameters.{csv,md,tex}` | Final Platt parameters | Supplementary |
| Table 4 | `tables/table4_threshold_mapping.{csv,md,tex}` | Raw to calibrated threshold mapping, decision mismatches | Results / Supplementary |
| Technical report | `CALIBRATION_ANALYSIS.md` | Full methods, results, limitations, decision | Supplementary / internal |
| Methods draft | `MANUSCRIPT_C5_METHODS.md` | Manuscript-ready Methods text | Methods |
| Results draft | `MANUSCRIPT_C5_RESULTS.md` | Manuscript-ready Results text | Results |
| Captions | `FIGURE_CAPTIONS.md`, `TABLE_CAPTIONS.md` | Journal captions | All |
| Calibrators | `final_calibrators.json` | Machine-readable final calibrators | Code release / Supplementary |
| Per-image output | `validation_predictions_calibrated.csv.gz` | Scores, calibrated probabilities, thresholds and decisions per image | Supplementary |
| Figure data | `figures/source_data/*.csv` | Exact values plotted in each figure | Supplementary |
