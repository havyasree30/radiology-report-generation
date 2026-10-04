# G1A journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_three_system_main_comparison.*` | B0 / G1A / G1 finding F1 0.366 / 0.196 / 0.383 | Results |
| Table 2 `tables/table2_g1_minus_g1a_controlled_comparison.*` | Retrieval effect: F1 +0.186 (95% CI +0.136 to +0.232), hallucination -0.497 (95% CI -0.561 to -0.431) | Results |
| Table 3 `tables/table3_per_finding_results.*` | Per-finding retention for B0 / G1A / G1 | Results / Supplementary |
| Table 4 `tables/table4_retrieval_induced_findings.*` | 136 findings added and 306 removed by retrieval context; net +33 matching / -203 unsupported | Results |
| Table 5 `tables/table5_normal_abnormal_analysis.*` | Abnormal recall classifier 0.726, G1A 0.710, G1 0.581 | Results / Discussion |
| Table 6 `tables/table6_supplementary_stratified_by_classifier_state.*` | Controlled comparison within classifier-abnormal studies (G1A not degenerate) | Supplementary |
| Figure 1 `figures/fig1_finding_precision_recall_f1.*` | Finding agreement of three systems | Results |
| Figure 2 `figures/fig2_hallucination_vs_omission.*` | Hallucination versus omission | Results |
| Figure 3 `figures/fig3_fp_propagation_vs_tp_retention.*` | FP propagation versus TP retention | Results |
| Figure 4 `figures/fig4_normal_abnormal_recall.*` | Over-normalisation origin | Results / Discussion |
| Figure 5 `figures/fig5_retrieval_effect_summary.*` | Controlled retrieval effect | Results / Discussion |
| `NO_RETRIEVAL_LLM_ABLATION.md` | Full G1A analysis | Supplementary / internal |
| `MANUSCRIPT_G1A_METHODS.md`, `MANUSCRIPT_G1A_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_G1A.md` | Content for Methodology, Results, Discussion | University report |
| `g1a_prompt_frozen.json`, `g1a_generator_metadata.json`, `g1a_prompt_diff_vs_g1.txt` | Frozen G1A prompt, generator metadata, prompt diff | Methods / code release |
| `g1a_per_study_results.csv`, `g1a_paired_differences.csv` | All per-study and paired results | Supplementary data |
| `g1a_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
