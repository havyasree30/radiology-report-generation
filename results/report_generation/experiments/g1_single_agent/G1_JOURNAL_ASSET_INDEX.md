# G1 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_evaluation_population.*` | 547 primary / 378 clinical-subset validation studies | Methods |
| Table 2 `tables/table2_b0_vs_g1_main_metrics.*` | Finding F1 0.366 (B0) versus 0.383 (G1); difference +0.017 (95% CI -0.024 to +0.053) | Results |
| Table 3 `tables/table3_per_finding_generation_results.*` | Per-finding agreement | Results / Supplementary |
| Table 4 `tables/table4_hallucination_omission_analysis.*` | Hallucination rate 40.7% (B0) versus 25.7% (G1) | Results |
| Table 5 `tables/table5_false_positive_propagation.*` | FP propagation 1.000 versus 0.405; TP retention 1.000 versus 0.718 | Results / Discussion |
| Table 6 `tables/table6_representative_cases.*` | Six representative cases | Results / Supplementary |
| Figure 1 `figures/fig1_finding_precision_recall_f1.*` | Finding agreement B0 versus G1 | Results |
| Figure 2 `figures/fig2_hallucination_omission.*` | Hallucination and omission | Results |
| Figure 3 `figures/fig3_fp_propagation_tp_retention.*` | FP propagation versus TP retention | Results / Discussion |
| Figure 4 `figures/fig4_lexical_metrics.*` | Secondary lexical metrics | Results |
| Figure 5 `figures/fig5_finding_support_provenance.*` | Source of stated findings; support versus truth | Results / Discussion |
| Figure 6 `figures/fig6_error_category_breakdown.*` | Hallucination and omission sources | Discussion |
| `SINGLE_AGENT_RAG_ANALYSIS.md` | Full G1 analysis | Supplementary / internal |
| `MANUSCRIPT_G1_METHODS.md`, `MANUSCRIPT_G1_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_G1.md` | Content for Methodology, Results, Discussion | University report |
| `generated_reports.csv`, `g1_per_study_results.csv` | Every B0 / G1 report with inputs, evidence and references | Supplementary data |
| `GENERATOR_FREEZE.json`, `g1_prompt_frozen.json`, `REPORT_GENERATION_OUTPUT_SCHEMA.json` | Frozen generator and prompt; output schema | Methods / code release / application |
| `g1_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
