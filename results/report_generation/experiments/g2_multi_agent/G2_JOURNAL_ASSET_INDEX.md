# G2 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_g2_architecture_configuration.*` | Three-agent architecture, frozen configuration, prompt hashes | Methods |
| Table 2 `tables/table2_g1_vs_g2_main_metrics.*` | Finding F1 G1 0.383 vs G2 0.382 | Results |
| Table 3 `tables/table3_paired_bootstrap_differences.*` | G2 minus G1: F1 -0.001 (95% CI -0.038 to +0.034), FP propagation +0.426 (95% CI +0.358 to +0.490), TP retention +0.214 (95% CI +0.138 to +0.287) | Results |
| Table 4 `tables/table4_agent1_evidence_analysis.*` | Agent 1 support status vs reference | Results |
| Table 5 `tables/table5_agent3_intervention_analysis.*` | Critic modified 43.7% of reports; retrieval-only funnel | Results / Discussion |
| Table 6 `tables/table6_per_finding_results.*` | Per-finding retention and P/R/F1 | Supplementary |
| Table 7 `tables/table7_latency_and_copying_analysis.*` | G2 4.8 times G1 latency; copying 0.0% vs 29.5% | Results / Discussion |
| Table 8 `tables/table8_supplementary_normal_abnormal_and_rare_findings.*` | Normal/abnormal and rare-finding behaviour | Supplementary |
| Table 9 `tables/table9_supplementary_stratified_by_classifier_state.*` | G2 minus G1 within classifier states | Supplementary |
| Figure 1 `figures/fig1_g1_vs_g2_finding_precision_recall_f1.*` | Finding agreement | Results |
| Figure 2 `figures/fig2_hallucination_and_omission.*` | Hallucination and omission | Results |
| Figure 3 `figures/fig3_fp_propagation_and_tp_retention.*` | FP propagation versus TP retention | Results |
| Figure 4 `figures/fig4_normal_vs_abnormal_recall.*` | Normal/abnormal behaviour | Results |
| Figure 5 `figures/fig5_agent1_evidence_support_categories.*` | Agent 1 evidence categories and retrieval-only funnel | Results |
| Figure 6 `figures/fig6_critic_actions_and_copying_reduction.*` | Critic actions and copying | Results / Discussion |
| `MULTI_AGENT_RAG_ANALYSIS.md` | Full G2 analysis | Supplementary / internal |
| `MANUSCRIPT_G2_METHODS.md`, `MANUSCRIPT_G2_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_G2.md` | Content for Methodology, Results, Discussion | University report |
| `g2_prompts_frozen.json`, `g2_prompts_frozen.txt`, `g2_generator_metadata.json` | Frozen prompts, schemas, hashes, generator metadata | Methods / code release |
| `g2_generated_reports.csv`, `agent_outputs/` | Every agent's output per study | Supplementary data |
| `g2_sent_messages_sha256.csv`, `g2_leakage_checks.json` | Hashes of every sent message and leakage checks | Supplementary |
| `g2_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
