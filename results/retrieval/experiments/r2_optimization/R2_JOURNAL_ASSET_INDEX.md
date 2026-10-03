# R2 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_query_policy_comparison.*` | Expansion, weighting and Top-N did not reliably improve dense Jaccard@3 (Q0 0.514) | Results |
| Table 2 `tables/table2_retriever_comparison.*` | Hybrid RRF 0.544 versus dense 0.518 Jaccard@3 | Results |
| Table 3 `tables/table3_top_k_comparison.*` | Coverage@K 0.647 (K=3), 0.748 (K=5), 0.792 (K=10); provisional K=5 | Results / Discussion |
| Table 4 `tables/table4_diversity_mmr_analysis.*` | MMR did not reduce exact duplicates and was not adopted | Results |
| Table 5 `tables/table5_final_r2_candidate_configuration.*` | Candidate retrieval configuration | Methods / Supplementary |
| Table 6 `tables/table6_primary_failure_analysis.*` | 98.9% of top-1 failures are upstream | Results / Discussion |
| Figure 1 `figures/fig1_query_policy_comparison.*` | No query-policy change closes the oracle gap | Results |
| Figure 2 `figures/fig2_dense_bm25_hybrid.*` | Hybrid improves on dense retrieval | Results |
| Figure 3 `figures/fig3_performance_across_k.*` | Coverage gains versus K; provisional K=5 | Results / Discussion |
| Figure 4 `figures/fig4_relevance_vs_redundancy.*` | Relevance versus redundancy of retrieval settings | Results / Supplementary |
| Figure 5 `figures/fig5_oracle_vs_final_pipeline.*` | Oracle gap before and after R2 | Results |
| Figure 6 `figures/fig6_failure_categories.*` | Failures are dominated by upstream errors | Results / Discussion |
| `RETRIEVAL_OPTIMIZATION_ANALYSIS.md` | Full R2 analysis | Supplementary / internal |
| `MANUSCRIPT_R2_METHODS.md`, `MANUSCRIPT_R2_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_R2.md` | Content for Methodology and Results and Discussion | University report |
| `R2_RETRIEVAL_CANDIDATE_CONFIG.json`, `r2_selection_protocol.json`, `finding_query_expansion.json` | Candidate configuration, pre-declared protocol, fixed expansion mapping | Methods / Supplementary / code release |
| `per_query_metrics_all_configs.csv`, `r2_paired_comparisons.csv` | All per-query metrics; all paired comparisons | Supplementary data |
| `r2_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
