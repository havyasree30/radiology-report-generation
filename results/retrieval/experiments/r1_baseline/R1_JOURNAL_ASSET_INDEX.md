# R1 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_dataset_and_split_summary.*` | 2,675 / 578 / 578 corpus / validation / locked-test studies, zero overlaps | Methods |
| Table 2 `tables/table2_dense_vs_lexical_baseline.*` | Oracle Jaccard@3 0.805 dense, 0.787 lexical, random 0.213 | Results |
| Table 3 `tables/table3_oracle_vs_classifier_queries.*` | Classifier queries Jaccard@3 0.514-0.533 versus oracle about 0.805 | Results |
| Table 4 `tables/table4_performance_across_k.*` | Jaccard flat across K; coverage rises with K | Results / Discussion |
| Table 5 `tables/table5_classifier_query_domain_shift.*` | Per-finding over- and under-prediction on IU | Results / Supplementary |
| Table 6 `tables/table6_precision_aware_gates.*` | Gates for 12 of 13 findings | Methods / Supplementary |
| Figure 1 `figures/fig1_dense_vs_lexical_across_k.*` | No reliable dense-lexical difference with oracle queries | Results |
| Figure 2 `figures/fig2_oracle_vs_classifier_queries.*` | Large oracle-to-classifier drop; gating does not close it | Results |
| Figure 3 `figures/fig3_performance_vs_k.*` | Top-K behaviour of agreement, coverage and redundancy | Results / Discussion |
| Figure 4 `figures/fig4_top3_agreement_distribution.*` | Bimodal top-3 agreement for classifier queries | Results |
| Figure 5 `figures/fig5_domain_shift_query_quality.*` | Classifier domain shift and query-status composition | Results / Discussion |
| `RETRIEVAL_BASELINE_ANALYSIS.md` | Full R1 analysis | Supplementary / internal |
| `MANUSCRIPT_R1_METHODS.md`, `MANUSCRIPT_R1_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_R1.md` | Content for Chapters 3, 4 and Results and Discussion | University report |
| `precision_aware_query_gates.json`, `classifier_query_comparison.csv` | Gate definitions; policy comparison | Methods / Results |
| `retrieval_results_top10.csv.gz`, `per_query_metrics.csv` | Full retrieval outputs and per-query metrics | Supplementary data |
| `r1_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
