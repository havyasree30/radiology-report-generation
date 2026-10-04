# Results: query construction, hybrid retrieval and diversity re-ranking (R2)

## Baseline reproduction and query construction

The retrieval-baseline results were reproduced exactly (2,238 query rows; largest metric difference 1e-16). On the 358-study primary set, Q0 finding names with dense retrieval gave a Jaccard@3 of 0.514 (95% CI 0.465 to 0.560). Phrase expansion (Q1) did not help: 0.503 (95% CI 0.455 to 0.548), a paired difference of -0.012 (95% CI -0.024 to +0.001) (Table 1, Figure 1). Probability weighting did not help either: uniform, probability and margin weights differed from Q0 by -0.000, +0.005 and +0.009, all with intervals including zero. Limiting the query to the most probable findings changed little: Top-1 and Top-2 differed from all positives by +0.000 and -0.002, and Top-3 by +0.003 (95% CI +0.000 to +0.007), the only difference whose interval excluded zero by the protocol rule, although it is too small to matter in practice; Top-1 reduced coverage@3 to 0.625 from 0.646. The normal phrase `no acute abnormality` was kept after two alternatives did not improve on it.

## Retriever comparison

With the Top-3 finding-name query, Jaccard@3 was 0.518 (95% CI 0.469 to 0.562) for dense retrieval, 0.533 (95% CI 0.485 to 0.576) for BM25 and 0.544 (95% CI 0.495 to 0.590) for hybrid RRF (Table 2, Figure 2). Hybrid fusion improved on dense retrieval by +0.026 (95% CI +0.016 to +0.038), whereas BM25 minus dense was +0.016 (95% CI -0.000 to +0.033). Union coverage@5 was 0.748 for hybrid, 0.676 for dense and 0.663 for BM25. Hybrid retrieval was therefore selected.

## Diversity re-ranking

MMR was not adopted (Table 4, Figure 4). At K=5 the duplicate-text rate changed by -0.0067, -0.0022 and -0.0022 for lambda 0.5, 0.7 and 0.9, and at K=3 by at most 0.0019; the number of distinct report texts in the top five stayed near 3.0. Relevance did not collapse, and at K=3 lambda 0.7 and 0.9 raised Jaccard@3 by +0.019 and +0.022, but this exploratory gain vanished at K=5.

## Top-K selection

For the selected pipeline, coverage@K was 0.582, 0.647, 0.748, 0.792 at K = 1, 3, 5 and 10, against random-retrieval expectations of 0.225, 0.491, 0.631, 0.781, while Jaccard@K stayed between 0.541 and 0.557 (Table 3, Figure 3). K=5 added +0.101 (95% CI +0.071 to +0.129) coverage over K=3, whereas K=10 added only 0.044 over K=5, below the declared 0.05 minimum. The duplicate-text rate was 0.333 at K=3 and 0.404 at K=5, and the context grew from 88 to 148 words. K = 5 was selected provisionally, before any retrieval-test evaluation.

## Oracle-to-classifier gap

The selected pipeline improved Jaccard@3 over the baseline classifier query by +0.029 (95% CI +0.019 to +0.042) (dense Q0 0.514, selected 0.544), and coverage@5 by +0.072 (95% CI +0.037 to +0.106) (Figure 5). The gap to the oracle dense reference fell from 0.291 to 0.261 at K=3 and was 0.252 at K=5. The same pipeline with oracle queries reached a Jaccard@3 of 0.819.

## Failure analysis and sensitivity

Of 358 queries, 181 had a top-1 report that was not an exact finding-set match (200 for the baseline) (Table 6, Figure 6). 77 failures arose from normal/abnormal disagreement and 102 from other classification or query errors, together 98.9% of failures; 2 were retriever failures despite a correct query, and none was classified as corpus limitation, duplicate/template or terminology mismatch. Excluding the 100 queries whose report text has an exact corpus copy preserved the ordering of configurations (Spearman 0.895).

## Other relevance and cost measures

Against the baseline classifier query with dense retrieval, the selected pipeline raised nDCG@3 from 0.518 to 0.549 and Hit@3 (an exact finding-set match among the top three) from 0.516 to 0.536, while the share of retrieved top-three reports without any finding overlap changed from 0.379 to 0.347. The mean query length was 2.84 words with 1.40 findings per query, compared with 2.98 words and 1.50 findings for the baseline, and 5.34 words for the expanded query. With oracle queries the selected pipeline reached a Hit@3 of 0.883.

## Interpretation

Hybrid fusion was the only change with a clear benefit (the Top-3 query rule passed the protocol test but gained only about 0.003); query expansion, weighting and diversity re-ranking were not adopted. Most remaining failures originate in the classifier's finding query. Selection and evaluation used the same validation queries, so the gains are optimistic, and the retrieval-test partition has not been evaluated.
