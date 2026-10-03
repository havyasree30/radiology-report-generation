# Table captions (R2)

**Table 1. Query-policy comparison.** Dense retrieval on the primary paired set: Jaccard@3 with 95% interval, nDCG@3, coverage@3, query length, findings per query, and the paired Jaccard@3 difference from Q0.

**Table 2. Retriever comparison.** Dense, BM25 and hybrid RRF with the selected query: Jaccard@1/3/5, nDCG@3, coverage@3/5, Hit@3 and the paired Jaccard@3 difference from dense retrieval.

**Table 3. Top-K comparison.** Jaccard, nDCG, Hit, union coverage (with the random-retrieval expectation), zero-overlap rate, duplicate-text rate, distinct report texts, mean context length and oracle coverage at K = 1, 3, 5, 10 for the selected pipeline.

**Table 4. Diversity / MMR analysis.** Standard hybrid retrieval versus MMR re-ranking (pool 30, lambda 0.5/0.7/0.9): relevance, coverage, duplicate-text rate, distinct texts, pairwise cosine, the paired duplicate-rate difference at K=5 and the adoption decision.

**Table 5. R2 candidate configuration.** Parameters of the candidate retrieval pipeline (not frozen; the retrieval test split is unopened).

**Table 6. Primary failure analysis.** Failures (top-1 report not an exact finding-set match) by category for the R1 dense baseline and the selected R2 pipeline.
