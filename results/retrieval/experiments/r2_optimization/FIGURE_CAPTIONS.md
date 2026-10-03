# Figure captions (R2)

Retrieval-validation queries only (358 primary paired studies); the locked retrieval-test split was not used. Relevance = finding agreement with the study's true (MeSH-mapped) finding set. Error bars: 95% bootstrap intervals over queries. Classifier queries come from the frozen classifier without any tuning.

**Figure 1. Query-policy comparison (dense retrieval).** Finding Jaccard@3 (left) and union coverage of the true findings at K=3 (right) for Q0 finding names (grey dotted line), Q1 expanded phrases, Q2 weighted queries (uniform, probability, margin), Q3 Top-N findings and the R1 precision-aware gated query. The blue dashed line marks the oracle reference.

**Figure 2. Dense, BM25 and hybrid retrieval.** Jaccard@K, nDCG@K and union coverage@K for K = 1, 3, 5, 10 with the selected query (Q0, Top-3 findings). Hybrid is Reciprocal Rank Fusion (k = 60, depth 100) of the dense and BM25 rankings.

**Figure 3. Performance across K for the selected pipeline.** Jaccard@K, union coverage@K and duplicate-text rate@K for the classifier-query pipeline (solid) and the same pipeline with oracle queries (dashed); dotted line: random-retrieval expectation; the green band marks the provisional K.

**Figure 4. Relevance versus redundancy.** Jaccard (left) and union coverage (right) against the mean pairwise cosine similarity among the retrieved reports at K=3 (top) and K=5 (bottom) for dense, BM25, hybrid and hybrid with MMR (lambda 0.5, 0.7, 0.9). Lower similarity indicates less redundancy.

**Figure 5. Oracle versus classifier-query pipelines.** Jaccard@K and union coverage@K for oracle and frozen-classifier queries with the R1 dense baseline (dashed) and the selected R2 pipeline (solid).

**Figure 6. Failure categories.** Share of the 358 queries by top-1 outcome for the R1 dense baseline and the selected R2 pipeline; a failure is a top-1 report that is not an exact finding-set match, assigned to one exclusive category by fixed rules.
