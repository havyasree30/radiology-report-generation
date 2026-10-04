# Table captions (R1)

**Table 1. IU X-Ray dataset and retrieval split summary.** Dataset inventory, study-level split sizes (reference corpus, retrieval-validation queries, locked test), usable query sets, split-overlap checks and embedding model.

**Table 2. Dense versus lexical baseline.** Oracle-query retrieval quality of MiniLM dense retrieval, BM25 and the random expectation on the primary paired set (finding Jaccard@K, nDCG, Hit@3, MRR@10).

**Table 3. Oracle versus classifier queries.** Retrieval quality for the oracle query, the all-positive classifier query and the precision-aware gated classifier query with both retrievers, with mean query length, findings per query, and empty and fallback query rates (over the eligible studies with classifier output; rates over all frontal studies are in Table 5 and the domain-shift summary).

**Table 4. Performance across K.** Jaccard@K, nDCG@K, Hit@K, union coverage, zero-overlap rate and duplicate-text rate at K = 1, 3, 5 and 10 for each retriever and query policy.

**Table 5. Classifier query and domain-shift summary.** Per-finding frequency in the mapped IU truth and in the frozen classifier's predictions, descriptive precision, recall and F1, retrieval gate, and positives removed by the gate.

**Table 6. Precision-aware retrieval gates.** Validation-derived gates on the calibrated probability, with precision and recall before and after gating, and the cross-fitted F0.5 gain with its bootstrap interval. Gates are query-building rules, not classifier thresholds.
