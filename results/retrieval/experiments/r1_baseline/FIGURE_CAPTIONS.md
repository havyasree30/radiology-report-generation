# Figure captions (R1)

Retrieval-validation queries only (358 primary paired studies); the locked retrieval-test split was not used. Relevance = finding agreement with the study's true (MeSH-mapped) finding set. Classifier queries come from the frozen classifier applied to the IU frontal image without any tuning.

**Figure 1. Dense versus lexical retrieval across K.** Mean finding Jaccard@K for dense (MiniLM + FAISS, solid) and lexical (BM25, dashed) retrieval, with oracle finding queries (left) and all-positive frozen-classifier queries (right). Error bars: 95% bootstrap intervals over queries. Dotted line: expected value for random retrieval.

**Figure 2. Oracle versus classifier queries.** Jaccard@3 (left) and nDCG@3 (right) for oracle, all-positive classifier and precision-aware gated classifier queries with both retrievers; error bars are 95% bootstrap intervals. The comparison between gated and all-positive queries is a downstream query-policy comparison and does not change classifier performance.

**Figure 3. Retrieval performance versus K for the three query policies.** Finding Jaccard@K (left), union coverage of the true findings by the top K (centre) and the share of retrieved reports whose text duplicates another retrieved report (right); colours: query policy; solid, dense; dashed, lexical.

**Figure 4. Distribution of Top-3 finding agreement.** Per-query mean Jaccard of the three highest-ranked reports (dense retrieval) for each query policy, as a percentage of queries; annotations give the mean and the share of queries with zero overlap.

**Figure 5. Frozen-classifier behaviour on IU X-Ray validation studies.** Left: predicted frequency of each finding against its frequency in the mapped IU truth. Right: composition of the query outcomes (pathology findings, No Finding query, strongest-positive fallback, empty) for the all-positive and precision-aware policies. External application of the classifier; nothing was tuned on IU X-Ray.
