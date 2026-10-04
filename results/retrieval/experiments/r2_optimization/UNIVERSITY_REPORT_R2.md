# University report content: R2 (query construction, hybrid retrieval, diversity)

_Insert-ready material generated from the R2 artifacts. Retrieval-validation data only; the locked retrieval-test split is unopened; the classifier is unchanged._

## Methodology

### Query construction

The retrieval stage receives the findings predicted by the frozen classifier. Four ways of turning them into a query were compared on the 358-study validation benchmark of the previous stage. (Q0) The finding names, as before. (Q1) A fixed short radiology phrase for each finding, defined once before evaluation. (Q2) A weighted combination of the embeddings of the finding phrases, using uniform weights, the calibrated probability, or the margin above the operating threshold; the calibrated probability is used only as a weight and never changes a classification. (Q3) Only the N most probable positive findings (N = 1, 2, 3), without padding. A fixed normal-study phrase is used when the classifier predicts no finding. Options were compared in stages and a new option replaced the current one only if the paired bootstrap interval of the Jaccard@3 difference excluded zero on the favourable side.

### Hybrid retrieval

The dense retriever (MiniLM embeddings, cosine similarity) and the BM25 lexical retriever were combined by Reciprocal Rank Fusion: each report receives the sum of 1/(60 + rank) over the two rankings, restricted to each ranker's top 100, with ties broken by corpus order. The dense and lexical rank of every fused report are stored so that each result can be traced to its sources.

### Diversity re-ranking and Top-K

Maximal Marginal Relevance was applied to the 30 best candidates to reduce near-duplicate references: at each step the report with the best trade-off between relevance and dissimilarity to the already chosen reports is added (lambda 0.5, 0.7, 0.9). It was to be adopted only if exact duplicates fell reliably without loss of relevance or coverage. The number of references K was chosen from 1, 3, 5, 10 by a rule declared in advance that rewards a reliable gain in coverage of the true findings while limiting redundancy and context length.

## Results and Discussion

### Query-policy comparison

On the validation benchmark, finding names with dense retrieval reached a Jaccard@3 of 0.514 (95% CI 0.465 to 0.560). Neither phrase expansion (-0.012 (95% CI -0.024 to +0.001)) nor probability weighting (differences of -0.000 to +0.009) improved retrieval; the Top-3 rule gave a difference of +0.003 (95% CI +0.000 to +0.007), which is negligible in practice, while restricting the query to one finding lowered coverage (Table 1, Figure 1). The simple weight max(p - 0.5, 0) was not used because 74.0% of the positive findings have p below 0.5.

### Retrieval-method comparison

Dense retrieval, BM25 and hybrid fusion reached Jaccard@3 of 0.518, 0.533 and 0.544; the hybrid was reliably better than dense retrieval (+0.026 (95% CI +0.016 to +0.038)) and was selected (Table 2, Figure 2). MMR did not reduce exact duplicate reports and was not adopted: the corpus consists largely of near-identical templated reports, so the similarity penalty cannot separate exact duplicates from similar ones (Table 4, Figure 4).

### Top-K analysis

Coverage of the true findings rose from 0.582 at K=1 to 0.647 at K=3, 0.748 at K=5 and 0.792 at K=10, while Jaccard@K stayed nearly constant and the duplicate-text rate increased from 0.333 (K=3) to 0.457 (K=10). Starting at K=1, the rule advanced to K=3 (coverage gain +0.065 (95% CI +0.045 to +0.090)), then to K=5 (gain +0.101 (95% CI +0.071 to +0.129)) and stopped there: K=10 added a coverage gain of 0.044 over K=5 (95% CI 0.026 to 0.064), below the declared 0.05 minimum. K = 5 is therefore the provisional number of references for generation, to be confirmed on the untouched retrieval test split and on generated reports (Table 3, Figure 3).

### Error analysis and remaining gap

The selected pipeline narrowed the gap to oracle queries from 0.291 to 0.261 Jaccard@3 (Figure 5), but 98.9% of the remaining top-1 failures stem from the classifier's finding query, mainly normal-versus-abnormal disagreements (77) and missing or extra findings (102); only 2 were retriever failures despite a correct query (Table 6, Figure 6). Improving retrieval further is therefore unlikely to help unless the classifier's findings improve.

### Limitations

Selection and evaluation used the same validation queries, so gains are optimistic; most differences between query policies are small; the reference annotations come from mapped MeSH terms with several rare findings; the Top-K rule constants are design choices; and the retrieval test split has not been evaluated.
