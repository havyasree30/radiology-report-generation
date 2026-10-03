# Methods: query construction, hybrid retrieval and diversity re-ranking (R2)

## Evaluation set and protocol

R2 reused the primary paired validation set of the retrieval baseline (358 studies with a usable finding set, a frontal image and a non-empty classifier query). The locked retrieval-test partition was not used, and the classifier, its thresholds and its calibration were not changed. The baseline results (oracle and classifier queries; dense and BM25 retrieval) were reproduced exactly before any new method was evaluated. Before evaluation we fixed a selection protocol: methods were compared in stages, and a challenger replaced the incumbent only if the lower bound of the paired bootstrap 95% confidence interval of the Jaccard@3 difference exceeded zero (1000 resamples of queries, one shared set of resampling indices, seed 42). The same finding-agreement metrics as in the baseline were used: Jaccard@K and graded nDCG@K against the study's true finding set, union coverage of true findings, and exact-match Hit@K.

## Query construction

The baseline query lists finding names (Q0). Q1 replaced each name by one fixed short radiology phrase from a mapping written before evaluation. Q2 weighted the per-finding phrase embeddings and re-normalised their sum, using uniform weights, the calibrated probability, or the margin of the probability above the frozen operating point; the simple form max(p - 0.5, 0) was not used because most frozen positives have p < 0.5. Q3 kept only the N most probable positive findings (N = 1, 2, 3), never padding. Calibrated probabilities served only as query weights or for ordering; no binary decision was altered. Studies predicted as No Finding received a fixed normal phrase chosen among three formulations on oracle-normal validation studies. Staged selection first compared Q0 with Q1, then the weightings, then Top-N.

## Retrievers

Dense retrieval used the unmodified all-MiniLM-L6-v2 encoder with a flat inner-product index over normalised vectors; the lexical baseline was Okapi BM25 without stop-word removal. Hybrid retrieval combined the two rankings by Reciprocal Rank Fusion, score = sum of 1/(60 + rank) over the top 100 of each ranker, with ties broken by corpus order, and the dense and BM25 rank of every fused result was stored. All rankings used one deterministic tie rule (score, then corpus order).

## Diversity re-ranking and Top-K

Maximal Marginal Relevance re-ranked the top 30 candidates of the selected pipeline (relevance min-max normalised within the pool; similarity the cosine of report embeddings; lambda 0.5, 0.7, 0.9). MMR was adopted only if, at both K = 3 and K = 5, the duplicate-text rate fell reliably, Jaccard and nDCG showed no reliable loss and coverage did not fall; the largest eligible lambda was to be chosen. Redundancy was measured by the exact duplicate-text rate, the number of distinct report texts and the mean pairwise cosine of retrieved reports. The provisional number of references K in {1, 3, 5, 10} was chosen by a declared rule: move to a larger K only if union coverage rose reliably and by at least 0.05, Jaccard@K fell by at most 0.03, the duplicate-text rate stayed at most 0.50 with more distinct reports, and the context stayed within 1,000 words; otherwise the smaller K was kept.

## Sensitivity and failure analysis

As a sensitivity analysis the primary set was restricted to queries whose own report text has no exact copy in the corpus. Failures (top-1 report not an exact finding-set match) were assigned by fixed rules to normal/abnormal disagreement, other upstream classification or query error, corpus limitation, duplicate/template issue, terminology mismatch, or retriever failure despite a good query.
