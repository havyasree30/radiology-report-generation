# Methods: one-shot locked end-to-end evaluation

## Frozen system and pre-registration

The final system comprised a frozen DenseNet-121 classifier (square-root-weighted binary cross-entropy, F1-optimal operating thresholds with a No Finding consistency rule, Platt calibration), a deterministic routing guard, frozen hybrid retrieval and a frozen single-agent retrieval-augmented generator. The retrieval query consisted of the three highest-probability classifier-positive findings; MiniLM dense retrieval and BM25 were fused by reciprocal-rank fusion (k = 60, depth 100) and the Top-5 reports (K = 5; no diversity reranking, phrase expansion or probability weighting) were given to the local model medgemma1.5:4b (greedy decoding, fixed seed, digest verified) with the frozen prompt. The guard routed each study from the classifier output alone: normal (No Finding positive and no pathology label positive), abnormal (any other study with a positive label) or indeterminate (no label positive; no retrieval, no model call, fixed message). The configuration (SHA-256 cc04996cf8e8f644...), prompt and guard hashes were verified before the test was touched. A protocol listing the populations, metrics, figures, tables, bootstrap, failure handling and failure taxonomy was hashed before any locked-test content was read, and all analysis code was frozen after it had reproduced the stored validation results.

## Test data and populations

The locked IU X-Ray test split contained 578 studies. Each study's first frontal image was classified. Populations followed the validation rules: studies with a frontal image (550), classified studies, studies with a usable reference report (primary end-to-end set, 549) and studies with a usable MeSH-mapped reference (368 in the clinical-finding subset); no study was excluded for its content.

## Outcomes

Classification used the C6 definitions (AUROC, AUPRC, precision, recall, specificity, F1, balanced accuracy, Brier score, log loss and adaptive-bin ECE) against the mapped labels. Retrieval was evaluated at K = 1, 3, 5 and 10 by finding agreement (Jaccard, nDCG, finding coverage, exact-match Hit, duplicate-text rate), with an oracle-query diagnostic. Report quality used finding precision, recall and F1, macro F1, hallucination and omission rates, propagation of classifier false positives, retention of classifier true positives, ROUGE-L, BLEU-4 and an exact-match METEOR variant. The three-state analysis reported the routing distribution, routing by reference state, normal and abnormal recall among decided studies and decision coverage, and the abstention composition. We also measured the share of abnormal-routed studies whose prose reads as normal, the provenance of stated findings, copying of retrieved text, rare-finding behaviour, a frozen ten-category failure taxonomy and runtime. Intervals were 95% percentile intervals from 1,000 study-level bootstrap resamples (seed 42). Validation-to-test differences were computed with the same code on the validation partition.

## Safeguards

Each generation request was cached with its hash and the frozen options, failures would have been recorded and not replaced, and all prompts were verified against their rebuild. No parameter, prompt, threshold, routing rule or model selection was changed after the test was opened.
