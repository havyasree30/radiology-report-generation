# Methods: single-agent retrieval-augmented report generation (G1)

## Pipeline and frozen components

Each chest radiograph was processed by the frozen classifier (DenseNet-121 trained with a square-root-weighted loss, class-specific F1-optimal thresholds, a rule that suppresses No Finding when a pathology is positive, and Platt-calibrated probabilities). The positive findings formed a query of at most the three most probable finding names, and the five best reports were retrieved from the reference corpus with the frozen hybrid retriever (dense MiniLM and BM25 fused by Reciprocal Rank Fusion). Classifier, thresholds, calibration, retrieval configuration and Top-5 were not changed. All experiments used the retrieval-validation studies; the locked test partition was not opened.

## Evaluation populations

The primary report-generation set contained every validation study with a frontal image and a non-empty reference report (547 studies), including studies with empty or imperfect classifier output (49 had no classifier output and no retrieval context). The clinical-finding subset (378 studies) additionally had a usable mapped finding set derived from MeSH terms. B0 and G1 were compared on identical studies.

## Systems

The rule-based baseline (B0) wrote one sentence per classifier-positive finding with canonical terms and no retrieved context; it wrote a normal report for No Finding and an explicit indeterminate report when the classifier produced neither a pathology nor No Finding. The single-agent system (G1) made one call per study to a local language model (medgemma1.5:4b, Q4_K_M, served by Ollama 0.35.1; greedy decoding with temperature 0, fixed seed, 8192-token context, at most 400 new tokens, no tools or web access). Its input contained the classifier positives with calibrated probabilities and the No Finding state; the Findings and Impression of the five retrieved reports with rank and score information; and a deterministic evidence table giving, for each classifier-positive finding, its probability and the number of retrieved reports whose mapped findings contain it. The reference report and the reference findings were structurally excluded from the input and verified to be absent. The prompt instructed the model to use only the supplied evidence, to give more weight to findings supported by several retrieved reports, to omit or hedge unsupported classifier positives, not to copy retrieved text, and to add no history, measurements, comparisons or recommendations. The prompt was frozen (SHA-256 recorded) after an implementation smoke test on four studies, and generation was cached per study.

## Outcome measures

Findings were extracted from report text with an assertion-aware rule-based lexicon (negated mentions excluded); any affirmative mention counted as stated (primary) and non-hedged mentions as definite (sensitivity). Against the mapped reference we computed micro precision, recall and F1, per-finding results, hallucinated findings (stated but absent from the reference) and omitted findings (present in the reference but not stated). The primary comparison metrics were finding F1, hallucination rate, propagation of classifier false positives (classifier-positive, reference-negative findings mentioned in the report) and retention of classifier true positives. Retrieval grounding classified each stated finding as supported by the classifier only, retrieval only, both or neither. Normal/abnormal consistency, report length, repeated sentences and copying of retrieved text were also measured. Secondary lexical metrics were BLEU-1, BLEU-4, ROUGE-L and an exact-match METEOR variant.

## Statistics

Differences between systems were estimated with a study-level paired bootstrap (1,000 resamples, fixed seed) and are reported with 95% percentile intervals. The bootstrap quantifies uncertainty only and was not used for tuning.
