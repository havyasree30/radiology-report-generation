# Methods: retrieval baseline and vision-to-retrieval evaluation (R1)

## Data and leakage-safe split

Reports and images came from the locally available IU X-Ray (Open-I) collection: 3,851 studies, 7,466 images (3,818 frontal). The retrieval unit was the study, so every image of a report remained in the same partition. We reused the study-level split fixed before any retrieval work (seed 42; 70% reference corpus, 15% retrieval-validation queries, 15% locked test), and verified that corpus, validation and test studies are disjoint and that no study's images span two partitions. The corpus held 2,675 training studies; 578 validation studies served as queries. The locked retrieval-test partition was not used. Retrieval text was the cleaned Findings plus Impression of each corpus report (markup, repeated headings and de-identification placeholders removed; indication and comparison excluded); no text was rewritten.

## Finding representation

Independent of the retrieval text, each study's MeSH indexing was mapped to the 14-class vocabulary of the frozen classifier using explicit rules; ambiguous terms (for example pulmonary congestion) and terms outside the vocabulary were not mapped. A study's finding set was its mapped abnormal findings, or {No Finding} if indexed normal without a mapped abnormality; 403 of 578 validation studies had a usable set.

## Retrievers

Dense retrieval used the unmodified sentence-transformers/all-MiniLM-L6-v2 encoder (embedding dimension verified as 384) with L2-normalised vectors in a FAISS flat inner-product index, equivalent to cosine similarity. The lexical baseline was Okapi BM25 (k1 = 1.5, b = 0.75) without stop-word removal or stemming. Neither retriever was tuned. The top ten reports were retrieved per query, and we confirmed that no query study could be retrieved.

## Query sources

Three query policies were compared on the same studies. The oracle query listed the study's true mapped findings. Classifier queries were built from the frozen classifier applied, unchanged, to the first frontal image of each study (frozen preprocessing, checkpoint, Platt calibration, operating thresholds and No Finding rule), which is an external application to a different dataset. The all-positive policy used every positive finding; the precision-aware policy used only positives that passed a per-class gate derived on the CheXpert validation set (candidate gates from calibrated-probability quantiles, F0.5 criterion, adopted only when the cross-fitted gain was reliably above zero), without changing any classifier decision. If nothing remained, the fixed normal query was used when No Finding was positive; otherwise the policy fell back to the strongest positive finding, flagged as such, and studies with no output at all were recorded as empty.

## Evaluation

Relevance was judged by independent finding agreement rather than the retriever's similarity score. For each retrieved report we computed the Jaccard similarity between its finding set and the study's true set (No Finding is a single element that never overlaps abnormal findings), reported as the mean over the top K = 1, 3, 5 and 10, and graded nDCG@K using the Jaccard as gain. Hit@K and reciprocal rank were computed with an exact finding-set match as the only binary criterion. We also report the coverage of true findings by the union of the top K, the share of retrieved reports without overlap, and the share duplicating another retrieved report's text. Confidence intervals came from 1000 bootstrap resamples of queries (seed 42), with paired differences between policies and retrievers. The primary set contained the 358 studies with a usable truth, a frontal image and a non-empty classifier query.
