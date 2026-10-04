# University report content: R1 (retrieval baseline)

_Insert-ready material generated from the R1 artifacts. Upstream classifier frozen at commit `55ba4bb` (C5) / C6 freeze manifest `2026-10-03T05:00:47+00:00`; R1 uses retrieval-validation data only._

## Chapter 3 — Materials and Methods

### 3.x IU X-Ray (Open-I) dataset

The retrieval component uses the Indiana University chest X-ray collection (Open-I) in the form distributed as CSV files and normalised PNG images, held locally and treated as read-only. It contains 3,851 radiology reports, each linked to one or more images (7,466 images listed: 3,818 frontal and 3,648 lateral; 3,405 studies have more than one image and 162 have no frontal image). Each report may contain Findings and Impression sections (3,331 contain both, 25 neither), together with Indication and Comparison fields, which are metadata and are not used for retrieval. The data also include MeSH and Problems annotations (121 distinct MeSH terms; 1,379 studies indexed as normal). The files carry no patient identifier, so independence can only be guaranteed at the level of the report.

### 3.x Report preprocessing and study-level split

The retrieval unit is the report (study), never a single image, so that the frontal and lateral images of one report cannot fall in different partitions. The pre-defined split (seed 42; 70% / 15% / 15%, stratified on normal versus abnormal indexing) gives a reference corpus of 2,675 studies, 578 retrieval-validation studies used as queries during development, and 578 locked test studies that remain unused. The checks confirm no overlap between corpus, validation and test studies and no study with images in more than one partition. For retrieval, each corpus report is represented by its cleaned Findings followed by its Impression (or the single available section); markup, repeated headings and the de-identification placeholder are removed, while the original text is stored unchanged. No report text is rewritten or generated.

### 3.x Finding representation of IU studies

To evaluate retrieval independently of the text being searched, the MeSH annotations of each study were mapped to the 14 observation names of the frozen chest X-ray classifier. Direct mappings follow the project's concept lexicon; two rules depend on the MeSH qualifier (an enlarged cardiac shadow is cardiomegaly; pleural thickening is the 'pleural other' observation); ambiguous terms (for example pulmonary congestion) are kept apart and not mapped; terms outside the vocabulary are not mapped. A study's finding set is its mapped abnormal findings, or 'No Finding' for normal studies; 403 of the 578 validation studies have a usable set. No IU term corresponds to enlarged cardiomediastinum.

## Chapter 4 — Methodology

### 4.x Retrieval architecture

The retrieval stage sits between the classifier and the (future) generation stage. Two retrievers were implemented over the reference corpus. The dense retriever encodes each report with the pre-trained all-MiniLM-L6-v2 sentence encoder (verified embedding dimension 384, not fine-tuned), normalises the vectors to unit length and searches a FAISS flat inner-product index, which equals cosine similarity for unit vectors. The lexical baseline is Okapi BM25 (k1 = 1.5, b = 0.75) over lowercase tokens. Both return the ten highest-ranked reports; no hyper-parameter was tuned.

### 4.x Query generation from the classifier

The classifier output for an image is a set of independent findings. Queries are built deterministically from finding names in a fixed order. Three policies are compared: (1) an oracle query from the true mapped findings, which measures retrieval independently of vision errors; (2) an all-positive query from every finding the frozen classifier calls positive; and (3) a precision-aware query that keeps only positives passing a per-class gate on the calibrated probability. If no abnormal finding remains, a fixed normal-study query is used when 'No Finding' is positive; otherwise the strongest positive finding is used as a flagged fallback, and an output with no positive finding is recorded as an empty query. The classifier, its thresholds, its calibration and its 'No Finding' rule are not changed.

The gates are derived from the classifier's validation outputs: candidate gates are quantiles of each class's positive predictions, the criterion is the precision-weighted F0.5 score, and a gate is adopted only if its cross-validated improvement is reliably above zero. Gates were adopted for 12 of 13 findings and not for Pleural Other.

### 4.x Evaluation protocol

Retrieval is judged by agreement between the finding set of each retrieved report and the true finding set of the query study (Jaccard similarity, graded nDCG, and Hit@K / MRR with an exact finding-set match as the only binary criterion), not by the retriever's own similarity score. Confidence intervals use bootstrap resampling of queries.

## Results and Discussion

### Retrieval baseline

With oracle queries both retrievers recover the true findings well (Jaccard@3 0.805 dense and 0.787 lexical, against 0.213 for random retrieval) and the difference between them is not reliable (paired difference +0.018, 95% interval -0.002 to +0.038). A simple lexical baseline is therefore competitive with a dense encoder on this short, templated corpus (Table 2, Figure 1).

### Vision-to-retrieval and query gating

When the query comes from the frozen classifier applied to the IU frontal image, Jaccard@3 falls to 0.514 (dense) and 0.533 (lexical); about 37.9% of retrieved reports share no finding with the study. The error analysis attributes the non-good cases mainly to the classifier query (n = 190 of 200), so the bottleneck is upstream of retrieval (Table 3, Figures 2 and 4). The classifier is applied here to a different dataset without any tuning, and its behaviour shifts: it predicts 'No Finding' for 48.8% of studies, over-predicts Fracture and under-predicts Support Devices relative to the mapped truth (Table 5, Figure 5). Precision-aware gating shortened the queries and removed 251 of 448 positive findings, yet no consistent improvement was found (Jaccard@3 difference +0.006 (95% CI -0.008 to +0.021) dense, -0.003 (95% CI -0.015 to +0.010) lexical), and 18.7% of studies needed the fallback; gating changes only the query, not classifier performance.

### Choice of the number of references

With oracle queries the finding Jaccard changes little across K (range 0.038 for dense; dense 0.797, 0.805, 0.809, 0.771 at K = 1, 3, 5, 10), while the union coverage of the true findings rises from 0.855 (K=1) to 0.964 (K=3), 0.971 (K=5) and 0.977 (K=10); most of the gain is reached by K=3. With all-positive classifier queries the Jaccard range is 0.012, but coverage keeps rising (0.574, 0.646, 0.676, 0.814), and the share of retrieved reports with zero finding overlap stays near 37.9% at every K. Redundancy grows with K: with dense oracle retrieval 34.8% of the top-3 and 47.3% of the top-10 reports have text identical to another retrieved report (templated normal reports).

Top-3 is a defensible default for oracle-quality queries (most of the achievable coverage, no Jaccard loss) but it is not shown to be optimal: for classifier-driven queries coverage still improves at larger K, redundancy grows with K while the share of retrieved reports without finding overlap stays roughly constant, and the choice of K is deferred to R2 where it can be judged on generated reports.

### Limitations

The reference annotations come from mapped MeSH terms, so some studies cannot be evaluated and several findings have very few positives; finding agreement is a proxy for usefulness, not a clinical judgement; the corpus contains many near-identical normal reports; the classifier was not adapted to IU X-Ray; and only validation queries were used, with the locked retrieval-test partition reserved for later.
