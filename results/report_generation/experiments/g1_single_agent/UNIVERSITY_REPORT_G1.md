# University report content: G1 (single-agent RAG baseline)

_Insert-ready material generated from the G1 artifacts. Validation data only; the locked retrieval-test split is unopened; classifier and retrieval are frozen._

## Methodology

### Single-Agent RAG architecture and data flow

The system turns a chest radiograph into a preliminary report in four stages. (1) The frozen DenseNet-121 classifier outputs fourteen independent probabilities, which are thresholded and calibrated. (2) The positive findings (at most the three most probable) form a query, and five reports are retrieved from a corpus of other patients' reports by fusing a dense MiniLM ranking and a BM25 ranking. (3) A deterministic evidence table is computed that lists, for every classifier-positive finding, its calibrated probability and how many of the five retrieved reports mention that finding. (4) A single call to a local language model (medgemma1.5:4b, run offline with Ollama, greedy decoding) writes a short FINDINGS and IMPRESSION text. Nothing is learned or tuned in stages 2 to 4: the classifier, thresholds, calibration, retrieval configuration and number of references are those fixed in earlier stages. The reference report is never part of the input.

### Grounding prompt and evidence summary

The prompt tells the model to use only the classifier output and the retrieved evidence, not to invent findings that appear in neither, to give more confidence to findings supported by several retrieved reports, to omit or hedge classifier-positive findings without support, not to mention every classifier-positive finding, not to copy retrieved reports, and not to add history, measurements, comparisons or recommendations. A normal classifier state yields a normal report even if retrieved reports describe disease. The evidence table is information for the generator, not a second classifier, and never changes the stored classifier prediction. A rule-based baseline (B0) that verbalises the classifier output without retrieval allows the question of whether retrieval-augmented generation improves on simply stating the classifier findings.

### Evaluation

Findings are extracted from generated text by an assertion-aware lexicon and compared with the findings mapped from the MeSH terms of the study. The main outcomes are finding precision, recall and F1, hallucinated and omitted findings, the proportion of classifier false positives that appear in the report, and the proportion of classifier true positives that are kept. Differences are estimated with a paired study-level bootstrap of 1,000 resamples. Lexical metrics (BLEU, ROUGE-L, METEOR-like) are secondary. The primary set has 547 validation studies and the clinical subset 378.

## Results

### Report-generation metrics

Finding F1 was 0.383 (95% CI 0.330 to 0.435) for G1 and 0.366 (95% CI 0.318 to 0.410) for B0 (difference +0.017 (95% CI -0.024 to +0.053)); precision 0.411 versus 0.331 and recall 0.358 versus 0.411 (Table 2, Figure 1). ROUGE-L was 0.247 versus 0.121, BLEU-4 0.039 versus 0.010 (Figure 4).

### Hallucination, omission and false-positive propagation

At least one hallucinated finding occurred in 25.7% of G1 reports and 40.7% of B0 reports; at least one omission in 38.1% and 34.9%. Classifier false-positive propagation was 0.405 for G1 and 1.000 for B0 (difference -0.595 (95% CI -0.650 to -0.538)), and true-positive retention 0.718 versus 1.000 (difference -0.282 (95% CI -0.359 to -0.211)). Yes, in this comparison G1 mentioned fewer classifier false positives than B0, at the price of retaining fewer classifier true positives (retention 0.718 (95% CI 0.641 to 0.789) versus 1.000 (95% CI 1.000 to 1.000)). (Tables 4 and 5, Figures 2, 3, 5 and 6.)

### Qualitative examples

Six deterministically chosen validation cases (a correct normal report, a correct abnormal report, a classifier false positive that G1 left out, one that G1 repeated, an omitted finding and a retrieval mismatch) are listed in `qualitative_examples.md`, each with the classifier output, the retrieved evidence, the generated report, the reference report and a short factual error analysis.

## Discussion

Benefits: the report is grounded in two sources, the evidence table makes the support of every classifier finding explicit, and the whole pipeline runs offline on a local model. Limits: This is a comparison of two complete systems: G1 differs from B0 in both the language model with its grounding prompt and the retrieved context, and no no-retrieval LLM ablation was run, so the effect cannot be attributed to retrieval alone. The model is small (4 billion parameters, 4-bit quantisation) and one prompt was used; the reference is derived from MeSH terms, so findings visible in the image but not indexed count as hallucinations; finding extraction is rule-based; and retrieved support separates classifier true from false positives only partly (mean support 0.680 versus 0.344). The results are validation-set results; the locked retrieval test has not been used, and the multi-agent comparison (G2) is the next stage.
