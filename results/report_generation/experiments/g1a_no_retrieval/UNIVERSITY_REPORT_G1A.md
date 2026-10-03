# University report content: G1A (no-retrieval ablation)

_Insert-ready material generated from the G1A artifacts. Validation data only; classifier, generator model and retrieval configuration are unchanged; the locked retrieval-test split is unopened._

## Methodology

### Purpose of the ablation

The single-agent RAG system (G1) was compared with the rule-based baseline, but the two differ in the language model and in the retrieved context. The ablation G1A removes only the retrieval: the same local model (medgemma1.5:4b, offline, greedy decoding, fixed seed) writes the report from the classifier output alone for the same 547 validation studies. All other components (classifier, thresholds, calibration, report format, finding extractor, reference labels, metrics and evaluation populations) are identical to G1, and the identity of the model, its digest and the generation options was checked automatically.

### Inputs, prompt and comparisons

G1A receives the classifier-positive findings with calibrated probabilities and the No Finding state; it receives no retrieved report, retrieval score or evidence summary. The prompt is that of G1 with the retrieval-dependent wording removed; the full difference is stored in the artifacts. Three systems are compared on identical studies (rule-based B0, G1A, G1), and the effect of retrieval is estimated as G1 minus G1A with a paired study-level bootstrap of 1,000 resamples. A difference is attributed to retrieval only if its interval excludes zero. Additional analyses quantify which findings the retrieved context adds or removes, whether over-normalisation (reports describing normal studies despite abnormal classifier output) originates from the model or the context, how rare findings are retained, and how much text is copied from corpus reports.

## Results

### Three-system comparison and the controlled retrieval effect

Finding F1 was 0.366 for B0, 0.196 for G1A and 0.383 for G1; hallucination rates were 40.7%, 75.4% and 25.7%. With the model held constant, retrieval changed finding precision by +0.246 (95% CI +0.193 to +0.300), recall by +0.116 (95% CI +0.060 to +0.170), F1 by +0.186 (95% CI +0.136 to +0.232), the hallucination rate by -0.497 (95% CI -0.561 to -0.431), and classifier false-positive propagation by -0.156 (95% CI -0.243 to -0.066) and true-positive retention by +0.154 (95% CI +0.053 to +0.259) (Tables 1 and 2, Figures 1 to 3 and 5).

### Retrieval-induced findings, normalisation and rare findings

The retrieval context added 136 findings and removed 306; of the added findings 46 matched the reference and 90 did not, and of the removed ones 13 were correct and 293 unsupported (Table 4). Abnormal recall was 0.726 for the classifier, 0.710 for G1A and 0.581 for G1; Over-normalisation is present in G1A and is larger with the retrieved context: the controlled comparison supports a contribution of the retrieval context in addition to the language-model/prompt behaviour. This reading is tentative: G1A abnormal recall is inflated by its unsupported abnormal reports for classifier-normal studies, and the share of normal reports given an abnormal classifier output did not differ reliably between G1A and G1. Pooled true-positive retention over the six rare findings: B0 1.000, G1A 0.381 (8 of 21), G1 0.333 (7 of 21); for the other findings B0 1.000, G1A 0.604, G1 0.802. G1 minus G1A retention is -0.048 (95% CI -0.300 to +0.211) for rare and +0.198 (95% CI +0.090 to +0.302) for other findings; the rare-minus-other retention gap differs between G1 and G1A by -0.246 (95% CI -0.496 to +0.025), so the data do not show that retrieval disproportionately suppresses rare findings (small counts limit this conclusion).

### Copying

Against all corpus reports, 4.0% of G1A and 47.0% of G1 sentences of at least six words occurred verbatim, so verbatim reuse is partly a property of the generic IU reporting style; the study's own retrieved reports were copied in 0.0% (G1A, control) versus 26.8% (G1) of long sentences.

### Interpretation caveat

G1A showed degenerate behaviour that confounds the contrast: for 92.0% of classifier-normal studies it wrote an unsupported enlarged-cardiomediastinum report (the first term of the prompt's term list), 45.3% of its reports are one identical text and 6 contain leaked reasoning tokens (G1: 36.4%, a normal report, and none). The prompt was frozen and not changed. Among the 178 classifier-abnormal studies, where G1A is not degenerate, G1 minus G1A was precision +0.092 (95% CI +0.034 to +0.150), recall +0.149 (95% CI +0.079 to +0.217) and F1 +0.119 (95% CI +0.060 to +0.175) (supplementary Table 6).

## Discussion

The ablation shows what the retrieval context adds under this model and prompt: the controlled intervals above, not the difference to the rule-based baseline, are the evidence for an effect of retrieval. Limits: one small model and one prompt per system; the G1 − G1A contrast includes the retrieval-linked instruction that could not be kept; rule-based finding extraction and a MeSH-derived reference; small counts for rare findings; and validation data only. The multi-agent system (G2) has not been started.
