# University report content: G1B (sham-retrieval control)

_Insert-ready material generated from the G1B artifacts. Validation data only; classifier, prompt, generator model and retrieval configuration are unchanged; the locked retrieval-test split is unopened._

## Methodology

### Purpose of the control

G1A removed the retrieved context entirely, but the model then behaved degenerately on classifier-normal studies, so G1A cannot isolate the contribution of retrieved content. The sham-retrieval control G1B keeps everything of G1 (prompt, offline model medgemma1.5:4b, decoding, message format, five context reports per study, classifier output, extractor and metrics) and changes only which five reports are shown: deterministic, seeded random reports from the training corpus, never the study itself and never its true Top-5. The sham mapping was frozen and hashed before generation, and the generator was not told that the context was a control.

### Comparisons

Four systems are compared on identical studies (rule-based B0, G1A, G1B, G1). The primary comparison is G1 minus G1B with a paired study-level bootstrap of 1,000 resamples; a difference is interpreted as an effect of context relevance only if its interval excludes zero. Additional analyses cover findings introduced beyond the classifier output and findings removed, copying of context text, normal/abnormal behaviour and rare findings.

## Results

### Four-system and controlled comparison

Finding F1 was 0.366 for B0, 0.196 for G1A, 0.185 for G1B and 0.383 for G1; hallucination rates were 40.7%, 75.4%, 19.8% and 25.7%. With relevant instead of sham context, finding precision changed by +0.139 (95% CI +0.065 to +0.210), recall by +0.218 (95% CI +0.160 to +0.276), F1 by +0.198 (95% CI +0.141 to +0.257), the hallucination rate by +0.058 (95% CI +0.005 to +0.114), classifier false-positive propagation by +0.135 (95% CI +0.052 to +0.219) and true-positive retention by +0.402 (95% CI +0.299 to +0.505) (Tables 1 and 2, Figures 1 to 3 and 5). G1 improved finding F1 over G1B. Because the prompt, model, decoding, classifier output and context structure were identical and only the identity of the five context reports differed, this is evidence that the relevance of the retrieved reports, and not merely the presence of radiology-report text in the context, contributes to performance under this model and prompt. The gain is a trade-off: relevant context kept far more classifier-positive findings (true-positive retention +0.402 (95% CI +0.299 to +0.505)) and had a higher hallucination rate (+0.058 (95% CI +0.005 to +0.114)), whereas sham context suppressed classifier-positive findings and so produced fewer hallucinated statements together with lower recall; a lower hallucination rate with sham context is therefore not evidence of better grounding.

### Introduced findings, copying, normal/abnormal behaviour and rare findings

G1 introduced 68 findings beyond the classifier output (26.5% reference-supported); G1B introduced 46 (6.5%) (Table 3). Against the study's own context reports, 29.5% of G1 sentences and 26.1% of G1B sentences were copied verbatim (paired difference +0.034 (95% CI -0.021 to +0.089)); whole-report copies were 24.1% (G1) and 20.5% (G1B) (difference +0.036 (95% CI -0.014 to +0.088)), over the 498 studies that had context. G1 copied 1.3% of its sentences from the sham reports (control level) and G1B 0.7% from the true Top-5. Against all corpus reports, verbatim sentence rates were 4.0% (G1A), 69.4% (G1B) and 47.0% (G1), and exact duplicates of an entire corpus report 0.0%, 9.0% and 10.2%. Copying from the supplied context did not differ reliably between relevant and sham context, so the copying seen in G1 is largely a property of this prompt and model given any report-like context and is not specific to relevance. Abnormal recall was 0.296 for G1B and 0.581 for G1. Pooled true-positive retention over the six rare findings: B0 1.000, G1A 0.381, G1B 0.048 (1 of 21), G1 0.333 (7 of 21); other findings: G1A 0.604, G1B 0.375, G1 0.802. G1 minus G1B retention is +0.286 (95% CI +0.115 to +0.467) for rare and +0.427 (95% CI +0.295 to +0.564) for other findings, so relevance of the context changes rare-finding retention.

### Interpretation caveat

This is a single-model, single-prompt, single-run validation result; it does not show clinical benefit, and relevance is only one of the differences the sham context removes (the displayed rank scores and the evidence-support counts follow the context supplied). Stability of G1B (descriptive): 8.8% of G1B reports are one identical text (G1: 36.4%); 0 G1B reports contain leaked reasoning tokens and 0 stopped at the token limit (G1: 0 and 0); 2 of 261 classifier-normal studies received an unsupported enlarged-cardiomediastinum statement (G1: 0).

## Discussion

The sham control, and not the no-retrieval ablation, is the primary evidence for whether context relevance matters: it holds the prompt and context structure fixed. G1A remains useful as a classifier-only robustness check. Limits: one small model, one prompt and one sham draw per study; rule-based finding extraction and a MeSH-derived reference; small counts for rare findings; and validation data only. The multi-agent system (G2) has not been started.
