# Results: sham-retrieval control (G1B)

## Generation and context check

G1B produced 547 of 547 reports without failure (98% format-compliant); mean length 23.9 words (G1 23.9, G1A 15.1). Stored G1 and G1A results were reproduced exactly. The relevant context shared a finding with the reference in 66.1% of context reports and the sham context in 24.1% (Table 8).

## Four-system comparison

On the 378 clinical-subset studies, finding precision was 0.331 (95% CI 0.286 to 0.371) for B0, 0.165 for G1A, 0.272 for G1B and 0.411 (95% CI 0.354 to 0.469) for G1; recall 0.411, 0.242, 0.140 and 0.358; F1 0.366, 0.196, 0.185 and 0.383; macro F1 0.325, 0.199, 0.115 and 0.215 (Table 1, Figure 1). Reports with at least one hallucinated finding were 40.7%, 75.4%, 19.8% and 25.7%, and with an omission 34.9%, 40.5%, 44.2% and 38.1% (Figure 2).

## Relevant versus sham context

G1 minus G1B (Table 2, Figure 5): precision +0.139 (95% CI +0.065 to +0.210), recall +0.218 (95% CI +0.160 to +0.276), F1 +0.198 (95% CI +0.141 to +0.257), macro F1 +0.101 (95% CI +0.062 to +0.136), hallucination rate +0.058 (95% CI +0.005 to +0.114), omission rate -0.061 (95% CI -0.090 to -0.034), classifier false-positive propagation +0.135 (95% CI +0.052 to +0.219), true-positive retention +0.402 (95% CI +0.299 to +0.505), normal recall +0.094 (95% CI +0.030 to +0.156), abnormal recall +0.285 (95% CI +0.203 to +0.365) and ROUGE-L -0.008 (95% CI -0.018 to +0.004). Intervals excluded zero for: precision, recall, F1, macro F1, hallucination rate, omission rate, FP propagation, TP retention, normal recall, abnormal recall. G1 improved finding F1 over G1B. Because the prompt, model, decoding, classifier output and context structure were identical and only the identity of the five context reports differed, this is evidence that the relevance of the retrieved reports, and not merely the presence of radiology-report text in the context, contributes to performance under this model and prompt. The gain is a trade-off: relevant context kept far more classifier-positive findings (true-positive retention +0.402 (95% CI +0.299 to +0.505)) and had a higher hallucination rate (+0.058 (95% CI +0.005 to +0.114)), whereas sham context suppressed classifier-positive findings and so produced fewer hallucinated statements together with lower recall; a lower hallucination rate with sham context is therefore not evidence of better grounding. Within the 178 classifier-abnormal studies the F1 gain remained (G1 minus G1B +0.197 (95% CI +0.128 to +0.269)) and was driven by recall (+0.281 (95% CI +0.210 to +0.353)); the precision difference was not reliable (+0.063 (95% CI -0.012 to +0.143)).

## Introduced and removed findings

G1 introduced 68 findings beyond the classifier output, 18 (26.5%) matching the reference; G1B introduced 46, 3 (6.5%) matching. Relative to the classifier, G1 removed 174 positives and G1B 253 (Table 3, Figure 6). Findings stated with relevant context only numbered 184 (69 reference-matching) and with sham context only 83 (7 matching).

## Copying

Against the study's own context reports, 29.5% of G1 sentences and 26.1% of G1B sentences were copied verbatim (paired difference +0.034 (95% CI -0.021 to +0.089)); whole-report copies were 24.1% (G1) and 20.5% (G1B) (difference +0.036 (95% CI -0.014 to +0.088)), over the 498 studies that had context. G1 copied 1.3% of its sentences from the sham reports (control level) and G1B 0.7% from the true Top-5. Against all corpus reports, verbatim sentence rates were 4.0% (G1A), 69.4% (G1B) and 47.0% (G1), and exact duplicates of an entire corpus report 0.0%, 9.0% and 10.2%. Copying from the supplied context did not differ reliably between relevant and sham context, so the copying seen in G1 is largely a property of this prompt and model given any report-like context and is not specific to relevance.

## Normal and abnormal behaviour; rare findings

Abnormal recall was 0.296 for G1B and 0.581 for G1; normal recall 0.818 and 0.911 (Table 5, Figure 4). Pooled true-positive retention over the six rare findings: B0 1.000, G1A 0.381, G1B 0.048 (1 of 21), G1 0.333 (7 of 21); other findings: G1A 0.604, G1B 0.375, G1 0.802. G1 minus G1B retention is +0.286 (95% CI +0.115 to +0.467) for rare and +0.427 (95% CI +0.295 to +0.564) for other findings, so relevance of the context changes rare-finding retention.

## Interpretation

This is a single-model, single-prompt, single-run validation result; it does not show clinical benefit, and relevance is only one of the differences the sham context removes (the displayed rank scores and the evidence-support counts follow the context supplied). Stability of G1B (descriptive): 8.8% of G1B reports are one identical text (G1: 36.4%); 0 G1B reports contain leaked reasoning tokens and 0 stopped at the token limit (G1: 0 and 0); 2 of 261 classifier-normal studies received an unsupported enlarged-cardiomediastinum statement (G1: 0). The classifier, generator, prompt and retrieval configuration were unchanged, the locked retrieval-test partition was not opened, and no multi-agent system was evaluated.
