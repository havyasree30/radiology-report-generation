# Results: no-retrieval language-model ablation (G1A)

## Generation and reproduction

G1A produced 547 of 547 reports without failure for the same studies as G1 (99% format-compliant), with a mean of 15.1 words (G1 23.9, B0 10.5). The G1 results were reproduced exactly from the stored reports (maximum difference 1e-16).

## Three-system comparison

On the 378 clinical-subset studies, finding precision was 0.331 (95% CI 0.286 to 0.371) for B0, 0.165 (95% CI 0.126 to 0.207) for G1A and 0.411 (95% CI 0.354 to 0.469) for G1; recall 0.411, 0.242 and 0.358; F1 0.366, 0.196 and 0.383; and macro F1 0.325, 0.199 and 0.215 (Table 1, Figure 1). Reports with at least one hallucinated finding were 40.7%, 75.4% and 25.7%, and with at least one omission 34.9%, 40.5% and 38.1% (Figure 2). Classifier false-positive propagation and true-positive retention were 0.561 and 0.564 for G1A and 0.405 and 0.718 for G1 (B0 repeats every classifier positive: 1.000 for both) (Figure 3). ROUGE-L was 0.121, 0.104 and 0.247.

## Effect of retrieval with the model held constant

G1 minus G1A (Table 2, Figure 5): precision +0.246 (95% CI +0.193 to +0.300), recall +0.116 (95% CI +0.060 to +0.170), F1 +0.186 (95% CI +0.136 to +0.232), hallucination rate -0.497 (95% CI -0.561 to -0.431), omission rate -0.024 (95% CI -0.050 to +0.003), classifier false-positive propagation -0.156 (95% CI -0.243 to -0.066), true-positive retention +0.154 (95% CI +0.053 to +0.259), abnormal recall -0.129 (95% CI -0.232 to -0.031) and ROUGE-L +0.143 (95% CI +0.130 to +0.155). Intervals excluded zero for: precision, recall, F1, hallucination rate, FP propagation, TP retention, abnormal recall, ROUGE-L; differences in the remaining metrics are not attributed to retrieval.

## Findings induced by the retrieval context

136 findings were present in G1 but not in G1A: 133 appeared in the retrieved reports, 46 matched the reference and 90 were unsupported by it; 64 had not been flagged by the classifier. 306 findings were present in G1A but not in G1: 13 were correct findings that were lost and 293 were unsupported findings that were suppressed. The net effect was +33 reference-matching and -203 unsupported findings (Table 4).

## Normal and abnormal consistency

Abnormal recall was 0.726 for the classifier, 0.710 for G1A and 0.581 for G1; normal recall 0.729, 0.115 and 0.911 (Table 5, Figure 4). When the classifier output was abnormal, 25.8% of G1A reports and 29.8% of G1 reports described a normal study (difference +0.039 (95% CI -0.056 to +0.141)). Over-normalisation is present in G1A and is larger with the retrieved context: the controlled comparison supports a contribution of the retrieval context in addition to the language-model/prompt behaviour. This reading is tentative: G1A abnormal recall is inflated by its unsupported abnormal reports for classifier-normal studies, and the share of normal reports given an abnormal classifier output did not differ reliably between G1A and G1.

## Rare findings

Pooled true-positive retention over the six rare findings: B0 1.000, G1A 0.381 (8 of 21), G1 0.333 (7 of 21); for the other findings B0 1.000, G1A 0.604, G1 0.802. G1 minus G1A retention is -0.048 (95% CI -0.300 to +0.211) for rare and +0.198 (95% CI +0.090 to +0.302) for other findings; the rare-minus-other retention gap differs between G1 and G1A by -0.246 (95% CI -0.496 to +0.025), so the data do not show that retrieval disproportionately suppresses rare findings (small counts limit this conclusion).

## Copying

Against all corpus reports, 4.0% of G1A and 47.0% of G1 sentences of at least six words occurred verbatim in some corpus report, and 0.0% and 10.2% of reports duplicated an entire corpus report. Against the study's own retrieved reports, the copied-sentence rate was 0.0% for G1A, which never saw them, and 26.8% for G1.

## Interpretation

G1A showed degenerate behaviour that confounds the contrast: for 92.0% of classifier-normal studies it wrote an unsupported enlarged-cardiomediastinum report (the first term of the prompt's term list), 45.3% of its reports are one identical text and 6 contain leaked reasoning tokens (G1: 36.4%, a normal report, and none). The prompt was frozen and not changed. Among the 178 classifier-abnormal studies, where G1A is not degenerate, G1 minus G1A was precision +0.092 (95% CI +0.034 to +0.150), recall +0.149 (95% CI +0.079 to +0.217) and F1 +0.119 (95% CI +0.060 to +0.175) (supplementary Table 6).

These are validation-set results for one small local model with one frozen prompt per system. The G1 − G1A contrast measures the retrieval context together with the retrieval-linked instruction that could not be kept in G1A. The classifier, generator model and retrieval configuration were unchanged, the locked retrieval-test partition was not opened, and no multi-agent system was evaluated.
