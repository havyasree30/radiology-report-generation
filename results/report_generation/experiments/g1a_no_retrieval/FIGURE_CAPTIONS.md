# Figure captions (G1A)

Retrieval-validation studies only (clinical subset 378 studies; text metrics 547 studies). B0 = rule-based report from the frozen classifier output; G1A = the same local language model as G1 given the classifier output only (no retrieval); G1 = the same model given the classifier output and the Top-5 retrieved reports with the evidence summary. Error bars: 95% study-level paired bootstrap intervals.

**Figure 1. Finding precision, recall and F1 for B0, G1A and G1.** Micro-averaged agreement between the findings stated in the report (any affirmative mention) and the MeSH-mapped IU reference findings.

**Figure 2. Hallucination versus omission.** Left: share of reports with at least one hallucinated finding and with at least one omitted finding. Right: mean hallucinated and omitted findings per report.

**Figure 3. Classifier false-positive propagation versus true-positive retention.** Proportion of classifier false positives mentioned in the report and of classifier true positives retained in the report. B0 repeats every classifier-positive finding by construction.

**Figure 4. Normal and abnormal recall.** Share of reference-normal and reference-abnormal studies whose report state is correct. The B0 report state equals the classifier state by construction.

**Figure 5. Retrieval-effect summary.** Left: paired differences G1 minus G1A with 95% intervals (blue: interval excludes zero; grey: it does not). Right: findings that differ between G1 and G1A, split by whether they match the reference.
