# Figure captions (G1B)

Retrieval-validation studies only (clinical subset 378 studies; text metrics 547 studies). B0 = rule-based report from the frozen classifier output; G1A = the same local model given the classifier output only; G1B = the same model and prompt given the classifier output and five sham corpus reports in the G1 format; G1 = the same model and prompt given the true Top-5 retrieved reports. Error bars: 95% study-level paired bootstrap intervals.

**Figure 1. Finding precision, recall and F1 for B0, G1A, G1B and G1.** Micro-averaged agreement between the findings stated in the report (any affirmative mention) and the MeSH-mapped IU reference findings.

**Figure 2. Hallucination versus omission.** Left: share of reports with at least one hallucinated finding and with at least one omitted finding. Right: mean hallucinated and omitted findings per report.

**Figure 3. Classifier false-positive propagation versus true-positive retention.** Proportion of classifier false positives mentioned in the report and of classifier true positives retained. B0 repeats every classifier-positive finding by construction.

**Figure 4. Normal and abnormal recall.** Share of reference-normal and reference-abnormal studies whose report state is correct. The B0 report state equals the classifier state by construction.

**Figure 5. Effect of context relevance.** Left: paired differences G1 (relevant context) minus G1B (sham context) with 95% intervals (blue: interval excludes zero; grey: it does not). Right: findings stated under one condition only, split by whether they match the reference.

**Figure 6. Introduced findings and copying.** Left: findings stated beyond the classifier output with sham (G1B) and relevant (G1) context, split by agreement with the reference. Right: share of sentences copied verbatim from, and of reports that duplicate, the study's own context reports.
