# Figure captions (G1)

Retrieval-validation studies only (clinical subset 378 studies; text metrics 547 studies). B0 = rule-based report from the frozen classifier output; G1 = single-agent RAG (classifier + Top-5 hybrid retrieval + evidence summary + local language model). Error bars: 95% study-level bootstrap intervals.

**Figure 1. Finding precision, recall and F1.** Micro-averaged agreement between the findings stated in the generated report (any affirmative mention) and the MeSH-mapped IU reference findings for B0 and G1.

**Figure 2. Hallucination and omission.** Left: share of reports with at least one hallucinated finding (stated, absent from the reference) and at least one omitted finding (in the reference, not stated). Right: the corresponding mean numbers of findings per report.

**Figure 3. Classifier false-positive propagation and true-positive retention.** Proportion of classifier false positives mentioned in the report and of classifier true positives retained in the report.

**Figure 4. Lexical report-generation metrics.** BLEU-1, BLEU-4, ROUGE-L and an exact-match METEOR variant (no stemming or synonyms) against the reference report. Lexical overlap is secondary and does not measure clinical correctness.

**Figure 5. Finding-support provenance.** Left: share of findings stated by G1 supported by the classifier only, by retrieval only, by both, or by neither. Right: share of classifier-positive findings mentioned in the G1 report by the number of Top-5 retrieved reports supporting the finding, for classifier true and false positives.

**Figure 6. Error-category breakdown.** Left: hallucinated findings split into classifier false positives and other sources. Right: omitted findings split into classifier true positives dropped by the report and findings the classifier never predicted.
