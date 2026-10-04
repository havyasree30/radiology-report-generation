# Figure captions (G2)

Retrieval-validation studies only (clinical subset 378 studies; text metrics 547 studies). B0 = rule-based report from the frozen classifier output; G1 = single-agent RAG; G2 draft = Agent 2 report before the critic; G2 = final multi-agent report. Error bars: 95% study-level paired bootstrap intervals.

**Figure 1. Finding precision, recall and F1 for B0, G1, the G2 Agent 2 draft and final G2.** Micro-averaged agreement between the findings stated in the report (any affirmative mention) and the MeSH-mapped IU reference findings.

**Figure 2. Hallucination and omission.** Left: share of reports with at least one hallucinated finding and with at least one omitted finding. Right: mean hallucinated and omitted findings per report.

**Figure 3. Classifier false-positive propagation and true-positive retention.** Proportion of classifier false positives appearing in the report and of classifier true positives retained. B0 repeats every classifier-positive finding by construction.

**Figure 4. Normal versus abnormal recall.** Share of reference-normal and reference-abnormal studies whose report state is correct. The B0 report state equals the classifier state by construction.

**Figure 5. Agent 1 evidence-support categories.** Left: classifier positives by validated support status, split by agreement with the reference. Right: retrieval-only candidate findings proposed by Agent 1, promoted into the Agent 2 draft and surviving the critic, split by agreement with the reference.

**Figure 6. Critic actions and copying.** Left: findings removed and added by the critic (clinical subset), split by agreement with the reference. Right: copied-sentence rate and whole-report copy rate against the study's own Top-5 and repeated-sentence rate for G1, the draft and final G2.
