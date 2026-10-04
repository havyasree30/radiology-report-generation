# Final test: discussion points (for later manuscript assembly)

- **Generalisation.** Classification, retrieval and report metrics on the locked test are close to the validation estimates (retrieval Jaccard@5 0.557 to 0.558; macro AUROC 0.847 to 0.830), with a larger drop in macro AUPRC (-0.091) and lower finding recall (-0.052); intervals are wide because the test has 368 clinical-subset studies.
- **Out-of-domain classifier.** The classifier was trained on CheXpert and applied to IU X-Ray with MeSH-derived labels; per-class AUROC ranges from 0.665 to 0.986, and some classes have one to five positives or none.
- **Retrieval is limited by the query.** At K = 5 the classifier-query Jaccard is 0.558 against 0.820 for the oracle query, and 44.4% of retrieved texts are duplicates of another retrieved text; better classifier findings, not a larger K, would be needed to close the gap.
- **Abstention is a coverage trade-off.** 11.5% of studies were indeterminate; decision coverage was 89.1%; decided-case recalls were 0.740 (normal) and 0.735 (abnormal). 19 of the 40 abstained studies with a reference were reference-abnormal.
- **Routing/prose inconsistency persists.** 36.6% of abnormal-routed studies have prose read as normal (validation 31.0%); an application must show the routing state next to the prose.
- **Grounding.** Findings stated only because of the retrieved reports are mostly unsupported (16.0% supported); findings supported by both the classifier and retrieval are supported in 54.5% of cases.
- **Copying.** 33.3% of sentences and 28.0% of reports are copied from the study's own Top-5; the earlier multi-agent variant did not reduce propagation of classifier findings and was not selected.
- **Rare findings.** Retention of rare-finding classifier true positives is 6 of 22; Lung Lesion and Pleural Other true positives were not retained in any report.
- **Reference limitation.** 120 of 368 clinical-subset studies have unmapped or uncertain MeSH terms; hallucination and omission rates are therefore partly reference artefacts.
- **Runtime.** End-to-end compute was 37.6 minutes for 486 generated reports (mean 4.5 s per report) on a consumer GPU.
- **Reproducibility.** Regenerating three validation reports in the pre-test dry run gave identical text for two and a semantically equivalent rewording for one, so local GPU generation is not bitwise reproducible.
- **Scope.** IU X-Ray labels are MeSH-derived and incomplete, the classifier is applied out of domain (CheXpert to IU), one small quantised model was run once with greedy decoding, the bootstrap resamples studies (no patient identifier), and nothing here shows clinical benefit; the system is a research prototype.
