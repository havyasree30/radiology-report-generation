# University report material: final locked test (F1)

_Insert-ready material generated from the stored artifacts. The test was opened once, after the protocol and the code were frozen; no component changed afterwards._

## Experimental setup

The frozen system is a DenseNet-121 chest X-ray classifier with F1-optimised thresholds, a No Finding consistency rule and Platt calibration; a deterministic routing guard; hybrid dense and BM25 retrieval of the Top-5 IU reports from the top-3 classifier findings; and a local single-agent language model (medgemma1.5:4b, greedy decoding) that writes a preliminary Findings and Impression. Studies with no positive classifier label are reported as indeterminate without retrieval or generation. The configuration, prompt and guard hashes were verified, a protocol was hashed, and analysis code that reproduced the validation results was frozen before the 578 locked IU X-Ray studies were opened. 550 had a frontal image, 549 formed the primary set and 368 the clinical-finding subset. Metrics follow the validation definitions; intervals are 1,000-resample study-level bootstrap intervals (seed 42).

## Final classification results

Macro AUROC was 0.830 (95% CI 0.771 to 0.876), macro AUPRC 0.374 (95% CI 0.353 to 0.474) and macro F1 0.312 (95% CI 0.264 to 0.352); Brier score 0.054 and ECE 0.051. Per-class AUROC ranged from 0.665 to 0.986; cardiomegaly (0.927) and pleural effusion (0.913) were among the best and support devices (0.665) among the weakest. The classifier is applied out of domain and labels are derived from MeSH terms, so these values are descriptive.

## Retrieval results

At K = 5 Jaccard was 0.558, nDCG 0.562, finding coverage 0.738 and Hit 0.639; with the oracle query Jaccard was 0.820, so the classifier-derived query limits retrieval. 44.4% of the retrieved texts duplicate another retrieved text.

## Report-generation results

Finding precision 0.413, recall 0.306, F1 0.351; hallucination rate 0.217, omission rate 0.364; classifier false-positive propagation 0.356 and true-positive retention 0.702. 33.3% of sentences and 28.0% of reports are copied from the retrieved reports.

## End-to-end system results

The guard routed 262 studies normal, 224 abnormal and 63 indeterminate. Among decided studies, normal recall was 0.740 and abnormal recall 0.735; decision coverage was 89.1%. 36.6% of abnormal-routed studies had prose read as normal (validation 31.0%). Compared with validation, retrieval and classification are similar, while recall, F1 and coverage are lower (Table 8).

## Failure analysis

142 of 368 clinical-subset studies (38.6%) had none of the first eight failure categories. Classifier error (199 studies) and report-generator omission (134) were the most frequent; abnormal routing with normal prose occurred in 47 and indeterminate abstention in 40.

## Limitations

IU X-Ray labels are MeSH-derived and incomplete, the classifier is applied out of domain (CheXpert to IU), one small quantised model was run once with greedy decoding, the bootstrap resamples studies (no patient identifier), and nothing here shows clinical benefit; the system is a research prototype. Regenerating three validation reports in the pre-test dry run gave identical text for two and a semantically equivalent rewording for one, so local GPU generation is not bitwise reproducible. Several classes and all rare findings have very few test cases, the reference is incomplete for 120 of 368 clinical-subset studies, and the frozen prompt's tendency to write normal-sounding prose for some abnormal-routed studies was reported and not repaired.
