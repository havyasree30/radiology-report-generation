# Final test figure captions

Locked end-to-end test (IU X-Ray, 578 studies; clinical subset 368; primary set 549). Blue = locked test; orange = validation dry run of the same code. Error bars: 95% study-level bootstrap intervals.

**Figure 1. Per-class classifier AUROC and AUPRC.** Frozen classifier on the locked test with MeSH-mapped labels; the number of test positives is shown per class; Enlarged Cardiomediastinum is undefined (no positives).

**Figure 2. Calibration.** Left: pooled reliability diagram with equal-frequency bins for calibrated probabilities (test and validation). Right: per-class adaptive-bin expected calibration error on the test.

**Figure 3. Retrieval at K = 1, 3, 5 and 10.** Jaccard, nDCG, finding coverage and exact-match Hit for the classifier query and for the oracle query (diagnostic only). The dotted line marks the frozen K = 5.

**Figure 4. Validation versus test classification metrics.** Macro AUROC, AUPRC, F1, Brier score and ECE.

**Figure 5. Final report precision, recall and F1.** Finding agreement with the mapped IU reference on the clinical subset (validation versus test).

**Figure 6. Hallucination, omission, classifier false-positive propagation and true-positive retention.** Clinical subset, validation versus test.

**Figure 7. Three-state system results.** Left: routing of reference-normal and reference-abnormal studies to normal, abnormal and indeterminate. Right: normal and abnormal recall among decided studies and decision coverage.

**Figure 8. Failure-category summary.** Share of the clinical subset in each frozen category; studies can belong to several categories; grey bars are limitation flags.
