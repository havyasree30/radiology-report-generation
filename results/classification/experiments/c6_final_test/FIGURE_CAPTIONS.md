# Figure captions (C6)

Locked test: 28,706 frontal images from 9,680 patients. Validation: 28,671 images. Metrics use valid labels only (uncertain excluded, blank scored negative). Binary metrics use the frozen C4 F1-optimal thresholds with the No Finding rule; calibrated values use the frozen C5 Platt calibrators.

**Figure 1. Validation versus locked-test performance.** Macro and micro ranking metrics, macro binary metrics and macro calibration metrics for the frozen classifier on validation (open circles) and the locked test set (filled diamonds). Panels use different horizontal axes. Differences are descriptive; nothing was tuned on the test set.

**Figure 2. Per-class AUROC and AUPRC.** Validation (open circles) and locked-test (filled diamonds) values for the 14 observations. The dashed line marks AUROC 0.5; grey ticks in the right panel mark the test prevalence, which is the AUPRC of a non-informative score.

**Figure 3. Per-class precision, recall and F1 on the locked test set** at the frozen operating thresholds (No Finding after the consistency rule).

**Figure 4. Rare-class performance.** The six rare classes defined in the loss study. Left: AUPRC (grey tick: test prevalence). Centre: precision (circles) and recall (triangles). Right: false positives per true positive. Open markers: validation; filled markers: locked test.

**Figure 5. Reliability diagrams on the locked test set** for the rare (Pleural Other), medium (Pneumothorax) and common (Support Devices) classes fixed in the calibration study. Observed positive fraction against mean predicted value in 10 equal-frequency bins, for the raw sigmoid score and the frozen Platt-calibrated probability; the diagonal is ideal calibration.

**Figure 6. True versus predicted number of positive findings per image** on the locked test set (13 observations excluding No Finding; true = explicit positive labels; predicted = final policy), as a percentage of images.

**Figure 7. Patient-level bootstrap uncertainty.** Locked-test macro metrics (diamonds) with 95% percentile intervals from 1000 patient-level resamples (seed 42); each panel has its own horizontal range. The intervals cover test-sample variability only.
