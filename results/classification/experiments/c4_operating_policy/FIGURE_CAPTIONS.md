# Figure captions (C4)

All figures: CheXpert validation partition (28,671 frontal images, 9,530 patients); uncertain labels excluded, blank labels scored negative; sigmoid output scores of a DenseNet-121 trained with √-weighted binary cross-entropy. Thresholds were selected on this same partition.

**Figure 1. Macro precision, recall and F1 under four operating policies.** Fixed threshold 0.50, Youden J thresholds, F1-optimal thresholds, and F1-optimal thresholds with the No Finding consistency rule are compared (macro average over 14 observations). Youden thresholds gave the highest recall (0.75) and the lowest precision (0.32); F1-optimal thresholds gave the highest macro F1 (0.45) with intermediate recall (0.54) and precision (0.40).

**Figure 2. Macro specificity and balanced accuracy under four operating policies.** Same policies and averaging as Figure 1. Youden thresholds had the highest balanced accuracy (0.73) and the lowest specificity (0.71).

**Figure 3. Number of positive findings per image.** Share of validation images with 0 to 10+ positive findings among the 13 observations other than No Finding, for the reference labels (mean 2.31), the 0.50 policy (2.48), Youden thresholds (5.12) and F1-optimal thresholds (3.43). Youden thresholds shifted the distribution toward more than five findings per image (44.8% of images).

**Figure 4. Per-class F1 under three thresholding policies.** F1 for each of the 14 observations at the 0.50 threshold, Youden J thresholds and F1-optimal thresholds. Lines connect the three values for each class.

**Figure 5. Precision-recall operating points per class.** For each observation, the operating point moves from the 0.50 threshold to the Youden threshold to the F1-optimal threshold. Youden thresholds moved points toward higher recall and lower precision in most classes.

**Figure 6. Rare-class false-alarm burden.** Precision, recall and false positives per true positive for the six rare observations under the 0.50, Youden and F1-optimal policies. Youden thresholds raised recall to 0.61–0.83 but required 6.6–28.1 false positives per true positive; F1-optimal thresholds required 2.9–7.5.

**Figure 7. No Finding before and after the consistency rule.** Left: No Finding precision, recall, specificity and F1 for F1-optimal thresholds with and without the rule. Right: percentage of images with No Finding and a pathology observation predicted positive together (of all images, and of images predicted No Finding). The rule removed all contradictions (2.6% to 0.0% of images) and reduced No Finding recall from 0.57 to 0.47.

**Figure 8. Final per-class thresholds.** Selected F1-optimal thresholds on the sigmoid score for the 14 observations; the vertical line marks the conventional 0.50 threshold.
