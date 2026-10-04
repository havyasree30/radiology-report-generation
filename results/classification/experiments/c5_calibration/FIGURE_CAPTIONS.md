# Figure captions (C5)

All figures: CheXpert validation partition (28,671 frontal images, 9,530 patients); calibrated values are out-of-fold from patient-level 5-fold cross-validation; raw values are the sigmoid scores of a DenseNet-121 trained with √-weighted binary cross-entropy. Uncertain labels are excluded and blank labels are scored negative. Where markers coincide, the raw marker is hidden behind the calibrated one.

**Figure 1. Brier score before and after calibration.** Per-class Brier score of the raw sigmoid score and of Platt-calibrated probabilities (out-of-fold). The x-axis is logarithmic. Calibration lowered the Brier score in all 14 classes, most for low-prevalence classes (e.g. Lung Lesion, 37%).

**Figure 2. Log loss before and after calibration.** As Figure 1 for the log loss (logarithmic x-axis). Log loss decreased in all 14 classes.

**Figure 3. Reliability curves for three representative classes.** Observed positive fraction against mean predicted value in 10 equal-frequency bins, for the rarest (Pleural Other), median-prevalence (Pneumothorax) and most prevalent (Support Devices) classes, using the raw score and the out-of-fold Platt-calibrated probability. The diagonal is ideal calibration. Raw scores are over-predicted for Pneumothorax and in the highest bins for Pleural Other, and under-predicted in most bins for Support Devices; the calibrated values follow the diagonal closely.

**Figure 4. Change in calibration metrics per class.** Percentage change in Brier score and log loss from the raw score to the out-of-fold calibrated probability (negative values denote improvement). Changes ranged from -36.9% to -0.2% for the Brier score.

**Figure 5. Mean predicted value against observed prevalence.** For each class, the observed prevalence, the mean raw sigmoid score and the mean out-of-fold calibrated probability (logarithmic x-axis). Raw means exceeded prevalence in 12 of 14 classes (not in Lung Opacity, Support Devices); calibrated means coincided with prevalence.
