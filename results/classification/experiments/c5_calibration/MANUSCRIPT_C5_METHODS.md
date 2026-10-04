# Methods: probability calibration (C5)

## Rationale and calibration methods

The sigmoid outputs of a classifier trained with a class-weighted loss are model scores, not necessarily probabilities. To obtain interpretable probabilities, we calibrated the output of each of the 14 CheXpert observations separately, using the logit of the stored sigmoid score (recovered as log(p/(1−p)); the scores were stored in single precision, giving a worst-case logit error of 8e-05). Four options were compared: the raw score; temperature scaling, p = σ(z/T) with T > 0; Platt scaling, p = σ(az + b), with a flag for non-positive or extreme slopes; and isotonic regression, a non-decreasing map used as an exploratory comparator because it can overfit in rare classes. Calibrators were fitted by minimising log loss on valid labels only (uncertain labels excluded, blank labels treated as negative).

## Patient-level cross-validation

Calibration quality was estimated from out-of-fold predictions. The 28,671 validation images (9,530 patients) were split into 5 folds with a fixed seed; all images of a patient were assigned to a single fold, using multi-label iterative stratification over patients to balance positives per class. For each method, calibrators were fitted on four folds and applied to the fifth, and the held-out predictions were pooled. No patient contributed to both fitting and evaluation within a fold.

## Metrics and method selection

We report the Brier score and log loss (primary), the expected calibration error with ten equal-frequency bins (supplementary), the mean prediction against prevalence, and the calibration slope and intercept. Methods were not chosen using AUROC or AUPRC. A selection rule was fixed before the results were examined: among calibrators without pathological fits (non-positive or extreme Platt slope, temperature at the search boundary, isotonic fits with very few output levels) and able to reproduce the binary decisions of the operating policy, the simplest method within 0.5% of the best on both Brier score and log loss was chosen for each class. If one method was chosen for at least 10 classes and applying it to all classes cost at most 1% in every class, it was adopted uniformly.

## Final calibrators and relation to the operating policy

The selected calibrators were then refitted on the full validation set. The binary decision policy was left unchanged: findings were still called positive using the raw score and the class-specific F1-optimal thresholds, with the No Finding rule suppressing No Finding when any of the twelve pathology observations was positive. Each raw threshold was mapped through its calibrator to a calibrated-equivalent threshold, and we verified for every image and class that the raw and calibrated decisions were identical. As a sanity check, AUROC and AUPRC were compared before and after calibration without using them for selection.

## Data handling

All calibration analyses used the validation partition only, with the stored model scores; no model was retrained and no inference was repeated. The held-out test partition was not used at any point and remains unevaluated, so that it can provide an independent assessment once the complete classifier configuration, including the calibrators, has been frozen. Calibrated outputs are described as probabilities only where they refer to the calibrated values; raw sigmoid outputs are referred to as scores.
