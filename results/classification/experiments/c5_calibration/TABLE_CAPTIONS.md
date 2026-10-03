# Table captions (C5)

**Table 1. Comparison of calibration methods.** Macro Brier score, log loss and expected calibration error (10 equal-frequency bins) over the 14 observations for the raw score, temperature scaling, Platt scaling and isotonic regression, from patient-level five-fold out-of-fold predictions on the validation set.

**Table 2. Per-class calibration, raw versus selected calibrator.** Prevalence, Brier score, log loss and expected calibration error of the raw sigmoid score and of the selected calibrator (out-of-fold), with the selected method.

**Table 3. Final calibration parameters.** Platt-scaling slope and intercept of p = sigmoid(a * logit(score) + b) for each observation, with the number of validation images and positives used for fitting (full validation set, after method selection).

**Table 4. Mapping of the C4 thresholds to the calibrated scale.** Raw F1-optimal threshold, calibrated-equivalent threshold, and the number of validation images whose binary decision differs between raw and calibrated outputs.
