# Results: probability calibration (C5)

## Baseline calibration quality

On the validation set (28,671 images, 9,530 patients), the raw sigmoid scores were systematically over-confident, most strongly for low-prevalence observations. Across the 14 observations the macro Brier score was 0.0962, the macro log loss 0.3125 and the macro expected calibration error 0.0643. The mean raw score exceeded the observed prevalence in 12 of 14 classes (not in Lung Opacity, Support Devices): for example 0.088 versus 0.027 for Pneumonia, 0.031 versus 0.013 for Pleural Other, and 0.201 versus 0.089 for No Finding (Figure 5). Raw calibration slopes ranged from 0.48 to 0.99.

## Comparison of calibration methods

Using patient-level five-fold out-of-fold predictions (Table 1), temperature scaling gave a small improvement (macro Brier score 0.0950, log loss 0.3082, ECE 0.0533), consistent with it being unable to correct a prevalence offset. Platt scaling gave substantially lower values (Brier 0.0839, log loss 0.2776, ECE 0.0056), corresponding to reductions of 12.8%, 11.2% and 91% relative to the raw scores. Isotonic regression performed similarly to Platt scaling on the Brier score (0.0840) and log loss (0.2784) with a lower ECE (0.0030), but offered no advantage on the primary metrics. No calibrator showed a pathological fit in any fold.

## Per-class findings

Improvements were concentrated in the low-prevalence classes (Table 2, Figures 1, 2 and 4). The largest Brier score reductions were for Lung Lesion (36.9%), Pneumonia (35.8%), No Finding (32.2%), Enlarged Cardiomediastinum (29.9%). The smallest changes were for the most prevalent observations, Lung Opacity (0.2%), Pleural Effusion (0.4%), Edema (1.6%), Support Devices (3.1%), whose raw scores already matched prevalence closely. Calibration did not worsen the Brier score or log loss of any class. After calibration, out-of-fold calibration slopes were 0.995 to 1.000, and the mean predicted probability was within 0.0000 of the observed prevalence for every class.

## Selected calibrator and reliability

Under the pre-specified rule, Platt scaling was selected for 12 classes individually; the two remaining classes were within the tolerance of Platt scaling, and applying it uniformly cost at most 0.29% in any class. Platt scaling was therefore used for all 14 observations (Table 3), with slopes between 0.479 and 0.989 and intercepts between -2.232 and 0.393. Reliability curves based on equal-frequency bins showed class-specific raw behaviour (Figure 3): for Pneumothorax the observed fraction was below the mean prediction in 9 of 10 bins (largest gap 0.330), for Pleural Other the raw scores were over-predicted mainly in the highest bins (largest gap 0.171), and for Support Devices they were under-predicted in most bins (9 of 10). The out-of-fold calibrated values followed the diagonal closely, with largest gaps of 0.004, 0.010 and 0.039.

## Threshold equivalence and ranking

The binary decisions of the operating policy were unchanged. Each class-specific F1-optimal threshold was mapped through its calibrator (Table 4), and the calibrated decisions matched the raw decisions for all 28,671 images in all 14 classes (0 mismatches); applying the No Finding rule to the calibrated-side decisions reproduced the final predictions exactly. Because the Platt maps are strictly increasing, AUROC and AUPRC were unchanged (maximum absolute differences 0.0e+00 and 0.0e+00). ## Stability and interpretation

Fits were stable across folds: the Platt slope of any class varied by at most 0.043 between folds, and the smallest number of positive labels in any class and fold was 76, so rare-class estimates rest on few events and should be read with that in mind. Pooled out-of-fold AUROC differed from the raw AUROC by at most 0.0007, reflecting small differences between the fold-specific monotone maps rather than any change in ranking. Calibration therefore changed what the numbers mean, not which images are called positive. Raw outputs remain model scores; only the calibrated values should be presented as probabilities.

## Scope of the evidence

These results are validation-set estimates; the final calibrators were fitted on the same validation data, the held-out test set has not been evaluated, and the probabilities refer to the project's report-derived label definition, with uncertain labels excluded and blank labels treated as negative. Calibration may not transfer to other institutions or labelling methods.
