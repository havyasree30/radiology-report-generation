# Results: locked-test evaluation (C6)

## Test set and independence

The locked test partition contained 28,706 frontal images from 9,680 patients. No patient, split group or image path overlapped with the training or validation partitions (all overlap counts were zero), and the frozen configuration was recorded before inference.

## Ranking performance

On the locked test set the macro AUROC was 0.803 (95% CI 0.799 to 0.806) and the macro AUPRC 0.427 (95% CI 0.422 to 0.433); the micro-averaged values were 0.893 and 0.672. Per-class AUROC ranged from 0.682 (Enlarged Cardiomediastinum) to 0.888 (Pleural Effusion), and per-class AUPRC from 0.093 (Pleural Other) to 0.873 (Support Devices); AUPRC exceeded the class prevalence in 14 of 14 classes (Table 2, Figure 2).

## Binary decisions

With the frozen operating policy, the macro precision was 0.399 (95% CI 0.394 to 0.404), recall 0.535 (95% CI 0.528 to 0.542), specificity 0.837 (95% CI 0.835 to 0.840), F1 0.450 (95% CI 0.445 to 0.455) and balanced accuracy 0.686 (95% CI 0.683 to 0.690) (Table 1, Figure 3). Pooled micro precision, recall and F1 were 0.559, 0.760 and 0.644. Recall exceeded precision in 14 of 14 classes, consistent with F1-optimal thresholds on imbalanced labels.

## Calibration

The frozen Platt calibrators gave a macro Brier score of 0.0837 (95% CI 0.0829 to 0.0844), a macro log loss of 0.2768 and a macro ECE of 0.0069, compared with 0.0960, 0.3116 and 0.0648 for the raw scores (Table 5). The largest per-class ECE was 0.0206 (Support Devices). Reliability curves for Pleural Other, Pneumothorax and Support Devices are shown in Figure 5. The calibrated-equivalent thresholds reproduced the raw-score decisions for all 28,706 images in every class.

## Rare classes

The six rare classes defined in the loss study were: Pleural Other (test prevalence 0.013, AUROC 0.809, AUPRC 0.093, precision 0.113, recall 0.249, 7.8 false positives per true positive); Pneumonia (test prevalence 0.027, AUROC 0.750, AUPRC 0.122, precision 0.142, recall 0.232, 6.0 false positives per true positive); Fracture (test prevalence 0.039, AUROC 0.779, AUPRC 0.199, precision 0.261, recall 0.285, 2.8 false positives per true positive); Lung Lesion (test prevalence 0.037, AUROC 0.784, AUPRC 0.213, precision 0.244, recall 0.274, 3.1 false positives per true positive); Enlarged Cardiomediastinum (test prevalence 0.051, AUROC 0.682, AUPRC 0.140, precision 0.134, recall 0.319, 6.4 false positives per true positive); Consolidation (test prevalence 0.079, AUROC 0.751, AUPRC 0.209, precision 0.196, recall 0.447, 4.1 false positives per true positive). Validation values were similar (Table 4, Figure 4); for example, Pleural Other had 7.5 false positives per true positive on validation.

## No Finding

No Finding had a test prevalence of 0.088. Before the consistency rule, 851 images (2.96% of images) were predicted No Finding together with at least one pathology finding; the rule modified 851 images and left no such contradictions. After the rule the precision, recall, specificity and F1 for No Finding were 0.491, 0.496, 0.950 and 0.494, and 8.9% of images were predicted No Finding. The test labels contained 0 contradictions between No Finding and the 12 pathology labels.

## Findings per image

The labels contained a mean of 2.32 positive findings per image (median 2; 95th percentile 5; maximum 7), whereas the final policy predicted a mean of 3.43 (median 4; 95th percentile 6; maximum 8). Zero findings were present in 9.4% of images and predicted for 6.8%; more than five findings were present in 0.7% and predicted for 10.4% (Figure 6).

## Validation versus test

Test and validation results were close (Table 3, Figure 1). Macro AUROC differed by +0.0014, macro AUPRC by +0.0034, macro F1 by +0.0005, macro Brier score by -0.0002 and macro ECE by +0.0013 (test minus validation). Per-class AUROC differences ranged from -0.017 (Fracture) to +0.023 (Pleural Other). The 95% bootstrap intervals were narrow (for example macro AUROC 0.7988 to 0.8062; Figure 7) and reflect test-sample variability only.

## Error examples

For each class we listed the five highest-scoring false positives and five lowest-scoring false negatives (140 descriptive examples with anonymised identifiers). The median raw score was 0.972 among the false-positive examples and 0.004 among the false-negative examples; 63 of 70 false-positive examples came from images with at least one other positive label. These listings were not interpreted radiologically and did not influence the classifier.

## Interpretation and limits

These results describe agreement with automatically extracted CheXpert labels, with uncertain labels excluded, on a single-institution dataset. They do not establish clinical performance, and the frequent false positives for rare classes and the larger number of predicted than labelled findings should be considered in any downstream use. No post-test tuning was performed.
