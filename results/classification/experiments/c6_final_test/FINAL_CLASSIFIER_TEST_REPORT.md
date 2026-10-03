# Final classifier locked-test report (C6)

_Generated from the C6 artifacts by `scripts/c6_05_write_docs.py`._

## 1. Objective

Evaluate the frozen classification pipeline once on the locked test split and freeze it as the classifier for downstream retrieval and report generation. No tuning followed the test evaluation; results are reported as obtained.

## 2. Frozen configuration

DenseNet-121, √-weighted BCE, checkpoint SHA-256 `6645ed44be7cb7d4cf4679b1fd561c7ac36be63f964ea320666199883126794b`; C4 F1-optimal raw-score thresholds with the No Finding rule (12 pathology findings suppress it, Support Devices does not); C5 Platt calibration for all 14 classes. The full configuration, with hashes of every frozen file, was written to `FINAL_CLASSIFIER_FREEZE_MANIFEST.json` at 2026-10-03T05:00:47+00:00, and the test inference finished at 2026-10-03T05:04:44+00:00. Both are re-verified by `c6_integrity_report.json`. See `FINAL_CLASSIFIER_SPECIFICATION.md` for every parameter.

## 3. Test-set integrity

- Images (frontal, after the frozen exclusion list): train 133,644, validation 28,671, test 28,706. Patients: train 45,324, validation 9,530, test 9,680.
- Overlap counts (patients / split groups / image paths): train ∩ val = 0 / 0 / 0; train ∩ test = 0 / 0 / 0; val ∩ test = 0 / 0 / 0.
- Status: **PASS**. Provenance: stratified_group split, seed 42, unit = split_group_id (patient merged with patients sharing byte-identical images).
- The `locked_test_split_used` flag is false in every C3–C5 result file that records it; no code before C6 unlocked the test split; no result file before C6 carries test predictions. Thresholds and calibrators were fitted on validation only, and model/loss/checkpoint selection used validation only.
- Inference: one pass, 28,706 images in 134 s, policy `fp32_strict`, no test-time augmentation, no ensembling; scores range 8.7e-07–0.9989.
- Validation reference: recomputed with the same functions and identical to the stored C2–C5 results (`validation_reproduction.json`: reproduced = True).

## 4. Ranking results

Macro AUROC 0.803 (95% CI 0.799 to 0.806), macro AUPRC 0.427 (95% CI 0.422 to 0.433); micro AUROC 0.893, micro AUPRC 0.672. Per-class AUROC ranges from 0.682 (Enlarged Cardiomediastinum) to 0.888 (Pleural Effusion); AUPRC from 0.093 (Pleural Other) to 0.873 (Support Devices).

**Table 2. Per-class performance.**

| Observation | Prevalence | Positives | AUROC | AUPRC | Threshold | Precision | Recall | Specificity | F1 | Bal. acc. | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 0.088 | 2530 | 0.885 | 0.471 | 0.643 | 0.491 | 0.496 | 0.950 | 0.494 | 0.723 | 1255 | 1300 | 24876 | 1275 |
| Enlarged Cardiomediastinum | 0.051 | 1400 | 0.682 | 0.140 | 0.304 | 0.134 | 0.319 | 0.889 | 0.189 | 0.604 | 447 | 2880 | 22972 | 953 |
| Cardiomegaly | 0.127 | 3527 | 0.867 | 0.558 | 0.452 | 0.500 | 0.606 | 0.911 | 0.548 | 0.759 | 2139 | 2138 | 22015 | 1388 |
| Lung Opacity | 0.504 | 14172 | 0.730 | 0.696 | 0.327 | 0.607 | 0.897 | 0.409 | 0.724 | 0.653 | 12707 | 8238 | 5693 | 1465 |
| Lung Lesion | 0.037 | 1068 | 0.784 | 0.213 | 0.514 | 0.244 | 0.274 | 0.967 | 0.258 | 0.621 | 293 | 906 | 26561 | 775 |
| Edema | 0.278 | 7481 | 0.859 | 0.683 | 0.379 | 0.594 | 0.738 | 0.806 | 0.658 | 0.772 | 5519 | 3767 | 15692 | 1962 |
| Consolidation | 0.079 | 1979 | 0.751 | 0.209 | 0.333 | 0.196 | 0.447 | 0.843 | 0.273 | 0.645 | 884 | 3622 | 19400 | 1095 |
| Pneumonia | 0.027 | 723 | 0.750 | 0.122 | 0.381 | 0.142 | 0.232 | 0.960 | 0.176 | 0.596 | 168 | 1014 | 24603 | 555 |
| Atelectasis | 0.184 | 4457 | 0.723 | 0.363 | 0.330 | 0.303 | 0.654 | 0.661 | 0.414 | 0.658 | 2917 | 6701 | 13091 | 1540 |
| Pneumothorax | 0.095 | 2693 | 0.861 | 0.513 | 0.615 | 0.494 | 0.534 | 0.942 | 0.513 | 0.738 | 1439 | 1475 | 24152 | 1254 |
| Pleural Effusion | 0.424 | 11597 | 0.888 | 0.844 | 0.362 | 0.717 | 0.876 | 0.745 | 0.788 | 0.810 | 10157 | 4017 | 11730 | 1440 |
| Pleural Other | 0.013 | 378 | 0.809 | 0.093 | 0.254 | 0.113 | 0.249 | 0.974 | 0.156 | 0.611 | 94 | 735 | 27358 | 284 |
| Fracture | 0.039 | 1128 | 0.779 | 0.199 | 0.326 | 0.261 | 0.285 | 0.967 | 0.272 | 0.626 | 321 | 910 | 26596 | 807 |
| Support Devices | 0.565 | 16125 | 0.868 | 0.873 | 0.343 | 0.792 | 0.885 | 0.698 | 0.836 | 0.792 | 14277 | 3754 | 8686 | 1848 |

## 5. Binary results

Under the frozen policy the macro (14-class) precision is 0.399 (95% CI 0.394 to 0.404), recall 0.535 (95% CI 0.528 to 0.542), specificity 0.837 (95% CI 0.835 to 0.840), F1 0.450 (95% CI 0.445 to 0.455) and balanced accuracy 0.686 (95% CI 0.683 to 0.690). Excluding No Finding (13 classes) the macro F1 is 0.447. Pooled micro precision / recall / F1 are 0.559 / 0.760 / 0.644.

**Table 1. Overall locked-test performance.**

| Metric | Locked test | 95% CI (patient bootstrap) |
|---|---:|---:|
| Macro AUROC (14 classes) | 0.803 | 0.799 to 0.806 |
| Micro AUROC | 0.893 | n/a |
| Macro AUPRC (14 classes) | 0.427 | 0.422 to 0.433 |
| Micro AUPRC | 0.672 | n/a |
| Macro precision | 0.399 | 0.394 to 0.404 |
| Macro recall | 0.535 | 0.528 to 0.542 |
| Macro specificity | 0.837 | 0.835 to 0.840 |
| Macro F1 | 0.450 | 0.445 to 0.455 |
| Macro balanced accuracy | 0.686 | 0.683 to 0.690 |
| Micro precision / recall / F1 (pooled) | 0.559 / 0.760 / 0.644 | n/a |
| Macro F1 excluding No Finding (13 classes) | 0.447 | n/a |
| Macro Brier score (calibrated) | 0.0837 | 0.0829 to 0.0844 |
| Macro log loss (calibrated) | 0.2768 | n/a |
| Macro ECE (calibrated) | 0.0069 | n/a |

## 6. Calibration results

With the frozen Platt calibrators the macro Brier score is 0.0837 (95% CI 0.0829 to 0.0844), the macro log loss 0.2768 and the macro ECE 0.0069 (10 adaptive bins, as in C5). For reference the raw scores give 0.0960, 0.3116 and 0.0648. The calibrated Brier score and log loss are lower than the raw values in 14 and 14 of 14 classes, respectively. The largest per-class ECE is 0.0206 (Support Devices).

**Table 5. Locked-test calibration.**

| Observation | Prevalence | Mean calibrated probability | Brier | Log loss | ECE | Raw-score Brier | Raw-score ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 0.088 | 0.092 | 0.0585 | 0.2020 | 0.0071 | 0.0894 | 0.1188 |
| Enlarged Cardiomediastinum | 0.051 | 0.052 | 0.0469 | 0.1896 | 0.0033 | 0.0687 | 0.1037 |
| Cardiomegaly | 0.127 | 0.126 | 0.0783 | 0.2646 | 0.0046 | 0.0878 | 0.0603 |
| Lung Opacity | 0.504 | 0.504 | 0.2085 | 0.6036 | 0.0122 | 0.2094 | 0.0317 |
| Lung Lesion | 0.037 | 0.037 | 0.0327 | 0.1354 | 0.0031 | 0.0510 | 0.0705 |
| Edema | 0.278 | 0.276 | 0.1333 | 0.4083 | 0.0093 | 0.1350 | 0.0282 |
| Consolidation | 0.079 | 0.079 | 0.0680 | 0.2457 | 0.0068 | 0.0902 | 0.1065 |
| Pneumonia | 0.027 | 0.027 | 0.0255 | 0.1128 | 0.0020 | 0.0385 | 0.0583 |
| Atelectasis | 0.184 | 0.183 | 0.1354 | 0.4269 | 0.0072 | 0.1519 | 0.1088 |
| Pneumothorax | 0.095 | 0.093 | 0.0622 | 0.2210 | 0.0060 | 0.0853 | 0.0962 |
| Pleural Effusion | 0.424 | 0.423 | 0.1334 | 0.4145 | 0.0102 | 0.1337 | 0.0153 |
| Pleural Other | 0.013 | 0.013 | 0.0126 | 0.0605 | 0.0020 | 0.0174 | 0.0207 |
| Fracture | 0.039 | 0.038 | 0.0346 | 0.1419 | 0.0023 | 0.0398 | 0.0299 |
| Support Devices | 0.565 | 0.565 | 0.1418 | 0.4485 | 0.0206 | 0.1463 | 0.0585 |
| Macro mean |  |  | 0.0837 | 0.2768 | 0.0069 | 0.0960 | 0.0648 |

## 7. Rare-class analysis

The six rare classes are those defined in C2 (not redefined from test prevalence).

| Observation | Split | Prevalence | AUROC | AUPRC | Precision | Recall | Specificity | F1 | TP | FP | FP per TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pleural Other | validation | 0.013 | 0.786 | 0.076 | 0.118 | 0.257 | 0.974 | 0.162 | 98 | 732 | 7.47 |
| Pleural Other | test | 0.013 | 0.809 | 0.093 | 0.113 | 0.249 | 0.974 | 0.156 | 94 | 735 | 7.82 |
| Pneumonia | validation | 0.027 | 0.762 | 0.134 | 0.151 | 0.271 | 0.957 | 0.194 | 194 | 1091 | 5.62 |
| Pneumonia | test | 0.027 | 0.750 | 0.122 | 0.142 | 0.232 | 0.960 | 0.176 | 168 | 1014 | 6.04 |
| Fracture | validation | 0.038 | 0.797 | 0.196 | 0.256 | 0.283 | 0.967 | 0.269 | 309 | 898 | 2.91 |
| Fracture | test | 0.039 | 0.779 | 0.199 | 0.261 | 0.285 | 0.967 | 0.272 | 321 | 910 | 2.83 |
| Lung Lesion | validation | 0.037 | 0.770 | 0.190 | 0.235 | 0.281 | 0.964 | 0.256 | 299 | 976 | 3.26 |
| Lung Lesion | test | 0.037 | 0.784 | 0.213 | 0.244 | 0.274 | 0.967 | 0.258 | 293 | 906 | 3.09 |
| Enlarged Cardiomediastinum | validation | 0.051 | 0.687 | 0.137 | 0.145 | 0.313 | 0.902 | 0.199 | 430 | 2527 | 5.88 |
| Enlarged Cardiomediastinum | test | 0.051 | 0.682 | 0.140 | 0.134 | 0.319 | 0.889 | 0.189 | 447 | 2880 | 6.44 |
| Consolidation | validation | 0.078 | 0.746 | 0.204 | 0.190 | 0.435 | 0.843 | 0.265 | 853 | 3625 | 4.25 |
| Consolidation | test | 0.079 | 0.751 | 0.209 | 0.196 | 0.447 | 0.843 | 0.273 | 884 | 3622 | 4.10 |

Pleural Other: precision 0.113, recall 0.249, 7.82 false positives per true positive on test (7.47 on validation). Across the six classes the test precision ranges 0.113–0.261 and the false positives per true positive 2.83–7.82.

## 8. No Finding analysis

- Test prevalence 0.088; after the rule: precision 0.491, recall 0.496, specificity 0.950, F1 0.494; 8.90% of images are predicted No Finding.
- Before the rule 11.87% of images were predicted No Finding, and 851 images (2.96% of images; 25.0% of those predicted No Finding) also had a pathology finding predicted positive. After the rule: 0. Images modified: 851.
- Ground-truth contradictions in the test labels (No Finding with an explicit positive among the 12 pathology labels): 0; No Finding together with Support Devices: 1096.
- Pre-rule vs post-rule No Finding F1: 0.518 vs 0.494 (the rule trades recall for consistency; it was not changed).

## 9. Validation-to-test generalization

**Table 3. Validation versus locked test.**

| Metric | Validation | Locked test | Test - validation |
|---|---:|---:|---:|
| macro auroc | 0.8011 | 0.8026 | +0.0014 |
| micro auroc | 0.8922 | 0.8932 | +0.0010 |
| macro auprc | 0.4235 | 0.4269 | +0.0034 |
| micro auprc | 0.6674 | 0.6720 | +0.0046 |
| macro precision | 0.3985 | 0.3992 | +0.0007 |
| macro recall | 0.5354 | 0.5352 | -0.0002 |
| macro specificity | 0.8374 | 0.8374 | +0.0000 |
| macro f1 | 0.4495 | 0.4500 | +0.0005 |
| macro balanced accuracy | 0.6864 | 0.6863 | -0.0001 |
| macro brier | 0.0839 | 0.0837 | -0.0002 |
| macro log loss | 0.2774 | 0.2768 | -0.0006 |
| macro ece | 0.0056 | 0.0069 | +0.0013 |

Differences are descriptive only. Macro AUROC differs by +0.0014, macro AUPRC by +0.0034, macro F1 by +0.0005, macro Brier by -0.0002 and macro ECE by +0.0013. Per-class AUROC differences range -0.017 (Fracture) to +0.023 (Pleural Other). The validation calibration values are in-sample for the frozen calibrators (out-of-fold macro ECE 0.0056).

## 10. Bootstrap uncertainty

1000 patient-level resamples (seed 42); all images of a resampled patient are repeated together; model, thresholds and calibrators are held fixed, so the intervals reflect test-sample variability only.

| Metric | Point estimate | 95% CI |
|---|---:|---|
| macro_auroc | 0.8026 | 0.7988 to 0.8062 |
| macro_auprc | 0.4269 | 0.4218 to 0.4332 |
| macro_brier | 0.0837 | 0.0829 to 0.0844 |
| macro_precision | 0.3992 | 0.3945 to 0.4039 |
| macro_recall | 0.5352 | 0.5283 to 0.5415 |
| macro_specificity | 0.8374 | 0.8348 to 0.8400 |
| macro_f1 | 0.4500 | 0.4451 to 0.4547 |
| macro_balanced_accuracy | 0.6863 | 0.6826 to 0.6896 |

## 11. Finding counts and error analysis

Findings per image (13 observations excluding No Finding): true mean 2.32 (median 2, 25th–75th percentile 1–3, 95th percentile 5, maximum 7); predicted mean 3.43 (median 4, 2–5, 95th 6, maximum 8). Share of images with 0 / 1 / 2–3 / 4–5 / >5 findings: true 9.4 / 19.9 / 51.0 / 19.0 / 0.7%; predicted 6.8 / 10.0 / 30.1 / 42.7 / 10.4%. The policy therefore predicts more findings per image than the labels contain, which downstream retrieval and report generation must tolerate.

`test_error_examples.csv` lists, for every class, the five highest-scoring false positives and five lowest-scoring false negatives (140 rows; anonymised identifiers). Among the false-positive examples the median raw score is 0.972 and 63 of 70 come from images with at least one other true positive label; among the false-negative examples the median raw score is 0.004. This is a descriptive listing only; no radiological interpretation was made and nothing was changed on its basis.

## 12. Final classifier freeze

The classifier, thresholds, No Finding rule and calibrators are frozen as documented in `FINAL_CLASSIFIER_SPECIFICATION.md`; the downstream output record is defined in `FINAL_CLASSIFIER_OUTPUT_SCHEMA.json`. No further tuning will be performed using this test set.

## 13. Limitations

- Single dataset (CheXpert, one institution) and frontal views only; no external validation.
- Reference labels are automatically extracted from reports (CheXpert labeler), with uncertain labels excluded: 0.0–15.5% of images are excluded per class (Atelectasis the most). Metrics describe agreement with these labels, not radiologist-verified findings.
- Several classes have modest discrimination: AUROC 0.682 for Enlarged Cardiomediastinum, and macro precision is 0.399; F1-optimal thresholds favour recall, so false positives are frequent for rare classes (7.8 per true positive at most).
- The policy predicts more findings per image (3.43) than are labelled (2.32).
- Bootstrap intervals cover test-sample variability only, not training variability, threshold or calibrator uncertainty.
- Calibrated probabilities estimate P(CheXpert label positive | image) under the project label policy, not clinical ground-truth probabilities; calibration was fitted on the same validation set used to select thresholds and may drift under dataset shift.
- No expert review of errors, no assessment of clinical utility or safety; this is a research prototype component, not a diagnostic system.
