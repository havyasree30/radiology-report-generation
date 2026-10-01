# DenseNet121 baseline: validation analysis

**Experiment.** One DenseNet121 run on CheXpert frontal images, trained on 2026-09-30.

**Status.**
- Validation evaluation and Youden thresholds are complete.
- **The locked test split has not been opened.**
- The official CheXpert valid set has not been used.

> **C1 finalization note (2026-10-01).** This document describes the ORIGINAL evaluation, which used AMP fp16 inference. The official C1 reference numbers come from the canonical fp32_strict re-evaluation of the same checkpoint: see `../C1_BASELINE_SUMMARY.md`.
>
> **Corrections made after the independent verification:**
> - The LR column for epochs 6 and 7 was corrected against `training_history.csv`: the drop to 1e-6 happened after epoch 7, not after epoch 6.
> - `validation_predictions.csv.gz` was re-serialised losslessly (17 digits, plus `.npz`).
> - `operating_points*` were recomputed from exact probabilities.
>
> PPVs and counts quoted in §5 may differ from the corrected files by rounding-level amounts (max PPV change in `../C1_FINALIZATION_LOG.json`).

**Sources.** Every number below comes from files in this directory: `training_history.csv`, `per_class_metrics.csv`, `summary_metrics.json`, `thresholds/youden_j_thresholds.json`, `operating_points*.{csv,json}` and `environment.json`.

## 1. Setup

| Item | Value |
|---|---|
| **Data** | Phase 1 stratified patient-group split (seed 42), frontal (AP/PA) images only |
| **Train** | 133,644 images, 45,324 patients (5 confirmed blank images excluded) |
| **Validation** | 28,671 images, 9,530 patients (1 blank image excluded) |
| **Labels** | positive 1, negative 0, uncertain masked (U-Ignore), blank → 0 (`../LABEL_POLICY.md`) |
| **Preprocessing** | grayscale → longest side 224 (PIL bilinear) → centre-pad to 224×224 with 0 → replicate to 3 channels → ImageNet mean/std |
| **Augmentation** (train only) | rotation ±5°, translation ±5%, scale 0.95–1.05; no flips, no colour jitter, no CLAHE |
| **Model** | torchvision DenseNet121 (IMAGENET1K_V1) with the classifier replaced by `Linear(1024, 14)`; outputs logits |
| **Loss** | masked BCE-with-logits, `reduction="none"`, normalised over valid entries; √-tempered training-only pos_weight |
| **Optimiser** | AdamW, lr 1e-4, weight decay 1e-4, betas (0.9, 0.999) |
| **Scheduler** | ReduceLROnPlateau on val macro AUROC (factor 0.1, patience 1) |
| **Early stopping** | patience 3 on val macro AUROC; max 8 epochs |
| **Batch** | 32 physical = 32 effective (no accumulation); AMP fp16 |
| **Environment** | torch 2.11.0+cu128, CUDA 12.8, RTX 3050 6 GB Laptop GPU; seed 42 with cuDNN deterministic |

## 2. Why this loss

Phase 1 found effective negative:positive ratios from 0.92:1 to 61.7:1, so an unweighted BCE baseline would under-train the rare labels. The training-split pos_weights recomputed for this run (`train_pos_weights.csv`) range from 0.77 to **74.8**.

**The risk with raw weights.**
- With a raw weight of 74.8, one Pleural Other positive contributes as much gradient as about 75 negatives.
- Pleural Other prevalence is 1.3%. At batch size 32, about one batch in three contains at least one Pleural Other positive, which produces intermittent loss and gradient spikes under fp16.
- That trades noisy optimisation for rare-label recall.

**The choice: √-tempered weights.** The square root compresses the weight range to 0.88–8.65 while keeping the rare-class emphasis monotone in the imbalance.

**This is a documented heuristic, not a proven optimum.** Raw pos_weight, plain BCE, focal loss and ASL remain the planned comparison. They are implemented and tested but not run.

**Observed stability.**
- The median per-step gradient norm stayed between 1.05 and 1.45 in every epoch.
- The maximum was 4.78, in epoch 1; later epochs stayed at or below 2.85.
- 0–2 steps per epoch were skipped by the AMP loss scaler. That is normal fp16 scale calibration, not divergence.
- No non-finite loss occurred.

## 3. Training behaviour

| Epoch | Train loss | Val loss | Val macro AUROC | Val micro AUROC | LR |
|---:|---:|---:|---:|---:|---|
| 1 | 0.5082 | 0.4942 | 0.7848 | 0.8854 | 1e-4 |
| 2 | 0.4790 | 0.4845 | 0.7983 | 0.8837 | 1e-4 |
| 3 | 0.4652 | 0.4972 | 0.7939 | 0.8821 | 1e-4 |
| 4 | 0.4531 | 0.5146 | 0.7952 | 0.8630 | 1e-4 → 1e-5 |
| **5** | **0.4209** | **0.4855** | **0.8038** | **0.8916** | 1e-5 |
| 6 | 0.4093 | 0.4901 | 0.8035 | 0.8938 | 1e-5 |
| 7 | 0.4018 | 0.4943 | 0.8019 | 0.8901 | 1e-5 → 1e-6 |
| 8 | 0.3932 | 0.4955 | 0.8015 | 0.8909 | 1e-6 |

**Selected checkpoint.** **Epoch 5**, chosen by validation macro AUROC. Training ran all 8 epochs; the counter reached 3 non-improving epochs exactly at the maximum.

**Timing.**
- Each epoch took 818–912 s of training plus 56–91 s of validation.
- Total wall time was 7,375 s (2 h 03 min).
- Peak GPU memory was 2.28 GB.

**Overfitting.** There is mild overfitting from epoch 3 onwards:
- Training loss falls monotonically, from 0.508 to 0.393.
- Validation loss reaches its minimum at epoch 2 (0.4845) and ends at 0.4955.
- Validation macro AUROC plateaus at about 0.80 after the learning-rate drop.

The selected epoch 5 sits at the plateau's peak, and its validation loss (0.4855) is close to the minimum. More epochs at this learning rate would not help. Better generalisation more likely needs regularisation or data-side changes, which belong to later controlled experiments.

**No underfitting signal.** Training loss is still falling, and the model separates most classes well above chance.

## 4. Validation performance (epoch 5, 28,671 images)

| Summary | Value |
|---|---:|
| Macro AUROC (14 classes, all defined) | **0.8038** |
| Micro AUROC | **0.8916** |
| Macro AUPRC | **0.4237** |
| Micro AUPRC | **0.6709** |
| Mean AUROC over the 5 commonly reported CheXpert labels | 0.8144 |

The five labels are Atelectasis, Cardiomegaly, Consolidation, Edema and Pleural Effusion.

**Per-class results.** Support counts valid entries only, since uncertain entries are masked. Operating points are at the Youden threshold.

| Observation | Valid n | Positives | Prevalence | AUROC | AUPRC | Youden threshold | Sensitivity | Specificity | J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 28,671 | 2,538 | 0.089 | 0.8838 | 0.4814 | 0.3242 | 0.822 | 0.809 | 0.630 |
| Enlarged Cardiomediastinum | 27,122 | 1,373 | 0.051 | **0.6984** | 0.1423 | 0.1726 | 0.561 | 0.730 | 0.291 |
| Cardiomegaly | 27,658 | 3,465 | 0.125 | 0.8685 | 0.5646 | 0.2309 | 0.776 | 0.813 | 0.590 |
| Lung Opacity | 27,998 | 14,127 | 0.505 | 0.7366 | 0.7094 | 0.5010 | 0.733 | 0.616 | 0.349 |
| Lung Lesion | 28,502 | 1,064 | 0.037 | 0.7773 | 0.1935 | 0.1402 | 0.614 | 0.794 | 0.408 |
| Edema | 26,875 | 7,426 | 0.276 | 0.8526 | 0.6787 | 0.3122 | 0.816 | 0.733 | 0.549 |
| Consolidation | 24,998 | 1,960 | 0.078 | 0.7497 | 0.2049 | 0.1791 | 0.758 | 0.619 | 0.377 |
| Pneumonia | 26,242 | 715 | 0.027 | 0.7698 | 0.1301 | 0.1144 | 0.681 | 0.724 | 0.405 |
| Atelectasis | 24,115 | 4,462 | 0.185 | **0.7159** | 0.3552 | 0.2766 | 0.723 | 0.591 | 0.314 |
| Pneumothorax | 28,263 | 2,632 | 0.093 | 0.8648 | 0.4985 | 0.3390 | 0.747 | 0.826 | 0.573 |
| Pleural Effusion | 27,209 | 11,543 | 0.424 | 0.8852 | 0.8395 | 0.4066 | 0.832 | 0.779 | 0.611 |
| Pleural Other | 28,400 | 381 | 0.013 | 0.7966 | **0.0829** | 0.0329 | 0.698 | 0.760 | 0.458 |
| Fracture | 28,590 | 1,092 | 0.038 | 0.8004 | 0.1881 | 0.0554 | 0.758 | 0.708 | 0.466 |
| Support Devices | 28,543 | 16,003 | 0.561 | 0.8534 | 0.8625 | 0.4309 | 0.809 | 0.768 | 0.577 |

**Weak classes.**
- **Low AUROC:** Enlarged Cardiomediastinum (0.698), Atelectasis (0.716), Lung Opacity (0.737) and Consolidation (0.750).
- **Low AUPRC:** every rare class has an AUPRC below 0.21 (Pleural Other 0.083, Pneumonia 0.130, Enlarged Cardiomediastinum 0.142, Fracture 0.188, Lung Lesion 0.194). Each is still well above its prevalence, the chance level.
- **Why these labels are hard:** Enlarged Cardiomediastinum, Atelectasis, Consolidation and Pneumonia are exactly the labels Phase 1 flagged for heavy uncertainty (a 52–76% uncertain share of positive-leaning mentions). Their valid supports are the smallest because uncertain entries are masked.

## 5. Operating-point behaviour (important for Phase 3)

These numbers are from `operating_points_summary.json` and `operating_points.csv`.

**Youden's J weights sensitivity and specificity equally, ignoring prevalence.** For rare classes this yields low thresholds and many false positives:

- **Positives per image:** the model predicts a mean of **5.09** positive observations per validation image (median 5), against 2.40 explicit true positives.
- **Every image gets at least one positive:** 0 images have no predicted positive, so the "no positive findings" path is never exercised on validation data.
- **Precision (PPV) at the thresholds:**
  - Pleural Other 0.038, Pneumonia 0.065, Fracture 0.093
  - Enlarged Cardiomediastinum 0.100, Lung Lesion 0.104, Consolidation 0.145
  - Support Devices 0.816, Pleural Effusion 0.735
  - Some "false positives" may be findings that were present but unmentioned (blank → 0).
- **No Finding contradictions:** the "No Finding positive together with a pathology positive" warning fires on **6,529 of 28,671 images (22.8%)**.

**Consequence.** Retrieval queries built from these statuses will often list rare findings that are not present. This is the requested Youden method working as specified, not a bug.

It must, however, be addressed or explicitly accepted before Phase 3, for example:
- a precision-aware operating point, fitted on validation only;
- query weighting by probability margin;
- letting the Multi-Agent RAG treat low-PPV classes as tentative.

No alternative has been implemented or selected.

## 6. Example (actual validation image)

**Image:** `CheXpert-v1.0-small/train/patient00019/study4/view1_frontal.jpg`. It passes validation with no warnings; the full output is in `example_inference.json`.

- **Reference labels:** positive for Cardiomegaly, Lung Opacity, Lung Lesion, Atelectasis, Pneumothorax and Pleural Effusion; Support Devices = 0; everything else blank.
- **Correct positives:** Lung Opacity, Lung Lesion, Atelectasis, Pleural Effusion.
- **Missed:** Cardiomegaly (p = 0.037) and Pneumothorax (p = 0.130).
- **Extra positives:** Consolidation and Pneumonia, which are blank in the reference.
- **Consistency:** single-image fp32 inference reproduces the batch fp16 validation probabilities to within 0.0010.

## 7. Validator audit

`../validator_audit/` covers all 162,321 frontal train and val images. Only the six Phase 1 blank images are hard-rejected, and it is exactly the same set. The remaining checks produced warnings only: 13 unusual aspect ratios, 9 dominant flat regions and 2 very bright images.

## 8. Limitations

- **A single seed and a single configuration.** No confidence intervals, and no comparison against other losses or uncertainty policies yet.
- **Label noise.** Validation labels come from the automatic report labeler, blanks are scored as negatives and uncertain entries are excluded.
- **Code provenance.** The checkpoint's recorded git commit (`0304f42`) predates the uncommitted Phase 2 code (`dirty_working_tree: true` in `environment.json`). Commit the code to fix this provenance.
