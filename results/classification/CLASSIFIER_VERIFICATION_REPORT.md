# DenseNet121 baseline: independent verification audit

**Audited experiment:** `results/classification/densenet121_baseline/`, best checkpoint, epoch 5.
**Audit date:** 2026-10-01.

**How the audit was produced.** `scripts/verify_classifier.py` generated every number here. It does **not** reuse the pipeline's metric or threshold code:
- Ground truth is re-read from the original `train.csv`.
- Metrics come from sklearn directly.
- Youden thresholds are recomputed by brute force over every distinct cut-off.
- Validation inference was re-run twice from the saved checkpoint.

**Machine-readable outputs:**
- `results/classification/CLASSIFIER_VERIFICATION.csv`
- `results/classification/verification/` (`audit_results.json`, tables, figures)

**Constraints respected.**
- **No retraining.** The checkpoint, thresholds, preprocessing and split are unchanged.
- **The locked test split was not evaluated.** Its rows were read from the manifest only to prove they were never used.
- **All 41 pre-existing files** under `results/classification/` are byte-identical (SHA-256) before and after the audit.

**Status legend.**
- **VERIFIED:** checked and correct.
- **WARNING:** correct or usable, but an issue should be addressed.
- **FAILED:** cannot currently be trusted.

## Summary

| § | Audit item | Status |
|---|---|---|
| 1 | Checkpoint | **WARNING** (correct; provenance commit predates the Phase 2 code) |
| 2 | Validation population | **VERIFIED** |
| 3 | Locked-test leakage | **VERIFIED** (9/9 PASS) |
| 4 | Prediction integrity | **VERIFIED** |
| 5 | Metric recomputation | **VERIFIED** (all differences ≤ 5.6e-17) |
| 6 | Prevalence baseline | **WARNING** (four rare classes have very low normalised lift) |
| 7 | ROC curves | **VERIFIED** |
| 8 | PR / AUPRC definition | **VERIFIED** |
| 9 | Youden thresholds | **VERIFIED** (bit-exact on the model's in-memory outputs) |
| 9b | Saved-prediction precision | **WARNING: confirmed artifact bug** (see below) |
| 10 | Confusion matrices | **VERIFIED** |
| 11 | Threshold sanity | **WARNING** (two thresholds below 0.10; one class with fewer than 500 positives) |
| 14 | Training history | **VERIFIED** (mild overfitting, already documented) |
| 15 | Leakage / duplicates | **VERIFIED** |
| 17 | Score distributions | **WARNING** (8 classes with substantial positive/negative overlap) |
| 20–21 | Structured findings and query builder | **VERIFIED** (one caveat on the contradiction case) |
| 22 | Artifact and documentation consistency | **WARNING** (2 wrong LR cells in `BASELINE_ANALYSIS.md`) |
| 23 | Reproducibility | **WARNING** (deterministic; fp32 single-image path differs from the fp16 path used for thresholds) |

**Final verdict: VERIFIED WITH WARNINGS — usable, but issues should be addressed.** This verdict covers technical validity and reproducibility only. It is **not** a clinical validation.

---

## Bugs found (reported, NOT fixed: awaiting approval)

> **Erratum (C1 finalization, 2026-10-01): the mechanism stated in B1 is wrong; its impact numbers stand.**
> - **What was wrong:** the original CSV was *written* losslessly (pandas uses Python's shortest round-trip repr). Re-reading it with `float_precision="round_trip"` recovers every value exactly.
> - **Actual root cause:** pandas' **default `read_csv` float parser** is not correctly rounded. That parser was used by this audit and by `analyze_operating_points.py`.
> - **Why the impact numbers stand:** the counts below were measured with that default parser and remain correct.
> - **Fix:** a lossless `.npz` plus a round-trip CSV reader. See `C1_BASELINE_SUMMARY.md` and `C1_FINALIZATION_LOG.json`.

**B1. Saved validation predictions are not bit-exact (confirmed bug, small impact).**
`validation_predictions.csv.gz` stores probabilities with 16 significant digits (pandas default). An exact float64 round-trip needs 17.

Two facts make that matter:
- Youden thresholds lie exactly **on** observed scores.
- fp16 inference produces many tied scores. For example, 16 Pneumonia images share the exact Pneumonia threshold value.

**Impact:**
- 77 valid label entries tie exactly at their class threshold. After the round-trip, 69 of them fall about 1e-17 below it, so `p >= threshold` flips from positive to negative for those entries (out of 383,186 valid label decisions).
- Statistics recomputed from the CSV shift slightly: sensitivity by up to 0.0026 (Pleural Other, 1 of 381 positives) and specificity by up to 0.0006.
- AUROC and AUPRC are **unaffected** (difference 0).

**What is correct:**
- The stored thresholds, sensitivities and specificities are correct. On the bit-exact probabilities from re-inference they reproduce with difference 0.0 (≤ 1.1e-16).
- Downstream inference uses in-memory probabilities, so it is not affected.

**What is affected:** anything computed *from the CSV*. `operating_points.csv` (from `scripts/analyze_operating_points.py`) is slightly off. For example, its Enlarged Cardiomediastinum PPV is 0.100, against 0.0998 from bit-exact probabilities.

**Proposed fix** (not applied):
1. Write probabilities with `float_format="%.17g"`, or as a binary `.npz`.
2. Regenerate `validation_predictions.csv.gz`, `per_class_metrics.csv` and `operating_points*` from the same checkpoint. No retraining is needed, and the thresholds would not change.

**B2. Documentation error in `BASELINE_ANALYSIS.md` §3.** The LR column claims the learning rate dropped 1e-5 → 1e-6 after **epoch 6**.
- `training_history.csv` shows LR 1e-5 for epochs 5–7 and 1e-6 only in epoch 8, so the drop came after epoch 7.
- That matches ReduceLROnPlateau with patience 1: two non-improving epochs are needed (6 and 7).
- The same mis-statement was made in chat during training.
- Every other number in that document matched its artifact.

**W1. Precision mismatch between threshold fitting and single-image inference (design issue).**
- Thresholds were fitted on fp16-autocast batch outputs, but `scripts/infer_classifier.py` runs fp32.
- On 512 deterministically sampled validation images, fp32 and fp16 probabilities differ by at most 0.0053 (median 0.00019), changing **5 of 7,168** label decisions (0.07%).
- Inference and threshold fitting should use the same numerical path.

---

## 1. Checkpoint — WARNING

| Item | Value |
|---|---|
| Path | `results/classification/densenet121_baseline/checkpoints/best.pt` |
| Size | 28,439,162 bytes (SHA-256 `e1c38989…7e6e`) |
| Architecture | `densenet121`; the state dict loads with `strict=True` into a fresh torchvision DenseNet121 whose head is `Linear(1024, 14)` |
| Output dimension | 14 (classifier weight shape `[14, 1024]`) |
| Label order | identical to the dataset/config order (14/14) |
| Best epoch | 5 (metadata) = 5 (argmax of `training_history.csv`) = 5 (`training_summary.json`) = 5 (threshold provenance) |
| Selection metric | `val_macro_auroc` = 0.8037923325159262 (metadata and history identical) |
| Metadata | embedded metadata = `best.meta.json`; label policy, loss and preprocessing equal the config; `subset = null` (not the smoke run) |
| Git commit | `0304f4271967689dcc91a201665c38572c720991`, with `dirty_working_tree: true` |

**Warning:** the recorded commit is the last Phase 1 commit. The Phase 2 training code was uncommitted when the model was trained, and is still uncommitted, so the checkpoint cannot be tied to a code revision.

## 2. Validation population — VERIFIED

| Item | Count |
|---|---:|
| Validation patients (all views) | 9,531 |
| Validation studies (all views) | 28,110 |
| Validation images before filtering | 33,512 |
| After frontal-only filtering | 28,672 |
| Excluded | 1: `patient44163/study1/view1_frontal.jpg`, reason `near_blank` (visually confirmed in Phase 1) |
| Evaluated images = prediction rows | **28,671 = 28,671** |
| Evaluated patients / studies | 9,530 / 28,109 |
| Duplicate prediction paths | 0 |
| Prediction path set = expected path set | yes |
| Train ∩ validation patients | **∅ (0)** |

## 3. Locked-test leakage audit — VERIFIED

| Item | Result | Evidence |
|---|---|---|
| Training | PASS | The recorded 133,644 training images equal the manifest's train frontal images minus exclusions; 0 are test paths |
| Early stopping / checkpoint selection | PASS | Monitor `val_macro_auroc` on `val_split = val` |
| Threshold optimisation | PASS | Provenance `source_split = val`, n = 28,671 = prediction rows; 0 test paths or patients |
| Preprocessing selection | PASS | Fixed a priori in config; no search |
| Class weights | PASS | `train_pos_weights.csv` counts exactly equal counts recomputed from TRAIN images only |
| Calibration | PASS | No calibration step exists |
| Hyperparameter tuning | PASS | Single config-fixed run |
| Code never unlocks the test split | PASS | `allow_locked=True` occurs 0 times in training, evaluation and inference code |
| Validation labels not used in training | PASS | Training label counts come from the train split only; 0 shared patients |

## 4. Prediction integrity — VERIFIED

**Range and validity.**
- No NaN, no ±inf, nothing below 0 or above 1.
- No all-zero or all-one rows, and no collapsed class (every class has a standard deviation ≥ 0.092 and 3,877–9,888 unique values).
- No single value covers more than 0.18% of a class.

**Identical prediction rows.** 5 rows have identical 14-vectors. All 10 images involved are pairs of **byte-identical images** found in Phase 1 (same patient), so identical outputs are expected.

**Sum of the 14 probabilities per image:**

| min | Q1 | median | mean | Q3 | max |
|---:|---:|---:|---:|---:|---:|
| 1.224 | 2.782 | 3.349 | 3.322 | 3.885 | 5.589 |

0.0% of images have a sum within 0.05 of 1. **There is no softmax normalisation; the outputs are independent sigmoids.**
- The model code: the forward pass returns logits, and inference applies `torch.sigmoid` element-wise.
- Figure: `verification/figures/probability_sum_distribution.png`.

Per-class statistics (min, Q1, median, mean, Q3, max, SD) are in `verification/probability_statistics.csv`.

## 5. Metric recomputation — VERIFIED

**Ground truth.** Labels re-read from the source CSV are identical to the saved labels and masks. Metrics are computed over valid entries only (uncertain excluded, blank = 0).

| Metric | Saved | Recomputed | Abs. diff |
|---|---:|---:|---:|
| Macro AUROC | 0.8037923325 | 0.8037923325 | 0 |
| Micro AUROC | 0.8916017873 | 0.8916017873 | 0 |
| Macro AUPRC | 0.4236865083 | 0.4236865083 | 0 |
| Micro AUPRC | 0.6708725125 | 0.6708725125 | 0 |
| 5-label CheXpert mean AUROC | 0.8143725381 | 0.8143725381 | 0 |

**Per class.** The AUROC difference is 0 for all 14 classes; the AUPRC difference is ≤ 5.6e-17. All positive and negative supports match `per_class_metrics.csv` and the threshold file.

## 6. Prevalence baseline — WARNING

AUPRC/prevalence cannot exceed 1/prevalence, so it understates common classes (for Support Devices the maximum is 1.78). The normalised lift (AUPRC − prevalence)/(1 − prevalence) is therefore also shown. On that scale 0 is random ranking and 1 is perfect.

| Observation | Pos | Neg | Prevalence | AUPRC | AUPRC / prev. | Normalised lift |
|---|---:|---:|---:|---:|---:|---:|
| No Finding | 2,538 | 26,133 | 0.089 | 0.481 | 5.44 | 0.431 |
| Enlarged Cardiomediastinum | 1,373 | 25,749 | 0.051 | 0.142 | 2.81 | **0.097** |
| Cardiomegaly | 3,465 | 24,193 | 0.125 | 0.565 | 4.51 | 0.502 |
| Lung Opacity | 14,127 | 13,871 | 0.505 | 0.709 | 1.41 | 0.414 |
| Lung Lesion | 1,064 | 27,438 | 0.037 | 0.193 | 5.18 | 0.162 |
| Edema | 7,426 | 19,449 | 0.276 | 0.679 | 2.46 | 0.556 |
| Consolidation | 1,960 | 23,038 | 0.078 | 0.205 | 2.61 | **0.137** |
| Pneumonia | 715 | 25,527 | 0.027 | 0.130 | 4.77 | **0.106** |
| Atelectasis | 4,462 | 19,653 | 0.185 | 0.355 | 1.92 | 0.209 |
| Pneumothorax | 2,632 | 25,631 | 0.093 | 0.498 | 5.35 | 0.447 |
| Pleural Effusion | 11,543 | 15,666 | 0.424 | 0.840 | 1.98 | 0.721 |
| Pleural Other | 381 | 28,019 | 0.013 | 0.083 | 6.18 | **0.070** |
| Fracture | 1,092 | 27,498 | 0.038 | 0.188 | 4.92 | 0.156 |
| Support Devices | 16,003 | 12,540 | 0.561 | 0.862 | 1.54 | 0.687 |

**Barely above the random baseline in absolute terms:** Pleural Other (0.083 vs 0.013), Enlarged Cardiomediastinum (0.142 vs 0.051), Pneumonia (0.130 vs 0.027) and Consolidation (0.205 vs 0.078). Their ratios look large only because their prevalence is tiny: each recovers ≤ 14% of the achievable precision gain.

## 7. ROC curves — VERIFIED

- All 14 classes have both positives and negatives, so there are no undefined classes.
- Every recomputed ROC starts at (0, 0), ends at (1, 1) and is monotone.
- `roc_auc_score` equals the trapezoidal integral of the ROC curve exactly (max difference 0.0).

## 8. Precision-recall — VERIFIED

**Definition.** The pipeline's "AUPRC" is `sklearn.metrics.average_precision_score`: step-wise average precision, Σ(Rₙ − Rₙ₋₁)·Pₙ, with no interpolation. A manual step-wise re-implementation agrees to within 2.2e-16.

**Alternative definition.** Trapezoidal integration of the PR curve would differ by −0.0013 to +0.00001 per class. It is **not** used.

**Project convention:** AUPRC = average precision. Historical values are unchanged.

## 9. Youden J thresholds — VERIFIED (see 9b)

**Bit-exact check.** On bit-exact in-memory probabilities (fresh re-inference), the brute-force threshold, argmax of TPR − FPR with ties going to the highest cut-off, **equals the stored threshold exactly for all 14 classes**. The stored sensitivity and specificity reproduce with difference 0.0 (≤ 1.1e-16).

**CSV check.** From the saved CSV, thresholds agree to ≤ 8.3e-17. Sensitivity and specificity differ slightly only where scores tie at the threshold (B1).

**Source.** All thresholds come from validation (`source_split = val`, 28,671 images). No test data was involved.

**Operating points** (bit-exact confusion counts; `CLASSIFIER_VERIFICATION.csv`, `_exact` columns):

| Observation | Threshold | Sens | Spec | Precision | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Finding | 0.3242 | 0.822 | 0.809 | 0.294 | 0.433 | 2,085 | 5,001 | 21,132 | 453 |
| Enlarged Cardiomediastinum | 0.1726 | 0.561 | 0.730 | 0.100 | 0.169 | 770 | 6,948 | 18,801 | 603 |
| Cardiomegaly | 0.2309 | 0.776 | 0.813 | 0.373 | 0.504 | 2,690 | 4,516 | 19,677 | 775 |
| Lung Opacity | 0.5010 | 0.733 | 0.616 | 0.660 | 0.695 | 10,352 | 5,330 | 8,541 | 3,775 |
| Lung Lesion | 0.1402 | 0.614 | 0.794 | 0.104 | 0.178 | 653 | 5,640 | 21,798 | 411 |
| Edema | 0.3122 | 0.816 | 0.733 | 0.538 | 0.649 | 6,058 | 5,196 | 14,253 | 1,368 |
| Consolidation | 0.1791 | 0.758 | 0.619 | 0.145 | 0.243 | 1,485 | 8,768 | 14,270 | 475 |
| Pneumonia | 0.1144 | 0.681 | 0.724 | 0.065 | 0.118 | 487 | 7,041 | 18,486 | 228 |
| Atelectasis | 0.2766 | 0.723 | 0.591 | 0.286 | 0.410 | 3,225 | 8,044 | 11,609 | 1,237 |
| Pneumothorax | 0.3390 | 0.747 | 0.826 | 0.306 | 0.434 | 1,966 | 4,460 | 21,171 | 666 |
| Pleural Effusion | 0.4066 | 0.832 | 0.779 | 0.735 | 0.781 | 9,605 | 3,457 | 12,209 | 1,938 |
| Pleural Other | 0.0329 | 0.698 | 0.760 | 0.038 | 0.072 | 266 | 6,719 | 21,300 | 115 |
| Fracture | 0.0554 | 0.758 | 0.708 | 0.093 | 0.166 | 828 | 8,043 | 19,455 | 264 |
| Support Devices | 0.4309 | 0.809 | 0.768 | 0.816 | 0.813 | 12,947 | 2,915 | 9,625 | 3,056 |

**9b.** The CSV-based columns (`tp`, `fp`, … without `_exact`) differ by 0–16 entries per class. That is the B1 bug.

## 10. Confusion matrices — VERIFIED

For all 14 classes:
- TP + FN = positive support;
- TN + FP = negative support;
- TP + FP + TN + FN = evaluated support.

This holds for both the CSV-based and the bit-exact counts.

## 11. Threshold sanity — WARNING

- **No unusable values:** no threshold is ≤ 0, ≥ 1 or infinite, and none is above 0.90.
- **Very low thresholds:** Pleural Other **0.0329** and Fracture **0.0554**.
- **Small support:** Pleural Other has the smallest positive support, 381, the only class under 500.

**Why low thresholds are legitimate here.** Each sigmoid estimates P(label | image), which is pulled down by low prevalence. The √-tempered pos_weight raises rare-class outputs only partially; the median score of true Pleural Other positives is 0.088.

Youden's J ignores prevalence and sets the cut-off where TPR − FPR peaks, which for a rare class lies at low absolute probabilities. The value is correct for its definition. The consequence is the very low precision shown in §9: 0.038 for Pleural Other, meaning 25 false alarms per true positive.

## 13. Macro vs micro — explanation from the observed numbers

- **Micro AUROC (0.892) > macro AUROC (0.804)** and **micro AUPRC (0.671) > macro AUPRC (0.424)**.
- **The four most common labels** (Support Devices, Lung Opacity, Pleural Effusion, Edema) contribute **71.4%** of all valid positives.
- **With those four removed**, micro AUROC drops to **0.839** and micro AUPRC to **0.370**.
- **Pooling also mixes classes.** Micro metrics pool all (image, label) pairs. They reward ranking a common label's positives above a rare label's negatives, which is a between-class base-rate effect rather than per-finding discrimination.

**Conclusion:** the common classes dominate the micro metrics. Macro values, and the per-class table, are the honest summary of per-finding performance.

## 14. Training history — VERIFIED

| Epoch | Train loss | Val loss | Macro AUROC | Macro AUPRC | LR (during epoch) |
|---:|---:|---:|---:|---:|---|
| 1 | 0.5082 | 0.4942 | 0.7848 | 0.3940 | 1e-4 |
| 2 | 0.4790 | 0.4845 | 0.7983 | 0.4102 | 1e-4 |
| 3 | 0.4652 | 0.4972 | 0.7939 | 0.4097 | 1e-4 |
| 4 | 0.4531 | 0.5146 | 0.7952 | 0.4125 | 1e-4 |
| **5** | **0.4209** | **0.4855** | **0.8038** | **0.4237** | 1e-5 |
| 6 | 0.4093 | 0.4901 | 0.8035 | 0.4248 | 1e-5 |
| 7 | 0.4018 | 0.4943 | 0.8019 | 0.4233 | 1e-5 |
| 8 | 0.3932 | 0.4955 | 0.8015 | 0.4226 | 1e-6 |

- **Best epoch:** 5.
- **Convergence:** healthy. Training loss decreases monotonically.
- **Overfitting:** mild, and present.
  - Validation loss is lowest at epoch 2 (0.4845), and the train–validation gap grows from −0.014 (epoch 1) to +0.102 (epoch 8).
  - Macro AUROC plateaus at about 0.80 after the LR drop.
- **Underfitting:** none apparent.
- **Stability:**
  - The largest epoch-to-epoch change in macro AUROC is +0.0134 (epoch 1→2).
  - Epoch 4 shows a transient dip in validation loss and micro AUROC (0.863), recovered after the LR drop.
  - Maximum gradient norm per epoch is ≤ 4.78, and 12 AMP-skipped steps in total.
  - Nothing is unstable.
- **Suspiciously perfect results:** none.
- **Log message:** the "early stopping after epoch 8" line coincided with the 8-epoch maximum. `training_summary.json` records `stopped_early: false`, which is consistent but worded ambiguously in the log.

## 15. Leakage and duplicate audit — VERIFIED

- **Overlap.** Train ∩ validation is 0 patients, 0 split groups and 0 image paths.
- **Phase 1 exact-duplicate groups.** 22 CheXpert groups (SHA-256) were checked; 0 cross a split boundary.
- **Cross-patient identical-image links.** 4 were checked; 0 cross a split boundary.
- **Prediction paths.** All map to the `val` split in the manifest, and a sample of 29 resolves on disk.
- **Hashes** came from the Phase 1 artifacts; none were regenerated.

## 16. Class-wise failure analysis

| Category | Classes (from the tables above) |
|---|---|
| **Strongest** | AUROC: Pleural Effusion 0.885, No Finding 0.884, Cardiomegaly 0.869. Normalised AUPRC lift: Pleural Effusion 0.721, Support Devices 0.687, Edema 0.556 |
| **Weakest** | AUROC: Enlarged Cardiomediastinum 0.698, Atelectasis 0.716, Lung Opacity 0.737. Normalised lift: Pleural Other 0.070, Enlarged Cardiomediastinum 0.097, Pneumonia 0.106 |
| **High AUROC, poor AUPRC** | Fracture (0.800 / 0.188), Pleural Other (0.797 / 0.083), Lung Lesion (0.777 / 0.193), Pneumonia (0.770 / 0.130) |
| **Many false positives** | Consolidation 8,768; Atelectasis 8,044; Fracture 8,043; Pneumonia 7,041; Enlarged Cardiomediastinum 6,948; Pleural Other 6,719 |
| **Many false negatives** (absolute) | Lung Opacity 3,775; Support Devices 3,056; Pleural Effusion 1,938; Edema 1,368; Atelectasis 1,237 |
| **Youden favours sensitivity** (sens − spec > 0.10) | Consolidation (+0.14), Atelectasis (+0.13), Lung Opacity (+0.12) |
| **Youden favours specificity** | Enlarged Cardiomediastinum (−0.17), Lung Lesion (−0.18), Pneumothorax (−0.08) |
| **Precision particularly poor** (< 0.15) | Pleural Other 0.038, Pneumonia 0.065, Fracture 0.093, Enlarged Cardiomediastinum 0.100, Lung Lesion 0.104, Consolidation 0.145 |
| **Recall particularly poor** (< 0.70) | Enlarged Cardiomediastinum 0.561, Lung Lesion 0.614, Pneumonia 0.681, Pleural Other 0.698 |

**Role of prevalence.**
- **False positives.** At a fixed sensitivity/specificity pair, the false positives scale with the number of negatives. The six low-precision classes have 12–74 negatives per positive, so even specificity of about 0.6–0.8 yields more false than true positives.
- **High-AUROC, low-precision classes.** Every one of them has prevalence ≤ 3.8%.
- **The weakest AUROC classes** are also those with the most uncertain labels (Phase 1). Uncertain labels are excluded from evaluation, but blank labels scored as negative may also contain unreported findings.

## 17. Score distributions — WARNING

Figure: `verification/figures/positive_vs_negative_distributions.png`. Table: `verification/distribution_overlap.csv`.

- **The model separates positives from negatives in every class:** the median score of positives exceeds that of negatives for all 14.
- **But overlap is substantial (descriptive overlap coefficient ≥ 0.5) for 8 classes:**
  - Enlarged Cardiomediastinum 0.71
  - Atelectasis 0.69
  - Lung Opacity 0.65
  - Consolidation 0.62
  - Pneumonia 0.60
  - Lung Lesion 0.59
  - Pleural Other 0.55
  - Fracture 0.54

For these classes no single threshold separates positives from negatives cleanly.

## 18. Random-baseline comparison

| | Random baseline | Model |
|---|---|---|
| AUROC | 0.5 for every class | 0.698–0.885 |
| AUPRC | equals prevalence | above prevalence for every class |

- **Good separation:** Pleural Effusion, No Finding, Cardiomegaly, Pneumothorax, Edema and Support Devices (AUROC ≥ 0.85).
- **Modest:** Enlarged Cardiomediastinum (0.698), Atelectasis (0.716) and Lung Opacity (0.737). These are meaningful but not strong.
- **Not strong in precision terms:** Pleural Other, Enlarged Cardiomediastinum, Pneumonia and Consolidation. Their AUPRC gain over chance is real but small in absolute terms (§6).

## 19–21. Examples, structured findings and query builder — VERIFIED

**Examples.** All are validation images, selected by deterministic rules rather than by eye. Each lists all 14 outputs (probability, threshold, prediction, ground truth); see `verification/examples.json`.

| Example | Rule | Image | True pos | Pred pos | FP | FN |
|---|---|---|---:|---:|---:|---:|
| 1. Correct multi-label | most correct valid labels among images with ≥ 2 true positives | patient12283/study4 | 6 | 7 | 0 | 0 |
| 2. False-positive-heavy | maximum FP count | patient33073/study1 | 2 | 10 | 10 | 2 |
| 3. False-negative-heavy | maximum FN count | patient13213/study2 | 6 | 6 | 3 | 5 |
| 4. Rare class | true Pleural Other positive with the median score | patient26481/study1 | 2 | 4 | 2 | 0 |
| 5. No positive prediction | none exists | **not available** | | | | |
| 6. No Finding contradiction | first such image in file order | patient00005/study1 | 2 | 5 | 3 | 0 |

- In example 1, the seventh prediction is Pneumonia, whose ground truth is *uncertain* (excluded from scoring). It still enters the query.
- No validation image has zero predicted positives. The no-positive path was therefore exercised on a clearly labelled **synthetic** vector, with every probability set to 0.5 × its threshold.

**Checks passed in all 5 available examples and the synthetic case:**
- Every status equals `p >= threshold`.
- The positive and negative sets partition the 14 labels exactly.
- Probabilities are passed through unaltered.
- No highest-probability label is forced positive (synthetic case: 0 positives).
- The query names only observations at or above threshold.
- The query contains no location, laterality, severity or causal wording.
- The no-positive query reads "Chest radiograph without any classifier finding above its operating threshold." It does not claim a normal study.
- The No Finding contradiction warning fires exactly when expected (example 6: No Finding together with Enlarged Cardiomediastinum, Lung Lesion and Pleural Other).

**Caveat.** When No Finding and pathologies are both positive, the query **text** lists only the pathologies. The contradiction is carried only in the structured `warnings` field, so Phase 3 consumers must read that field.

## 22. Artifact and documentation consistency — WARNING

These artifact cross-checks all hold:
- the best epoch agrees across history, checkpoint metadata, training summary, summary metrics and threshold provenance;
- `summary_metrics.n_images` equals the 28,671 prediction rows;
- the threshold-file label order is the expected order;
- the experiment `config.yaml` equals `configs/classifier/densenet121.yaml`;
- the thresholds in `per_class_metrics.csv` equal the threshold file to within 1e-15 (CSV precision).

**`BASELINE_ANALYSIS.md` was parsed and every number compared with its artifact.**
- **Mismatch:** only the **LR cells for epochs 6 and 7** are wrong (B2).
- **Per-class table:** all values match the bit-exact artifacts.
- **Summary values:** macro and micro AUROC/AUPRC, 5.09 and 2.40 positives per image, 6,529 warnings, 2.28 GB and 7,375 s all match their source files.
- **Exception:** the PPV values and the 22.8% warning rate come from the CSV-based `operating_points` output, so they inherit the tiny B1 offsets.

## 23. Reproducibility — WARNING

| Comparison | Max abs. probability difference | Status flips |
|---|---:|---:|
| Fresh fp16 run 1 vs fresh fp16 run 2 (same checkpoint) | **0.0** | 0 |
| Fresh run vs saved CSV | 1.1e-16 | 72 (B1: ties at the threshold) |
| fp32 single-image path vs fp16 batch path (512 images) | 0.0053 | 5 of 7,168 (W1) |

- **Determinism:** batch evaluation is fully deterministic. The re-run macro AUROC is 0.8037923325159262, identical to the saved value.
- **Remaining warnings:** they come from serialisation precision (B1) and the fp32/fp16 mismatch (W1), not from randomness.

---

## Final verdict: **VERIFIED WITH WARNINGS — usable, but issues should be addressed**

1. **Reported AUROCs correct?** Yes. All 14, plus macro and micro, recompute exactly (difference 0).
2. **Reported AUPRCs correct?** Yes, as average precision (difference ≤ 5.6e-17).
3. **Youden J thresholds correct?** Yes. They are bit-exact on the model's in-memory outputs, with stored sensitivity and specificity exact. The saved CSV cannot reproduce them exactly (B1).
4. **Validation split leak-free?** Yes. 0 shared patients, split groups or paths; 0 duplicate groups or identity links crossing.
5. **Locked test set untouched?** Yes, 9/9 PASS. It was not evaluated by this audit either.
6. **Probabilities genuine independent sigmoids?** Yes. Sums range from 1.22 to 5.59, and none is near 1.
7. **Any collapsed classes?** No.
8. **Three strongest classes** (AUROC): Pleural Effusion (0.885), No Finding (0.884), Cardiomegaly (0.869).
9. **Three weakest classes** (AUROC): Enlarged Cardiomediastinum (0.698), Atelectasis (0.716), Lung Opacity (0.737). By normalised AUPRC: Pleural Other, Enlarged Cardiomediastinum, Pneumonia.
10. **Concerning precision:** Pleural Other 0.038, Pneumonia 0.065, Fracture 0.093, Enlarged Cardiomediastinum 0.100, Lung Lesion 0.104, Consolidation 0.145.
11. **Concerning recall:** Enlarged Cardiomediastinum 0.561, Lung Lesion 0.614, Pneumonia 0.681, Pleural Other 0.698.
12. **Overfitting?** Mild. Validation loss is lowest at epoch 2 while training loss keeps falling; the AUROC plateau from epoch 5 limits the damage.
13. **Underfitting?** No evidence.
14. **Suspiciously good results?** None. All values are moderate and consistent with the label noise and prevalence.
15. **Stored results internally consistent?** Yes, apart from B1 (CSV precision) and B2 (two documentation LR cells).
16. **Structured-findings conversion correct?** Yes, in all checked cases, including the contradiction warning and the synthetic no-positive case.
17. **Query builder faithful to classifier output?** Yes. It uses CheXpert names only, invents nothing, and handles no-positive cases. Caveat: the No Finding contradiction lives in `warnings`, not in the text.
18. **Technically ready to feed MiniLM + Qdrant?** Technically yes: the outputs are correct, reproducible and leak-free. Resolve or explicitly accept these first:
    - **B1:** regenerate the prediction file at full precision, with no retraining.
    - **W1:** align the inference precision with the threshold-fitting precision.
    - **The commit-provenance gap.**
    - **The scientific issue already raised in `BASELINE_ANALYSIS.md`:** the Youden operating points over-call findings. About 5.1 predicted positives per image against 2.4 true; precision below 0.15 for 6 classes. Retrieval queries would carry those false findings.
