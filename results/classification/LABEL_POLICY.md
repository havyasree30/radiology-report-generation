# CheXpert Label Policy (Phase 2 classifier)

**Scope.** The DenseNet121 classifier (System A).
**Implementation.**
- `src/data/label_policies.py`, which maps labels in memory. Source CSVs are never modified.
- `src/classification/dataset.py`, which builds the per-sample validity mask.

**Configuration:** `configs/classifier/densenet121.yaml` → `label_policy`.

## 1. Source semantics (verified)

The CheXpert paper (Irvin et al., AAAI 2019, arXiv:1901.07031) describes the labeler output as follows:
- **Positive (1):** at least one mention of the observation is classified positive.
- **Uncertain (−1):** there are no positive mentions and at least one uncertain mention.
- **Negative (0):** there is at least one negated mention (and no positive or uncertain one).
- **Blank:** the observation is **not mentioned** in the report.

The paper defines three uncertainty-handling approaches:
- **U-Ignore:** the uncertain labels are ignored (masked from the loss).
- **U-Zeros:** uncertain → 0.
- **U-Ones:** uncertain → 1.

It reports that the best approach differed by observation. For example, U-Ones worked best for Atelectasis and Edema.

**The paper does not explicitly specify how blank labels were handled during training.** The blank policy below is therefore *our* documented modelling assumption, justified by the Phase 1 data rather than attributed to the original authors.

The official validation set was labelled by three board-certified radiologists (majority vote), and it has no uncertain and no blank labels. This was verified in Phase 1.

## 2. Policy used for the Phase 2 baseline

| Raw state | Meaning | Training target | In loss? (validity mask) | Configurable |
|---|---|---|---|---|
| 1 | positive mention | 1 | yes | n/a |
| 0 | explicitly negated | 0 | yes | n/a |
| −1 | uncertain mention | none | **no: U-Ignore (masked)** | `u_ignore` / `u_zero` / `u_one` |
| blank | not mentioned | 0 | yes | `zero` / `ignore` |

### Uncertain → masked (U-Ignore)

Masked labels contribute **exactly zero loss and zero gradient**, which is tested. The loss is normalised by the number of valid entries only.

Reasons for choosing this for the baseline:
- **It injects no assumed label.** Phase 1 showed that the uncertainty policy changes the effective positive class 2–4× for Pneumonia (4.11×), Consolidation (2.88×), Enlarged Cardiomediastinum (2.15×) and Atelectasis (2.01×). U-Zero or U-One would bake a strong untested assumption into the baseline.
- **It matches the Phase 1 plan.** U-Ignore was the proposed reference for the loss-comparison stage, with U-Zero and U-One to be compared later under otherwise identical settings.
- **It is not declared the winner.** `u_zero` and `u_one` are implemented and tested, but they are not run in this phase.

**Evaluation.** Uncertain entries are also excluded from validation metrics and Youden thresholds, because they have no reference answer. Each metric's valid support is reported per class.

### Blank → 0 (negative)

Evidence from Phase 1 (`results/eda/tables/chexpert_missing_analysis.csv`, `training_pos_weights.csv`):

- **Blank is common for most labels.** 40–97% of labels are blank; 11 of 14 observations are at least 50% blank.
- **Masking blanks would invert the class balance.** Only explicit negatives would remain, and pos_weight would fall below 1 for 10 of 14 labels. Explicit negation is a biased sample: a finding is typically negated only when it was clinically in question.
- **No Finding would lose every negative under masking.** It is never explicitly 0, so its training class would contain only positives.
- **Clinical reading.** A report that does not mention a finding most often reflects that the finding was not seen.

This is an **assumption**, and it introduces some false negatives: findings that were present but not reported. The **`ignore`** option is implemented for a later sensitivity analysis, which is not run in this phase.

## 3. Mask semantics (implementation contract)

- **Shapes.** `targets[i, c]` ∈ {0, 1} and `mask[i, c]` ∈ {True, False}, for c in the fixed order of `src/classification/labels.py`.
- **Loss.** Element-wise terms (`reduction="none"`) are combined as follows, so masked entries have zero influence:
  - `where(mask, elem, 0)`, which stops even non-finite values at masked positions from leaking;
  - then summed and divided by `max(mask.sum(), 1)`.
- **Metrics and thresholds** use only entries with `mask == True`. A class with no valid positives or no valid negatives gets `NaN` or `None` plus a reason, never 0.

## 4. Label order (fixed)

1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices

This is the CheXpert CSV column order. It is asserted against the CSV header, the Phase 1 artifacts and every checkpoint.
