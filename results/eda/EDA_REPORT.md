# Phase 1 EDA Report: CheXpert and IU X-Ray

**Project:** A Framework for AI-Based Preliminary Radiology Report Generation
**Phase:** 1 (Dataset Exploration and EDA). No model has been trained.
**Analysis date:** 2026-09-30 · **Seed:** 42 · **Environment:** Python 3.11.9, Windows 11. Full record in `summaries/reproducibility.json`.

Every number in this report was produced by the pipeline below from the local datasets. The source tables and figures are cited inline.
Where a number is a *descriptive heuristic* (a threshold or band), the threshold lives in `configs/eda.yaml`.

```
scripts.eda_01_image_audit   -> image integrity, hashes, duplicate groups
scripts.eda_02_chexpert      -> label states, imbalance, multi-label structure, cohort, views, demographics
scripts.eda_03_chexpert_splits -> patient-group splits, leakage checks, pos_weights, sampling analysis
scripts.eda_04_iu_xray       -> IU structure, text quality, concept analysis, report-level splits
scripts.eda_05_summary       -> dataset_summary.csv, eda_summary.json, reproducibility.json
```

---

## 1. Dataset discovery

The datasets are not in the locations named in the Phase 1 brief (`data/chexpert/`, `data/iu xray/`). The actual layout is:

| Item | Actual location | Notes |
|---|---|---|
| CheXpert images | `data/CheXpert-v1.0-small/{train,valid}/patientXXXXX/studyN/viewK_{frontal,lateral}.jpg` | CheXpert **v1.0-small** (downsampled JPEGs) |
| CheXpert train labels | `data/train.csv` | Sits *outside* the release folder. Its `Path` column is relative to `data/` |
| CheXpert valid labels | `data/CheXpert-v1.0-small/valid.csv` | Official validation set |
| IU X-Ray | `data/IU X-ray/{indiana_reports.csv, indiana_projections.csv, images/images_normalized/*.png}` | **Kaggle CSV layout** (not Open-I XML) |
| IU archive | `data/IU X-ray.zip` (14.1 GB) | Source archive. Its 7,472 entries match the extracted files |

All locations are resolved through `configs/paths.yaml` (git-ignored). `.gitignore` now ignores everything under `data/` except `data/README.md` and `data/splits/`. The datasets were only read; no file under `data/CheXpert-v1.0-small`, `data/IU X-ray` or `data/train.csv` was written, renamed or moved.

**The official CheXpert test set is not present.** Only the official train and valid partitions exist locally.

## 2. Dataset structure

**CheXpert.** The CSV columns are `Path, Sex, Age, Frontal/Lateral, AP/PA` followed by 14 observation columns. The targets were detected from the header and value vocabulary, not hard-coded, and validated against the expected 14 names:
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices.

- **Identifiers.** The patient ID (`patientNNNNN`) and study ID come from the path. Study numbers restart per patient, so the study ID is `patient/study`.
- **Label values.** Every value is in {1, 0, −1, blank}. No out-of-vocabulary value was found (`tables/chexpert_audit_log.csv` is empty).
- **Labels are study-level.** In all 33,994 multi-image training studies, every image carries identical labels.

**IU X-Ray.**
- `indiana_reports.csv` has one row per report (`uid, MeSH, Problems, image, indication, comparison, findings, impression`).
- `indiana_projections.csv` has one row per image (`uid, filename, projection`).
- The `uid` is the report/study unit.
- **There is no patient identifier**, so patient-level grouping is impossible for IU X-Ray. This is a documented limitation.

## 3. Data integrity (`tables/image_audit_summary.csv`, `tables/image_quality_flags.csv`)

Every file was fully decoded (not just header-read). The audit also computed a SHA-256, 64-bit and 256-bit difference hashes, and intensity statistics.

| | CheXpert | IU X-Ray |
|---|---:|---:|
| Image files on disk | 223,649 | 7,470 |
| CSV-referenced records / unique paths | 223,648 / 223,648 | 7,466 / 7,466 |
| Missing / zero-byte / corrupt referenced images | 0 / 0 / 0 | 0 / 0 / 0 |
| Files on disk not referenced by any CSV | 1 (macOS `._` metadata file, unreadable, not an image) | 4 |
| Image mode | 100% `L` (8-bit grayscale) | 100% `L` |
| Width range (median) | 320–930 px (390) | 1,529–2,891 px (2,048) |
| Height range (median) | 320–642 px (320) | 1,760–3,001 px (2,048) |
| Near-blank images | **6** | 2 (both unreferenced files) |
| Aspect ratio outside [0.5, 2.0] | 15 | 0 |
| Exact-duplicate groups (SHA-256) | 22 (44 images) | 3 (6 images) |

**Near-blank images.** The first rule tried (pixel SD < 5) caught only 1 of the defective images. Visual inspection showed that images with SD up to about 24 were also blank. The final rule is SD < 5 **or** ≥ 90% near-saturated pixels. All six flagged CheXpert images were visually confirmed to contain no anatomy (`tables/manual_image_review.csv`). Five are in the train split and one is in val.

**Unusual aspect ratios.** An IQR-based aspect-ratio rule was tried first and rejected: CheXpert-small sizes are discrete, so it flagged 20,889 images. Absolute bounds replaced it. Inspection shows the flagged wide images are valid, tightly collimated radiographs. They are not defects, but naive square resizing would distort them.

**Duplicate identities (a leakage risk).**
- Four exact-duplicate pairs span **different CheXpert patient IDs**: 14431↔36994, 44529↔50936, and 05189↔35139 (two pairs).
- These patient IDs may belong to the same person. They were **merged into one split group** (3 merges), so identical images can never straddle partitions (`tables/chexpert_cross_patient_duplicate_links.csv`).
- The remaining exact duplicates are within a single study or patient.

**Perceptual duplicates.**
- The 64-bit dHash is **not discriminative** for chest radiographs: 8,320 "duplicate" groups in CheXpert, the largest with 63 images.
- The 256-bit dHash yields 28 CheXpert groups. Every group not already an exact duplicate lies **within one study**, so no new cross-patient candidates appeared.
- Perceptual matches are reported as candidates only.

**IU duplicates.**
- uid 1015 lists its frontal/lateral pair twice under two series numbers, and the copies are byte-identical.
- uid 3245 has two byte-identical images, both labelled *Frontal*.

## 4. CheXpert cohort characteristics

| | Images | Patients | Studies |
|---|---:|---:|---:|
| Official train | 223,414 | 64,540 | 187,641 |
| Official valid | 234 | 200 | 200 |
| **Total** | **223,648** | **64,740** | **187,841** |

No patient appears in both official partitions.

## 5. Label distribution (`tables/chexpert_label_distribution.csv`, fig. `chexpert_01`)

All four states are kept separate. The table covers the official training set, all 223,414 images:

| Observation | Positive | Negative | Uncertain | Missing |
|---|---:|---:|---:|---:|
| Support Devices | 116,001 (51.9%) | 6,137 | 1,079 | 100,197 (44.8%) |
| Lung Opacity | 105,581 (47.3%) | 6,599 | 5,598 | 105,636 (47.3%) |
| Pleural Effusion | 86,187 (38.6%) | 35,396 | 11,628 | 90,203 (40.4%) |
| Edema | 52,246 (23.4%) | 20,726 | 12,984 | 137,458 (61.5%) |
| Atelectasis | 33,376 (14.9%) | 1,328 | 33,739 | 154,971 (69.4%) |
| Cardiomegaly | 27,000 (12.1%) | 11,116 | 8,087 | 177,211 (79.3%) |
| No Finding | 22,381 (10.0%) | **0** | 0 | 201,033 (90.0%) |
| Pneumothorax | 19,448 (8.7%) | 56,341 | 3,145 | 144,480 (64.7%) |
| Consolidation | 14,783 (6.6%) | 28,097 | 27,742 | 152,792 (68.4%) |
| Enlarged Cardiomediastinum | 10,798 (4.8%) | 21,638 | 12,403 | 178,575 (79.9%) |
| Lung Lesion | 9,186 (4.1%) | 1,270 | 1,488 | 211,470 (94.7%) |
| Fracture | 9,040 (4.0%) | 2,512 | 642 | 211,220 (94.5%) |
| Pneumonia | 6,039 (2.7%) | 2,799 | 18,770 | 195,806 (87.6%) |
| Pleural Other | 3,523 (1.6%) | 316 | 2,653 | 216,922 (97.1%) |

**No Finding is never explicitly negative:** it is either 1 or blank.

The **official valid set contains no uncertain and no blank labels**, because it was radiologist-labelled. It also has a very different prevalence profile:
- Enlarged Cardiomediastinum: 46.6% vs 4.8% in train.
- Fracture: 0 positives.
- Lung Lesion and Pleural Other: 1 positive each.

## 6. Class imbalance (`tables/chexpert_imbalance_analysis.csv`, figs. `chexpert_02`, `03`, `17`)

Negatives were **not** computed as N − positives. Two ratios are reported:

- **Explicit** negatives : positives. This is misleading on its own: for 11 of 14 observations there are *fewer* explicit negatives than positives, because reports mention a finding mostly when it is present.
- **(Explicit negatives + unmentioned) : positives.** This is the ratio the loss faces under the conventional *NaN → 0, uncertain excluded* treatment. It is the basis of the severity band.

Ranked from least to most imbalanced by the second ratio:

| Rank | Observation | Explicit neg : pos | (Neg + unmentioned) : pos | Severity |
|---:|---|---:|---:|---|
| 1 | Support Devices | 0.05 | 0.92 | mild |
| 2 | Lung Opacity | 0.06 | 1.06 | mild |
| 3 | Pleural Effusion | 0.41 | 1.46 | mild |
| 4 | Edema | 0.40 | 3.03 | moderate |
| 5 | Atelectasis | 0.04 | 4.68 | moderate |
| 6 | Cardiomegaly | 0.41 | 6.98 | moderate |
| 7 | No Finding | 0 (no explicit negatives) | 8.98 | moderate |
| 8 | Pneumothorax | 2.90 | 10.33 | severe |
| 9 | Consolidation | 1.90 | 12.24 | severe |
| 10 | Enlarged Cardiomediastinum | 2.00 | 18.54 | severe |
| 11 | Lung Lesion | 0.14 | 23.16 | severe |
| 12 | Fracture | 0.28 | 23.64 | severe |
| 13 | Pneumonia | 0.46 | 32.89 | severe |
| 14 | Pleural Other | 0.09 | **61.66** | **extreme** |

The ratio spans **0.92 : 1 to 61.7 : 1**, a 67-fold range across labels.
- **Common findings (≥ 15% prevalence):** Support Devices, Lung Opacity, Pleural Effusion, Edema.
- **Rare (1–5%):** Enlarged Cardiomediastinum, Lung Lesion, Fracture, Pneumonia, Pleural Other.
- **Extremely rare (< 1%):** none.
- **Negative-dominated by explicit labels (≥ 10 : 1):** none. The imbalance comes almost entirely from *unmentioned* labels, not from explicit negatives.

## 7. Multi-label characteristics (`tables/chexpert_label_cardinality.csv`, fig. `chexpert_06`)

These figures use frontal training images (n = 191,027) and count positive = label 1.

- **Label cardinality** (14 observations) is **2.41**, and **label density** is **0.172**. Excluding No Finding, cardinality is 2.32; over the 12 pathologies (excluding No Finding and Support Devices) it is 1.76.
- **Positive observations per image:** 0 → 7,766; 1 → 40,388; 2 → 56,728; 3 → 49,365; 4 → 27,666; 5 → 7,865; 6 → 1,128; 7 → 111; 8 → 10. The maximum observed is 8.

Most images carry two or more simultaneous positives. This confirms the multi-label formulation: 14 independent sigmoids, with no softmax and no sum-to-one constraint.

## 8. Label co-occurrence (`tables/chexpert_cooccurrence.csv`, figs. `chexpert_07`–`09b`)

These are measured on frontal training images, positive vs not positive. Everything here is statistical association, not causation.

- **Strongest positive associations (phi):**
  - Lung Opacity–Pleural Effusion: 0.210 (47,764 images with both; Jaccard 0.387).
  - Cardiomegaly–Edema: 0.172 (lift 1.78).
  - Pleural Effusion–Support Devices: 0.146.
  - Edema–Pleural Effusion: 0.128.
- **Conditional probabilities:**
  - P(Lung Opacity | Pneumonia) = 0.69.
  - P(Support Devices | Edema) = 0.66.
  - P(Support Devices | Pleural Effusion) = 0.65.
  - P(Edema | Cardiomegaly) = 0.46.
- **Rare labels travel with common ones.** Given Pleural Other, Lung Lesion or Pneumonia, Lung Opacity is positive in 0.55, 0.59 and 0.69 of images respectively. This is the mechanism that makes image-level oversampling leak into common labels (§ 23).
- **Negative associations** are dominated by No Finding, which is mutually exclusive with every pathology (phi with Lung Opacity −0.31).

## 9. Uncertainty (`tables/chexpert_uncertainty_analysis.csv`, fig. `chexpert_04`)

| Observation | Uncertain (% of images) | Uncertain / (pos + unc) | Positives ×, if U-One |
|---|---:|---:|---:|
| Atelectasis | 15.1% | 0.50 | 2.01 |
| Consolidation | 12.4% | 0.65 | 2.88 |
| Pneumonia | 8.4% | **0.76** | **4.11** |
| Edema | 5.8% | 0.20 | 1.25 |
| Enlarged Cardiomediastinum | 5.6% | 0.53 | 2.15 |
| Pleural Effusion | 5.2% | 0.12 | 1.13 |
| Cardiomegaly | 3.6% | 0.23 | 1.30 |
| Lung Opacity | 2.5% | 0.05 | 1.05 |
| Pneumothorax | 1.4% | 0.14 | 1.16 |
| Pleural Other | 1.2% | 0.43 | 1.75 |
| Lung Lesion | 0.7% | 0.14 | 1.16 |
| Support Devices | 0.5% | 0.01 | 1.01 |
| Fracture | 0.3% | 0.07 | 1.07 |
| No Finding | 0.0% | 0.00 | 1.00 |

- **Uncertainty is concentrated in Pneumonia, Consolidation, Atelectasis and Enlarged Cardiomediastinum.** For these, the uncertainty policy changes the effective positive class by 2–4×.
- For Pneumonia, uncertain mentions outnumber certain positives 3.1 : 1. Pneumonia is also a clinical diagnosis that is only partly determinable from a radiograph.
- The official valid set has **no** uncertain labels. Any uncertainty policy is therefore evaluated against certain labels.

## 10. Missing / unmentioned labels (`tables/chexpert_missing_analysis.csv`, fig. `chexpert_05`)

- **Missing-dominated labels:** 11 of 14 observations have ≥ 50% blank labels. The exceptions are Support Devices (44.8%), Lung Opacity (47.3%) and Pleural Effusion (40.4%). Pleural Other (97.1%), Lung Lesion (94.7%) and Fracture (94.5%) are almost never mentioned.
- **Blank dwarfs uncertain.** The ratio ranges from 4.6 : 1 (Atelectasis) to 329 : 1 (Fracture), so the two states are quantitatively different phenomena, not interchangeable noise.
- **Masking NaN would invert the class balance.** If NaN entries were excluded from the loss, only explicit negatives would remain. The training-only pos_weight would then fall **below 1** for 10 of 14 labels: for example 0.03 for Atelectasis and 0.05 for Lung Opacity, frontal U-Masked (`tables/training_pos_weights.csv`).
- **No Finding would have zero negatives** under NaN masking, which makes it degenerate (the fallback caught this). Explicit negatives are a biased sample: a finding is typically negated only when it was clinically in question.

**Proposal: NaN → 0 during training.** An unmentioned finding in a radiology report is, in the large majority of cases, absent. The alternative (masking) destroys most negative supervision and No Finding entirely. This is a *modelling assumption* and will be documented as such. The raw CSV is never modified; the mapping happens in `src/data/label_policies.py` at load time. A NaN-masked run can be kept as a sensitivity check but is not proposed as a primary candidate.

## 11. Patient and study structure (`tables/chexpert_patient_study_summary.csv`, figs. `chexpert_10`, `11`)

| Official train + valid | Mean | Median | Q1–Q3 | Max |
|---|---:|---:|---:|---:|
| Images per patient | 3.45 | 2 | 1–4 | 92 |
| Studies per patient | 2.90 | 1 | 1–3 | 91 |
| Images per study | 1.19 | 1 | 1–1 | 3 |

- **Heavy patients:** 62 patients exceed the 99.9th percentile of studies per patient (46 studies) (`tables/chexpert_patients_many_studies.csv`).
- **Why patient-level splitting is mandatory:** these heavy-tailed patients would dominate any image-level split and leak between partitions.

## 12. View distributions (`tables/chexpert_view_summary.csv`, figs. `chexpert_12`, `13`)

| | Count | % of images |
|---|---:|---:|
| Frontal | 191,229 | 85.5% |
| Lateral | 32,419 | 14.5% |
| AP (all frontal) | 161,759 | 72.3% |
| PA (all frontal) | 29,453 | 13.2% |
| AP/PA blank (exactly the lateral images) | 32,419 | 14.5% |
| LL / RL on a *frontal* image | 16 / 1 | < 0.01% |

- **CSV and filename agree.** The CSV view and the filename view agree for every image.
- **Studies without a frontal view:** 16 training studies have no frontal image.
- **AP dominance:** AP makes up 84.6% of frontal images. AP films are typical of bedside/ICU imaging and correlate with Support Devices, which makes AP/PA a plausible confounder. It is recorded here, not acted upon.

## 13. Demographics (`tables/chexpert_demographics.csv`, figs. `chexpert_14`, `15`)

**Age (image level, n = 223,648, 0 missing):**
- mean 60.4, SD 17.8, median 62
- Q1 49, Q3 74
- min 0, max 90

**Implausible ages:** three images have **age 0**. They belong to patients 64538, 64539 and 64540, the last three training IDs. They are flagged, not corrected (`tables/chexpert_age_flags.csv`). The maximum of 90 is consistent with top-coding of older ages.

**Sex:**
- Images: Male 132,764 (59.4%), Female 90,883 (40.6%), Unknown 1.
- Patients: Male 35,917, Female 28,822, Unknown 1.
- No patient has inconsistent sex across images.

These distributions are descriptive only. No fairness conclusion can be drawn from them.

## 14. Data-quality issues (summary)

| Issue | Count | Handling (none applied automatically) |
|---|---:|---|
| Near-blank CheXpert images (visually confirmed) | 6 | recommend exclusion in Phase 2 |
| Cross-patient identical images | 4 links (3 patient merges) | merged into one split group ✔ |
| Frontal images with LL/RL AP-PA code | 17 | keep; exclude from AP/PA analyses |
| Age = 0 | 3 | keep; treat age as missing |
| Sex = Unknown | 1 | keep |
| No Finding + positive pathology | **0** | none needed |
| macOS `._` metadata file in CheXpert | 1 | not referenced; ignore |
| IU empty reports (no findings, no impression) | 25 | excluded from retrieval corpus ✔ |
| IU duplicated image sets (uid 1015, 3245) | 2 reports | same report ⇒ no split leakage |

The consolidated candidate list is in `tables/chexpert_exclusion_candidates.csv`.

## 15. Leakage analysis

- **CheXpert patients:** Train ∩ Val = Train ∩ Test = Val ∩ Test = ∅. None of them intersects the official valid set either. This is asserted in `scripts/eda_03_chexpert_splits.py` (hard failure) and in `tests/test_splits.py`, which runs against the committed manifests.
- **Split groups** (patients merged by identical images): no overlap.
- **IU X-Ray:**
  - Every report uid is in exactly one split, and every image follows its report.
  - The retrieval corpus contains only training reports (`tests/test_iu_xray.py`).
- **Residual risk 1: IU patient identity.** IU X-Ray has no patient ID. A patient with several reports could appear in more than one split, and this cannot be detected from the available metadata.
- **Residual risk 2: templated text.** 20.9% of val reports and 24.6% of test reports have *exact* normalized text matches in the training reports, because templated normal reports repeat. This is not identity leakage. It **will inflate** text-overlap metrics for retrieval and generation, and must be reported alongside them (§ 20).

## 16. Split methodology

- **Source.** Custom partitions are carved from the **official training CSV only**. The official valid set (radiologist-labelled, no uncertain/blank labels, 200 patients) is kept intact as a separate secondary evaluation set. It is never used for model selection.
- **Grouping.** The unit is the patient, merged with any patient sharing a byte-identical image.
- **Pre-declared decision rule** (`configs/eda.yaml`), written before the results were seen: use group-aware multi-label stratification if any label with ≥ 100 positives deviates by more than 10% relative prevalence in any split. That applies to the seeded random split or to the 95th percentile over 50 random seeds.
- **Outcome.** The seeded random patient split reached 11.7%. The 95th percentile over seeds was 14.9% (median 9.2%, worst 17.7%). **The rule selected stratification.**
- **Stratification algorithm.** Iterative multi-label stratification (Sechidis et al., 2011) lifted to groups: each group carries per-label positive-image counts. It is deterministic (seeded tie-breaking) and invariant to input order, which is tested.

## 17. Split quality (`tables/chexpert_split_distribution.csv`, `tables/chexpert_split_sizes.csv`, figs. `chexpert_16`, `16b`)

| Split | Images | Frontal | Patients | Studies | Image share |
|---|---:|---:|---:|---:|---:|
| Train | 156,390 | 133,649 | 45,329 | 131,329 | 70.0% |
| Val | 33,512 | 28,672 | 9,531 | 28,110 | 15.0% |
| Test | 33,512 | 28,706 | 9,680 | 28,202 | 15.0% |
| Official valid | 234 | 202 | 200 | 200 | n/a |

- **All images:** the worst relative prevalence deviation of any label in any split is **0.13%**.
- **Frontal subset:** the worst deviation is 2.9% (test), because stratification balanced all images rather than frontal images alone.
- **Rare labels:** Pleural Other prevalence among frontal images is 1.31% / 1.33% / 1.32% (train / val / test).
- **Test distribution:** the test set was not balanced or altered. It mirrors the cohort's natural prevalence.

## 18. IU X-Ray structural characteristics (`tables/iu_dataset_summary.csv`, `tables/iu_projection_summary.csv`, figs. `iu_21`, `iu_22`)

**Counts.**
- 3,851 reports (3,851 unique uids).
- 7,466 image rows (7,466 unique files), plus 7,470 files on disk. The 4 unreferenced files include 2 blank images of uid 2084.
- Every report has at least one image, and every image has a report.

**Projections.** Frontal 3,818 (51.1%), Lateral 3,648 (48.9%).

**Images per report.**

| Images per report | Reports | % |
|---:|---:|---:|
| 1 | 446 | 11.6% |
| 2 | 3,210 | 83.4% |
| 3 | 181 | 4.7% |
| 4 | 13 | 0.3% |
| 5 | 1 | < 0.1% |

**Projection patterns.**
- The dominant pattern is one frontal plus one lateral (1F+1L): 3,194 reports, 82.9%.
- **162 reports have no frontal image** (0F+1L: 151, 0F+2L: 11). This matters if retrieval is ever keyed on frontal images.

## 19. IU report characteristics (`tables/iu_report_lengths.csv`, figs. `iu_18`–`iu_20`)

**Section completeness.**

| Sections present | Reports | % |
|---|---:|---:|
| Findings + Impression | 3,331 | 86.5% |
| Impression only | 489 | 12.7% |
| Findings only | 6 | 0.2% |
| Neither | 25 | 0.6% |

**Section lengths** (whitespace words, among non-empty sections).

| Section | Median | Q1–Q3 | Max | Approx. tokens, median |
|---|---:|---:|---:|---:|
| Findings | 29 | 21–38 | 169 | 35 |
| Impression | 5 | 4–12 | 130 | 7 |
| Combined | 34 | 25–46 | 230 | 41 |

Token counts are regex tokens (words, numbers, punctuation), not any language model's tokenizer.

## 20. IU text quality (`tables/iu_report_quality.csv`, fig. `iu_24`)

**Flag counts** (flags overlap; nothing was deleted).
- 25 empty reports.
- 12 near-empty (≤ 3 words), and 18 extremely short (≤ 5 words); all of the short ones are impression-only.
- 39 extremely long, above the 99th percentile of 108.75 words.
- 514 missing Findings and 31 missing Impression.
- 6 with Findings identical to Impression.
- 0 malformed, and 0 with a high share of anonymisation tokens.

**Anonymisation.** 1,780 reports contain the anonymisation token `XXXX`, 3,544 occurrences in total. It sometimes replaces clinically meaningful words, for example "There are no XXXX of a pleural effusion".

**Duplicate text.**
- 945 reports share their normalized text with at least one other report.
- Only 3,036 distinct texts exist among the 3,826 non-empty reports.
- The most frequent single text occurs 51 times, a templated normal report.

**Implications for retrieval evaluation.**
- A retriever can score well on text similarity simply by returning a normal template.
- Report-level metrics must be broken down by normal vs abnormal reports.
- Semantic or n-gram similarity must not be read as clinical correctness.

## 21. Clinical terminology analysis (`tables/iu_concept_frequency.csv`, `tables/iu_concept_mesh_agreement.csv`, fig. `iu_23`)

**Method.** A transparent, rule-based, NegEx-style matcher (`src/analysis/text_concepts.py`, lexicon in `configs/iu_concept_lexicon.yaml`). Every CheXpert concept mention is classified as positive, negated or uncertain within its clause, and each report takes the strongest status per concept. **This is an approximate EDA instrument, not a clinical NLP labeler.**

**Test phrases.** It handles, for example:
- "No pneumothorax." → negated
- "No pericardial effusion" → not pleural effusion
- "No interval change in cardiomegaly" → positive
- "Pneumonia cannot be excluded" → uncertain

**Report-level concept prevalence** (n = 3,826 reports with text).

| Concept | Positive | Uncertain | Negated |
|---|---:|---:|---:|
| Lung Opacity | 13.7% | 0.3% | 24.7% |
| Atelectasis | 8.1% | 1.5% | 0.2% |
| Cardiomegaly | 7.7% | 0.5% | 0.2% |
| Support Devices | 4.8% | 0.1% | 0.0% |
| Pleural Effusion | 3.9% | 0.9% | **71.0%** |
| Lung Lesion | 3.8% | 0.6% | 5.0% |
| Fracture | 2.7% | 0.4% | 2.3% |
| Edema | 1.8% | 0.7% | 7.0% |
| Pneumonia | 1.2% | 1.1% | 4.5% |
| Pleural Other | 0.9% | 0.2% | 0.0% |
| Consolidation | 0.9% | 0.2% | **30.7%** |
| Pneumothorax | 0.6% | 0.0% | **67.9%** |
| Enlarged Cardiomediastinum | 0.3% | 0.0% | 0.9% |

**Negation handling is not optional.** Pleural Effusion, Pneumothorax and Consolidation are mentioned mostly to rule them out. Counting every keyword hit as positive would overstate their prevalence about 19× (Pleural Effusion), 37× (Consolidation) and 118× (Pneumothorax).

**Agreement with the dataset's own MeSH indexing** (extractor-positive vs MeSH heading present; MeSH is not ground truth):

| Concept | Precision vs MeSH | Recall vs MeSH |
|---|---:|---:|
| Lung Opacity | 0.98 | 0.97 |
| Cardiomegaly | 0.96 | 0.83 |
| Atelectasis | 0.96 | 0.94 |
| Pleural Effusion | 0.92 | 0.93 |
| Pneumothorax | 0.86 | 0.86 |
| Consolidation | 0.88 | 0.97 |
| Lung Lesion | 0.72 | 0.87 |
| Pneumonia | 0.70 | 0.83 |
| Edema | 0.91 | 0.69 |
| Support Devices | 0.94 | **0.54** |

- **Support Devices recall** is low because the lexicon misses many device phrasings.
- **Lung Lesion and Pneumonia precision** is moderate.
- **Review sample:** a random sample of 300 mentions for manual review is in `tables/iu_concept_mentions_review_sample.csv`.

**Distribution shift from CheXpert.** IU X-Ray is mostly outpatient and normal: 1,379 reports (35.8%) are indexed "normal". CheXpert is inpatient-heavy (51.9% Support Devices, 72.3% AP). The retrieval corpus will therefore be dominated by normal and near-normal language.

## 22. Implications for DenseNet121 training

1. **Output head:** 14 logits → 14 independent sigmoids, trained with per-label binary losses. No softmax, and no normalisation to sum 1. Label cardinality of 2.41 means co-existing findings are the norm.
2. **Views:** use frontal images only for the primary experiment (§ 12, Q14).
3. **Imbalance:** use loss-level handling first, not sampling (§ 23).
4. **Missing labels:** NaN → 0 (§ 10). **Uncertain labels:** compare U-Masked, U-Zero and U-One (§ 24).
5. **Evaluation must be per label.** Prevalence spans 1.3%–56% in frontal images, so report AUROC **and** AUPRC. AUPRC is prevalence-dependent and must be read against each label's prevalence. The official valid set cannot support per-label AUROC for Fracture (0 positives), Lung Lesion (1) or Pleural Other (1).
6. **Preprocessing:** aspect ratio varies (15 very wide crops, a range of 0.50–2.91). Resizing must preserve aspect ratio (pad or crop consistently). This is a preprocessing choice for Phase 2, and no strategy is pre-selected. CLAHE is not assumed.
7. **Output semantics:** thresholds and calibration are later, validation-only steps.

## 23. Recommended imbalance experiments

**The data justify all three loss-level candidates:**
- **Positive-weighted BCE (B).** Effective ratios reach 61.7 : 1. The training-only pos_weight (frontal, U-Masked, NaN→0) ranges from **0.77** (Support Devices) to **74.8** (Pleural Other).
  - Risk: a weight of 74.8 may produce over-confident positives and poor calibration. Calibration must therefore be measured, never assumed.
- **Focal loss (C, γ ∈ {1, 2}).** With density 0.17, about 83% of label slots per image are negative, and most of them are easy. Focal loss targets exactly this.
- **Asymmetric loss (D).** It was designed for multi-label settings with many negatives per sample. Its probability margin also dampens the influence of possibly mislabeled negatives, which is relevant because NaN → 0 injects some false negatives.

**Sampling** (`summaries/sampling_analysis.md`, `tables/chexpert_sampling_effect.csv`). Inverse-prevalence sampling:
- inflates the four rarest labels by 2.5–7.0×;
- cuts the Kish effective sample size to 46.6% of N;
- concentrates 7.0% of the sampling mass in 1% of images;
- shifts P(B | A) by up to 0.13.

The pre-stated "costly" rule is met. **No sampler in the primary experiments.** A tempered √-sampler (ESS 80.8%) remains an optional later ablation.

## 24. Recommended uncertainty experiments

Compare three strategies:
- **U-Masked** (−1 excluded from that label's loss);
- **U-Zero** (−1 → 0);
- **U-One** (−1 → 1).

In each case NaN → 0 and the pos_weights are policy-matched (all precomputed in `data/splits/chexpert/training_pos_weights.json`).

**Evidence to watch:**
- **Pneumonia, Consolidation, Atelectasis and Enlarged Cardiomediastinum** are where the policies diverge most (U-One multiplies positives by 4.1×, 2.9×, 2.0× and 2.1×).
- **Support Devices, Fracture and Lung Opacity** are essentially unaffected.

**Report per-label results.** The best policy may differ by observation, but choosing a policy per label is itself a hyperparameter selection and must be made on validation only.

## 25. Remaining methodological decisions (require approval)

1. Approve excluding the 6 confirmed near-blank images (5 train, 1 val).
2. Approve NaN → 0 as the training assumption, with an optional NaN-masked sensitivity run.
3. Define the primary evaluation label set. Options: all 14 macro, or also the 5 labels commonly reported for CheXpert (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion) as a secondary summary.
4. Define the role of the official valid set: secondary evaluation only, with low-positive labels reported descriptively.
5. Settle the preprocessing policy: aspect-ratio-preserving resize, input resolution, and augmentation. This is to be decided by experiment or by a documented convention, not assumed.
6. Settle the compute budget. The proposed matrix (Q20) has 21 runs on an RTX 3050 6 GB GPU. The CUDA build of PyTorch is not yet installed.
7. For IU: whether retrieval evaluation should also exclude or stratify templated normal reports.

---

## Answers to the Phase 1 analytical questions

1. **Overall severity.** Imbalance is severe and highly heterogeneous. The effective negative : positive ratio ranges from 0.92 : 1 to 61.7 : 1. Seven of 14 labels are in the severe or extreme bands (≥ 10 : 1) under NaN → 0.
2. **Most affected.** Pleural Other (61.7 : 1), Pneumonia (32.9), Fracture (23.6), Lung Lesion (23.2), Enlarged Cardiomediastinum (18.5).
3. **Enough positives to train.** All 14 labels have ≥ 1,000 positive frontal training images (minimum: Pleural Other, 1,746) and ≥ 378 positive frontal images in each of val and test (minimum: Pleural Other, test). Every label is trainable, but the five rare labels will have wide confidence intervals.
4. **Labels needing caution.**
   - Pneumonia: 76% of positive-leaning mentions are uncertain, and it is a clinical diagnosis.
   - Pleural Other: rarest, 97% unmentioned.
   - Consolidation, Atelectasis and Enlarged Cardiomediastinum: high uncertainty.
   - No Finding: never explicitly negative; a derived label.
   - Enlarged Cardiomediastinum also shows a large prevalence shift in the official valid set (46.6% vs 4.8%).
5. **Uncertainty per class.** From 0% (No Finding) to 15.1% of images (Atelectasis). Relative to positive mentions it reaches 0.76 (Pneumonia) and 0.65 (Consolidation). See § 9.
6. **Missing labels.** Between 40.4% (Pleural Effusion) and 97.1% (Pleural Other) of labels are blank. 11 of 14 labels are ≥ 50% blank. The official valid set has none.
7. **Weighted BCE.** Yes, it is justified. Training-only weights range 0.77–74.8, and the extreme weights require calibration monitoring.
8. **Focal loss.** Yes: about 83% of label slots are negatives, most of them easy.
9. **Asymmetric loss.** Yes: there are many negatives per image, and NaN → 0 introduces some label noise that ASL's margin mitigates.
10. **Sampling.** Not for the primary experiments. The measured ESS loss (to 46.6%) and co-occurrence distortion (up to 0.13) meet the pre-declared "costly" rule.
11. **Would sampling distort co-occurrence?** Yes. Upweighting rare labels also reweights their frequent partners: Lung Opacity accompanies 55–69% of Pleural Other, Lung Lesion and Pneumonia positives. Conditional probabilities shift by up to 0.13.
12. **Is a random patient split balanced enough?** No, by the pre-declared criterion: 11.7% seeded, 14.9% at the 95th percentile, up to 17.7%.
13. **Is group-aware stratification necessary?** Yes, by the rule. It reduced the worst deviation to 0.13% without breaking patient grouping or altering test prevalence.
14. **Frontal-only primary experiment?** Yes.
    - Frontal images are 85.5% of the data, and labels are study-level, so laterals add no new labels.
    - Laterals have different anatomy and projection.
    - A single-image DenseNet121 is naturally a frontal model.
    - Only 16 training studies lack a frontal image.
15. **Laterals.** Keep them in all manifests (they are split with their patients). Do not use them in the primary experiment. Candidate later ablation: laterals as extra training images, or study-level frontal + lateral fusion.
16. **Uncertainty strategies to compare.** U-Masked vs U-Zero vs U-One, with NaN → 0, reported per label.
17. **NaN handling.** Map NaN → 0 at load time, as a documented assumption. Masking NaN inverts the class ratio for 10 of 14 labels and leaves No Finding with zero negatives.
18. **Label consistency problems.** None of the No Finding + pathology type: 0 conflicts among 22,419 No Finding images. 8,808 No Finding images also have Support Devices = 1, which is consistent with CheXpert's definition. Labels are identical across all images of every study.
19. **Quality issues to fix before training.** Exclude 6 blank images (pending approval). The cross-patient duplicate identities are already handled by split grouping. Nothing else requires exclusion: 3 age-0 records, 17 frontal LL/RL codes and 1 unknown sex are metadata issues only.
20. **Controlled experiment matrix.** Fixed across all runs:
    - DenseNet121 (ImageNet-pretrained), frontal only, stratified patient split (seed 42);
    - NaN → 0, the same preprocessing and augmentation;
    - model selection on val only, 3 seeds per configuration.

    | Stage | Run | Loss | Uncertainty | Purpose |
    |---|---|---|---|---|
    | A | A1 | BCE | U-Masked | baseline |
    | A | A2 | Positive-weighted BCE (train-only weights) | U-Masked | imbalance |
    | A | A3 | Focal γ = 1 | U-Masked | imbalance |
    | A | A4 | Focal γ = 2 | U-Masked | imbalance |
    | A | A5 | Asymmetric loss (γ⁺ 0, γ⁻ 4, clip 0.05) | U-Masked | imbalance |
    | B | B1 | best-on-val loss from Stage A | U-Zero | uncertainty |
    | B | B2 | best-on-val loss from Stage A | U-One | uncertainty |

    - **Size:** 7 configurations × 3 seeds = 21 runs.
    - **U-Masked** is a *reference* for Stage A so that loss comparisons are not confounded by uncertain-label noise. It is not a pre-selected winner, and Stage B tests it directly.
    - **Evaluation:** per-label AUROC and AUPRC, with patient-level bootstrap CIs and paired comparisons.
    - **Test split:** touched once, only for the final selected configuration and the baseline.
