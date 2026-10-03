"""C6 documents, generated from the stored C6 artifacts (no number is typed by hand):
FINAL_CLASSIFIER_OUTPUT_SCHEMA.json, FINAL_CLASSIFIER_SPECIFICATION.md, FINAL_CLASSIFIER_TEST_REPORT.md,
MANUSCRIPT_C6_METHODS.md, MANUSCRIPT_C6_RESULTS.md, FIGURE_CAPTIONS.md, TABLE_CAPTIONS.md, C6_JOURNAL_ASSET_INDEX.md

    .venv\\Scripts\\python.exe -m scripts.c6_05_write_docs
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.classification.final_test import apply_frozen_pipeline, load_frozen, output_record, output_schema, validate_record
from src.classification.labels import LABELS
from src.classification.serialization import load_predictions_npz
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
OUT = EXP / "c6_final_test"
rd = lambda n: pd.read_csv(OUT / n, float_precision="round_trip")  # noqa: E731
jl = lambda n: json.loads((OUT / n).read_text(encoding="utf-8"))  # noqa: E731
f3, f4 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}")  # noqa: E731
wc = lambda t: len(t.split())  # noqa: E731


def main() -> int:
    man, integ, run, res = jl("FINAL_CLASSIFIER_FREEZE_MANIFEST.json"), jl("test_split_integrity.json"), jl("test_inference_run.json"), jl("c6_results.json")
    rk, bt, cal = rd("test_ranking_metrics.csv").set_index("observation"), rd("test_binary_metrics_per_class.csv").set_index("observation"), rd("test_calibration_per_class.csv").set_index("observation")
    cmp_, pcv, boot = rd("validation_vs_test_summary.csv").set_index("metric"), rd("validation_vs_test_per_class.csv").set_index("observation"), rd("test_bootstrap_summary.csv").set_index("metric")
    rare, nf_df, cnt = rd("test_rare_class_analysis.csv"), rd("test_no_finding_analysis.csv").set_index("split"), rd("test_finding_counts.csv")
    dist, err = rd("test_finding_count_distribution.csv"), rd("test_error_examples.csv")
    sm, b14, b13, bmi, cs = res["ranking"], res["binary_macro14"], res["binary_macro13"], res["binary_micro"], res["calibration"]
    mc, mr = cs["macro_calibrated"], cs["macro_raw_score_for_reference"]
    n_te, n_pat = res["n_images"], res["n_patients"]
    ic = integ["image_counts_after_frontal_and_exclusion_filters"]
    pcn = integ["patient_counts"]
    ci = lambda k, g=f3: f"{g(boot.loc[k, 'point_estimate'])} (95% CI {g(boot.loc[k, 'ci95_low'])} to {g(boot.loc[k, 'ci95_high'])})"  # noqa: E731
    d = lambda k: cmp_.loc[k, "difference_test_minus_validation"]  # noqa: E731
    nft, nfv = nf_df.loc["test"], nf_df.loc["validation"]
    cte = lambda who, defn="13_non_no_finding", sp="test": cnt[(cnt.split == sp) & (cnt.definition == defn) & (cnt.source == who)].iloc[0]  # noqa: E731
    gt, pr = cte("ground_truth"), cte("predicted")
    auroc, auprc = rk["auroc"], rk["auprc"]
    dau = (pcv.test_auroc - pcv.validation_auroc)
    rt, rv = rare[rare.split == "test"].set_index("observation"), rare[rare.split == "validation"].set_index("observation")
    fpe, fne = err[err.error_type == "high_confidence_false_positive"], err[err.error_type == "high_confidence_false_negative"]
    thr = man["f1_thresholds_raw_score"]
    plat = man["calibration"]["parameters"]
    ece_worst = cal.ece.idxmax()
    n_auprc_gt = int((rk['auprc'] > rk['prevalence']).sum())
    n_rec_gt = int((bt['recall'] > bt['precision']).sum())
    excl = (1 - rk["n_valid"] / n_te)
    sha = man["checkpoint_sha256"]
    ef = man["environment"]

    # ============================================================ output schema + example (validation image, not a test image)
    policy, calib = load_frozen(EXP / "c4_operating_policy/final_operating_policy.json", EXP / "c5_calibration/final_calibrators.json")
    va = load_predictions_npz(EXP / "c2b_sqrt_weighted_bce/evaluation/fp32_strict/validation_predictions.npz")
    o = apply_frozen_pipeline(va["probs"][:1], policy, calib)
    ex = output_record("example_validation_image_0", o["scores"][0], o["prob"][0], o["thresholds"], o["calibrated_thresholds"], o["pred_final"][0])
    validate_record(ex)
    (OUT / "FINAL_CLASSIFIER_OUTPUT_SCHEMA.json").write_text(json.dumps({
        "version": "c6-v1", "label_order": list(LABELS),
        "notes": ["Multi-label output: the 14 entries are independent; probabilities are never normalised or softmaxed across classes.",
                  "raw_score is the DenseNet-121 sigmoid output and is a model score, not a probability.",
                  "calibrated_probability (C5 Platt scaling) is for interpretable probability reporting only.",
                  "threshold (frozen C4 F1-optimal, raw-score scale) alone determines `positive`; calibrated_equivalent_threshold reproduces the same decision on the calibrated scale.",
                  "No Finding is positive only if its own raw score >= its threshold AND none of the 12 pathology findings is positive (Support Devices does not suppress it).",
                  "calibrated_probability estimates P(CheXpert label positive | image) under the project label policy, not a clinical ground-truth probability.",
                  "The example below was computed on the first VALIDATION image; no locked-test image is used as an example."],
        "schema": output_schema(), "example": ex}, indent=2, allow_nan=False), encoding="utf-8")

    # ============================================================ specification
    S = []
    A = S.append
    A("# Final classifier specification (frozen at C6)")
    A("")
    A(f"_Generated by `scripts/c6_05_write_docs.py`. Freeze manifest: `FINAL_CLASSIFIER_FREEZE_MANIFEST.json` (written {man['created_utc']}, before any test inference)._")
    A("")
    A("## Architecture")
    A(f"{man['model_architecture']}.")
    A("")
    A("## Training loss")
    A(f"√-weighted binary cross-entropy (`{man['loss']}`: positive weights from the training split, square-root tempered). Selected in the C2 controlled study by validation macro AUPRC; no test information was used.")
    A("")
    A("## Checkpoint")
    A(f"- Path: `{man['checkpoint']}` (epoch {man['checkpoint_epoch']})")
    A(f"- SHA-256: `{sha}`")
    A("")
    A("## Preprocessing (exact)")
    pp = man["preprocessing"]
    A(f"Frontal view only; images in the exclusion list `{man['exclusions_file']}` are dropped. Image size {pp['image_size']}; resize `{pp['resize']}` ({pp['interpolation']} interpolation) with `{pp['pad_position']}` padding of value {pp['pad_value']}; {pp['channels']} channels (grayscale replicated); normalisation mean {pp['mean']}, std {pp['std']}. No augmentation at inference, no test-time augmentation, no ensembling. Numerical policy `{man['inference_policy']}` (float32, TF32 off, deterministic cuDNN).")
    A("")
    A("## Output classes (fixed order)")
    A(", ".join(f"{i + 1}. {l}" for i, l in enumerate(LABELS)) + ". Fourteen independent sigmoid outputs; never softmax, never normalised across classes.")
    A("")
    A("Label policy used for all metrics: uncertain (−1) labels are excluded; blank labels are scored negative.")
    A("")
    A("## Binary policy")
    A("C4 F1-optimal per-class thresholds on the **raw** score: `positive = raw_score >= threshold`.")
    A("")
    A("| Observation | Raw-score threshold | Calibrated-equivalent threshold |\n|---|---:|---:|")
    for l in LABELS:
        A(f"| {l} | {thr[l]:.6f} | {plat[l]['calibrated_equivalent_threshold']:.6f} |")
    A("")
    A("## No Finding rule")
    A(f"`{man['no_finding_rule']}`: No Finding is positive only if its own threshold is met and none of the 12 pathology findings ({', '.join(man['no_finding_rule_abnormal_labels'])}) is positive. Support Devices does **not** suppress No Finding.")
    A("")
    A("## Calibration")
    A("C5 Platt scaling for all 14 classes, p = sigmoid(a · logit(raw score) + b), fitted on the full validation set only.")
    A("")
    A("| Observation | a | b |\n|---|---:|---:|")
    for l in LABELS:
        A(f"| {l} | {plat[l]['a']:.6f} | {plat[l]['b']:.6f} |")
    A("")
    A("## Final output for every class")
    A("- raw model score")
    A("- calibrated probability")
    A("- positive/negative binary decision")
    A("")
    A("Machine-readable record: `FINAL_CLASSIFIER_OUTPUT_SCHEMA.json`.")
    A("")
    A("> Calibrated probability is used for interpretable probability reporting. The frozen C4 raw-score operating thresholds determine binary finding activation.")
    A("")
    A("The calibrated probability estimates the probability of a positive CheXpert label under the stated label policy, not a clinical ground-truth probability.")
    A("")
    A("## Final locked-test performance")
    A(f"Locked test: {n_te:,} frontal images, {n_pat:,} patients (one inference pass).")
    A("")
    A("| Metric | Value (95% patient-bootstrap CI where available) |\n|---|---|")
    for lab, k, g in (("Macro AUROC", "macro_auroc", f3), ("Macro AUPRC", "macro_auprc", f3), ("Macro precision", "macro_precision", f3), ("Macro recall", "macro_recall", f3),
                      ("Macro specificity", "macro_specificity", f3), ("Macro F1", "macro_f1", f3), ("Macro balanced accuracy", "macro_balanced_accuracy", f3), ("Macro Brier (calibrated)", "macro_brier", f4)):
        A(f"| {lab} | {ci(k, g)} |")
    A(f"| Micro AUROC / micro AUPRC | {f3(sm['micro_auroc'])} / {f3(sm['micro_auprc'])} |")
    A(f"| Macro log loss / ECE (calibrated) | {f4(mc['log_loss'])} / {f4(mc['ece'])} |")
    A("")
    A("These are research-prototype performance figures for CheXpert-label prediction. They are not evidence of clinical safety or readiness.")
    A("")
    A("> No further classifier, threshold, calibration or post-processing tuning will be performed based on this test set.")
    A("")
    (OUT / "FINAL_CLASSIFIER_SPECIFICATION.md").write_text("\n".join(S), encoding="utf-8")

    # ============================================================ test report
    t = {n: (OUT / f"tables/{n}.md").read_text(encoding="utf-8") for n in
         ("table1_final_overall_performance", "table2_per_class_final_performance", "table3_validation_vs_test", "table4_rare_class_results", "table5_locked_test_calibration")}
    R = []
    A = R.append
    A("# Final classifier locked-test report (C6)")
    A("")
    A("_Generated from the C6 artifacts by `scripts/c6_05_write_docs.py`._")
    A("")
    A("## 1. Objective")
    A("")
    A("Evaluate the frozen classification pipeline once on the locked test split and freeze it as the classifier for downstream retrieval and report generation. No tuning followed the test evaluation; results are reported as obtained.")
    A("")
    A("## 2. Frozen configuration")
    A("")
    A(f"DenseNet-121, √-weighted BCE, checkpoint SHA-256 `{sha}`; C4 F1-optimal raw-score thresholds with the No Finding rule (12 pathology findings suppress it, Support Devices does not); C5 Platt calibration for all 14 classes. The full configuration, with hashes of every frozen file, was written to `FINAL_CLASSIFIER_FREEZE_MANIFEST.json` at {man['created_utc']}, and the test inference finished at {run['finished_utc']}. Both are re-verified by `c6_integrity_report.json`. See `FINAL_CLASSIFIER_SPECIFICATION.md` for every parameter.")
    A("")
    A("## 3. Test-set integrity")
    A("")
    A(f"- Images (frontal, after the frozen exclusion list): train {ic['train']:,}, validation {ic['val']:,}, test {ic['test']:,}. Patients: train {pcn['train']:,}, validation {pcn['val']:,}, test {pcn['test']:,}.")
    A(f"- Overlap counts (patients / split groups / image paths): " + "; ".join(f"{k.replace('_and_', ' ∩ ')} = {v['patients']} / {v['split_groups']} / {v['image_paths']}" for k, v in integ["overlap_counts"].items()) + ".")
    A(f"- Status: **{integ['status']}**. Provenance: {integ['split_provenance']['method']} split, seed {integ['split_provenance']['seed']}, unit = {integ['split_provenance']['unit']}.")
    A("- The `locked_test_split_used` flag is false in every C3–C5 result file that records it; no code before C6 unlocked the test split; no result file before C6 carries test predictions. Thresholds and calibrators were fitted on validation only, and model/loss/checkpoint selection used validation only.")
    A(f"- Inference: one pass, {run['n_images']:,} images in {run['seconds']:.0f} s, policy `{run['inference_policy']['name']}`, no test-time augmentation, no ensembling; scores range {run['score_min']:.1e}–{run['score_max']:.4f}.")
    A(f"- Validation reference: recomputed with the same functions and identical to the stored C2–C5 results (`validation_reproduction.json`: reproduced = {res['validation_reproduction']['reproduced']}).")
    A("")
    A("## 4. Ranking results")
    A("")
    A(f"Macro AUROC {ci('macro_auroc')}, macro AUPRC {ci('macro_auprc')}; micro AUROC {f3(sm['micro_auroc'])}, micro AUPRC {f3(sm['micro_auprc'])}. Per-class AUROC ranges from {f3(auroc.min())} ({auroc.idxmin()}) to {f3(auroc.max())} ({auroc.idxmax()}); AUPRC from {f3(auprc.min())} ({auprc.idxmin()}) to {f3(auprc.max())} ({auprc.idxmax()}).")
    A("")
    A("**Table 2. Per-class performance.**")
    A("")
    A(t["table2_per_class_final_performance"])
    A("## 5. Binary results")
    A("")
    A(f"Under the frozen policy the macro (14-class) precision is {ci('macro_precision')}, recall {ci('macro_recall')}, specificity {ci('macro_specificity')}, F1 {ci('macro_f1')} and balanced accuracy {ci('macro_balanced_accuracy')}. Excluding No Finding (13 classes) the macro F1 is {f3(b13['macro_f1'])}. Pooled micro precision / recall / F1 are {f3(bmi['micro_precision'])} / {f3(bmi['micro_recall'])} / {f3(bmi['micro_f1'])}.")
    A("")
    A("**Table 1. Overall locked-test performance.**")
    A("")
    A(t["table1_final_overall_performance"])
    A("## 6. Calibration results")
    A("")
    A(f"With the frozen Platt calibrators the macro Brier score is {ci('macro_brier', f4)}, the macro log loss {f4(mc['log_loss'])} and the macro ECE {f4(mc['ece'])} (10 adaptive bins, as in C5). For reference the raw scores give {f4(mr['brier'])}, {f4(mr['log_loss'])} and {f4(mr['ece'])}. The calibrated Brier score and log loss are lower than the raw values in {int(round(cs['fraction_of_classes_with_lower_brier_after_calibration'] * 14))} and {int(round(cs['fraction_of_classes_with_lower_log_loss_after_calibration'] * 14))} of 14 classes, respectively. The largest per-class ECE is {f4(cal.ece.max())} ({ece_worst}).")
    A("")
    A("**Table 5. Locked-test calibration.**")
    A("")
    A(t["table5_locked_test_calibration"])
    A("## 7. Rare-class analysis")
    A("")
    A("The six rare classes are those defined in C2 (not redefined from test prevalence).")
    A("")
    A(t["table4_rare_class_results"])
    A("Pleural Other: " + f"precision {f3(rt.loc['Pleural Other', 'precision'])}, recall {f3(rt.loc['Pleural Other', 'recall'])}, {rt.loc['Pleural Other', 'fp_per_tp']:.2f} false positives per true positive on test ({rv.loc['Pleural Other', 'fp_per_tp']:.2f} on validation). Across the six classes the test precision ranges {f3(rt.precision.min())}–{f3(rt.precision.max())} and the false positives per true positive {rt.fp_per_tp.min():.2f}–{rt.fp_per_tp.max():.2f}.")
    A("")
    A("## 8. No Finding analysis")
    A("")
    A(f"- Test prevalence {f3(nft.no_finding_prevalence)}; after the rule: precision {f3(nft.precision)}, recall {f3(nft.recall)}, specificity {f3(nft.specificity)}, F1 {f3(nft.f1)}; {nft.pct_predicted_no_finding_after_rule:.2f}% of images are predicted No Finding.")
    A(f"- Before the rule {nft.pct_predicted_no_finding_before_rule:.2f}% of images were predicted No Finding, and {int(nft.contradiction_n_before_rule)} images ({nft.contradiction_pct_of_images_before_rule:.2f}% of images; {nft.contradiction_pct_of_predicted_no_finding_before_rule:.1f}% of those predicted No Finding) also had a pathology finding predicted positive. After the rule: {int(nft.contradiction_n_after_rule)}. Images modified: {int(nft.images_modified_by_rule)}.")
    A(f"- Ground-truth contradictions in the test labels (No Finding with an explicit positive among the 12 pathology labels): {int(nft.ground_truth_contradiction_with_12_pathology_n)}; No Finding together with Support Devices: {int(nft.ground_truth_no_finding_with_support_devices_n)}.")
    A(f"- Pre-rule vs post-rule No Finding F1: {f3(nft.pre_rule_f1)} vs {f3(nft.f1)} (the rule trades recall for consistency; it was not changed).")
    A("")
    A("## 9. Validation-to-test generalization")
    A("")
    A("**Table 3. Validation versus locked test.**")
    A("")
    A(t["table3_validation_vs_test"])
    A(f"Differences are descriptive only. Macro AUROC differs by {d('macro_auroc'):+.4f}, macro AUPRC by {d('macro_auprc'):+.4f}, macro F1 by {d('macro_f1'):+.4f}, macro Brier by {d('macro_brier'):+.4f} and macro ECE by {d('macro_ece'):+.4f}. Per-class AUROC differences range {dau.min():+.3f} ({dau.idxmin()}) to {dau.max():+.3f} ({dau.idxmax()}). The validation calibration values are in-sample for the frozen calibrators (out-of-fold macro ECE {f4(cmp_.loc['macro_ece', 'validation_out_of_fold'])}).")
    A("")
    A("## 10. Bootstrap uncertainty")
    A("")
    A(f"{res['bootstrap']['n_resamples']} patient-level resamples (seed {res['bootstrap']['seed']}); all images of a resampled patient are repeated together; model, thresholds and calibrators are held fixed, so the intervals reflect test-sample variability only.")
    A("")
    A("| Metric | Point estimate | 95% CI |\n|---|---:|---|")
    for k in boot.index:
        A(f"| {k} | {f4(boot.loc[k, 'point_estimate'])} | {f4(boot.loc[k, 'ci95_low'])} to {f4(boot.loc[k, 'ci95_high'])} |")
    A("")
    A("## 11. Finding counts and error analysis")
    A("")
    A(f"Findings per image (13 observations excluding No Finding): true mean {gt['mean']:.2f} (median {gt['median']:.0f}, 25th–75th percentile {gt['p25']:.0f}–{gt['p75']:.0f}, 95th percentile {gt['p95']:.0f}, maximum {gt['max']:.0f}); predicted mean {pr['mean']:.2f} (median {pr['median']:.0f}, {pr['p25']:.0f}–{pr['p75']:.0f}, 95th {pr['p95']:.0f}, maximum {pr['max']:.0f}). Share of images with 0 / 1 / 2–3 / 4–5 / >5 findings: true {gt['pct_0']:.1f} / {gt['pct_1']:.1f} / {gt['pct_2_3']:.1f} / {gt['pct_4_5']:.1f} / {gt['pct_gt5']:.1f}%; predicted {pr['pct_0']:.1f} / {pr['pct_1']:.1f} / {pr['pct_2_3']:.1f} / {pr['pct_4_5']:.1f} / {pr['pct_gt5']:.1f}%. The policy therefore predicts more findings per image than the labels contain, which downstream retrieval and report generation must tolerate.")
    A("")
    A(f"`test_error_examples.csv` lists, for every class, the five highest-scoring false positives and five lowest-scoring false negatives ({len(err)} rows; anonymised identifiers). Among the false-positive examples the median raw score is {fpe.raw_score.median():.3f} and {int((fpe.n_true_positive_labels_in_image > 0).sum())} of {len(fpe)} come from images with at least one other true positive label; among the false-negative examples the median raw score is {fne.raw_score.median():.3f}. This is a descriptive listing only; no radiological interpretation was made and nothing was changed on its basis.")
    A("")
    A("## 12. Final classifier freeze")
    A("")
    A("The classifier, thresholds, No Finding rule and calibrators are frozen as documented in `FINAL_CLASSIFIER_SPECIFICATION.md`; the downstream output record is defined in `FINAL_CLASSIFIER_OUTPUT_SCHEMA.json`. No further tuning will be performed using this test set.")
    A("")
    A("## 13. Limitations")
    A("")
    A("- Single dataset (CheXpert, one institution) and frontal views only; no external validation.")
    A("- Reference labels are automatically extracted from reports (CheXpert labeler), with uncertain labels excluded: " + f"{100 * excl.min():.1f}–{100 * excl.max():.1f}% of images are excluded per class ({excl.idxmax()} the most). Metrics describe agreement with these labels, not radiologist-verified findings.")
    A("- Several classes have modest discrimination: " + f"AUROC {f3(auroc.min())} for {auroc.idxmin()}, and macro precision is {f3(b14['macro_precision'])}; F1-optimal thresholds favour recall, so false positives are frequent for rare classes ({rt.fp_per_tp.max():.1f} per true positive at most).")
    A(f"- The policy predicts more findings per image ({pr['mean']:.2f}) than are labelled ({gt['mean']:.2f}).")
    A("- Bootstrap intervals cover test-sample variability only, not training variability, threshold or calibrator uncertainty.")
    A("- Calibrated probabilities estimate P(CheXpert label positive | image) under the project label policy, not clinical ground-truth probabilities; calibration was fitted on the same validation set used to select thresholds and may drift under dataset shift.")
    A("- No expert review of errors, no assessment of clinical utility or safety; this is a research prototype component, not a diagnostic system.")
    A("")
    (OUT / "FINAL_CLASSIFIER_TEST_REPORT.md").write_text("\n".join(R), encoding="utf-8")

    # ============================================================ manuscript Methods
    meth = f"""# Methods: locked-test evaluation and classifier freeze (C6)

## Data and split independence

Experiments used frontal chest radiographs from CheXpert, partitioned at the patient level (patients with byte-identical images were merged into one split group) into training ({ic['train']:,} images, {pcn['train']:,} patients), validation ({ic['val']:,} images, {pcn['val']:,} patients) and a locked test partition ({ic['test']:,} images, {pcn['test']:,} patients). Before inference we verified that no patient, split group or image path occurred in more than one partition, and we confirmed from the repository records that the test partition had not been used for training, loss selection, checkpoint selection, threshold selection or calibration fitting. Uncertain labels were excluded from every metric and blank labels were scored as negative.

## Frozen classifier

The classifier was a DenseNet-121 with 14 independent sigmoid outputs, trained with a square-root-weighted binary cross-entropy loss; the checkpoint was selected on validation data (SHA-256 recorded). Binary decisions used class-specific thresholds that maximised the F1 score on the validation set, applied to the raw sigmoid score. A consistency rule then suppressed No Finding whenever any of the 12 pathology findings was positive; Support Devices did not suppress it. Probabilities were obtained by per-class Platt scaling of the logit of the raw score, fitted on the validation set only. Calibration affected the displayed probabilities but not the binary decisions, because the calibrated-equivalent thresholds reproduce the raw decisions exactly. Before any test metric was computed, the complete configuration (checkpoint hash, preprocessing, label policy, all thresholds, all calibration parameters, code commits, software versions) and a hash of every frozen file were written to a freeze manifest, and the manifest was re-verified before evaluation.

## Test inference and metrics

The frozen model was run once on the test partition with canonical single-precision inference, without test-time augmentation or ensembling. We report per-class and macro-averaged AUROC and AUPRC, micro-averaged AUROC and AUPRC, and, for the binary decisions, precision, recall, specificity, F1 score and balanced accuracy (macro averages over the 14 classes; micro-averaged precision, recall and F1 as secondary measures). Calibration was summarised by the Brier score, log loss and expected calibration error with 10 equal-frequency bins, as in the calibration study. Rare classes were the six defined during the loss study, not redefined from test prevalence. We additionally describe No Finding behaviour before and after the consistency rule, the distribution of positive findings per image, and the highest-scoring false positives and lowest-scoring false negatives; no radiological interpretation was attempted.

## Uncertainty and generalisation

Uncertainty was estimated with {man['bootstrap']['n_resamples']} patient-level bootstrap resamples (seed {man['bootstrap']['seed']}): patients were drawn with replacement and all of a patient's images were included together, with the model, thresholds and calibrators held fixed. Validation results were recomputed with the same code and matched the stored validation results exactly; differences between validation and test are reported descriptively. No threshold, calibrator, rule or checkpoint was modified after the test partition was opened, and the bootstrap was used for uncertainty reporting only.
"""
    mw = wc(meth)
    (OUT / "MANUSCRIPT_C6_METHODS.md").write_text(meth, encoding="utf-8")

    # ============================================================ manuscript Results
    rare_txt = "; ".join(f"{l} (test prevalence {f3(rt.loc[l, 'prevalence'])}, AUROC {f3(rt.loc[l, 'auroc'])}, AUPRC {f3(rt.loc[l, 'auprc'])}, precision {f3(rt.loc[l, 'precision'])}, recall {f3(rt.loc[l, 'recall'])}, {rt.loc[l, 'fp_per_tp']:.1f} false positives per true positive)" for l in rare.observation.drop_duplicates())
    resu = f"""# Results: locked-test evaluation (C6)

## Test set and independence

The locked test partition contained {n_te:,} frontal images from {n_pat:,} patients. No patient, split group or image path overlapped with the training or validation partitions (all overlap counts were zero), and the frozen configuration was recorded before inference.

## Ranking performance

On the locked test set the macro AUROC was {ci('macro_auroc')} and the macro AUPRC {ci('macro_auprc')}; the micro-averaged values were {f3(sm['micro_auroc'])} and {f3(sm['micro_auprc'])}. Per-class AUROC ranged from {f3(auroc.min())} ({auroc.idxmin()}) to {f3(auroc.max())} ({auroc.idxmax()}), and per-class AUPRC from {f3(auprc.min())} ({auprc.idxmin()}) to {f3(auprc.max())} ({auprc.idxmax()}); AUPRC exceeded the class prevalence in {n_auprc_gt} of 14 classes (Table 2, Figure 2).

## Binary decisions

With the frozen operating policy, the macro precision was {ci('macro_precision')}, recall {ci('macro_recall')}, specificity {ci('macro_specificity')}, F1 {ci('macro_f1')} and balanced accuracy {ci('macro_balanced_accuracy')} (Table 1, Figure 3). Pooled micro precision, recall and F1 were {f3(bmi['micro_precision'])}, {f3(bmi['micro_recall'])} and {f3(bmi['micro_f1'])}. Recall exceeded precision in {n_rec_gt} of 14 classes, consistent with F1-optimal thresholds on imbalanced labels.

## Calibration

The frozen Platt calibrators gave a macro Brier score of {ci('macro_brier', f4)}, a macro log loss of {f4(mc['log_loss'])} and a macro ECE of {f4(mc['ece'])}, compared with {f4(mr['brier'])}, {f4(mr['log_loss'])} and {f4(mr['ece'])} for the raw scores (Table 5). The largest per-class ECE was {f4(cal.ece.max())} ({ece_worst}). Reliability curves for Pleural Other, Pneumothorax and Support Devices are shown in Figure 5. The calibrated-equivalent thresholds reproduced the raw-score decisions for all {n_te:,} images in every class.

## Rare classes

The six rare classes defined in the loss study were: {rare_txt}. Validation values were similar (Table 4, Figure 4); for example, Pleural Other had {rv.loc['Pleural Other', 'fp_per_tp']:.1f} false positives per true positive on validation.

## No Finding

No Finding had a test prevalence of {f3(nft.no_finding_prevalence)}. Before the consistency rule, {int(nft.contradiction_n_before_rule)} images ({nft.contradiction_pct_of_images_before_rule:.2f}% of images) were predicted No Finding together with at least one pathology finding; the rule modified {int(nft.images_modified_by_rule)} images and left no such contradictions. After the rule the precision, recall, specificity and F1 for No Finding were {f3(nft.precision)}, {f3(nft.recall)}, {f3(nft.specificity)} and {f3(nft.f1)}, and {nft.pct_predicted_no_finding_after_rule:.1f}% of images were predicted No Finding. The test labels contained {int(nft.ground_truth_contradiction_with_12_pathology_n)} contradictions between No Finding and the 12 pathology labels.

## Findings per image

The labels contained a mean of {gt['mean']:.2f} positive findings per image (median {gt['median']:.0f}; 95th percentile {gt['p95']:.0f}; maximum {gt['max']:.0f}), whereas the final policy predicted a mean of {pr['mean']:.2f} (median {pr['median']:.0f}; 95th percentile {pr['p95']:.0f}; maximum {pr['max']:.0f}). Zero findings were present in {gt['pct_0']:.1f}% of images and predicted for {pr['pct_0']:.1f}%; more than five findings were present in {gt['pct_gt5']:.1f}% and predicted for {pr['pct_gt5']:.1f}% (Figure 6).

## Validation versus test

Test and validation results were close (Table 3, Figure 1). Macro AUROC differed by {d('macro_auroc'):+.4f}, macro AUPRC by {d('macro_auprc'):+.4f}, macro F1 by {d('macro_f1'):+.4f}, macro Brier score by {d('macro_brier'):+.4f} and macro ECE by {d('macro_ece'):+.4f} (test minus validation). Per-class AUROC differences ranged from {dau.min():+.3f} ({dau.idxmin()}) to {dau.max():+.3f} ({dau.idxmax()}). The 95% bootstrap intervals were narrow (for example macro AUROC {f4(boot.loc['macro_auroc', 'ci95_low'])} to {f4(boot.loc['macro_auroc', 'ci95_high'])}; Figure 7) and reflect test-sample variability only.

## Error examples

For each class we listed the five highest-scoring false positives and five lowest-scoring false negatives ({len(err)} descriptive examples with anonymised identifiers). The median raw score was {fpe.raw_score.median():.3f} among the false-positive examples and {fne.raw_score.median():.3f} among the false-negative examples; {int((fpe.n_true_positive_labels_in_image > 0).sum())} of {len(fpe)} false-positive examples came from images with at least one other positive label. These listings were not interpreted radiologically and did not influence the classifier.

## Interpretation and limits

These results describe agreement with automatically extracted CheXpert labels, with uncertain labels excluded, on a single-institution dataset. They do not establish clinical performance, and the frequent false positives for rare classes and the larger number of predicted than labelled findings should be considered in any downstream use. No post-test tuning was performed.
"""
    rw = wc(resu)
    (OUT / "MANUSCRIPT_C6_RESULTS.md").write_text(resu, encoding="utf-8")
    ok_len = 450 <= mw <= 650 and 700 <= rw <= 1000

    # ============================================================ captions + asset index
    fc = f"""# Figure captions (C6)

Locked test: {n_te:,} frontal images from {n_pat:,} patients. Validation: {ic['val']:,} images. Metrics use valid labels only (uncertain excluded, blank scored negative). Binary metrics use the frozen C4 F1-optimal thresholds with the No Finding rule; calibrated values use the frozen C5 Platt calibrators.

**Figure 1. Validation versus locked-test performance.** Macro and micro ranking metrics, macro binary metrics and macro calibration metrics for the frozen classifier on validation (open circles) and the locked test set (filled diamonds). Panels use different horizontal axes. Differences are descriptive; nothing was tuned on the test set.

**Figure 2. Per-class AUROC and AUPRC.** Validation (open circles) and locked-test (filled diamonds) values for the 14 observations. The dashed line marks AUROC 0.5; grey ticks in the right panel mark the test prevalence, which is the AUPRC of a non-informative score.

**Figure 3. Per-class precision, recall and F1 on the locked test set** at the frozen operating thresholds (No Finding after the consistency rule).

**Figure 4. Rare-class performance.** The six rare classes defined in the loss study. Left: AUPRC (grey tick: test prevalence). Centre: precision (circles) and recall (triangles). Right: false positives per true positive. Open markers: validation; filled markers: locked test.

**Figure 5. Reliability diagrams on the locked test set** for the rare (Pleural Other), medium (Pneumothorax) and common (Support Devices) classes fixed in the calibration study. Observed positive fraction against mean predicted value in 10 equal-frequency bins, for the raw sigmoid score and the frozen Platt-calibrated probability; the diagonal is ideal calibration.

**Figure 6. True versus predicted number of positive findings per image** on the locked test set (13 observations excluding No Finding; true = explicit positive labels; predicted = final policy), as a percentage of images.

**Figure 7. Patient-level bootstrap uncertainty.** Locked-test macro metrics (diamonds) with 95% percentile intervals from {res['bootstrap']['n_resamples']} patient-level resamples (seed {res['bootstrap']['seed']}); each panel has its own horizontal range. The intervals cover test-sample variability only.
"""
    tc = """# Table captions (C6)

**Table 1. Final overall locked-test performance.** Macro and micro ranking metrics, macro binary metrics under the frozen operating policy and macro calibration metrics, with 95% patient-level bootstrap intervals where computed.

**Table 2. Per-class final performance.** Prevalence, positive count, AUROC, AUPRC, frozen raw-score threshold, precision, recall, specificity, F1, balanced accuracy and confusion counts for each observation on the locked test set.

**Table 3. Validation versus locked test.** Ranking, binary and calibration summary metrics on validation and the locked test set, with the test-minus-validation difference. Validation calibration values are in-sample for the frozen calibrators.

**Table 4. Rare-class results.** Prevalence, AUROC, AUPRC, precision, recall, specificity, F1, true positives, false positives and false positives per true positive for the six rare classes, on validation and locked test.

**Table 5. Locked-test calibration.** Prevalence, mean calibrated probability, Brier score, log loss and ECE (10 equal-frequency bins) for each observation after frozen Platt calibration, with raw-score Brier and ECE for reference.
"""
    (OUT / "FIGURE_CAPTIONS.md").write_text(fc, encoding="utf-8")
    (OUT / "TABLE_CAPTIONS.md").write_text(tc, encoding="utf-8")
    best_gap = dau.abs().idxmax()
    idx = f"""# C6 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_final_overall_performance.*` | Macro AUROC {f3(sm['macro_auroc'])}, AUPRC {f3(sm['macro_auprc'])}, F1 {f3(b14['macro_f1'])} with bootstrap CIs | Results |
| Table 2 `tables/table2_per_class_final_performance.*` | AUROC {f3(auroc.min())} ({auroc.idxmin()}) to {f3(auroc.max())} ({auroc.idxmax()}) | Results / Supplementary |
| Table 3 `tables/table3_validation_vs_test.*` | Macro AUROC difference {d('macro_auroc'):+.4f} (test minus validation) | Results |
| Table 4 `tables/table4_rare_class_results.*` | Rare-class precision {f3(rt.precision.min())}–{f3(rt.precision.max())} on test | Results |
| Table 5 `tables/table5_locked_test_calibration.*` | Macro ECE {f4(mc['ece'])}, Brier {f4(mc['brier'])} on test | Results |
| Figure 1 `figures/fig1_validation_vs_test.*` | Validation and test summary metrics are close | Results |
| Figure 2 `figures/fig2_per_class_auroc_auprc.*` | Per-class discrimination; AUPRC above prevalence in all classes | Results |
| Figure 3 `figures/fig3_precision_recall_f1.*` | Recall exceeds precision in {n_rec_gt} of 14 classes at F1-optimal thresholds | Results |
| Figure 4 `figures/fig4_rare_class_performance.*` | Rare classes: low precision and several false positives per true positive | Results / Discussion |
| Figure 5 `figures/fig5_test_reliability.*` | Frozen Platt calibration on test vs raw scores | Results |
| Figure 6 `figures/fig6_finding_counts.*` | Policy predicts more findings per image than the labels contain ({pr['mean']:.2f} vs {gt['mean']:.2f}) | Results / Discussion |
| Figure 7 `figures/fig7_bootstrap_ci.*` | Narrow patient-level bootstrap intervals | Results / Supplementary |
| `FINAL_CLASSIFIER_TEST_REPORT.md` | Full locked-test report | Supplementary / internal |
| `MANUSCRIPT_C6_METHODS.md`, `MANUSCRIPT_C6_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `FINAL_CLASSIFIER_SPECIFICATION.md`, `FINAL_CLASSIFIER_FREEZE_MANIFEST.json` | Permanent frozen specification and hashes | Methods / Supplementary / code release |
| `FINAL_CLASSIFIER_OUTPUT_SCHEMA.json` | Downstream output record | Methods / downstream use |
| `test_*.csv`, `test_predictions_*.csv.gz`, `test_bootstrap_draws.npz` | Per-image predictions and all metric tables | Supplementary data |
| `c6_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
"""
    (OUT / "C6_JOURNAL_ASSET_INDEX.md").write_text(idx, encoding="utf-8")

    # ============================================================ wording discipline
    bad = []
    for n in ("MANUSCRIPT_C6_METHODS.md", "MANUSCRIPT_C6_RESULTS.md", "FINAL_CLASSIFIER_TEST_REPORT.md", "FINAL_CLASSIFIER_SPECIFICATION.md", "FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md", "C6_JOURNAL_ASSET_INDEX.md"):
        txt = (OUT / n).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"superior", r"\{[a-z_]+[^}]*\}", r"\bnan\b"):
            if re.search(pat, txt, re.I):
                bad.append((n, pat))
    print(json.dumps({"methods_words": mw, "results_words": rw, "length_ok": ok_len, "wording_flags": bad}))
    return 0 if ok_len and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
