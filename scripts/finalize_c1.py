"""Freeze the verified DenseNet121 run as the official C1 baseline (no retraining).

Prerequisite: `scripts.evaluate_classifier` has produced the canonical evaluation
(<exp>/evaluation/fp32_strict/).

Steps
  1. hash the checkpoint (must stay byte-identical)
  2. ORIGINAL C1 (amp_fp16_legacy_c1): reproduce the original in-memory probabilities,
     prove they match the stored thresholds/sens/spec exactly, and replace the lossy
     16-digit prediction CSV with lossless artifacts (17-digit CSV + .npz);
     regenerate the CSV-derived operating-point files from exact probabilities
  3. compare ORIGINAL vs CANONICAL (same checkpoint, evaluation-consistency correction)
  4. write C1_BASELINE_METRICS.json, C1_BASELINE_SUMMARY.md, C1_PROVENANCE.json

    .venv\\Scripts\\python.exe -m scripts.finalize_c1
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.classification.checkpointing import load_checkpoint
from src.classification.dataset import CheXpertDataset, label_policy_from_config, load_split_frame
from src.classification.evaluator import predict
from src.classification.inference_policy import CANONICAL, LEGACY_C1_AMP
from src.classification.labels import LABELS
from src.classification.serialization import (load_predictions_csv, load_predictions_npz, roundtrip_report,
                                              save_predictions)
from src.preprocessing.transforms import EvalTransform, PreprocessConfig
from src.utils.config import PROJECT_ROOT, load_paths

CFG_PATH = PROJECT_ROOT / "configs/classifier/densenet121.yaml"
EXP = PROJECT_ROOT / "results/classification/densenet121_baseline"
CANON = EXP / "evaluation" / CANONICAL.name
CKPT = EXP / "checkpoints/best.pt"
OUTROOT = PROJECT_ROOT / "results/classification"
# SHA-256 of the original files as audited in CLASSIFIER_VERIFICATION_REPORT.md (recorded before the audit)
ORIGINAL_CSV_SHA256 = "288ae5912e473e59072286202f5397e67649344a47d9d6c82460583a4cbd5a33"
ORIGINAL_OP_SHA256 = "21634762d28e46dc75e9d2ccbbacf5f62899729e46235926a547d94ea6593a8a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rel(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def thresholded(raw: np.ndarray, valid: np.ndarray, p: np.ndarray, thr: np.ndarray) -> pd.DataFrame:
    y = raw == 1
    rows = []
    for j, lab in enumerate(LABELS):
        v = valid[:, j]
        yj, pr = y[v, j], p[v, j] >= thr[j]
        tp, fp, fn, tn = int((pr & yj).sum()), int((pr & ~yj).sum()), int((~pr & yj).sum()), int((~pr & ~yj).sum())
        sens, prec = tp / (tp + fn), (tp / (tp + fp) if tp + fp else np.nan)
        rows.append({"observation": lab, "threshold": thr[j], "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "sensitivity": sens, "specificity": tn / (tn + fp), "precision": prec, "recall": sens,
                     "f1": 2 * prec * sens / (prec + sens), "prevalence": yj.mean(), "predicted_positive_rate": pr.mean()})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    log: dict = {"checkpoint_sha256_before": sha256(CKPT)}
    if not (CANON / "validation_predictions.npz").exists():
        raise FileNotFoundError(f"run scripts.evaluate_classifier first ({CANON} missing)")

    # ------------------------------------------------ 2. ORIGINAL C1 (legacy amp) reproduced + lossless re-serialisation
    device = torch.device("cuda")
    model, meta = load_checkpoint(CKPT, device)
    paths = load_paths()
    frame = load_split_frame(paths, cfg["data"], "val")
    ds = CheXpertDataset(frame, paths.chexpert_image_base, label_policy_from_config(cfg["label_policy"]),
                         EvalTransform(PreprocessConfig.from_dict(cfg["preprocessing"])))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    p_leg = predict(model, loader, device, policy=LEGACY_C1_AMP)["probs"]
    raw = frame[list(LABELS)].to_numpy(dtype=np.float64)
    valid = raw != -1.0
    audit_npz = np.load(OUTROOT / "verification/reinference_probs_fp16.npz", allow_pickle=False)
    thr_orig_file = json.loads((EXP / "thresholds/youden_j_thresholds.json").read_text(encoding="utf-8"))
    thr_orig = np.array([e["threshold"] for e in thr_orig_file["thresholds"]], dtype=np.float64)
    old_csv = EXP / "validation_predictions.csv.gz"
    if sha256(old_csv) != ORIGINAL_CSV_SHA256 or sha256(EXP / "operating_points.csv") != ORIGINAL_OP_SHA256:
        raise RuntimeError("original C1 prediction/operating-point files are not the verified originals; "
                           "refusing to re-serialise (would lose the before/after evidence)")
    old = load_predictions_csv(old_csv)
    if old["paths"] != frame["Path"].tolist():
        raise AssertionError("row order of original prediction CSV differs from the validation frame")
    tab_leg = thresholded(raw, valid, p_leg, thr_orig)
    stored_sens = np.array([e["sensitivity"] for e in thr_orig_file["thresholds"]])
    stored_spec = np.array([e["specificity"] for e in thr_orig_file["thresholds"]])
    log["original_c1_reproduction"] = {
        "policy": LEGACY_C1_AMP.as_dict(),
        "bit_identical_to_audit_reinference": bool(np.array_equal(p_leg, audit_npz["probs"])),
        "max_abs_diff_vs_old_16digit_csv": float(np.abs(p_leg - old["probs"]).max()),
        "stored_sensitivity_max_abs_diff": float(np.abs(tab_leg["sensitivity"].to_numpy() - stored_sens).max()),
        "stored_specificity_max_abs_diff": float(np.abs(tab_leg["specificity"].to_numpy() - stored_spec).max()),
    }
    if not log["original_c1_reproduction"]["bit_identical_to_audit_reinference"] or \
            log["original_c1_reproduction"]["stored_sensitivity_max_abs_diff"] > 1e-12:
        raise AssertionError(f"original C1 probabilities not reproduced exactly: {log['original_c1_reproduction']}")
    # How the old file was actually consumed: pandas' DEFAULT parser (not correctly rounded).
    old_default = pd.read_csv(old_csv)[[f"prob::{l}" for l in LABELS]].to_numpy(dtype=np.float64)
    before = roundtrip_report(p_leg, old_default, thr_orig)
    before["status_mismatches_valid_entries_only"] = int((((p_leg >= thr_orig) != (old_default >= thr_orig)) & valid).sum())
    exact_reader = roundtrip_report(p_leg, old["probs"], thr_orig)  # load_predictions_csv uses round_trip parsing
    log["serialization_before_fix"] = {
        "file": rel(old_csv), "sha256": sha256(old_csv),
        "writer": "pandas to_csv default (Python repr, shortest round-trip digits): lossless",
        "reader_used_downstream": "pandas read_csv default C parser: NOT correctly rounded (root cause)",
        "with_default_reader": before, "with_round_trip_reader": exact_reader,
        "status_mismatches": before["status_mismatches"],
        "status_mismatches_valid_entries_only": before["status_mismatches_valid_entries_only"]}
    files = save_predictions(EXP, "validation_predictions", frame["Path"], frame["patient_id"], p_leg, raw, valid,
                             meta={"policy": LEGACY_C1_AMP.name, "checkpoint_epoch": meta["epoch"], "role": "original C1"})
    after = {"npz": roundtrip_report(p_leg, load_predictions_npz(files["npz"])["probs"], thr_orig),
             "csv_17_digits": roundtrip_report(p_leg, load_predictions_csv(files["csv"])["probs"], thr_orig)}
    log["serialization_after_fix"] = {k: dict(v, file=rel(files[k.split("_")[0]])) for k, v in after.items()}
    if after["npz"]["status_mismatches"] or after["csv_17_digits"]["status_mismatches"]:
        raise AssertionError(f"lossless serialisation still changes statuses: {after}")

    # regenerate the CSV-derived operating points of the ORIGINAL evaluation from exact probabilities
    old_op = pd.read_csv(EXP / "operating_points.csv")
    old_ops = json.loads((EXP / "operating_points_summary.json").read_text(encoding="utf-8"))
    op = tab_leg.rename(columns={"precision": "ppv"})[["observation", "threshold", "prevalence", "predicted_positive_rate",
                                                       "tp", "fp", "fn", "ppv", "sensitivity"]]
    op.to_csv(EXP / "operating_points.csv", index=False, float_format="%.17g")
    from scripts.evaluate_classifier import overcalling
    oc_leg = overcalling(raw, p_leg, thr_orig)
    ops_new = {"n_images": oc_leg["n_images"],  # same schema as the original file
               "predicted_positives_per_image_mean": oc_leg["predicted_positive_findings_per_image_mean"],
               "predicted_positives_per_image_median": oc_leg["predicted_positive_findings_per_image_median"],
               "explicit_true_positives_per_image_mean": oc_leg["actual_positive_findings_per_image_mean"],
               "images_with_zero_predicted_positives": oc_leg["images_with_zero_predicted_positives"],
               "images_with_no_finding_pathology_warning": oc_leg["images_with_no_finding_pathology_warning"],
               "pct_images_with_no_finding_pathology_warning": oc_leg["pct_images_with_no_finding_pathology_warning"]}
    if set(ops_new) != set(old_ops):
        raise AssertionError(f"operating-point schema mismatch: {sorted(set(ops_new) ^ set(old_ops))}")
    (EXP / "operating_points_summary.json").write_text(json.dumps(ops_new, indent=2), encoding="utf-8")
    log["original_operating_points_correction"] = {
        "reason": "previously computed from the lossy 16-digit CSV",
        "ppv_max_abs_change": float(np.abs(old_op.set_index("observation").loc[list(LABELS), "ppv"].to_numpy()
                                           - op["ppv"].to_numpy()).max()),
        "summary_before": old_ops, "summary_after": ops_new}

    # ------------------------------------------------ 3. ORIGINAL vs CANONICAL
    can = load_predictions_npz(CANON / "validation_predictions.npz")
    if can["paths"] != frame["Path"].tolist():
        raise AssertionError("canonical predictions row order differs")
    p_can = can["probs"]
    pcm_orig = pd.read_csv(EXP / "per_class_metrics.csv").set_index("observation").loc[list(LABELS)]
    pcm_can = pd.read_csv(CANON / "per_class_metrics.csv").set_index("observation").loc[list(LABELS)]
    sum_orig = json.loads((EXP / "summary_metrics.json").read_text(encoding="utf-8"))
    sum_can = json.loads((CANON / "summary_metrics.json").read_text(encoding="utf-8"))
    thr_can_file = json.loads((CANON / "thresholds/youden_j_thresholds.json").read_text(encoding="utf-8"))
    thr_can = np.array([e["threshold"] for e in thr_can_file["thresholds"]], dtype=np.float64)
    tab_can = thresholded(raw, valid, p_can, thr_can)
    comp_rows = []
    for j, lab in enumerate(LABELS):
        for metric in ("auroc", "auprc"):
            comp_rows.append({"scope": lab, "metric": metric, "original": pcm_orig.loc[lab, metric],
                              "canonical": pcm_can.loc[lab, metric]})
        comp_rows.append({"scope": lab, "metric": "youden_threshold", "original": thr_orig[j], "canonical": thr_can[j]})
    for metric in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc", "competition5_macro_auroc"):
        comp_rows.append({"scope": "overall", "metric": metric, "original": sum_orig[metric], "canonical": sum_can[metric]})
    comp = pd.DataFrame(comp_rows)
    comp["abs_diff"] = (comp["canonical"] - comp["original"]).abs()
    comp.to_csv(OUTROOT / "C1_original_vs_canonical.csv", index=False, float_format="%.17g")
    s_orig, s_can = p_leg >= thr_orig, p_can >= thr_can
    changes = {
        "probabilities_total": int(p_can.size),
        "probabilities_changed": int((p_can != p_leg).sum()),
        "probability_max_abs_diff": float(np.abs(p_can - p_leg).max()),
        "probability_median_abs_diff": float(np.median(np.abs(p_can - p_leg))),
        "thresholds_changed": int((thr_can != thr_orig).sum()),
        "threshold_max_abs_diff": float(np.abs(thr_can - thr_orig).max()),
        "statuses_changed_all_entries": int((s_orig != s_can).sum()),
        "statuses_changed_valid_entries": int(((s_orig != s_can) & valid).sum()),
        "statuses_changed_if_original_thresholds_kept": int(((p_can >= thr_orig) != s_orig).sum()),
        "auroc_auprc_changed_classes": int(((comp["metric"].isin(["auroc", "auprc"])) & (comp["scope"] != "overall")
                                            & (comp["abs_diff"] > 0)).sum()),
        "max_abs_diff_auroc_auprc_any": float(comp.loc[comp["metric"].isin(["auroc", "auprc", "macro_auroc", "micro_auroc",
                                                                             "macro_auprc", "micro_auprc"]), "abs_diff"].max()),
    }
    log["original_vs_canonical"] = changes

    # ------------------------------------------------ 4. official C1 artifacts
    frozen_cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    tsum = json.loads((EXP / "training_summary.json").read_text(encoding="utf-8"))
    env = json.loads((EXP / "environment.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(EXP / "training_history.csv")
    pw = pd.read_csv(EXP / "train_pos_weights.csv")
    oc = sum_can["overcalling"]
    per_class = []
    for j, lab in enumerate(LABELS):
        r, t = pcm_can.loc[lab], tab_can.iloc[j]
        per_class.append({"observation": lab, "positive_support": int(r["n_positive"]), "negative_support": int(r["n_negative"]),
                          "prevalence": float(r["prevalence"]), "auroc": float(r["auroc"]), "auprc": float(r["auprc"]),
                          "youden_threshold": float(t["threshold"]), "sensitivity": float(t["sensitivity"]),
                          "specificity": float(t["specificity"]), "precision": float(t["precision"]),
                          "recall": float(t["recall"]), "f1": float(t["f1"]), "tp": int(t["tp"]), "fp": int(t["fp"]),
                          "tn": int(t["tn"]), "fn": int(t["fn"]),
                          "predicted_positive_rate": float(t["predicted_positive_rate"])})
    weak = {
        "low_auroc_lt_0.75": [d["observation"] for d in per_class if d["auroc"] < 0.75],
        "precision_lt_0.15": [d["observation"] for d in per_class if d["precision"] < 0.15],
        "recall_lt_0.70": [d["observation"] for d in per_class if d["recall"] < 0.70],
        "overcalling": f"{oc['predicted_positive_findings_per_image_mean']:.2f} predicted vs "
                       f"{oc['actual_positive_findings_per_image_mean']:.2f} actual positive findings per image",
        "no_finding_contradiction_rate_pct": oc["pct_images_with_no_finding_pathology_warning"],
        "mild_overfitting": f"val loss minimum at epoch {int(hist.loc[hist['val_loss'].idxmin(), 'epoch'])}, "
                            f"selected epoch {tsum['best_epoch']}",
        "single_seed": True,
    }
    metrics = {
        "baseline_id": "C1",
        "description": "DenseNet121 baseline, verified (results/classification/CLASSIFIER_VERIFICATION_REPORT.md); "
                       "official reference for all future classification experiments",
        "reference_evaluation": f"canonical re-evaluation ({CANONICAL.name}) of the SAME checkpoint; "
                                "an evaluation-consistency correction, NOT a new trained model",
        "checkpoint": {"path": rel(CKPT), "sha256": log["checkpoint_sha256_before"], "size_bytes": CKPT.stat().st_size,
                       "architecture": meta["architecture"], "num_labels": meta["num_labels"], "best_epoch": meta["epoch"],
                       "selection_metric": meta["val_metric_name"], "selection_metric_value_at_training": meta["val_metric"]},
        "label_order": list(LABELS),
        "data": {"split": "data/splits/chexpert/chexpert_image_manifest.csv.gz (stratified patient-group split, seed 42)",
                 "view_policy": "frontal (AP/PA) only", "exclusions": "6 visually confirmed blank images (5 train, 1 val)",
                 "train_images": meta["counts"]["train_images"], "train_patients": meta["counts"]["train_patients"],
                 "val_images": meta["counts"]["val_images"], "val_patients": meta["counts"]["val_patients"],
                 "locked_test": "not used for anything"},
        "label_policy": {"positive": "1 -> 1", "negative": "0 -> 0", "uncertain": "-1 -> ignored (masked, zero loss/gradient)",
                         "blank": "NaN -> 0", "config": frozen_cfg["label_policy"]},
        "preprocessing": frozen_cfg["preprocessing"], "augmentation": frozen_cfg["augmentation"],
        "loss": dict(frozen_cfg["loss"], pos_weight_raw_range=[float(pw["pos_weight_raw"].min()), float(pw["pos_weight_raw"].max())],
                     pos_weight_used_range=[float(pw["pos_weight_used"].min()), float(pw["pos_weight_used"].max())]),
        "optimizer": frozen_cfg["optimizer"], "scheduler": frozen_cfg["scheduler"],
        "training": {"batch_size": frozen_cfg["training"]["batch_size"],
                     "effective_batch_size": frozen_cfg["training"]["batch_size"] * frozen_cfg["training"]["grad_accumulation"],
                     "amp_training": frozen_cfg["training"]["amp"], "cudnn_tf32_during_training": True,
                     "max_epochs": frozen_cfg["training"]["max_epochs"], "epochs_run": tsum["epochs_run"],
                     "best_epoch": tsum["best_epoch"], "training_seconds": tsum["total_seconds"],
                     "peak_gpu_mem_gb": tsum["peak_gpu_mem_gb"], "seed": frozen_cfg["seed"],
                     "lr_by_epoch": dict(zip(hist["epoch"].astype(int).astype(str), hist["lr"]))},
        "environment": env["cuda"],
        "canonical_inference_policy": CANONICAL.as_dict(),
        "metrics": {k: sum_can[k] for k in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc", "competition5_macro_auroc")},
        "metrics_original_c1_amp_fp16": {k: sum_orig[k] for k in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc")},
        "per_class": per_class,
        "overcalling": oc,
        "original_vs_canonical": changes,
        "known_weaknesses": weak,
        "artifacts": {"canonical_evaluation_dir": rel(CANON),
                      "canonical_thresholds": rel(CANON / "thresholds/youden_j_thresholds.json"),
                      "canonical_predictions": rel(CANON / "validation_predictions.npz"),
                      "original_thresholds": rel(EXP / "thresholds/youden_j_thresholds.json"),
                      "original_predictions": rel(EXP / "validation_predictions.npz"),
                      "comparison": rel(OUTROOT / "C1_original_vs_canonical.csv")},
    }
    (OUTROOT / "C1_BASELINE_METRICS.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    code_files = ["src/classification/model.py", "src/classification/losses.py", "src/classification/dataset.py",
                  "src/classification/trainer.py", "src/classification/evaluator.py", "src/classification/thresholds.py",
                  "src/classification/findings.py", "src/classification/inference_policy.py",
                  "src/classification/serialization.py", "src/preprocessing/transforms.py", "src/data/label_policies.py",
                  "scripts/evaluate_classifier.py", "scripts/infer_classifier.py", "scripts/finalize_c1.py"]
    prov = {
        "baseline_id": "C1",
        "checkpoint": {"path": rel(CKPT), "sha256": log["checkpoint_sha256_before"],
                       "embedded_git_commit_at_training": meta["git_commit"],
                       "embedded_note": "training ran with an uncommitted (dirty) working tree; the checkpoint is NOT modified "
                                        "to add a commit. The code commit for C1 is the commit that adds this file "
                                        "(git log --format=%H -n 1 -- results/classification/C1_PROVENANCE.json)."},
        "config": {"training_config_frozen": rel(EXP / "config.yaml"), "training_config_frozen_sha256": sha256(EXP / "config.yaml"),
                   "current_config": rel(CFG_PATH), "current_config_sha256": sha256(CFG_PATH),
                   "current_config_difference": "adds the `inference` section (canonical policy) at C1 finalization"},
        "code_sha256": {f: sha256(PROJECT_ROOT / f) for f in code_files},
        "label_order": list(LABELS),
        "split": {"manifest": "data/splits/chexpert/chexpert_image_manifest.csv.gz",
                  "manifest_sha256": sha256(PROJECT_ROOT / "data/splits/chexpert/chexpert_image_manifest.csv.gz"),
                  "split_metadata_sha256": sha256(PROJECT_ROOT / "data/splits/chexpert/split_metadata.json"),
                  "exclusions_sha256": sha256(PROJECT_ROOT / "results/eda/tables/chexpert_exclusion_candidates.csv")},
        "preprocessing": frozen_cfg["preprocessing"],
        "evaluation_precision_policy": CANONICAL.as_dict(),
        "thresholds": {"canonical": rel(CANON / "thresholds/youden_j_thresholds.json"),
                       "canonical_sha256": sha256(CANON / "thresholds/youden_j_thresholds.json"),
                       "original_amp_fp16": rel(EXP / "thresholds/youden_j_thresholds.json"),
                       "original_sha256": sha256(EXP / "thresholds/youden_j_thresholds.json"),
                       "source_split": "val"},
        "predictions": {"canonical_npz_sha256": sha256(CANON / "validation_predictions.npz"),
                        "original_npz_sha256": sha256(EXP / "validation_predictions.npz")},
    }
    log["checkpoint_sha256_after"] = sha256(CKPT)
    log["checkpoint_byte_identical"] = log["checkpoint_sha256_after"] == log["checkpoint_sha256_before"]
    if not log["checkpoint_byte_identical"]:
        raise AssertionError("checkpoint changed!")
    prov["finalization_checks"] = {"checkpoint_byte_identical": True}
    (OUTROOT / "C1_PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (OUTROOT / "C1_FINALIZATION_LOG.json").write_text(json.dumps(log, indent=2, default=float), encoding="utf-8")
    write_summary_md(metrics, log)
    print(json.dumps({k: log[k] for k in ("checkpoint_byte_identical", "serialization_before_fix", "serialization_after_fix",
                                          "original_vs_canonical")}, indent=1, default=float))
    return 0


def write_summary_md(m: dict, log: dict) -> None:
    f = lambda v, n=4: f"{v:.{n}f}"  # noqa: E731
    ck, d, tr, oc, ch = m["checkpoint"], m["data"], m["training"], m["overcalling"], m["original_vs_canonical"]
    lines = [
        "# C1 baseline: DenseNet121 (official reference)",
        "",
        "_Generated by `scripts/finalize_c1.py` from the stored artifacts. Machine-readable: "
        "`C1_BASELINE_METRICS.json`, provenance: `C1_PROVENANCE.json`._",
        "",
        "**What C1 is.** The verified DenseNet121 baseline, frozen without retraining.",
        "",
        f"**Reference evaluation.** The reference numbers below are the **canonical re-evaluation** (`{m['canonical_inference_policy']['name']}`) "
        "of the same checkpoint. This is an evaluation-consistency correction, **not a new trained model**.",
        "",
        "**Historical numbers kept.** The original AMP-fp16 evaluation is preserved alongside it.",
        "",
        "## Model and data",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Architecture | torchvision DenseNet121 (ImageNet init) with head `Linear(1024, 14)`; logits only; inference applies 14 independent sigmoids |",
        f"| Checkpoint | `{ck['path']}` ({ck['size_bytes']:,} bytes) |",
        f"| Checkpoint SHA-256 | `{ck['sha256']}` |",
        f"| Best epoch / selection | {ck['best_epoch']} / validation macro AUROC |",
        f"| Split | {d['split']}; locked test split unused |",
        f"| Views | {d['view_policy']} |",
        f"| Train | {d['train_images']:,} images, {d['train_patients']:,} patients |",
        f"| Validation | {d['val_images']:,} images, {d['val_patients']:,} patients |",
        f"| Exclusions | {d['exclusions']} |",
        f"| Label policy | positive 1 → 1; negative 0 → 0; **uncertain −1 → ignored** (masked: zero loss, zero gradient); **blank → 0** |",
        f"| Preprocessing | grayscale → longest side {m['preprocessing']['image_size']} (PIL {m['preprocessing']['interpolation']}) → centre-pad (value {m['preprocessing']['pad_value']}) to {m['preprocessing']['image_size']}² → 3 channels → ImageNet mean/std |",
        f"| Augmentation | rotation ±{m['augmentation']['rotation_degrees']}°, translation ±{m['augmentation']['translate_fraction']:.0%}, scale {m['augmentation']['scale_range'][0]}–{m['augmentation']['scale_range'][1]}; no flips, colour jitter or CLAHE |",
        f"| Loss | masked BCE-with-logits; pos_weight = √(N_neg / N_pos) from the TRAIN split (raw {m['loss']['pos_weight_raw_range'][0]:.2f}–{m['loss']['pos_weight_raw_range'][1]:.2f} → used {m['loss']['pos_weight_used_range'][0]:.2f}–{m['loss']['pos_weight_used_range'][1]:.2f}) |",
        f"| Optimiser | AdamW, lr {m['optimizer']['lr']}, weight decay {m['optimizer']['weight_decay']} |",
        f"| Scheduler | ReduceLROnPlateau on val macro AUROC (factor {m['scheduler']['factor']}, patience {m['scheduler']['patience']}); LR by epoch: " + ", ".join(f"{e}: {v:.0e}" for e, v in tr["lr_by_epoch"].items()) + " |",
        f"| Batch | {tr['batch_size']} physical = {tr['effective_batch_size']} effective; AMP fp16 training (cuDNN TF32 on during training) |",
        f"| Seed | {tr['seed']} |",
        f"| Epochs / time | {tr['epochs_run']} epochs, {tr['training_seconds'] / 3600:.2f} h; peak GPU memory {tr['peak_gpu_mem_gb']:.2f} GB |",
        f"| **Canonical inference precision** | **{m['canonical_inference_policy']['name']}**: float32, no autocast, TF32 off, cuDNN deterministic |",
        "",
        "## Canonical validation metrics (official C1 reference)",
        "",
        "| Metric | Canonical (fp32_strict) | Original (AMP fp16) | Abs. diff |",
        "|---|---:|---:|---:|",
    ]
    for k in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc"):
        a, b = m["metrics"][k], m["metrics_original_c1_amp_fp16"][k]
        lines.append(f"| {k.replace('_', ' ').title().replace('Auroc', 'AUROC').replace('Auprc', 'AUPRC')} | {f(a)} | {f(b)} | {abs(a - b):.2e} |")
    lines += ["", "| Observation | Pos | Prevalence | AUROC | AUPRC | Youden thr. | Sens / recall | Spec | Precision | F1 | TP | FP | TN | FN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in m["per_class"]:
        lines.append(f"| {r['observation']} | {r['positive_support']:,} | {f(r['prevalence'], 3)} | {f(r['auroc'])} | {f(r['auprc'])} | "
                     f"{f(r['youden_threshold'])} | {f(r['sensitivity'], 3)} | {f(r['specificity'], 3)} | {f(r['precision'], 3)} | "
                     f"{f(r['f1'], 3)} | {r['tp']:,} | {r['fp']:,} | {r['tn']:,} | {r['fn']:,} |")
    lines += ["",
              "Thresholds are fitted on validation only. A label is positive when p ≥ threshold. "
              "AUPRC = average precision (`average_precision_score`).",
              "",
              "## Over-calling (baseline characteristic, deliberately NOT fixed)",
              "",
              f"- **Actual vs predicted positives.** Actual explicit positive findings average **{oc['actual_positive_findings_per_image_mean']:.2f}** per image; the model predicts **{oc['predicted_positive_findings_per_image_mean']:.2f}** positives per image (median {oc['predicted_positive_findings_per_image_median']:.0f}).",
              f"- **No image goes without a positive.** Images with zero predicted positives: {oc['images_with_zero_predicted_positives']}.",
              f"- **No Finding contradictions.** No Finding plus a pathology positive occurs in {oc['images_with_no_finding_pathology_warning']:,} images ({oc['pct_images_with_no_finding_pathology_warning']:.1f}%).",
              "",
              "| Observation | Prevalence | Predicted positive rate | Ratio |", "|---|---:|---:|---:|"]
    for r in m["per_class"]:
        lines.append(f"| {r['observation']} | {f(r['prevalence'], 3)} | {f(r['predicted_positive_rate'], 3)} | {r['predicted_positive_rate'] / r['prevalence']:.1f}× |")
    w = m["known_weaknesses"]
    lines += ["", "## Known weaknesses", "",
              f"- **AUROC below 0.75:** {', '.join(w['low_auroc_lt_0.75'])}.",
              f"- **Thresholded precision below 0.15:** {', '.join(w['precision_lt_0.15'])}.",
              f"- **Recall below 0.70:** {', '.join(w['recall_lt_0.70'])}.",
              f"- **Over-calling at the Youden operating point:** {w['overcalling']}.",
              f"- **Mild overfitting:** {w['mild_overfitting']}. Single seed, so no confidence intervals yet.",
              "",
              "## Original vs canonical evaluation (same checkpoint)", "",
              f"- **Probabilities:** {ch['probabilities_changed']:,} of {ch['probabilities_total']:,} changed (max abs diff {ch['probability_max_abs_diff']:.2e}, median {ch['probability_median_abs_diff']:.2e}).",
              f"- **Thresholds:** {ch['thresholds_changed']} of 14 changed (max abs diff {ch['threshold_max_abs_diff']:.2e}).",
              f"- **Statuses:** {ch['statuses_changed_all_entries']:,} changed ({ch['statuses_changed_valid_entries']:,} on valid entries).",
              f"- **AUROC/AUPRC:** largest absolute change {ch['max_abs_diff_auroc_auprc_any']:.2e}. This is numerical precision, **not** a model change.",
              "- **Full comparison:** `C1_original_vs_canonical.csv`.",
              "",
              "## Serialization fix", "",
              f"- **Before:** reading the original CSV with pandas' default (not correctly rounded) float parser produced "
              f"{log['serialization_before_fix']['status_mismatches']} status mismatches ({log['serialization_before_fix']['status_mismatches_valid_entries_only']} on valid entries) against the stored thresholds. "
              "The values written were in fact exact; the loss occurred on reading. This corrects the mechanism stated in the verification report.",
              f"- **After:** a lossless `.npz` (canonical), plus a 17-digit CSV read with `float_precision=\"round_trip\"`. Max abs diff "
              f"{log['serialization_after_fix']['csv_17_digits']['max_abs_diff']:.1e}; **{log['serialization_after_fix']['csv_17_digits']['status_mismatches']}** mismatches.",
              ""]
    (OUTROOT / "C1_BASELINE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
