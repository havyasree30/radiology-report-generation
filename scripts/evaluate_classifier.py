"""Validation evaluation of the selected checkpoint under ONE inference policy.

    .venv\\Scripts\\python.exe -m scripts.evaluate_classifier --config configs/classifier/densenet121.yaml
    (optional) --policy fp32_strict | amp_fp16_legacy_c1     default: config inference.policy

Outputs go to <output_dir>/evaluation/<policy>/ and never overwrite other policies'
results. Inference is run twice to prove determinism. Thresholds are fitted on
the validation split only; the locked test split is never loaded here.
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from src.analysis.plotting import INK_2, MUTED, SERIES, apply_style, save_figure, subtitle
from src.classification import inference_policy
from src.classification.checkpointing import load_checkpoint
from src.classification.dataset import CheXpertDataset, label_policy_from_config, load_split_frame
from src.classification.evaluator import predict
from src.classification.labels import LABELS, NO_FINDING, PATHOLOGY_LABELS
from src.classification.metrics import per_class_metrics, summary_metrics
from src.classification.serialization import load_predictions_csv, load_predictions_npz, roundtrip_report, save_predictions
from src.classification.thresholds import save_thresholds, youden_thresholds
from src.classification.trainer import load_config
from src.preprocessing.transforms import EvalTransform, PreprocessConfig
from src.utils.audit import setup_logging
from src.utils.config import PROJECT_ROOT, load_paths

log = logging.getLogger("evaluate")


def thresholded_table(y, p, m, thr: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(LABELS):
        v = m[:, j]
        yj, pred = y[v, j] == 1, p[v, j] >= thr[j]
        tp, fp = int((pred & yj).sum()), int((pred & ~yj).sum())
        fn, tn = int((~pred & yj).sum()), int((~pred & ~yj).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        rows.append({"observation": lab, "threshold": thr[j], "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "sensitivity": sens, "recall": sens, "specificity": tn / (tn + fp) if tn + fp else np.nan,
                     "precision": prec, "f1": 2 * prec * sens / (prec + sens) if (prec + sens) else np.nan,
                     "prevalence": yj.mean(), "predicted_positive_rate": pred.mean()})
    return pd.DataFrame(rows)


def overcalling(raw: np.ndarray, p: np.ndarray, thr: np.ndarray) -> dict:
    pos, y = p >= thr, raw == 1
    nf = LABELS.index(NO_FINDING)
    patho = [LABELS.index(l) for l in PATHOLOGY_LABELS]
    conflict = pos[:, nf] & pos[:, patho].any(axis=1)
    return {"n_images": int(len(p)),
            "actual_positive_findings_per_image_mean": float(y.sum(1).mean()),
            "predicted_positive_findings_per_image_mean": float(pos.sum(1).mean()),
            "predicted_positive_findings_per_image_median": float(np.median(pos.sum(1))),
            "images_with_zero_predicted_positives": int((pos.sum(1) == 0).sum()),
            "images_with_no_finding_pathology_warning": int(conflict.sum()),
            "pct_images_with_no_finding_pathology_warning": float(100 * conflict.mean()),
            "note": "actual = explicit positive (1) labels; uncertain and blank are not counted as positive"}


def grid_plot(curves, fname, title, xlabel, ylabel, out_dir, diag=False, baselines=None):
    fig, axes = plt.subplots(4, 4, figsize=(14, 13), sharex=True, sharey=True)
    for ax, lab in zip(axes.flat, LABELS):
        if lab in curves:
            x, y, txt = curves[lab]
            ax.plot(x, y, color=SERIES[0], lw=1.6)
            ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)
        if diag:
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
        if baselines and lab in baselines:
            ax.axhline(baselines[lab], color=MUTED, lw=0.8)
        ax.set_title(lab, fontsize=9, loc="left")
    for ax in axes.flat[len(LABELS):]:
        ax.axis("off")
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, fname, out_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classifier/densenet121.yaml")
    ap.add_argument("--policy", default=None)
    args = ap.parse_args()
    setup_logging()
    apply_style()
    cfg = load_config(PROJECT_ROOT / args.config)
    policy = inference_policy.get(args.policy or cfg["inference"]["policy"])
    exp = PROJECT_ROOT / cfg["output_dir"]
    out = exp / "evaluation" / policy.name
    fig_dir = out / "figures"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    model, meta = load_checkpoint(exp / "checkpoints" / "best.pt", device)
    split = cfg["thresholds"]["source_split"]
    paths = load_paths()
    frame = load_split_frame(paths, cfg["data"], split)
    ds = CheXpertDataset(frame, paths.chexpert_image_base, label_policy_from_config(cfg["label_policy"]),
                         EvalTransform(PreprocessConfig.from_dict(cfg["preprocessing"])))

    def run():
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
        return predict(model, loader, device, policy=policy)

    pred, pred2 = run(), run()
    flags = inference_policy.current_flags()
    y, p, m = pred["targets"], pred["probs"], pred["mask"]
    determinism = {"max_abs_diff_run1_vs_run2": float(np.abs(p - pred2["probs"]).max()),
                   "bit_identical": bool(np.array_equal(p, pred2["probs"]))}
    raw = frame[list(LABELS)].to_numpy(dtype=np.float64)

    pc = per_class_metrics(y, p, m)
    sm = summary_metrics(y, p, m)
    thr_entries = youden_thresholds(y, p, m, source_split=split)
    thr = np.array([e["threshold"] for e in thr_entries], dtype=np.float64)
    tt = thresholded_table(y, p, m, thr)
    pc = pc.merge(tt.drop(columns=["prevalence"]), on="observation")
    out.mkdir(parents=True, exist_ok=True)
    pc.to_csv(out / "per_class_metrics.csv", index=False, float_format="%.17g")

    files = save_predictions(out, "validation_predictions", frame["Path"], frame["patient_id"], p, raw, m,
                             meta={"policy": policy.name, "checkpoint_epoch": meta["epoch"]})
    rt = {"npz": roundtrip_report(p, load_predictions_npz(files["npz"])["probs"], thr),
          "csv_17_digits": roundtrip_report(p, load_predictions_csv(files["csv"])["probs"], thr)}
    if rt["npz"]["status_mismatches"] or rt["csv_17_digits"]["status_mismatches"]:
        raise AssertionError(f"serialization changed threshold statuses: {rt}")

    prov = {"source_split": split, "checkpoint": str((exp / "checkpoints" / "best.pt").relative_to(PROJECT_ROOT)),
            "checkpoint_epoch": meta["epoch"], "git_commit_at_training": meta["git_commit"],
            "n_images": int(len(frame)), "label_policy": cfg["label_policy"], "inference_policy": policy.as_dict(),
            "backend_flags": flags, "rule": "positive if probability >= threshold; threshold = argmax(TPR - FPR) on validation ROC"}
    save_thresholds(thr_entries, out / "thresholds" / "youden_j_thresholds.json", prov)
    oc = overcalling(raw, p, thr)
    summary = {"split": split, "n_images": int(len(frame)), "n_patients": int(frame["patient_id"].nunique()),
               "checkpoint_epoch": meta["epoch"], "inference_policy": policy.name, "backend_flags": flags, **sm,
               "competition5_macro_auroc": float(pc.set_index("observation").loc[
                   ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"], "auroc"].mean()),
               "determinism": determinism, "serialization_roundtrip": rt, "overcalling": oc}
    (out / "summary_metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log.info("summary: %s", {k: summary[k] for k in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc")})

    # ---- figures ----
    for metric, fname in (("auroc", "per_class_auroc"), ("auprc", "per_class_auprc")):
        d = pc.sort_values(metric)
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.barh(d["observation"], d[metric], color=SERIES[0], height=0.6)
        if metric == "auprc":
            ax.scatter(d["prevalence"], d["observation"], color=SERIES[1], zorder=3, s=30,
                       label="Prevalence (chance-level AUPRC)")
            ax.legend(loc="lower right")
        for yy, v in enumerate(d[metric]):
            ax.text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0.5 if metric == "auroc" else 0, 1.0)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"Validation {metric.upper()} per observation", pad=22)
        subtitle(ax, f"{len(frame):,} frontal validation images; policy {policy.name}; epoch {meta['epoch']}")
        ax.grid(axis="y", visible=False)
        save_figure(fig, fname, fig_dir)
    roc, pr, prev = {}, {}, {}
    for j, lab in enumerate(LABELS):
        mm = m[:, j]
        if pc.loc[j, "undefined_reason"]:
            continue
        fpr, tpr, _ = roc_curve(y[mm, j], p[mm, j])
        roc[lab] = (fpr, tpr, f"AUROC {pc.loc[j, 'auroc']:.3f}")
        prec, rec, _ = precision_recall_curve(y[mm, j], p[mm, j])
        pr[lab] = (rec, prec, f"AUPRC {pc.loc[j, 'auprc']:.3f}")
        prev[lab] = pc.loc[j, "prevalence"]
    grid_plot(roc, "roc_curves", f"Validation ROC curves ({policy.name})", "False positive rate", "True positive rate",
              fig_dir, diag=True)
    grid_plot(pr, "pr_curves", f"Validation precision-recall curves ({policy.name}; line = prevalence)", "Recall",
              "Precision", fig_dir, baselines=prev)
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    bins = np.linspace(0, 1, 41)
    for ax, (j, lab) in zip(axes.flat, enumerate(LABELS)):
        mm = m[:, j]
        ax.hist(p[mm & (y[:, j] == 0), j], bins=bins, color=SERIES[0], alpha=0.6, density=True, label="negative")
        ax.hist(p[mm & (y[:, j] == 1), j], bins=bins, color=SERIES[1], alpha=0.6, density=True, label="positive")
        ax.axvline(thr[j], color="#0b0b0b", lw=1)
        ax.set_title(lab, fontsize=9, loc="left")
        ax.set_yticks([])
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[len(LABELS):]:
        ax.axis("off")
    fig.suptitle(f"Validation probability distributions by true label ({policy.name}; line = Youden threshold)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, "probability_distributions", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
