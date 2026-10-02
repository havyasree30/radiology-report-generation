"""C2 loss study: build all comparison artifacts from the four completed arms.

    .venv\\Scripts\\python.exe -m scripts.compare_losses

Reads only stored artifacts (canonical fp32_strict evaluation of each arm's
macro-AUPRC-selected checkpoint). Threshold-based numbers are NOT used for the
comparison. C1 appears as an external historical reference column only.

Outputs: results/classification/experiments/LOSS_COMPARISON.{csv,md},
         per-class / rare / common / bootstrap tables, figures/, <arm>/ANALYSIS.md
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.plotting import INK_2, SERIES, apply_style, save_figure
from src.classification.labels import LABELS
from src.classification.serialization import load_predictions_npz
from src.utils.config import PROJECT_ROOT

EXPD = PROJECT_ROOT / "results/classification/experiments"
ARMS = {"c2a_bce": "C2-A BCE", "c2b_sqrt_weighted_bce": "C2-B sqrt-weighted BCE",
        "c2c_focal_gamma2": "C2-C Focal (gamma=2)", "c2d_asl": "C2-D ASL"}
SHORT = {"c2a_bce": "BCE", "c2b_sqrt_weighted_bce": "WBCE", "c2c_focal_gamma2": "Focal", "c2d_asl": "ASL"}
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]
COMMON = ["Support Devices", "Lung Opacity", "Pleural Effusion", "Edema", "Atelectasis"]
N_BOOT, SEED = 1000, 42


def load_arm(a: str) -> dict:
    d = EXPD / a
    ev = d / "evaluation" / "fp32_strict"
    return {"pc": pd.read_csv(ev / "per_class_metrics.csv", float_precision="round_trip").set_index("observation").loc[list(LABELS)],
            "sm": json.loads((ev / "summary_metrics.json").read_text(encoding="utf-8")),
            "hist": pd.read_csv(d / "training_history.csv"),
            "ts": json.loads((d / "training_summary.json").read_text(encoding="utf-8")),
            "sha": json.loads((d / "checkpoint_sha256.json").read_text(encoding="utf-8"))["best.pt"]["sha256"],
            "pw": pd.read_csv(d / "train_pos_weights.csv"),
            "loss_cfg": yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["loss"],
            "pred": load_predictions_npz(ev / "validation_predictions.npz")}


def macro_metrics(y, p, v, idx):
    au, ap = [], []
    for j in range(len(LABELS)):
        sel = idx[v[idx, j]]
        yj = y[sel, j]
        if 0 < yj.sum() < len(yj):
            au.append(roc_auc_score(yj, p[sel, j]))
            ap.append(average_precision_score(yj, p[sel, j]))
    return np.mean(au), np.mean(ap)


def paired_bootstrap(arms: dict, patients: np.ndarray, y, v) -> pd.DataFrame:
    """Resample PATIENTS (all their images) with replacement; same resample for every arm."""
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(patients, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    res = {a: [] for a in arms}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[k] for k in pick])
        for a, d in arms.items():
            res[a].append(macro_metrics(y, d["pred"]["probs"], v, idx))
    out = []
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = np.array(res[a]), np.array(res[b])
            for k, metric in enumerate(("macro_auroc", "macro_auprc")):
                diff = rb[:, k] - ra[:, k]
                out.append({"comparison": f"{SHORT[b]} - {SHORT[a]}", "metric": metric,
                            "point_estimate": arms[b]["sm"][metric] - arms[a]["sm"][metric],
                            "ci95_low": float(np.percentile(diff, 2.5)), "ci95_high": float(np.percentile(diff, 97.5)),
                            "share_of_resamples_positive": float((diff > 0).mean())})
    return pd.DataFrame(out)


def main() -> int:
    apply_style()
    (EXPD / "figures").mkdir(parents=True, exist_ok=True)
    arms = {a: load_arm(a) for a in ARMS}
    c1 = json.loads((EXPD / "C1_REFERENCE.json").read_text(encoding="utf-8"))
    ref_paths = arms["c2a_bce"]["pred"]["paths"]
    for a, d in arms.items():
        if d["pred"]["paths"] != ref_paths:
            raise AssertionError(f"{a}: validation rows differ from c2a_bce")
        if not np.array_equal(d["pred"]["valid"], arms["c2a_bce"]["pred"]["valid"]):
            raise AssertionError(f"{a}: validity mask differs")
    y = arms["c2a_bce"]["pred"]["raw_labels"] == 1
    v = arms["c2a_bce"]["pred"]["valid"]

    # ---------------- summary ----------------
    summ = []
    for a, d in arms.items():
        h, ts = d["hist"], d["ts"]
        summ.append({"arm": a, "loss": ARMS[a], "macro_auroc": d["sm"]["macro_auroc"], "micro_auroc": d["sm"]["micro_auroc"],
                     "macro_auprc": d["sm"]["macro_auprc"], "micro_auprc": d["sm"]["micro_auprc"],
                     "best_epoch": ts["best_epoch"], "epochs_run": ts["epochs_run"], "stopped_early": ts["stopped_early"],
                     "peak_vram_gb": ts["peak_gpu_mem_gb"], "training_hours": ts["total_seconds"] / 3600,
                     "mean_epoch_train_minutes": h["train_seconds"].mean() / 60, "mean_epoch_val_minutes": h["val_seconds"].mean() / 60,
                     "grad_norm_max": h["grad_norm_max"].max(), "amp_skipped_steps": int(h["amp_skipped_or_nonfinite_steps"].sum()),
                     "val_loss_min_epoch": int(h.loc[h["val_loss"].idxmin(), "epoch"]),
                     "checkpoint_sha256": d["sha"]})
    summ.append({"arm": "C1 (external reference)", "loss": "sqrt-weighted BCE, AUROC-selected, TF32 on",
                 **{k: c1["overall"][k] for k in ("macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc")},
                 "best_epoch": 5, "checkpoint_sha256": c1["checkpoint_sha256"]})
    summary = pd.DataFrame(summ)
    summary.to_csv(EXPD / "LOSS_COMPARISON.csv", index=False, float_format="%.17g")

    # ---------------- per class ----------------
    rows = []
    for lab in LABELS:
        r = {"observation": lab, "positive_support": int(arms["c2a_bce"]["pc"].loc[lab, "n_positive"]),
             "prevalence": arms["c2a_bce"]["pc"].loc[lab, "prevalence"]}
        for metric in ("auroc", "auprc"):
            for a in ARMS:
                r[f"{SHORT[a]}_{metric}"] = arms[a]["pc"].loc[lab, metric]
            r[f"C1ref_{metric}"] = c1["per_class"][lab][metric]
        rows.append(r)
    pc = pd.DataFrame(rows)
    pc.to_csv(EXPD / "LOSS_COMPARISON_per_class.csv", index=False, float_format="%.17g")

    def nlift(ap, prev):
        return (ap - prev) / (1 - prev)

    rare = []
    for lab in RARE:
        r = pc.set_index("observation").loc[lab]
        for a in ARMS:
            s = SHORT[a]
            rare.append({"observation": lab, "loss": s, "prevalence": r["prevalence"], "auroc": r[f"{s}_auroc"],
                         "auprc": r[f"{s}_auprc"], "auprc_over_prevalence": r[f"{s}_auprc"] / r["prevalence"],
                         "normalized_lift": nlift(r[f"{s}_auprc"], r["prevalence"]),
                         "delta_auprc_vs_bce": r[f"{s}_auprc"] - r["BCE_auprc"]})
    rare = pd.DataFrame(rare)
    rare.to_csv(EXPD / "LOSS_COMPARISON_rare_classes.csv", index=False, float_format="%.17g")
    common = []
    for lab in COMMON:
        r = pc.set_index("observation").loc[lab]
        for a in ARMS:
            s = SHORT[a]
            common.append({"observation": lab, "loss": s, "auroc": r[f"{s}_auroc"], "auprc": r[f"{s}_auprc"],
                           "delta_auroc_vs_bce": r[f"{s}_auroc"] - r["BCE_auroc"],
                           "delta_auprc_vs_bce": r[f"{s}_auprc"] - r["BCE_auprc"]})
    common = pd.DataFrame(common)
    common.to_csv(EXPD / "LOSS_COMPARISON_common_classes.csv", index=False, float_format="%.17g")

    patients = np.asarray(arms["c2a_bce"]["pred"]["patient_ids"])
    boot = paired_bootstrap(arms, patients, y, v)
    boot.to_csv(EXPD / "LOSS_COMPARISON_bootstrap.csv", index=False, float_format="%.17g")

    weights = arms["c2b_sqrt_weighted_bce"]["pw"][["observation", "n_positive", "n_negative", "pos_weight_raw", "pos_weight_used"]]
    weights.to_csv(EXPD / "c2b_sqrt_weighted_bce" / "pos_weights_used.csv", index=False, float_format="%.17g")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (a, d) in enumerate(arms.items()):
        h = d["hist"]
        for ax, col in zip(axes, ("val_macro_auprc", "val_macro_auroc")):
            ax.plot(h["epoch"], h[col], marker="o", lw=2, color=SERIES[i], label=SHORT[a])
            be = d["ts"]["best_epoch"]
            ax.scatter([be], [h.loc[h["epoch"] == be, col]], s=90, facecolor="none", edgecolor=SERIES[i], lw=2)
    for ax, t in zip(axes, ("Validation macro AUPRC (selection metric)", "Validation macro AUROC")):
        ax.set_title(t, pad=10)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle("C2 loss study, one seed; ring = selected epoch", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, "learning_curves_metrics", EXPD / "figures")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (i, (a, d)) in zip(axes, enumerate(arms.items())):
        h = d["hist"]
        ax.plot(h["epoch"], h["train_loss"], marker="o", color=SERIES[0], label="train")
        ax.plot(h["epoch"], h["val_loss"], marker="o", color=SERIES[1], label="val")
        ax.axvline(d["ts"]["best_epoch"], color=INK_2, lw=0.8)
        ax.set_title(SHORT[a], loc="left")
        ax.set_xlabel("Epoch")
    axes[0].legend()
    fig.suptitle("Train vs validation loss per arm (loss scales differ between loss functions; compare shapes only)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, "learning_curves_loss", EXPD / "figures")

    for metric in ("auprc", "auroc"):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        yy = np.arange(len(LABELS))
        for i, a in enumerate(ARMS):
            ax.barh(yy + (i - 1.5) * 0.2, pc[f"{SHORT[a]}_{metric}"], height=0.2, color=SERIES[i], label=SHORT[a])
        if metric == "auprc":
            ax.scatter(pc["prevalence"], yy, marker="|", s=200, color="#0b0b0b", label="prevalence (chance)")
        ax.set_yticks(yy, LABELS)
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper())
        ax.set_xlim(0.5 if metric == "auroc" else 0, 1)
        ax.set_title(f"Per-class validation {metric.upper()} by loss (all 14 classes)", pad=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", visible=False)
        save_figure(fig, f"per_class_{metric}", EXPD / "figures")

    # ---------------- per-arm analysis ----------------
    for a, d in arms.items():
        h, ts = d["hist"], d["ts"]
        lines = [f"# {ARMS[a]}: run analysis", "",
                 f"_Generated by `scripts/compare_losses.py`. Loss config (from this run's frozen `config.yaml`): `{json.dumps(d['loss_cfg'])}`_",
                 "",
                 f"- **Checkpoint** `checkpoints/best.pt`: SHA-256 `{d['sha']}`, selected epoch {ts['best_epoch']} by validation macro AUPRC.",
                 f"- **Run length:** {ts['epochs_run']} epochs; stopped early: {ts['stopped_early']}; {ts['total_seconds'] / 3600:.2f} h; peak VRAM {ts['peak_gpu_mem_gb']:.2f} GB.",
                 f"- **Canonical (fp32_strict) validation:** macro AUROC {d['sm']['macro_auroc']:.4f}, micro AUROC {d['sm']['micro_auroc']:.4f}, "
                 f"macro AUPRC {d['sm']['macro_auprc']:.4f}, micro AUPRC {d['sm']['micro_auprc']:.4f}.",
                 f"- **Stability:** max gradient norm {h['grad_norm_max'].max():.2f}; AMP-skipped steps {int(h['amp_skipped_or_nonfinite_steps'].sum())}; "
                 f"non-finite losses: none (training aborts on any).",
                 f"- **Overfitting signal:** validation-loss minimum at epoch {int(h.loc[h['val_loss'].idxmin(), 'epoch'])}; "
                 f"train loss {h['train_loss'].iloc[0]:.4f} → {h['train_loss'].iloc[-1]:.4f}.", "",
                 "| Epoch | Train loss | Val loss | Macro AUROC | Macro AUPRC | LR | Train min | Val min |",
                 "|---:|---:|---:|---:|---:|---|---:|---:|"]
        for r in h.itertuples():
            lines.append(f"| {r.epoch} | {r.train_loss:.4f} | {r.val_loss:.4f} | {r.val_macro_auroc:.4f} | {r.val_macro_auprc:.4f} | "
                         f"{r.lr:.0e} | {r.train_seconds / 60:.1f} | {r.val_seconds / 60:.1f} |")
        lines += ["", "Per-epoch validation uses the canonical policy; the table's last-epoch values are not the reported metrics "
                  "unless that epoch was selected. Reported metrics come from `evaluation/fp32_strict/`.", ""]
        (EXPD / a / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.drop(columns=["checkpoint_sha256"]).to_string(index=False))
    print(boot.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
