"""C6 journal assets: 7 figures (PNG 300 dpi + vector PDF + SVG, with source data) and 5 tables (CSV, Markdown, LaTeX).

Reads only the artifacts written by scripts.c6_03_evaluate. No titles inside plots (captions are separate).

    .venv\\Scripts\\python.exe -m scripts.c6_04_figures_tables
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import INK_2, SERIES, apply_style
from src.classification.labels import LABELS
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/classification/experiments/c6_final_test"
FIG, SRC, TAB = OUT / "figures", OUT / "figures/source_data", OUT / "tables"
C_TEST, C_VAL, C_THIRD = SERIES[0], SERIES[1], SERIES[2]
rd = lambda n: pd.read_csv(OUT / n, float_precision="round_trip")  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
f4 = lambda v: f"{v:.4f}"  # noqa: E731


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")  # noqa: E731
    body = "\n".join(" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False))
    return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n\\hline\n"
            + " & ".join(esc(c) for c in df.columns) + " \\\\\n\\hline\n" + body + "\n\\hline\n\\end{tabular}\n"
            + f"\\caption{{{esc(caption)}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def write_table(name: str, df: pd.DataFrame, caption: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (TAB / f"{name}.md").write_text(md_table(df) + "\n", encoding="utf-8")
    (TAB / f"{name}.tex").write_text(tex_table(df, caption, f"tab:{name}"), encoding="utf-8")


def src(name: str, df: pd.DataFrame) -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRC / f"{name}.csv", index=False, float_format="%.17g")


def main() -> int:
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    res = json.loads((OUT / "c6_results.json").read_text(encoding="utf-8"))
    cmp_ = rd("validation_vs_test_summary.csv")
    pc = rd("validation_vs_test_per_class.csv")
    rk = rd("test_ranking_metrics.csv").set_index("observation")
    bt = rd("test_binary_metrics_per_class.csv").set_index("observation")
    cal = rd("test_calibration_per_class.csv")
    boot = rd("test_bootstrap_summary.csv").set_index("metric")
    rare = rd("test_rare_class_analysis.csv")
    dist = rd("test_finding_count_distribution.csv")
    bins = rd("test_reliability_bins.csv")
    sm, b14, b13, bmicro = res["ranking"], res["binary_macro14"], res["binary_macro13"], res["binary_micro"]
    cs = res["calibration"]
    ci = lambda k: f"{f3(boot.loc[k, 'ci95_low'])} to {f3(boot.loc[k, 'ci95_high'])}" if k in boot.index else "n/a"  # noqa: E731
    ci4 = lambda k: f"{f4(boot.loc[k, 'ci95_low'])} to {f4(boot.loc[k, 'ci95_high'])}"  # noqa: E731

    # ================================================================== tables
    t1 = pd.DataFrame([
        ("Macro AUROC (14 classes)", f3(sm["macro_auroc"]), ci("macro_auroc")),
        ("Micro AUROC", f3(sm["micro_auroc"]), "n/a"),
        ("Macro AUPRC (14 classes)", f3(sm["macro_auprc"]), ci("macro_auprc")),
        ("Micro AUPRC", f3(sm["micro_auprc"]), "n/a"),
        ("Macro precision", f3(b14["macro_precision"]), ci("macro_precision")),
        ("Macro recall", f3(b14["macro_recall"]), ci("macro_recall")),
        ("Macro specificity", f3(b14["macro_specificity"]), ci("macro_specificity")),
        ("Macro F1", f3(b14["macro_f1"]), ci("macro_f1")),
        ("Macro balanced accuracy", f3(b14["macro_balanced_accuracy"]), ci("macro_balanced_accuracy")),
        ("Micro precision / recall / F1 (pooled)", f"{f3(bmicro['micro_precision'])} / {f3(bmicro['micro_recall'])} / {f3(bmicro['micro_f1'])}", "n/a"),
        ("Macro F1 excluding No Finding (13 classes)", f3(b13["macro_f1"]), "n/a"),
        ("Macro Brier score (calibrated)", f4(cs["macro_calibrated"]["brier"]), ci4("macro_brier")),
        ("Macro log loss (calibrated)", f4(cs["macro_calibrated"]["log_loss"]), "n/a"),
        ("Macro ECE (calibrated)", f4(cs["macro_calibrated"]["ece"]), "n/a")], columns=["Metric", "Locked test", "95% CI (patient bootstrap)"])
    write_table("table1_final_overall_performance", t1, "Final locked-test performance of the frozen classifier.")
    t2 = pd.DataFrame({"Observation": list(LABELS), "Prevalence": [f3(rk.loc[l, "prevalence"]) for l in LABELS],
                       "Positives": [int(rk.loc[l, "positive_count"]) for l in LABELS],
                       "AUROC": [f3(rk.loc[l, "auroc"]) for l in LABELS], "AUPRC": [f3(rk.loc[l, "auprc"]) for l in LABELS],
                       "Threshold": [f3(bt.loc[l, "threshold"]) for l in LABELS], "Precision": [f3(bt.loc[l, "precision"]) for l in LABELS],
                       "Recall": [f3(bt.loc[l, "recall"]) for l in LABELS], "Specificity": [f3(bt.loc[l, "specificity"]) for l in LABELS],
                       "F1": [f3(bt.loc[l, "f1"]) for l in LABELS], "Bal. acc.": [f3(bt.loc[l, "balanced_accuracy"]) for l in LABELS],
                       "TP": [int(bt.loc[l, "tp"]) for l in LABELS], "FP": [int(bt.loc[l, "fp"]) for l in LABELS],
                       "TN": [int(bt.loc[l, "tn"]) for l in LABELS], "FN": [int(bt.loc[l, "fn"]) for l in LABELS]})
    write_table("table2_per_class_final_performance", t2, "Per-class locked-test performance under the frozen operating policy.")
    t3 = pd.DataFrame({"Metric": cmp_["metric"].str.replace("_", " "), "Validation": [f4(v) for v in cmp_["validation"]],
                       "Locked test": [f4(v) for v in cmp_["test"]], "Test - validation": [f"{v:+.4f}" for v in cmp_["difference_test_minus_validation"]]})
    write_table("table3_validation_vs_test", t3, "Validation versus locked-test results of the frozen classifier.")
    t4 = pd.DataFrame({"Observation": rare.observation, "Split": rare.split, "Prevalence": rare.prevalence.map(f3), "AUROC": rare.auroc.map(f3),
                       "AUPRC": rare.auprc.map(f3), "Precision": rare.precision.map(f3), "Recall": rare.recall.map(f3),
                       "Specificity": rare.specificity.map(f3), "F1": rare.f1.map(f3), "TP": rare.tp, "FP": rare.fp,
                       "FP per TP": rare.fp_per_tp.map(lambda v: f"{v:.2f}")})
    write_table("table4_rare_class_results", t4, "Rare-class results on validation and locked test.")
    t5 = pd.DataFrame({"Observation": cal.observation, "Prevalence": cal.prevalence.map(f3), "Mean calibrated probability": cal.mean_calibrated_probability.map(f3),
                       "Brier": cal.brier.map(f4), "Log loss": cal.log_loss.map(f4), "ECE": cal.ece.map(f4),
                       "Raw-score Brier": cal.raw_brier.map(f4), "Raw-score ECE": cal.raw_ece.map(f4)})
    t5.loc[len(t5)] = ["Macro mean", "", "", f4(cal.brier.mean()), f4(cal.log_loss.mean()), f4(cal.ece.mean()), f4(cal.raw_brier.mean()), f4(cal.raw_ece.mean())]
    write_table("table5_locked_test_calibration", t5, "Locked-test calibration of the frozen Platt calibrators.")

    # ================================================================== Figure 1: validation vs test
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    groups = [("Ranking", ["macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc"]),
              ("Binary decisions (macro)", ["macro_precision", "macro_recall", "macro_specificity", "macro_f1", "macro_balanced_accuracy"]),
              ("Calibration (macro)", ["macro_brier", "macro_log_loss", "macro_ece"])]
    cm = cmp_.set_index("metric")
    f1src = []
    for ax, (nm, ms) in zip(axes, groups):
        y = np.arange(len(ms))
        ax.scatter(cm.loc[ms, "validation"], y - 0.12, s=42, marker="o", facecolor="white", edgecolor=C_VAL, linewidth=1.5, label="Validation", zorder=3)
        ax.scatter(cm.loc[ms, "test"], y + 0.12, s=42, marker="D", color=C_TEST, label="Locked test", zorder=3)
        ax.set_yticks(y, [m.replace("_", " ").replace("balanced accuracy", "bal. accuracy") for m in ms])
        ax.invert_yaxis()
        ax.set_xlabel(nm)
        ax.grid(axis="y", visible=False)
        lo, hi = float(cm.loc[ms, ["validation", "test"]].min().min()), float(cm.loc[ms, ["validation", "test"]].max().max())
        pad = (hi - lo) * 0.15 + 1e-4
        ax.set_xlim(lo - pad, hi + pad)
        f1src.append(cm.loc[ms, ["validation", "test"]].reset_index().assign(panel=nm))
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    src("fig1_validation_vs_test", pd.concat(f1src))
    save(fig, "fig1_validation_vs_test")

    # ================================================================== Figure 2: per-class AUROC and AUPRC
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    y = np.arange(14)
    for ax, m, lab in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        ax.scatter(pc[f"validation_{m}"], y - 0.12, s=36, facecolor="white", edgecolor=C_VAL, linewidth=1.4, label="Validation", zorder=3)
        ax.scatter(pc[f"test_{m}"], y + 0.12, s=36, marker="D", color=C_TEST, label="Locked test", zorder=3)
        if m == "auroc":
            ax.axvline(0.5, color="#898781", lw=0.9, ls="--")
            ax.text(0.505, -0.45, "chance", fontsize=7, color=INK_2, va="center")
        else:
            ax.scatter(pc["test_prevalence"], y, s=26, marker="|", color="#898781", linewidth=1.6, label="Chance level (test prevalence)", zorder=2)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, pc.observation)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.45, 0.95)
    axes[1].set_xlim(0, 0.95)
    axes[1].legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig2_per_class_auroc_auprc", pc[["observation", "validation_auroc", "test_auroc", "validation_auprc", "test_auprc", "test_prevalence"]])
    save(fig, "fig2_per_class_auroc_auprc")

    # ================================================================== Figure 3: precision / recall / F1 per class
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    h = 0.26
    for k, (m, colr, nm) in enumerate((("precision", C_TEST, "Precision"), ("recall", C_VAL, "Recall"), ("f1", C_THIRD, "F1"))):
        ax.barh(y + (k - 1) * h, pc[f"test_{m}"], height=h, color=colr, label=nm)
    ax.set_yticks(y, pc.observation)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Locked-test value at the frozen operating thresholds")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    src("fig3_precision_recall_f1", pc[["observation", "test_precision", "test_recall", "test_f1"]])
    save(fig, "fig3_precision_recall_f1")

    # ================================================================== Figure 4: rare classes
    classes = list(rare.observation.drop_duplicates())
    yr = np.arange(len(classes))
    v, t = rare[rare.split == "validation"].set_index("observation"), rare[rare.split == "test"].set_index("observation")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, ms, xl in ((axes[0], [("auprc", "AUPRC")], "AUPRC (tick = prevalence)"), (axes[1], [("precision", "Precision"), ("recall", "Recall")], "Precision (circle) and recall (triangle)"),
                       (axes[2], [("fp_per_tp", "FP per TP")], "False positives per true positive")):
        for m, _ in ms:
            mk = {"auprc": "o", "fp_per_tp": "o", "precision": "o", "recall": "^"}[m]
            ax.scatter(v.loc[classes, m], yr - 0.13, s=40, marker=mk, facecolor="white", edgecolor=C_VAL, linewidth=1.4, zorder=3,
                       label="Validation" if m == ms[0][0] else None)
            ax.scatter(t.loc[classes, m], yr + 0.13, s=40, marker=mk, color=C_TEST, zorder=3, label="Locked test" if m == ms[0][0] else None)
        if ms[0][0] == "auprc":
            ax.scatter(t.loc[classes, "prevalence"], yr, marker="|", s=60, color="#898781", linewidth=1.6, zorder=2)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yr, classes)
    axes[0].invert_yaxis()
    axes[2].legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig4_rare_class_performance", rare)
    save(fig, "fig4_rare_class_performance")

    # ================================================================== Figure 5: test reliability
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    prev = cal.set_index("observation").prevalence
    sdata = []
    for ax, (role, lab) in zip(axes, res["representative_classes"].items()):
        d = bins[bins.observation == lab]
        hi = float(max(d.mean_predicted.max(), d.observed_fraction.max())) * 1.08
        ax.plot([0, hi], [0, hi], color="#898781", lw=1.0, label="Ideal calibration")
        for series, colr, mk, nm in (("raw", SERIES[1], "o", "Raw sigmoid score"), ("calibrated", C_TEST, "D", "Frozen Platt-calibrated")):
            b = d[d.series == series].sort_values("bin")
            ax.plot(b.mean_predicted, b.observed_fraction, color=colr, lw=1.3, marker=mk, ms=4.5, label=nm)
        nb, npb = int(d[d.series == "raw"].shape[0]), int(d[d.series == "raw"].n.median())
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Mean predicted value per bin")
        ax.set_ylabel("Observed positive fraction")
        ax.text(0.04, 0.96, f"{lab} ({role}; prevalence {prev[lab]:.3f})" + chr(10) + f"{nb} equal-frequency bins, ~{npb:,} images each",
                transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
        sdata.append(d)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig5_test_reliability_bins", pd.concat(sdata))
    save(fig, "fig5_test_reliability")

    # ================================================================== Figure 6: finding counts
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    cols = [str(i) if i < 10 else "10+" for i in range(11)]
    gt, pr = dist[dist.source == "ground_truth"].iloc[0], dist[dist.source == "predicted"].iloc[0]
    x = np.arange(len(cols))
    ax.bar(x - 0.2, [gt[c] for c in cols], width=0.4, color=C_VAL, label="True (explicit positive labels)")
    ax.bar(x + 0.2, [pr[c] for c in cols], width=0.4, color=C_TEST, label="Predicted (final policy)")
    ax.set_xticks(x, cols)
    ax.set_xlabel("Number of positive findings per image (13 observations excluding No Finding)")
    ax.set_ylabel("Images (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    src("fig6_finding_counts", dist)
    save(fig, "fig6_finding_counts")

    # ================================================================== Figure 7 (optional): bootstrap CIs
    keys = ["macro_auroc", "macro_auprc", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "macro_balanced_accuracy", "macro_brier"]
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 3.8))
    for ax, k in zip(axes.flat, keys):
        pt, lo_, hi_ = (boot.loc[k, c] for c in ("point_estimate", "ci95_low", "ci95_high"))
        ax.errorbar([pt], [0], xerr=[[pt - lo_], [hi_ - pt]], fmt="D", color=C_TEST, ecolor=C_TEST, capsize=4, ms=5, lw=1.6)
        w = hi_ - lo_
        ax.set_xlim(lo_ - 0.6 * w, hi_ + 0.6 * w)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel(k.replace("macro_", "macro ").replace("balanced_accuracy", "balanced accuracy"))
        ax.text(0.5, 0.82, f"{pt:.4f}  [{lo_:.4f}, {hi_:.4f}]", transform=ax.transAxes, ha="center", fontsize=7, color=INK_2)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    src("fig7_bootstrap_ci", boot.loc[keys].reset_index())
    save(fig, "fig7_bootstrap_ci")

    dpis = [Image.open(p).info.get("dpi", (0, 0))[0] for p in sorted(FIG.glob("*.png"))]
    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(dpis)),
           "formats_per_figure": ["png", "pdf", "svg"], "formats_per_table": ["csv", "md", "tex"]}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
