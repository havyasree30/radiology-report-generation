"""C4 journal assets: 8 figures (PNG 300 dpi + vector PDF + SVG) and 4 tables (CSV, Markdown, LaTeX).

Reads only the CSVs written by scripts.c4_operating_policy. No titles inside the plots (captions are
in FIGURE_CAPTIONS.md). Per-figure source data is written to figures/source_data/.

    .venv\\Scripts\\python.exe -m scripts.c4_figures_tables
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from scripts.compare_losses import RARE
from src.analysis.plotting import INK_2, MUTED, SERIES, apply_style
from src.classification.labels import LABELS
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/classification/experiments/c4_operating_policy"
FIG, SRC, TAB = OUT / "figures", OUT / "figures/source_data", OUT / "tables"
POL = ["thr_050", "youden", "f1", "f1_nf"]
POL_LAB = {"thr_050": "Threshold 0.50", "youden": "Youden J", "f1": "F1-optimal", "f1_nf": "F1-optimal + No Finding rule",
           "ground_truth": "Ground truth"}
COL = {"thr_050": SERIES[0], "youden": SERIES[1], "f1": SERIES[2], "f1_nf": SERIES[6], "ground_truth": "#0b0b0b"}
METRIC_LAB = {"precision": "Precision", "recall": "Recall", "specificity": "Specificity", "f1": "F1",
              "balanced_accuracy": "Balanced accuracy"}


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def grouped_bars(ax, groups: list[str], series: dict[str, list[float]], labels: dict, colors: dict, ylim=(0, 1.0), fmt="{:.2f}"):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (k, vals) in enumerate(series.items()):
        xs = x + (i - (n - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.9, color=colors[k], label=labels[k])
        for xv, v in zip(xs, vals):
            ax.text(xv, v + (ylim[1] - ylim[0]) * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")  # noqa: E731
    head = " & ".join(esc(c) for c in df.columns) + r" \\"
    body = "\n".join(" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False))
    return ("\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n\\hline\n"
            + head + "\n\\hline\n" + body + "\n\\hline\n\\end{tabular}\n" + f"\\caption{{{esc(caption)}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def write_table(name: str, df: pd.DataFrame, caption: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    pretty = df.copy()
    (TAB / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (TAB / f"{name}.md").write_text(md_table(pretty) + "\n", encoding="utf-8")
    (TAB / f"{name}.tex").write_text(tex_table(pretty, caption, f"tab:{name}"), encoding="utf-8")


def main() -> int:
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    comp = pd.read_csv(OUT / "policy_comparison.csv", float_precision="round_trip")
    long = pd.read_csv(OUT / "policy_per_class_long.csv", float_precision="round_trip")
    rare = pd.read_csv(OUT / "rare_class_false_alarm.csv", float_precision="round_trip")
    nf = pd.read_csv(OUT / "no_finding_before_after.csv", float_precision="round_trip").set_index("policy")
    dist = pd.read_csv(SRC / "fig3_finding_count_distribution.csv", float_precision="round_trip")
    c14 = comp[comp.scope == "14 classes"].set_index("policy")
    SRC.mkdir(parents=True, exist_ok=True)

    # ---------------- Fig 1: macro precision / recall / F1
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ms = ["precision", "recall", "f1"]
    ser = {k: [c14.loc[k, f"macro_{m}"] for m in ms] for k in POL}
    grouped_bars(ax, [METRIC_LAB[m] for m in ms], ser, POL_LAB, COL, ylim=(0, 1.0))
    ax.set_ylabel("Macro average over 14 observations")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    pd.DataFrame({"metric": ms, **{k: v for k, v in ser.items()}}).to_csv(SRC / "fig1_policy_precision_recall_f1.csv", index=False, float_format="%.17g")
    save(fig, "fig1_policy_precision_recall_f1")

    # ---------------- Fig 2: balanced accuracy & specificity
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ms = ["specificity", "balanced_accuracy"]
    ser = {k: [c14.loc[k, f"macro_{m}"] for m in ms] for k in POL}
    grouped_bars(ax, [METRIC_LAB[m] for m in ms], ser, POL_LAB, COL, ylim=(0, 1.1))
    ax.set_ylabel("Macro average over 14 observations")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    pd.DataFrame({"metric": ms, **ser}).to_csv(SRC / "fig2_policy_specificity_balanced_accuracy.csv", index=False, float_format="%.17g")
    save(fig, "fig2_policy_specificity_balanced_accuracy")

    # ---------------- Fig 3: predicted abnormal findings per image
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    keys = ["ground_truth", "thr_050", "youden", "f1"]
    ser = {k: dist[f"pct_images_{k}"].tolist() for k in keys}
    grouped_bars(ax, dist["n_findings"].astype(str).tolist(), ser, POL_LAB, COL, ylim=(0, 32), fmt="{:.0f}")
    ax.set_xlabel("Number of positive findings per image (13 observations excluding No Finding)")
    ax.set_ylabel("Share of validation images (%)")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig3_predicted_findings_per_image")

    # ---------------- Fig 4: per-class F1
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(LABELS))
    wide = {k: long[long.policy == k].set_index("observation").loc[list(LABELS)] for k in ("thr_050", "youden", "f1")}
    for k, mk in (("thr_050", "o"), ("youden", "s"), ("f1", "D")):
        ax.scatter(wide[k]["f1"], y, s=42, marker=mk, color=COL[k], label=POL_LAB[k], zorder=3, edgecolor="white", linewidth=0.8)
    for i in y:
        v = [wide[k]["f1"].iloc[i] for k in wide]
        ax.plot([min(v), max(v)], [i, i], color="#c3c2b7", lw=1.5, zorder=1)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("F1 (validation, per observation)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    pd.DataFrame({"observation": LABELS, **{f"f1_{k}": wide[k]["f1"].to_numpy() for k in wide}}).to_csv(
        SRC / "fig4_per_class_f1.csv", index=False, float_format="%.17g")
    save(fig, "fig4_per_class_f1")

    # ---------------- Fig 5: per-class precision-recall movement 0.50 -> Youden -> F1
    fig, axes = plt.subplots(4, 4, figsize=(9.6, 9.6), sharex=True, sharey=True)
    for ax, (j, lab) in zip(axes.flat, enumerate(LABELS)):
        pts = [(wide[k]["recall"].iloc[j], wide[k]["precision"].iloc[j]) for k in ("thr_050", "youden", "f1")]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#c3c2b7", lw=1.2, zorder=1)
        for (rx, py), k, mk in zip(pts, ("thr_050", "youden", "f1"), ("o", "s", "D")):
            ax.scatter([rx], [py], s=34, marker=mk, color=COL[k], zorder=3, edgecolor="white", linewidth=0.6, label=POL_LAB[k])
        ax.set_title(lab, fontsize=8, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linewidth=0.4)
    for ax in axes.flat[14:]:
        ax.axis("off")
    axes.flat[14].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=8)
    fig.supxlabel("Recall")
    fig.supylabel("Precision")
    fig.tight_layout()
    pd.DataFrame([{"observation": lab, "policy": k, "recall": wide[k]["recall"].iloc[j], "precision": wide[k]["precision"].iloc[j]}
                  for j, lab in enumerate(LABELS) for k in wide]).to_csv(SRC / "fig5_per_class_precision_recall.csv", index=False, float_format="%.17g")
    save(fig, "fig5_per_class_precision_recall_tradeoff")

    # ---------------- Fig 6: rare-class false-alarm burden
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    rk = ("thr_050", "youden", "f1")
    for ax, (m, lab, ylim, fmt) in zip(axes, (("precision", "Precision", (0, 0.45), "{:.2f}"), ("recall", "Recall", (0, 1.0), "{:.2f}"),
                                               ("fp_per_tp", "False positives per true positive", None, "{:.1f}"))):
        ser = {k: [rare[(rare.policy == k) & (rare.observation == c)][m].iloc[0] for c in RARE] for k in rk}
        top = ylim[1] if ylim else max(max(v) for v in ser.values()) * 1.15
        grouped_bars(ax, list(RARE), ser, POL_LAB, COL, ylim=(0, top), fmt=fmt)
        ax.set_ylabel(lab)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    rare.to_csv(SRC / "fig6_rare_class_false_alarm.csv", index=False, float_format="%.17g")
    save(fig, "fig6_rare_class_false_alarm_burden")

    # ---------------- Fig 7: No Finding consistency
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ms = ["precision", "recall", "specificity", "f1"]
    ser = {k: [nf.loc[k, m] for m in ms] for k in ("f1", "f1_nf")}
    grouped_bars(axes[0], [METRIC_LAB[m] for m in ms], ser, POL_LAB, COL, ylim=(0, 1.08))
    axes[0].set_ylabel("No Finding (validation)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    ser2 = {k: [nf.loc[k, "contradiction_pct_of_all_images_pathology12"], nf.loc[k, "contradiction_pct_of_no_finding_positive_pathology12"]] for k in ("f1", "f1_nf")}
    grouped_bars(axes[1], ["% of all images", "% of images predicted\nNo Finding"], ser2, POL_LAB, COL, ylim=(0, 30), fmt="{:.1f}")
    axes[1].set_ylabel("Contradictory predictions (%)\n(No Finding positive with a pathology positive)")
    fig.tight_layout()
    pd.DataFrame({"metric": ms, **ser}).to_csv(SRC / "fig7a_no_finding_metrics.csv", index=False, float_format="%.17g")
    pd.DataFrame({"measure": ["pct_all_images", "pct_no_finding_positive"], **ser2}).to_csv(SRC / "fig7b_no_finding_contradiction.csv", index=False, float_format="%.17g")
    save(fig, "fig7_no_finding_consistency")

    # ---------------- Fig 8: final thresholds
    fin = json.loads((OUT / "final_operating_policy.json").read_text(encoding="utf-8"))
    t = np.array([fin["classes"][l]["threshold"] for l in LABELS])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = np.arange(len(LABELS))
    ax.barh(y, t, color=SERIES[2], height=0.6)
    ax.axvline(0.5, color="#0b0b0b", lw=1.1, label="Fixed threshold 0.50")
    for i, v in enumerate(t):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Final per-class threshold on the sigmoid score (F1-optimal)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    pd.DataFrame({"observation": LABELS, "final_threshold": t}).to_csv(SRC / "fig8_final_thresholds.csv", index=False, float_format="%.17g")
    save(fig, "fig8_final_thresholds")

    # ---------------- Tables
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    t1 = []
    for scope, fname, cap in (("14 classes", "table1_policy_comparison", "Operating-policy comparison on the validation set (macro average over 14 observations)."),
                              ("13 abnormal findings (excl. No Finding)", "table1b_policy_comparison_13_abnormal",
                               "Operating-policy comparison on the 13 abnormal observations only (No Finding excluded).")):
        d = comp[comp.scope == scope].set_index("policy").loc[POL]
        df = pd.DataFrame({"Policy": [POL_LAB[k] for k in POL], "Macro precision": d["macro_precision"].map(f3).values,
                           "Macro recall": d["macro_recall"].map(f3).values, "Macro specificity": d["macro_specificity"].map(f3).values,
                           "Macro F1": d["macro_f1"].map(f3).values, "Macro balanced accuracy": d["macro_balanced_accuracy"].map(f3).values,
                           "Mean predicted abnormal findings per image": d["mean_predicted_abnormal_findings_per_image"].map(lambda v: f"{v:.2f}").values,
                           "No Finding contradiction rate (% of images)": d["no_finding_contradiction_pct_of_images"].map(lambda v: f"{v:.1f}").values})
        write_table(fname, df, cap)
    fl = long[long.policy == "f1_nf"].set_index("observation").loc[list(LABELS)]
    t2 = pd.DataFrame({"Observation": LABELS, "Prevalence": fl["prevalence"].map(f3).values, "Final threshold": fl["threshold"].map(lambda v: f"{v:.4f}").values,
                       "Precision": fl["precision"].map(f3).values, "Recall": fl["recall"].map(f3).values,
                       "Specificity": fl["specificity"].map(f3).values, "F1": fl["f1"].map(f3).values,
                       "Balanced accuracy": fl["balanced_accuracy"].map(f3).values})
    write_table("table2_final_per_class_operating_points", t2, "Final per-class operating points (F1-optimal thresholds with No Finding consistency rule) on the validation set.")
    r3 = rare.copy()
    r3["policy"] = r3["policy"].map(POL_LAB)
    t3 = pd.DataFrame({"Observation": r3["observation"], "Policy": r3["policy"], "Prevalence": r3["prevalence"].map(f3), "Threshold": r3["threshold"].map(lambda v: f"{v:.4f}"),
                       "TP": r3["tp"].astype(int), "FP": r3["fp"].astype(int), "Precision": r3["precision"].map(f3), "Recall": r3["recall"].map(f3),
                       "F1": r3["f1"].map(f3), "FP per TP": r3["fp_per_tp"].map(lambda v: f"{v:.1f}")})
    t3 = t3.assign(_o=t3["Observation"].map({c: i for i, c in enumerate(RARE)}), _p=r3["policy"].map({POL_LAB[k]: i for i, k in enumerate(POL)})).sort_values(["_o", "_p"]).drop(columns=["_o", "_p"])
    write_table("table3_rare_class_performance", t3, "Rare-class performance under three thresholding policies on the validation set.")
    rows = []
    for k, nm in (("f1", "F1-optimal (raw)"), ("f1_nf", "F1-optimal + No Finding rule")):
        r = nf.loc[k]
        rows.append({"Policy": nm, "TP": int(r.tp), "FP": int(r.fp), "TN": int(r.tn), "FN": int(r.fn), "Precision": f3(r.precision), "Recall": f3(r.recall),
                     "Specificity": f3(r.specificity), "F1": f3(r.f1), "% images predicted No Finding": f"{r.pct_predicted_no_finding:.1f}",
                     "Contradiction (% of images)": f"{r.contradiction_pct_of_all_images_pathology12:.1f}",
                     "Contradiction (% of predicted No Finding)": f"{r.contradiction_pct_of_no_finding_positive_pathology12:.1f}"})
    write_table("table4_no_finding_consistency", pd.DataFrame(rows), "No Finding prediction before and after the deterministic consistency rule (validation set).")

    # ---------------- asset checks (resolution, vector output, source-data agreement)
    report = {"png_dpi": {}, "pdf_has_embedded_raster": {}, "svg_present": {}}
    for f in sorted(FIG.glob("fig*.png")):
        im = Image.open(f)
        dpi = im.info.get("dpi", (0, 0))[0]
        report["png_dpi"][f.stem] = {"dpi": round(float(dpi)), "pixels": im.size}
        pdf = (FIG / f"{f.stem}.pdf").read_bytes()
        report["pdf_has_embedded_raster"][f.stem] = b"/Subtype /Image" in pdf
        report["svg_present"][f.stem] = (FIG / f"{f.stem}.svg").exists()
    assert all(v["dpi"] >= 299 for v in report["png_dpi"].values()), report
    assert not any(report["pdf_has_embedded_raster"].values()), report
    assert all(report["svg_present"].values())
    assert len(report["png_dpi"]) == 8
    # figure source data == recomputation from the raw long table
    s1 = pd.read_csv(SRC / "fig1_policy_precision_recall_f1.csv", float_precision="round_trip").set_index("metric")
    for k in ("thr_050", "youden", "f1"):
        ref = long[long.policy == k][["precision", "recall", "f1"]].mean()
        assert abs(s1.loc["precision", k] - ref["precision"]) < 1e-12 and abs(s1.loc["f1", k] - ref["f1"]) < 1e-12
    report["figure_source_agrees_with_raw_tables"] = True
    (FIG / "asset_checks.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"figures": len(report["png_dpi"]), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": min(v["dpi"] for v in report["png_dpi"].values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
