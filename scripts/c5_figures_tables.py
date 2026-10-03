"""C5 journal assets: 5 figures (PNG 300 dpi + vector PDF + SVG) and 4 tables (CSV, Markdown, LaTeX).

Reads only the CSV/JSON artifacts written by scripts.c5_calibration. No titles inside the plots (captions
are in FIGURE_CAPTIONS.md); per-figure source data goes to figures/source_data/.

    .venv\\Scripts\\python.exe -m scripts.c5_figures_tables
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

OUT = PROJECT_ROOT / "results/classification/experiments/c5_calibration"
FIG, SRC, TAB = OUT / "figures", OUT / "figures/source_data", OUT / "tables"
C_RAW, C_CAL, C_OBS, C_TEMP, C_ISO = SERIES[1], SERIES[0], "#0b0b0b", SERIES[2], SERIES[6]
METH_LAB = {"raw": "Raw sigmoid score", "temperature": "Temperature scaling", "platt": "Platt scaling", "isotonic": "Isotonic regression"}


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
    return ("\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n\\hline\n"
            + " & ".join(esc(c) for c in df.columns) + " \\\\\n\\hline\n" + body + "\n\\hline\n\\end{tabular}\n"
            + f"\\caption{{{esc(caption)}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def write_table(name: str, df: pd.DataFrame, caption: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (TAB / f"{name}.md").write_text(md_table(df) + "\n", encoding="utf-8")
    (TAB / f"{name}.tex").write_text(tex_table(df, caption, f"tab:{name}"), encoding="utf-8")


def plain_log_axis(ax, ticks):
    """Log x-axis with a few explicit, plainly formatted ticks (no 10^n clutter, no minor-tick labels)."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())


def dumbbell(ax, df, col_raw, col_cal, xlabel):
    y = np.arange(len(df))
    for i in y:
        ax.plot([df[col_raw].iloc[i], df[col_cal].iloc[i]], [i, i], color="#c3c2b7", lw=1.8, zorder=1)
    ax.scatter(df[col_raw], y, s=46, color=C_RAW, label="Raw sigmoid score", zorder=3, edgecolor="white", linewidth=0.8)
    ax.scatter(df[col_cal], y, s=46, color=C_CAL, marker="D", label="Platt-calibrated (out-of-fold)", zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, df["observation"])
    ax.invert_yaxis()
    lo, hi = float(min(df[col_raw].min(), df[col_cal].min())), float(max(df[col_raw].max(), df[col_cal].max()))
    cand = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if lo * 0.7 <= t <= hi * 1.4]
    plain_log_axis(ax, cand)
    ax.set_xlim(lo * 0.8, hi * 1.25)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False)


def main() -> int:
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    sel = pd.read_csv(OUT / "raw_vs_selected_per_class.csv", float_precision="round_trip")
    mac = pd.read_csv(OUT / "method_comparison_macro.csv", float_precision="round_trip").set_index("method")
    mapping = pd.read_csv(OUT / "threshold_mapping.csv", float_precision="round_trip")
    rel = pd.read_csv(OUT / "reliability_bins.csv", float_precision="round_trip")
    cal = json.loads((OUT / "final_calibrators.json").read_text(encoding="utf-8"))
    res = json.loads((OUT / "c5_results.json").read_text(encoding="utf-8"))
    SRC.mkdir(parents=True, exist_ok=True)

    # ---- Fig 1 / Fig 2: Brier and log loss (log x-axis: values span about 25x between classes)
    for col, lab, name in (("brier", "Brier score (log scale; lower is better)", "fig1_brier_raw_vs_calibrated"),
                           ("log_loss", "Log loss (log scale; lower is better)", "fig2_logloss_raw_vs_calibrated")):
        fig, ax = plt.subplots(figsize=(7.0, 5.4))
        dumbbell(ax, sel, f"raw_{col}", f"calibrated_{col}", lab)
        sel[["observation", "prevalence", f"raw_{col}", f"calibrated_{col}"]].to_csv(SRC / f"{name}.csv", index=False, float_format="%.17g")
        save(fig, name)

    # ---- Fig 3: reliability curves for three representative classes
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    src = []
    for ax, (role, lab) in zip(axes, res["representative"].items()):
        d = rel[rel.observation == lab]
        hi = float(max(d.mean_predicted.max(), d.observed_fraction.max())) * 1.08
        ax.plot([0, hi], [0, hi], color="#898781", lw=1.0, label="Ideal calibration")
        for series, colr, mk, nm in (("raw", C_RAW, "o", "Raw sigmoid score"), ("calibrated_oof", C_CAL, "D", "Platt-calibrated (out-of-fold)")):
            b = d[d.series == series].sort_values("bin")
            ax.plot(b.mean_predicted, b.observed_fraction, color=colr, lw=1.3, marker=mk, ms=4.5, label=nm)
        n_bins, n_per = int(d[d.series == "raw"].shape[0]), int(d[d.series == "raw"].n.median())
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Mean predicted value per bin")
        ax.set_ylabel("Observed positive fraction")
        prev_lab = sel.set_index("observation").loc[lab, "prevalence"]
        ax.text(0.04, 0.96, f"{lab} ({role}; prevalence {prev_lab:.3f})" + chr(10)
                + f"{n_bins} equal-frequency bins, ~{n_per:,} images each", transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
        src.append(d)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    pd.concat(src).to_csv(SRC / "fig3_reliability_bins.csv", index=False, float_format="%.17g")
    save(fig, "fig3_reliability_curves")

    # ---- Fig 4: relative change in calibration metrics per class
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(sel))
    d_b = 100 * (sel.calibrated_brier / sel.raw_brier - 1)
    d_l = 100 * (sel.calibrated_log_loss / sel.raw_log_loss - 1)
    ax.barh(y - 0.2, d_b, height=0.38, color=C_CAL, label="Brier score")
    ax.barh(y + 0.2, d_l, height=0.38, color=C_TEMP, label="Log loss")
    ax.axvline(0, color="#0b0b0b", lw=1.0)
    ax.set_yticks(y, sel.observation)
    ax.invert_yaxis()
    ax.set_xlabel("Change from raw to calibrated, out-of-fold (%; negative = better calibrated)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", frameon=False)
    pd.DataFrame({"observation": sel.observation, "brier_change_pct": d_b, "log_loss_change_pct": d_l}).to_csv(
        SRC / "fig4_relative_change.csv", index=False, float_format="%.17g")
    save(fig, "fig4_calibration_change_per_class")

    # ---- Fig 5: mean predicted value vs observed prevalence
    order = sel.sort_values("prevalence").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(order))
    ax.scatter(order.prevalence, y, s=70, marker="|", color=C_OBS, label="Observed prevalence", zorder=4, linewidth=2.2)
    ax.scatter(order.raw_mean_score, y, s=44, color=C_RAW, label="Mean raw sigmoid score", zorder=3, edgecolor="white", linewidth=0.8)
    ax.scatter(order.calibrated_oof_mean, y, s=40, color=C_CAL, marker="D", label="Mean calibrated probability (out-of-fold)", zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, order.observation)
    ax.invert_yaxis()
    plain_log_axis(ax, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlim(0.0105, 0.75)
    ax.set_xlabel("Mean value over validation images (log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False)
    order[["observation", "prevalence", "raw_mean_score", "calibrated_oof_mean"]].to_csv(SRC / "fig5_mean_vs_prevalence.csv", index=False, float_format="%.17g")
    save(fig, "fig5_mean_probability_vs_prevalence")

    # ---- Tables
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    t1 = pd.DataFrame({"Method": [METH_LAB[m] for m in ("raw", "temperature", "platt", "isotonic")],
                       "Macro Brier score": [f4(mac.loc[m, "brier"]) for m in ("raw", "temperature", "platt", "isotonic")],
                       "Macro log loss": [f4(mac.loc[m, "log_loss"]) for m in ("raw", "temperature", "platt", "isotonic")],
                       "Macro ECE": [f4(mac.loc[m, "ece"]) for m in ("raw", "temperature", "platt", "isotonic")]})
    write_table("table1_calibration_method_comparison", t1, "Macro calibration metrics (14 observations) from patient-level 5-fold out-of-fold predictions; the raw row is unaffected by cross-validation.")
    t2 = pd.DataFrame({"Observation": sel.observation, "Prevalence": sel.prevalence.map(lambda v: f"{v:.3f}"),
                       "Raw Brier": sel.raw_brier.map(f4), "Calibrated Brier": sel.calibrated_brier.map(f4),
                       "Raw log loss": sel.raw_log_loss.map(f4), "Calibrated log loss": sel.calibrated_log_loss.map(f4),
                       "Raw ECE": sel.raw_ece.map(f4), "Calibrated ECE": sel.calibrated_ece.map(f4),
                       "Selected method": sel.method.map(lambda m: METH_LAB[m].split()[0])})
    write_table("table2_per_class_raw_vs_calibrated", t2, "Per-class calibration metrics, raw sigmoid score versus the selected calibrator (out-of-fold).")
    rows = []
    for lab in LABELS:
        s = cal["classes"][lab]
        rows.append({"Observation": lab, "Method": METH_LAB[s["method"]].split()[0], "Slope a": f"{s['a']:.3f}", "Intercept b": f"{s['b']:.3f}",
                     "Images used": f"{s['n_fit']:,}", "Positives used": f"{s['n_fit_positive']:,}"})
    assert all(cal["classes"][l]["method"] == "platt" for l in LABELS)         # all-Platt -> slope/intercept columns apply to every row
    write_table("table3_final_calibration_parameters", pd.DataFrame(rows), "Final Platt-scaling parameters, p = sigmoid(a * logit(score) + b), fitted on the full validation set.")
    t4 = pd.DataFrame({"Observation": mapping.observation, "Raw F1 threshold": mapping.raw_f1_threshold.map(f4),
                       "Calibrated-equivalent threshold": mapping.calibrated_equivalent_threshold.map(f4),
                       "Decision mismatches": mapping.decision_mismatches_all_images.astype(int)})
    write_table("table4_threshold_mapping", t4, "Raw C4 F1-optimal thresholds and their calibrated-equivalents, with the number of validation images whose binary decision differs.")

    # ---- asset checks
    chk = {"png_dpi": {}, "pdf_has_embedded_raster": {}, "svg_present": {}}
    for f in sorted(FIG.glob("fig*.png")):
        im = Image.open(f)
        chk["png_dpi"][f.stem] = {"dpi": round(float(im.info.get("dpi", (0, 0))[0])), "pixels": im.size}
        chk["pdf_has_embedded_raster"][f.stem] = b"/Subtype /Image" in (FIG / f"{f.stem}.pdf").read_bytes()
        chk["svg_present"][f.stem] = (FIG / f"{f.stem}.svg").exists()
    assert len(chk["png_dpi"]) == 5 and all(v["dpi"] >= 299 for v in chk["png_dpi"].values())
    assert not any(chk["pdf_has_embedded_raster"].values()) and all(chk["svg_present"].values())
    s4 = pd.read_csv(SRC / "fig4_relative_change.csv", float_precision="round_trip")
    assert np.allclose(s4.brier_change_pct, 100 * (sel.calibrated_brier / sel.raw_brier - 1))
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"figures": 5, "tables": len(list(TAB.glob("*.csv"))), "min_dpi": min(v["dpi"] for v in chk["png_dpi"].values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
