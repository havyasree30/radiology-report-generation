"""G1A journal assets: 5 figures (PNG 300 dpi + vector PDF + SVG, with source data) and 5 tables (CSV, Markdown, LaTeX).
Reads only artifacts written by g1a_04_evaluate. No titles inside plots.

    .venv\\Scripts\\python.exe -m scripts.g1a_05_figures_tables
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import SERIES, apply_style
from src.utils.config import PROJECT_ROOT

G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
FIG, SRC, TAB = G1A / "figures", G1A / "figures/source_data", G1A / "tables"
B0S, GAS, G1S = "B0_rule_based", "G1A_llm_no_retrieval", "G1_single_agent_rag"
SYSTEMS = (B0S, GAS, G1S)
NAME = {B0S: "B0 rule-based", GAS: "G1A LLM, no retrieval", G1S: "G1 LLM + retrieval"}
COL = {B0S: SERIES[1], GAS: SERIES[3], G1S: SERIES[0]}
f3 = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"  # noqa: E731


def main() -> int:
    for d in (FIG, SRC, TAB):
        d.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    rd = lambda n: pd.read_csv(G1A / n, float_precision="round_trip")  # noqa: E731
    main_c, pair = rd("g1a_main_comparison.csv").set_index("metric"), rd("g1a_paired_differences.csv")
    pf, ri, nad, dec = rd("g1a_per_finding_results.csv"), rd("g1a_retrieval_induced_findings.csv"), rd("g1a_normal_abnormal.csv"), rd("g1a_report_state_given_classifier_state.csv")
    rare = rd("g1a_rare_finding_retention.csv")
    M = lambda m, s: (main_c.loc[m, s], main_c.loc[m, s + "_ci95_low"], main_c.loc[m, s + "_ci95_high"])  # noqa: E731
    D = lambda a, b, m: pair[(pair.comparison == f"{a} minus {b}") & (pair.metric == m)].iloc[0]  # noqa: E731
    cell = lambda m, s: f"{f3(M(m, s)[0])} ({f3(M(m, s)[1])} to {f3(M(m, s)[2])})"  # noqa: E731
    dcell = lambda r: f"{r['diff']:+.3f} ({r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731

    def save(fig, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    def src(name, df):
        df.to_csv(SRC / f"{name}.csv", index=False, float_format="%.17g")

    def md_table(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|"] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()])

    def tex_table(df, cap, lab):
        esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")  # noqa: E731
        body = "\n".join(" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False))
        return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n\\hline\n" + " & ".join(esc(c) for c in df.columns) + " \\\\\n\\hline\n" + body
                + "\n\\hline\n\\end{tabular}\n" + f"\\caption{{{esc(cap)}}}\n\\label{{{lab}}}\n\\end{{table}}\n")

    def write_table(name, df, cap):
        (TAB / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        (TAB / f"{name}.md").write_text(md_table(df) + "\n", encoding="utf-8")
        (TAB / f"{name}.tex").write_text(tex_table(df, cap, f"tab:{name}"), encoding="utf-8")

    def grouped(ax, metrics, labels, ylabel, ylim=None, legend=True):
        rows = []
        for j, s in enumerate(SYSTEMS):
            mid = np.array([M(m, s)[0] for m in metrics])
            lo = np.array([M(m, s)[1] for m in metrics])
            hi = np.array([M(m, s)[2] for m in metrics])
            ax.bar(np.arange(len(metrics)) + (j - 1) * 0.27, mid, width=0.25, color=COL[s], yerr=[mid - lo, hi - mid], capsize=2.5, label=NAME[s] if legend else None)
            rows += [{"system": s, "metric": m, "value": a, "ci95_low": b, "ci95_high": c} for m, a, b, c in zip(metrics, mid, lo, hi)]
        ax.set_xticks(range(len(metrics)), labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        if ylim:
            ax.set_ylim(*ylim)
        return rows

    # ================================================================== tables
    T1 = [("Finding precision", "precision"), ("Finding recall", "recall"), ("Finding F1 (micro)", "f1"), ("Macro finding F1", "macro_f1"), ("Hallucination rate (reports with >=1 hallucinated finding)", "hallucination_rate"),
          ("Omission rate (reports with >=1 omitted finding)", "omission_rate"), ("Classifier FP propagation", "clf_fp_propagation"), ("Classifier TP retention", "tp_retention"), ("Normal recall (report state)", "normal_recall"),
          ("Abnormal recall (report state)", "abnormal_recall"), ("ROUGE-L", "rouge_l"), ("BLEU-4", "bleu4"), ("METEOR (exact-match variant)", "meteor_exact"), ("Mean report length (words)", "mean_words")]
    t1 = pd.DataFrame([{"Metric": n, **{NAME[s]: (f"{M(m, s)[0]:.1f} ({M(m, s)[1]:.1f} to {M(m, s)[2]:.1f})" if m == "mean_words" else cell(m, s)) for s in SYSTEMS}} for n, m in T1])
    write_table("table1_three_system_main_comparison", t1, "B0 rule-based, G1A LLM without retrieval and G1 LLM with retrieval on identical studies (95% bootstrap intervals).")
    T2 = T1 + [("Mean hallucinated findings per report", "mean_hallucinated"), ("Mean omitted findings per report", "mean_omitted"), ("Over-normalisation: report normal although classifier abnormal", "over_normalisation_given_classifier_abnormal"),
               ("Rare-finding TP retention", "rare_tp_retention"), ("Other-finding TP retention", "common_tp_retention"), ("BLEU-1", "bleu1")]
    t2 = pd.DataFrame([{"Metric": n, "G1A (no retrieval)": f3(D(G1S, GAS, m).b), "G1 (retrieval)": f3(D(G1S, GAS, m).a), "G1 - G1A (95% CI)": dcell(D(G1S, GAS, m)), "CI excludes 0": "yes" if D(G1S, GAS, m).excludes_zero else "no"} for n, m in T2])
    write_table("table2_g1_minus_g1a_controlled_comparison", t2, "Controlled comparison G1 minus G1A: effect of the retrieval context with classifier, model and decoding held constant (paired study-level bootstrap).")
    t3 = pf.pivot(index="finding", columns="system", values=["n_truth", "classifier_true_positives", "tp_retention", "precision", "recall", "f1"]).reset_index()
    t3.columns = ["finding"] + [f"{a}|{b}" for a, b in t3.columns[1:]]
    order = list(pf[pf.system == B0S].finding)
    t3 = t3.set_index("finding").loc[order].reset_index()
    t3 = pd.DataFrame({"Finding": t3.finding + np.where(pf[pf.system == B0S].set_index("finding").loc[order].rare_c2.to_numpy(), " (rare)", ""), "Reference positives": t3[f"n_truth|{B0S}"].astype(int),
                       "Classifier TPs": t3[f"classifier_true_positives|{B0S}"].astype(int), **{f"TP retention {NAME[s]}": t3[f"tp_retention|{s}"].map(f3) for s in SYSTEMS},
                       **{f"P / R / F1 {NAME[s]}": [f"{f3(a)} / {f3(b)} / {f3(c)}" for a, b, c in zip(t3[f"precision|{s}"], t3[f"recall|{s}"], t3[f"f1|{s}"])] for s in SYSTEMS}})
    write_table("table3_per_finding_results", t3, "Per-finding results on the clinical subset for the three systems; rare findings follow the C2 definition.")
    t4 = ri.copy()
    keep = ["group", "n_findings", "appear_in_retrieved_reports", "not_in_any_retrieved_report", "match_reference", "unsupported_by_reference", "also_classifier_positive", "not_classifier_positive", "match_reference_and_not_classifier_positive",
            "unsupported_and_not_classifier_positive"]
    t4a = t4[t4.group.str.startswith(("added", "removed"))][keep].copy()
    t4a.columns = ["Group", "Findings", "In retrieved reports", "Not in any retrieved report", "Match reference", "Unsupported by reference", "Also classifier positive", "Not classifier positive",
                   "Match reference and not classifier positive", "Unsupported and not classifier positive"]
    net = t4[t4.group.str.startswith("net")].iloc[0]
    t4a.loc[len(t4a)] = ["Net (added - removed)", "", "", "", int(net["net_correct_findings (added matching reference - removed matching reference)"]), int(net["net_unsupported_findings (added unsupported - removed unsupported)"]), "", "", "", ""]
    write_table("table4_retrieval_induced_findings", t4a, "Findings that differ between G1 and G1A (clinical subset): added and removed by the retrieval context.")
    na_rows = []
    for who, nm in (("classifier", "Classifier (= B0 report state)"), (B0S, NAME[B0S]), (GAS, NAME[GAS]), (G1S, NAME[G1S])):
        nn = nad[(nad.source == who)].set_index("reference")
        s_key = B0S if who in ("classifier", B0S) else who
        na_rows.append({"Source": nm, "Normal recall": cell("normal_recall", s_key), "Abnormal recall": cell("abnormal_recall", s_key),
                        "Abnormal studies reported normal": f"{int(nn.loc['abnormal', 'predicted_normal'])} of {int(nn.loc['abnormal', 'n'])}", "Normal studies reported abnormal": f"{int(nn.loc['normal', 'predicted_abnormal'])} of {int(nn.loc['normal', 'n'])}",
                        "Indeterminate": int(nn.loc["abnormal", "predicted_indeterminate"] + nn.loc["normal", "predicted_indeterminate"]),
                        "Report normal given classifier abnormal": "n/a" if who == "classifier" else cell("over_normalisation_given_classifier_abnormal", s_key)})
    write_table("table5_normal_abnormal_analysis", pd.DataFrame(na_rows), "Normal / abnormal consistency of the classifier and the three report generators against the reference.")

    # supplementary table (post hoc, descriptive): G1 minus G1A within each classifier state
    st = rd("g1a_stratified_by_classifier_state.csv")
    keep = st[st.metric.isin(["precision", "recall", "f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall"])]
    t6 = pd.DataFrame({"Classifier state": keep.classifier_state, "Studies": keep.n_studies, "Metric": keep.metric, "B0": keep[B0S].map(f3), "G1A": keep[GAS].map(f3), "G1": keep[G1S].map(f3),
                       "G1 - G1A (95% CI)": [f"{a:+.3f} ({b:+.3f} to {c:+.3f})" for a, b, c in zip(keep.g1_minus_g1a, keep.ci95_low, keep.ci95_high)], "CI excludes 0": np.where(keep.excludes_zero, "yes", "no")})
    write_table("table6_supplementary_stratified_by_classifier_state", t6, "Supplementary (post hoc): G1 minus G1A within each classifier state (abnormal, normal, indeterminate).")

    # ================================================================== Figure 1
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    f1 = grouped(ax, ["precision", "recall", "f1"], ["Precision", "Recall", "F1 (micro)"], "Finding agreement with the IU reference", (0, 0.8))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    src("fig1_finding_prf", pd.DataFrame(f1))
    save(fig, "fig1_finding_precision_recall_f1")

    # ================================================================== Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    f2 = grouped(axes[0], ["hallucination_rate", "omission_rate"], ["Reports with at least one\nhallucinated finding", "Reports with at least one\nomitted finding"], "Share of reports", (0, 0.95))
    f2 += grouped(axes[1], ["mean_hallucinated", "mean_omitted"], ["Hallucinated findings\nper report", "Omitted findings\nper report"], "Findings per report", (0, 1.15), legend=False)
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(1.1, 1.02), ncol=3)
    fig.tight_layout()
    src("fig2_hallucination_omission", pd.DataFrame(f2))
    save(fig, "fig2_hallucination_vs_omission")

    # ================================================================== Figure 3
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    f3d = grouped(ax, ["clf_fp_propagation", "tp_retention"], ["Classifier false positives\nmentioned in report", "Classifier true positives\nretained in report"], "Proportion", (0, 1.1))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    src("fig3_fp_propagation_tp_retention", pd.DataFrame(f3d))
    save(fig, "fig3_fp_propagation_vs_tp_retention")

    # ================================================================== Figure 4
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    f4 = grouped(ax, ["normal_recall", "abnormal_recall"], ["Normal recall", "Abnormal recall"], "Share of reference studies with the correct state", (0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.text(0.5, -0.2, "B0 report state equals the classifier state by construction", transform=ax.transAxes, ha="center", fontsize=7)
    fig.tight_layout()
    src("fig4_normal_abnormal_recall", pd.DataFrame(f4))
    save(fig, "fig4_normal_abnormal_recall")

    # ================================================================== Figure 5: retrieval effect summary
    keys = [("precision", "Finding precision"), ("recall", "Finding recall"), ("f1", "Finding F1"), ("hallucination_rate", "Hallucination rate"), ("omission_rate", "Omission rate"), ("clf_fp_propagation", "Classifier FP propagation"),
            ("tp_retention", "Classifier TP retention"), ("abnormal_recall", "Abnormal recall"), ("rouge_l", "ROUGE-L")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(keys))
    rows5 = []
    for i, (m, lab) in enumerate(keys):
        r = D(G1S, GAS, m)
        colr = SERIES[0] if r.excludes_zero else "#898781"
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r.ci95_low], [r.ci95_high - r["diff"]]], fmt="o", color=colr, ecolor=colr, capsize=3, ms=5)
        rows5.append({"metric": m, "diff_g1_minus_g1a": r["diff"], "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "ci_excludes_zero": bool(r.excludes_zero)})
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.set_yticks(y, [l for _, l in keys])
    ax.invert_yaxis()
    ax.set_xlabel("G1 (retrieval) minus G1A (no retrieval), paired difference with 95% CI")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    addr = ri[ri.group.str.startswith("added")].iloc[0]
    remr = ri[ri.group.str.startswith("removed")].iloc[0]
    parts = [("Added by retrieval context", addr, "match_reference", "unsupported_by_reference", ("Matches reference", "Unsupported by reference")),
             ("Removed by retrieval context", remr, "match_reference", "unsupported_by_reference", ("Correct finding lost", "Unsupported finding removed"))]
    f5b = []
    for i, (lab, r, k1, k2, (l1, l2)) in enumerate(parts):
        ax.barh(i, r[k1], color=SERIES[2] if i == 0 else SERIES[1], height=0.5, label=l1)
        ax.barh(i, r[k2], left=r[k1], color=SERIES[3] if i == 0 else "#898781", height=0.5, label=l2)
        ax.text(r[k1] / 2, i, f"{int(r[k1])}", ha="center", va="center", fontsize=8, color="white")
        ax.text(r[k1] + r[k2] / 2, i, f"{int(r[k2])}", ha="center", va="center", fontsize=8, color="white")
        f5b.append({"group": lab, "match_reference": int(r[k1]), "unsupported_by_reference": int(r[k2])})
    ax.set_yticks(range(2), [p[0] for p in parts])
    ax.invert_yaxis()
    ax.set_xlabel("Findings that differ between G1 and G1A (count, clinical subset)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    src("fig5_retrieval_effect_forest", pd.DataFrame(rows5))
    src("fig5_retrieval_induced_counts", pd.DataFrame(f5b))
    save(fig, "fig5_retrieval_effect_summary")
    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(Image.open(x).info.get("dpi", (0, 0))[0] for x in FIG.glob("*.png")))}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
