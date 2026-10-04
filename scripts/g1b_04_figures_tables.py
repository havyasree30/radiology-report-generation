"""G1B journal assets: 6 figures (PNG 300 dpi + vector PDF + SVG, with source data) and 6 main + 2 supplementary tables (CSV, Markdown, LaTeX).
Reads only artifacts written by g1b_03_evaluate. No titles inside plots.

    .venv\\Scripts\\python.exe -m scripts.g1b_04_figures_tables
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import SERIES, apply_style
from src.utils.config import PROJECT_ROOT

G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
FIG, SRC, TAB = G1B / "figures", G1B / "figures/source_data", G1B / "tables"
B0S, GAS, GBS, G1S = "B0_rule_based", "G1A_llm_no_retrieval", "G1B_llm_sham_retrieval", "G1_single_agent_rag"
SYSTEMS = (B0S, GAS, GBS, G1S)
NAME = {B0S: "B0 rule-based", GAS: "G1A LLM, no retrieval", GBS: "G1B LLM + sham retrieval", G1S: "G1 LLM + relevant retrieval"}
COL = {B0S: SERIES[1], GAS: SERIES[3], GBS: SERIES[2], G1S: SERIES[0]}
f3 = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
pct = lambda v: "n/a" if pd.isna(v) else f"{100 * v:.1f}%"  # noqa: E731


def main() -> int:
    for d in (FIG, SRC, TAB):
        d.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    rd = lambda n: pd.read_csv(G1B / n, float_precision="round_trip")  # noqa: E731
    main_c, pair = rd("g1b_main_comparison.csv").set_index("metric"), rd("g1b_paired_differences.csv")
    pf, nad = rd("g1b_per_finding_results.csv"), rd("g1b_normal_abnormal.csv")
    intro, dif, cpy, cpd, manip = rd("g1b_context_introduced_and_removed_findings.csv"), rd("g1b_g1_vs_g1b_findings.csv"), rd("g1b_copying_analysis.csv"), rd("g1b_copying_paired_difference.csv"), rd("g1b_context_manipulation_check.csv")
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
            ax.bar(np.arange(len(metrics)) + (j - 1.5) * 0.21, mid, width=0.2, color=COL[s], yerr=[mid - lo, hi - mid], capsize=2, label=NAME[s] if legend else None)
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
    write_table("table1_four_system_main_comparison", t1, "B0 rule-based, G1A LLM without retrieval, G1B LLM with sham retrieval and G1 LLM with relevant retrieval on identical studies (95% bootstrap intervals).")
    T2 = T1 + [("Mean hallucinated findings per report", "mean_hallucinated"), ("Mean omitted findings per report", "mean_omitted"), ("Over-normalisation: report normal although classifier abnormal", "over_normalisation_given_classifier_abnormal"),
               ("Rare-finding TP retention", "rare_tp_retention"), ("Other-finding TP retention", "common_tp_retention"), ("BLEU-1", "bleu1")]
    t2 = pd.DataFrame([{"Metric": n, "G1B (sham context)": f3(D(G1S, GBS, m).b), "G1 (relevant context)": f3(D(G1S, GBS, m).a), "G1 - G1B (95% CI)": dcell(D(G1S, GBS, m)), "CI excludes 0": "yes" if D(G1S, GBS, m).excludes_zero else "no"} for n, m in T2])
    write_table("table2_g1_minus_g1b_controlled_comparison", t2, "Primary controlled comparison G1 minus G1B: relevant versus structurally matched sham context with prompt, model, decoding and classifier output held constant (paired study-level bootstrap).")
    ti = intro.set_index("system")
    ctxrow = lambda s: "n/a" if s in (B0S, GAS) else int(ti.loc[s, "introduced_in_own_context_reports"])  # noqa: E731
    t3 = pd.DataFrame({"Quantity": ["Stated findings", "Introduced findings (stated, not classifier positive)", "  present in own context reports", "  match reference", "  unsupported by reference", "Share of introduced findings supported by reference",
                                    "Classifier positives removed (not stated)", "  of which reference-correct (lost)", "  of which unsupported (classifier false positive suppressed)"],
                       **{NAME[s]: [int(ti.loc[s, "stated_findings"]), int(ti.loc[s, "introduced_not_classifier_positive"]), ctxrow(s), int(ti.loc[s, "introduced_match_reference"]), int(ti.loc[s, "introduced_unsupported_by_reference"]),
                                    f3(ti.loc[s, "share_introduced_supported_by_reference"]), int(ti.loc[s, "classifier_positives_removed"]), int(ti.loc[s, "removed_were_correct_(reference_positive)"]),
                                    int(ti.loc[s, "removed_were_unsupported_(classifier_false_positive)"])] for s in SYSTEMS}})
    write_table("table3_introduced_and_removed_findings", t3, "Findings introduced beyond the classifier output and classifier positives removed, per system (clinical subset).")
    cc = cpy.set_index("system")
    own_c, own_w = "copied_sentence_rate_from_own_context (G1 definition)", "whole_report_copy_rate_from_own_context (G1 definition)"
    t4 = pd.DataFrame({"Quantity": ["Copied-sentence rate from own context (G1 definition)", "Whole-report copies of own context", "Copied-sentence rate from the G1 true Top-5", "Copied-sentence rate from the sham context",
                                    "Sentences (>=6 words) found verbatim in the corpus", "Generic template sentences (>=10 corpus reports) among long sentences", "Reports duplicating a whole corpus report", "Mean report length (words)"],
                       **{NAME[s]: [pct(cc.loc[s, own_c]), pct(cc.loc[s, own_w]), pct(cc.loc[s, "copied_sentence_rate_from_g1_true_top5 (studies with context)"]), pct(cc.loc[s, "copied_sentence_rate_from_sham_context (studies with context)"]),
                                    pct(cc.loc[s, "corpus_copied_sentence_rate"]), pct(cc.loc[s, "corpus_template_sentence_share_of_long (df>=10)"]), pct(cc.loc[s, "exact_whole_report_duplicate_of_a_corpus_report_rate"]),
                                    f"{cc.loc[s, 'mean_words']:.1f}"] for s in SYSTEMS}})
    write_table("table4_copying_analysis", t4, "Copying of context and corpus text by each system (studies with context for the context rows; all studies for the corpus rows).")
    na_rows = []
    for who, nm in (("classifier", "Classifier (= B0 report state)"), (B0S, NAME[B0S]), (GAS, NAME[GAS]), (GBS, NAME[GBS]), (G1S, NAME[G1S])):
        nn = nad[(nad.source == who)].set_index("reference")
        k = B0S if who in ("classifier", B0S) else who
        na_rows.append({"Source": nm, "Normal recall": cell("normal_recall", k), "Abnormal recall": cell("abnormal_recall", k),
                        "Abnormal studies reported normal": f"{int(nn.loc['abnormal', 'predicted_normal'])} of {int(nn.loc['abnormal', 'n'])}", "Normal studies reported abnormal": f"{int(nn.loc['normal', 'predicted_abnormal'])} of {int(nn.loc['normal', 'n'])}",
                        "Report normal given classifier abnormal": "n/a" if who == "classifier" else cell("over_normalisation_given_classifier_abnormal", k),
                        "Rare-finding TP retention": "n/a" if who == "classifier" else f3(M("rare_tp_retention", k)[0]), "Other-finding TP retention": "n/a" if who == "classifier" else f3(M("common_tp_retention", k)[0])})
    write_table("table5_normal_abnormal_and_rare_findings", pd.DataFrame(na_rows), "Normal/abnormal consistency and rare-finding retention for the classifier and the four report generators (rare findings: C2 definition, pooled).")
    t6 = pf.pivot(index="finding", columns="system", values=["n_truth", "classifier_true_positives", "tp_retention", "precision", "recall", "f1"]).reset_index()
    t6.columns = ["finding"] + [f"{a}|{b}" for a, b in t6.columns[1:]]
    order = list(pf[pf.system == B0S].finding)
    t6 = t6.set_index("finding").loc[order].reset_index()
    t6 = pd.DataFrame({"Finding": t6.finding + np.where(pf[pf.system == B0S].set_index("finding").loc[order].rare_c2.to_numpy(), " (rare)", ""), "Reference positives": t6[f"n_truth|{B0S}"].astype(int),
                       "Classifier TPs": t6[f"classifier_true_positives|{B0S}"].astype(int), **{f"TP retention {NAME[s]}": t6[f"tp_retention|{s}"].map(f3) for s in SYSTEMS},
                       **{f"P / R / F1 {NAME[s]}": [f"{f3(a)} / {f3(b)} / {f3(c)}" for a, b, c in zip(t6[f"precision|{s}"], t6[f"recall|{s}"], t6[f"f1|{s}"])] for s in SYSTEMS}})
    write_table("table6_per_finding_results", t6, "Per-finding results on the clinical subset for the four systems; rare findings follow the C2 definition.")
    st = rd("g1b_stratified_by_classifier_state.csv")
    keep = st[st.metric.isin(["precision", "recall", "f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall"])]
    t7 = pd.DataFrame({"Classifier state": keep.classifier_state, "Studies": keep.n_studies, "Metric": keep.metric, "G1A": keep[GAS].map(f3), "G1B": keep[GBS].map(f3), "G1": keep[G1S].map(f3),
                       "G1 - G1B (95% CI)": [f"{a:+.3f} ({b:+.3f} to {c:+.3f})" for a, b, c in zip(keep.g1_minus_g1b, keep.ci95_low, keep.ci95_high)], "CI excludes 0": np.where(keep.excludes_zero, "yes", "no")})
    write_table("table7_supplementary_stratified_by_classifier_state", t7, "Supplementary: G1 minus G1B within each classifier state (descriptive).")
    mm = manip.set_index("system")
    t8 = pd.DataFrame({"Quantity": ["Studies with context (clinical subset)", "Mean Jaccard of context finding sets with the reference findings", "Context reports with the identical finding set as the reference",
                                    "Context reports sharing at least one reference finding", "Classifier positives present in the context", "Mean words per context report"],
                       **{lab: [int(mm.loc[s, "n_clinical_studies_with_context"]), f3(mm.loc[s, "mean_jaccard_context_vs_reference_findings"]), pct(mm.loc[s, "mean_share_context_reports_with_identical_finding_set_as_reference"]),
                                pct(mm.loc[s, "mean_share_context_reports_sharing_a_reference_finding"]), pct(mm.loc[s, "mean_share_of_classifier_positives_present_in_context"]), f"{mm.loc[s, 'mean_context_words']:.1f}"]
                          for s, lab in ((G1S, "G1 relevant context"), (GBS, "G1B sham context"))}})
    write_table("table8_supplementary_context_manipulation_check", t8, "Supplementary: relevance of the supplied context to the reference findings and the classifier output for relevant (G1) and sham (G1B) context.")

    # ================================================================== figures
    leg = lambda ax, anchor=(0.5, 1.0): ax.legend(frameon=False, loc="lower center", bbox_to_anchor=anchor, ncol=2)  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    f1 = grouped(ax, ["precision", "recall", "f1"], ["Precision", "Recall", "F1 (micro)"], "Finding agreement with the IU reference", (0, 0.8))
    leg(ax)
    fig.tight_layout()
    src("fig1_finding_prf", pd.DataFrame(f1))
    save(fig, "fig1_finding_precision_recall_f1")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw={"wspace": 0.25})
    f2 = grouped(axes[0], ["hallucination_rate", "omission_rate"], ["Reports with at least one\nhallucinated finding", "Reports with at least one\nomitted finding"], "Share of reports", (0, 0.95))
    f2 += grouped(axes[1], ["mean_hallucinated", "mean_omitted"], ["Hallucinated findings\nper report", "Omitted findings\nper report"], "Findings per report", (0, 1.15), legend=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    src("fig2_hallucination_omission", pd.DataFrame(f2))
    save(fig, "fig2_hallucination_vs_omission")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    f3d = grouped(ax, ["clf_fp_propagation", "tp_retention"], ["Classifier false positives\nmentioned in report", "Classifier true positives\nretained in report"], "Proportion", (0, 1.1))
    leg(ax)
    fig.tight_layout()
    src("fig3_fp_propagation_tp_retention", pd.DataFrame(f3d))
    save(fig, "fig3_fp_propagation_vs_tp_retention")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    f4 = grouped(ax, ["normal_recall", "abnormal_recall"], ["Normal recall", "Abnormal recall"], "Share of reference studies with the correct state", (0, 1.05))
    leg(ax)
    ax.text(0.5, -0.2, "B0 report state equals the classifier state by construction", transform=ax.transAxes, ha="center", fontsize=7)
    fig.tight_layout()
    src("fig4_normal_abnormal_recall", pd.DataFrame(f4))
    save(fig, "fig4_normal_abnormal_recall")

    keys = [("precision", "Finding precision"), ("recall", "Finding recall"), ("f1", "Finding F1"), ("macro_f1", "Macro finding F1"), ("hallucination_rate", "Hallucination rate"), ("omission_rate", "Omission rate"),
            ("clf_fp_propagation", "Classifier FP propagation"), ("tp_retention", "Classifier TP retention"), ("normal_recall", "Normal recall"), ("abnormal_recall", "Abnormal recall"), ("rouge_l", "ROUGE-L")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    rows5 = []
    for i, (m, lab) in enumerate(keys):
        r = D(G1S, GBS, m)
        colr = SERIES[0] if r.excludes_zero else "#898781"
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r.ci95_low], [r.ci95_high - r["diff"]]], fmt="o", color=colr, ecolor=colr, capsize=3, ms=5)
        rows5.append({"metric": m, "diff_g1_minus_g1b": r["diff"], "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "ci_excludes_zero": bool(r.excludes_zero)})
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.set_yticks(np.arange(len(keys)), [l for _, l in keys])
    ax.invert_yaxis()
    ax.set_xlabel("G1 (relevant context) minus G1B (sham context): paired difference, 95% CI")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    labs = ["Stated with relevant\ncontext only (G1)", "Stated with sham\ncontext only (G1B)"]
    f5b = []
    for i, r in enumerate((dif.iloc[0], dif.iloc[1])):
        ax.barh(i, r.match_reference, color=SERIES[2], height=0.5, label="Matches reference" if i == 0 else None)
        ax.barh(i, r.unsupported_by_reference, left=r.match_reference, color=SERIES[3], height=0.5, label="Unsupported by reference" if i == 0 else None)
        ax.text(r.match_reference / 2, i, f"{int(r.match_reference)}", ha="center", va="center", fontsize=8, color="white")
        ax.text(r.match_reference + r.unsupported_by_reference / 2, i, f"{int(r.unsupported_by_reference)}", ha="center", va="center", fontsize=8, color="white")
        f5b.append({"group": labs[i].replace("\n", " "), "match_reference": int(r.match_reference), "unsupported_by_reference": int(r.unsupported_by_reference)})
    ax.set_yticks(range(2), labs)
    ax.invert_yaxis()
    ax.set_xlabel("Findings that differ between G1 and G1B (count, clinical subset)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    src("fig5_relevance_effect_forest", pd.DataFrame(rows5))
    src("fig5_g1_vs_g1b_counts", pd.DataFrame(f5b))
    save(fig, "fig5_relevance_effect_summary")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    f6 = []
    for j, s in enumerate((GBS, G1S)):
        r = ti.loc[s]
        ax.bar(j * 2 - 0.2, r.introduced_match_reference, width=0.4, color=SERIES[2], label="Matches reference" if j == 0 else None)
        ax.bar(j * 2 + 0.2, r.introduced_unsupported_by_reference, width=0.4, color=SERIES[3], label="Unsupported by reference" if j == 0 else None)
        ax.text(j * 2 - 0.2, r.introduced_match_reference, f"{int(r.introduced_match_reference)}", ha="center", va="bottom", fontsize=8)
        ax.text(j * 2 + 0.2, r.introduced_unsupported_by_reference, f"{int(r.introduced_unsupported_by_reference)}", ha="center", va="bottom", fontsize=8)
        f6.append({"system": s, "panel": "introduced_findings", "match_reference": int(r.introduced_match_reference), "unsupported_by_reference": int(r.introduced_unsupported_by_reference)})
    ax.set_xticks([0, 2], ["G1B sham context", "G1 relevant context"])
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylabel("Introduced findings (stated, not classifier positive)")
    ax.set_ylim(0, max(ti.loc[[GBS, G1S], "introduced_unsupported_by_reference"].max(), ti.loc[[GBS, G1S], "introduced_match_reference"].max()) * 1.3)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    cm = {"copied_sentence_rate_from_own_context": "Sentences copied from\nown context", "whole_report_copy_rate_from_own_context": "Whole-report copies\nof own context"}
    for i, (_, r) in enumerate(cpd.iterrows()):
        ax.bar(i - 0.2, r.G1, width=0.4, color=SERIES[0], label="G1 relevant context" if i == 0 else None)
        ax.bar(i + 0.2, r.G1B, width=0.4, color=SERIES[2], label="G1B sham context" if i == 0 else None)
        ax.text(i - 0.2, r.G1, f"{100 * r.G1:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, r.G1B, f"{100 * r.G1B:.1f}%", ha="center", va="bottom", fontsize=8)
        f6 += [{"system": "G1", "panel": r.metric, "value": r.G1}, {"system": "G1B", "panel": r.metric, "value": r.G1B}]
    ax.set_xticks(range(len(cpd)), [cm[m] for m in cpd.metric])
    ax.set_ylabel("Share")
    ax.set_ylim(0, max(cpd.G1.max(), cpd.G1B.max()) * 1.3)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    src("fig6_introduced_findings_and_copying", pd.DataFrame(f6))
    save(fig, "fig6_introduced_findings_and_copying")
    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(Image.open(x).info.get("dpi", (0, 0))[0] for x in FIG.glob("*.png")))}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
