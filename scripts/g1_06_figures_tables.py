"""G1 journal assets: 6 figures (PNG 300 dpi + vector PDF + SVG, with source data) and 6 tables (CSV, Markdown, LaTeX).
Reads only artifacts written by g1_04 / g1_05. No titles inside plots.

    .venv\\Scripts\\python.exe -m scripts.g1_06_figures_tables [--dry-run]
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import SERIES, apply_style
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
SYS = {"B0_rule_based": "B0 rule-based", "G1_single_agent_rag": "G1 single-agent RAG"}
COL = {"B0_rule_based": SERIES[1], "G1_single_agent_rag": SERIES[0]}
f3 = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"  # noqa: E731


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    D = G1 / "dry_run" if args.dry_run else G1
    FIG, SRC, TAB = D / "figures", D / "figures/source_data", D / "tables"
    for d in (FIG, SRC, TAB):
        d.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    rd = lambda n: pd.read_csv(D / n, float_precision="round_trip")  # noqa: E731
    pops = json.loads((D / "g1_evaluation_populations.json").read_text(encoding="utf-8"))
    summ = json.loads((D / "g1_summary_metrics.json").read_text(encoding="utf-8"))
    bt, pf, ho, pr = rd("g1_paired_bootstrap.csv"), rd("g1_per_finding_results.csv"), rd("g1_hallucination_omission.csv"), rd("g1_fp_propagation.csv")
    sb, ps, lr = rd("g1_support_bucket_analysis.csv"), rd("g1_per_study_results.csv"), rd("g1_length_redundancy.csv")
    B = lambda a, m: bt[(bt.analysis == a) & (bt.metric == m)].iloc[0]  # noqa: E731

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

    ci = lambda r, s: f"{f3(r[s])} ({f3(r[s + '_ci95_low'])} to {f3(r[s + '_ci95_high'])})"  # noqa: E731
    dci = lambda r: f"{r['diff']:+.3f} ({r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731

    # ================================================================== tables
    p = pops
    t1 = pd.DataFrame([("Retrieval-validation studies", p["validation_studies"]), ("With a frontal image (classifier output available)", p["validation_studies_with_frontal_image"]),
                       ("Frontal studies without a usable reference report (excluded)", p["frontal_without_usable_reference_report"]), ("Primary report-generation set", p["primary_report_generation_set"]),
                       ("  classifier query: findings / No Finding / empty", " / ".join(str(p["primary_set_classifier_query_status"].get(k, 0)) for k in ("findings", "no_finding", "empty"))),
                       ("  studies without retrieval context (empty query)", p["primary_set_no_retrieval_context_studies"]), ("Clinical-finding evaluation subset (usable mapped IU findings)", p["clinical_finding_evaluation_subset"]),
                       ("  of which reference normal", p["in_clinical_subset_reference_normal"]), ("Paired model-comparison set (B0 and G1 reports exist)", p["paired_model_comparison_set"]),
                       ("Paired clinical subset", p["paired_clinical_subset"]), ("Studies without a G1 report", p["studies_without_g1_report"]), ("Locked retrieval-test studies opened", p["locked_retrieval_test_studies_opened"])],
                      columns=["Population", "Studies"])
    write_table("table1_evaluation_population", t1, "G1 evaluation populations (retrieval-validation studies only).")
    rows = [("Finding precision", "finding_any_mention", "precision"), ("Finding recall", "finding_any_mention", "recall"), ("Finding F1", "finding_any_mention", "f1"),
            ("Hallucination rate (reports with >=1 hallucinated finding)", "finding_any_mention", "hallucination_rate"), ("Omission rate (reports with >=1 omitted finding)", "finding_any_mention", "omission_rate"),
            ("Classifier FP propagation", "propagation_any_mention", "clf_fp_propagation"), ("True-positive retention", "propagation_any_mention", "tp_retention"),
            ("ROUGE-L", "text_primary_set", "rouge_l"), ("BLEU-1", "text_primary_set", "bleu1"), ("BLEU-4", "text_primary_set", "bleu4"), ("METEOR (exact-match variant)", "text_primary_set", "meteor_exact")]
    t2 = pd.DataFrame([{"Metric": n, "Rule-based B0 (95% CI)": ci(B(a, m), "b0"), "Single-Agent RAG G1 (95% CI)": ci(B(a, m), "g1"), "Difference G1 - B0 (95% CI)": dci(B(a, m))} for n, a, m in rows])
    write_table("table2_b0_vs_g1_main_metrics", t2, "Rule-based baseline versus single-agent RAG (paired study-level bootstrap).")
    t3 = pf.pivot(index="finding", columns="system", values=["n_truth", "n_generated", "precision", "recall", "f1"]).reset_index()
    t3.columns = ["finding"] + [f"{a}|{b}" for a, b in t3.columns[1:]]
    t3 = pd.DataFrame({"Finding": t3.finding, "Reference positives": t3["n_truth|B0_rule_based"].astype(int), "B0 stated": t3["n_generated|B0_rule_based"].astype(int), "G1 stated": t3["n_generated|G1_single_agent_rag"].astype(int),
                       "B0 P / R / F1": [f"{f3(a)} / {f3(b)} / {f3(c)}" for a, b, c in zip(t3["precision|B0_rule_based"], t3["recall|B0_rule_based"], t3["f1|B0_rule_based"])],
                       "G1 P / R / F1": [f"{f3(a)} / {f3(b)} / {f3(c)}" for a, b, c in zip(t3["precision|G1_single_agent_rag"], t3["recall|G1_single_agent_rag"], t3["f1|G1_single_agent_rag"])]})
    write_table("table3_per_finding_generation_results", t3, "Per-finding generation results on the clinical subset (any affirmative mention).")
    t4 = ho.copy()
    t4 = pd.DataFrame({"System": t4.system.map(SYS), "Definition": t4.definition, "Hallucinated / report": t4.hallucinated_per_report.map(f3), "Omitted / report": t4.omitted_per_report.map(f3),
                       "Reports with >=1 hallucination (%)": t4.pct_reports_ge1_hallucinated.map(lambda v: f"{v:.1f}"), "Reports with >=1 omission (%)": t4.pct_reports_ge1_omission.map(lambda v: f"{v:.1f}"), "Reports": t4.n_reports})
    write_table("table4_hallucination_omission_analysis", t4, "Hallucination and omission analysis (clinical subset).")
    t5a = pr.copy()
    t5 = pd.DataFrame({"System": t5a.system.map(SYS), "Definition": t5a.definition, "Classifier FPs": t5a.n_clf_fp, "FPs mentioned": t5a.fp_mentioned, "FP propagation": t5a.clf_fp_propagation.map(f3),
                       "Classifier TPs": t5a.n_clf_tp, "TPs retained": t5a.tp_retained, "TP retention": t5a.tp_retention.map(f3)})
    write_table("table5_false_positive_propagation", t5, "Classifier false-positive propagation and true-positive retention (clinical subset).")
    rc = pd.read_csv(G1 / "g1_representative_cases.csv") if (G1 / "g1_representative_cases.csv").exists() else pd.DataFrame({"category": ["not generated in dry run"]})
    t6 = rc.fillna("").astype(str)
    write_table("table6_representative_cases", t6, "Representative validation cases (deterministic selection; details in qualitative_examples.md).")

    # ================================================================== Figure 1
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    f1 = []
    for j, (sysn, key) in enumerate((("B0_rule_based", "b0"), ("G1_single_agent_rag", "g1"))):
        v = [B("finding_any_mention", m) for m in ("precision", "recall", "f1")]
        mid, lo, hi = [r[key] for r in v], [r[key + "_ci95_low"] for r in v], [r[key + "_ci95_high"] for r in v]
        x = np.arange(3) + (j - 0.5) * 0.36
        ax.bar(x, mid, width=0.34, color=COL[sysn], yerr=[np.array(mid) - lo, np.array(hi) - mid], capsize=3, label=SYS[sysn])
        f1 += [{"system": sysn, "metric": m, "value": a, "ci95_low": b_, "ci95_high": c} for m, a, b_, c in zip(("precision", "recall", "f1"), mid, lo, hi)]
    ax.set_xticks(range(3), ["Precision", "Recall", "F1"])
    ax.set_ylabel("Finding agreement with IU reference (micro)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    src("fig1_finding_prf", pd.DataFrame(f1))
    save(fig, "fig1_finding_precision_recall_f1")

    # ================================================================== Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    f2 = []
    for ax, (ms, labs, yl) in zip(axes, ((("hallucination_rate", "omission_rate"), ("Reports with at least one\nhallucinated finding", "Reports with at least one\nomitted finding"), "Share of reports"),
                                         (("mean_hallucinated", "mean_omitted"), ("Hallucinated findings\nper report", "Omitted findings\nper report"), "Findings per report"))):
        for j, (sysn, key) in enumerate((("B0_rule_based", "b0"), ("G1_single_agent_rag", "g1"))):
            v = [B("finding_any_mention", m) for m in ms]
            mid, lo, hi = [r[key] for r in v], [r[key + "_ci95_low"] for r in v], [r[key + "_ci95_high"] for r in v]
            ax.bar(np.arange(2) + (j - 0.5) * 0.36, mid, width=0.34, color=COL[sysn], yerr=[np.array(mid) - lo, np.array(hi) - mid], capsize=3, label=SYS[sysn] if ax is axes[0] else None)
            f2 += [{"system": sysn, "metric": m, "value": a, "ci95_low": b_, "ci95_high": c} for m, a, b_, c in zip(ms, mid, lo, hi)]
        ax.set_xticks(range(2), labs)
        ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(1.1, 1.02), ncol=2)
    fig.tight_layout()
    src("fig2_hallucination_omission", pd.DataFrame(f2))
    save(fig, "fig2_hallucination_omission")

    # ================================================================== Figure 3
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    f3d = []
    for j, (sysn, key) in enumerate((("B0_rule_based", "b0"), ("G1_single_agent_rag", "g1"))):
        v = [B("propagation_any_mention", m) for m in ("clf_fp_propagation", "tp_retention")]
        mid, lo, hi = [r[key] for r in v], [r[key + "_ci95_low"] for r in v], [r[key + "_ci95_high"] for r in v]
        ax.bar(np.arange(2) + (j - 0.5) * 0.36, mid, width=0.34, color=COL[sysn], yerr=[np.array(mid) - lo, np.array(hi) - mid], capsize=3, label=SYS[sysn])
        f3d += [{"system": sysn, "metric": m, "value": a, "ci95_low": b_, "ci95_high": c} for m, a, b_, c in zip(("fp_propagation", "tp_retention"), mid, lo, hi)]
    ax.set_xticks(range(2), ["Classifier false positives\nmentioned in report", "Classifier true positives\nretained in report"])
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    src("fig3_fp_propagation_tp_retention", pd.DataFrame(f3d))
    save(fig, "fig3_fp_propagation_tp_retention")

    # ================================================================== Figure 4
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    f4 = []
    names = (("bleu1", "BLEU-1"), ("bleu4", "BLEU-4"), ("rouge_l", "ROUGE-L"), ("meteor_exact", "METEOR\n(exact-match variant)"))
    for j, (sysn, key) in enumerate((("B0_rule_based", "b0"), ("G1_single_agent_rag", "g1"))):
        v = [B("text_primary_set", m) for m, _ in names]
        mid, lo, hi = [r[key] for r in v], [r[key + "_ci95_low"] for r in v], [r[key + "_ci95_high"] for r in v]
        ax.bar(np.arange(4) + (j - 0.5) * 0.36, mid, width=0.34, color=COL[sysn], yerr=[np.array(mid) - lo, np.array(hi) - mid], capsize=3, label=SYS[sysn])
        f4 += [{"system": sysn, "metric": m, "value": a, "ci95_low": b_, "ci95_high": c} for (m, _), a, b_, c in zip(names, mid, lo, hi)]
    ax.set_xticks(range(4), [n for _, n in names])
    ax.set_ylabel("Mean score against the reference report")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    src("fig4_lexical_metrics", pd.DataFrame(f4))
    save(fig, "fig4_lexical_metrics")

    # ================================================================== Figure 5
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    prov = pd.read_csv(G1 / ("dry_run/g1_grounding_provenance.csv" if args.dry_run else "g1_grounding_provenance.csv"))
    left, labs = 0.0, {"A_classifier_only": "Classifier only", "B_retrieval_only": "Retrieval only", "C_both": "Both", "D_neither": "Neither"}
    cols = {"A_classifier_only": SERIES[1], "B_retrieval_only": SERIES[3], "C_both": SERIES[2], "D_neither": "#898781"}
    for _, r in prov.iterrows():
        axes[0].barh(0, r.share, left=left, color=cols[r.category], label=f"{labs[r.category]} (n={int(r.n_findings)})", height=0.5)
        if r.share >= 0.06:
            axes[0].text(left + r.share / 2, 0, f"{100 * r.share:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += r.share
    axes[0].set_yticks([])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Share of findings stated by G1 (clinical subset)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=7, ncol=2)
    sbp = sb.copy()
    for j, (tp_flag, nm, colr) in enumerate(((True, "Classifier true positive", SERIES[2]), (False, "Classifier false positive", SERIES[1]))):
        d = sbp[sbp.is_true_positive == tp_flag]
        axes[1].bar(np.arange(3) + (j - 0.5) * 0.36, d.g1_mention_rate.fillna(0), width=0.34, color=colr, label=nm)
        for x, (n_, v) in enumerate(zip(d.n, d.g1_mention_rate)):
            axes[1].text(x + (j - 0.5) * 0.36, (0 if pd.isna(v) else v) + 0.02, f"n={int(n_)}", ha="center", fontsize=7)
    axes[1].set_xticks(range(3), ["0/5", "1-2/5", "3-5/5"])
    axes[1].set_xlabel("Retrieved support for the classifier-positive finding")
    axes[1].set_ylabel("Mentioned in the G1 report")
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(frameon=False, loc="upper left", fontsize=7)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    src("fig5_provenance", prov)
    src("fig5_support_buckets", sbp)
    save(fig, "fig5_finding_support_provenance")

    # ================================================================== Figure 6: error-category breakdown
    def sets(v):
        return set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()

    cat = []
    for sysn in SYS:
        d = ps[(ps.system == sysn) & (ps.in_clinical)]
        c = {"hall_clf_fp": 0, "hall_other": 0, "omit_clf_dropped": 0, "omit_clf_missed": 0}
        for _, r in d.iterrows():
            st, tr, cl = sets(r.stated), sets(r.truth), sets(r.clf_pos)
            for f in st - tr:
                c["hall_clf_fp" if f in cl else "hall_other"] += 1
            for f in tr - st:
                c["omit_clf_dropped" if f in cl else "omit_clf_missed"] += 1
        cat.append({"system": sysn, **c, "n_reports": len(d)})
    cd = pd.DataFrame(cat)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, (keys, labs_, title) in zip(axes, ((("hall_clf_fp", "hall_other"), ("Classifier false positive", "Not a classifier positive (retrieval-derived or other)"), "Hallucinated findings"),
                                                 (("omit_clf_dropped", "omit_clf_missed"), ("Classifier true positive dropped by the report", "Never predicted by the classifier"), "Omitted findings"))):
        left = np.zeros(2)
        for k, lab_, colr in zip(keys, labs_, (SERIES[1], SERIES[3])):
            v = cd[k].to_numpy(float)
            ax.barh(np.arange(2), v, left=left, color=colr, label=lab_, height=0.5)
            for i_, (l_, w_) in enumerate(zip(left, v)):
                if w_ > 0:
                    ax.text(l_ + w_ / 2, i_, f"{int(w_)}", ha="center", va="center", fontsize=8)
            left += v
        ax.set_yticks(range(2), [SYS[s] for s in cd.system])
        ax.invert_yaxis()
        ax.set_xlabel(f"{title} (count, clinical subset)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=7)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    src("fig6_error_categories", cd)
    save(fig, "fig6_error_category_breakdown")
    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(Image.open(x).info.get("dpi", (0, 0))[0] for x in FIG.glob("*.png")))}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
