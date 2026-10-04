"""R1 journal assets: 5 figures (PNG 300 dpi + vector PDF + SVG, with source data) and 6 tables (CSV, Markdown, LaTeX).
Reads only artifacts written by the earlier R1 scripts. No titles inside plots.

    .venv\\Scripts\\python.exe -m scripts.r1_06_figures_tables
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import INK_2, SERIES, apply_style
from src.classification.labels import LABELS
from src.retrieval.findings import ABNORMAL
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
FIG, SRC, TAB = OUT / "figures", OUT / "figures/source_data", OUT / "tables"
KS = (1, 3, 5, 10)
COL = {"oracle": SERIES[0], "classifier_all_positive": SERIES[1], "classifier_gated": SERIES[2]}
NAME = {"oracle": "Oracle finding query", "classifier_all_positive": "Classifier: all positives", "classifier_gated": "Classifier: precision-aware gated"}
RNAME = {"dense_minilm": "Dense (MiniLM)", "lexical_bm25": "Lexical (BM25)"}
rd = lambda n: pd.read_csv(OUT / n, float_precision="round_trip")  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731


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
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("@", r"@")  # noqa: E731
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
    inv, split = json.loads((OUT / "iu_xray_dataset_inventory.json").read_text(encoding="utf-8")), json.loads((OUT / "retrieval_split.json").read_text(encoding="utf-8"))
    qs, stats = json.loads((OUT / "r1_query_sets.json").read_text(encoding="utf-8")), json.loads((OUT / "r1_mapping_and_corpus_stats.json").read_text(encoding="utf-8"))
    emb, rnd = json.loads((OUT / "embedding_metadata.json").read_text(encoding="utf-8")), json.loads((OUT / "random_baseline_expectation.json").read_text(encoding="utf-8"))
    summ, ci = rd("retrieval_summary_primary.csv"), rd("retrieval_bootstrap_ci.csv")
    cq, gate = rd("classifier_query_comparison.csv"), rd("retrieval_gate_table.csv")
    shift, rem = rd("iu_domain_shift_per_finding.csv"), rd("gating_removed_terms.csv")
    dsum = json.loads((OUT / "iu_domain_shift_summary.json").read_text(encoding="utf-8"))
    pq = rd("per_query_metrics.csv")
    S = lambda rt, pol, c: float(summ[(summ.retriever == rt) & (summ.policy == pol)][c].iloc[0])  # noqa: E731
    CI = lambda rt, pol, m: ci[(ci.retriever == rt) & (ci.policy == pol) & (ci.metric == m)].iloc[0]  # noqa: E731

    # ================================================================== tables
    sz, sc = split["sizes"], inv["sections_after_cleaning"]
    t1 = pd.DataFrame([
        ("IU X-Ray reports / studies", f"{inv['reports_studies']:,}"), ("Images listed (frontal / lateral)", f"{inv['images_listed_in_projections_csv']:,} ({inv['frontal_images']:,} / {inv['lateral_images']:,})"),
        ("Studies with more than one image", f"{inv['studies_with_multiple_images']:,}"), ("Studies without a frontal image", f"{inv['studies_without_frontal']:,}"),
        ("Reports with Findings / Impression / both / neither", f"{sc['with_findings']:,} / {sc['with_impression']:,} / {sc['with_both']:,} / {sc['with_neither']:,}"),
        ("Reference corpus (train split, non-empty text)", f"{sz['reference_corpus_studies']:,} studies"),
        ("Retrieval validation queries (val split)", f"{sz['validation_studies']:,} studies, {sz['validation_images']:,} images"),
        ("Locked retrieval test (test split; not opened)", f"{sz['locked_test_studies']:,} studies, {sz['locked_test_images']:,} images"),
        ("Validation studies with a usable finding representation", f"{stats['validation_with_eval_set']:,} ({stats['validation_indexed_normal']:,} indexed normal)"),
        ("Primary paired query set (frontal image, non-empty classifier query)", f"{qs['primary_paired_set_non_empty_classifier_query']:,}"),
        ("Split overlaps (corpus-validation / corpus-test / validation-test)", "0 / 0 / 0"),
        ("Embedding model; dimension (verified)", f"{emb['model_identifier']}; {emb['embedding_dimension_verified']}")], columns=["Item", "Value"])
    write_table("table1_dataset_and_split_summary", t1, "IU X-Ray dataset and leakage-safe retrieval split summary.")
    orc = []
    for rt in ("dense_minilm", "lexical_bm25"):
        orc.append({"Retriever": RNAME[rt], **{f"Jaccard@{k}": f3(S(rt, "oracle", f"jaccard_truth@{k}")) for k in KS}, "nDCG@3": f3(S(rt, "oracle", "ndcg@3")), "nDCG@5": f3(S(rt, "oracle", "ndcg@5")),
                    "Hit@3": f3(S(rt, "oracle", "hit@3")), "MRR@10": f3(S(rt, "oracle", "mrr@10"))})
    orc.append({"Retriever": "Random (expected)", **{f"Jaccard@{k}": f3(rnd[f"jaccard_truth@{k}"]) for k in KS}, "nDCG@3": f3(rnd["ndcg@3"]), "nDCG@5": f3(rnd["ndcg@5"]), "Hit@3": "n/a", "MRR@10": "n/a"})
    write_table("table2_dense_vs_lexical_baseline", pd.DataFrame(orc), "Dense and lexical retrieval with oracle finding queries (primary paired set).")
    t3 = cq[["retriever", "query_policy", "jaccard@1", "jaccard@3", "ndcg@3", "ndcg@5", "hit@3", "mean_query_words", "mean_findings_per_query", "empty_query_rate_eligible_classified_studies", "fallback_query_rate_eligible_classified_studies"]].copy()
    t3.insert(0, "Retriever", t3.pop("retriever").map(RNAME))
    t3["query_policy"] = t3["query_policy"].map(NAME)
    for c in ("jaccard@1", "jaccard@3", "ndcg@3", "ndcg@5", "hit@3", "mean_query_words", "mean_findings_per_query"):
        t3[c] = t3[c].map(lambda v: f"{v:.3f}" if c.startswith(("jacc", "ndcg", "hit")) else f"{v:.2f}")
    for c in ("empty_query_rate_eligible_classified_studies", "fallback_query_rate_eligible_classified_studies"):
        t3[c] = t3[c].map(lambda v: f"{100 * v:.1f}%")
    t3.columns = ["Retriever", "Query policy", "Jaccard@1", "Jaccard@3", "nDCG@3", "nDCG@5", "Hit@3", "Query words", "Findings/query", "Empty rate", "Fallback rate"]
    write_table("table3_oracle_vs_classifier_queries", t3, "Oracle versus frozen-classifier queries (all positives and precision-aware gated) on the primary paired set.")
    r4 = []
    for rt in ("dense_minilm", "lexical_bm25"):
        for pol in ("oracle", "classifier_all_positive", "classifier_gated"):
            for k in KS:
                r4.append({"Retriever": RNAME[rt], "Query policy": NAME[pol], "K": k, "Jaccard@K": f3(S(rt, pol, f"jaccard_truth@{k}")), "nDCG@K": f3(S(rt, pol, f"ndcg@{k}")),
                           "Hit@K": f3(S(rt, pol, f"hit@{k}")), "Union coverage@K": f3(S(rt, pol, f"union_coverage@{k}")),
                           "Zero-overlap rate@K": f3(S(rt, pol, f"zero_overlap_rate@{k}")), "Duplicate-text rate@K": f3(S(rt, pol, f"duplicate_text_rate@{k}"))})
    write_table("table4_performance_across_k", pd.DataFrame(r4), "Retrieval performance at K = 1, 3, 5 and 10.")
    t5 = shift[["finding", "iu_truth_frequency_eligible_studies", "predicted_frequency_all_frontal_studies", "precision", "recall", "f1"]].merge(
        rem[["finding", "gate", "removed_by_gate_low_confidence", "frozen_positive_studies"]].rename(columns={"removed_by_gate_low_confidence": "removed", "frozen_positive_studies": "pos_all_frontal"}), on="finding", how="left")
    t5 = pd.DataFrame({"Finding": t5.finding, "IU truth frequency": t5.iu_truth_frequency_eligible_studies.map(f3), "Predicted frequency": t5.predicted_frequency_all_frontal_studies.map(f3),
                       "Precision": t5.precision.map(lambda v: "n/a" if pd.isna(v) else f3(v)), "Recall": t5.recall.map(lambda v: "n/a" if pd.isna(v) else f3(v)), "F1": t5.f1.map(lambda v: "n/a" if pd.isna(v) else f3(v)),
                       "Retrieval gate": t5.gate.map(lambda v: "none" if pd.isna(v) else f3(v)) .where(t5.finding != "No Finding", "n/a"),
                       "Positives removed by gate": [("n/a" if pd.isna(a) else f"{int(a)} of {int(b)}") for a, b in zip(t5.removed, t5.pos_all_frontal)]})
    t5.loc[len(t5)] = ["Summary", f"{dsum['iu_indexed_normal_frequency_eligible']:.3f} indexed normal", f"No Finding {dsum['no_finding_positive_frequency']:.3f}; empty query {dsum['empty_query_frequency']:.3f}",
                       f3(dsum["descriptive_agreement_macro_over_13_abnormal"]["precision"]), f3(dsum["descriptive_agreement_macro_over_13_abnormal"]["recall"]), f3(dsum["descriptive_agreement_macro_over_13_abnormal"]["f1"]),
                       f"fallback {dsum['fallback_frequency']:.3f}", "macro over 13 abnormal findings"]
    write_table("table5_classifier_query_domain_shift", t5, "Frozen-classifier behaviour on IU X-Ray validation studies and query construction.")
    t6 = pd.DataFrame({"Finding": gate.finding, "Frozen raw threshold": gate.frozen_raw_threshold.map(f3), "Frozen calibrated-equivalent threshold": gate.frozen_calibrated_equivalent_threshold.map(f3),
                       "Retrieval gate": gate.retrieval_gate_calibrated_probability.map(lambda v: "none" if pd.isna(v) else f3(v)), "Precision before": gate.precision_before.map(f3),
                       "Precision after": gate.precision_after.map(f3), "Recall before": gate.recall_before.map(f3), "Recall after": gate.recall_after.map(f3),
                       "OOF F0.5 gain (95% CI)": [f"{a:+.3f} ({b:+.3f} to {c:+.3f})" for a, b, c in zip(gate.oof_f05_gain, gate.oof_gain_ci95_low, gate.oof_gain_ci95_high)]})
    write_table("table6_precision_aware_gates", t6, "Validation-derived precision-aware retrieval gates (CheXpert validation set; not classifier thresholds).")

    # ================================================================== Figure 1: dense vs lexical across K
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    f1s = []
    for ax, pol in zip(axes, ("oracle", "classifier_all_positive")):
        for rt, ls, mk in (("dense_minilm", "-", "o"), ("lexical_bm25", "--", "s")):
            m = [S(rt, pol, f"jaccard_truth@{k}") for k in KS]
            lo = [CI(rt, pol, f"jaccard_truth@{k}").ci95_low for k in KS]
            hi = [CI(rt, pol, f"jaccard_truth@{k}").ci95_high for k in KS]
            off = -0.06 if rt == "dense_minilm" else 0.06
            ax.errorbar(np.arange(4) + off, m, yerr=[np.array(m) - lo, np.array(hi) - m], color=COL[pol], ls=ls, marker=mk, ms=5, capsize=3, lw=1.4, label=RNAME[rt])
            f1s += [{"policy": pol, "retriever": rt, "K": k, "jaccard": a, "ci95_low": b, "ci95_high": c} for k, a, b, c in zip(KS, m, lo, hi)]
        ax.axhline(rnd["jaccard_truth@3"], color="#898781", lw=1.0, ls=":", label="Random (expected)")
        ax.set_xticks(range(4), [f"K={k}" for k in KS])
        ax.set_xlabel(NAME[pol])
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Finding Jaccard@K vs study truth")
    axes[0].legend(loc="lower left", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig1_dense_vs_lexical", pd.DataFrame(f1s))
    save(fig, "fig1_dense_vs_lexical_across_k")

    # ================================================================== Figure 2: oracle vs classifier policies (K=3)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    f2 = []
    for ax, (m, lab) in zip(axes, (("jaccard_truth@3", "Finding Jaccard@3 vs study truth"), ("ndcg@3", "nDCG@3"))):
        for j, pol in enumerate(("oracle", "classifier_all_positive", "classifier_gated")):
            vals, los, his = [], [], []
            for rt in ("dense_minilm", "lexical_bm25"):
                r = CI(rt, pol, m)
                vals.append(r["mean"]); los.append(r.ci95_low); his.append(r.ci95_high)
                f2.append({"metric": m, "policy": pol, "retriever": rt, "mean": r["mean"], "ci95_low": r.ci95_low, "ci95_high": r.ci95_high})
            x = np.arange(2) + (j - 1) * 0.26
            ax.bar(x, vals, width=0.24, color=COL[pol], yerr=[np.array(vals) - los, np.array(his) - vals], capsize=3, label=NAME[pol] if m == "ndcg@3" else None)
        ax.set_xticks(range(2), [RNAME["dense_minilm"], RNAME["lexical_bm25"]])
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig2_oracle_vs_classifier", pd.DataFrame(f2))
    save(fig, "fig2_oracle_vs_classifier_queries")

    # ================================================================== Figure 3: performance vs K (three policies, both retrievers)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    f3s = []
    for ax, (m, lab) in zip(axes, (("jaccard_truth", "Finding Jaccard@K"), ("union_coverage", "Union coverage of true findings@K"), ("duplicate_text_rate", "Duplicate-text rate@K"))):
        for pol in ("oracle", "classifier_all_positive", "classifier_gated"):
            for rt, ls, mk in (("dense_minilm", "-", "o"), ("lexical_bm25", "--", "s")):
                v = [S(rt, pol, f"{m}@{k}") for k in KS]
                ax.plot(range(4), v, color=COL[pol], ls=ls, marker=mk, ms=4.5, lw=1.4, label=f"{NAME[pol]}, {RNAME[rt].split()[0].lower()}" if m == "jaccard_truth" else None)
                f3s += [{"metric": m, "policy": pol, "retriever": rt, "K": k, "value": x} for k, x in zip(KS, v)]
        ax.set_xticks(range(4), [f"K={k}" for k in KS])
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower left", frameon=False, fontsize=6, ncol=1)
    fig.tight_layout()
    src("fig3_performance_vs_k", pd.DataFrame(f3s))
    save(fig, "fig3_performance_vs_k")

    # ================================================================== Figure 4: distribution of Top-3 finding agreement (dense)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    f4s = []
    edges = np.linspace(0, 1, 11)
    for ax, pol in zip(axes, ("oracle", "classifier_all_positive", "classifier_gated")):
        v = pq[(pq.retriever == "dense_minilm") & (pq.policy == pol) & pq.in_primary_set & (pq.retrieved > 0)]["jaccard_truth@3"].to_numpy()
        h, _ = np.histogram(v, bins=edges)
        ax.bar(edges[:-1] + 0.05, 100 * h / len(v), width=0.09, color=COL[pol])
        ax.set_xlabel(f"Per-query Jaccard@3 ({NAME[pol]})")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="x", visible=False)
        f4s += [{"policy": pol, "bin_low": a, "bin_high": b, "percent_queries": 100 * c / len(v), "n_queries": len(v)} for a, b, c in zip(edges[:-1], edges[1:], h)]
        ax.text(0.5, 0.93, f"mean {v.mean():.3f}; zero overlap {100 * np.mean(v == 0):.1f}%", transform=ax.transAxes, ha="center", fontsize=8, color=INK_2)
    axes[0].set_ylabel("Queries (%)")
    fig.tight_layout()
    src("fig4_top3_agreement_distribution", pd.DataFrame(f4s))
    save(fig, "fig4_top3_agreement_distribution")

    # ================================================================== Figure 5: domain shift / query quality
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(shift))
    ax.barh(y - 0.19, shift.iu_truth_frequency_eligible_studies, height=0.36, color=SERIES[1], label="IU truth (mapped MeSH)")
    ax.barh(y + 0.19, shift.predicted_frequency_all_frontal_studies, height=0.36, color=SERIES[0], label="Frozen classifier prediction")
    ax.set_yticks(y, shift.finding)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of validation studies")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax = axes[1]
    cl = rd("iu_validation_classifier_outputs.csv").fillna("")
    cats = [("findings", "Pathology findings"), ("no_finding", "No Finding query"), ("fallback_strongest_positive", "Fallback: strongest positive"), ("empty", "Empty / undefined")]
    cols_ = [SERIES[0], SERIES[2], SERIES[3], "#898781"]
    f5 = []
    for i, (col, lab) in enumerate((("query_all_positive_status", "All positives"), ("query_gated_status", "Precision-aware"))):
        left = 0.0
        for (c, cn), colr in zip(cats, cols_):
            v = float((cl[col] == c).mean())
            ax.barh(i, v, left=left, color=colr, label=cn if i == 0 else None, height=0.5)
            f5.append({"policy": lab, "status": c, "fraction": v})
            left += v
    ax.set_yticks([0, 1], ["All positives", "Precision-aware"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of validation studies with a frontal image")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    src("fig5_domain_shift_left", shift[["finding", "iu_truth_frequency_eligible_studies", "predicted_frequency_all_frontal_studies"]])
    src("fig5_query_status_right", pd.DataFrame(f5))
    save(fig, "fig5_domain_shift_query_quality")

    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(Image.open(p).info.get("dpi", (0, 0))[0] for p in FIG.glob("*.png")))}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
