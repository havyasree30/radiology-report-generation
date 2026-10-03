"""R2 journal assets: 6 figures (PNG 300 dpi + vector PDF + SVG, with source data) and 6 tables (CSV, Markdown, LaTeX).
Reads only artifacts written by the earlier R2 scripts. No titles inside plots.

    .venv\\Scripts\\python.exe -m scripts.r2_04_figures_tables
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import INK_2, SERIES, apply_style
from src.retrieval.r2_context import R2
from src.utils.config import PROJECT_ROOT

OUT = R2
FIG, SRC, TAB = OUT / "figures", OUT / "figures/source_data", OUT / "tables"
KS = (1, 3, 5, 10)
rd = lambda n: pd.read_csv(OUT / n, float_precision="round_trip")  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
LABEL = {"R1_Q0_dense": "Q0 finding names (R1 baseline)", "A1_Q1_expanded_dense": "Q1 expanded phrases", "A2_Q2_uniform_dense": "Q2 uniform-weighted",
         "A2_Q2_probability_dense": "Q2 probability-weighted", "A2_Q2_margin_dense": "Q2 margin-weighted", "A3_Q3_top1_dense": "Q3 top-1 finding",
         "A3_Q3_top2_dense": "Q3 top-2 findings", "A3_Q3_top3_dense": "Q3 top-3 findings", "R1_gated_dense": "R1 precision-aware gated",
         "B_dense": "Dense (MiniLM)", "B_bm25": "BM25", "B_hybrid_rrf": "Hybrid RRF"}


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
    return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n\\hline\n" + " & ".join(esc(c) for c in df.columns)
            + " \\\\\n\\hline\n" + body + "\n\\hline\n\\end{tabular}\n" + f"\\caption{{{esc(caption)}}}\n\\label{{{label}}}\n\\end{{table}}\n")


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
    dec = json.loads((OUT / "r2_stage_decisions.json").read_text(encoding="utf-8"))
    cfg = json.loads((OUT / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    S = rd("r2_summary_all_configs.csv").set_index("config")
    PC, MM, KL, FC = rd("r2_paired_comparisons.csv"), rd("r2_mmr_adoption_checks.csv"), rd("r2_k_selection_log.csv"), rd("r2_failure_categories.csv")
    rnd = json.loads((OUT / "random_reference.json").read_text(encoding="utf-8"))
    K = int(dec["selected_K"])
    C = {"blue": SERIES[0], "orange": SERIES[1], "green": SERIES[2], "yellow": SERIES[3], "grey": "#898781"}
    pcmp = lambda group, comp, metric: PC[(PC.group == group) & (PC.comparison == comp) & (PC.metric == metric)].iloc[0]  # noqa: E731

    # ================================================================== tables
    pol = ["R1_Q0_dense", "A1_Q1_expanded_dense", "A2_Q2_uniform_dense", "A2_Q2_probability_dense", "A2_Q2_margin_dense", "A3_Q3_top1_dense", "A3_Q3_top2_dense", "A3_Q3_top3_dense", "R1_gated_dense"]
    t1 = []
    for n in pol:
        d = PC[(PC.group == "query_policy_vs_Q0") & (PC.a == n) & (PC.metric == "jaccard_truth@3")]
        t1.append({"Query policy (dense)": LABEL[n], "Jaccard@3 (95% CI)": f"{f3(S.loc[n, 'jaccard_truth@3'])} ({f3(S.loc[n, 'jaccard_truth@3__ci95_low'])} to {f3(S.loc[n, 'jaccard_truth@3__ci95_high'])})",
                   "nDCG@3": f3(S.loc[n, "ndcg@3"]), "Coverage@3": f3(S.loc[n, "union_coverage@3"]), "Query words": f"{S.loc[n, 'mean_query_words']:.2f}", "Findings/query": f"{S.loc[n, 'mean_findings_per_query']:.2f}",
                   "Difference vs Q0 in Jaccard@3 (95% CI)": "reference" if n == "R1_Q0_dense" else (f"{d.iloc[0].mean_difference:+.3f} ({d.iloc[0].ci95_low:+.3f} to {d.iloc[0].ci95_high:+.3f})" if len(d) else "n/a")})
    write_table("table1_query_policy_comparison", pd.DataFrame(t1), "Query-policy comparison with dense retrieval on the primary paired set.")
    rets = ["B_dense", "B_bm25", "B_hybrid_rrf"]
    t2 = []
    for n in rets:
        d = PC[(PC.group == "retriever") & (PC.a == n) & (PC.b == "B_dense") & (PC.metric == "jaccard_truth@3")]
        t2.append({"Retriever": LABEL[n], **{f"Jaccard@{k}": f3(S.loc[n, f"jaccard_truth@{k}"]) for k in (1, 3, 5)}, "nDCG@3": f3(S.loc[n, "ndcg@3"]), "Coverage@3": f3(S.loc[n, "union_coverage@3"]), "Coverage@5": f3(S.loc[n, "union_coverage@5"]),
                   "Hit@3": f3(S.loc[n, "hit@3"]), "Difference vs dense in Jaccard@3 (95% CI)": "reference" if n == "B_dense" else f"{d.iloc[0].mean_difference:+.3f} ({d.iloc[0].ci95_low:+.3f} to {d.iloc[0].ci95_high:+.3f})"})
    write_table("table2_retriever_comparison", pd.DataFrame(t2), "Dense, BM25 and hybrid RRF retrieval with the selected query policy.")
    t3 = []
    for k in KS:
        r = {"K": k, "Jaccard@K": f3(S.loc["FINAL_pipeline", f"jaccard_truth@{k}"]), "nDCG@K": f3(S.loc["FINAL_pipeline", f"ndcg@{k}"]), "Hit@K": f3(S.loc["FINAL_pipeline", f"hit@{k}"]),
             "Coverage@K": f3(S.loc["FINAL_pipeline", f"union_coverage@{k}"]), "Random coverage@K": f3(rnd[f"union_coverage@{k}"]), "Zero-overlap rate": f3(S.loc["FINAL_pipeline", f"zero_overlap_rate@{k}"]),
             "Duplicate-text rate": f3(S.loc["FINAL_pipeline", f"duplicate_text_rate@{k}"]), "Unique templates": f"{S.loc['FINAL_pipeline', f'unique_templates@{k}']:.2f}",
             "Context words": f"{S.loc['FINAL_pipeline', f'context_words@{k}']:.0f}", "Oracle coverage@K": f3(S.loc["O_final_pipeline", f"union_coverage@{k}"])}
        t3.append(r)
    sel_row = {"K": f"Selected: K={K}"} | {c: "" for c in list(t3[0])[1:]}
    t3.append(sel_row)
    write_table("table3_top_k_comparison", pd.DataFrame(t3), "Top-K comparison for the selected classifier-query pipeline (hybrid RRF).")
    t4 = []
    base_n = f"B_{'hybrid_rrf' if dec['mmr']['base'] == 'hybrid' else 'dense'}"
    for n in [base_n] + [f"C_{dec['mmr']['base']}_mmr{l}" for l in (0.5, 0.7, 0.9)]:
        lam = "standard (no MMR)" if n == base_n else n.split("mmr")[1]
        chk = MM[MM["lambda"].astype(str) == lam] if n != base_n else None
        t4.append({"Setting": lam if n == base_n else f"MMR lambda={lam}", "Jaccard@3": f3(S.loc[n, "jaccard_truth@3"]), "Jaccard@5": f3(S.loc[n, "jaccard_truth@5"]), "nDCG@5": f3(S.loc[n, "ndcg@5"]),
                   "Coverage@3": f3(S.loc[n, "union_coverage@3"]), "Coverage@5": f3(S.loc[n, "union_coverage@5"]), "Duplicate rate@5": f3(S.loc[n, "duplicate_text_rate@5"]),
                   "Unique templates@5": f"{S.loc[n, 'unique_templates@5']:.2f}", "Pairwise cosine@5": f3(S.loc[n, "mean_pairwise_cosine@5"]),
                   "Duplicate-rate difference@5 (95% CI)": "reference" if chk is None else f"{chk[chk.K == 5].iloc[0].dup_mean_difference:+.4f} ({chk[chk.K == 5].iloc[0].dup_ci95_low:+.4f} to {chk[chk.K == 5].iloc[0].dup_ci95_high:+.4f})",
                   "Adopted": "n/a" if chk is None else ("yes" if dec["mmr"]["lambda"] is not None and str(dec["mmr"]["lambda"]) == lam else "no")})
    write_table("table4_diversity_mmr_analysis", pd.DataFrame(t4), "Standard hybrid retrieval versus MMR re-ranking (pool 30).")
    q = cfg["query_construction"]
    t5 = pd.DataFrame([("Status", "candidate; locked retrieval test unopened"), ("Embedding model; dimension", f"{cfg['embedding_model']['identifier']}; {cfg['embedding_model']['dimension']}"),
                       ("Query phrase set", q["phrase_set"]), ("Clinical phrase expansion selected", str(q["clinical_phrase_expansion_selected"])), ("Probability weighting selected", str(q["weighting_selected"])),
                       ("Top-N findings", str(q["top_n_findings"])), ("No Finding query phrase", q["no_finding_query_phrase"]), ("Lexical", f"BM25 k1={cfg['lexical']['k1']}, b={cfg['lexical']['b']}"),
                       ("Retriever", cfg["retriever"]["selected"]), ("Fusion", f"RRF k={cfg['retriever']['hybrid']['rrf_k']}, depth {cfg['retriever']['hybrid']['depth_per_ranker']}"),
                       ("MMR", "adopted" if cfg["diversity_reranking"]["mmr_selected"] else "not adopted"), ("Provisional Top-K", str(cfg["provisional_top_k"])),
                       ("Candidate Jaccard@K / nDCG@K / coverage@K", f"{f3(cfg['validation_performance_of_candidate']['jaccard@K'])} / {f3(cfg['validation_performance_of_candidate']['ndcg@K'])} / {f3(cfg['validation_performance_of_candidate']['union_coverage@K'])}")],
                      columns=["Parameter", "Value"])
    write_table("table5_final_r2_candidate_configuration", t5, "R2 retrieval candidate configuration (not frozen).")
    t6 = []
    for sysn, nm in (("R1_Q0_dense", "R1 Q0 dense baseline"), ("R2_final", "R2 selected pipeline")):
        g = FC[(FC.system == sysn) & FC.failure_definition.str.startswith("top-1")]
        t6.append({"System": nm, "Failures (top-1 not exact)": int(g.n_failures.iloc[0]), "Queries": int(g.n_queries.iloc[0]),
                   **{f"{c} {g[g.category == c].category_name.iloc[0].lower()}": f"{int(g[g.category == c].n.iloc[0])} ({g[g.category == c].pct_of_failures.iloc[0]:.1f}%)" for c in ("D", "A", "E", "F", "B", "C")}})
    write_table("table6_primary_failure_analysis", pd.DataFrame(t6), "Failure-category breakdown (percent of failures).")

    # ================================================================== Figure 1: query policies
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    y = np.arange(len(pol))
    f1 = []
    for ax, (m, lab) in zip(axes, (("jaccard_truth@3", "Finding Jaccard@3 vs study truth"), ("union_coverage@3", "Union coverage of true findings@3"))):
        mid, lo, hi = (S.loc[pol, m].to_numpy(), S.loc[pol, f"{m}__ci95_low"].to_numpy(), S.loc[pol, f"{m}__ci95_high"].to_numpy())
        ax.errorbar(mid, y, xerr=[mid - lo, hi - mid], fmt="o", color=C["orange"], ecolor=C["orange"], capsize=3, ms=5)
        ax.axvline(S.loc["R1_Q0_dense", m], color=C["grey"], lw=1, ls=":")
        ax.axvline(S.loc["O_Q0_dense", m], color=C["blue"], lw=1, ls="--")
        ax.text(S.loc["O_Q0_dense", m], -0.8, "oracle", color=C["blue"], fontsize=7, ha="center")
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        f1 += [{"metric": m, "config": n, "mean": a, "ci95_low": b, "ci95_high": c_} for n, a, b, c_ in zip(pol, mid, lo, hi)]
    axes[0].set_yticks(y, [LABEL[n] for n in pol])
    axes[0].invert_yaxis()
    fig.tight_layout()
    src("fig1_query_policy_comparison", pd.DataFrame(f1))
    save(fig, "fig1_query_policy_comparison")

    # ================================================================== Figure 2: dense vs BM25 vs hybrid across K
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    f2 = []
    for ax, (m, lab) in zip(axes, (("jaccard_truth", "Finding Jaccard@K"), ("ndcg", "nDCG@K"), ("union_coverage", "Union coverage@K"))):
        for n, colr, mk, off in (("B_dense", C["blue"], "o", -0.07), ("B_bm25", C["orange"], "s", 0.0), ("B_hybrid_rrf", C["green"], "D", 0.07)):
            mid = np.array([S.loc[n, f"{m}@{k}"] for k in KS])
            lo = np.array([S.loc[n, f"{m}@{k}__ci95_low"] for k in KS])
            hi = np.array([S.loc[n, f"{m}@{k}__ci95_high"] for k in KS])
            ax.errorbar(np.arange(4) + off, mid, yerr=[mid - lo, hi - mid], color=colr, marker=mk, ms=4.5, capsize=2.5, lw=1.3, label=LABEL[n] if m == "jaccard_truth" else None)
            f2 += [{"metric": m, "config": n, "K": k, "mean": a, "ci95_low": b, "ci95_high": c_} for k, a, b, c_ in zip(KS, mid, lo, hi)]
        ax.set_xticks(range(4), [f"K={k}" for k in KS])
        ax.set_ylabel(lab)
        ax.set_ylim(0.3, 1.0 if m == "union_coverage" else 0.8)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig2_retriever_comparison", pd.DataFrame(f2))
    save(fig, "fig2_dense_bm25_hybrid")

    # ================================================================== Figure 3: performance across K (selected pipeline)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    f3s = []
    for ax, (m, lab) in zip(axes, (("jaccard_truth", "Finding Jaccard@K"), ("union_coverage", "Union coverage@K"), ("duplicate_text_rate", "Duplicate-text rate@K"))):
        for n, colr, ls, nm in (("FINAL_pipeline", C["orange"], "-", "Selected pipeline (classifier query)"), ("O_final_pipeline", C["blue"], "--", "Same pipeline, oracle query")):
            mid = np.array([S.loc[n, f"{m}@{k}"] for k in KS])
            if f"{m}@1__ci95_low" in S.columns:
                lo = np.array([S.loc[n, f"{m}@{k}__ci95_low"] for k in KS])
                hi = np.array([S.loc[n, f"{m}@{k}__ci95_high"] for k in KS])
                ax.errorbar(range(4), mid, yerr=[mid - lo, hi - mid], color=colr, ls=ls, marker="o", ms=4.5, capsize=2.5, lw=1.3, label=nm if m == "jaccard_truth" else None)
            else:
                ax.plot(range(4), mid, color=colr, ls=ls, marker="o", ms=4.5, lw=1.3, label=nm if m == "jaccard_truth" else None)
            f3s += [{"metric": m, "config": n, "K": k, "value": v} for k, v in zip(KS, mid)]
        if m in ("union_coverage", "jaccard_truth"):
            ax.plot(range(4), [rnd[f"{m}@{k}"] for k in KS], color=C["grey"], ls=":", lw=1.2, label="Random (expected)" if m == "jaccard_truth" else None)
        ax.axvline(KS.index(K), color=C["green"], lw=6, alpha=0.18)
        ax.set_xticks(range(4), [f"K={k}" for k in KS])
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower left", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig3_performance_across_k", pd.DataFrame(f3s))
    save(fig, "fig3_performance_across_k")

    # ================================================================== Figure 4: relevance versus redundancy
    base_n = f"B_{'hybrid_rrf' if dec['mmr']['base'] == 'hybrid' else 'dense'}"
    pts = [("B_dense", "Dense", C["blue"], "o"), ("B_bm25", "BM25", C["orange"], "s"), ("B_hybrid_rrf", "Hybrid", C["green"], "D")] + \
          [(f"C_{dec['mmr']['base']}_mmr{l}", f"MMR lambda={l}", c_, mk_) for l, c_, mk_ in ((0.5, "#f2c14e", "^"), (0.7, "#d98e04", "v"), (0.9, "#8a5a00", "P"))]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex="col")
    f4 = []
    for r, k in enumerate((3, 5)):
        for c_, (m, lab) in enumerate((("jaccard_truth", f"Jaccard@{k}"), ("union_coverage", f"Coverage@{k}"))):
            ax = axes[r, c_]
            for n, nm, colr, mk in pts:
                x_, y_ = S.loc[n, f"mean_pairwise_cosine@{k}"], S.loc[n, f"{m}@{k}"]
                ax.scatter(x_, y_, color=colr, marker=mk, s=55, zorder=3, label=nm if (r == 0 and c_ == 0) else None, edgecolor="white", linewidth=0.6, alpha=0.9)
                f4.append({"K": k, "metric": m, "config": n, "mean_pairwise_cosine": x_, "value": y_, "duplicate_text_rate": S.loc[n, f"duplicate_text_rate@{k}"]})
            ax.set_ylabel(lab)
            if r == 1:
                ax.set_xlabel("Mean pairwise cosine among retrieved reports")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig4_relevance_vs_redundancy", pd.DataFrame(f4))
    save(fig, "fig4_relevance_vs_redundancy")

    # ================================================================== Figure 5: oracle vs final pipeline
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    f5 = []
    for ax, (m, lab) in zip(axes, (("jaccard_truth", "Finding Jaccard@K"), ("union_coverage", "Union coverage@K"))):
        for n, colr, ls, mk, nm in (("O_final_pipeline", C["blue"], "-", "o", "Oracle query, R2 pipeline"), ("O_Q0_dense", C["blue"], "--", "s", "Oracle query, R1 dense"),
                                     ("FINAL_pipeline", C["orange"], "-", "o", "Classifier query, R2 pipeline"), ("R1_Q0_dense", C["orange"], "--", "s", "Classifier query, R1 dense")):
            v = [S.loc[n, f"{m}@{k}"] for k in KS]
            ax.plot(range(4), v, color=colr, ls=ls, marker=mk, ms=4.5, lw=1.4, label=nm if m == "jaccard_truth" else None)
            f5 += [{"metric": m, "config": n, "K": k, "value": x} for k, x in zip(KS, v)]
        ax.set_xticks(range(4), [f"K={k}" for k in KS])
        ax.set_ylabel(lab)
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="center right", frameon=False, fontsize=7)
    fig.tight_layout()
    src("fig5_oracle_vs_final", pd.DataFrame(f5))
    save(fig, "fig5_oracle_vs_final_pipeline")

    # ================================================================== Figure 6: failure categories
    fig, ax = plt.subplots(figsize=(8.8, 2.9))
    cats = [("none", "No failure", "#c3c2b7"), ("D", "Normal/abnormal disagreement", C["orange"]), ("A", "Upstream classification/query error", C["yellow"]), ("E", "Corpus limitation", C["grey"]),
            ("F", "Duplicate/template issue", C["green"]), ("B", "Terminology mismatch", "#6a4fd0"), ("C", "Retriever failure despite good query", C["blue"])]
    f6 = []
    for i, (sysn, nm) in enumerate((("R1_Q0_dense", "R1 Q0 dense"), ("R2_final", "R2 selected pipeline"))):
        g = FC[(FC.system == sysn) & FC.failure_definition.str.startswith("top-1")]
        n_q, n_f = int(g.n_queries.iloc[0]), int(g.n_failures.iloc[0])
        left = 0.0
        for c_, lab, colr in cats:
            v = (n_q - n_f) / n_q if c_ == "none" else int(g[g.category == c_].n.iloc[0]) / n_q
            ax.barh(i, 100 * v, left=left, color=colr, height=0.55, label=lab if i == 0 else None)
            if v >= 0.04:
                ax.text(left + 50 * v, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7, color="white" if c_ in ("C", "B", "D") else "black")
            f6.append({"system": sysn, "category": c_, "percent_of_all_queries": 100 * v})
            left += 100 * v
    ax.set_yticks([0, 1], ["R1 Q0 dense", "R2 selected pipeline"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Queries (%); failure = top-1 report is not an exact finding-set match")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=7, ncol=3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    src("fig6_failure_categories", pd.DataFrame(f6))
    save(fig, "fig6_failure_categories")

    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(Image.open(p).info.get("dpi", (0, 0))[0] for p in FIG.glob("*.png")))}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
