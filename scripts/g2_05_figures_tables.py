"""G2 journal assets: 6 figures (PNG 300 dpi + vector PDF + SVG, with source data) and 7 main + 2 supplementary tables (CSV, Markdown, LaTeX).
Reads only artifacts written by g2_04_evaluate (and the frozen prompt/generator files). No titles inside plots.

    .venv\\Scripts\\python.exe -m scripts.g2_05_figures_tables
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import SERIES, apply_style
from src.utils.config import PROJECT_ROOT

G2 = PROJECT_ROOT / "results/report_generation/experiments/g2_multi_agent"
FIG, SRC, TAB = G2 / "figures", G2 / "figures/source_data", G2 / "tables"
B0S, G1S, G2DS, G2S = "B0_rule_based", "G1_single_agent_rag", "G2_agent2_draft_pre_critic", "G2_multi_agent_rag"
SHOW = (B0S, G1S, G2DS, G2S)
NAME = {B0S: "B0 rule-based", G1S: "G1 single-agent RAG", G2DS: "G2 Agent 2 draft (before critic)", G2S: "G2 multi-agent RAG (final)"}
COL = {B0S: SERIES[1], G1S: SERIES[0], G2DS: SERIES[3], G2S: SERIES[2]}
f3 = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
pct = lambda v: "n/a" if pd.isna(v) else f"{100 * v:.1f}%"  # noqa: E731


def main() -> int:
    for d in (FIG, SRC, TAB):
        d.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    rd = lambda n: pd.read_csv(G2 / n, float_precision="round_trip")  # noqa: E731
    jl = lambda n: json.loads((G2 / n).read_text(encoding="utf-8"))  # noqa: E731
    main_c, pair = rd("g2_main_comparison.csv").set_index("metric"), rd("g2_paired_differences.csv")
    pf, nad, dec = rd("g2_per_finding_results.csv"), rd("g2_normal_abnormal.csv"), rd("g2_report_state_given_classifier_state.csv")
    a1tab, byc, funnel, intro, cpy, cpd = rd("g2_agent1_status_vs_reference.csv"), rd("g2_agent1_support_count_vs_reference.csv"), rd("g2_retrieval_only_funnel.csv"), rd("g2_introduced_findings.csv"), rd("g2_copying_analysis.csv").set_index("system"), rd("g2_copying_paired_difference.csv")
    a1s, a3, lat, frozen, meta = jl("g2_agent1_summary.json"), jl("g2_agent3_summary.json"), jl("g2_latency_compute.json"), jl("g2_prompts_frozen.json"), jl("g2_generator_metadata.json")
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
        for j, s in enumerate(SHOW):
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
    h = frozen["prompt_sha256"]
    opt = meta["generation_options"]
    t1 = pd.DataFrame([
        ["Pipeline", "Agent 1 (Evidence Verifier) -> deterministic validation -> Agent 2 (Grounded Report Writer) -> Agent 3 (Grounding Critic); one pass, no loop"],
        ["Frozen upstream components", "DenseNet-121 classifier, C4 operating policy, No Finding rule, C5 Platt calibration, R2 retrieval (top-3 finding query, dense + BM25 RRF, Top-5)"],
        ["Generator (all agents)", f"{meta['model_tag']} (Ollama {meta['ollama_version']}, digest {meta['model_digest'][:12]}, {meta['model']['quantization_level']}), local only, no tools, no web"],
        ["Decoding", f"temperature {opt['temperature']}, top_k {opt['top_k']}, top_p {opt['top_p']}, seed {opt['seed']}, context {opt['num_ctx']} tokens"],
        ["Output budget (new tokens)", ", ".join(f"{a.split('_')[0]} {v['num_predict']}" for a, v in meta["agent_specific"].items()) + " (all agents use schema-constrained JSON output)"],
        ["Agent 1 input", "classifier-positive findings, calibrated probabilities, No Finding state, frozen Top-5 retrieved reports (rank and text only)"],
        ["Agent 1 output", "JSON: supporting ranks, short evidence summary and support status per classifier finding; retrieval_only candidates; ranks of normal-describing reports"],
        ["Deterministic validation", "valid ranks only; support count recomputed from ranks; status from count (>=2 supported, 1 partially supported, 0 unsupported); summaries reproducing retrieved prose replaced"],
        ["Agent 2 input", "classifier state, calibrated probabilities, validated Agent 1 evidence JSON; no retrieved report text"],
        ["Agent 3 input", "classifier state, validated Agent 1 evidence JSON, Agent 2 draft; no retrieved report text; APPROVE keeps the draft unchanged, otherwise one corrected report"],
        ["Prompt hash Agent 1", h["agent1_evidence_verifier"]], ["Prompt hash Agent 2", h["agent2_grounded_report_writer"]], ["Prompt hash Agent 3", h["agent3_grounding_critic"]], ["Combined prompt hash", h["combined"]]], columns=["Component", "Specification"])
    write_table("table1_g2_architecture_configuration", t1, "G2 architecture and frozen configuration.")
    T2 = [("Finding precision", "precision"), ("Finding recall", "recall"), ("Finding F1 (micro)", "f1"), ("Macro finding F1", "macro_f1"), ("Hallucination rate (reports with >=1 hallucinated finding)", "hallucination_rate"),
          ("Omission rate (reports with >=1 omitted finding)", "omission_rate"), ("Classifier FP propagation", "clf_fp_propagation"), ("Classifier TP retention", "tp_retention"), ("Normal recall (report state)", "normal_recall"),
          ("Abnormal recall (report state)", "abnormal_recall"), ("ROUGE-L", "rouge_l"), ("BLEU-4", "bleu4"), ("METEOR (exact-match variant)", "meteor_exact"), ("Mean report length (words)", "mean_words")]
    t2 = pd.DataFrame([{"Metric": n, **{NAME[s]: (f"{M(m, s)[0]:.1f} ({M(m, s)[1]:.1f} to {M(m, s)[2]:.1f})" if m == "mean_words" else cell(m, s)) for s in SHOW}} for n, m in T2])
    write_table("table2_g1_vs_g2_main_metrics", t2, "Main metrics for B0, G1, the G2 Agent 2 draft and the final G2 report on identical validation studies (95% bootstrap intervals).")
    T3 = T2 + [("Mean hallucinated findings per report", "mean_hallucinated"), ("Mean omitted findings per report", "mean_omitted"), ("Over-normalisation: report normal although classifier abnormal", "over_normalisation_given_classifier_abnormal"),
               ("Rare-finding TP retention", "rare_tp_retention"), ("Other-finding TP retention", "common_tp_retention")]
    t3 = pd.DataFrame([{"Metric": n, "G1": f3(D(G2S, G1S, m).b), "G2": f3(D(G2S, G1S, m).a), "G2 - G1 (95% CI)": dcell(D(G2S, G1S, m)), "CI excludes 0": "yes" if D(G2S, G1S, m).excludes_zero else "no",
                       "Draft - G1 (95% CI)": dcell(D(G2DS, G1S, m)), "G2 - draft (95% CI)": dcell(D(G2S, G2DS, m))} for n, m in T3])
    write_table("table3_paired_bootstrap_differences", t3, "Primary comparison G2 minus G1 and the contribution of the critic (G2 minus Agent 2 draft); paired study-level bootstrap, 1,000 resamples.")
    ag1 = a1tab[a1tab.status_source.str.startswith(("validated", "all"))]
    t4 = pd.DataFrame({"Quantity": ["Supported classifier positives", "Partially supported classifier positives", "Unsupported classifier positives", "All classifier positives (clinical subset)", "Model-stated status inconsistent with the cited ranks (validation corrected)",
                                    "Retrieval-only candidates proposed (clinical subset)", "  reference-supported", "  with >=3 supporting reports", "Citation agreement with IU labels of the retrieved report", "Recall of IU-labelled supporting reports",
                                    "Status agreement with the deterministic G1 support table"],
                       "Value": [f"{int(r.n_classifier_positives)} ({int(r.n_reference_positive)} match reference, {pct(r.share_matching_reference)})" for r in ag1.itertuples() if r.support_status in ("supported", "partially_supported", "unsupported", "all")] +
                                [str(a1s["validation_flag_counts"].get("agent1_status_inconsistent_with_count", 0)), str(a1s["n_retrieval_only_candidates_clinical"]), f"{a1s['retrieval_only_candidates_reference_supported']} ({pct(a1s['retrieval_only_candidates_reference_supported'] / max(a1s['n_retrieval_only_candidates_clinical'], 1))})",
                                 str(a1s["candidate_ge_promotion_threshold_clinical"]), pct(a1s["citation_agreement_with_iu_labels"]), pct(a1s["recall_of_iu_labelled_support"]), pct(a1s["status_agreement_with_deterministic_g1_support_table"])]})
    write_table("table4_agent1_evidence_analysis", t4, "Agent 1 evidence analysis: support status of classifier positives against the reference, retrieval-only candidates and agreement with the IU labels of the retrieved reports.")
    c3 = a3["clinical_subset"]
    t5 = pd.DataFrame({"Quantity": ["Reports approved unchanged", "Reports modified by the critic", "Critic actions (all studies)", "Findings removed by critic (clinical subset)", "  correct removals (not in reference)", "  incorrect removals (in reference)",
                                    "Findings added by critic (clinical subset)", "  correct additions (in reference)", "  unsupported additions (not in reference)", "Retrieval-only candidates proposed by Agent 1", "  promoted into the Agent 2 draft", "  surviving the critic (final report)"],
                       "Value": [f"{a3['approved_unchanged_pct']:.1f}%", f"{a3['modified_pct']:.1f}% ({a3['n_modified']} of {a3['n_studies']})", ", ".join(f"{k} {v}" for k, v in a3["actions"].items()), str(c3["findings_removed"]),
                                 str(c3["correct_removals_(not_in_reference)"]), str(c3["incorrect_removals_(in_reference)"]), str(c3["findings_added"]), str(c3["correct_additions_(in_reference)"]), str(c3["unsupported_additions_(not_in_reference)"])] +
                                [f"{int(r.n_candidates)} ({int(r.reference_supported)} reference-supported, {int(r.reference_unsupported)} unsupported)" for r in funnel.itertuples()]})
    write_table("table5_agent3_intervention_analysis", t5, "Agent 3 (critic) interventions and the fate of retrieval-only candidates through the pipeline.")
    t6 = pf.pivot(index="finding", columns="system", values=["n_truth", "classifier_true_positives", "tp_retention", "precision", "recall", "f1"]).reset_index()
    t6.columns = ["finding"] + [f"{a}|{b}" for a, b in t6.columns[1:]]
    order = list(pf[pf.system == B0S].finding)
    t6 = t6.set_index("finding").loc[order].reset_index()
    t6 = pd.DataFrame({"Finding": t6.finding + np.where(pf[pf.system == B0S].set_index("finding").loc[order].rare_c2.to_numpy(), " (rare)", ""), "Reference positives": t6[f"n_truth|{B0S}"].astype(int),
                       "Classifier TPs": t6[f"classifier_true_positives|{B0S}"].astype(int), **{f"TP retention {NAME[s]}": t6[f"tp_retention|{s}"].map(f3) for s in (B0S, G1S, G2S)},
                       **{f"P / R / F1 {NAME[s]}": [f"{f3(a)} / {f3(b)} / {f3(c)}" for a, b, c in zip(t6[f"precision|{s}"], t6[f"recall|{s}"], t6[f"f1|{s}"])] for s in (G1S, G2S)}})
    write_table("table6_per_finding_results", t6, "Per-finding results on the clinical subset for B0, G1 and G2; rare findings follow the C2 definition.")
    ag = {a: lat["agents"][a] for a in lat["agents"]}
    rows7 = [["Agent 1 mean seconds / call (n calls)", f"{ag['agent1_evidence_verifier']['mean_seconds']:.2f} ({ag['agent1_evidence_verifier']['n_calls']})"], ["Agent 2 mean seconds / call", f"{ag['agent2_grounded_report_writer']['mean_seconds']:.2f}"],
             ["Agent 3 mean seconds / call", f"{ag['agent3_grounding_critic']['mean_seconds']:.2f}"], ["G2 mean seconds / study", f"{lat['g2_mean_seconds_per_study']:.2f}"], ["G1 mean seconds / study", f"{lat['g1_mean_seconds_per_study']:.2f}"],
             ["G2 / G1 time ratio", f"{lat['g2_over_g1_time_ratio']:.2f}"], ["G2 total recorded call time (minutes)", f"{lat['g2_total_recorded_call_seconds'] / 60:.1f}"], ["G1 total recorded call time (minutes)", f"{lat['g1_total_recorded_call_seconds'] / 60:.1f}"],
             ["Mean output tokens Agent 1 / 2 / 3", " / ".join(f"{ag[a]['mean_output_tokens']:.0f}" for a in ag)], ["Mean prompt tokens Agent 1 / 2 / 3", " / ".join(f"{ag[a]['mean_prompt_tokens']:.0f}" for a in ag)],
             ["Peak GPU memory used (MiB, sampled)", f"{lat['gpu_memory_used_mib_peak_sampled']:.0f}"]]
    for lab, key in (("Copied-sentence rate from Top-5", "copied_sentence_rate_from_top5"), ("Whole-report copies of Top-5", "whole_report_copy_rate_from_top5"), ("Repeated-sentence rate", "repeated_sentence_rate")):
        r = cpd[(cpd.comparison == f"{G2S} minus {G1S}") & (cpd.metric == key)].iloc[0]
        rows7.append([f"{lab}: G1 / G2 (G2 - G1, 95% CI)", f"{pct(r.b)} / {pct(r.a)} ({100 * r['diff']:+.1f} points, {100 * r.ci95_low:+.1f} to {100 * r.ci95_high:+.1f})"])
    rows7 += [["Sentences found verbatim in the corpus: G1 / G2", f"{pct(cpy.loc[G1S, 'corpus_copied_sentence_rate'])} / {pct(cpy.loc[G2S, 'corpus_copied_sentence_rate'])}"],
              ["Reports duplicating a whole corpus report: G1 / G2", f"{pct(cpy.loc[G1S, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])} / {pct(cpy.loc[G2S, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])}"]]
    write_table("table7_latency_and_copying_analysis", pd.DataFrame(rows7, columns=["Quantity", "Value"]), "Latency, compute and copying analysis for G1 and G2.")
    na_rows = []
    for who, nm in (("classifier", "Classifier (= B0 report state)"), (G1S, NAME[G1S]), (G2DS, NAME[G2DS]), (G2S, NAME[G2S])):
        nn = nad[(nad.source == who)].set_index("reference")
        k = B0S if who == "classifier" else who
        na_rows.append({"Source": nm, "Normal recall": cell("normal_recall", k), "Abnormal recall": cell("abnormal_recall", k), "Abnormal studies reported normal": f"{int(nn.loc['abnormal', 'predicted_normal'])} of {int(nn.loc['abnormal', 'n'])}",
                        "Normal studies reported abnormal": f"{int(nn.loc['normal', 'predicted_abnormal'])} of {int(nn.loc['normal', 'n'])}", "Report normal given classifier abnormal": "n/a" if who == "classifier" else cell("over_normalisation_given_classifier_abnormal", k),
                        "Rare-finding TP retention": "n/a" if who == "classifier" else f3(M("rare_tp_retention", k)[0]), "Other-finding TP retention": "n/a" if who == "classifier" else f3(M("common_tp_retention", k)[0])})
    write_table("table8_supplementary_normal_abnormal_and_rare_findings", pd.DataFrame(na_rows), "Supplementary: normal/abnormal consistency and rare-finding retention.")
    st = rd("g2_stratified_by_classifier_state.csv")
    t9 = pd.DataFrame({"Classifier state": st.classifier_state, "Studies": st.n_studies, "Metric": st.metric, "G1": st[G1S].map(f3), "G2": st[G2S].map(f3), "G2 - G1 (95% CI)": [f"{a:+.3f} ({b:+.3f} to {c:+.3f})" for a, b, c in zip(st.g2_minus_g1, st.ci95_low, st.ci95_high)],
                       "CI excludes 0": np.where(st.excludes_zero, "yes", "no")})
    write_table("table9_supplementary_stratified_by_classifier_state", t9, "Supplementary: G2 minus G1 within each classifier state (descriptive).")

    # ================================================================== figures
    leg = lambda ax: ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    f1 = grouped(ax, ["precision", "recall", "f1"], ["Precision", "Recall", "F1 (micro)"], "Finding agreement with the IU reference", (0, 0.8))
    leg(ax)
    fig.tight_layout()
    src("fig1_finding_prf", pd.DataFrame(f1))
    save(fig, "fig1_g1_vs_g2_finding_precision_recall_f1")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw={"wspace": 0.25})
    f2 = grouped(axes[0], ["hallucination_rate", "omission_rate"], ["Reports with >=1\nhallucinated finding", "Reports with >=1\nomitted finding"], "Share of reports", (0, 0.8))
    f2 += grouped(axes[1], ["mean_hallucinated", "mean_omitted"], ["Hallucinated findings\nper report", "Omitted findings\nper report"], "Findings per report", (0, 1.0), legend=False)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="upper center", ncol=2, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    src("fig2_hallucination_omission", pd.DataFrame(f2))
    save(fig, "fig2_hallucination_and_omission")
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    f3d = grouped(ax, ["clf_fp_propagation", "tp_retention"], ["Classifier false positives\nmentioned in report", "Classifier true positives\nretained in report"], "Proportion", (0, 1.1))
    leg(ax)
    fig.tight_layout()
    src("fig3_fp_propagation_tp_retention", pd.DataFrame(f3d))
    save(fig, "fig3_fp_propagation_and_tp_retention")
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    f4 = grouped(ax, ["normal_recall", "abnormal_recall"], ["Normal recall", "Abnormal recall"], "Share of reference studies with the correct state", (0, 1.05))
    leg(ax)
    ax.text(0.5, -0.2, "B0 report state equals the classifier state by construction", transform=ax.transAxes, ha="center", fontsize=7)
    fig.tight_layout()
    src("fig4_normal_abnormal_recall", pd.DataFrame(f4))
    save(fig, "fig4_normal_vs_abnormal_recall")
    # Figure 5: Agent 1 evidence-support categories
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    f5 = []
    for i, r in enumerate(ag1[ag1.support_status != "all"].itertuples()):
        ax.bar(i, r.n_reference_positive, color=SERIES[2], label="Matches reference" if i == 0 else None)
        ax.bar(i, r.n_classifier_positives - r.n_reference_positive, bottom=r.n_reference_positive, color=SERIES[3], label="Not in reference" if i == 0 else None)
        ax.text(i, r.n_classifier_positives, f"n={int(r.n_classifier_positives)}\n{pct(r.share_matching_reference)} match", ha="center", va="bottom", fontsize=7)
        f5.append({"panel": "status", "category": r.support_status, "n": int(r.n_classifier_positives), "n_reference_positive": int(r.n_reference_positive)})
    ax.set_xticks(range(3), ["Supported", "Partially\nsupported", "Unsupported"])
    ax.set_ylabel("Classifier positives (clinical subset)")
    ax.set_ylim(0, max(ag1[ag1.support_status != "all"].n_classifier_positives) * 1.25)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for i, r in enumerate(funnel.itertuples()):
        ax.bar(i, r.reference_supported, color=SERIES[2], label="Reference-supported" if i == 0 else None)
        ax.bar(i, r.reference_unsupported, bottom=r.reference_supported, color=SERIES[3], label="Unsupported" if i == 0 else None)
        ax.text(i, r.n_candidates, f"{int(r.n_candidates)}", ha="center", va="bottom", fontsize=8)
        f5.append({"panel": "retrieval_only_funnel", "category": r.stage, "n": int(r.n_candidates), "n_reference_positive": int(r.reference_supported)})
    ax.set_xticks(range(3), ["Proposed by\nAgent 1", "Promoted into\nAgent 2 draft", "Surviving\nAgent 3"])
    ax.set_ylabel("Retrieval-only candidate findings")
    ax.set_ylim(0, max(funnel.n_candidates) * 1.2)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    src("fig5_agent1_evidence_categories", pd.DataFrame(f5))
    save(fig, "fig5_agent1_evidence_support_categories")
    # Figure 6: critic actions and copying
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    f6 = []
    cats = [("Removed:\nnot in reference", c3["correct_removals_(not_in_reference)"], SERIES[2]), ("Removed:\nin reference", c3["incorrect_removals_(in_reference)"], SERIES[3]), ("Added:\nin reference", c3["correct_additions_(in_reference)"], SERIES[2]),
            ("Added:\nnot in reference", c3["unsupported_additions_(not_in_reference)"], SERIES[3])]
    for i, (lab, v, colr) in enumerate(cats):
        ax.bar(i, v, color=colr)
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
        f6.append({"panel": "critic_changes", "category": lab.replace("\n", " "), "value": int(v)})
    ax.set_xticks(range(4), [c[0] for c in cats])
    ax.set_ylabel("Findings added or removed by the critic (clinical subset)")
    ax.set_ylim(0, max(max(c[1] for c in cats), 1) * 1.25)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    keys = [("copied_sentence_rate_from_top5 (G1 definition)", "Sentences copied\nfrom Top-5"), ("whole_report_copy_rate_from_top5 (G1 definition)", "Whole-report copies\nof Top-5"), ("repeated_sentence_rate", "Repeated-sentence\nrate")]
    for j, s in enumerate((G1S, G2DS, G2S)):
        vals = [cpy.loc[s, k] for k, _ in keys]
        ax.bar(np.arange(3) + (j - 1) * 0.27, vals, width=0.25, color=COL[s], label=NAME[s])
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * 0.27, v, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=6.5)
            f6.append({"panel": "copying", "system": s, "category": keys[i][1].replace("\n", " "), "value": float(v)})
    ax.set_xticks(range(3), [l for _, l in keys])
    ax.set_ylabel("Share")
    ax.set_ylim(0, max(cpy.loc[G1S, keys[0][0]], 0.05) * 1.35)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    src("fig6_critic_actions_and_copying", pd.DataFrame(f6))
    save(fig, "fig6_critic_actions_and_copying_reduction")
    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(Image.open(x).info.get("dpi", (0, 0))[0] for x in FIG.glob("*.png")))}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
