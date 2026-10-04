"""F1 final-test assets: 8 figures (PNG 300 dpi + PDF + SVG + source data) and 10 tables + 2 supplementary (CSV, Markdown, LaTeX). They only FORMAT stored values
produced by the frozen analysis code; no metric is computed or changed here. No titles inside plots.

    .venv\\Scripts\\python.exe -m scripts.f1_07_figures_tables            (locked test)
    .venv\\Scripts\\python.exe -m scripts.f1_07_figures_tables validation_dry_run   (format check on the validation dry run)
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.plotting import SERIES, apply_style
from src.f1.pipeline import F1

f3 = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"  # noqa: E731
pct = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}%"  # noqa: E731
ci = lambda a, b, c: f"{f3(a)} ({f3(b)} to {f3(c)})"  # noqa: E731
CATNAME = {"1_classifier_error": "Classifier error", "2_normal_abnormal_classifier_mismatch": "Normal/abnormal classifier mismatch", "3_retrieval_query_mismatch": "Retrieval/query mismatch", "4_retrieval_only_unsupported_finding": "Retrieval-only unsupported finding",
           "5_report_generator_omission": "Report-generator omission", "6_report_generator_hallucination": "Report-generator hallucination", "7_abnormal_routing_normal_prose": "Abnormal routing, normal prose", "8_indeterminate_abstention": "Indeterminate / abstention",
           "9_corpus_limitation": "Corpus limitation (flag)", "10_reference_label_limitation": "Reference-label limitation (flag)"}


def main(test_dir: str = "test") -> int:
    dry = test_dir != "test"
    T, V = F1 / test_dir, F1 / "validation_dry_run"
    base = (V / "assets_format_check") if dry else F1
    FIG, SRC, TAB = base / "figures", base / "figures/source_data", base / "tables"
    for d in (FIG, SRC, TAB):
        d.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    t, v = json.loads((T / "f1_summary.json").read_text(encoding="utf-8")), json.loads((V / "f1_summary.json").read_text(encoding="utf-8"))
    cpt, cpv = pd.read_csv(T / "classification_per_class.csv"), pd.read_csv(V / "classification_per_class.csv")
    rt, rv = pd.read_csv(T / "retrieval_metrics_by_k.csv"), pd.read_csv(V / "retrieval_metrics_by_k.csv")
    relt, relv = pd.read_csv(T / "classification_reliability_pooled_bins.csv"), pd.read_csv(V / "classification_reliability_pooled_bins.csv")
    pft, fa = pd.read_csv(T / "report_per_finding_results.csv"), pd.read_csv(T / "failure_analysis.csv")
    prov, cmp_ = pd.read_csv(T / "provenance.csv"), pd.read_csv(F1 / ("validation_to_test_comparison.csv" if not dry else "dry_run_comparison_validation_vs_validation.csv"))
    abf = pd.read_csv(T / "abstention_finding_distribution.csv")
    TS, VS = t["report"], v["report"]
    COLT, COLV = SERIES[0], SERIES[3]

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
        body = "\n".join(" & ".join(esc(x) for x in r) + r" \\" for r in df.itertuples(index=False))
        return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n\\hline\n" + " & ".join(esc(c) for c in df.columns) + " \\\\\n\\hline\n" + body
                + "\n\\hline\n\\end{tabular}\n" + f"\\caption{{{esc(cap)}}}\n\\label{{{lab}}}\n\\end{{table}}\n")

    def write_table(name, df, cap):
        (TAB / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        (TAB / f"{name}.md").write_text(md_table(df) + "\n", encoding="utf-8")
        (TAB / f"{name}.tex").write_text(tex_table(df, cap, f"tab:{name}"), encoding="utf-8")

    # ============================================================================ tables
    fl = t["population_flow"]
    t1 = pd.DataFrame([["Locked test studies", fl["locked_or_validation_studies"]], ["With a frontal image", fl["with_frontal_image"]], ["Without a frontal image (not evaluable end to end)", fl["without_frontal_image"]], ["Failing the image validator", fl["failing_image_validator"]],
                       ["Classified (P2)", fl["classified_P2"]], ["Classified with a usable reference report: primary end-to-end set (P3)", fl["primary_end_to_end_set_P3_(classified_and_usable_reference_report)"]],
                       ["Classified without a usable reference report", fl["classified_without_usable_reference_report"]], ["Clinical-finding subset with usable mapped truth (P4)", fl["clinical_finding_subset_P4"]], ["  of which reference-normal", fl["reference_normal_in_P4"]],
                       ["Classification population (classified with mapped truth)", fl["classification_population_(classified_with_mapped_truth)"]], ["Retrieval population (non-empty classifier query)", fl["retrieval_primary_population_(non_empty_classifier_query)"]],
                       ["Routed to Path A / normal (P3)", fl["routing_counts_on_P3"]["normal"]], ["Routed to Path B / abnormal (P3)", fl["routing_counts_on_P3"]["abnormal"]], ["Routed to Path C / indeterminate (P3)", fl["routing_counts_on_P3"]["indeterminate"]],
                       ["Routed normal / abnormal / indeterminate (P2)", " / ".join(str(fl["routing_counts_on_P2"].get(k, 0)) for k in ("normal", "abnormal", "indeterminate"))]], columns=["Population step", "Studies"])
    write_table("table1_locked_test_population", t1, "Locked test population flow and routing counts (no study excluded silently).")
    cc, cp = t["classification"]["ci95"], t["classification"]["point"]
    rows2 = [[lab, ci(cc[k]["point"], cc[k]["ci95_low"], cc[k]["ci95_high"])] for k, lab in (("macro_auroc", "Macro AUROC"), ("macro_auprc", "Macro AUPRC"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"), ("macro_specificity", "Macro specificity"),
                                                                                                ("macro_f1", "Macro F1"), ("macro_balanced_accuracy", "Macro balanced accuracy"), ("macro_brier", "Brier score (macro)"), ("macro_log_loss", "Log loss (macro)"), ("macro_ece", "ECE (macro)"))]
    rows2 = [["Micro AUROC", f3(cp["micro_auroc"])], ["Micro AUPRC", f3(cp["micro_auprc"])]] + rows2
    rows2.append(["Classes with defined AUROC / undefined", f"{t['classification']['n_classes_defined']} / {', '.join(t['classification']['undefined_classes']) or 'none'}"])
    rows2.append(["Studies (classification population)", t["classification"]["n_studies"]])
    write_table("table2_final_classifier_metrics", pd.DataFrame(rows2, columns=["Metric", "Frozen classifier on the locked test (95% study-level bootstrap CI)"]), "Final classifier metrics on the locked end-to-end test population (IU X-Ray, MeSH-mapped labels).")
    t3 = cpt[["observation", "support", "auroc", "auprc", "precision", "recall", "specificity", "f1"]].copy()
    t3.columns = ["Observation", "Support (test positives)", "AUROC", "AUPRC", "Precision", "Recall", "Specificity", "F1"]
    for c in t3.columns[2:]:
        t3[c] = t3[c].map(f3)
    write_table("table3_per_class_classifier_metrics", t3, "Per-class classifier metrics on the locked test; undefined values (no positives) are shown as n/a.")
    t4 = rt[rt.metric.isin(["jaccard_truth", "ndcg", "union_coverage", "hit", "duplicate_text_rate"])].copy()
    t4 = pd.DataFrame({"K": t4.K, "Metric": t4.metric, "Classifier query (95% CI)": [ci(a, b, c) for a, b, c in zip(t4.classifier_query, t4.classifier_ci95_low, t4.classifier_ci95_high)],
                       "Oracle query (95% CI)": [ci(a, b, c) for a, b, c in zip(t4.oracle_query, t4.oracle_ci95_low, t4.oracle_ci95_high)], "Gap, classifier minus oracle (95% CI)": [ci(a, b, c) for a, b, c in zip(t4.gap_classifier_minus_oracle, t4.gap_ci95_low, t4.gap_ci95_high)]})
    write_table("table4_final_retrieval_metrics", t4, f"Frozen retrieval on the locked test (K = 1, 3, 5, 10; primary K = 5); oracle diagnostic only; {int(rt.n_queries.iloc[0])} queries.")
    m = TS["metrics"]
    nice = {"precision": "Finding precision", "recall": "Finding recall", "f1": "Finding F1 (micro)", "macro_f1": "Macro finding F1", "hallucination_rate": "Hallucination rate", "omission_rate": "Omission rate", "clf_fp_propagation": "Classifier FP propagation",
            "tp_retention": "Classifier TP retention", "rouge_l": "ROUGE-L", "bleu4": "BLEU-4", "meteor_exact": "METEOR (exact-match variant)", "mean_words": "Mean report length (words)"}
    t5 = pd.DataFrame([[nice[k], ci(m[k]["value"], m[k]["ci95_low"], m[k]["ci95_high"]) if k != "mean_words" else f"{m[k]['value']:.1f} ({m[k]['ci95_low']:.1f} to {m[k]['ci95_high']:.1f})"] for k in nice], columns=["Metric", "Final system on the locked test (95% CI)"])
    t5.loc[len(t5)] = ["Studies (clinical subset / primary set)", f"{TS['n_clinical_P4']} / {TS['n_primary_P3_with_final_report']}"]
    t5.loc[len(t5)] = ["Persistent generation failures", TS["n_generation_failures"]]
    write_table("table5_final_report_generation_metrics", t5, "Final report-generation metrics on the locked test (frozen G1 with the deterministic guard).")
    th, ab, pr = TS["three_state"], TS["abstention"], TS["routing_prose_consistency"]
    rn, ra = (next(r for r in th["reference_stratified"] if r["reference_state"] == k) for k in ("normal", "abnormal"))
    dd = th["decided"]
    rows6 = [["System states, primary set P3 (normal / abnormal / indeterminate)", " / ".join(f"{th['state_counts_P3'][k]} ({100 * th['state_counts_P3'][k] / TS['n_primary_P3_with_final_report']:.1f}%)" for k in ("normal", "abnormal", "indeterminate"))],
             ["System states, clinical subset P4", " / ".join(f"{th['state_counts_P4'][k]} ({100 * th['state_counts_P4'][k] / TS['n_clinical_P4']:.1f}%)" for k in ("normal", "abnormal", "indeterminate"))],
             [f"Reference-normal studies (n = {rn['n']}): routed normal / abnormal / indeterminate", " / ".join(f"{rn[f'routed_{k}_pct']:.1f}%" for k in ("normal", "abnormal", "indeterminate"))],
             [f"Reference-abnormal studies (n = {ra['n']}): routed normal / abnormal / indeterminate", " / ".join(f"{ra[f'routed_{k}_pct']:.1f}%" for k in ("normal", "abnormal", "indeterminate"))],
             ["Normal recall among decided studies (95% CI)", ci(dd["normal_recall_among_decided"]["value"], dd["normal_recall_among_decided"]["ci95_low"], dd["normal_recall_among_decided"]["ci95_high"])],
             ["Abnormal recall among decided studies (95% CI)", ci(dd["abnormal_recall_among_decided"]["value"], dd["abnormal_recall_among_decided"]["ci95_low"], dd["abnormal_recall_among_decided"]["ci95_high"])],
             [f"Decision coverage, decided / eligible ({th['decided_n']} of {th['eligible_n']}) (95% CI)", ci(dd["decision_coverage"]["value"], dd["decision_coverage"]["ci95_low"], dd["decision_coverage"]["ci95_high"])],
             ["Abstained studies P3 / P4", f"{ab['n_abstained_P3']} ({ab['pct_abstained_P3']:.1f}%) / {ab['n_abstained_P4']} ({ab['pct_abstained_P4']:.1f}%)"],
             ["Abstained with reference: reference-normal / reference-abnormal", f"{ab['reference_normal_among_abstained']} / {ab['reference_abnormal_among_abstained']}"],
             ["Abnormal-routing -> normal prose (P3), 95% CI", f"{pr['abnormal_routed_P3']['prose_normal']} of {pr['abnormal_routed_P3']['n']} ({pr['abnormal_routing_to_normal_prose_rate_P3']:.1f}%; {100 * pr['abnormal_routing_to_normal_prose_rate_ci95'][0]:.1f}% to {100 * pr['abnormal_routing_to_normal_prose_rate_ci95'][1]:.1f}%)"]]
    write_table("table6_three_state_system_metrics", pd.DataFrame(rows6, columns=["Quantity", "Locked test"]), "Three-state (normal / abnormal / indeterminate) system results, decided-case recalls, coverage, abstention and routing/prose consistency.")
    t7 = pft[pft.rare_c2][["finding", "support_test_truth_positive", "precision", "recall", "f1", "classifier_true_positives", "classifier_tp_retained", "tp_retention"]].copy()
    t7.columns = ["Rare finding", "Test support", "Precision", "Recall", "F1", "Classifier TPs", "TPs retained in report", "TP retention"]
    for c in ("Precision", "Recall", "F1", "TP retention"):
        t7[c] = t7[c].map(f3)
    write_table("table7_rare_finding_analysis", t7, "Rare findings (C2 definition) on the locked test; counts are very small, no strong claim is made.")
    t8 = cmp_[["group", "metric", "validation", "test", "test_minus_validation"]].copy()
    for c in ("validation", "test", "test_minus_validation"):
        t8[c] = t8[c].map(f3)
    t8.columns = ["Group", "Metric", "Validation", "Test", "Test minus validation"]
    write_table("table8_validation_to_test_comparison", t8, "Descriptive validation-to-test comparison (same code, IU validation dry run versus locked test); not used for tuning.")
    t9 = pd.DataFrame({"Category": [CATNAME.get(c, c.replace("_", " ")) for c in fa.category], "Studies": fa.n_studies, "Share of clinical subset": [f"{x:.1f}%" for x in fa.pct_of_clinical_subset]})
    write_table("table9_failure_analysis", t9, "Frozen failure taxonomy on the locked clinical subset (multi-label; categories 9 and 10 are limitation flags).")
    rtm = t["runtime"]
    rows10 = [["Classifier inference: model load / image loading and preprocessing / inference / total (s)", " / ".join(f"{rtm['classifier'][k]:.1f}" for k in ("model_load", "image_loading_and_preprocessing", "model_inference", "total"))],
              ["Query construction and hybrid retrieval, all studies (s); mean per study (s)", f"{rtm['prepare']['case_building_incl_query_embedding_and_hybrid_retrieval_seconds']:.1f}; {rtm['prepare']['mean_query_and_retrieval_seconds_per_study']:.3f}"]]
    if "generation" in rtm:
        g = rtm["generation"]
        rows10 += [["G1 generation: reports generated; mean / median seconds per report", f"{g['n_generated']}; {g['mean_seconds_per_report']:.2f} / {g['median_seconds_per_report']:.2f}"], ["G1 generation total call time (min); wall time of the run (min)", f"{g['total_call_seconds'] / 60:.1f}; {g['wall_seconds_this_run'] / 60:.1f}"],
                   ["Mean prompt / output tokens per report", f"{g['mean_prompt_tokens']:.0f} / {g['mean_output_tokens']:.0f}"], ["Total end-to-end compute (min)", f"{rtm['total_end_to_end_seconds'] / 60:.1f}"], ["Sampled peak GPU memory used (MiB)", f"{g['gpu_memory_used_mib_peak_sampled']:.0f}"],
                   ["Ollama / model / digest", f"{g['ollama_version']} / {g['model_tag']} / {g['model_digest'][:12]}"]]
    write_table("table10_runtime_compute", pd.DataFrame(rows10, columns=["Quantity", "Value"]), "Runtime and compute summary of the locked run (local RTX 3050 6 GB GPU).")
    pv = prov.copy()
    pv = pd.DataFrame({"Provenance of stated finding": pv.provenance, "Stated findings": pv.stated_findings, "Reference-supported": pv.reference_supported, "Unsupported": pv.reference_unsupported, "Share supported": pv.share_supported.map(f3)})
    write_table("table11_supplementary_provenance", pv, "Provenance of stated findings (classifier only, retrieval only, both, neither) and reference support, locked clinical subset.")
    cp_ = TS["copying"]
    write_table("table12_supplementary_copying", pd.DataFrame([["Copied-sentence rate against the study's own Top-5 (studies with context)", f"{pct(cp_['copied_sentence_rate_mean'])} ({pct(cp_['copied_sentence_rate_ci95'][0])} to {pct(cp_['copied_sentence_rate_ci95'][1])})"],
                                                              ["Whole-report copy rate", f"{pct(cp_['whole_report_copy_rate'])} ({pct(cp_['whole_report_copy_rate_ci95'][0])} to {pct(cp_['whole_report_copy_rate_ci95'][1])})"],
                                                              ["Repeated-sentence rate (all studies)", pct(cp_["repeated_sentence_rate_mean_P3"])], ["Studies with context", cp_["n_studies_with_context"]]], columns=["Quantity", "Locked test"]), "Copying analysis with the frozen G1 definitions.")

    # ============================================================================ figures
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    order = list(cpt.observation)
    yy = np.arange(len(order))
    for ax, col, lab in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        vals = cpt[col].to_numpy(float)
        ax.barh(yy, np.nan_to_num(vals), color=COLT)
        for i, x in enumerate(vals):
            ax.text(0.01 if np.isnan(x) else x + 0.01, i, "undefined" if np.isnan(x) else f"{x:.2f}", va="center", fontsize=7, color="#555")
        ax.set_xlabel(lab)
        ax.set_xlim(0, 1.15)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yy, [f"{o} (n={int(s)})" for o, s in zip(order, cpt.support)])
    axes[0].invert_yaxis()
    fig.tight_layout()
    src("fig1_per_class_auroc_auprc", cpt[["observation", "support", "auroc", "auprc"]])
    save(fig, "fig1_per_class_classifier_auroc_auprc")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#898781", lw=1, ls="--")
    ax.plot(relt.mean_predicted, relt.observed_fraction, "o-", color=COLT, label="Locked test", ms=4)
    ax.plot(relv.mean_predicted, relv.observed_fraction, "s-", color=COLV, label="Validation (dry run)", ms=3.5, alpha=0.8)
    ax.set_xlabel("Mean calibrated probability (equal-frequency bin)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_xlim(0, max(0.5, relt.mean_predicted.max() * 1.1))
    ax.set_ylim(0, max(0.5, relt.observed_fraction.max() * 1.1))
    ax.legend(frameon=False)
    ax = axes[1]
    ax.bar(np.arange(len(cpt)), cpt.ece, color=COLT)
    ax.set_xticks(np.arange(len(cpt)), [o[:14] for o in cpt.observation], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("ECE (adaptive bins) per class")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    src("fig2_reliability_pooled", relt)
    src("fig2_ece_per_class", cpt[["observation", "ece", "brier"]])
    save(fig, "fig2_calibration_reliability")
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.6))
    for ax, (m_, lab) in zip(axes, (("jaccard_truth", "Jaccard (truth)"), ("ndcg", "nDCG"), ("union_coverage", "Finding coverage"), ("hit", "Hit (exact match)"))):
        sub = rt[rt.metric == m_].sort_values("K")
        ax.errorbar(sub.K, sub.classifier_query, yerr=[np.maximum(0, sub.classifier_query - sub.classifier_ci95_low), np.maximum(0, sub.classifier_ci95_high - sub.classifier_query)], fmt="o-", color=COLT, capsize=2.5, label="Classifier query")
        ax.errorbar(sub.K + 0.12, sub.oracle_query, yerr=[np.maximum(0, sub.oracle_query - sub.oracle_ci95_low), np.maximum(0, sub.oracle_ci95_high - sub.oracle_query)], fmt="s--", color=COLV, capsize=2.5, label="Oracle query (diagnostic)")
        ax.axvline(5, color="#898781", lw=0.8, ls=":")
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xlabel("K")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    src("fig3_retrieval_k_analysis", rt)
    save(fig, "fig3_retrieval_k_analysis")
    cl_rows = [(k, lab) for k, lab in (("macro_auroc", "Macro AUROC"), ("macro_auprc", "Macro AUPRC"), ("macro_f1", "Macro F1"), ("macro_brier", "Brier (macro)"), ("macro_ece", "ECE (macro)"))]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    src4 = []
    for j, (k, lab) in enumerate(cl_rows):
        for off, (S, col, nm) in zip((-0.2, 0.2), ((v, COLV, "Validation (dry run)"), (t, COLT, "Locked test"))):
            e = S["classification"]["ci95"][k]
            ax.bar(j + off, e["point"], width=0.38, color=col, yerr=[[max(0.0, e["point"] - e["ci95_low"])], [max(0.0, e["ci95_high"] - e["point"])]], capsize=2.5, label=nm if j == 0 else None)
            src4.append({"metric": k, "set": nm, "value": e["point"], "ci95_low": e["ci95_low"], "ci95_high": e["ci95_high"]})
    ax.set_xticks(range(len(cl_rows)), [l for _, l in cl_rows])
    ax.set_ylabel("Value")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    src("fig4_validation_vs_test_classification", pd.DataFrame(src4))
    save(fig, "fig4_validation_vs_test_classification")

    def paired_bars(keys, labels, fname, ylabel, ylim=None, width=6.8):
        fig, ax = plt.subplots(figsize=(width, 3.9))
        s5 = []
        for j, k in enumerate(keys):
            for off, (S, col, nm) in zip((-0.2, 0.2), ((VS, COLV, "Validation (dry run)"), (TS, COLT, "Locked test"))):
                e = S["metrics"][k]
                ax.bar(j + off, e["value"], width=0.38, color=col, yerr=[[max(0.0, e["value"] - e["ci95_low"])], [max(0.0, e["ci95_high"] - e["value"])]], capsize=2.5, label=nm if j == 0 else None)
                s5.append({"metric": k, "set": nm, "value": e["value"], "ci95_low": e["ci95_low"], "ci95_high": e["ci95_high"]})
        ax.set_xticks(range(len(keys)), labels)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        src(fname, pd.DataFrame(s5))
        return fig

    save(paired_bars(["precision", "recall", "f1", "macro_f1"], ["Precision", "Recall", "F1 (micro)", "Macro F1"], "fig5_report_prf", "Finding agreement with the IU reference", (0, 0.8)), "fig5_final_report_precision_recall_f1")
    save(paired_bars(["hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention"], ["Hallucination\nrate", "Omission\nrate", "Classifier FP\npropagation", "Classifier TP\nretention"], "fig6_hall_omit_fp_tp", "Rate", (0, 1.0), 7.6), "fig6_hallucination_omission_fp_tp")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    s7 = []
    cols = {"normal": SERIES[2], "abnormal": SERIES[3], "indeterminate": "#898781"}
    for i, r in enumerate((rn, ra)):
        left = 0
        for s_ in ("normal", "abnormal", "indeterminate"):
            w = r[f"routed_{s_}_pct"]
            ax.barh(i, w, left=left, color=cols[s_], label=f"Routed {s_}" if i == 0 else None, height=0.55)
            if w > 6:
                ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=8)
            left += w
            s7.append({"reference_state": r["reference_state"], "routed": s_, "pct": w})
    ax.set_yticks([0, 1], [f"Reference normal\n(n={rn['n']})", f"Reference abnormal\n(n={ra['n']})"])
    ax.set_xlabel("Share of studies (%)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    for j, k in enumerate(("normal_recall_among_decided", "abnormal_recall_among_decided", "decision_coverage")):
        e = dd[k]
        ax.bar(j, e["value"], color=COLT, yerr=[[max(0.0, e["value"] - e["ci95_low"])], [max(0.0, e["ci95_high"] - e["value"])]], capsize=3)
        ax.text(j + 0.3, e["value"], f"{e['value']:.2f}", ha="left", va="center", fontsize=8)
        s7.append({"reference_state": "decided", "routed": k, "pct": 100 * e["value"]})
    ax.set_xticks(range(3), ["Normal recall\n(decided)", "Abnormal recall\n(decided)", "Decision\ncoverage"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    src("fig7_three_state", pd.DataFrame(s7))
    save(fig, "fig7_three_state_system_results")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fa8 = fa[fa.category.isin(CATNAME)]
    yy = np.arange(len(fa8))
    ax.barh(yy, fa8.pct_of_clinical_subset, color=[COLT if int(c.split("_")[0]) <= 8 else "#898781" for c in fa8.category])
    for i, (nn, p) in enumerate(zip(fa8.n_studies, fa8.pct_of_clinical_subset)):
        ax.text(p + 0.8, i, f"{int(nn)} ({p:.1f}%)", va="center", fontsize=7)
    ax.set_yticks(yy, [CATNAME[c] for c in fa8.category])
    ax.invert_yaxis()
    ax.set_xlim(0, max(fa8.pct_of_clinical_subset) * 1.25)
    ax.set_xlabel("Share of the clinical subset (%; studies can fall in several categories; grey = limitation flags)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    src("fig8_failure_categories", fa)
    save(fig, "fig8_failure_category_summary")
    chk = {"figures": len(list(FIG.glob("*.png"))), "tables": len(list(TAB.glob("*.csv"))), "min_dpi": round(min(Image.open(x).info.get("dpi", (0, 0))[0] for x in FIG.glob("*.png")))}
    (FIG / "asset_checks.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print(json.dumps(chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
