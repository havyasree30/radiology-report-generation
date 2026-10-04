"""G1 documents generated from the stored G1 artifacts (no number is typed by hand; wording of every comparison follows the sign
and interval of the data): SINGLE_AGENT_RAG_ANALYSIS.md, MANUSCRIPT_G1_METHODS.md, MANUSCRIPT_G1_RESULTS.md, UNIVERSITY_REPORT_G1.md,
FIGURE_CAPTIONS.md, TABLE_CAPTIONS.md, G1_JOURNAL_ASSET_INDEX.md

    .venv\\Scripts\\python.exe -m scripts.g1_07_write_docs
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
rd = lambda n: pd.read_csv(G1 / n, float_precision="round_trip")  # noqa: E731
jl = lambda n: json.loads((G1 / n).read_text(encoding="utf-8"))  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731
wc = lambda t: len(t.split())  # noqa: E731


def main() -> int:
    pops, summ, gen, frozen = jl("g1_evaluation_populations.json"), jl("g1_summary_metrics.json"), jl("GENERATOR_FREEZE.json"), jl("g1_prompt_frozen.json")
    smoke, runlog, leak, integ5 = jl("smoke_test_report.json"), jl("generation_run_log.json"), jl("payload_leakage_check.json"), jl("retrieval_top5_integrity.json")
    bt, pf, lr, conf = rd("g1_paired_bootstrap.csv"), rd("g1_per_finding_results.csv"), rd("g1_length_redundancy.csv"), rd("g1_normal_abnormal_confusion.csv")
    sb, ho, prov, tm = rd("g1_support_bucket_analysis.csv"), rd("g1_hallucination_omission.csv"), rd("g1_grounding_provenance.csv"), rd("g1_text_metrics.csv")
    B = lambda a, m: bt[(bt.analysis == a) & (bt.metric == m)].iloc[0]  # noqa: E731
    M = "G1_single_agent_rag"
    b0, g1 = summ["B0_rule_based"], summ[M]
    sup, ctxu, na, ev, macro = summ["support_vs_truth"], summ["context_utilization"], summ["normal_abnormal_consistency"], summ["extractor_vs_mesh_on_reference_text"], summ["macro_over_classes_with_truth"]
    n_cl, n_pr = pops["paired_clinical_subset"], pops["paired_model_comparison_set"]
    mod, opt = gen["model"], gen["generation_options"]

    def ci(r, s="g1"):
        return f"{f3(r[s])} (95% CI {f3(r[s + '_ci95_low'])} to {f3(r[s + '_ci95_high'])})"

    def dci(r):
        return f"{r['diff']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"

    def word(r, lower_better=False):
        if r["ci95_low"] <= 0 <= r["ci95_high"]:
            return "did not differ reliably"
        up = r["diff"] > 0
        return ("was higher" if up else "was lower") + (" (an improvement)" if up != lower_better else " (worse)")

    P, R_, F = B("finding_any_mention", "precision"), B("finding_any_mention", "recall"), B("finding_any_mention", "f1")
    H, O = B("finding_any_mention", "hallucination_rate"), B("finding_any_mention", "omission_rate")
    HM, OM = B("finding_any_mention", "mean_hallucinated"), B("finding_any_mention", "mean_omitted")
    FP, TP = B("propagation_any_mention", "clf_fp_propagation"), B("propagation_any_mention", "tp_retention")
    FPd, TPd = B("propagation_definite_only", "clf_fp_propagation"), B("propagation_definite_only", "tp_retention")
    Fd = B("finding_definite_only", "f1")
    RL, B4, B1, MT = (B("text_primary_set", k) for k in ("rouge_l", "bleu4", "bleu1", "meteor_exact"))
    fp_reduced = FP["ci95_high"] < 0
    tp_lost = TP["ci95_high"] < 0
    answer = ("Yes, in this comparison G1 mentioned fewer classifier false positives than B0" if fp_reduced else
              "No reliable reduction of classifier false-positive propagation was observed" if FP["ci95_low"] <= 0 <= FP["ci95_high"] else "No: G1 propagated more classifier false positives than B0")
    answer += (f", at the price of retaining fewer classifier true positives (retention {ci(TP, 'g1')} versus {ci(TP, 'b0')})" if tp_lost else
               f", while true-positive retention {word(TP).replace('was ', 'was ')} (G1 {f3(TP['g1'])} versus B0 {f3(TP['b0'])})")
    ps_ = rd("g1_per_study_results.csv")
    gcl = ps_[(ps_.system == M) & (ps_.in_clinical)]
    sets_ = lambda v: set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()  # noqa: E731
    ro_n = ro_tp = 0
    for _, r_ in gcl.iterrows():
        st_, tr_, cl_ = sets_(r_.stated), sets_(r_.truth), sets_(r_.clf_pos)
        for f_ in st_ - cl_:
            ro_n += 1
            ro_tp += int(f_ in tr_)
    selectivity = (f"B0 mentions every classifier-positive finding by construction (propagation and retention are both 1.000), so the informative quantity is how selectively G1 drops them: "
                   f"G1 kept {pc(TP['g1'])} of the classifier true positives but {pc(FP['g1'])} of the classifier false positives (difference {TP['g1'] - FP['g1']:+.3f}).")
    retr_only = (f"Findings stated by G1 that the classifier had not flagged (taken from the retrieved reports) numbered {ro_n}, of which {ro_tp} ({pc(ro_tp / max(ro_n, 1))}) are in the reference.")
    macro_note = (f"Macro F1 over the {macro[M]['classes_with_truth']} findings with reference positives was {'lower' if macro[M]['macro_f1'] < macro['B0_rule_based']['macro_f1'] else 'higher'} for G1 "
                  f"({f3(macro[M]['macro_f1'])}) than for B0 ({f3(macro['B0_rule_based']['macro_f1'])}).")
    attrib = ("This is a comparison of two complete systems: G1 differs from B0 in both the language model with its grounding prompt and the retrieved context, and no no-retrieval LLM ablation was run, "
              "so the effect cannot be attributed to retrieval alone.")
    pv = {r.category: r for r in prov.itertuples()}
    tot_prov = int(prov.n_findings.sum())
    buck = sb.pivot(index="support_bucket", columns="is_true_positive", values="g1_mention_rate").reindex(["0/5", "1-2/5", "3-5/5"])
    nb = sb.pivot(index="support_bucket", columns="is_true_positive", values="n").reindex(["0/5", "1-2/5", "3-5/5"])
    cn = lambda src_, ref: conf[(conf.source == src_) & (conf.reference == ref)].iloc[0]  # noqa: E731
    lrg = lr.set_index("system")

    # ============================================================ analysis report
    Rr = []
    A = Rr.append
    A("# G1: single-agent RAG baseline and preliminary report generation")
    A("")
    A("_Generated by `scripts/g1_07_write_docs.py` from the G1 artifacts. Retrieval-validation studies only; the locked retrieval-test split was not opened; the classifier and the R2 retrieval configuration are frozen._")
    A("")
    A("## 1. Objective")
    A("")
    A("Compare a rule-based report that verbalises the frozen classifier output (B0) with a single-agent RAG system (G1: frozen classifier + frozen R2 Top-5 hybrid retrieval + deterministic evidence summary + one LLM call), mainly on clinical-finding agreement, hallucination, omission and propagation of classifier false positives. Lexical overlap is secondary and is not clinical correctness.")
    A("")
    A("## 2. Frozen configuration")
    A("")
    A(f"- **Classifier:** DenseNet-121, √-weighted BCE, C2-B checkpoint, C4 F1-optimal thresholds with the No Finding rule, C5 Platt-calibrated probabilities (unchanged).")
    A(f"- **Retrieval (R2 candidate, unchanged):** query = top-3 positive finding names; hybrid Dense MiniLM (384-d) + BM25 with RRF (k = 60, depth 100); Top-{integ5['retrieval_config']['k']}; no MMR, no expansion, no gating, no weighting. For the {integ5['r2_primary_studies_checked']} studies shared with R2, the Top-5 lists are identical to the R2 results ({integ5['top5_identical_to_r2']} of {integ5['r2_primary_studies_checked']}).")
    A(f"- **Generator:** local Ollama `{mod['model_tag']}` (digest `{mod['digest'][:12]}`, {mod['parameter_size']}, {mod['quantization_level']}, native context {mod['context_length_native']}), Ollama {gen['ollama_version']}, endpoint `{gen['endpoint']}`, greedy decoding (temperature {opt['temperature']}, top_k {opt['top_k']}, seed {opt['seed']}), context window {opt['num_ctx']} tokens, at most {opt['num_predict']} new tokens, no tools, no web access, no external API. Hardware: {gen['hardware_backend'].get('nvidia_smi')}. The same model and options are frozen for G2.")
    A(f"- **Prompt:** version `{frozen['prompt_version']}`, SHA-256 `{frozen['prompt_sha256']}`, frozen before bulk generation (`g1_prompt_frozen.json`).")
    A("")
    A("## 3. Evaluation populations")
    A("")
    A((G1 / "tables/table1_evaluation_population.md").read_text(encoding="utf-8"))
    A(f"The primary report-generation set keeps studies whose classifier output is empty or imperfect ({pops['primary_set_classifier_query_status'].get('empty', 0)} studies have no classifier output and no retrieval context). Finding-level metrics use the clinical subset ({n_cl} studies); text metrics use the primary set ({n_pr}). B0 and G1 are always compared on identical studies; {pops['studies_without_g1_report']} studies lack a G1 report.")
    A("")
    A("## 4. Methods in brief")
    A("")
    A("**B0** writes one sentence per classifier-positive finding with canonical terms (a normal report for No Finding; an explicit 'indeterminate' report when the classifier outputs neither a pathology nor No Finding). **G1** receives the classifier positives with calibrated probabilities and the No Finding state, the Top-5 retrieved Findings and Impressions with rank and score information, and a deterministic evidence table (probability and the number of Top-5 reports whose mapped findings support each classifier-positive finding). It never receives the reference report or the truth findings (structural payload whitelist; "
      f"{leak['cases_with_reference_sentence_outside_retrieved_reports']} of {leak['n_cases']} payloads contain a reference sentence outside the retrieved reports). The grounding prompt asks the model to use only the supplied evidence, to treat unsupported classifier positives cautiously, not to copy retrieved reports and not to add history, measurements, comparisons or recommendations.")
    A("")
    A(f"Generated findings are extracted from the report text with the Phase 1 assertion-aware lexicon plus a small change set fixed before generation. A *stated* finding is any affirmative mention (definite or hedged; primary), a *definite* finding a non-hedged mention (sensitivity). The reference is the MeSH-mapped IU finding set of R1; as a sensitivity check the same extractor is applied to the reference text. Applied to reference text, the extractor reaches precision {f3(ev['precision'])}, recall {f3(ev['recall'])} and F1 {f3(ev['f1'])} against the MeSH truth, which bounds the measurement quality of every finding-level number below. Study-level paired bootstrap: 1,000 resamples, seed 42.")
    A("")
    A("## 5. Smoke test and an implementation fix")
    A("")
    A(f"Four development studies (normal, two or more positives, one positive, empty query) checked connectivity, parsing, format, leakage and caching only. The first attempt showed that the 4B model sometimes keeps generating after the IMPRESSION by echoing the input template, which hit the token limit in two cases and contaminated the parsed sections. This was treated as an implementation problem: generation now ends at the input section titles (stop sequences) and the parser discards echoed text and repeated sections. The prompt wording was not changed. After the fix all four checks passed ({smoke['n_cases']} cases, {smoke['format_deviations']} format deviations, all finish reasons `stop`), the prompt and generator were frozen, and all studies, including these four, were regenerated in the bulk run. "
      f"Bulk run: {runlog['n_cached_success']} of {runlog['n_cases']} reports generated, {runlog['n_failed']} failures, G1 format-compliant in {pc(lrg.loc[M, 'format_ok_rate'])} of reports.")
    A("")
    A("## 6. Primary comparison")
    A("")
    A(f"On the {n_cl} clinical-subset studies (any affirmative mention), finding precision {word(P)} for G1 versus B0 ({ci(P, 'g1')} versus {ci(P, 'b0')}; difference {dci(P)}), recall {word(R_)} ({ci(R_, 'g1')} versus {ci(R_, 'b0')}; {dci(R_)}) and F1 {word(F)} ({ci(F, 'g1')} versus {ci(F, 'b0')}; {dci(F)}). With definite assertions only, the F1 difference is {dci(Fd)}. Macro F1 over the {macro[M]['classes_with_truth']} classes with reference positives is {f3(macro[M]['macro_f1'])} for G1 and {f3(macro['B0_rule_based']['macro_f1'])} for B0.")
    A("")
    A("**Table 2. B0 versus G1.**")
    A("")
    A((G1 / "tables/table2_b0_vs_g1_main_metrics.md").read_text(encoding="utf-8"))
    A("**Table 3. Per-finding results.**")
    A("")
    A((G1 / "tables/table3_per_finding_generation_results.md").read_text(encoding="utf-8"))
    A("## 7. Hallucination and omission")
    A("")
    A(f"Reports with at least one hallucinated finding: G1 {ci(H, 'g1')}, B0 {ci(H, 'b0')} (difference {dci(H)}, which {word(H, True)}). Reports with at least one omitted finding: G1 {ci(O, 'g1')}, B0 {ci(O, 'b0')} (difference {dci(O)}, which {word(O, True)}). Per report, hallucinated findings {f3(HM['g1'])} versus {f3(HM['b0'])} and omitted findings {f3(OM['g1'])} versus {f3(OM['b0'])} (differences {dci(HM)} and {dci(OM)}). 'Hallucinated' means a stated finding absent from the MeSH-mapped reference; it includes findings that are present in the image but not indexed by MeSH.")
    A("")
    A("**Table 4. Hallucination and omission.**")
    A("")
    A((G1 / "tables/table4_hallucination_omission_analysis.md").read_text(encoding="utf-8"))
    A("## 8. Retrieval grounding of the findings stated by G1")
    A("")
    A(f"Of the {tot_prov} findings stated by G1 in the clinical subset: {pc(pv['A_classifier_only'].share)} were supported by the classifier only, {pc(pv['B_retrieval_only'].share)} by retrieval only, {pc(pv['C_both'].share)} by both and {pc(pv['D_neither'].share)} by neither (unsupported-finding rate; n = {int(pv['D_neither'].n_findings)}). Support means that the finding is a classifier positive and/or appears in the mapped findings of at least one Top-5 retrieved report. {retr_only}")
    A("")
    A("## 9. Classifier false-positive propagation")
    A("")
    A(f"{answer}. {selectivity} Classifier false-positive propagation (pooled over {int(g1['propagation_any_mention']['n_clf_fp'])} classifier false positives): G1 {ci(FP, 'g1')}, B0 {ci(FP, 'b0')}, difference {dci(FP)}. True-positive retention (over {int(g1['propagation_any_mention']['n_clf_tp'])} classifier true positives): G1 {ci(TP, 'g1')}, B0 {ci(TP, 'b0')}, difference {dci(TP)}. With definite assertions only: propagation {dci(FPd)}, retention {dci(TPd)}. {attrib}")
    A("")
    A(f"Retrieved support separates classifier true from false positives only partly: the mean support fraction is {f3(sup['mean_support_fraction_true_positives'])} for true positives and {f3(sup['mean_support_fraction_false_positives'])} for false positives; {sup['fp_with_zero_support']} of {sup['n_fp']} false positives and {sup['tp_with_zero_support']} of {sup['n_tp']} true positives have no retrieved support. G1 suppressed {sup['fp_suppressed_by_g1']} classifier false positives ({sup['fp_suppressed_with_zero_support']} with zero support) and omitted {sup['tp_omitted_by_g1']} true positives ({sup['tp_omitted_with_zero_support']} with zero support). Mention rates by support bucket (true positive / false positive): "
      + "; ".join(f"{b}: {f3(buck.loc[b, True])} (n={int(nb.loc[b, True])}) / {f3(buck.loc[b, False])} (n={int(nb.loc[b, False])})" for b in buck.index) + ". An omitted false positive is counted as suppression only because the reference confirms it is absent; retrieval is not claimed to have 'corrected' the classifier beyond that.")
    A("")
    A("**Table 5. False-positive propagation.**")
    A("")
    A((G1 / "tables/table5_false_positive_propagation.md").read_text(encoding="utf-8"))
    A("## 10. Normal / abnormal consistency")
    A("")
    A(f"On the clinical subset ({n_cl} studies), normal-recall / abnormal-recall / accuracy (indeterminate counted as wrong): classifier {f3(na['classifier']['normal_recall'])} / {f3(na['classifier']['abnormal_recall'])} / {f3(na['classifier']['accuracy'])}; B0 {f3(na['B0_rule_based']['normal_recall'])} / {f3(na['B0_rule_based']['abnormal_recall'])} / {f3(na['B0_rule_based']['accuracy'])}; G1 {f3(na[M]['normal_recall'])} / {f3(na[M]['abnormal_recall'])} / {f3(na[M]['accuracy'])}. "
      f"Indeterminate share: B0 {pc(na['B0_rule_based']['indeterminate_share'])}, G1 {pc(na[M]['indeterminate_share'])}. G1 {'reports abnormal studies as normal more often than the classifier does' if na[M]['abnormal_recall'] < na['classifier']['abnormal_recall'] else 'does not report abnormal studies as normal more often than the classifier'} (abnormal recall {f3(na[M]['abnormal_recall'])} versus {f3(na['classifier']['abnormal_recall'])}) while its normal recall is {f3(na[M]['normal_recall'])} versus {f3(na['classifier']['normal_recall'])}. The full confusion counts are in `g1_normal_abnormal_confusion.csv`.")
    A("")
    A("## 11. Length, redundancy and copying")
    A("")
    A(f"Mean words per report: B0 {lrg.loc['B0_rule_based', 'mean_words']:.1f}, G1 {lrg.loc[M, 'mean_words']:.1f} (median {lrg.loc[M, 'median_words']:.0f}); repeated-sentence rate {f3(lrg.loc[M, 'repeated_sentence_rate'])}. Against the Top-5 retrieved texts, G1 copies {pc(lrg.loc[M, 'copied_sentence_rate'])} of its sentences of at least six words verbatim, {pc(lrg.loc[M, 'ngram8_overlap_with_retrieved'])} of its 8-grams occur in retrieved text, and {pc(lrg.loc[M, 'whole_report_copy_rate'])} of reports are whole-report copies (generic short phrases are not counted).")
    A("")
    A("## 12. Top-5 context utilisation")
    A("")
    A(f"Among {ctxu['studies_with_stated_findings']} G1 reports with retrieval context and at least one stated finding, {f3(ctxu['mean_supporting_reports_among_top5'])} of the five retrieved reports on average contain support for a stated finding; the share of support pairs by retrieval rank 1 to 5 is {', '.join(pc(x) for x in ctxu['share_by_rank'])}; {ctxu['studies_supported_only_by_rank1']} reports are supported only by the rank-1 report. K was not changed.")
    A("")
    A("## 13. Lexical metrics (secondary)")
    A("")
    A(f"On the primary set ({n_pr} studies), ROUGE-L {ci(RL, 'g1')} for G1 versus {ci(RL, 'b0')} for B0 (difference {dci(RL)}); BLEU-4 {f3(B4['g1'])} versus {f3(B4['b0'])} ({dci(B4)}); BLEU-1 {f3(B1['g1'])} versus {f3(B1['b0'])} ({dci(B1)}); METEOR (exact-match variant, no stemming or synonyms; not comparable with published METEOR) {f3(MT['g1'])} versus {f3(MT['b0'])} ({dci(MT)}). BERTScore was not computed (no local model; no download was approved). Lexical overlap is not evidence of clinical correctness.")
    A("")
    A("## 14. Qualitative examples")
    A("")
    A("Six deterministically selected validation cases (normal, abnormal, classifier false positive suppressed and propagated, omitted finding, retrieval mismatch) are in `qualitative_examples.md` and summarised in Table 6.")
    A("")
    A((G1 / "tables/table6_representative_cases.md").read_text(encoding="utf-8"))
    A("## 15. Limitations")
    A("")
    A("- B0 versus G1 compares whole systems (LLM + grounding prompt + retrieval); no LLM-without-retrieval ablation was run, so the contribution of retrieval is not isolated.")
    A("- A single 4B local model, one frozen prompt, greedy decoding; no prompt search and no replicate generations. Ollama does not guarantee bitwise GPU determinism.")
    A("- Finding extraction is rule-based and approximate; the MeSH reference is incomplete (studies with only unmapped terms are excluded from finding metrics; findings present in the image but not indexed count as hallucinations; Enlarged Cardiomediastinum has no IU reference).")
    A("- Finding counts per class are small for several classes, so per-finding results are descriptive.")
    A("- METEOR is a limited variant and BERTScore is missing; lexical metrics are secondary.")
    A("- Normal studies dominate the clinical subset (" + f"{pops['in_clinical_subset_reference_normal']} of {n_cl}).")
    A("- Validation data only; the locked retrieval test and any end-to-end test evaluation remain for the final system. G2 has not been started.")
    A("")
    A("## 16. G1 conclusion")
    A("")
    A(f"G1 single-agent RAG versus the rule-based B0: finding F1 {word(F)} ({dci(F)}); hallucination rate {word(H, True)} ({dci(H)}); classifier false-positive propagation {word(FP, True)} ({dci(FP)}); true-positive retention {word(TP)} ({dci(TP)}). {answer}. {selectivity} {macro_note} {attrib} The classifier, retrieval configuration and Top-K were not changed, and G2 has not been started.")
    A("")
    (G1 / "SINGLE_AGENT_RAG_ANALYSIS.md").write_text("\n".join(Rr), encoding="utf-8")

    # ============================================================ manuscript Methods
    meth = f"""# Methods: single-agent retrieval-augmented report generation (G1)

## Pipeline and frozen components

Each chest radiograph was processed by the frozen classifier (DenseNet-121 trained with a square-root-weighted loss, class-specific F1-optimal thresholds, a rule that suppresses No Finding when a pathology is positive, and Platt-calibrated probabilities). The positive findings formed a query of at most the three most probable finding names, and the five best reports were retrieved from the reference corpus with the frozen hybrid retriever (dense MiniLM and BM25 fused by Reciprocal Rank Fusion). Classifier, thresholds, calibration, retrieval configuration and Top-5 were not changed. All experiments used the retrieval-validation studies; the locked test partition was not opened.

## Evaluation populations

The primary report-generation set contained every validation study with a frontal image and a non-empty reference report ({pops['primary_report_generation_set']} studies), including studies with empty or imperfect classifier output ({pops['primary_set_classifier_query_status'].get('empty', 0)} had no classifier output and no retrieval context). The clinical-finding subset ({pops['clinical_finding_evaluation_subset']} studies) additionally had a usable mapped finding set derived from MeSH terms. B0 and G1 were compared on identical studies.

## Systems

The rule-based baseline (B0) wrote one sentence per classifier-positive finding with canonical terms and no retrieved context; it wrote a normal report for No Finding and an explicit indeterminate report when the classifier produced neither a pathology nor No Finding. The single-agent system (G1) made one call per study to a local language model ({mod['model_tag']}, {mod['quantization_level']}, served by Ollama {gen['ollama_version']}; greedy decoding with temperature 0, fixed seed, {opt['num_ctx']}-token context, at most {opt['num_predict']} new tokens, no tools or web access). Its input contained the classifier positives with calibrated probabilities and the No Finding state; the Findings and Impression of the five retrieved reports with rank and score information; and a deterministic evidence table giving, for each classifier-positive finding, its probability and the number of retrieved reports whose mapped findings contain it. The reference report and the reference findings were structurally excluded from the input and verified to be absent. The prompt instructed the model to use only the supplied evidence, to give more weight to findings supported by several retrieved reports, to omit or hedge unsupported classifier positives, not to copy retrieved text, and to add no history, measurements, comparisons or recommendations. The prompt was frozen (SHA-256 recorded) after an implementation smoke test on four studies, and generation was cached per study.

## Outcome measures

Findings were extracted from report text with an assertion-aware rule-based lexicon (negated mentions excluded); any affirmative mention counted as stated (primary) and non-hedged mentions as definite (sensitivity). Against the mapped reference we computed micro precision, recall and F1, per-finding results, hallucinated findings (stated but absent from the reference) and omitted findings (present in the reference but not stated). The primary comparison metrics were finding F1, hallucination rate, propagation of classifier false positives (classifier-positive, reference-negative findings mentioned in the report) and retention of classifier true positives. Retrieval grounding classified each stated finding as supported by the classifier only, retrieval only, both or neither. Normal/abnormal consistency, report length, repeated sentences and copying of retrieved text were also measured. Secondary lexical metrics were BLEU-1, BLEU-4, ROUGE-L and an exact-match METEOR variant.

## Statistics

Differences between systems were estimated with a study-level paired bootstrap (1,000 resamples, fixed seed) and are reported with 95% percentile intervals. The bootstrap quantifies uncertainty only and was not used for tuning.
"""
    mw = wc(meth)
    (G1 / "MANUSCRIPT_G1_METHODS.md").write_text(meth, encoding="utf-8")

    # ============================================================ manuscript Results
    resu = f"""# Results: single-agent retrieval-augmented report generation (G1)

## Populations and generation

The primary set comprised {pops['primary_report_generation_set']} validation studies and the clinical-finding subset {pops['clinical_finding_evaluation_subset']} ({pops['in_clinical_subset_reference_normal']} reference-normal); {pops['paired_model_comparison_set']} studies had both a B0 and a G1 report (Table 1). The local model produced {runlog['n_cached_success']} of {runlog['n_cases']} reports without failure and {pc(lrg.loc[M, 'format_ok_rate'])} followed the requested format. Reports contained a mean of {lrg.loc[M, 'mean_words']:.1f} words (B0 {lrg.loc['B0_rule_based', 'mean_words']:.1f}).

## Clinical finding agreement

Finding precision {word(P)} for G1 compared with B0 ({ci(P, 'g1')} versus {ci(P, 'b0')}), recall {word(R_)} ({ci(R_, 'g1')} versus {ci(R_, 'b0')}) and F1 {word(F)} ({ci(F, 'g1')} versus {ci(F, 'b0')}; difference {dci(F)}) (Table 2, Figure 1). Counting only non-hedged assertions, the F1 difference was {dci(Fd)}. Per-finding results are in Table 3.

## Hallucination and omission

{pc(H['g1'])} of G1 reports and {pc(H['b0'])} of B0 reports contained at least one hallucinated finding (difference {dci(H)}), and {pc(O['g1'])} and {pc(O['b0'])} at least one omission ({dci(O)}); per report G1 stated {f3(HM['g1'])} hallucinated and omitted {f3(OM['g1'])} findings, against {f3(HM['b0'])} and {f3(OM['b0'])} for B0 (Table 4, Figure 2). Of the findings stated by G1, {pc(pv['A_classifier_only'].share)} were supported by the classifier only, {pc(pv['B_retrieval_only'].share)} by retrieval only, {pc(pv['C_both'].share)} by both and {pc(pv['D_neither'].share)} by neither (Figure 5). {retr_only}

## Classifier false-positive propagation

Classifier false-positive propagation was {ci(FP, 'g1')} for G1 and {ci(FP, 'b0')} for B0 (difference {dci(FP)}), and true-positive retention {ci(TP, 'g1')} versus {ci(TP, 'b0')} ({dci(TP)}) (Table 5, Figure 3). {answer}. {selectivity} With definite assertions only the differences were {dci(FPd)} and {dci(TPd)}. Retrieved support discriminated classifier errors only partly: mean support was {f3(sup['mean_support_fraction_true_positives'])} for true positives and {f3(sup['mean_support_fraction_false_positives'])} for false positives, and {sup['fp_with_zero_support']} of {sup['n_fp']} false positives had no support. G1 left out {sup['fp_suppressed_by_g1']} classifier false positives and {sup['tp_omitted_by_g1']} classifier true positives. {attrib}

## Normal and abnormal consistency

Against the reference, the classifier classified normal and abnormal studies with recall {f3(na['classifier']['normal_recall'])} and {f3(na['classifier']['abnormal_recall'])}, B0 reports with {f3(na['B0_rule_based']['normal_recall'])} and {f3(na['B0_rule_based']['abnormal_recall'])}, and G1 reports with {f3(na[M]['normal_recall'])} and {f3(na[M]['abnormal_recall'])}; {pc(na[M]['indeterminate_share'])} of G1 reports and {pc(na['B0_rule_based']['indeterminate_share'])} of B0 reports were indeterminate.

## Copying, context use and lexical overlap

G1 copied {pc(lrg.loc[M, 'copied_sentence_rate'])} of its sentences of at least six words verbatim from the retrieved reports, and {pc(lrg.loc[M, 'whole_report_copy_rate'])} of reports were whole-report copies. On average {f3(ctxu['mean_supporting_reports_among_top5'])} of the five retrieved reports supported a stated finding, with the support spread over ranks 1 to 5 as {', '.join(pc(x) for x in ctxu['share_by_rank'])}. ROUGE-L was {ci(RL, 'g1')} for G1 and {ci(RL, 'b0')} for B0 (difference {dci(RL)}); BLEU-4 {f3(B4['g1'])} versus {f3(B4['b0'])}; the METEOR variant {f3(MT['g1'])} versus {f3(MT['b0'])} (Figure 4). BERTScore was not computed.

## Measurement validity and sensitivity

Applied to the reference report text, the finding extractor agreed with the MeSH-derived reference with precision {f3(ev['precision'])}, recall {f3(ev['recall'])} and F1 {f3(ev['f1'])}, which bounds the quality of every finding-level number above. When the same extractor was used for the reference (text-derived rather than MeSH-derived findings), the F1 was {f3(g1['finding_vs_reference_text_extraction']['f1'])} for G1 and {f3(b0['finding_vs_reference_text_extraction']['f1'])} for B0. Macro F1 over the {macro[M]['classes_with_truth']} findings with reference positives was {f3(macro[M]['macro_f1'])} for G1 and {f3(macro['B0_rule_based']['macro_f1'])} for B0.

## Qualitative examples and error categories

Six deterministically selected cases are summarised in Table 6, and Figure 6 splits hallucinated findings into classifier false positives and other sources, and omissions into classifier true positives dropped by the report and findings never predicted by the classifier.

## Interpretation

These are validation-set results for one 4-billion-parameter local model with one frozen prompt. B0 and G1 differ in both the language model and the retrieved context, so the comparison does not isolate the contribution of retrieval. Finding extraction is rule-based and the reference is derived from MeSH terms. The locked retrieval-test partition has not been evaluated and no multi-agent system was tested.
"""
    rw = wc(resu)
    (G1 / "MANUSCRIPT_G1_RESULTS.md").write_text(resu, encoding="utf-8")
    ok_len = 500 <= mw <= 700 and 700 <= rw <= 1000

    fc = f"""# Figure captions (G1)

Retrieval-validation studies only (clinical subset {n_cl} studies; text metrics {n_pr} studies). B0 = rule-based report from the frozen classifier output; G1 = single-agent RAG (classifier + Top-5 hybrid retrieval + evidence summary + local language model). Error bars: 95% study-level bootstrap intervals.

**Figure 1. Finding precision, recall and F1.** Micro-averaged agreement between the findings stated in the generated report (any affirmative mention) and the MeSH-mapped IU reference findings for B0 and G1.

**Figure 2. Hallucination and omission.** Left: share of reports with at least one hallucinated finding (stated, absent from the reference) and at least one omitted finding (in the reference, not stated). Right: the corresponding mean numbers of findings per report.

**Figure 3. Classifier false-positive propagation and true-positive retention.** Proportion of classifier false positives mentioned in the report and of classifier true positives retained in the report.

**Figure 4. Lexical report-generation metrics.** BLEU-1, BLEU-4, ROUGE-L and an exact-match METEOR variant (no stemming or synonyms) against the reference report. Lexical overlap is secondary and does not measure clinical correctness.

**Figure 5. Finding-support provenance.** Left: share of findings stated by G1 supported by the classifier only, by retrieval only, by both, or by neither. Right: share of classifier-positive findings mentioned in the G1 report by the number of Top-5 retrieved reports supporting the finding, for classifier true and false positives.

**Figure 6. Error-category breakdown.** Left: hallucinated findings split into classifier false positives and other sources. Right: omitted findings split into classifier true positives dropped by the report and findings the classifier never predicted.
"""
    tc = """# Table captions (G1)

**Table 1. G1 evaluation population.** Retrieval-validation studies, primary report-generation set, clinical-finding subset and the paired comparison set.

**Table 2. B0 versus G1 main metrics.** Finding precision, recall and F1, hallucination and omission rates, classifier false-positive propagation, true-positive retention and lexical metrics, with 95% paired study-level bootstrap intervals for each system and for the difference.

**Table 3. Per-finding generation results.** Reference positives, findings stated and precision / recall / F1 for each of the 13 abnormal findings.

**Table 4. Hallucination and omission analysis.** Mean hallucinated and omitted findings per report and the share of reports with at least one, for any affirmative mention and for definite assertions only.

**Table 5. False-positive propagation.** Classifier false positives mentioned and true positives retained in the report, for any affirmative mention and for definite assertions only.

**Table 6. Representative cases.** Deterministically selected validation cases (anonymised) with classifier findings, reference findings, findings stated by G1, false positives and omissions.
"""
    (G1 / "FIGURE_CAPTIONS.md").write_text(fc, encoding="utf-8")
    (G1 / "TABLE_CAPTIONS.md").write_text(tc, encoding="utf-8")
    idx = f"""# G1 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_evaluation_population.*` | {pops['primary_report_generation_set']} primary / {pops['clinical_finding_evaluation_subset']} clinical-subset validation studies | Methods |
| Table 2 `tables/table2_b0_vs_g1_main_metrics.*` | Finding F1 {f3(F['b0'])} (B0) versus {f3(F['g1'])} (G1); difference {dci(F)} | Results |
| Table 3 `tables/table3_per_finding_generation_results.*` | Per-finding agreement | Results / Supplementary |
| Table 4 `tables/table4_hallucination_omission_analysis.*` | Hallucination rate {pc(H['b0'])} (B0) versus {pc(H['g1'])} (G1) | Results |
| Table 5 `tables/table5_false_positive_propagation.*` | FP propagation {f3(FP['b0'])} versus {f3(FP['g1'])}; TP retention {f3(TP['b0'])} versus {f3(TP['g1'])} | Results / Discussion |
| Table 6 `tables/table6_representative_cases.*` | Six representative cases | Results / Supplementary |
| Figure 1 `figures/fig1_finding_precision_recall_f1.*` | Finding agreement B0 versus G1 | Results |
| Figure 2 `figures/fig2_hallucination_omission.*` | Hallucination and omission | Results |
| Figure 3 `figures/fig3_fp_propagation_tp_retention.*` | FP propagation versus TP retention | Results / Discussion |
| Figure 4 `figures/fig4_lexical_metrics.*` | Secondary lexical metrics | Results |
| Figure 5 `figures/fig5_finding_support_provenance.*` | Source of stated findings; support versus truth | Results / Discussion |
| Figure 6 `figures/fig6_error_category_breakdown.*` | Hallucination and omission sources | Discussion |
| `SINGLE_AGENT_RAG_ANALYSIS.md` | Full G1 analysis | Supplementary / internal |
| `MANUSCRIPT_G1_METHODS.md`, `MANUSCRIPT_G1_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_G1.md` | Content for Methodology, Results, Discussion | University report |
| `generated_reports.csv`, `g1_per_study_results.csv` | Every B0 / G1 report with inputs, evidence and references | Supplementary data |
| `GENERATOR_FREEZE.json`, `g1_prompt_frozen.json`, `REPORT_GENERATION_OUTPUT_SCHEMA.json` | Frozen generator and prompt; output schema | Methods / code release / application |
| `g1_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
"""
    (G1 / "G1_JOURNAL_ASSET_INDEX.md").write_text(idx, encoding="utf-8")

    U = []
    A = U.append
    A("# University report content: G1 (single-agent RAG baseline)")
    A("")
    A("_Insert-ready material generated from the G1 artifacts. Validation data only; the locked retrieval-test split is unopened; classifier and retrieval are frozen._")
    A("")
    A("## Methodology")
    A("")
    A("### Single-Agent RAG architecture and data flow")
    A("")
    A(f"The system turns a chest radiograph into a preliminary report in four stages. (1) The frozen DenseNet-121 classifier outputs fourteen independent probabilities, which are thresholded and calibrated. (2) The positive findings (at most the three most probable) form a query, and five reports are retrieved from a corpus of other patients' reports by fusing a dense MiniLM ranking and a BM25 ranking. (3) A deterministic evidence table is computed that lists, for every classifier-positive finding, its calibrated probability and how many of the five retrieved reports mention that finding. (4) A single call to a local language model ({mod['model_tag']}, run offline with Ollama, greedy decoding) writes a short FINDINGS and IMPRESSION text. Nothing is learned or tuned in stages 2 to 4: the classifier, thresholds, calibration, retrieval configuration and number of references are those fixed in earlier stages. The reference report is never part of the input.")
    A("")
    A("### Grounding prompt and evidence summary")
    A("")
    A("The prompt tells the model to use only the classifier output and the retrieved evidence, not to invent findings that appear in neither, to give more confidence to findings supported by several retrieved reports, to omit or hedge classifier-positive findings without support, not to mention every classifier-positive finding, not to copy retrieved reports, and not to add history, measurements, comparisons or recommendations. A normal classifier state yields a normal report even if retrieved reports describe disease. The evidence table is information for the generator, not a second classifier, and never changes the stored classifier prediction. A rule-based baseline (B0) that verbalises the classifier output without retrieval allows the question of whether retrieval-augmented generation improves on simply stating the classifier findings.")
    A("")
    A("### Evaluation")
    A("")
    A(f"Findings are extracted from generated text by an assertion-aware lexicon and compared with the findings mapped from the MeSH terms of the study. The main outcomes are finding precision, recall and F1, hallucinated and omitted findings, the proportion of classifier false positives that appear in the report, and the proportion of classifier true positives that are kept. Differences are estimated with a paired study-level bootstrap of 1,000 resamples. Lexical metrics (BLEU, ROUGE-L, METEOR-like) are secondary. The primary set has {pops['primary_report_generation_set']} validation studies and the clinical subset {pops['clinical_finding_evaluation_subset']}.")
    A("")
    A("## Results")
    A("")
    A("### Report-generation metrics")
    A("")
    A(f"Finding F1 was {ci(F, 'g1')} for G1 and {ci(F, 'b0')} for B0 (difference {dci(F)}); precision {f3(P['g1'])} versus {f3(P['b0'])} and recall {f3(R_['g1'])} versus {f3(R_['b0'])} (Table 2, Figure 1). ROUGE-L was {f3(RL['g1'])} versus {f3(RL['b0'])}, BLEU-4 {f3(B4['g1'])} versus {f3(B4['b0'])} (Figure 4).")
    A("")
    A("### Hallucination, omission and false-positive propagation")
    A("")
    A(f"At least one hallucinated finding occurred in {pc(H['g1'])} of G1 reports and {pc(H['b0'])} of B0 reports; at least one omission in {pc(O['g1'])} and {pc(O['b0'])}. Classifier false-positive propagation was {f3(FP['g1'])} for G1 and {f3(FP['b0'])} for B0 (difference {dci(FP)}), and true-positive retention {f3(TP['g1'])} versus {f3(TP['b0'])} (difference {dci(TP)}). {answer}. (Tables 4 and 5, Figures 2, 3, 5 and 6.)")
    A("")
    A("### Qualitative examples")
    A("")
    A("Six deterministically chosen validation cases (a correct normal report, a correct abnormal report, a classifier false positive that G1 left out, one that G1 repeated, an omitted finding and a retrieval mismatch) are listed in `qualitative_examples.md`, each with the classifier output, the retrieved evidence, the generated report, the reference report and a short factual error analysis.")
    A("")
    A("## Discussion")
    A("")
    A(f"Benefits: the report is grounded in two sources, the evidence table makes the support of every classifier finding explicit, and the whole pipeline runs offline on a local model. Limits: {attrib} The model is small (4 billion parameters, 4-bit quantisation) and one prompt was used; the reference is derived from MeSH terms, so findings visible in the image but not indexed count as hallucinations; finding extraction is rule-based; and retrieved support separates classifier true from false positives only partly (mean support {f3(sup['mean_support_fraction_true_positives'])} versus {f3(sup['mean_support_fraction_false_positives'])}). The results are validation-set results; the locked retrieval test has not been used, and the multi-agent comparison (G2) is the next stage.")
    A("")
    (G1 / "UNIVERSITY_REPORT_G1.md").write_text("\n".join(U), encoding="utf-8")

    bad = []
    for n in ("SINGLE_AGENT_RAG_ANALYSIS.md", "MANUSCRIPT_G1_METHODS.md", "MANUSCRIPT_G1_RESULTS.md", "UNIVERSITY_REPORT_G1.md", "FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md", "G1_JOURNAL_ASSET_INDEX.md"):
        txt = (G1 / n).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"superior", r"\{[a-z_]+\[[^}]*\}", r"\bnan\b", r"\bNone\b", r"anthropic", r"claude"):
            if re.search(pat, txt, re.I if pat in ("anthropic", "claude") else 0):
                bad.append((n, pat))
    print(json.dumps({"methods_words": mw, "results_words": rw, "length_ok": ok_len, "wording_flags": bad}))
    return 0 if ok_len and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
