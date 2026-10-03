"""G1A documents generated from the stored G1A artifacts (no number typed by hand; the wording of every comparison follows the sign
and interval of the data, and a difference is attributed to retrieval only where the controlled G1 - G1A interval excludes zero):
NO_RETRIEVAL_LLM_ABLATION.md, MANUSCRIPT_G1A_METHODS.md, MANUSCRIPT_G1A_RESULTS.md, UNIVERSITY_REPORT_G1A.md, FIGURE_CAPTIONS.md,
TABLE_CAPTIONS.md, G1A_JOURNAL_ASSET_INDEX.md

    .venv\\Scripts\\python.exe -m scripts.g1a_06_write_docs
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.utils.config import PROJECT_ROOT

G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
B0S, GAS, G1S = "B0_rule_based", "G1A_llm_no_retrieval", "G1_single_agent_rag"
rd = lambda n: pd.read_csv(G1A / n, float_precision="round_trip")  # noqa: E731
jl = lambda n, d=G1A: json.loads((d / n).read_text(encoding="utf-8"))  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731
wc = lambda t: len(t.split())  # noqa: E731


def main() -> int:
    summ, repro, run, meta, frozen = jl("g1a_summary.json"), jl("g1_reproduction_check.json"), jl("generation_run_log.json"), jl("g1a_generator_metadata.json"), jl("g1a_prompt_frozen.json")
    g1prompt, checks, smoke = jl("g1_prompt_frozen.json", G1), jl("g1a_input_checks.json"), jl("smoke_test_report.json")
    mc, pair, pf, ri, nad, dec = rd("g1a_main_comparison.csv").set_index("metric"), rd("g1a_paired_differences.csv"), rd("g1a_per_finding_results.csv"), rd("g1a_retrieval_induced_findings.csv"), rd("g1a_normal_abnormal.csv"), rd("g1a_report_state_given_classifier_state.csv")
    rare, cp = rd("g1a_rare_finding_retention.csv"), rd("g1a_copying_analysis.csv").set_index("system")
    n_c, n_p = summ["n_clinical_subset"], summ["n_paired_studies"]
    Dd = lambda m, a=G1S, b=GAS: pair[(pair.comparison == f"{a} minus {b}") & (pair.metric == m)].iloc[0]  # noqa: E731
    P_ = lambda m, s: mc.loc[m, s]  # noqa: E731
    cis = lambda m, s: f"{f3(mc.loc[m, s])} (95% CI {f3(mc.loc[m, s + '_ci95_low'])} to {f3(mc.loc[m, s + '_ci95_high'])})"  # noqa: E731
    dci = lambda r: f"{r['diff']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731

    def word(r, lower_better=False):
        if not r.excludes_zero:
            return "did not differ reliably"
        up = r["diff"] > 0
        return ("was higher" if up else "was lower") + (" (favourable)" if up != lower_better else " (unfavourable)")

    def attributed(r, lower_better=False):
        return (f"the controlled comparison supports an effect of the retrieval context ({word(r, lower_better)})" if r.excludes_zero else "the controlled comparison does not support an effect of retrieval")

    nm = lambda s: {B0S: "B0", GAS: "G1A", G1S: "G1"}[s]  # noqa: E731
    gen = meta["model"]
    same = all(meta["same_generator_as_g1"].values())
    PR, RC, F1, MF = (Dd(k) for k in ("precision", "recall", "f1", "macro_f1"))
    HL, OM = Dd("hallucination_rate"), Dd("omission_rate")
    FP, TP = Dd("clf_fp_propagation"), Dd("tp_retention")
    AR, NR, ON = Dd("abnormal_recall"), Dd("normal_recall"), Dd("over_normalisation_given_classifier_abnormal")
    RL, B4, MT, WD = Dd("rouge_l"), Dd("bleu4"), Dd("meteor_exact"), Dd("mean_words")
    RT, CT, RMC = Dd("rare_tp_retention"), Dd("common_tp_retention"), Dd("rare_minus_common_retention")
    add = ri[ri.group.str.startswith("added")].iloc[0]
    rem = ri[ri.group.str.startswith("removed")].iloc[0]
    net = ri[ri.group.str.startswith("net")].iloc[0]
    n_add, n_rem = int(add.n_findings), int(rem.n_findings)
    nc_net = int(net["net_correct_findings (added matching reference - removed matching reference)"])
    nu_net = int(net["net_unsupported_findings (added unsupported - removed unsupported)"])
    cla = {s: nad[(nad.source == s) & (nad.reference == "abnormal")].iloc[0] for s in ("classifier", B0S, GAS, G1S)}
    clnor = {s: nad[(nad.source == s) & (nad.reference == "normal")].iloc[0] for s in ("classifier", B0S, GAS, G1S)}
    decg = lambda s, cs: dec[(dec.system == s) & (dec.classifier_state == cs)].iloc[0]  # noqa: E731
    # attribution of over-normalisation
    ga_ar, g1_ar, clf_ar = P_("abnormal_recall", GAS), P_("abnormal_recall", G1S), P_("abnormal_recall", B0S)
    if AR.excludes_zero and AR["diff"] < 0:
        norm_attr = "Over-normalisation is present in G1A and is larger with the retrieved context: the controlled comparison supports a contribution of the retrieval context in addition to the language-model/prompt behaviour. This reading is tentative: G1A abnormal recall is inflated by its unsupported abnormal reports for classifier-normal studies, and the share of normal reports given an abnormal classifier output did not differ reliably between G1A and G1."
    elif not AR.excludes_zero:
        norm_attr = "The controlled comparison does not show that retrieval changes abnormal recall; the over-normalisation therefore originates mainly from the language model and prompt behaviour, which are shared by G1A and G1."
    else:
        norm_attr = "Abnormal recall is higher with the retrieved context, so retrieval does not explain the over-normalisation."
    base_note = (f"G1A abnormal recall {f3(ga_ar)} versus classifier {f3(clf_ar)}" + (" (already below the classifier)" if ga_ar < clf_ar else ""))
    rr = rare.set_index(["system", "group"])
    rare_g = lambda s, g: rr.loc[(s, g)]  # noqa: E731
    rare_txt = (f"Pooled true-positive retention over the six rare findings: B0 {f3(rare_g(B0S, 'rare (C2 definition)').tp_retention)}, G1A {f3(rare_g(GAS, 'rare (C2 definition)').tp_retention)} ({int(rare_g(GAS, 'rare (C2 definition)').retained)} of {int(rare_g(GAS, 'rare (C2 definition)').classifier_true_positives)}), "
                f"G1 {f3(rare_g(G1S, 'rare (C2 definition)').tp_retention)} ({int(rare_g(G1S, 'rare (C2 definition)').retained)} of {int(rare_g(G1S, 'rare (C2 definition)').classifier_true_positives)}); for the other findings B0 {f3(rare_g(B0S, 'other findings').tp_retention)}, "
                f"G1A {f3(rare_g(GAS, 'other findings').tp_retention)}, G1 {f3(rare_g(G1S, 'other findings').tp_retention)}.")
    rare_attr = (f"G1 minus G1A retention is {dci(RT)} for rare and {dci(CT)} for other findings; the rare-minus-other retention gap differs between G1 and G1A by {dci(RMC)}, so " +
                 ("retrieval disproportionately affects rare findings." if RMC.excludes_zero else "the data do not show that retrieval disproportionately suppresses rare findings (small counts limit this conclusion)."))

    # post hoc diagnostics of G1A degenerate behaviour and stratified (supplementary) analysis
    dg, strat = jl("g1a_degenerate_output_diagnostics.json"), rd("g1a_stratified_by_classifier_state.csv")
    dgA, dgG = dg[GAS], dg[G1S]
    n_nor, n_abn = dgA["studies_by_classifier_state"]["normal"], dgA["studies_by_classifier_state"]["abnormal"]
    ec_nor = dgA["enlarged_cardiomediastinum_stated_without_classifier_positive_by_classifier_state"].get("normal", 0)
    sv = lambda st, m: strat[(strat.classifier_state == st) & (strat.metric == m)].iloc[0]  # noqa: E731
    sci = lambda r: f"{r['g1_minus_g1a']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731
    s_pr, s_rc, s_f1, s_fp, s_tp = (sv("abnormal", m) for m in ("precision", "recall", "f1", "clf_fp_propagation", "tp_retention"))
    s_hn = sv("normal", "hallucination_rate")
    s_n = int(s_pr.n_studies)
    degen = (f"**Degenerate G1A behaviour (post hoc diagnostic; the prompt was not changed).** For {ec_nor} of the {n_nor} studies in which the classifier output was normal ({pc(ec_nor / n_nor)}), G1A wrote an enlarged-cardiomediastinum report that no classifier output supported: "
             f"the model reproduced the first term of the prompt's standard-term list when given 'No Finding: positive'. {pc(dgA['most_common_text_share'])} of all G1A reports are one identical text, versus {pc(dgG['most_common_text_share'])} for G1 (whose most common text is the normal report); "
             f"{dgA['reports_with_leaked_reasoning_tokens']} G1A reports contain leaked reasoning tokens and {dgA['reports_stopped_by_token_limit']} stopped at the token limit (G1: {dgG['reports_with_leaked_reasoning_tokens']} and {dgG['reports_stopped_by_token_limit']}). "
             "The G1 − G1A contrast therefore reflects the robustness of this prompt/model combination without retrieval context, and the removed retrieval-linked rule, in addition to retrieved text itself, and it must not be read as a pure effect of retrieved reports. "
             f"Within the {s_n} classifier-abnormal studies (supplementary stratified analysis, where G1A is not degenerate), G1 minus G1A was: precision {sci(s_pr)}, recall {sci(s_rc)}, F1 {sci(s_f1)}, classifier FP propagation {sci(s_fp)}, TP retention {sci(s_tp)}. "
             f"Within classifier-normal studies the hallucination-rate difference was {sci(s_hn)}, driven by the degenerate output described above.")

    short = (f"G1A showed degenerate behaviour that confounds the contrast: for {pc(ec_nor / n_nor)} of classifier-normal studies it wrote an unsupported enlarged-cardiomediastinum report (the first term of the prompt's term list), "
             f"{pc(dgA['most_common_text_share'])} of its reports are one identical text and {dgA['reports_with_leaked_reasoning_tokens']} contain leaked reasoning tokens (G1: {pc(dgG['most_common_text_share'])}, a normal report, and none). The prompt was frozen and not changed. "
             f"Among the {s_n} classifier-abnormal studies, where G1A is not degenerate, G1 minus G1A was precision {sci(s_pr)}, recall {sci(s_rc)} and F1 {sci(s_f1)} (supplementary Table 6).")

    # ============================================================ analysis
    R = []
    A = R.append
    A("# G1A: no-retrieval LLM ablation")
    A("")
    A("_Generated by `scripts/g1a_06_write_docs.py` from the G1A artifacts. Retrieval-validation studies only; the locked retrieval-test split was not opened; classifier, retrieval configuration, finding extractor and populations are those of G1._")
    A("")
    A("> **Read first.** " + degen)
    A("")
    A("## 1. Question and design")
    A("")
    A(f"What does retrieval add when the classifier and the language model are held constant? G1A uses the same local model, decoding, classifier output, studies, report format, finding extractor and metrics as G1; the only experimental difference is that G1A receives no retrieved reports, no retrieval scores and no evidence-support counts. Generator check: Ollama {meta['ollama_version']}, `{gen['model_tag']}` digest `{gen['digest'][:12]}`, {gen['quantization_level']}; identical to G1 on version, tag, digest, quantization, generation options, context and output limits ({'all checks passed' if same else 'MISMATCH'}). Options: temperature {meta['generation_options']['temperature']}, top_k {meta['generation_options']['top_k']}, seed {meta['generation_options']['seed']}, {meta['generation_options']['num_ctx']}-token context, {meta['generation_options']['num_predict']} new tokens at most.")
    A("")
    A("## 2. Inputs and prompt")
    A("")
    A(f"G1A receives only the frozen classifier-positive findings, their calibrated probabilities and the No Finding state. The classifier block of the user message is byte-identical to G1's for {checks['classifier_block_identical_to_g1']} of {checks['n']} studies; {checks['retrieval_content_in_any_message']} messages contain retrieval wording, and {checks['reference_sentences_in_message']} contain a reference sentence. The G1A system prompt is the G1 prompt with exactly the retrieval-dependent wording removed (the clause about retrieved excerpts; the rule on retrieved support; the rule on treating classifier positives without retrieved support cautiously; the retrieval mentions in the remaining rules); `g1a_prompt_diff_vs_g1.txt` lists every change. Prompt version `{frozen['prompt_version']}`, SHA-256 `{frozen['prompt_sha256']}`, frozen before bulk generation (G1 prompt `{g1prompt['prompt_sha256'][:12]}…`). "
      "Because the G1 rule on treating classifier-positive findings without retrieved support cautiously cannot exist without retrieval, the G1 − G1A contrast measures the retrieval context together with that retrieval-linked instruction. Empty-classifier studies keep the G1 behaviour (rule 9 of the G1A prompt, the same classifier block).")
    A("")
    A(f"Smoke test (4 studies, implementation checks only): {'passed' if smoke['all_implementation_checks_passed'] else 'failed'}, {smoke['format_deviations']} format deviations; the prompt was not revised. Bulk generation: {run['n_cached_success']} of {run['n_cases']} reports for the same {run['n_cases']} studies as G1, {run['n_failed']} failures, {summ['format_length'][GAS]['format_ok_rate'] * 100:.1f}% format-compliant, {run['wall_seconds_this_run'] / 60:.0f} minutes.")
    A("")
    A("## 3. G1 reproduction")
    A("")
    A(f"B0 and G1 were re-derived from the stored reports through the shared evaluation function and compared with the stored G1 per-study results on {repro[G1S]['n_studies']} studies: maximum absolute numeric difference {repro[G1S]['max_abs_numeric_difference']:.1e} ({repro[G1S]['numeric_columns_compared']} numeric columns), all stated-finding, truth, state columns identical: {all(repro[G1S]['string_columns_identical'].values())}. G1 reproduced: **{repro['g1_reproduced']}**. The G1 results themselves were not modified.")
    A("")
    A(f"## 4. Three-way comparison (paired studies: {n_p} for text metrics, {n_c} clinical-subset studies for finding metrics)")
    A("")
    A((G1A / "tables/table1_three_system_main_comparison.md").read_text(encoding="utf-8"))
    A(f"Against B0, G1A finding precision {word(Dd('precision', GAS, B0S))} ({dci(Dd('precision', GAS, B0S))}), recall {word(Dd('recall', GAS, B0S))} ({dci(Dd('recall', GAS, B0S))}), F1 {word(Dd('f1', GAS, B0S))} ({dci(Dd('f1', GAS, B0S))}) and hallucination rate {word(Dd('hallucination_rate', GAS, B0S), True)} ({dci(Dd('hallucination_rate', GAS, B0S))}). Even without retrieval the language model changes the propagation of classifier false positives (G1A {cis('clf_fp_propagation', GAS)}; B0 repeats every classifier positive by construction) and retains {cis('tp_retention', GAS)} of the classifier true positives, so part of what G1 showed against B0 is not caused by retrieval.")
    A("")
    A("## 5. Controlled comparison G1 − G1A: what retrieval adds")
    A("")
    A((G1A / "tables/table2_g1_minus_g1a_controlled_comparison.md").read_text(encoding="utf-8"))
    A(f"Finding precision {word(PR)} with retrieval ({dci(PR)}); {attributed(PR)}. Recall {word(RC)} ({dci(RC)}); {attributed(RC)}. F1 {word(F1)} ({dci(F1)}); {attributed(F1)}. Macro F1 {word(MF)} ({dci(MF)}). Hallucination rate {word(HL, True)} ({dci(HL)}); {attributed(HL, True)}. Omission rate {word(OM, True)} ({dci(OM)}); {attributed(OM, True)}. Classifier FP propagation {word(FP, True)} ({dci(FP)}); {attributed(FP, True)}. TP retention {word(TP)} ({dci(TP)}); {attributed(TP)}. Abnormal recall {word(AR)} ({dci(AR)}). ROUGE-L {word(RL)} ({dci(RL)}), BLEU-4 {dci(B4)}, METEOR variant {dci(MT)}, report length {dci(WD)} words.")
    A("")
    A("### 5b. Interpretation caveat and stratified analysis")
    A("")
    A(degen)
    A("")
    A((G1A / "tables/table6_supplementary_stratified_by_classifier_state.md").read_text(encoding="utf-8"))
    A("## 6. Retrieval-induced findings (G1 versus G1A)")
    A("")
    A(f"Per study and finding, {n_add} findings are present in G1 but not in G1A, and {n_rem} in G1A but not in G1. Of the {n_add} added by the retrieval context, {int(add.appear_in_retrieved_reports)} appear in at least one retrieved report, {int(add.match_reference)} match the reference and {int(add.unsupported_by_reference)} are unsupported by the reference ({int(add.also_classifier_positive)} were also classifier positives; {int(add.not_classifier_positive)} were not classifier positives). Of the {n_rem} removed, {int(rem.match_reference)} were correct findings that were lost and {int(rem.unsupported_by_reference)} were unsupported findings that were suppressed. Net effect: {nc_net:+d} findings matching the reference and {nu_net:+d} unsupported findings. "
      "This quantifies retrieval-induced benefit (added correct findings, suppressed unsupported findings) against retrieval-induced harm (added unsupported findings, lost correct findings).")
    A("")
    A((G1A / "tables/table4_retrieval_induced_findings.md").read_text(encoding="utf-8"))
    A("## 7. Normalisation bias")
    A("")
    A(f"Abnormal recall (reference-abnormal studies reported abnormal): classifier {f3(clf_ar)}, G1A {f3(ga_ar)}, G1 {f3(g1_ar)}; normal recall: classifier {f3(P_('normal_recall', B0S))}, G1A {f3(P_('normal_recall', GAS))}, G1 {f3(P_('normal_recall', G1S))}. G1 − G1A: abnormal recall {dci(AR)}, normal recall {dci(NR)}. "
      f"Given that the classifier reported an abnormal state, the report was normal in {pc(P_('over_normalisation_given_classifier_abnormal', GAS))} of G1A reports and {pc(P_('over_normalisation_given_classifier_abnormal', G1S))} of G1 reports (difference {dci(ON)}); the classifier-abnormal studies number {int(decg(GAS, 'abnormal').n)}. {norm_attr} ({base_note}.) The prompt was not changed.")
    A("")
    A((G1A / "tables/table5_normal_abnormal_analysis.md").read_text(encoding="utf-8"))
    A("## 8. Rare-finding retention")
    A("")
    A(f"{rare_txt} {rare_attr} Per-finding retention for B0, G1A and G1 is in Table 3.")
    A("")
    A((G1A / "tables/table3_per_finding_results.md").read_text(encoding="utf-8"))
    A("## 9. Copying")
    A("")
    c = cp
    A(f"Corpus-wide (all IU corpus reports; sentences of at least six words): G1A copies {pc(c.loc[GAS, 'corpus_copied_sentence_rate'])} of its long sentences verbatim from some corpus report and G1 {pc(c.loc[G1S, 'corpus_copied_sentence_rate'])}; of long sentences, {pc(c.loc[GAS, 'corpus_template_sentence_share_of_long (df>=%d)' % summ['template_min_df']])} (G1A) and {pc(c.loc[G1S, 'corpus_template_sentence_share_of_long (df>=%d)' % summ['template_min_df']])} (G1) are generic template sentences occurring in at least {summ['template_min_df']} corpus reports. "
      f"Whole reports whose long sentences all occur in the corpus: G1A {pc(c.loc[GAS, 'whole_report_all_long_sentences_in_corpus_rate'])}, G1 {pc(c.loc[G1S, 'whole_report_all_long_sentences_in_corpus_rate'])}; exact duplicates of an entire corpus report: G1A {pc(c.loc[GAS, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])}, G1 {pc(c.loc[G1S, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])}. With the G1 definition against the study's own Top-5 retrieved texts, the copied-sentence rate is {pc(c.loc[GAS, 'g1_top5_copied_sentence_rate (G1 definition)'])} for G1A (which never saw them: the control level) and {pc(c.loc[G1S, 'g1_top5_copied_sentence_rate (G1 definition)'])} for G1; whole-report copies {pc(c.loc[GAS, 'g1_top5_whole_report_copy_rate (G1 definition)'])} versus {pc(c.loc[G1S, 'g1_top5_whole_report_copy_rate (G1 definition)'])}. Mean report length: G1A {summ['format_length'][GAS]['mean_words']:.1f} words, G1 {summ['format_length'][G1S]['mean_words']:.1f}. "
      "Short generic sentences recur in IU reports, so copying is interpreted relative to the G1A control and not as plagiarism.")
    A("")
    A("## 10. Limitations")
    A("")
    A("- One 4-billion-parameter local model, one frozen prompt per system, greedy decoding, one run; no replicate generations; Ollama does not guarantee bitwise GPU determinism.")
    A("- The G1 − G1A contrast includes the retrieval-linked instruction that had to be removed from the G1A prompt; it is the effect of the retrieval context as G1 implemented it, not of retrieved text in isolation.")
    A("- Finding extraction is rule-based and the reference is MeSH-derived and incomplete; per-finding counts are small, especially for rare findings, and Enlarged Cardiomediastinum has no reference positives.")
    A("- Normal studies dominate the clinical subset; the normal/abnormal analysis is descriptive.")
    A("- Validation data only; selection of nothing was made on these results; the locked retrieval test remains unopened; G2 has not been started.")
    A("")
    A("## 11. Conclusion")
    A("")
    A(f"With classifier, model and decoding held constant, the retrieval context {('changed' if any(r.excludes_zero for r in (PR, RC, F1, HL, OM, FP, TP, AR)) else 'did not reliably change')} the main finding metrics: precision {word(PR)}, recall {word(RC)}, F1 {word(F1)}, hallucination rate {word(HL, True)}, omission rate {word(OM, True)}, classifier FP propagation {word(FP, True)}, TP retention {word(TP)}, abnormal recall {word(AR)}. Retrieval added {n_add} and removed {n_rem} findings, with a net {nc_net:+d} reference-matching and {nu_net:+d} unsupported findings. These differences are confounded by the degenerate G1A behaviour on classifier-normal studies described at the top of this document. {norm_attr} {rare_attr} No change was made to the classifier, the generator model or the retrieval configuration.")
    A("")
    (G1A / "NO_RETRIEVAL_LLM_ABLATION.md").write_text("\n".join(R), encoding="utf-8")

    # ============================================================ manuscript methods
    meth = f"""# Methods: no-retrieval language-model ablation (G1A)

## Purpose and design

The single-agent retrieval-augmented system (G1) differed from the rule-based baseline in both the language model and the retrieved context. To isolate the contribution of retrieval we ran an ablation (G1A) in which the same language model produced reports for the same {n_p} validation studies without any retrieved reports. Everything else was held constant: the frozen classifier with its operating thresholds, the No Finding rule and calibrated probabilities; the local model {gen['model_tag']} ({gen['quantization_level']}, served by Ollama {meta['ollama_version']}, digest recorded and verified to be identical to G1); greedy decoding with temperature 0, top-k 1, seed {meta['generation_options']['seed']}, a {meta['generation_options']['num_ctx']}-token context and at most {meta['generation_options']['num_predict']} new tokens; the Findings and Impression format; the finding extractor; the reference labels; and all metrics.

## Inputs and prompt

G1A received only the classifier-positive findings, their calibrated probabilities and the No Finding state. It received no retrieved report, retrieval score, evidence-support count, reference report or reference finding; this was verified for every message, and the classifier block was byte-identical to that of G1. The G1A prompt was the G1 prompt with only the retrieval-dependent wording removed (the introduction of retrieved excerpts, the two rules about retrieved support, and the retrieval references inside other rules); all remaining instructions, including the grounding rules on invention, copying, identifiers, history and recommendations, the normal-report and empty-classifier rules, the standard finding terms and the output format, were unchanged. Because the G1 rule on cautiously treating unsupported classifier positives depends on retrieval, the contrast between G1 and G1A includes that retrieval-linked instruction. The prompt was frozen with its hash after a four-study implementation test, and responses were cached per study.

## Comparisons and outcomes

Three systems were compared on identical studies: the rule-based baseline (B0), G1A and G1. Outcomes were finding precision, recall, micro and macro F1, hallucination and omission rates, propagation of classifier false positives, retention of classifier true positives, normal and abnormal recall of the report state, ROUGE-L, BLEU-4, an exact-match METEOR variant and report length, computed exactly as in G1 on the primary set ({n_p} studies) and the clinical-finding subset ({n_c} studies). G1 results were re-derived from the stored reports to confirm exact reproduction.

## Controlled analyses

The main contrast was G1 minus G1A with a paired study-level bootstrap (1,000 resamples, the fixed seed used in G1); a difference was attributed to retrieval only when its interval excluded zero. Retrieval-induced findings were quantified per study and finding: findings present in G1 but not G1A (and the reverse), whether they appear in the retrieved reports, whether they match the reference, and whether they were classifier positives. Over-normalisation was assessed as the abnormal recall of the classifier, G1A and G1 and as the share of reports stated to be normal when the classifier output was abnormal. Retention of true positives was compared for the six rare findings defined in the loss study and for all other findings. Copying was measured against all corpus reports (sentences of at least six words; template sentences occurring in at least {summ['template_min_df']} corpus reports; whole-report duplication) and with the G1 definition against the study's own retrieved reports.
"""
    mw = wc(meth)
    (G1A / "MANUSCRIPT_G1A_METHODS.md").write_text(meth, encoding="utf-8")

    resu = f"""# Results: no-retrieval language-model ablation (G1A)

## Generation and reproduction

G1A produced {run['n_cached_success']} of {run['n_cases']} reports without failure for the same studies as G1 ({summ['format_length'][GAS]['format_ok_rate'] * 100:.0f}% format-compliant), with a mean of {summ['format_length'][GAS]['mean_words']:.1f} words (G1 {summ['format_length'][G1S]['mean_words']:.1f}, B0 {summ['format_length'][B0S]['mean_words']:.1f}). The G1 results were reproduced exactly from the stored reports (maximum difference {repro[G1S]['max_abs_numeric_difference']:.0e}).

## Three-system comparison

On the {n_c} clinical-subset studies, finding precision was {cis('precision', B0S)} for B0, {cis('precision', GAS)} for G1A and {cis('precision', G1S)} for G1; recall {f3(P_('recall', B0S))}, {f3(P_('recall', GAS))} and {f3(P_('recall', G1S))}; F1 {f3(P_('f1', B0S))}, {f3(P_('f1', GAS))} and {f3(P_('f1', G1S))}; and macro F1 {f3(P_('macro_f1', B0S))}, {f3(P_('macro_f1', GAS))} and {f3(P_('macro_f1', G1S))} (Table 1, Figure 1). Reports with at least one hallucinated finding were {pc(P_('hallucination_rate', B0S))}, {pc(P_('hallucination_rate', GAS))} and {pc(P_('hallucination_rate', G1S))}, and with at least one omission {pc(P_('omission_rate', B0S))}, {pc(P_('omission_rate', GAS))} and {pc(P_('omission_rate', G1S))} (Figure 2). Classifier false-positive propagation and true-positive retention were {f3(P_('clf_fp_propagation', GAS))} and {f3(P_('tp_retention', GAS))} for G1A and {f3(P_('clf_fp_propagation', G1S))} and {f3(P_('tp_retention', G1S))} for G1 (B0 repeats every classifier positive: 1.000 for both) (Figure 3). ROUGE-L was {f3(P_('rouge_l', B0S))}, {f3(P_('rouge_l', GAS))} and {f3(P_('rouge_l', G1S))}.

## Effect of retrieval with the model held constant

G1 minus G1A (Table 2, Figure 5): precision {dci(PR)}, recall {dci(RC)}, F1 {dci(F1)}, hallucination rate {dci(HL)}, omission rate {dci(OM)}, classifier false-positive propagation {dci(FP)}, true-positive retention {dci(TP)}, abnormal recall {dci(AR)} and ROUGE-L {dci(RL)}. Intervals excluded zero for: {', '.join(n for n, r in (('precision', PR), ('recall', RC), ('F1', F1), ('hallucination rate', HL), ('omission rate', OM), ('FP propagation', FP), ('TP retention', TP), ('abnormal recall', AR), ('ROUGE-L', RL)) if r.excludes_zero) or 'none of these metrics'}; differences in the remaining metrics are not attributed to retrieval.

## Findings induced by the retrieval context

{n_add} findings were present in G1 but not in G1A: {int(add.appear_in_retrieved_reports)} appeared in the retrieved reports, {int(add.match_reference)} matched the reference and {int(add.unsupported_by_reference)} were unsupported by it; {int(add.not_classifier_positive)} had not been flagged by the classifier. {n_rem} findings were present in G1A but not in G1: {int(rem.match_reference)} were correct findings that were lost and {int(rem.unsupported_by_reference)} were unsupported findings that were suppressed. The net effect was {nc_net:+d} reference-matching and {nu_net:+d} unsupported findings (Table 4).

## Normal and abnormal consistency

Abnormal recall was {f3(clf_ar)} for the classifier, {f3(ga_ar)} for G1A and {f3(g1_ar)} for G1; normal recall {f3(P_('normal_recall', B0S))}, {f3(P_('normal_recall', GAS))} and {f3(P_('normal_recall', G1S))} (Table 5, Figure 4). When the classifier output was abnormal, {pc(P_('over_normalisation_given_classifier_abnormal', GAS))} of G1A reports and {pc(P_('over_normalisation_given_classifier_abnormal', G1S))} of G1 reports described a normal study (difference {dci(ON)}). {norm_attr}

## Rare findings

{rare_txt} {rare_attr}

## Copying

Against all corpus reports, {pc(cp.loc[GAS, 'corpus_copied_sentence_rate'])} of G1A and {pc(cp.loc[G1S, 'corpus_copied_sentence_rate'])} of G1 sentences of at least six words occurred verbatim in some corpus report, and {pc(cp.loc[GAS, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])} and {pc(cp.loc[G1S, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])} of reports duplicated an entire corpus report. Against the study's own retrieved reports, the copied-sentence rate was {pc(cp.loc[GAS, 'g1_top5_copied_sentence_rate (G1 definition)'])} for G1A, which never saw them, and {pc(cp.loc[G1S, 'g1_top5_copied_sentence_rate (G1 definition)'])} for G1.

## Interpretation

{short}

These are validation-set results for one small local model with one frozen prompt per system. The G1 − G1A contrast measures the retrieval context together with the retrieval-linked instruction that could not be kept in G1A. The classifier, generator model and retrieval configuration were unchanged, the locked retrieval-test partition was not opened, and no multi-agent system was evaluated.
"""
    rw = wc(resu)
    (G1A / "MANUSCRIPT_G1A_RESULTS.md").write_text(resu, encoding="utf-8")
    ok_len = 450 <= mw <= 700 and 600 <= rw <= 1000

    fc = f"""# Figure captions (G1A)

Retrieval-validation studies only (clinical subset {n_c} studies; text metrics {n_p} studies). B0 = rule-based report from the frozen classifier output; G1A = the same local language model as G1 given the classifier output only (no retrieval); G1 = the same model given the classifier output and the Top-5 retrieved reports with the evidence summary. Error bars: 95% study-level paired bootstrap intervals.

**Figure 1. Finding precision, recall and F1 for B0, G1A and G1.** Micro-averaged agreement between the findings stated in the report (any affirmative mention) and the MeSH-mapped IU reference findings.

**Figure 2. Hallucination versus omission.** Left: share of reports with at least one hallucinated finding and with at least one omitted finding. Right: mean hallucinated and omitted findings per report.

**Figure 3. Classifier false-positive propagation versus true-positive retention.** Proportion of classifier false positives mentioned in the report and of classifier true positives retained in the report. B0 repeats every classifier-positive finding by construction.

**Figure 4. Normal and abnormal recall.** Share of reference-normal and reference-abnormal studies whose report state is correct. The B0 report state equals the classifier state by construction.

**Figure 5. Retrieval-effect summary.** Left: paired differences G1 minus G1A with 95% intervals (blue: interval excludes zero; grey: it does not). Right: findings that differ between G1 and G1A, split by whether they match the reference.
"""
    tc = """# Table captions (G1A)

**Table 1. Three-system main comparison.** B0 rule-based, G1A LLM without retrieval and G1 LLM with retrieval on identical studies: finding precision, recall, F1, macro F1, hallucination and omission rates, classifier false-positive propagation and true-positive retention, normal and abnormal recall, ROUGE-L, BLEU-4, the exact-match METEOR variant and report length, with 95% bootstrap intervals.

**Table 2. Controlled comparison G1 minus G1A.** Effect of the retrieval context with classifier, model and decoding held constant; paired study-level bootstrap intervals.

**Table 3. Per-finding results.** Reference positives, classifier true positives, true-positive retention and precision / recall / F1 for each finding and each system; rare findings follow the C2 definition.

**Table 4. Retrieval-induced findings.** Findings present in G1 but not in G1A (added) and in G1A but not in G1 (removed): whether they appear in retrieved reports, match the reference, or were classifier positives; net effect.

**Table 6 (supplementary). G1 minus G1A by classifier state.** Post hoc stratification (classifier normal, abnormal, indeterminate) used to interpret the controlled comparison; paired bootstrap intervals.

**Table 5. Normal / abnormal analysis.** Normal and abnormal recall, counts of misclassified studies, and the share of reports stated to be normal when the classifier output was abnormal, for the classifier, B0, G1A and G1.
"""
    (G1A / "FIGURE_CAPTIONS.md").write_text(fc, encoding="utf-8")
    (G1A / "TABLE_CAPTIONS.md").write_text(tc, encoding="utf-8")
    idx = f"""# G1A journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_three_system_main_comparison.*` | B0 / G1A / G1 finding F1 {f3(P_('f1', B0S))} / {f3(P_('f1', GAS))} / {f3(P_('f1', G1S))} | Results |
| Table 2 `tables/table2_g1_minus_g1a_controlled_comparison.*` | Retrieval effect: F1 {dci(F1)}, hallucination {dci(HL)} | Results |
| Table 3 `tables/table3_per_finding_results.*` | Per-finding retention for B0 / G1A / G1 | Results / Supplementary |
| Table 4 `tables/table4_retrieval_induced_findings.*` | {n_add} findings added and {n_rem} removed by retrieval context; net {nc_net:+d} matching / {nu_net:+d} unsupported | Results |
| Table 5 `tables/table5_normal_abnormal_analysis.*` | Abnormal recall classifier {f3(clf_ar)}, G1A {f3(ga_ar)}, G1 {f3(g1_ar)} | Results / Discussion |
| Table 6 `tables/table6_supplementary_stratified_by_classifier_state.*` | Controlled comparison within classifier-abnormal studies (G1A not degenerate) | Supplementary |
| Figure 1 `figures/fig1_finding_precision_recall_f1.*` | Finding agreement of three systems | Results |
| Figure 2 `figures/fig2_hallucination_vs_omission.*` | Hallucination versus omission | Results |
| Figure 3 `figures/fig3_fp_propagation_vs_tp_retention.*` | FP propagation versus TP retention | Results |
| Figure 4 `figures/fig4_normal_abnormal_recall.*` | Over-normalisation origin | Results / Discussion |
| Figure 5 `figures/fig5_retrieval_effect_summary.*` | Controlled retrieval effect | Results / Discussion |
| `NO_RETRIEVAL_LLM_ABLATION.md` | Full G1A analysis | Supplementary / internal |
| `MANUSCRIPT_G1A_METHODS.md`, `MANUSCRIPT_G1A_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_G1A.md` | Content for Methodology, Results, Discussion | University report |
| `g1a_prompt_frozen.json`, `g1a_generator_metadata.json`, `g1a_prompt_diff_vs_g1.txt` | Frozen G1A prompt, generator metadata, prompt diff | Methods / code release |
| `g1a_per_study_results.csv`, `g1a_paired_differences.csv` | All per-study and paired results | Supplementary data |
| `g1a_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
"""
    (G1A / "G1A_JOURNAL_ASSET_INDEX.md").write_text(idx, encoding="utf-8")

    U = []
    A = U.append
    A("# University report content: G1A (no-retrieval ablation)")
    A("")
    A("_Insert-ready material generated from the G1A artifacts. Validation data only; classifier, generator model and retrieval configuration are unchanged; the locked retrieval-test split is unopened._")
    A("")
    A("## Methodology")
    A("")
    A("### Purpose of the ablation")
    A("")
    A(f"The single-agent RAG system (G1) was compared with the rule-based baseline, but the two differ in the language model and in the retrieved context. The ablation G1A removes only the retrieval: the same local model ({gen['model_tag']}, offline, greedy decoding, fixed seed) writes the report from the classifier output alone for the same {n_p} validation studies. All other components (classifier, thresholds, calibration, report format, finding extractor, reference labels, metrics and evaluation populations) are identical to G1, and the identity of the model, its digest and the generation options was checked automatically.")
    A("")
    A("### Inputs, prompt and comparisons")
    A("")
    A("G1A receives the classifier-positive findings with calibrated probabilities and the No Finding state; it receives no retrieved report, retrieval score or evidence summary. The prompt is that of G1 with the retrieval-dependent wording removed; the full difference is stored in the artifacts. Three systems are compared on identical studies (rule-based B0, G1A, G1), and the effect of retrieval is estimated as G1 minus G1A with a paired study-level bootstrap of 1,000 resamples. A difference is attributed to retrieval only if its interval excludes zero. Additional analyses quantify which findings the retrieved context adds or removes, whether over-normalisation (reports describing normal studies despite abnormal classifier output) originates from the model or the context, how rare findings are retained, and how much text is copied from corpus reports.")
    A("")
    A("## Results")
    A("")
    A("### Three-system comparison and the controlled retrieval effect")
    A("")
    A(f"Finding F1 was {f3(P_('f1', B0S))} for B0, {f3(P_('f1', GAS))} for G1A and {f3(P_('f1', G1S))} for G1; hallucination rates were {pc(P_('hallucination_rate', B0S))}, {pc(P_('hallucination_rate', GAS))} and {pc(P_('hallucination_rate', G1S))}. With the model held constant, retrieval changed finding precision by {dci(PR)}, recall by {dci(RC)}, F1 by {dci(F1)}, the hallucination rate by {dci(HL)}, and classifier false-positive propagation by {dci(FP)} and true-positive retention by {dci(TP)} (Tables 1 and 2, Figures 1 to 3 and 5).")
    A("")
    A("### Retrieval-induced findings, normalisation and rare findings")
    A("")
    A(f"The retrieval context added {n_add} findings and removed {n_rem}; of the added findings {int(add.match_reference)} matched the reference and {int(add.unsupported_by_reference)} did not, and of the removed ones {int(rem.match_reference)} were correct and {int(rem.unsupported_by_reference)} unsupported (Table 4). Abnormal recall was {f3(clf_ar)} for the classifier, {f3(ga_ar)} for G1A and {f3(g1_ar)} for G1; {norm_attr} {rare_txt} {rare_attr}")
    A("")
    A("### Copying")
    A("")
    A(f"Against all corpus reports, {pc(cp.loc[GAS, 'corpus_copied_sentence_rate'])} of G1A and {pc(cp.loc[G1S, 'corpus_copied_sentence_rate'])} of G1 sentences of at least six words occurred verbatim, so verbatim reuse is partly a property of the generic IU reporting style; the study's own retrieved reports were copied in {pc(cp.loc[GAS, 'g1_top5_copied_sentence_rate (G1 definition)'])} (G1A, control) versus {pc(cp.loc[G1S, 'g1_top5_copied_sentence_rate (G1 definition)'])} (G1) of long sentences.")
    A("")
    A("### Interpretation caveat")
    A("")
    A(short)
    A("")
    A("## Discussion")
    A("")
    A("The ablation shows what the retrieval context adds under this model and prompt: the controlled intervals above, not the difference to the rule-based baseline, are the evidence for an effect of retrieval. Limits: one small model and one prompt per system; the G1 − G1A contrast includes the retrieval-linked instruction that could not be kept; rule-based finding extraction and a MeSH-derived reference; small counts for rare findings; and validation data only. The multi-agent system (G2) has not been started.")
    A("")
    (G1A / "UNIVERSITY_REPORT_G1A.md").write_text("\n".join(U), encoding="utf-8")

    bad = []
    for n in ("NO_RETRIEVAL_LLM_ABLATION.md", "MANUSCRIPT_G1A_METHODS.md", "MANUSCRIPT_G1A_RESULTS.md", "UNIVERSITY_REPORT_G1A.md", "FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md", "G1A_JOURNAL_ASSET_INDEX.md"):
        txt = (G1A / n).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"superior", r"\{[a-z_]+\[[^}]*\}", r"\bnan\b", r"\bNone\b", r"anthropic", r"claude"):
            if re.search(pat, txt, re.I if pat in ("anthropic", "claude") else 0):
                bad.append((n, pat))
    print(json.dumps({"methods_words": mw, "results_words": rw, "length_ok": ok_len, "wording_flags": bad}))
    return 0 if ok_len and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
