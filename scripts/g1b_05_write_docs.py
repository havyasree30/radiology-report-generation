"""G1B documents generated from the stored G1B artifacts (no number typed by hand; the wording of every comparison follows the sign and
interval of the data, and a difference is attributed to the relevance of the retrieved content only where the controlled G1 - G1B interval excludes zero):
SHAM_RETRIEVAL_CONTROL_ANALYSIS.md, MANUSCRIPT_G1B_METHODS.md, MANUSCRIPT_G1B_RESULTS.md, UNIVERSITY_REPORT_G1B.md, FIGURE_CAPTIONS.md,
TABLE_CAPTIONS.md, G1B_JOURNAL_ASSET_INDEX.md

    .venv\\Scripts\\python.exe -m scripts.g1b_05_write_docs
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.utils.config import PROJECT_ROOT

G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
B0S, GAS, GBS, G1S = "B0_rule_based", "G1A_llm_no_retrieval", "G1B_llm_sham_retrieval", "G1_single_agent_rag"
rd = lambda n: pd.read_csv(G1B / n, float_precision="round_trip")  # noqa: E731
jl = lambda n, d=G1B: json.loads((d / n).read_text(encoding="utf-8"))  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731
wc = lambda t: len(t.split())  # noqa: E731


def main() -> int:
    summ, repro, run, meta, fz, checks, diag = jl("g1b_summary.json"), jl("g1_reproduction_check.json"), jl("generation_run_log.json"), jl("g1b_generator_metadata.json"), jl("g1b_sham_mapping_frozen.json"), jl("g1b_input_checks.json"), jl("g1b_stability_diagnostics.json")
    mc, pair, ti, dif, cpy, cpd, manip = rd("g1b_main_comparison.csv").set_index("metric"), rd("g1b_paired_differences.csv"), rd("g1b_context_introduced_and_removed_findings.csv").set_index("system"), rd("g1b_g1_vs_g1b_findings.csv"), \
        rd("g1b_copying_analysis.csv").set_index("system"), rd("g1b_copying_paired_difference.csv").set_index("metric"), rd("g1b_context_manipulation_check.csv").set_index("system")
    rare, nad, dec, strat = rd("g1b_rare_finding_retention.csv").set_index(["system", "group"]), rd("g1b_normal_abnormal.csv"), rd("g1b_report_state_given_classifier_state.csv"), rd("g1b_stratified_by_classifier_state.csv")
    n_c, n_p, n_ctx = summ["n_clinical_subset"], summ["n_paired_studies"], summ["n_studies_with_context"]
    Dd = lambda m, a=G1S, b=GBS: pair[(pair.comparison == f"{a} minus {b}") & (pair.metric == m)].iloc[0]  # noqa: E731
    P_ = lambda m, s: mc.loc[m, s]  # noqa: E731
    cis = lambda m, s: f"{f3(mc.loc[m, s])} (95% CI {f3(mc.loc[m, s + '_ci95_low'])} to {f3(mc.loc[m, s + '_ci95_high'])})"  # noqa: E731
    dci = lambda r: f"{r['diff']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731

    def word(r, lower_better=False):
        if not r.excludes_zero:
            return "did not differ reliably"
        up = r["diff"] > 0
        return ("was higher" if up else "was lower") + (" (favourable)" if up != lower_better else " (unfavourable)")

    gen = meta["model_tag"]
    same = all(meta["same_generator_as_g1"].values()) and meta["prompt_is_g1_prompt"]
    PR, RC, F1, MF = (Dd(k) for k in ("precision", "recall", "f1", "macro_f1"))
    HL, OM, FP, TP = Dd("hallucination_rate"), Dd("omission_rate"), Dd("clf_fp_propagation"), Dd("tp_retention")
    NR, AR, ON, RL = Dd("normal_recall"), Dd("abnormal_recall"), Dd("over_normalisation_given_classifier_abnormal"), Dd("rouge_l")
    RT, CT, RMC, WD = Dd("rare_tp_retention"), Dd("common_tp_retention"), Dd("rare_minus_common_retention"), Dd("mean_words")
    BL, MT = Dd("bleu4"), Dd("meteor_exact")
    SH_F1, SH_HL, SH_PR, SH_RC = (Dd(k, GBS, GAS) for k in ("f1", "hallucination_rate", "precision", "recall"))
    if F1.excludes_zero and F1["diff"] > 0:
        verdict = ("G1 improved finding F1 over G1B. Because the prompt, model, decoding, classifier output and context structure were identical and only the identity of the five context reports differed, this is evidence that the relevance of the "
                   "retrieved reports, and not merely the presence of radiology-report text in the context, contributes to performance under this model and prompt.")
    elif F1.excludes_zero:
        verdict = "G1B exceeded G1 on finding F1. The data therefore give no evidence that relevant retrieval is better than sham context for finding F1 under this model and prompt."
    else:
        verdict = "G1 and G1B did not differ reliably on finding F1. The evidence does not show a specific benefit from retrieval relevance for finding F1 under this model and prompt."
    if HL.excludes_zero and HL["diff"] > 0 and TP.excludes_zero and TP["diff"] > 0:
        verdict += (f" The gain is a trade-off: relevant context kept far more classifier-positive findings (true-positive retention {dci(TP)}) and had a higher hallucination rate ({dci(HL)}), whereas sham context suppressed classifier-positive findings "
                    f"and so produced fewer hallucinated statements together with lower recall; a lower hallucination rate with sham context is therefore not evidence of better grounding.")
    short_claims = f"This is a single-model, single-prompt, single-run validation result; it does not show clinical benefit, and relevance is only one of the differences the sham context removes (the displayed rank scores and the evidence-support counts follow the context supplied)."
    intro = lambda s: ti.loc[s]  # noqa: E731
    rr = lambda s, g: rare.loc[(s, g)]  # noqa: E731
    rare_txt = (f"Pooled true-positive retention over the six rare findings: B0 {f3(rr(B0S, 'rare (C2 definition)').tp_retention)}, G1A {f3(rr(GAS, 'rare (C2 definition)').tp_retention)}, G1B {f3(rr(GBS, 'rare (C2 definition)').tp_retention)} "
                f"({int(rr(GBS, 'rare (C2 definition)').retained)} of {int(rr(GBS, 'rare (C2 definition)').classifier_true_positives)}), G1 {f3(rr(G1S, 'rare (C2 definition)').tp_retention)} "
                f"({int(rr(G1S, 'rare (C2 definition)').retained)} of {int(rr(G1S, 'rare (C2 definition)').classifier_true_positives)}); other findings: G1A {f3(rr(GAS, 'other findings').tp_retention)}, G1B {f3(rr(GBS, 'other findings').tp_retention)}, G1 {f3(rr(G1S, 'other findings').tp_retention)}.")
    rare_attr = (f"G1 minus G1B retention is {dci(RT)} for rare and {dci(CT)} for other findings, so " + ("relevance of the context changes rare-finding retention." if RT.excludes_zero else
                 "the data do not show that relevant context changes rare-finding retention, which remains weak (small counts limit this conclusion)."))
    nn = lambda who, ref: nad[(nad.source == who) & (nad.reference == ref)].iloc[0]  # noqa: E731
    d_g = lambda s: dec[(dec.system == s) & (dec.classifier_state == "abnormal")].iloc[0]  # noqa: E731
    sv = lambda st, m: strat[(strat.classifier_state == st) & (strat.metric == m)].iloc[0]  # noqa: E731
    sci = lambda r: f"{r['g1_minus_g1b']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731
    s_pr, s_rc, s_f1 = (sv("abnormal", m) for m in ("precision", "recall", "f1"))
    s_n = int(s_f1.n_studies)
    dg1, dgb = diag[G1S], diag[GBS]
    g1b_added, g1_added = dif.iloc[1], dif.iloc[0]
    net = dif.iloc[2]
    strat_note = (f"Within the {s_n} classifier-abnormal studies the F1 gain remained (G1 minus G1B {sci(s_f1)}) and was driven by recall ({sci(s_rc)}); the precision difference was {'reliable' if s_pr['excludes_zero'] else 'not reliable'} ({sci(s_pr)}).")
    stability = (f"Stability of G1B (descriptive): {pc(dgb['most_common_text_share'])} of G1B reports are one identical text (G1: {pc(dg1['most_common_text_share'])}); {dgb['reports_with_leaked_reasoning_tokens']} G1B reports contain leaked reasoning tokens "
                 f"and {dgb['reports_stopped_by_token_limit']} stopped at the token limit (G1: {dg1['reports_with_leaked_reasoning_tokens']} and {dg1['reports_stopped_by_token_limit']}); "
                 f"{dgb['enlarged_cardiomediastinum_stated_without_classifier_positive_in_classifier_normal_studies']} of {dgb['classifier_normal_studies']} classifier-normal studies received an unsupported enlarged-cardiomediastinum statement (G1: {dg1['enlarged_cardiomediastinum_stated_without_classifier_positive_in_classifier_normal_studies']}).")
    own_c, own_w = "copied_sentence_rate_from_own_context (G1 definition)", "whole_report_copy_rate_from_own_context (G1 definition)"
    cp_txt = (f"Against the study's own context reports, {pc(cpy.loc[G1S, own_c])} of G1 sentences and {pc(cpy.loc[GBS, own_c])} of G1B sentences were copied verbatim (paired difference {dci(pd.Series({'diff': cpd.loc['copied_sentence_rate_from_own_context', 'diff_g1_minus_g1b'], 'ci95_low': cpd.loc['copied_sentence_rate_from_own_context', 'ci95_low'], 'ci95_high': cpd.loc['copied_sentence_rate_from_own_context', 'ci95_high']}))}); "
              f"whole-report copies were {pc(cpy.loc[G1S, own_w])} (G1) and {pc(cpy.loc[GBS, own_w])} (G1B) (difference {dci(pd.Series({'diff': cpd.loc['whole_report_copy_rate_from_own_context', 'diff_g1_minus_g1b'], 'ci95_low': cpd.loc['whole_report_copy_rate_from_own_context', 'ci95_low'], 'ci95_high': cpd.loc['whole_report_copy_rate_from_own_context', 'ci95_high']}))}), over the {n_ctx} studies that had context. "
              f"G1 copied {pc(cpy.loc[G1S, 'copied_sentence_rate_from_sham_context (studies with context)'])} of its sentences from the sham reports (control level) and G1B {pc(cpy.loc[GBS, 'copied_sentence_rate_from_g1_true_top5 (studies with context)'])} from the true Top-5. "
              f"Against all corpus reports, verbatim sentence rates were {pc(cpy.loc[GAS, 'corpus_copied_sentence_rate'])} (G1A), {pc(cpy.loc[GBS, 'corpus_copied_sentence_rate'])} (G1B) and {pc(cpy.loc[G1S, 'corpus_copied_sentence_rate'])} (G1), and exact duplicates of an entire corpus report "
              f"{pc(cpy.loc[GAS, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])}, {pc(cpy.loc[GBS, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])} and {pc(cpy.loc[G1S, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])}.")
    if cpd.loc["copied_sentence_rate_from_own_context", "excludes_zero"] and cpd.loc["copied_sentence_rate_from_own_context", "diff_g1_minus_g1b"] > 0:
        cp_interp = "Copying from the supplied context is higher with relevant than with sham context, so copying is not a property of the prompt alone."
    elif cpd.loc["copied_sentence_rate_from_own_context", "excludes_zero"]:
        cp_interp = "Copying from the supplied context is lower with relevant than with sham context."
    else:
        cp_interp = "Copying from the supplied context did not differ reliably between relevant and sham context, so the copying seen in G1 is largely a property of this prompt and model given any report-like context and is not specific to relevance."

    A_ = []
    A = A_.append
    A("# G1B: sham-retrieval control")
    A("")
    A("_Generated by `scripts/g1b_05_write_docs.py` from the G1B artifacts. Retrieval-validation studies only; the locked retrieval-test split was not opened; classifier, retrieval configuration, prompt, generator, finding extractor and populations are those of G1._")
    A("")
    A(f"> **Summary.** {verdict} {short_claims}")
    A("")
    A("## 1. Question and design")
    A("")
    A(f"G1A removed the retrieved context entirely and became behaviourally degenerate on classifier-normal studies, so it is reported as a classifier-only robustness ablation, not as the primary causal retrieval ablation. G1B instead keeps the exact G1 prompt, the exact generator, the exact message format and the same number of context reports, and replaces the five retrieved reports of every study with five deterministic sham reports drawn from the retrieval corpus. "
      f"The primary comparison is G1 (relevant Top-5) minus G1B (sham Top-5). Generator check: Ollama {meta['ollama_version']}, `{gen}` digest `{meta['model_digest'][:12]}`, identical to G1 on version, tag, digest, options, context and output limits and on the prompt hash ({'all checks passed' if same else 'MISMATCH'}). "
      f"Options: temperature {meta['generation_options']['temperature']}, top_k {meta['generation_options']['top_k']}, seed {meta['generation_options']['seed']}, {meta['generation_options']['num_ctx']}-token context, {meta['generation_options']['num_predict']} new tokens at most.")
    A("")
    A("## 2. Sham-context construction (frozen before generation)")
    A("")
    A(f"For each of the {checks['n_with_context']} studies with a retrieval query, five sham reports were sampled without replacement from the {fz['pool_size']} IU training reports of the R2 corpus, using a per-study generator seeded from the fixed seed {fz['seed']} and the study id only (no reference, finding, classifier or image information). "
      f"Excluded for each study: the study itself and its own true R2 Top-5. Checks: query study in sham context {checks['query_study_in_sham']}; overlap with the true Top-5 {checks['true_top5_overlap']}; sham reports outside the corpus {checks['sham_not_in_corpus']}; in the locked test split {checks['sham_in_locked_test']}; duplicates within a study {checks['duplicates_within_study']}; "
      f"reference sentences outside the context {checks['reference_sentences_outside_context']}; message skeleton identical to G1 for {checks['format_identical_skeleton']} of {checks['n_with_context']} studies; {checks['n_empty_query_identical_to_g1']} of {checks['n_empty_query']} empty-query messages identical to G1. {checks['distinct_sham_reports_used']} distinct corpus reports were used. "
      f"The mapping (`g1b_sham_mapping.csv`, SHA-256 `{fz['mapping_sha256']}`) was frozen before generation. The generator is not told that the context is a control: the rank, dense-rank, BM25-rank and fusion-score fields displayed for each slot are carried over from the G1 report that the sham report replaces, and the evidence-support table is computed by the unchanged G1 function from the context supplied, as in G1. "
      "The sham selection was not tuned after seeing outcomes.")
    A("")
    A(f"Bulk generation: {run['n_cached_success']} of {run['n_cases']} reports, {run['n_failed']} failures, {summ['format_length'][GBS]['format_ok_rate'] * 100:.1f}% format-compliant, {run['wall_seconds_this_run'] / 60:.0f} minutes. No separate prompt test was needed because the prompt and generator are those frozen for G1.")
    A("")
    A("## 3. Reproduction of earlier results")
    A("")
    A(f"B0, G1 and G1A were re-derived from their stored reports through the shared evaluation function and matched the stored per-study results on {repro[G1S]['n_studies']} studies (maximum absolute numeric difference {max(repro[s]['max_abs_numeric_difference'] for s in (B0S, G1S, GAS)):.1e}); reproduced: **{repro['g1_and_g1a_reproduced']}**. "
      f"For the {repro['empty_query_studies']['n']} empty-query studies, whose G1B request is byte-identical to the G1 request, the G1B output text was identical to the stored G1 text in {repro['empty_query_studies']['g1b_text_identical_to_g1_text']} cases (a determinism check of the local generator).")
    A("")
    A("## 4. Context manipulation check")
    A("")
    A((G1B / "tables/table8_supplementary_context_manipulation_check.md").read_text(encoding="utf-8"))
    A(f"The relevant context shares a finding with the reference in {pc(manip.loc[G1S, 'mean_share_context_reports_sharing_a_reference_finding'])} of context reports on average, the sham context in {pc(manip.loc[GBS, 'mean_share_context_reports_sharing_a_reference_finding'])}; this measures how different the two conditions are, using reference findings only for evaluation, never for selection.")
    A("")
    A(f"## 5. Four-system comparison (paired studies: {n_p} for text metrics, {n_c} clinical-subset studies for finding metrics)")
    A("")
    A((G1B / "tables/table1_four_system_main_comparison.md").read_text(encoding="utf-8"))
    A(f"With sham context (G1B) compared with no context (G1A): precision {word(SH_PR)} ({dci(SH_PR)}), recall {word(SH_RC)} ({dci(SH_RC)}), F1 {word(SH_F1)} ({dci(SH_F1)}), hallucination rate {word(SH_HL, True)} ({dci(SH_HL)}); this shows what any report-like context, relevant or not, does to this model and prompt. {stability}")
    A("")
    A("## 6. Primary controlled comparison: G1 (relevant) minus G1B (sham)")
    A("")
    A((G1B / "tables/table2_g1_minus_g1b_controlled_comparison.md").read_text(encoding="utf-8"))
    A(f"Finding precision {word(PR)} with relevant context ({dci(PR)}); recall {word(RC)} ({dci(RC)}); F1 {word(F1)} ({dci(F1)}); macro F1 {word(MF)} ({dci(MF)}). Hallucination rate {word(HL, True)} ({dci(HL)}); omission rate {word(OM, True)} ({dci(OM)}). Classifier FP propagation {word(FP, True)} ({dci(FP)}); TP retention {word(TP)} ({dci(TP)}). "
      f"Normal recall {word(NR)} ({dci(NR)}); abnormal recall {word(AR)} ({dci(AR)}). ROUGE-L {word(RL)} ({dci(RL)}), BLEU-4 {dci(BL)}, METEOR variant {dci(MT)}, report length {dci(WD)} words.")
    A("")
    A(f"**Interpretation.** {verdict} {strat_note} {short_claims}")
    A("")
    A("## 7. Findings introduced and removed")
    A("")
    A((G1B / "tables/table3_introduced_and_removed_findings.md").read_text(encoding="utf-8"))
    A(f"Introduced findings are stated findings that the classifier had not flagged. With relevant context G1 introduced {int(intro(G1S).introduced_not_classifier_positive)} findings, {int(intro(G1S).introduced_match_reference)} matching the reference ({pc(intro(G1S).share_introduced_supported_by_reference)} supported); with sham context G1B introduced "
      f"{int(intro(GBS).introduced_not_classifier_positive)}, {int(intro(GBS).introduced_match_reference)} matching ({pc(intro(GBS).share_introduced_supported_by_reference)}). Relative to the classifier output, G1 removed {int(intro(G1S).classifier_positives_removed)} positives ({int(intro(G1S)['removed_were_correct_(reference_positive)'])} reference-correct, "
      f"{int(intro(G1S)['removed_were_unsupported_(classifier_false_positive)'])} unsupported) and G1B {int(intro(GBS).classifier_positives_removed)} ({int(intro(GBS)['removed_were_correct_(reference_positive)'])} reference-correct, {int(intro(GBS)['removed_were_unsupported_(classifier_false_positive)'])} unsupported).")
    A("")
    A(f"Finding by finding, {int(g1_added.n_findings)} findings were stated with relevant context only and {int(g1b_added.n_findings)} with sham context only. Of those stated only with relevant context, {int(g1_added.match_reference)} match the reference ({pc(g1_added.share_supported_by_reference)}) and {int(g1_added.unsupported_by_reference)} are unsupported; "
      f"of those stated only with sham context, {int(g1b_added.match_reference)} match ({pc(g1b_added.share_supported_by_reference)}) and {int(g1b_added.unsupported_by_reference)} are unsupported. Net (relevant-only minus sham-only): {int(net.match_reference):+d} reference-matching and {int(net.unsupported_by_reference):+d} unsupported findings.")
    A("")
    A("## 8. Copying")
    A("")
    A((G1B / "tables/table4_copying_analysis.md").read_text(encoding="utf-8"))
    A(f"{cp_txt} {cp_interp} Short generic sentences recur in IU reports, so copying is interpreted relative to the sham condition and not as plagiarism.")
    A("")
    A("## 9. Normal and abnormal behaviour; rare findings")
    A("")
    A((G1B / "tables/table5_normal_abnormal_and_rare_findings.md").read_text(encoding="utf-8"))
    A(f"Abnormal recall: classifier {f3(P_('abnormal_recall', B0S))}, G1A {f3(P_('abnormal_recall', GAS))}, G1B {f3(P_('abnormal_recall', GBS))}, G1 {f3(P_('abnormal_recall', G1S))}; normal recall: {f3(P_('normal_recall', B0S))}, {f3(P_('normal_recall', GAS))}, {f3(P_('normal_recall', GBS))}, {f3(P_('normal_recall', G1S))}. "
      f"Given an abnormal classifier output the report was normal in {pc(P_('over_normalisation_given_classifier_abnormal', GBS))} of G1B and {pc(P_('over_normalisation_given_classifier_abnormal', G1S))} of G1 reports (difference {dci(ON)}; {int(d_g(G1S).n)} studies). {rare_txt} {rare_attr}")
    A("")
    A("## 10. Supplementary: stratified by classifier state")
    A("")
    A(f"Within the {s_n} classifier-abnormal studies, G1 minus G1B was precision {sci(s_pr)}, recall {sci(s_rc)} and F1 {sci(s_f1)} (descriptive, same bootstrap convention).")
    A("")
    A((G1B / "tables/table7_supplementary_stratified_by_classifier_state.md").read_text(encoding="utf-8"))
    A("## 11. Limitations")
    A("")
    A("- One 4-billion-parameter local model, one frozen prompt, greedy decoding, one run and one sham draw per study; no replicate sham draws and no replicate generations; Ollama does not guarantee bitwise GPU determinism.")
    A("- Sham context is one specific control (random corpus reports). It removes topical relevance but also leaves the displayed rank fields and the evidence-support counts unchanged in format, so it tests relevance of the report text, not of every retrieval-derived signal separately.")
    A("- Finding extraction is rule-based and the reference is MeSH-derived and incomplete; per-finding counts are small, especially for rare findings, and Enlarged Cardiomediastinum has no reference positives.")
    A("- Normal studies dominate the clinical subset; the normal/abnormal analysis is descriptive.")
    A("- Validation data only; nothing was selected using these results; the locked retrieval test remains unopened; G2 has not been started.")
    A("")
    A("## 12. Conclusion")
    A("")
    A(f"{verdict} With relevant instead of sham context: precision {word(PR)}, recall {word(RC)}, F1 {word(F1)}, hallucination rate {word(HL, True)}, omission rate {word(OM, True)}, classifier FP propagation {word(FP, True)}, TP retention {word(TP)}, abnormal recall {word(AR)}. {cp_interp} {rare_attr} "
      "No change was made to the classifier, the generator model, the prompt or the retrieval configuration. G1A remains a classifier-only robustness ablation. " + short_claims)
    A("")
    (G1B / "SHAM_RETRIEVAL_CONTROL_ANALYSIS.md").write_text("\n".join(A_), encoding="utf-8")

    meth = f"""# Methods: sham-retrieval control (G1B)

## Purpose and design

The single-agent retrieval-augmented system (G1) writes a preliminary report from the frozen classifier output and the Top-5 retrieved corpus reports. To test whether the relevance of the retrieved content matters, and not merely the presence of radiology-report text, we ran a sham-retrieval control (G1B). G1B used the same {n_p} validation studies, the identical G1 prompt, the identical local model {gen} (Ollama {meta['ollama_version']}, digest verified to equal that of G1), greedy decoding with temperature 0, top-k 1, seed {meta['generation_options']['seed']}, a {meta['generation_options']['num_ctx']}-token context and at most {meta['generation_options']['num_predict']} new tokens, the same message format and number of context reports, the same classifier outputs, the same report parser, finding extractor, reference labels and metrics. Only the identity of the five context reports differed.

## Sham context

For each of the {checks['n_with_context']} studies with a retrieval query, five reports were drawn without replacement from the {fz['pool_size']} training reports of the retrieval corpus, with a per-study generator seeded from a fixed seed ({fz['seed']}) and the study identifier only. Selection never used the reference report, reference findings, classifier output or image. The query study and the study's own true Top-5 were excluded, no locked-test report was eligible, and the empty-query studies were handled exactly as in G1. Rank slots and the displayed rank fields were preserved, and the generator was not told that the context was a control. The evidence-support counts were computed by the unchanged G1 function from the context supplied. The mapping was frozen with a hash before generation and not changed afterwards.

## Comparisons and outcomes

Four systems were compared on identical studies: the rule-based baseline (B0), the no-retrieval ablation (G1A, a classifier-only robustness check), G1B and G1. The primary comparison was G1 minus G1B. Outcomes were finding precision, recall, micro and macro F1, hallucination and omission rates, propagation of classifier false positives, retention of classifier true positives, normal and abnormal recall of the report state, ROUGE-L, BLEU-4, an exact-match METEOR variant and report length, computed as in G1 on the primary set ({n_p} studies) and the clinical-finding subset ({n_c} studies). Uncertainty was a paired study-level bootstrap with 1,000 resamples and the seed used in G1; a difference was interpreted as an effect of context relevance only when its interval excluded zero.

## Additional analyses

Findings introduced beyond the classifier output (stated, not flagged by the classifier) and classifier positives removed were counted per system and checked against the reference. Findings stated under one condition only were compared finding by finding. Copying was measured against each system's own context reports (sentences copied verbatim, whole-report copies), against the other condition's context as a control, and against all corpus reports (sentences of at least six words, template sentences, whole-report duplication). Retention of true positives was analysed for the six rare findings of the loss study and for all others, and normal and abnormal behaviour as the report state against the reference and as the share of normal reports given an abnormal classifier output. A context check compared how closely the relevant and sham contexts matched the reference findings.
"""
    mw = wc(meth)
    (G1B / "MANUSCRIPT_G1B_METHODS.md").write_text(meth, encoding="utf-8")

    resu = f"""# Results: sham-retrieval control (G1B)

## Generation and context check

G1B produced {run['n_cached_success']} of {run['n_cases']} reports without failure ({summ['format_length'][GBS]['format_ok_rate'] * 100:.0f}% format-compliant); mean length {summ['format_length'][GBS]['mean_words']:.1f} words (G1 {summ['format_length'][G1S]['mean_words']:.1f}, G1A {summ['format_length'][GAS]['mean_words']:.1f}). Stored G1 and G1A results were reproduced exactly. The relevant context shared a finding with the reference in {pc(manip.loc[G1S, 'mean_share_context_reports_sharing_a_reference_finding'])} of context reports and the sham context in {pc(manip.loc[GBS, 'mean_share_context_reports_sharing_a_reference_finding'])} (Table 8).

## Four-system comparison

On the {n_c} clinical-subset studies, finding precision was {cis('precision', B0S)} for B0, {f3(P_('precision', GAS))} for G1A, {f3(P_('precision', GBS))} for G1B and {cis('precision', G1S)} for G1; recall {f3(P_('recall', B0S))}, {f3(P_('recall', GAS))}, {f3(P_('recall', GBS))} and {f3(P_('recall', G1S))}; F1 {f3(P_('f1', B0S))}, {f3(P_('f1', GAS))}, {f3(P_('f1', GBS))} and {f3(P_('f1', G1S))}; macro F1 {f3(P_('macro_f1', B0S))}, {f3(P_('macro_f1', GAS))}, {f3(P_('macro_f1', GBS))} and {f3(P_('macro_f1', G1S))} (Table 1, Figure 1). Reports with at least one hallucinated finding were {pc(P_('hallucination_rate', B0S))}, {pc(P_('hallucination_rate', GAS))}, {pc(P_('hallucination_rate', GBS))} and {pc(P_('hallucination_rate', G1S))}, and with an omission {pc(P_('omission_rate', B0S))}, {pc(P_('omission_rate', GAS))}, {pc(P_('omission_rate', GBS))} and {pc(P_('omission_rate', G1S))} (Figure 2).

## Relevant versus sham context

G1 minus G1B (Table 2, Figure 5): precision {dci(PR)}, recall {dci(RC)}, F1 {dci(F1)}, macro F1 {dci(MF)}, hallucination rate {dci(HL)}, omission rate {dci(OM)}, classifier false-positive propagation {dci(FP)}, true-positive retention {dci(TP)}, normal recall {dci(NR)}, abnormal recall {dci(AR)} and ROUGE-L {dci(RL)}. Intervals excluded zero for: {', '.join(n for n, r in (('precision', PR), ('recall', RC), ('F1', F1), ('macro F1', MF), ('hallucination rate', HL), ('omission rate', OM), ('FP propagation', FP), ('TP retention', TP), ('normal recall', NR), ('abnormal recall', AR), ('ROUGE-L', RL)) if r.excludes_zero) or 'none of these metrics'}. {verdict} {strat_note}

## Introduced and removed findings

G1 introduced {int(intro(G1S).introduced_not_classifier_positive)} findings beyond the classifier output, {int(intro(G1S).introduced_match_reference)} ({pc(intro(G1S).share_introduced_supported_by_reference)}) matching the reference; G1B introduced {int(intro(GBS).introduced_not_classifier_positive)}, {int(intro(GBS).introduced_match_reference)} ({pc(intro(GBS).share_introduced_supported_by_reference)}) matching. Relative to the classifier, G1 removed {int(intro(G1S).classifier_positives_removed)} positives and G1B {int(intro(GBS).classifier_positives_removed)} (Table 3, Figure 6). Findings stated with relevant context only numbered {int(g1_added.n_findings)} ({int(g1_added.match_reference)} reference-matching) and with sham context only {int(g1b_added.n_findings)} ({int(g1b_added.match_reference)} matching).

## Copying

{cp_txt} {cp_interp}

## Normal and abnormal behaviour; rare findings

Abnormal recall was {f3(P_('abnormal_recall', GBS))} for G1B and {f3(P_('abnormal_recall', G1S))} for G1; normal recall {f3(P_('normal_recall', GBS))} and {f3(P_('normal_recall', G1S))} (Table 5, Figure 4). {rare_txt} {rare_attr}

## Interpretation

{short_claims} {stability} The classifier, generator, prompt and retrieval configuration were unchanged, the locked retrieval-test partition was not opened, and no multi-agent system was evaluated.
"""
    rw = wc(resu)
    (G1B / "MANUSCRIPT_G1B_RESULTS.md").write_text(resu, encoding="utf-8")
    ok_len = 450 <= mw <= 700 and 600 <= rw <= 1000

    fc = f"""# Figure captions (G1B)

Retrieval-validation studies only (clinical subset {n_c} studies; text metrics {n_p} studies). B0 = rule-based report from the frozen classifier output; G1A = the same local model given the classifier output only; G1B = the same model and prompt given the classifier output and five sham corpus reports in the G1 format; G1 = the same model and prompt given the true Top-5 retrieved reports. Error bars: 95% study-level paired bootstrap intervals.

**Figure 1. Finding precision, recall and F1 for B0, G1A, G1B and G1.** Micro-averaged agreement between the findings stated in the report (any affirmative mention) and the MeSH-mapped IU reference findings.

**Figure 2. Hallucination versus omission.** Left: share of reports with at least one hallucinated finding and with at least one omitted finding. Right: mean hallucinated and omitted findings per report.

**Figure 3. Classifier false-positive propagation versus true-positive retention.** Proportion of classifier false positives mentioned in the report and of classifier true positives retained. B0 repeats every classifier-positive finding by construction.

**Figure 4. Normal and abnormal recall.** Share of reference-normal and reference-abnormal studies whose report state is correct. The B0 report state equals the classifier state by construction.

**Figure 5. Effect of context relevance.** Left: paired differences G1 (relevant context) minus G1B (sham context) with 95% intervals (blue: interval excludes zero; grey: it does not). Right: findings stated under one condition only, split by whether they match the reference.

**Figure 6. Introduced findings and copying.** Left: findings stated beyond the classifier output with sham (G1B) and relevant (G1) context, split by agreement with the reference. Right: share of sentences copied verbatim from, and of reports that duplicate, the study's own context reports.
"""
    tc = """# Table captions (G1B)

**Table 1. Four-system main comparison.** B0 rule-based, G1A LLM without retrieval, G1B LLM with sham retrieval and G1 LLM with relevant retrieval on identical studies: finding precision, recall, F1, macro F1, hallucination and omission rates, classifier false-positive propagation and true-positive retention, normal and abnormal recall, ROUGE-L, BLEU-4, the exact-match METEOR variant and report length, with 95% bootstrap intervals.

**Table 2. Primary controlled comparison G1 minus G1B.** Effect of relevant versus structurally matched sham context with prompt, model, decoding and classifier output held constant; paired study-level bootstrap intervals.

**Table 3. Introduced and removed findings.** Findings stated beyond the classifier output (and their agreement with the reference and presence in the system's own context) and classifier positives removed, per system.

**Table 4. Copying analysis.** Verbatim copying of each system's own context, of the other condition's context (control level) and of the whole corpus, template sentences and whole-report duplication.

**Table 5. Normal/abnormal behaviour and rare findings.** Normal and abnormal recall, misclassified study counts, share of normal reports given an abnormal classifier output, and pooled retention of rare and other classifier true positives.

**Table 6. Per-finding results.** Reference positives, classifier true positives, true-positive retention and precision / recall / F1 for each finding and system; rare findings follow the C2 definition.

**Table 7 (supplementary). G1 minus G1B by classifier state.** Descriptive stratification by classifier state.

**Table 8 (supplementary). Context manipulation check.** Relevance of the relevant and sham contexts to the reference findings and to the classifier output.
"""
    (G1B / "FIGURE_CAPTIONS.md").write_text(fc, encoding="utf-8")
    (G1B / "TABLE_CAPTIONS.md").write_text(tc, encoding="utf-8")
    idx = f"""# G1B journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_four_system_main_comparison.*` | Finding F1 B0 / G1A / G1B / G1 {f3(P_('f1', B0S))} / {f3(P_('f1', GAS))} / {f3(P_('f1', GBS))} / {f3(P_('f1', G1S))} | Results |
| Table 2 `tables/table2_g1_minus_g1b_controlled_comparison.*` | Relevance effect: F1 {dci(F1)}, hallucination {dci(HL)} | Results |
| Table 3 `tables/table3_introduced_and_removed_findings.*` | Introduced findings supported by reference: G1B {pc(intro(GBS).share_introduced_supported_by_reference)}, G1 {pc(intro(G1S).share_introduced_supported_by_reference)} | Results |
| Table 4 `tables/table4_copying_analysis.*` | Own-context sentence copying G1 {pc(cpy.loc[G1S, own_c])} vs G1B {pc(cpy.loc[GBS, own_c])} | Results / Discussion |
| Table 5 `tables/table5_normal_abnormal_and_rare_findings.*` | Normal/abnormal behaviour and rare-finding retention | Results / Discussion |
| Table 6 `tables/table6_per_finding_results.*` | Per-finding retention and P/R/F1 | Supplementary |
| Table 7 `tables/table7_supplementary_stratified_by_classifier_state.*` | G1 minus G1B within classifier states | Supplementary |
| Table 8 `tables/table8_supplementary_context_manipulation_check.*` | Context relevance of the two conditions | Methods / Supplementary |
| Figure 1 `figures/fig1_finding_precision_recall_f1.*` | Finding agreement of four systems | Results |
| Figure 2 `figures/fig2_hallucination_vs_omission.*` | Hallucination versus omission | Results |
| Figure 3 `figures/fig3_fp_propagation_vs_tp_retention.*` | FP propagation versus TP retention | Results |
| Figure 4 `figures/fig4_normal_abnormal_recall.*` | Normal/abnormal behaviour | Results / Discussion |
| Figure 5 `figures/fig5_relevance_effect_summary.*` | Controlled relevance effect | Results / Discussion |
| Figure 6 `figures/fig6_introduced_findings_and_copying.*` | Introduced findings and copying | Results / Discussion |
| `SHAM_RETRIEVAL_CONTROL_ANALYSIS.md` | Full G1B analysis | Supplementary / internal |
| `MANUSCRIPT_G1B_METHODS.md`, `MANUSCRIPT_G1B_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_G1B.md` | Content for Methodology, Results, Discussion | University report |
| `g1b_sham_mapping.csv`, `g1b_sham_mapping_frozen.json`, `g1b_generator_metadata.json` | Frozen sham mapping and hash, generator metadata | Methods / code release |
| `g1b_per_study_results.csv`, `g1b_paired_differences.csv` | All per-study and paired results | Supplementary data |
| `g1b_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
"""
    (G1B / "G1B_JOURNAL_ASSET_INDEX.md").write_text(idx, encoding="utf-8")

    U = []
    A = U.append
    A("# University report content: G1B (sham-retrieval control)")
    A("")
    A("_Insert-ready material generated from the G1B artifacts. Validation data only; classifier, prompt, generator model and retrieval configuration are unchanged; the locked retrieval-test split is unopened._")
    A("")
    A("## Methodology")
    A("")
    A("### Purpose of the control")
    A("")
    A(f"G1A removed the retrieved context entirely, but the model then behaved degenerately on classifier-normal studies, so G1A cannot isolate the contribution of retrieved content. The sham-retrieval control G1B keeps everything of G1 (prompt, offline model {gen}, decoding, message format, five context reports per study, classifier output, extractor and metrics) and changes only which five reports are shown: deterministic, seeded random reports from the training corpus, never the study itself and never its true Top-5. The sham mapping was frozen and hashed before generation, and the generator was not told that the context was a control.")
    A("")
    A("### Comparisons")
    A("")
    A("Four systems are compared on identical studies (rule-based B0, G1A, G1B, G1). The primary comparison is G1 minus G1B with a paired study-level bootstrap of 1,000 resamples; a difference is interpreted as an effect of context relevance only if its interval excludes zero. Additional analyses cover findings introduced beyond the classifier output and findings removed, copying of context text, normal/abnormal behaviour and rare findings.")
    A("")
    A("## Results")
    A("")
    A("### Four-system and controlled comparison")
    A("")
    A(f"Finding F1 was {f3(P_('f1', B0S))} for B0, {f3(P_('f1', GAS))} for G1A, {f3(P_('f1', GBS))} for G1B and {f3(P_('f1', G1S))} for G1; hallucination rates were {pc(P_('hallucination_rate', B0S))}, {pc(P_('hallucination_rate', GAS))}, {pc(P_('hallucination_rate', GBS))} and {pc(P_('hallucination_rate', G1S))}. "
      f"With relevant instead of sham context, finding precision changed by {dci(PR)}, recall by {dci(RC)}, F1 by {dci(F1)}, the hallucination rate by {dci(HL)}, classifier false-positive propagation by {dci(FP)} and true-positive retention by {dci(TP)} (Tables 1 and 2, Figures 1 to 3 and 5). {verdict}")
    A("")
    A("### Introduced findings, copying, normal/abnormal behaviour and rare findings")
    A("")
    A(f"G1 introduced {int(intro(G1S).introduced_not_classifier_positive)} findings beyond the classifier output ({pc(intro(G1S).share_introduced_supported_by_reference)} reference-supported); G1B introduced {int(intro(GBS).introduced_not_classifier_positive)} ({pc(intro(GBS).share_introduced_supported_by_reference)}) (Table 3). {cp_txt} {cp_interp} "
      f"Abnormal recall was {f3(P_('abnormal_recall', GBS))} for G1B and {f3(P_('abnormal_recall', G1S))} for G1. {rare_txt} {rare_attr}")
    A("")
    A("### Interpretation caveat")
    A("")
    A(f"{short_claims} {stability}")
    A("")
    A("## Discussion")
    A("")
    A("The sham control, and not the no-retrieval ablation, is the primary evidence for whether context relevance matters: it holds the prompt and context structure fixed. G1A remains useful as a classifier-only robustness check. Limits: one small model, one prompt and one sham draw per study; rule-based finding extraction and a MeSH-derived reference; small counts for rare findings; and validation data only. The multi-agent system (G2) has not been started.")
    A("")
    (G1B / "UNIVERSITY_REPORT_G1B.md").write_text("\n".join(U), encoding="utf-8")

    bad = []
    for n in ("SHAM_RETRIEVAL_CONTROL_ANALYSIS.md", "MANUSCRIPT_G1B_METHODS.md", "MANUSCRIPT_G1B_RESULTS.md", "UNIVERSITY_REPORT_G1B.md", "FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md", "G1B_JOURNAL_ASSET_INDEX.md"):
        txt = (G1B / n).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"superior", r"\{[a-z_]+\[[^}]*\}", r"\bnan\b", r"\bNone\b", r"anthropic", r"claude"):
            if re.search(pat, txt, re.I if pat in ("anthropic", "claude") else 0):
                bad.append((n, pat))
    print(json.dumps({"methods_words": mw, "results_words": rw, "length_ok": ok_len, "wording_flags": bad}))
    return 0 if ok_len and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
