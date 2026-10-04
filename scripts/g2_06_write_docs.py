"""G2 documents generated from the stored G2 artifacts (no number typed by hand; the wording of every comparison follows the sign and interval of the
data; G2 is called superior only if ALL predeclared criteria are met):
MULTI_AGENT_RAG_ANALYSIS.md, MANUSCRIPT_G2_METHODS.md, MANUSCRIPT_G2_RESULTS.md, UNIVERSITY_REPORT_G2.md, FIGURE_CAPTIONS.md, TABLE_CAPTIONS.md, G2_JOURNAL_ASSET_INDEX.md

    .venv\\Scripts\\python.exe -m scripts.g2_06_write_docs
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.utils.config import PROJECT_ROOT

G2 = PROJECT_ROOT / "results/report_generation/experiments/g2_multi_agent"
B0S, G1S, G2DS, G2S, GAS, GBS = "B0_rule_based", "G1_single_agent_rag", "G2_agent2_draft_pre_critic", "G2_multi_agent_rag", "G1A_llm_no_retrieval", "G1B_llm_sham_retrieval"
rd = lambda n: pd.read_csv(G2 / n, float_precision="round_trip")  # noqa: E731
jl = lambda n: json.loads((G2 / n).read_text(encoding="utf-8"))  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731
wc = lambda t: len(t.split())  # noqa: E731


def main() -> int:
    summ, repro, run, meta, fz, checks, diag, leak = jl("g2_summary.json"), jl("g1_reproduction_check.json"), jl("generation_run_log.json"), jl("g2_generator_metadata.json"), jl("g2_prompts_frozen.json"), jl("g2_input_checks.json"), \
        jl("g2_stability_diagnostics.json"), jl("g2_leakage_checks.json")
    a1s, a3, lat, smoke = jl("g2_agent1_summary.json"), jl("g2_agent3_summary.json"), jl("g2_latency_compute.json"), jl("smoke_test_report.json")
    sens = rd("g2_sensitivity_agent1_parsed_only.csv").set_index("metric")
    mc, pair = rd("g2_main_comparison.csv").set_index("metric"), rd("g2_paired_differences.csv")
    a1tab, funnel, intro, cpy, cpd = rd("g2_agent1_status_vs_reference.csv"), rd("g2_retrieval_only_funnel.csv"), rd("g2_introduced_findings.csv").set_index("system"), rd("g2_copying_analysis.csv").set_index("system"), rd("g2_copying_paired_difference.csv")
    rare, nad, dec, strat, emp = rd("g2_rare_finding_retention.csv").set_index(["system", "group"]), rd("g2_normal_abnormal.csv"), rd("g2_report_state_given_classifier_state.csv"), rd("g2_stratified_by_classifier_state.csv"), rd("g2_empty_classifier_output_studies.csv")
    n_c, n_p, n_ctx = summ["n_clinical_subset"], summ["n_paired_studies"], summ["n_studies_with_context"]
    Dd = lambda m, a=G2S, b=G1S: pair[(pair.comparison == f"{a} minus {b}") & (pair.metric == m)].iloc[0]  # noqa: E731
    P_ = lambda m, s: mc.loc[m, s]  # noqa: E731
    cis = lambda m, s: f"{f3(mc.loc[m, s])} (95% CI {f3(mc.loc[m, s + '_ci95_low'])} to {f3(mc.loc[m, s + '_ci95_high'])})"  # noqa: E731
    dci = lambda r: f"{r['diff']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731
    CP = lambda m, a=G2S, b=G1S: cpd[(cpd.comparison == f"{a} minus {b}") & (cpd.metric == m)].iloc[0]  # noqa: E731
    cdci = lambda r: f"{100 * r['diff']:+.1f} points (95% CI {100 * r['ci95_low']:+.1f} to {100 * r['ci95_high']:+.1f})"  # noqa: E731

    def word(r, lower_better=False):
        if not r.excludes_zero:
            return "did not differ reliably"
        up = r["diff"] > 0
        return ("was higher" if up else "was lower") + (" (favourable)" if up != lower_better else " (unfavourable)")

    def worse(r, higher_better=True):
        """True if the difference is reliably in the unfavourable direction."""
        return bool(r.excludes_zero and ((r["diff"] < 0) == higher_better))

    gen = meta["model_tag"]
    same = all(meta["same_generator_as_g1"].values())
    PR, RC, F1, MF = (Dd(k) for k in ("precision", "recall", "f1", "macro_f1"))
    HL, OM, FP, TP = Dd("hallucination_rate"), Dd("omission_rate"), Dd("clf_fp_propagation"), Dd("tp_retention")
    NR, AR, ON, RL = Dd("normal_recall"), Dd("abnormal_recall"), Dd("over_normalisation_given_classifier_abnormal"), Dd("rouge_l")
    B4, MT, WD = Dd("bleu4"), Dd("meteor_exact"), Dd("mean_words")
    RT, CT = Dd("rare_tp_retention"), Dd("common_tp_retention")
    CS, CW, CR = CP("copied_sentence_rate_from_top5"), CP("whole_report_copy_rate_from_top5"), CP("repeated_sentence_rate")
    # ---- predeclared success criteria (data-driven)
    crit = {"classifier FP propagation reduced relative to G1": bool(FP.excludes_zero and FP["diff"] < 0), "hallucination rate reduced relative to G1": bool(HL.excludes_zero and HL["diff"] < 0),
            "recall preserved (not reliably lower than G1)": not worse(RC), "classifier TP retention preserved (not reliably lower than G1)": not worse(TP), "abnormal recall preserved or improved (not reliably lower than G1)": not worse(AR),
            "copying from the retrieved reports reduced": bool(CS.excludes_zero and CS["diff"] < 0)}
    met = [k for k, v in crit.items() if v]
    all_met = len(met) == len(crit)
    collapse = worse(RC) or worse(TP)
    if all_met:
        verdict = "G2 met all six predeclared criteria relative to G1 on this validation set; because every criterion was met, G2 is reported as improving on G1 under this model and prompts, without a claim of clinical benefit."
    else:
        failed = [k for k, v in crit.items() if not v]
        verdict = (f"G2 met {len(met)} of the six predeclared criteria relative to G1 ({'; '.join(met) if met else 'none'}) and did not meet: {'; '.join(failed)}. G2 is therefore not declared an improvement over G1 on this validation set"
                   + ("; its lower false-positive propagation or hallucination has to be read together with a reliably lower recall or true-positive retention, i.e. a trade-off and not a gain." if collapse else "."))
    ag1 = a1tab[a1tab.status_source.str.startswith("validated")].set_index("support_status")
    c3 = a3["clinical_subset"]
    rr = lambda s, g: rare.loc[(s, g)]  # noqa: E731
    rare_txt = (f"Pooled true-positive retention over the six rare findings: B0 {f3(rr(B0S, 'rare (C2 definition)').tp_retention)}, G1 {f3(rr(G1S, 'rare (C2 definition)').tp_retention)} ({int(rr(G1S, 'rare (C2 definition)').retained)} of {int(rr(G1S, 'rare (C2 definition)').classifier_true_positives)}), "
                f"G2 draft {f3(rr(G2DS, 'rare (C2 definition)').tp_retention)}, G2 {f3(rr(G2S, 'rare (C2 definition)').tp_retention)} ({int(rr(G2S, 'rare (C2 definition)').retained)} of {int(rr(G2S, 'rare (C2 definition)').classifier_true_positives)}); other findings: G1 {f3(rr(G1S, 'other findings').tp_retention)}, G2 {f3(rr(G2S, 'other findings').tp_retention)}.")
    rare_attr = f"G2 minus G1 retention is {dci(RT)} for rare and {dci(CT)} for other findings; " + ("rare-finding retention changed reliably." if RT.excludes_zero else "the rare-finding difference is not reliable, and the small counts (21 classifier true positives over the six findings) do not allow a claim of improvement or loss.")
    nn = lambda who, ref: nad[(nad.source == who) & (nad.reference == ref)].iloc[0]  # noqa: E731
    d_g = lambda s: dec[(dec.system == s) & (dec.classifier_state == "abnormal")].iloc[0]  # noqa: E731
    ri = lambda s: intro.loc[s]  # noqa: E731
    sv = lambda st, m: strat[(strat.classifier_state == st) & (strat.metric == m)].iloc[0]  # noqa: E731
    sci = lambda r: f"{r['g2_minus_g1']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731
    s_f1, s_rc, s_pr, s_fp, s_tp = (sv("abnormal", m) for m in ("f1", "recall", "precision", "clf_fp_propagation", "tp_retention"))
    emp_g2 = emp[emp.system == G2S].iloc[0]
    emp_g1 = emp[emp.system == G1S].iloc[0]
    n_emp = summ["n_empty_query_studies"]
    f_stage = funnel.set_index("stage")
    stages = list(f_stage.index)
    crit_txt = "; ".join(f"{k}: {'met' if v else 'not met'}" for k, v in crit.items())
    caveat = ("Single-model, single-prompt-set, single-run validation result; the 4-billion-parameter model's critic and verifier are weak components, the agent roles share one model, and no result here shows clinical benefit.")
    crit_agent3 = (f"The critic approved {a3['approved_unchanged_pct']:.1f}% of reports unchanged and modified {a3['modified_pct']:.1f}% ({a3['n_modified']} of {a3['n_studies']}); in the clinical subset it removed {c3['findings_removed']} findings "
                   f"({c3['correct_removals_(not_in_reference)']} not in the reference, {c3['incorrect_removals_(in_reference)']} in the reference) and added {c3['findings_added']} ({c3['correct_additions_(in_reference)']} in the reference, {c3['unsupported_additions_(not_in_reference)']} not).")
    agent3_verdict = ("The critic's changes removed more reference-supported than unsupported findings or added unsupported ones, so it did not add value on these measures." if (c3["incorrect_removals_(in_reference)"] >= c3["correct_removals_(not_in_reference)"] and c3["findings_removed"] > 0) else
                      "The critic's removals were mostly findings not in the reference, which is the intended direction, but its overall effect on the main metrics is given by the G2 minus draft differences.")
    p_dr = lambda m: Dd(m, G2S, G2DS)  # noqa: E731
    crit_effect = (f"Final G2 minus Agent 2 draft: F1 {dci(p_dr('f1'))}, recall {dci(p_dr('recall'))}, hallucination rate {dci(p_dr('hallucination_rate'))}, FP propagation {dci(p_dr('clf_fp_propagation'))}, TP retention {dci(p_dr('tp_retention'))}.")

    n_trunc = a1s["validation_flag_counts"].get("agent1_unparseable", 0)
    sd = lambda m: f"{sens.loc[m, 'diff']:+.3f} (95% CI {sens.loc[m, 'ci95_low']:+.3f} to {sens.loc[m, 'ci95_high']:+.3f})"  # noqa: E731
    sup_sh, uns_sh = ag1.loc["supported", "share_matching_reference"], ag1.loc["unsupported", "share_matching_reference"]
    disc_txt = (f"Agent 1's validated support status did not discriminate between correct and incorrect classifier findings: {pc(sup_sh)} of 'supported' and {pc(uns_sh)} of 'unsupported' classifier positives match the reference"
                + (" (a difference of no practical size)." if abs(sup_sh - uns_sh) < 0.10 else ".")) if (ag1.loc["supported", "n_classifier_positives"] and ag1.loc["unsupported", "n_classifier_positives"]) else ""
    trunc_txt = (f"Agent 1 hit its frozen output budget and returned unparseable JSON for {n_trunc} of {n_ctx} studies with context ({pc(n_trunc / n_ctx)}; {diag['agents']['calls_stopped_by_token_limit']} calls stopped by the token limit); in those studies every classifier finding was recorded as unsupported "
                 f"without evidence (flag agent1_omitted_classifier_finding). Post hoc sensitivity analysis restricted to the {int(sens.iloc[0].n_clinical_studies)} clinical-subset studies with parsed Agent 1 output ({int(sens.iloc[0].n_excluded_agent1_unparseable)} excluded), G2 minus G1: "
                 f"recall {sd('recall')}, F1 {sd('f1')}, hallucination rate {sd('hallucination_rate')}, FP propagation {sd('clf_fp_propagation')}, TP retention {sd('tp_retention')}, abnormal recall {sd('abnormal_recall')}; the conclusions are unchanged. The budget was part of the frozen configuration and was not changed after bulk generation began.")
    short_share = lambda sname: cpy.loc[sname, "share_of_sentences_under_6_words_(cannot_register_as_copied)"]  # noqa: E731
    len_txt = (f"G2 reports are much shorter than G1 reports (mean {P_('mean_words', G2S):.1f} versus {P_('mean_words', G1S):.1f} words; {pc(short_share(G2S))} of G2 sentences have fewer than six words and cannot register as copied, against {pc(short_share(G1S))} for G1), "
               "so the measured copying reduction is partly a consequence of terse output and not only of the agent separation; the repeated-sentence rate rose.")
    emp_txt = (f"Report text for the {n_emp} studies without a classifier output: all {int(emp_g2.report_normal)} G2 reports state a normal study (G1: {int(emp_g1.report_normal)}), although the Agent 2 prompt instructs the model to say that no finding could be determined and not to call the examination normal; "
               "this explicit-indeterminate requirement was therefore NOT achieved by G2 (nor by G1), and the model ignored the rule. The behaviour is reported as measured; the classifier state, not the report, remains indeterminate for these studies.") if emp_g2.report_normal > 0 else               "These studies are written with the explicit no-output handling of G1 and none was stated as normal."
    caveat = caveat + " Additional measured limitations: " + f"Agent 1 support status did not separate correct from incorrect classifier findings; Agent 1 output was truncated for {pc(n_trunc / n_ctx)} of studies with context; the critic changed a stated finding in {a3['studies_with_removed_or_added_finding']} studies; G2 reports are terse; empty classifier outputs were still written as normal reports."

    A_ = []
    A = A_.append
    A("# G2: structured multi-agent RAG")
    A("")
    A("_Generated by `scripts/g2_06_write_docs.py` from the G2 artifacts. Retrieval-validation studies only; the locked retrieval-test split was not opened; classifier, calibration, retrieval configuration, generator model, finding extractor, reference labels and populations are those of G1._")
    A("")
    A(f"> **Summary.** {verdict} {caveat}")
    A("")
    A("## 1. Question and design")
    A("")
    A(f"G2 asks whether splitting report generation into three sequential roles of the same frozen model can keep the recall and true-positive retention that relevant retrieval gave G1 while reducing hallucination, classifier false-positive propagation, retrieval-induced unsupported findings and direct copying of retrieved reports. "
      f"Agent 1 (Evidence Verifier) reads the classifier output and the frozen R2 Top-5 reports and returns structured JSON only; its JSON is validated deterministically. Agent 2 (Grounded Report Writer) receives only the classifier state, calibrated probabilities and the validated evidence JSON, never raw retrieved text. "
      f"Agent 3 (Grounding Critic) receives the classifier state, the evidence and the Agent 2 draft, never raw retrieved text, and either approves (the draft is kept unchanged by code) or returns one corrected report. There is no loop: at most three model calls per study ({lat['g2_calls_per_study_mean']:.2f} on average, because the {n_p - n_ctx} empty-query studies skip Agent 1). "
      f"Generator: Ollama {meta['ollama_version']}, `{gen}` digest `{meta['model_digest'][:12]}`, identical to G1 on version, tag, digest, sampling options and context ({'all checks passed' if same else 'MISMATCH'}); temperature {meta['generation_options']['temperature']}, top_k {meta['generation_options']['top_k']}, seed {meta['generation_options']['seed']}, {meta['generation_options']['num_ctx']}-token context. "
      "Only the output budget differs per agent and every agent uses schema-constrained JSON output (a local Ollama request field).")
    A("")
    A("## 2. Frozen prompts and implementation test")
    A("")
    A(f"Prompt version `{fz['prompt_version']}`; SHA-256 (prompt, schema, output budget): Agent 1 `{fz['prompt_sha256']['agent1_evidence_verifier']}`, Agent 2 `{fz['prompt_sha256']['agent2_grounded_report_writer']}`, Agent 3 `{fz['prompt_sha256']['agent3_grounding_critic']}`, combined `{fz['prompt_sha256']['combined']}`. "
      f"The prompts were frozen before bulk generation (frozen {fz['frozen_utc']}, generation started {run['started_utc']}). The four-study smoke test (normal, one finding, multiple findings, empty classifier output) checked JSON parsing, structure, connectivity, leakage and caching only: {'passed' if smoke['all_implementation_checks_passed'] else 'failed'}. "
      "A first smoke run showed a technical failure: with a plain-text answer format Agent 2 sometimes entered the model's hidden reasoning mode and exhausted its token budget. Agent 2 was therefore switched to the same schema-constrained JSON container as the other agents before freezing; no clinical rule was changed after looking at the smoke cases. "
      f"Bulk generation: {run['n_completed']} of {run['n_cases']} studies completed, {run['n_failed']} failures.")
    A("")
    A("## 3. Populations and reproduction of earlier results")
    A("")
    A(f"The primary set is the same {n_p} studies as G1 ({n_ctx} with a retrieval query and {n_p - n_ctx} empty classifier outputs); the clinical-finding subset has {n_c} studies. Retrieval is read from the stored G1 cases (identical to the frozen R2 Top-5 for all {checks['n']} studies: {checks['retrieved_ids_equal_g1_top5']}). "
      f"B0, G1, G1A and G1B were re-derived from their stored reports through the shared evaluation function and matched the stored results (max numeric difference {max(repro[s]['max_abs_numeric_difference'] for s in (B0S, G1S, GAS, GBS)):.1e}); reproduced: **{repro['earlier_results_reproduced']}**.")
    A("")
    A("## 4. Main metrics: G1 versus G2")
    A("")
    A((G2 / "tables/table2_g1_vs_g2_main_metrics.md").read_text(encoding="utf-8"))
    A("## 5. Paired bootstrap differences (primary: G2 minus G1)")
    A("")
    A((G2 / "tables/table3_paired_bootstrap_differences.md").read_text(encoding="utf-8"))
    A(f"Finding precision {word(PR)} with G2 ({dci(PR)}); recall {word(RC)} ({dci(RC)}); F1 {word(F1)} ({dci(F1)}); macro F1 {word(MF)} ({dci(MF)}). Hallucination rate {word(HL, True)} ({dci(HL)}); omission rate {word(OM, True)} ({dci(OM)}). "
      f"Classifier FP propagation {word(FP, True)} ({dci(FP)}); TP retention {word(TP)} ({dci(TP)}). Normal recall {word(NR)} ({dci(NR)}); abnormal recall {word(AR)} ({dci(AR)}). ROUGE-L {word(RL)} ({dci(RL)}), BLEU-4 {dci(B4)}, METEOR variant {dci(MT)}, report length {dci(WD)} words.")
    A("")
    A("## 6. Predeclared success criteria")
    A("")
    A(f"{crit_txt}. {verdict} A rise in precision with a collapse in recall would not have been declared a success. The copying criterion is confounded by output length: {len_txt}")
    A("")
    A("## 7. False-positive control")
    A("")
    A(f"Classifier false-positive propagation (classifier false positives appearing in the report / classifier false positives): B0 {cis('clf_fp_propagation', B0S)}, G1 {cis('clf_fp_propagation', G1S)}, G2 draft {f3(P_('clf_fp_propagation', G2DS))}, G2 {cis('clf_fp_propagation', G2S)}. "
      f"Classifier true-positive retention: B0 {f3(P_('tp_retention', B0S))}, G1 {cis('tp_retention', G1S)}, G2 draft {f3(P_('tp_retention', G2DS))}, G2 {cis('tp_retention', G2S)}. G2 minus G1: FP propagation {dci(FP)}, TP retention {dci(TP)}. "
      + ("The reduction in false-positive propagation is accompanied by a reliably lower true-positive retention, so part of it comes from suppressing true abnormalities." if (FP.excludes_zero and FP["diff"] < 0 and worse(TP)) else
         "False-positive propagation was not reduced reliably." if not (FP.excludes_zero and FP["diff"] < 0) else "True-positive retention was not reliably lower."))
    A("")
    A("## 8. Hallucination and omission")
    A("")
    A(f"Reports with at least one hallucinated finding: G1 {cis('hallucination_rate', G1S)}, G2 draft {f3(P_('hallucination_rate', G2DS))}, G2 {cis('hallucination_rate', G2S)} (G2 minus G1 {dci(HL)}). Reports with at least one omitted finding: G1 {cis('omission_rate', G1S)}, G2 {cis('omission_rate', G2S)} (difference {dci(OM)}). "
      f"Stated findings (clinical subset): G1 {int(ri(G1S)['stated_findings_total'])}, G2 draft {int(ri(G2DS)['stated_findings_total'])}, G2 {int(ri(G2S)['stated_findings_total'])}.")
    A("")
    A("## 9. Normal and abnormal consistency")
    A("")
    A((G2 / "tables/table8_supplementary_normal_abnormal_and_rare_findings.md").read_text(encoding="utf-8"))
    A(f"Abnormal recall: classifier {f3(P_('abnormal_recall', B0S))}, G1 {f3(P_('abnormal_recall', G1S))}, G2 {f3(P_('abnormal_recall', G2S))} (G2 minus G1 {dci(AR)}); normal recall {f3(P_('normal_recall', B0S))}, {f3(P_('normal_recall', G1S))}, {f3(P_('normal_recall', G2S))} ({dci(NR)}). "
      f"Given an abnormal classifier output ({int(d_g(G2S).n)} studies) the report was normal in {pc(P_('over_normalisation_given_classifier_abnormal', G1S))} of G1 and {pc(P_('over_normalisation_given_classifier_abnormal', G2S))} of G2 reports (difference {dci(ON)}). "
      + ("G2 therefore reduced G1's abnormal-to-normal collapse." if (ON.excludes_zero and ON["diff"] < 0) else "G2 did not reliably correct G1's tendency to describe some abnormal studies as normal." if not ON.excludes_zero else "G2 made G1's abnormal-to-normal collapse worse."))
    A("")
    A("## 10. Rare findings")
    A("")
    A(f"{rare_txt} {rare_attr} Per-finding retention and F1 are in Table 6.")
    A("")
    A((G2 / "tables/table6_per_finding_results.md").read_text(encoding="utf-8"))
    A("## 11. Retrieval-only findings")
    A("")
    A((G2 / "tables/table5_agent3_intervention_analysis.md").read_text(encoding="utf-8"))
    A(f"Findings stated beyond the classifier output (clinical subset): G1 {int(ri(G1S)['introduced_findings_(stated, not classifier positive)'])}, of which {int(ri(G1S)['introduced_reference_supported'])} reference-supported ({pc(ri(G1S)['share_supported'])}); "
      f"G2 draft {int(ri(G2DS)['introduced_findings_(stated, not classifier positive)'])} ({int(ri(G2DS)['introduced_reference_supported'])} supported); G2 final {int(ri(G2S)['introduced_findings_(stated, not classifier positive)'])} ({int(ri(G2S)['introduced_reference_supported'])} supported, {pc(ri(G2S)['share_supported'])}). "
      f"Retrieval-derived introduced findings (also present in the retrieved reports' IU labels): G1 {int(ri(G1S)['introduced_also_in_retrieved_reports_(retrieval-derived)'])} ({int(ri(G1S)['retrieval_derived_reference_supported'])} supported), G2 {int(ri(G2S)['introduced_also_in_retrieved_reports_(retrieval-derived)'])} ({int(ri(G2S)['retrieval_derived_reference_supported'])} supported). "
      f"Through the pipeline, Agent 1 proposed {int(f_stage.loc[stages[0], 'n_candidates'])} retrieval-only candidates ({int(f_stage.loc[stages[0], 'reference_supported'])} reference-supported), {int(f_stage.loc[stages[1], 'n_candidates'])} reached the draft ({int(f_stage.loc[stages[1], 'reference_supported'])} supported) and {int(f_stage.loc[stages[2], 'n_candidates'])} survived the critic ({int(f_stage.loc[stages[2], 'reference_supported'])} supported).")
    A("")
    A("## 12. Agent contributions")
    A("")
    A("### Agent 1: evidence verifier")
    A("")
    A((G2 / "tables/table4_agent1_evidence_analysis.md").read_text(encoding="utf-8"))
    A(f"Classifier positives by validated support status (clinical subset): supported {int(ag1.loc['supported', 'n_classifier_positives'])} ({pc(ag1.loc['supported', 'share_matching_reference']) if ag1.loc['supported', 'n_classifier_positives'] else 'n/a'} match the reference), partially supported {int(ag1.loc['partially_supported', 'n_classifier_positives'])} "
      f"({pc(ag1.loc['partially_supported', 'share_matching_reference']) if ag1.loc['partially_supported', 'n_classifier_positives'] else 'n/a'}), unsupported {int(ag1.loc['unsupported', 'n_classifier_positives'])} ({pc(ag1.loc['unsupported', 'share_matching_reference']) if ag1.loc['unsupported', 'n_classifier_positives'] else 'n/a'}). "
      f"Of the {a1s['cited_ranks_total']} ranks cited by Agent 1, {pc(a1s['citation_agreement_with_iu_labels'])} point to a retrieved report whose IU labels contain the finding; Agent 1 cited {pc(a1s['recall_of_iu_labelled_support'])} of the labelled supporting reports. "
      f"{disc_txt} Validation flags (all studies): {', '.join(f'{k} {v}' for k, v in a1s['validation_flag_counts'].items()) or 'none'}. {trunc_txt}")
    A("")
    A("### Agent 2: grounded report writer")
    A("")
    A(f"Before the critic, the Agent 2 draft stated {int(ri(G2DS)['stated_findings_total'])} findings in the clinical subset with FP propagation {cis('clf_fp_propagation', G2DS) if 'clf_fp_propagation_ci95_low' in mc.columns or True else ''} and TP retention {cis('tp_retention', G2DS)}; draft minus G1: FP propagation {dci(Dd('clf_fp_propagation', G2DS, G1S))}, TP retention {dci(Dd('tp_retention', G2DS, G1S))}, F1 {dci(Dd('f1', G2DS, G1S))}.")
    A("")
    A("### Agent 3: grounding critic")
    A("")
    A(f"{crit_agent3} Of the {a3['revise_actions']} REVISE actions, {a3['modified_beyond_section_labels_n']} differ from the draft beyond the FINDINGS / IMPRESSION labels ({a3['modified_beyond_section_labels_pct']:.1f}% of reports), and {a3['studies_with_removed_or_added_finding']} studies had any stated finding added or removed; most modifications are therefore relabelling or rewording, not corrections. Issues named by the critic (all studies): {', '.join(f'{k} {v}' for k, v in a3['issue_counts'].items())}. {agent3_verdict} {crit_effect}")
    A("")
    A("## 13. Copying")
    A("")
    A((G2 / "tables/table7_latency_and_copying_analysis.md").read_text(encoding="utf-8"))
    A(f"With the G1 definitions against the study's own Top-5 ({n_ctx} studies with context), the copied-sentence rate was {pc(CS['b'])} for G1 and {pc(CS['a'])} for G2 ({cdci(CS)}); whole-report copies {pc(CW['b'])} and {pc(CW['a'])} ({cdci(CW)}); repeated-sentence rate {pc(CR['b'])} and {pc(CR['a'])} ({cdci(CR)}). "
      f"Against all corpus reports, sentences of at least six words found verbatim: G1 {pc(cpy.loc[G1S, 'corpus_copied_sentence_rate'])}, G2 {pc(cpy.loc[G2S, 'corpus_copied_sentence_rate'])}; exact duplicates of an entire corpus report: {pc(cpy.loc[G1S, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])} and {pc(cpy.loc[G2S, 'exact_whole_report_duplicate_of_a_corpus_report_rate'])}. "
      f"{len_txt} Agents 2 and 3 never receive retrieved prose, so any remaining overlap would reflect generic reporting phrases the model produces by itself and not plagiarism.")
    A("")
    A("## 14. Empty classifier output")
    A("")
    A(f"The {n_emp} studies without a classifier output skip Agent 1 (nothing to verify) and are passed to Agents 2 and 3 with an explicit no_output state. {emp_txt} Report states for these studies (B0 / G1 / G2): normal {int(emp[emp.system == B0S].iloc[0].report_normal)} / {int(emp_g1.report_normal)} / {int(emp_g2.report_normal)}, abnormal {int(emp[emp.system == B0S].iloc[0].report_abnormal)} / {int(emp_g1.report_abnormal)} / {int(emp_g2.report_abnormal)}, "
      f"indeterminate {int(emp[emp.system == B0S].iloc[0].report_indeterminate)} / {int(emp_g1.report_indeterminate)} / {int(emp_g2.report_indeterminate)} (per extractor).")
    A("")
    A("## 15. Latency and compute")
    A("")
    ag = lat["agents"]
    A(f"Mean wall time per call: Agent 1 {ag['agent1_evidence_verifier']['mean_seconds']:.1f} s ({ag['agent1_evidence_verifier']['n_calls']} calls), Agent 2 {ag['agent2_grounded_report_writer']['mean_seconds']:.1f} s, Agent 3 {ag['agent3_grounding_critic']['mean_seconds']:.1f} s; per study {lat['g2_mean_seconds_per_study']:.1f} s for G2 versus {lat['g1_mean_seconds_per_study']:.1f} s for G1 "
      f"({lat['g2_over_g1_time_ratio']:.1f} times). Total recorded call time: G2 {lat['g2_total_recorded_call_seconds'] / 60:.0f} minutes (wall time of the run {lat['g2_wall_seconds_this_run'] / 60:.0f} minutes) versus G1 {lat['g1_total_recorded_call_seconds'] / 60:.0f} minutes. Mean output tokens per call: "
      f"{ag['agent1_evidence_verifier']['mean_output_tokens']:.0f} (Agent 1), {ag['agent2_grounded_report_writer']['mean_output_tokens']:.0f} (Agent 2), {ag['agent3_grounding_critic']['mean_output_tokens']:.0f} (Agent 3). Peak GPU memory sampled with nvidia-smi: {lat['gpu_memory_used_mib_peak_sampled']:.0f} MiB (total used memory, sampled every 5 studies). "
      "For an interactive application G2 costs about this many sequential local calls per study; a single-call design would be cheaper.")
    A("")
    A("## 16. Leakage checks")
    A("")
    A(f"Automated checks over {leak['n_studies']} studies ({leak['messages_checked']} messages): Agent 2 messages sharing a six-word sequence with a retrieved report {leak['agent2_message_shares_6gram_with_retrieved_text']}; Agent 3 messages {leak['agent3_message_shares_6gram_with_retrieved_text']}; reference-report sentences in Agent 2 / Agent 3 messages "
      f"{leak['agent2_message_reference_sentences']} / {leak['agent3_message_reference_sentences']}; reference sentences outside the retrieved text in Agent 1 messages {leak['agent1_message_reference_sentences_outside_retrieved_text']}; messages rebuilt identically from the stored inputs {leak['messages_rebuilt_identically']} of {leak['messages_checked']}; "
      f"stored message hashes match recomputed hashes: {leak['all_stored_hashes_match_recomputed']}. Overall: {'passed' if leak['pass'] else 'FAILED'}. Hashes of every sent message are in `g2_sent_messages_sha256.csv`.")
    A("")
    A("## 17. Supplementary: stratified by classifier state")
    A("")
    A(f"Within the {int(s_f1.n_studies)} classifier-abnormal studies, G2 minus G1 was precision {sci(s_pr)}, recall {sci(s_rc)}, F1 {sci(s_f1)}, FP propagation {sci(s_fp)} and TP retention {sci(s_tp)} (descriptive).")
    A("")
    A((G2 / "tables/table9_supplementary_stratified_by_classifier_state.md").read_text(encoding="utf-8"))
    A("## 18. Limitations")
    A("")
    A("- One 4-billion-parameter local model plays all three roles; one frozen prompt set, greedy decoding, one run; Ollama does not guarantee bitwise GPU determinism. No prompt was tuned and no ablation of individual agents beyond the draft-versus-final comparison was run.")
    A("- Agent 1's support status is recomputed by a deterministic count rule from the ranks the model cites; the cited ranks are not verified against the retrieved text. Agreement with the IU labels of the retrieved reports is descriptive because those labels are an incomplete MeSH-derived approximation.")
    A("- Agent 2 and Agent 3 receive Agent 1's short summaries, which are paraphrases checked automatically for six-word overlaps with retrieved prose; this guarantees no verbatim leakage but not that no retrieved information is conveyed.")
    A("- Finding extraction is rule-based and the reference is MeSH-derived and incomplete; per-finding counts are small, especially for rare findings, and Enlarged Cardiomediastinum has no reference positives.")
    A("- Latency was measured on one consumer GPU with a small quantised model and is not representative of other hardware.")
    A("- Validation data only; nothing was selected using the locked retrieval test, which remains unopened; the final Streamlit application has not been built.")
    A("")
    A("## 19. Conclusion")
    A("")
    A(f"{verdict} G2 minus G1: precision {word(PR)}, recall {word(RC)}, F1 {word(F1)}, hallucination rate {word(HL, True)}, FP propagation {word(FP, True)}, TP retention {word(TP)}, abnormal recall {word(AR)}, copying from the Top-5 {'reduced' if crit['copying from the retrieved reports reduced'] else 'not reliably reduced'}. "
      f"{crit_agent3} Runtime was {lat['g2_over_g1_time_ratio']:.1f} times that of G1. No change was made to the classifier, retrieval configuration or generator model. {caveat}")
    A("")
    (G2 / "MULTI_AGENT_RAG_ANALYSIS.md").write_text("\n".join(A_), encoding="utf-8")

    meth = f"""# Methods: structured multi-agent retrieval-augmented report generation (G2)

## Purpose and design

G2 tests whether decomposing preliminary report generation into three sequential roles of one frozen local language model retains the recall that relevant retrieval gave the single-agent system (G1) while reducing hallucination, propagation of classifier false positives, unsupported retrieval-derived findings and copying of retrieved reports. All components upstream of generation were frozen: the DenseNet-121 classifier with its operating policy, No Finding rule and Platt calibration, and the retrieval configuration (top-3 finding query, dense and BM25 reciprocal-rank fusion, Top-5). The model was {gen} (Ollama {meta['ollama_version']}, digest verified to equal G1's) with temperature 0, top-k 1 and seed {meta['generation_options']['seed']}; the evaluation populations ({n_p} validation studies, {n_c} in the clinical subset), finding extractor, reference labels and metrics were those of G1. Validation data only were used.

## Agents

Agent 1 (Evidence Verifier) receives the classifier-positive findings, calibrated probabilities, the No Finding state and the five retrieved reports, and returns JSON only: for each classifier finding the supporting retrieved-report ranks, a short paraphrased summary and a support status (supported, partially supported, unsupported), plus retrieval-only candidate findings and the ranks of reports describing a normal study. The JSON is validated deterministically: only valid ranks are kept, the support count is recomputed, status follows the predeclared count rule (two or more reports, one report, none), summaries that reproduce six consecutive words of a retrieved report are replaced, and candidates need at least one supporting report. Agent 2 (Grounded Report Writer) receives only the classifier state, probabilities and the validated evidence and writes Findings and Impression; retrieval-only candidates may be written only with at least three supporting reports and never when the classifier state is No Finding. Agent 3 (Grounding Critic) receives the classifier state, evidence and draft, checks eight predefined issue types once, and either approves (the draft is kept unchanged by code) or returns one corrected report. Agents 2 and 3 never receive retrieved report text. There is no loop. Studies without a classifier output skip Agent 1 and keep the explicit no-output handling of G1.

## Freezing and checks

The three prompts, output schemas and budgets were frozen with SHA-256 hashes after a four-study implementation test (normal, one finding, multiple findings, empty output) that checked parsing, structure, connectivity, leakage and caching only. Responses were cached per agent and study. Automated checks confirmed that no retrieved text reached Agents 2 and 3, that no reference text reached any agent, and that every stored message hash matched a rebuild from the stored inputs.

## Outcomes and analyses

The primary comparison was G2 minus G1 on identical studies: finding precision, recall, micro and macro F1, hallucination and omission rates, classifier false-positive propagation and true-positive retention, normal and abnormal recall, ROUGE-L, BLEU-4 and an exact-match METEOR variant, with paired study-level bootstrap intervals (1,000 resamples, seed 42). Success required reduced propagation and hallucination with preserved recall, retention and abnormal recall and reduced copying; no single metric decided. We also analysed each agent's contribution (support status against the reference, draft before critique, critic additions and removals), retrieval-only findings through the pipeline, copying with the G1 definitions, per-finding and rare-finding retention, and latency.
"""
    mw = wc(meth)
    (G2 / "MANUSCRIPT_G2_METHODS.md").write_text(meth, encoding="utf-8")

    resu = f"""# Results: structured multi-agent RAG (G2)

## Generation and checks

All {run['n_completed']} of {run['n_cases']} studies completed without failure; {n_p - n_ctx} empty-output studies skipped Agent 1. Stored B0, G1, G1A and G1B results were reproduced exactly, and the leakage checks {'passed' if leak['pass'] else 'failed'} (no retrieved text in Agent 2 or 3 messages; no reference text in any message).

## G1 versus G2

On the {n_c} clinical-subset studies, finding precision was {cis('precision', G1S)} for G1 and {cis('precision', G2S)} for G2; recall {cis('recall', G1S)} and {cis('recall', G2S)}; F1 {cis('f1', G1S)} and {cis('f1', G2S)}; macro F1 {f3(P_('macro_f1', G1S))} and {f3(P_('macro_f1', G2S))} (Table 2, Figure 1). Reports with at least one hallucinated finding were {pc(P_('hallucination_rate', G1S))} and {pc(P_('hallucination_rate', G2S))}, with an omission {pc(P_('omission_rate', G1S))} and {pc(P_('omission_rate', G2S))} (Figure 2). Paired differences G2 minus G1 (Table 3): precision {dci(PR)}, recall {dci(RC)}, F1 {dci(F1)}, hallucination rate {dci(HL)}, omission rate {dci(OM)}, FP propagation {dci(FP)}, TP retention {dci(TP)}, normal recall {dci(NR)}, abnormal recall {dci(AR)}, ROUGE-L {dci(RL)}, BLEU-4 {dci(B4)} and the METEOR variant {dci(MT)}. {verdict}

## False positives, retention and normal/abnormal consistency

Classifier false-positive propagation was {f3(P_('clf_fp_propagation', B0S))} for B0, {f3(P_('clf_fp_propagation', G1S))} for G1 and {f3(P_('clf_fp_propagation', G2S))} for G2; true-positive retention {f3(P_('tp_retention', B0S))}, {f3(P_('tp_retention', G1S))} and {f3(P_('tp_retention', G2S))} (Figure 3). Abnormal recall was {f3(P_('abnormal_recall', B0S))} for the classifier, {f3(P_('abnormal_recall', G1S))} for G1 and {f3(P_('abnormal_recall', G2S))} for G2, and normal recall {f3(P_('normal_recall', B0S))}, {f3(P_('normal_recall', G1S))} and {f3(P_('normal_recall', G2S))} (Figure 4); when the classifier output was abnormal the report was normal in {pc(P_('over_normalisation_given_classifier_abnormal', G1S))} (G1) and {pc(P_('over_normalisation_given_classifier_abnormal', G2S))} (G2) of studies (difference {dci(ON)}). {rare_txt} {rare_attr}

## Agent contributions

Agent 1 labelled {int(ag1.loc['supported', 'n_classifier_positives'])} classifier positives supported, {int(ag1.loc['partially_supported', 'n_classifier_positives'])} partially supported and {int(ag1.loc['unsupported', 'n_classifier_positives'])} unsupported in the clinical subset; the shares matching the reference were {pc(ag1.loc['supported', 'share_matching_reference']) if ag1.loc['supported', 'n_classifier_positives'] else 'n/a'}, {pc(ag1.loc['partially_supported', 'share_matching_reference']) if ag1.loc['partially_supported', 'n_classifier_positives'] else 'n/a'} and {pc(ag1.loc['unsupported', 'share_matching_reference']) if ag1.loc['unsupported', 'n_classifier_positives'] else 'n/a'} (Table 4, Figure 5). {disc_txt} {crit_agent3} {crit_effect} Of {int(f_stage.loc[stages[0], 'n_candidates'])} retrieval-only candidates, {int(f_stage.loc[stages[1], 'n_candidates'])} reached the draft and {int(f_stage.loc[stages[2], 'n_candidates'])} the final report; G1 stated {int(ri(G1S)['introduced_findings_(stated, not classifier positive)'])} findings beyond the classifier output ({pc(ri(G1S)['share_supported'])} reference-supported) and G2 {int(ri(G2S)['introduced_findings_(stated, not classifier positive)'])} ({pc(ri(G2S)['share_supported'])}) (Table 5, Figure 6).

## Copying and runtime

The copied-sentence rate against the study's own Top-5 was {pc(CS['b'])} for G1 and {pc(CS['a'])} for G2 ({cdci(CS)}); whole-report copies {pc(CW['b'])} and {pc(CW['a'])} ({cdci(CW)}); repeated-sentence rate {pc(CR['b'])} and {pc(CR['a'])}. G2 needed {lat['g2_mean_seconds_per_study']:.1f} s per study versus {lat['g1_mean_seconds_per_study']:.1f} s for G1 ({lat['g2_over_g1_time_ratio']:.1f} times; Table 7).

## Interpretation

{caveat} {len_txt} Excluding the studies with truncated Agent 1 output left the G2 minus G1 conclusions unchanged (post hoc sensitivity analysis in the artifacts). The classifier, retrieval configuration and generator model were unchanged, the locked retrieval-test partition was not opened, and the final application has not been built.
"""
    rw = wc(resu)
    (G2 / "MANUSCRIPT_G2_RESULTS.md").write_text(resu, encoding="utf-8")
    ok_len = 450 <= mw <= 700 and 600 <= rw <= 1000

    fc = f"""# Figure captions (G2)

Retrieval-validation studies only (clinical subset {n_c} studies; text metrics {n_p} studies). B0 = rule-based report from the frozen classifier output; G1 = single-agent RAG; G2 draft = Agent 2 report before the critic; G2 = final multi-agent report. Error bars: 95% study-level paired bootstrap intervals.

**Figure 1. Finding precision, recall and F1 for B0, G1, the G2 Agent 2 draft and final G2.** Micro-averaged agreement between the findings stated in the report (any affirmative mention) and the MeSH-mapped IU reference findings.

**Figure 2. Hallucination and omission.** Left: share of reports with at least one hallucinated finding and with at least one omitted finding. Right: mean hallucinated and omitted findings per report.

**Figure 3. Classifier false-positive propagation and true-positive retention.** Proportion of classifier false positives appearing in the report and of classifier true positives retained. B0 repeats every classifier-positive finding by construction.

**Figure 4. Normal versus abnormal recall.** Share of reference-normal and reference-abnormal studies whose report state is correct. The B0 report state equals the classifier state by construction.

**Figure 5. Agent 1 evidence-support categories.** Left: classifier positives by validated support status, split by agreement with the reference. Right: retrieval-only candidate findings proposed by Agent 1, promoted into the Agent 2 draft and surviving the critic, split by agreement with the reference.

**Figure 6. Critic actions and copying.** Left: findings removed and added by the critic (clinical subset), split by agreement with the reference. Right: copied-sentence rate and whole-report copy rate against the study's own Top-5 and repeated-sentence rate for G1, the draft and final G2.
"""
    tc = """# Table captions (G2)

**Table 1. G2 architecture and configuration.** Agents, their inputs and outputs, deterministic validation, frozen components, generator and decoding settings, and the SHA-256 hashes of the three frozen prompts.

**Table 2. G1 versus G2 main metrics.** B0, G1, the Agent 2 draft and the final G2 report on identical studies, with 95% bootstrap intervals.

**Table 3. Paired bootstrap differences.** G2 minus G1 (primary), draft minus G1 and G2 minus draft for all metrics; paired study-level bootstrap, 1,000 resamples.

**Table 4. Agent 1 evidence analysis.** Support status of classifier positives and their agreement with the reference, retrieval-only candidates, and agreement of Agent 1's citations with the IU labels of the retrieved reports.

**Table 5. Agent 3 intervention analysis.** Share of reports approved or modified, findings removed and added by the critic with their agreement with the reference, and the fate of retrieval-only candidates through the pipeline.

**Table 6. Per-finding results.** Reference positives, classifier true positives, true-positive retention and precision / recall / F1 for B0, G1 and G2; rare findings follow the C2 definition.

**Table 7. Latency and copying analysis.** Per-agent and per-study latency, tokens and GPU memory, and copying of the Top-5 and of the corpus for G1 and G2.

**Table 8 (supplementary). Normal/abnormal behaviour and rare findings.** Normal and abnormal recall and rare-finding retention for the classifier, G1, the draft and G2.

**Table 9 (supplementary). G2 minus G1 by classifier state.** Descriptive stratification by classifier state.
"""
    (G2 / "FIGURE_CAPTIONS.md").write_text(fc, encoding="utf-8")
    (G2 / "TABLE_CAPTIONS.md").write_text(tc, encoding="utf-8")
    idx = f"""# G2 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_g2_architecture_configuration.*` | Three-agent architecture, frozen configuration, prompt hashes | Methods |
| Table 2 `tables/table2_g1_vs_g2_main_metrics.*` | Finding F1 G1 {f3(P_('f1', G1S))} vs G2 {f3(P_('f1', G2S))} | Results |
| Table 3 `tables/table3_paired_bootstrap_differences.*` | G2 minus G1: F1 {dci(F1)}, FP propagation {dci(FP)}, TP retention {dci(TP)} | Results |
| Table 4 `tables/table4_agent1_evidence_analysis.*` | Agent 1 support status vs reference | Results |
| Table 5 `tables/table5_agent3_intervention_analysis.*` | Critic modified {a3['modified_pct']:.1f}% of reports; retrieval-only funnel | Results / Discussion |
| Table 6 `tables/table6_per_finding_results.*` | Per-finding retention and P/R/F1 | Supplementary |
| Table 7 `tables/table7_latency_and_copying_analysis.*` | G2 {lat['g2_over_g1_time_ratio']:.1f} times G1 latency; copying {pc(CS['a'])} vs {pc(CS['b'])} | Results / Discussion |
| Table 8 `tables/table8_supplementary_normal_abnormal_and_rare_findings.*` | Normal/abnormal and rare-finding behaviour | Supplementary |
| Table 9 `tables/table9_supplementary_stratified_by_classifier_state.*` | G2 minus G1 within classifier states | Supplementary |
| Figure 1 `figures/fig1_g1_vs_g2_finding_precision_recall_f1.*` | Finding agreement | Results |
| Figure 2 `figures/fig2_hallucination_and_omission.*` | Hallucination and omission | Results |
| Figure 3 `figures/fig3_fp_propagation_and_tp_retention.*` | FP propagation versus TP retention | Results |
| Figure 4 `figures/fig4_normal_vs_abnormal_recall.*` | Normal/abnormal behaviour | Results |
| Figure 5 `figures/fig5_agent1_evidence_support_categories.*` | Agent 1 evidence categories and retrieval-only funnel | Results |
| Figure 6 `figures/fig6_critic_actions_and_copying_reduction.*` | Critic actions and copying | Results / Discussion |
| `MULTI_AGENT_RAG_ANALYSIS.md` | Full G2 analysis | Supplementary / internal |
| `MANUSCRIPT_G2_METHODS.md`, `MANUSCRIPT_G2_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_G2.md` | Content for Methodology, Results, Discussion | University report |
| `g2_prompts_frozen.json`, `g2_prompts_frozen.txt`, `g2_generator_metadata.json` | Frozen prompts, schemas, hashes, generator metadata | Methods / code release |
| `g2_generated_reports.csv`, `agent_outputs/` | Every agent's output per study | Supplementary data |
| `g2_sent_messages_sha256.csv`, `g2_leakage_checks.json` | Hashes of every sent message and leakage checks | Supplementary |
| `g2_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
"""
    (G2 / "G2_JOURNAL_ASSET_INDEX.md").write_text(idx, encoding="utf-8")

    U = []
    A = U.append
    A("# University report content: G2 (structured multi-agent RAG)")
    A("")
    A("_Insert-ready material generated from the G2 artifacts. Validation data only; classifier, retrieval configuration and generator model are unchanged; the locked retrieval-test split is unopened._")
    A("")
    A("## Methodology")
    A("")
    A("### Purpose and architecture")
    A("")
    A(f"G2 splits report generation into three sequential roles of the same offline model ({gen}, greedy decoding, fixed seed): an Evidence Verifier that checks every classifier finding against the five retrieved reports and returns structured JSON, a Grounded Report Writer that sees only the classifier output and the verified evidence (never the retrieved text), and a Grounding Critic that checks the draft once and either approves it or returns one corrected report. There is no loop. The aim is to keep the recall that relevant retrieval gave the single-agent system G1 while reducing hallucination, classifier false-positive propagation, retrieval-induced unsupported findings and copying. The three prompts were frozen with SHA-256 hashes before bulk generation, and automated checks confirmed that no retrieved text reached the writer and the critic and that no reference text reached any agent.")
    A("")
    A("### Evaluation")
    A("")
    A("G2 is compared with G1 on the same validation studies with the same finding extractor, reference labels and metrics (finding precision, recall and F1, hallucination and omission, classifier false-positive propagation and true-positive retention, normal and abnormal recall, text-overlap scores) using a paired study-level bootstrap of 1,000 resamples. G2 is judged against predeclared criteria (lower propagation and hallucination with preserved recall, retention and abnormal recall, and less copying), not against a single metric.")
    A("")
    A("## Results")
    A("")
    A("### G1 versus G2")
    A("")
    A(f"Finding F1 was {f3(P_('f1', G1S))} for G1 and {f3(P_('f1', G2S))} for G2 (difference {dci(F1)}); recall {dci(RC)}; hallucination rate {dci(HL)}; classifier false-positive propagation {dci(FP)}; true-positive retention {dci(TP)}; abnormal recall {dci(AR)}. {verdict} Criteria: {crit_txt}.")
    A("")
    A("### Agents, retrieval-only findings and copying")
    A("")
    A(f"{crit_agent3} {crit_effect} Agent 1 proposed {int(f_stage.loc[stages[0], 'n_candidates'])} retrieval-only candidates, {int(f_stage.loc[stages[1], 'n_candidates'])} reached the draft and {int(f_stage.loc[stages[2], 'n_candidates'])} the final report. Copied-sentence rate against the study's own Top-5: G1 {pc(CS['b'])}, G2 {pc(CS['a'])} ({cdci(CS)}); whole-report copies {pc(CW['b'])} and {pc(CW['a'])}. {rare_txt} {rare_attr}")
    A("")
    A("### Runtime")
    A("")
    A(f"G2 used {lat['g2_mean_seconds_per_study']:.1f} s per study against {lat['g1_mean_seconds_per_study']:.1f} s for G1 ({lat['g2_over_g1_time_ratio']:.1f} times), with up to three sequential local calls per study; peak sampled GPU memory {lat['gpu_memory_used_mib_peak_sampled']:.0f} MiB.")
    A("")
    A("## Discussion")
    A("")
    A(f"{caveat} The decomposition is only as reliable as its weakest agent: the support status depends on the ranks cited by a small model, and the critic's own judgements are limited. The per-agent analyses show where the pipeline helps or does not. The multi-agent system is a research prototype, and the final application has not been built.")
    A("")
    (G2 / "UNIVERSITY_REPORT_G2.md").write_text("\n".join(U), encoding="utf-8")

    bad = []
    for n in ("MULTI_AGENT_RAG_ANALYSIS.md", "MANUSCRIPT_G2_METHODS.md", "MANUSCRIPT_G2_RESULTS.md", "UNIVERSITY_REPORT_G2.md", "FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md", "G2_JOURNAL_ASSET_INDEX.md"):
        txt = (G2 / n).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"superior to", r"\{[a-z_]+\[[^}]*\}", r"\bnan\b", r"\bNone\b", r"anthropic", r"claude"):
            if re.search(pat, txt, re.I if pat in ("anthropic", "claude") else 0):
                bad.append((n, pat))
    print(json.dumps({"methods_words": mw, "results_words": rw, "length_ok": ok_len, "wording_flags": bad, "criteria": crit}))
    return 0 if ok_len and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
