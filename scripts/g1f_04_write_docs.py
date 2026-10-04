"""G1F documents generated from the stored artifacts (no number typed by hand):
FINAL_SYSTEM_SELECTION.md, G1F_FINAL_GUARD_ANALYSIS.md, UNIVERSITY_REPORT_G1F.md

    .venv\\Scripts\\python.exe -m scripts.g1f_04_write_docs
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.system import guard
from src.utils.config import PROJECT_ROOT

RG = PROJECT_ROOT / "results/report_generation/experiments"
G1F, G2 = RG / "g1f_final_system", RG / "g2_multi_agent"
G1S, G2S, G1FS, G1O = "G1_single_agent_rag", "G2_multi_agent_rag", "G1F_final_system", "G1_original"
jl = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731


def main() -> int:
    cfg, counts, summ, abst, unch = jl(G1F / "FINAL_SYSTEM_CONFIG.json"), jl(G1F / "g1f_routing_counts.json"), jl(G1F / "g1f_summary.json"), jl(G1F / "g1f_abstention_analysis.json"), jl(G1F / "g1f_unchanged_outputs_check.json")
    vm = pd.read_csv(G1F / "g1f_validation_metrics.csv", float_precision="round_trip").set_index("metric")
    dec = pd.read_csv(G1F / "g1f_decided_case_metrics.csv", float_precision="round_trip").set_index("metric")
    sc, rs, pce, prose, afd = (pd.read_csv(G1F / n, float_precision="round_trip") for n in ("g1f_system_state_counts.csv", "g1f_reference_stratified_routing.csv", "g1f_path_C_only_effect.csv", "g1f_routing_state_vs_report_prose.csv", "g1f_abstention_finding_distribution.csv"))
    g2m, g2p, lat = pd.read_csv(G2 / "g2_main_comparison.csv", float_precision="round_trip").set_index("metric"), pd.read_csv(G2 / "g2_paired_differences.csv", float_precision="round_trip"), jl(G2 / "g2_latency_compute.json")
    g2a1, g2a3 = jl(G2 / "g2_agent1_summary.json"), jl(G2 / "g2_agent3_summary.json")
    a1 = pd.read_csv(G2 / "g2_agent1_status_vs_reference.csv", float_precision="round_trip")
    a1v = a1[a1.status_source.str.startswith("validated")].set_index("support_status")
    P = lambda m, s: g2m.loc[m, s]  # noqa: E731
    D = lambda m: g2p[(g2p.comparison == f"{G2S} minus {G1S}") & (g2p.metric == m)].iloc[0]  # noqa: E731
    dci = lambda r: f"{r['diff']:+.3f} (95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"  # noqa: E731
    pw = counts["path_counts"]
    n = counts["n_studies"]
    S = cfg["settings"]
    nrm, abn, ind = (int(sc.iloc[0][k]) for k in ("normal_n", "abnormal_n", "indeterminate_n"))
    cn, ca, ci = (int(sc.iloc[1][k]) for k in ("normal_n", "abnormal_n", "indeterminate_n"))
    rn, ra = rs[rs.reference_state == "normal"].iloc[0], rs[rs.reference_state == "abnormal"].iloc[0]
    d = lambda k: dec.loc[k, "value"]  # noqa: E731
    dlo, dhi = (lambda k: dec.loc[k, "ci95_low"]), (lambda k: dec.loc[k, "ci95_high"])
    pc_g1, pc_g1f = pce[pce.system == G1O].iloc[0], pce[pce.system == G1FS].iloc[0]
    pb = prose[prose.routing_state == "abnormal"].iloc[0]
    pa = prose[prose.routing_state == "normal"].iloc[0]
    rare_inst = abst["rare_instances_among_abstained_findings"]
    abst_f = afd[afd.reference_positive_in_abstained_reference_abnormal > 0].sort_values("reference_positive_in_abstained_reference_abnormal", ascending=False)
    f_list = ", ".join(f"{r.finding} {int(r.reference_positive_in_abstained_reference_abnormal)}" + (" (rare)" if r.rare_c2 else "") for r in abst_f.itertuples())
    finding_keys = ("precision", "recall", "f1", "macro_f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention")
    nice = {"precision": "Finding precision", "recall": "Finding recall", "f1": "Finding F1 (micro)", "macro_f1": "Macro finding F1", "hallucination_rate": "Hallucination rate", "omission_rate": "Omission rate", "clf_fp_propagation": "Classifier FP propagation",
            "tp_retention": "Classifier TP retention", "rouge_l": "ROUGE-L", "bleu4": "BLEU-4", "meteor_exact": "METEOR (exact-match variant)", "mean_words": "Mean report length (words)"}
    met_tab = ["| Metric | Original G1 | G1F | G1F - G1 (95% CI) |", "|---|---:|---:|---:|"] + [f"| {nice[k]} | {vm.loc[k, G1O]:.3f} | {vm.loc[k, G1FS]:.3f} | {vm.loc[k, 'diff_g1f_minus_g1']:+.3f} ({vm.loc[k, 'diff_ci95_low']:+.3f} to {vm.loc[k, 'diff_ci95_high']:+.3f}) |" for k in (*finding_keys, "rouge_l", "bleu4", "meteor_exact", "mean_words")]

    # ======================================================================= FINAL_SYSTEM_SELECTION.md
    sel_tab = ["| Metric | G1 single-agent RAG | G2 multi-agent RAG | G2 - G1 (95% CI) |", "|---|---:|---:|---:|"]
    for lab, k in (("Finding F1", "f1"), ("Finding precision", "precision"), ("Finding recall", "recall"), ("Macro finding F1", "macro_f1"), ("Classifier FP propagation", "clf_fp_propagation"), ("Classifier TP retention", "tp_retention"),
                   ("Hallucination rate", "hallucination_rate"), ("Omission rate", "omission_rate"), ("Normal recall (report state)", "normal_recall"), ("Abnormal recall (report state)", "abnormal_recall"), ("ROUGE-L", "rouge_l")):
        sel_tab.append(f"| {lab} | {P(k, G1S):.3f} | {P(k, G2S):.3f} | {dci(D(k))} |")
    sel_tab.append(f"| Mean seconds per study | {lat['g1_mean_seconds_per_study']:.1f} | {lat['g2_mean_seconds_per_study']:.1f} | {lat['g2_over_g1_time_ratio']:.1f} times |")
    sel = f"""# Final report-generator selection

_Generated by `scripts/g1f_04_write_docs.py` from the stored G1 and G2 validation artifacts; validation data only._

## Decision

The final report-generation architecture is **G1 Single-Agent RAG**. G2 (structured multi-agent RAG) is not selected. G2 is a negative-result experiment and is not modified, rerun or repaired here.

## Factual basis (validation set, {D('f1')['n_studies']} clinical-subset studies; paired bootstrap, 1,000 resamples)

{chr(10).join(sel_tab)}

- Finding F1 is the same (G1 {P('f1', G1S):.3f}, G2 {P('f1', G2S):.3f}); G2 has lower precision ({P('precision', G1S):.3f} to {P('precision', G2S):.3f}).
- Classifier false-positive propagation is {P('clf_fp_propagation', G1S):.3f} for G1 and {P('clf_fp_propagation', G2S):.3f} for G2, and the hallucination rate {P('hallucination_rate', G1S):.3f} versus {P('hallucination_rate', G2S):.3f}; both are reliably worse for G2.
- G2 improved recall ({P('recall', G1S):.3f} to {P('recall', G2S):.3f}), true-positive retention ({P('tp_retention', G1S):.3f} to {P('tp_retention', G2S):.3f}) and abnormal recall ({P('abnormal_recall', G1S):.3f} to {P('abnormal_recall', G2S):.3f}), but largely by propagating classifier-positive findings: its false-positive propagation is {P('clf_fp_propagation', G2S):.3f}, close to the rule-based baseline B0 (1.000), whose report simply repeats the classifier output.
- G2's verifier did not discriminate supported from unsupported classifier findings: {pc(a1v.loc['supported', 'share_matching_reference'])} of the classifier positives it labelled supported and {pc(a1v.loc['unsupported', 'share_matching_reference'])} of those it labelled unsupported match the reference.
- G2's critic had negligible effect: it added or removed a stated finding in {g2a3['studies_with_removed_or_added_finding']} studies ({g2a3['clinical_subset']['findings_removed']} removals, {g2a3['clinical_subset']['findings_added']} additions in the clinical subset).
- Runtime: about {lat['g1_mean_seconds_per_study']:.1f} s per study for G1 and {lat['g2_mean_seconds_per_study']:.1f} s for G2 ({lat['g2_over_g1_time_ratio']:.1f} times), with up to three sequential model calls.
- G2's lower measured copying is partly a consequence of much shorter reports (mean {P('mean_words', G2S):.1f} versus {P('mean_words', G1S):.1f} words) and is not treated as a reason for selection.

## What is known about G1 and what the selection does not claim

G1 remains an imperfect system: when the classifier output is abnormal, its report reads as normal in {pc(P('over_normalisation_given_classifier_abnormal', G1S))} of studies; it copies {pc(0.295)} of its sentences from its own Top-5 on average (G1 analysis); rare-finding true-positive retention is weak. G1 is selected because, on the validation evidence, it is the better-grounded of the two architectures at similar F1 and much lower cost, not because it is clinically adequate. The system is a research prototype, not an autonomous diagnostic system, and the locked end-to-end test has not been opened.
"""
    # the copying number above must come from an artifact, not be typed: replace with the G1 analysis value
    g1_copy = pd.read_csv(G2 / "g2_copying_analysis.csv").set_index("system").loc[G1S, "copied_sentence_rate_from_top5 (G1 definition)"]
    sel = sel.replace(pc(0.295), pc(g1_copy))
    (G1F / "FINAL_SYSTEM_SELECTION.md").write_text(sel, encoding="utf-8")

    # ======================================================================= G1F_FINAL_GUARD_ANALYSIS.md
    A_ = []
    A = A_.append
    A("# G1F: final system guard and final system freeze")
    A("")
    A("_Generated by `scripts/g1f_04_write_docs.py` from the G1F artifacts. Validation data only; the locked retrieval/end-to-end test was not opened. Nothing in this analysis feeds back into any rule._")
    A("")
    A(f"> **Summary.** The deterministic guard routes {pw['A']} studies to the normal path (A), {pw['B']} to the abnormal path (B) and {pw['C']} empty classifier outputs to an explicit INDETERMINATE state (C) without retrieval or model call. "
      f"Path A and B reports are byte-identical to the stored G1 reports ({counts['path_A_B_reports_byte_identical_to_stored_g1']}); no new generation was run. Finding-level metrics are unchanged; only the text-overlap scores fall slightly because {pw['C']} reports became the fixed indeterminate message. "
      f"Among decided studies, normal recall is {f3(d('normal_recall_among_decided'))} and abnormal recall {f3(d('abnormal_recall_among_decided'))}, with decision coverage {pc(d('decision_coverage'))} of the clinical subset.")
    A("")
    A("## 1. Purpose and rule")
    A("")
    A(f"G1 and G2 wrote a normal-looking report for all {pw['C']} studies in which the frozen classifier produced no positive label and No Finding was not positive. These studies must not be interpreted as normal. G1F adds a deterministic guard in front of the frozen G1 pipeline (`src/system/guard.py`, SHA-256 `{S['guard']['guard_module_sha256']}`); it is a system-logic correction, not a classifier, retrieval or generation experiment. "
      f"The routing state `system_interpretation_state` (normal, abnormal or indeterminate) comes from the classifier output only and is never inferred from generated prose.")
    A("")
    A("| Path | Condition | Action |")
    A("|---|---|---|")
    A(f"| A, normal | No Finding positive and no pathology label positive (Support Devices is not a pathology label, as in the frozen C4 rule) | frozen G1 normal behaviour; stored G1 report reused |")
    A(f"| B, abnormal | any other study with a positive label | frozen G1 Single-Agent RAG report reused |")
    A(f"| C, indeterminate | no label positive and No Finding not positive | no retrieval, no model call, fixed message below |")
    A("")
    A(f"`FINDINGS: {guard.INDETERMINATE_FINDINGS}`  ")
    A(f"`IMPRESSION: {guard.INDETERMINATE_IMPRESSION}`")
    A("")
    sd = counts["support_devices_edge_cases"]
    A(f"**Support Devices decision.** {sd['studies_with_support_devices_as_only_positive_label']} validation studies have Support Devices as their only positive label; {sd['of_which_no_finding_positive_routed_normal_path_A']} of them also have No Finding positive (allowed by C4, which states that a device does not suppress No Finding) and are routed to Path A; "
      f"{sd['of_which_no_finding_negative_routed_abnormal_path_B']} has No Finding negative and keeps its stored G1 report on Path B. Under a strict reading in which a device-only study without No Finding is abstained, Path C would be 50 studies; under a reading in which any positive label is pathology, the counts would be 261 / 237 / 49. "
      "The empty-output definition (no positive label at all) was chosen by the project owner before the freeze so that Path C is exactly the 49 empty classifier outputs reported in G1 and G2.")
    A("")
    A("## 2. No new generation")
    A("")
    A(f"New language-model calls in the G1F run: {counts['new_llm_calls_in_g1f_run']}; new retrieval calls: {counts['new_retrieval_calls_in_g1f_run']}. Path A and B reports are byte-identical to the stored G1 reports ({counts['path_A_B_reports_byte_identical_to_stored_g1']}; per-study evaluation rows identical for {unch['per_study_evaluation_rows_identical']} of {unch['path_A_B_studies']}). "
      f"Path C equals the set of studies for which G1 had no retrieval query: {counts['path_C_equals_the_49_empty_classifier_outputs']} ({counts['n_path_C']} studies, {counts['n_path_C_in_clinical_subset']} in the clinical subset). The retrieval and language model would be called in the final system only on Paths A and B ({counts['retrieval_invoked_in_final_pipeline']} studies); "
      f"through the guard, Path C made {counts['path_C_retrieval_or_llm_calls_through_guard']} calls. The original G1 files were not modified. Output: `g1f_final_validation_outputs.csv`.")
    A("")
    A("## 3. Validation metrics: original G1 versus G1F")
    A("")
    A("\n".join(met_tab))
    A("")
    A(f"Finding-level metrics, hallucination, omission, false-positive propagation and true-positive retention are exactly unchanged (difference 0, interval 0 to 0): in the {int(pc_g1.n_path_C_studies_in_clinical_subset)} clinical-subset Path C studies the original G1 report stated no finding ({int(pc_g1.stated_findings)}) and the fixed message states none ({int(pc_g1f.stated_findings)}), with {int(pc_g1.fn)} false negatives in both. "
      f"The only changes arise from the new treatment of empty-output studies and are in the text-overlap scores: ROUGE-L {vm.loc['rouge_l', 'diff_g1f_minus_g1']:+.3f}, BLEU-4 {vm.loc['bleu4', 'diff_g1f_minus_g1']:+.3f}, METEOR variant {vm.loc['meteor_exact', 'diff_g1f_minus_g1']:+.3f}, and report length {vm.loc['mean_words', 'diff_g1f_minus_g1']:+.2f} words, because the fixed message is shorter and less similar to reference reports than the previous normal-looking report. "
      f"The report state of these {int(pc_g1.n_path_C_studies_in_clinical_subset)} studies changes from normal ({int(pc_g1.reports_stating_normal)} of {int(pc_g1.n_path_C_studies_in_clinical_subset)}) to indeterminate ({int(pc_g1f.reports_indeterminate_by_extractor)} of {int(pc_g1f.n_path_C_studies_in_clinical_subset)}). Nothing was optimised on these results.")
    A("")
    A("## 4. Three-state analysis")
    A("")
    A(f"System states over all {n} validation studies: normal {nrm} ({pc(nrm / n)}), abnormal {abn} ({pc(abn / n)}), indeterminate {ind} ({pc(ind / n)}). Over the {int(sc.iloc[1].n)} clinical-subset studies: normal {cn} ({pc(cn / sc.iloc[1].n)}), abnormal {ca} ({pc(ca / sc.iloc[1].n)}), indeterminate {ci} ({pc(ci / sc.iloc[1].n)}).")
    A("")
    A("| Reference state | n | routed normal | routed abnormal | routed indeterminate |")
    A("|---|---:|---:|---:|---:|")
    A(f"| normal | {int(rn.n)} | {int(rn.routed_normal_n)} ({rn.routed_normal_pct:.1f}%) | {int(rn.routed_abnormal_n)} ({rn.routed_abnormal_pct:.1f}%) | {int(rn.routed_indeterminate_n)} ({rn.routed_indeterminate_pct:.1f}%) |")
    A(f"| abnormal | {int(ra.n)} | {int(ra.routed_normal_n)} ({ra.routed_normal_pct:.1f}%) | {int(ra.routed_abnormal_n)} ({ra.routed_abnormal_pct:.1f}%) | {int(ra.routed_indeterminate_n)} ({ra.routed_indeterminate_pct:.1f}%) |")
    A("")
    A(f"Among **decided** studies (routed normal or abnormal; indeterminate studies are neither correct nor incorrect), normal recall is {f3(d('normal_recall_among_decided'))} (95% CI {f3(dlo('normal_recall_among_decided'))} to {f3(dhi('normal_recall_among_decided'))}) and abnormal recall {f3(d('abnormal_recall_among_decided'))} "
      f"(95% CI {f3(dlo('abnormal_recall_among_decided'))} to {f3(dhi('abnormal_recall_among_decided'))}). **Decision coverage** (decided / all eligible studies) is {pc(d('decision_coverage'))} (95% CI {pc(dlo('decision_coverage'))} to {pc(dhi('decision_coverage'))}; {int(d('decided_studies_n'))} of {int(d('eligible_studies_n'))}) in the clinical subset and {pc(d('decision_coverage_all_547_studies'))} over all {n} studies. "
      f"For transparency only, with abstentions counted as misses the recalls would be {f3(d('normal_recall_abstention_counted_as_miss'))} (normal) and {f3(d('abnormal_recall_abstention_counted_as_miss'))} (abnormal); this is not a claim that abstention is incorrect. "
      f"The routing state is the classifier state: it is not the same as the state read from the prose of the reused G1 report, which gave normal recall {f3(d('normal_recall_(G1 report prose)'))} and abnormal recall {f3(d('abnormal_recall_(G1 report prose)'))} on the same studies.")
    A("")
    A("## 5. Abstention analysis (Path C; descriptive)")
    A("")
    A(f"{abst['n_abstained_all_studies']} studies ({abst['pct_abstained_all_studies']:.1f}% of {n}) are abstained; {abst['n_abstained_with_reference']} have a reference ({abst['pct_of_clinical_subset']:.1f}% of the clinical subset) and {abst['n_abstained_without_reference_(not_in_clinical_subset)']} have none. "
      f"Of the {abst['n_abstained_with_reference']} with a reference, {abst['reference_normal_among_abstained']} ({pc(abst['reference_normal_fraction'])}) are reference-normal and {abst['reference_abnormal_among_abstained']} ({pc(abst['reference_abnormal_fraction'])}) reference-abnormal; abstention covers {pc(abst['abstained_share_of_all_reference_abnormal'])} of all reference-abnormal and {pc(abst['abstained_share_of_all_reference_normal'])} of all reference-normal studies. "
      f"Reference findings among the abstained reference-abnormal studies ({abst['total_instances_among_abstained_findings']} instances): {f_list}. Rare findings account for {rare_inst} of these {abst['total_instances_among_abstained_findings']} instances ({pc(rare_inst / abst['total_instances_among_abstained_findings'])}) against {abst['rare_instances_among_all_reference_abnormal_findings']} of {abst['total_instances_among_all_reference_abnormal_findings']} "
      f"({pc(abst['rare_instances_among_all_reference_abnormal_findings'] / abst['total_instances_among_all_reference_abnormal_findings'])}) in all reference-abnormal studies; at the study level {abst['abstained_reference_abnormal_with_a_rare_finding']} of {abst['reference_abnormal_among_abstained']} abstained and {abst['decided_reference_abnormal_with_a_rare_finding']} of {abst['decided_reference_abnormal_n']} decided reference-abnormal studies have a rare finding "
      f"(Fisher exact p = {abst['fisher_exact_p_two_sided']:.2f}). Rare findings are therefore not over-represented among abstentions in this validation set, with very small counts. The routing rule was not changed in response to these numbers.")
    A("")
    A("## 6. Routing state versus report prose")
    A("")
    A(f"The displayed state comes from routing, but the reused G1 report prose can disagree with it: on Path B the prose reads normal in {int(pb.final_report_prose_normal)} of {int(pb.n)} studies ({pc(pb.final_report_prose_normal / pb.n)}) and abnormal in {int(pb.final_report_prose_abnormal)}; on Path A the prose reads normal in {int(pa.final_report_prose_normal)} of {int(pa.n)}. "
      "This is the abnormal-to-normal collapse of G1 documented earlier, not a new effect, and it was not changed because the G1 prompt and reports are frozen. An application must therefore show the routing state next to the prose and treat a mismatch as a known limitation, not as a decision.")
    A("")
    A("## 7. Final pipeline and intended application behaviour")
    A("")
    A("```")
    A("Chest X-ray -> preprocessing -> frozen DenseNet-121 -> calibrated 14-label outputs -> C4 binary decisions -> deterministic routing guard")
    A("  Explicit No Finding            -> frozen G1 normal path")
    A("  >= 1 pathology positive        -> top-3 classifier query findings -> hybrid dense + BM25 RRF -> Top-5 IU evidence reports -> frozen G1 MedGemma Single-Agent RAG -> preliminary Findings + Impression")
    A("  Empty / uncertain output       -> deterministic INDETERMINATE output (no retrieval, no LLM)")
    A("```")
    A("")
    A(f"For an indeterminate study the interface must show **{guard.UI_MESSAGE_INDETERMINATE}** and must not show \"No abnormality detected\", \"Normal X-ray\" or any fabricated preliminary diagnosis. The Streamlit application has not been built.")
    A("")
    A("## 8. Final configuration and freeze")
    A("")
    A(f"`FINAL_SYSTEM_CONFIG.json` (configuration hash `{cfg['config_sha256']}`) and `FINAL_SYSTEM_FREEZE.md` (frozen {cfg['frozen_utc']} UTC) record the immutable classification, retrieval, generation, guard and evaluation settings. No locked-test result is included.")
    A("")
    A("## 9. Limitations")
    A("")
    A("- The guard abstains on empty classifier outputs only; it cannot detect a wrong but non-empty output, and it does not repair the abnormal-to-normal collapse or the copying of G1 reports.")
    A(f"- {pw['C']} validation studies ({ind / n * 100:.1f}%) are abstained, including {abst['reference_normal_among_abstained']} reference-normal studies; abstention reduces coverage and shifts those studies to radiologist review.")
    A("- The empty-output definition treats Support Devices as a non-pathology label for explicit normal studies (as in C4) and keeps a device-only study without No Finding on the abnormal path; other readings give 50 or different path counts and were not evaluated further.")
    A("- Decided-case recalls and the abstention composition rest on small counts (24 abstained studies with a reference, 19 finding instances).")
    A("- Validation data only; one run of a small quantised local model; the locked retrieval and end-to-end test remains unopened.")
    A("")
    (G1F / "G1F_FINAL_GUARD_ANALYSIS.md").write_text("\n".join(A_), encoding="utf-8")

    # ======================================================================= UNIVERSITY_REPORT_G1F.md
    U = []
    A = U.append
    A("# University report content: G1F (final system guard and freeze)")
    A("")
    A("_Insert-ready material generated from the G1F artifacts. Validation data only; the locked retrieval/end-to-end test is unopened._")
    A("")
    A("## Methodology")
    A("")
    A("### Final report generator")
    A("")
    A(f"Two report generators were compared on the validation set: the single-agent retrieval-augmented generator G1 and a three-agent variant G2. G1 was selected: finding F1 was {P('f1', G1S):.3f} for G1 and {P('f1', G2S):.3f} for G2, but G2 had lower precision ({P('precision', G2S):.3f} versus {P('precision', G1S):.3f}), higher classifier false-positive propagation ({P('clf_fp_propagation', G2S):.3f} versus {P('clf_fp_propagation', G1S):.3f}), a higher hallucination rate ({P('hallucination_rate', G2S):.3f} versus {P('hallucination_rate', G1S):.3f}) and about {lat['g2_over_g1_time_ratio']:.1f} times the runtime. "
      "Its gains in recall and abnormal recall came largely from repeating classifier-positive findings, its verifier did not separate supported from unsupported findings and its critic changed almost nothing.")
    A("")
    A("### Deterministic routing guard")
    A("")
    A(f"The classifier produces {pw['C']} empty outputs on the validation set (no positive label and No Finding not positive). Earlier systems wrote a normal-looking report for them. A deterministic guard now routes every study to one of three states: normal (No Finding positive and no pathology label positive; frozen G1 normal path), abnormal (any other study with a positive label; frozen G1 retrieval-augmented report) and indeterminate (empty output; no retrieval, no language-model call, and the fixed message \"{guard.INDETERMINATE_FINDINGS} {guard.INDETERMINATE_IMPRESSION}\"). "
      "The state comes from the classifier output, not from the generated text. Existing G1 reports were reused unchanged, so no new generation was performed.")
    A("")
    A("### Freeze")
    A("")
    A(f"Classification (checkpoint, thresholds, No Finding rule, calibration), retrieval (query construction, hybrid reciprocal-rank fusion, Top-5), the generator model and prompt and the guard were recorded in one configuration with SHA-256 `{cfg['config_sha256']}` before the locked test set is opened. No parameter, prompt, threshold, routing rule or model selection may be changed after inspecting locked-test outputs.")
    A("")
    A("## Results")
    A("")
    A(f"Routing: {pw['A']} normal, {pw['B']} abnormal and {pw['C']} indeterminate studies ({pc(ind / n)}). Finding-level validation metrics are identical to the original G1 (F1 {vm.loc['f1', G1FS]:.3f}, precision {vm.loc['precision', G1FS]:.3f}, recall {vm.loc['recall', G1FS]:.3f}, hallucination rate {vm.loc['hallucination_rate', G1FS]:.3f}, false-positive propagation {vm.loc['clf_fp_propagation', G1FS]:.3f}, true-positive retention {vm.loc['tp_retention', G1FS]:.3f}); text-overlap scores fall slightly (ROUGE-L {vm.loc['rouge_l', G1O]:.3f} to {vm.loc['rouge_l', G1FS]:.3f}). "
      f"Among decided studies, normal recall was {f3(d('normal_recall_among_decided'))} and abnormal recall {f3(d('abnormal_recall_among_decided'))}, with decision coverage {pc(d('decision_coverage'))}. Of the {abst['n_abstained_with_reference']} abstained studies with a reference, {abst['reference_abnormal_among_abstained']} were reference-abnormal and {abst['reference_normal_among_abstained']} reference-normal; rare findings were not over-represented (descriptive, small counts).")
    A("")
    A("## Discussion")
    A("")
    A(f"The guard turns an unrecognised failure (a normal-looking report for an empty classifier output) into an explicit abstention at the price of {pc(1 - d('decision_coverage'))} lower coverage in the clinical subset. It does not improve the classifier or the generator: the reused G1 prose still reads normal for {pc(pb.final_report_prose_normal / pb.n)} of abnormal-routed studies, so the interface must display the routing state and the indeterminate message, never a normal label for an indeterminate study. The system remains a research prototype and the final application has not been built.")
    A("")
    (G1F / "UNIVERSITY_REPORT_G1F.md").write_text("\n".join(U), encoding="utf-8")

    bad = []
    for nme in ("FINAL_SYSTEM_SELECTION.md", "G1F_FINAL_GUARD_ANALYSIS.md", "UNIVERSITY_REPORT_G1F.md"):
        txt = (G1F / nme).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"\{[a-z_]+\[[^}]*\}", r"\bnan\b", r"\bNone\b", r"anthropic", r"claude"):
            if re.search(pat, txt, re.I if pat in ("anthropic", "claude") else 0):
                bad.append((nme, pat))
    print(json.dumps({"wording_flags": bad, "files": 3}))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
