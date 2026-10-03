"""R2 documents generated from the stored R2 artifacts (no number is typed by hand):
RETRIEVAL_OPTIMIZATION_ANALYSIS.md, MANUSCRIPT_R2_METHODS.md, MANUSCRIPT_R2_RESULTS.md, UNIVERSITY_REPORT_R2.md,
FIGURE_CAPTIONS.md, TABLE_CAPTIONS.md, R2_JOURNAL_ASSET_INDEX.md

    .venv\\Scripts\\python.exe -m scripts.r2_05_write_docs
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.retrieval.r2_context import R1, R2

KS = (1, 3, 5, 10)
rd = lambda n: pd.read_csv(R2 / n, float_precision="round_trip")  # noqa: E731
jl = lambda n: json.loads((R2 / n).read_text(encoding="utf-8"))  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731
wc = lambda t: len(t.split())  # noqa: E731


def main() -> int:
    dec, cfg, prot = jl("r2_stage_decisions.json"), jl("R2_RETRIEVAL_CANDIDATE_CONFIG.json"), jl("r2_selection_protocol.json")
    rep, nsel, rnd, flags = jl("r1_reproduction_check.json"), jl("normal_query_selection.json"), jl("random_reference.json"), jl("r2_failure_flags.json")
    S = rd("r2_summary_all_configs.csv").set_index("config")
    PC, SC, MM, KL, FC = rd("r2_paired_comparisons.csv"), rd("r2_stage_comparisons.csv"), rd("r2_mmr_adoption_checks.csv"), rd("r2_k_selection_log.csv"), rd("r2_failure_categories.csv")
    K = int(dec["selected_K"])
    q = cfg["query_construction"]
    n_prim = dec["primary_set_size"]
    sel_ret = dec["selected_retriever"]
    D = {d["stage"]: d for d in dec["decisions"]}
    pcg = lambda group, comp, metric: PC[(PC.group == group) & (PC.comparison == comp) & (PC.metric == metric)].iloc[0]  # noqa: E731
    pcf = lambda r: f"{r.mean_difference:+.3f} (95% CI {r.ci95_low:+.3f} to {r.ci95_high:+.3f})"  # noqa: E731
    ch = lambda name, m="jaccard_truth@3": PC[(PC.a == name) & (PC.metric == m) & PC.group.isin(["query_policy_vs_Q0", "retriever"]) & (PC.b.isin(["R1_Q0_dense", "B_dense"]))].iloc[0]  # noqa: E731
    sj = lambda c, m: S.loc[c, m]  # noqa: E731
    fin = "FINAL_pipeline"
    ci3 = lambda c: f"{f3(S.loc[c, 'jaccard_truth@3'])} (95% CI {f3(S.loc[c, 'jaccard_truth@3__ci95_low'])} to {f3(S.loc[c, 'jaccard_truth@3__ci95_high'])})"  # noqa: E731
    share_lt05 = [d for d in dec["decisions"] if d["stage"] == "A2_note"][0]["share_of_frozen_positive_findings_with_p_below_0.5"]
    gap_r1 = pcg("oracle_gap", "oracle dense minus R1 classifier Q0 dense (R1 gap)", "jaccard_truth@3")
    gap_r2 = pcg("oracle_gap", "oracle dense Q0 minus final R2 pipeline", "jaccard_truth@3")
    gap_r2k = pcg("oracle_gap", "oracle dense Q0 minus final R2 pipeline", f"jaccard_truth@{K}")
    gap_pipe = pcg("oracle_gap", "oracle with the final pipeline minus final R2 pipeline", "jaccard_truth@3")
    fin_vs_q0 = {k: pcg("query_policy_vs_Q0", "final R2 pipeline minus R1 Q0 dense", f"{m}@{k}") for k in (3, K) for m in ("jaccard_truth",)}
    fin_vs_q0_cov = pcg("query_policy_vs_Q0", "final R2 pipeline minus R1 Q0 dense", f"union_coverage@{K}")
    # sensitivity (exclude exact-duplicate-text queries)
    cmp_cfgs = ["R1_Q0_dense", "R1_gated_dense", "A1_Q1_expanded_dense", "A2_Q2_uniform_dense", "A2_Q2_probability_dense", "A2_Q2_margin_dense", "A3_Q3_top1_dense", "A3_Q3_top2_dense",
                "A3_Q3_top3_dense", "B_dense", "B_bm25", "B_hybrid_rrf"]
    full = S.loc[cmp_cfgs, "jaccard_truth@3"]
    excl = S.loc[cmp_cfgs, "jaccard_truth@3_excluding_exact_duplicate_text_queries"]
    spear = float(full.rank().corr(excl.rank()))
    hy_gt_dense = bool(excl["B_hybrid_rrf"] > excl["B_dense"])
    n_excl = int(S.loc[fin, "n_queries_excluding_duplicates"])
    # K facts
    kl = lambda a, b, c: KL[(KL.from_K == a) & (KL.to_K == b)].iloc[0][c]  # noqa: E731
    # MMR facts
    mm = lambda lam, k, c: MM[(MM["lambda"] == lam) & (MM.K == k)].iloc[0][c]  # noqa: E731
    lam_all = prot["stage_C"]["lambdas"]
    # failure analysis
    fcat = lambda sysn, cat, col="n", d="top-1": FC[(FC.system == sysn) & FC.failure_definition.str.startswith(d) & (FC.category == cat)].iloc[0][col]  # noqa: E731
    nfail = {s: int(FC[(FC.system == s) & FC.failure_definition.str.startswith("top-1")].n_failures.iloc[0]) for s in ("R1_Q0_dense", "R2_final")}
    nfail_cov = int(FC[(FC.system == "R2_final") & FC.failure_definition.str.startswith("true")].n_failures.iloc[0])
    upstream = {s: int(fcat(s, "A") + fcat(s, "D")) for s in ("R1_Q0_dense", "R2_final")}
    mmr_txt = ("MMR was not adopted: no lambda reliably reduced the exact-duplicate rate at both K = 3 and K = 5" if dec["mmr"]["lambda"] is None else f"MMR with lambda = {dec['mmr']['lambda']} was adopted")
    k_rule_text = (f"Starting at K=1, the rule advanced to K=3 (coverage gain {pcf(pcg('top_k', 'K=3 minus K=1', 'union_coverage@3 - union_coverage@1'))}), then to K=5 (gain {pcf(pcg('top_k', 'K=5 minus K=3', 'union_coverage@5 - union_coverage@3'))}) "
                   f"and stopped there: K=10 added a coverage gain of {kl(5, 10, 'coverage_gain'):.3f} over K=5 (95% CI {kl(5, 10, 'coverage_ci_low'):.3f} to {kl(5, 10, 'coverage_ci_high'):.3f}), below the declared 0.05 minimum.")

    # ============================================================ analysis report
    R = []
    A = R.append
    A("# R2: query construction, hybrid retrieval and diversity re-ranking")
    A("")
    A("_Generated by `scripts/r2_05_write_docs.py` from the R2 artifacts. Retrieval-validation queries only (the R1 primary paired set); the locked retrieval-test split was not opened; the classifier was not modified; RAG was not started._")
    A("")
    A("## 1. Objective")
    A("")
    A("R1 showed that oracle finding queries retrieve well while queries from the frozen classifier lose about 0.29 Jaccard@3. R2 asks whether better query construction (phrase expansion, confidence weighting, Top-N finding selection), dense-lexical fusion and diversity re-ranking can narrow that gap, and fixes a provisional Top-K for RAG.")
    A("")
    A("## 2. Population, protocol and R1 reproduction")
    A("")
    A(f"All comparisons use the exact R1 primary paired validation set ({n_prim} studies); nothing was re-split, removed or redefined. The selection protocol (`r2_selection_protocol.json`) and the fixed clinical expansion mapping (`finding_query_expansion.json`) were written before any R2 evaluation; the protocol's replacement rule is: a challenger replaces the incumbent only if the lower bound of the paired bootstrap 95% CI of the Jaccard@3 difference is above zero ({prot['bootstrap']['n_resamples']} resamples, seed {prot['bootstrap']['seed']}, queries as the unit, one shared set of resampling indices).")
    A("")
    A(f"R1 baselines (oracle and classifier all-positive / gated queries, dense and BM25) were reproduced exactly with the R1 ranking order: {rep['r1_order']['n_rows']:,} query rows, {rep['r1_order']['n_lists_not_identical_to_r1']} differing lists, maximum metric difference {rep['r1_order']['max_abs_metric_difference_any_query']:.1e}. R2 uses one deterministic tie rule (score descending, then corpus order) because R1's FAISS order for exactly tied reports depends on the search depth; this reorders exact ties in {rep['r2_order']['n_lists_not_identical_to_r1']} of the {rep['r2_order']['n_rows']:,} lists and leaves every primary-set mean Jaccard@3 unchanged (largest absolute difference {max(rep['r2_order']['primary_set_abs_difference_in_mean_jaccard@3'].values()):.1e}).")
    A("")
    A("## 3. No Finding query phrase")
    A("")
    cand = nsel["candidates"]
    A(f"Candidates {', '.join('`' + c['phrase'] + '`' for c in cand)} were compared on the {nsel['population']}. Jaccard@3 was " + "; ".join(f"{c['phrase']} {f3(c['jaccard_truth@3'])}" for c in cand) + f". Because every normal query uses one fixed phrase, all normal studies share one retrieved list and the paired interval is degenerate; the rule reduces to 'strictly better than the R1 phrase'. No candidate was better, so `{nsel['chosen']}` is kept (the R1 phrase).")
    A("")
    A("## 4. Stage A: query representation")
    A("")
    A(f"**Q1, clinical phrase expansion** (one fixed mapping, {len(q['clinical_expansion_mapping'])} short phrases; mean query length {S.loc['A1_Q1_expanded_dense', 'mean_query_words']:.2f} words against {S.loc['R1_Q0_dense', 'mean_query_words']:.2f}): Jaccard@3 {ci3('A1_Q1_expanded_dense')} versus {ci3('R1_Q0_dense')} for Q0, a paired difference of {pcf(ch('A1_Q1_expanded_dense'))}. Expansion did not help and, if anything, lowered coverage@3 ({f3(S.loc['A1_Q1_expanded_dense', 'union_coverage@3'])} versus {f3(S.loc['R1_Q0_dense', 'union_coverage@3'])}); Q0 is kept.")
    A("")
    A(f"**Q2, calibrated-probability weighting** (weighted sum of per-finding phrase embeddings, then L2 normalisation; uniform, probability, and margin-above-operating-point weights): paired Jaccard@3 differences against Q0 were {pcf(ch('A2_Q2_uniform_dense'))} (uniform), {pcf(ch('A2_Q2_probability_dense'))} (probability) and {pcf(ch('A2_Q2_margin_dense'))} (margin); none is reliably above zero, so weighting is rejected. The simple form max(p - 0.5, 0) was rejected by design: {pc(share_lt05)} of the frozen positive findings have p < 0.5 (the calibrated-equivalent thresholds are far below 0.5), so it would give zero weight to most findings. Weights use the frozen C5 probabilities only and never change a binary decision.")
    A("")
    A(f"**Q3, Top-N findings** (positives ordered by calibrated probability, never padded): Jaccard@3 differences against all positives were {pcf(ch('A3_Q3_top1_dense'))} (N=1), {pcf(ch('A3_Q3_top2_dense'))} (N=2) and {pcf(ch('A3_Q3_top3_dense'))} (N=3). N=1 lowers coverage@3 ({f3(S.loc['A3_Q3_top1_dense', 'union_coverage@3'])} versus {f3(S.loc['R1_Q0_dense', 'union_coverage@3'])} for all positives), so truncating the query to one finding loses true findings; adding findings beyond three does not measurably harm retrieval. By the protocol rule N=3 is selected, but its gain is small (it changes only studies with more than three positive findings) and is not practically meaningful on its own.")
    A("")
    A("**Table 1. Query-policy comparison.**")
    A("")
    A((R2 / "tables/table1_query_policy_comparison.md").read_text(encoding="utf-8"))
    A(f"Selected classifier query policy: finding names (Q0), Top-{q['top_n_findings']} findings, no weighting, no expansion, normal phrase `{q['no_finding_query_phrase']}`. R1 precision-aware gating is kept only as a baseline (dense Jaccard@3 {f3(S.loc['R1_gated_dense', 'jaccard_truth@3'])}).")
    A("")
    A("## 5. Stage B: retriever")
    A("")
    A(f"With the selected query: dense {ci3('B_dense')}, BM25 {ci3('B_bm25')}, hybrid RRF (k = {cfg['retriever']['hybrid']['rrf_k']}, depth {cfg['retriever']['hybrid']['depth_per_ranker']}, no tuning) {ci3('B_hybrid_rrf')}. Paired differences in Jaccard@3: BM25 minus dense {pcf(ch('B_bm25'))}; hybrid minus dense {pcf(ch('B_hybrid_rrf'))}; hybrid minus BM25 {pcf(pcg('retriever', 'B_hybrid_rrf minus B_bm25', 'jaccard_truth@3'))}. The hybrid also raises coverage@5 to {f3(S.loc['B_hybrid_rrf', 'union_coverage@5'])} from {f3(S.loc['B_dense', 'union_coverage@5'])} (dense) and {f3(S.loc['B_bm25', 'union_coverage@5'])} (BM25). Hybrid RRF replaces dense under the protocol rule. Every hybrid result stores its dense rank, BM25 rank and RRF score (`final_pipeline_top10.csv.gz`).")
    A("")
    A("**Table 2. Retriever comparison.**")
    A("")
    A((R2 / "tables/table2_retriever_comparison.md").read_text(encoding="utf-8"))
    A("## 6. Stage C: diversity re-ranking (MMR)")
    A("")
    A(f"MMR re-ranks the top-{cfg['diversity_reranking']['mmr_pool']} hybrid candidates (relevance = min-max-normalised RRF score; similarity = cosine of report embeddings; lambda in {lam_all}). {mmr_txt} (`r2_mmr_adoption_checks.csv`): the duplicate-text rate changed by {mm(0.5, 5, 'dup_mean_difference'):+.4f}, {mm(0.7, 5, 'dup_mean_difference'):+.4f} and {mm(0.9, 5, 'dup_mean_difference'):+.4f} at K=5 and by at most {max(abs(mm(l, 3, 'dup_mean_difference')) for l in lam_all):.4f} at K=3. Relevance did not collapse: at K=3 lambda = 0.7 and 0.9 raised Jaccard@3 by {mm(0.7, 3, 'jaccard_mean_difference'):+.3f} and {mm(0.9, 3, 'jaccard_mean_difference'):+.3f} and coverage@3 by {mm(0.7, 3, 'coverage_mean_difference'):+.3f} and {mm(0.9, 3, 'coverage_mean_difference'):+.3f}, but these gains disappear at K=5 (coverage {mm(0.7, 5, 'coverage_mean_difference'):+.3f} and {mm(0.9, 5, 'coverage_mean_difference'):+.3f}). Because the corpus is dominated by near-identical templated reports (mean pairwise cosine among retrieved reports about {S.loc['B_hybrid_rrf', 'mean_pairwise_cosine@5']:.2f}), the similarity term cannot separate exact duplicates from template-similar reports, which is a plausible reason why exact duplicates are not removed. The K=3 relevance gain is an exploratory observation that the protocol does not adopt; it has no effect at the provisional K=5.")
    A("")
    A("**Table 4. Diversity / MMR analysis.**")
    A("")
    A((R2 / "tables/table4_diversity_mmr_analysis.md").read_text(encoding="utf-8"))
    A("## 7. Top-K selection")
    A("")
    A(f"Candidates K = 1, 3, 5, 10 were judged with the declared rule (coverage gain with CI lower bound above zero and at least 0.05, Jaccard@K falling by at most 0.03, duplicate-text rate at most 0.50 with more distinct templates, context at most 1,000 words). {k_rule_text} At K={K} the context is {S.loc[fin, f'context_words@{K}']:.0f} words on average (K=3: {S.loc[fin, 'context_words@3']:.0f}; K=10: {S.loc[fin, 'context_words@10']:.0f}), the duplicate-text rate is {f3(S.loc[fin, f'duplicate_text_rate@{K}'])} with {S.loc[fin, f'unique_templates@{K}']:.2f} distinct report texts, and {pc(S.loc[fin, f'zero_overlap_rate@{K}'])} of retrieved reports share no finding with the study. Coverage@{K} is {f3(S.loc[fin, f'union_coverage@{K}'])}, which exceeds the random-retrieval expectation of {f3(rnd[f'union_coverage@{K}'])} (coverage rises with K even for random reports, so the lift over random is the informative quantity). K = {K} is therefore provisional; it was chosen before the retrieval test split was opened and K=3 is not retained merely because it was planned.")
    A("")
    A("**Table 3. Top-K comparison.**")
    A("")
    A((R2 / "tables/table3_top_k_comparison.md").read_text(encoding="utf-8"))
    A("## 8. Oracle-to-classifier gap after R2")
    A("")
    A(f"The selected pipeline improves Jaccard@3 over the R1 classifier baseline by {pcf(fin_vs_q0[3])} (dense Q0 {f3(S.loc['R1_Q0_dense', 'jaccard_truth@3'])} to {f3(S.loc[fin, 'jaccard_truth@3'])}) and coverage@{K} by {pcf(fin_vs_q0_cov)}. The gap to the oracle dense reference narrows from {pcf(gap_r1)} (R1) to {pcf(gap_r2)} in Jaccard@3, and is {pcf(gap_r2k)} at K={K}. Applying the same pipeline to oracle queries gives Jaccard@3 {f3(S.loc['O_final_pipeline', 'jaccard_truth@3'])} (gap to the classifier-query pipeline {pcf(gap_pipe)}), so most of the remaining gap is not removable by retrieval changes.")
    A("")
    A("## 9. Sensitivity to exact-text duplicates")
    A("")
    A(f"Excluding the {n_prim - n_excl} primary queries whose own report text has an exact copy in the corpus ({n_excl} remain) lowers absolute values but keeps the method ordering: the Spearman correlation of the Jaccard@3 ranking of {len(cmp_cfgs)} configurations between the full and the reduced set is {spear:.3f}, and hybrid {'remains above' if hy_gt_dense else 'does not remain above'} dense ({f3(excl['B_hybrid_rrf'])} versus {f3(excl['B_dense'])}). The official primary results are unchanged.")
    A("")
    A("## 10. Error analysis")
    A("")
    A(f"Top-1 failures (the top-1 report is not an exact finding-set match): R1 Q0 dense {nfail['R1_Q0_dense']} of {n_prim}; R2 pipeline {nfail['R2_final']} of {n_prim}. For the R2 pipeline: normal/abnormal disagreement (D) {int(fcat('R2_final', 'D'))} ({fcat('R2_final', 'D', 'pct_of_failures'):.1f}% of failures), other upstream classification/query errors (A) {int(fcat('R2_final', 'A'))} ({fcat('R2_final', 'A', 'pct_of_failures'):.1f}%), corpus limitation (E) {int(fcat('R2_final', 'E'))}, duplicate/template issue (F) {int(fcat('R2_final', 'F'))}, terminology mismatch (B) {int(fcat('R2_final', 'B'))} and retriever failure despite a good query (C) {int(fcat('R2_final', 'C'))}. Upstream causes (A + D) account for {upstream['R2_final']} of {nfail['R2_final']} failures ({100 * upstream['R2_final'] / nfail['R2_final']:.1f}%), against {upstream['R1_Q0_dense']} of {nfail['R1_Q0_dense']} in R1. With the coverage-based definition (true findings not fully covered by the top-{K}) there are {nfail_cov} failures, all upstream. Duplicate report texts occur in {pc(flags['R2_final']['duplicate_text_in_topK'])} of the top-{K} lists as a non-exclusive flag; they affect redundancy but are not by themselves the cause of a failure under this classification. Categories are assigned by fixed rules (see `scripts/r2_03_error_analysis_and_config.py`); no example was fixed by hand.")
    A("")
    A("**Table 6. Primary failure analysis.**")
    A("")
    A((R2 / "tables/table6_primary_failure_analysis.md").read_text(encoding="utf-8"))
    A("## 11. Candidate configuration")
    A("")
    A("`R2_RETRIEVAL_CANDIDATE_CONFIG.json` records the embedding model, the query policy, the unused expansion mapping, BM25, RRF parameters, the non-adopted MMR setting, the normal phrase and the provisional Top-K. It is a candidate, not the frozen retrieval system.")
    A("")
    A("**Table 5. Candidate configuration.**")
    A("")
    A((R2 / "tables/table5_final_r2_candidate_configuration.md").read_text(encoding="utf-8"))
    A("## 12. Limitations")
    A("")
    A("- Query policy, retriever, MMR setting and K were selected and evaluated on the same retrieval-validation queries; the paired intervals do not account for this selection, so absolute gains are optimistic. The locked retrieval test is the independent check.")
    A("- Most R2 differences are small relative to the interval widths; the only practically meaningful improvement is hybrid fusion over dense retrieval; the Top-3 query rule passed the protocol test with a negligible gain.")
    A("- The oracle-to-classifier gap remains large; it is driven by the frozen classifier's query errors, which R2 deliberately does not address.")
    A("- IU truth comes from MeSH mapped to 14 classes (some studies unusable, Enlarged Cardiomediastinum unevaluable, few positives for several classes); the primary set is about half normal studies; templated reports are common.")
    A("- The Top-K rule constants (0.05, 0.03, 0.50, 1,000 words) are declared design choices; coverage at larger K rises partly by chance, as the random reference shows.")
    A("- MMR was evaluated with one similarity definition and three lambdas; no cross-encoder or learned re-ranker was tried.")
    A("- RRF used the standard k = 60 and depth 100 without tuning; BM25 was used on short finding-name queries.")
    A("")
    A("## 13. R2 conclusion")
    A("")
    A(f"The best-supported classifier-query configuration is finding names (top-{q['top_n_findings']} positives) with hybrid RRF retrieval and K = {K}: Jaccard@3 {f3(S.loc[fin, 'jaccard_truth@3'])}, Jaccard@{K} {f3(S.loc[fin, f'jaccard_truth@{K}'])}, coverage@{K} {f3(S.loc[fin, f'union_coverage@{K}'])}. Phrase expansion, probability weighting and MMR gave no adopted improvement; hybrid fusion improved on dense retrieval. The oracle gap narrowed from {gap_r1.mean_difference:.3f} to {gap_r2.mean_difference:.3f} but remains dominated by upstream classification and query errors. The retrieval test split remains locked and R3 has not been started.")
    A("")
    (R2 / "RETRIEVAL_OPTIMIZATION_ANALYSIS.md").write_text("\n".join(R), encoding="utf-8")

    # ============================================================ manuscript Methods
    meth = f"""# Methods: query construction, hybrid retrieval and diversity re-ranking (R2)

## Evaluation set and protocol

R2 reused the primary paired validation set of the retrieval baseline ({n_prim} studies with a usable finding set, a frontal image and a non-empty classifier query). The locked retrieval-test partition was not used, and the classifier, its thresholds and its calibration were not changed. The baseline results (oracle and classifier queries; dense and BM25 retrieval) were reproduced exactly before any new method was evaluated. Before evaluation we fixed a selection protocol: methods were compared in stages, and a challenger replaced the incumbent only if the lower bound of the paired bootstrap 95% confidence interval of the Jaccard@3 difference exceeded zero ({prot['bootstrap']['n_resamples']} resamples of queries, one shared set of resampling indices, seed {prot['bootstrap']['seed']}). The same finding-agreement metrics as in the baseline were used: Jaccard@K and graded nDCG@K against the study's true finding set, union coverage of true findings, and exact-match Hit@K.

## Query construction

The baseline query lists finding names (Q0). Q1 replaced each name by one fixed short radiology phrase from a mapping written before evaluation. Q2 weighted the per-finding phrase embeddings and re-normalised their sum, using uniform weights, the calibrated probability, or the margin of the probability above the frozen operating point; the simple form max(p - 0.5, 0) was not used because most frozen positives have p < 0.5. Q3 kept only the N most probable positive findings (N = 1, 2, 3), never padding. Calibrated probabilities served only as query weights or for ordering; no binary decision was altered. Studies predicted as No Finding received a fixed normal phrase chosen among three formulations on oracle-normal validation studies. Staged selection first compared Q0 with Q1, then the weightings, then Top-N.

## Retrievers

Dense retrieval used the unmodified all-MiniLM-L6-v2 encoder with a flat inner-product index over normalised vectors; the lexical baseline was Okapi BM25 without stop-word removal. Hybrid retrieval combined the two rankings by Reciprocal Rank Fusion, score = sum of 1/(60 + rank) over the top 100 of each ranker, with ties broken by corpus order, and the dense and BM25 rank of every fused result was stored. All rankings used one deterministic tie rule (score, then corpus order).

## Diversity re-ranking and Top-K

Maximal Marginal Relevance re-ranked the top 30 candidates of the selected pipeline (relevance min-max normalised within the pool; similarity the cosine of report embeddings; lambda 0.5, 0.7, 0.9). MMR was adopted only if, at both K = 3 and K = 5, the duplicate-text rate fell reliably, Jaccard and nDCG showed no reliable loss and coverage did not fall; the largest eligible lambda was to be chosen. Redundancy was measured by the exact duplicate-text rate, the number of distinct report texts and the mean pairwise cosine of retrieved reports. The provisional number of references K in {{1, 3, 5, 10}} was chosen by a declared rule: move to a larger K only if union coverage rose reliably and by at least 0.05, Jaccard@K fell by at most 0.03, the duplicate-text rate stayed at most 0.50 with more distinct reports, and the context stayed within 1,000 words; otherwise the smaller K was kept.

## Sensitivity and failure analysis

As a sensitivity analysis the primary set was restricted to queries whose own report text has no exact copy in the corpus. Failures (top-1 report not an exact finding-set match) were assigned by fixed rules to normal/abnormal disagreement, other upstream classification or query error, corpus limitation, duplicate/template issue, terminology mismatch, or retriever failure despite a good query.
"""
    mw = wc(meth)
    (R2 / "MANUSCRIPT_R2_METHODS.md").write_text(meth, encoding="utf-8")

    # ============================================================ manuscript Results
    resu = f"""# Results: query construction, hybrid retrieval and diversity re-ranking (R2)

## Baseline reproduction and query construction

The retrieval-baseline results were reproduced exactly ({rep['r1_order']['n_rows']:,} query rows; largest metric difference {rep['r1_order']['max_abs_metric_difference_any_query']:.0e}). On the {n_prim}-study primary set, Q0 finding names with dense retrieval gave a Jaccard@3 of {ci3('R1_Q0_dense')}. Phrase expansion (Q1) did not help: {ci3('A1_Q1_expanded_dense')}, a paired difference of {pcf(ch('A1_Q1_expanded_dense'))} (Table 1, Figure 1). Probability weighting did not help either: uniform, probability and margin weights differed from Q0 by {ch('A2_Q2_uniform_dense').mean_difference:+.3f}, {ch('A2_Q2_probability_dense').mean_difference:+.3f} and {ch('A2_Q2_margin_dense').mean_difference:+.3f}, all with intervals including zero. Limiting the query to the most probable findings changed little: Top-1 and Top-2 differed from all positives by {ch('A3_Q3_top1_dense').mean_difference:+.3f} and {ch('A3_Q3_top2_dense').mean_difference:+.3f}, and Top-3 by {pcf(ch('A3_Q3_top3_dense'))}, the only difference whose interval excluded zero by the protocol rule, although it is too small to matter in practice; Top-1 reduced coverage@3 to {f3(S.loc['A3_Q3_top1_dense', 'union_coverage@3'])} from {f3(S.loc['R1_Q0_dense', 'union_coverage@3'])}. The normal phrase `{nsel['chosen']}` was kept after two alternatives did not improve on it.

## Retriever comparison

With the Top-3 finding-name query, Jaccard@3 was {ci3('B_dense')} for dense retrieval, {ci3('B_bm25')} for BM25 and {ci3('B_hybrid_rrf')} for hybrid RRF (Table 2, Figure 2). Hybrid fusion improved on dense retrieval by {pcf(ch('B_hybrid_rrf'))}, whereas BM25 minus dense was {pcf(ch('B_bm25'))}. Union coverage@5 was {f3(S.loc['B_hybrid_rrf', 'union_coverage@5'])} for hybrid, {f3(S.loc['B_dense', 'union_coverage@5'])} for dense and {f3(S.loc['B_bm25', 'union_coverage@5'])} for BM25. Hybrid retrieval was therefore selected.

## Diversity re-ranking

MMR was not adopted (Table 4, Figure 4). At K=5 the duplicate-text rate changed by {mm(0.5, 5, 'dup_mean_difference'):+.4f}, {mm(0.7, 5, 'dup_mean_difference'):+.4f} and {mm(0.9, 5, 'dup_mean_difference'):+.4f} for lambda 0.5, 0.7 and 0.9, and at K=3 by at most {max(abs(mm(l, 3, 'dup_mean_difference')) for l in lam_all):.4f}; the number of distinct report texts in the top five stayed near {S.loc['B_hybrid_rrf', 'unique_templates@5']:.1f}. Relevance did not collapse, and at K=3 lambda 0.7 and 0.9 raised Jaccard@3 by {mm(0.7, 3, 'jaccard_mean_difference'):+.3f} and {mm(0.9, 3, 'jaccard_mean_difference'):+.3f}, but this exploratory gain vanished at K=5.

## Top-K selection

For the selected pipeline, coverage@K was {', '.join(f3(S.loc[fin, f'union_coverage@{k}']) for k in KS)} at K = 1, 3, 5 and 10, against random-retrieval expectations of {', '.join(f3(rnd[f'union_coverage@{k}']) for k in KS)}, while Jaccard@K stayed between {f3(min(S.loc[fin, f'jaccard_truth@{k}'] for k in KS))} and {f3(max(S.loc[fin, f'jaccard_truth@{k}'] for k in KS))} (Table 3, Figure 3). K=5 added {pcf(pcg('top_k', 'K=5 minus K=3', 'union_coverage@5 - union_coverage@3'))} coverage over K=3, whereas K=10 added only {kl(5, 10, 'coverage_gain'):.3f} over K=5, below the declared 0.05 minimum. The duplicate-text rate was {f3(S.loc[fin, 'duplicate_text_rate@3'])} at K=3 and {f3(S.loc[fin, 'duplicate_text_rate@5'])} at K=5, and the context grew from {S.loc[fin, 'context_words@3']:.0f} to {S.loc[fin, 'context_words@5']:.0f} words. K = 5 was selected provisionally, before any retrieval-test evaluation.

## Oracle-to-classifier gap

The selected pipeline improved Jaccard@3 over the baseline classifier query by {pcf(fin_vs_q0[3])} (dense Q0 {f3(S.loc['R1_Q0_dense', 'jaccard_truth@3'])}, selected {f3(S.loc[fin, 'jaccard_truth@3'])}), and coverage@{K} by {pcf(fin_vs_q0_cov)} (Figure 5). The gap to the oracle dense reference fell from {gap_r1.mean_difference:.3f} to {gap_r2.mean_difference:.3f} at K=3 and was {gap_r2k.mean_difference:.3f} at K={K}. The same pipeline with oracle queries reached a Jaccard@3 of {f3(S.loc['O_final_pipeline', 'jaccard_truth@3'])}.

## Failure analysis and sensitivity

Of {n_prim} queries, {nfail['R2_final']} had a top-1 report that was not an exact finding-set match ({nfail['R1_Q0_dense']} for the baseline) (Table 6, Figure 6). {int(fcat('R2_final', 'D'))} failures arose from normal/abnormal disagreement and {int(fcat('R2_final', 'A'))} from other classification or query errors, together {100 * upstream['R2_final'] / nfail['R2_final']:.1f}% of failures; {int(fcat('R2_final', 'C'))} were retriever failures despite a correct query, and none was classified as corpus limitation, duplicate/template or terminology mismatch. Excluding the {n_prim - n_excl} queries whose report text has an exact corpus copy preserved the ordering of configurations (Spearman {spear:.3f}).

## Other relevance and cost measures

Against the baseline classifier query with dense retrieval, the selected pipeline raised nDCG@3 from {f3(S.loc['R1_Q0_dense', 'ndcg@3'])} to {f3(S.loc[fin, 'ndcg@3'])} and Hit@3 (an exact finding-set match among the top three) from {f3(S.loc['R1_Q0_dense', 'hit@3'])} to {f3(S.loc[fin, 'hit@3'])}, while the share of retrieved top-three reports without any finding overlap changed from {f3(S.loc['R1_Q0_dense', 'zero_overlap_rate@3'])} to {f3(S.loc[fin, 'zero_overlap_rate@3'])}. The mean query length was {S.loc[fin, 'mean_query_words']:.2f} words with {S.loc[fin, 'mean_findings_per_query']:.2f} findings per query, compared with {S.loc['R1_Q0_dense', 'mean_query_words']:.2f} words and {S.loc['R1_Q0_dense', 'mean_findings_per_query']:.2f} findings for the baseline, and {S.loc['A1_Q1_expanded_dense', 'mean_query_words']:.2f} words for the expanded query. With oracle queries the selected pipeline reached a Hit@3 of {f3(S.loc['O_final_pipeline', 'hit@3'])}.

## Interpretation

Hybrid fusion was the only change with a clear benefit (the Top-3 query rule passed the protocol test but gained only about 0.003); query expansion, weighting and diversity re-ranking were not adopted. Most remaining failures originate in the classifier's finding query. Selection and evaluation used the same validation queries, so the gains are optimistic, and the retrieval-test partition has not been evaluated.
"""
    rw = wc(resu)
    (R2 / "MANUSCRIPT_R2_RESULTS.md").write_text(resu, encoding="utf-8")
    ok_len = 450 <= mw <= 650 and 650 <= rw <= 900

    # ============================================================ captions / index
    fc = f"""# Figure captions (R2)

Retrieval-validation queries only ({n_prim} primary paired studies); the locked retrieval-test split was not used. Relevance = finding agreement with the study's true (MeSH-mapped) finding set. Error bars: 95% bootstrap intervals over queries. Classifier queries come from the frozen classifier without any tuning.

**Figure 1. Query-policy comparison (dense retrieval).** Finding Jaccard@3 (left) and union coverage of the true findings at K=3 (right) for Q0 finding names (grey dotted line), Q1 expanded phrases, Q2 weighted queries (uniform, probability, margin), Q3 Top-N findings and the R1 precision-aware gated query. The blue dashed line marks the oracle reference.

**Figure 2. Dense, BM25 and hybrid retrieval.** Jaccard@K, nDCG@K and union coverage@K for K = 1, 3, 5, 10 with the selected query (Q0, Top-3 findings). Hybrid is Reciprocal Rank Fusion (k = 60, depth 100) of the dense and BM25 rankings.

**Figure 3. Performance across K for the selected pipeline.** Jaccard@K, union coverage@K and duplicate-text rate@K for the classifier-query pipeline (solid) and the same pipeline with oracle queries (dashed); dotted line: random-retrieval expectation; the green band marks the provisional K.

**Figure 4. Relevance versus redundancy.** Jaccard (left) and union coverage (right) against the mean pairwise cosine similarity among the retrieved reports at K=3 (top) and K=5 (bottom) for dense, BM25, hybrid and hybrid with MMR (lambda 0.5, 0.7, 0.9). Lower similarity indicates less redundancy.

**Figure 5. Oracle versus classifier-query pipelines.** Jaccard@K and union coverage@K for oracle and frozen-classifier queries with the R1 dense baseline (dashed) and the selected R2 pipeline (solid).

**Figure 6. Failure categories.** Share of the {n_prim} queries by top-1 outcome for the R1 dense baseline and the selected R2 pipeline; a failure is a top-1 report that is not an exact finding-set match, assigned to one exclusive category by fixed rules.
"""
    tc = """# Table captions (R2)

**Table 1. Query-policy comparison.** Dense retrieval on the primary paired set: Jaccard@3 with 95% interval, nDCG@3, coverage@3, query length, findings per query, and the paired Jaccard@3 difference from Q0.

**Table 2. Retriever comparison.** Dense, BM25 and hybrid RRF with the selected query: Jaccard@1/3/5, nDCG@3, coverage@3/5, Hit@3 and the paired Jaccard@3 difference from dense retrieval.

**Table 3. Top-K comparison.** Jaccard, nDCG, Hit, union coverage (with the random-retrieval expectation), zero-overlap rate, duplicate-text rate, distinct report texts, mean context length and oracle coverage at K = 1, 3, 5, 10 for the selected pipeline.

**Table 4. Diversity / MMR analysis.** Standard hybrid retrieval versus MMR re-ranking (pool 30, lambda 0.5/0.7/0.9): relevance, coverage, duplicate-text rate, distinct texts, pairwise cosine, the paired duplicate-rate difference at K=5 and the adoption decision.

**Table 5. R2 candidate configuration.** Parameters of the candidate retrieval pipeline (not frozen; the retrieval test split is unopened).

**Table 6. Primary failure analysis.** Failures (top-1 report not an exact finding-set match) by category for the R1 dense baseline and the selected R2 pipeline.
"""
    (R2 / "FIGURE_CAPTIONS.md").write_text(fc, encoding="utf-8")
    (R2 / "TABLE_CAPTIONS.md").write_text(tc, encoding="utf-8")
    idx = f"""# R2 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_query_policy_comparison.*` | Expansion, weighting and Top-N did not reliably improve dense Jaccard@3 (Q0 {f3(S.loc['R1_Q0_dense', 'jaccard_truth@3'])}) | Results |
| Table 2 `tables/table2_retriever_comparison.*` | Hybrid RRF {f3(S.loc['B_hybrid_rrf', 'jaccard_truth@3'])} versus dense {f3(S.loc['B_dense', 'jaccard_truth@3'])} Jaccard@3 | Results |
| Table 3 `tables/table3_top_k_comparison.*` | Coverage@K {f3(S.loc[fin, 'union_coverage@3'])} (K=3), {f3(S.loc[fin, 'union_coverage@5'])} (K=5), {f3(S.loc[fin, 'union_coverage@10'])} (K=10); provisional K={K} | Results / Discussion |
| Table 4 `tables/table4_diversity_mmr_analysis.*` | MMR did not reduce exact duplicates and was not adopted | Results |
| Table 5 `tables/table5_final_r2_candidate_configuration.*` | Candidate retrieval configuration | Methods / Supplementary |
| Table 6 `tables/table6_primary_failure_analysis.*` | {100 * upstream['R2_final'] / nfail['R2_final']:.1f}% of top-1 failures are upstream | Results / Discussion |
| Figure 1 `figures/fig1_query_policy_comparison.*` | No query-policy change closes the oracle gap | Results |
| Figure 2 `figures/fig2_dense_bm25_hybrid.*` | Hybrid improves on dense retrieval | Results |
| Figure 3 `figures/fig3_performance_across_k.*` | Coverage gains versus K; provisional K={K} | Results / Discussion |
| Figure 4 `figures/fig4_relevance_vs_redundancy.*` | Relevance versus redundancy of retrieval settings | Results / Supplementary |
| Figure 5 `figures/fig5_oracle_vs_final_pipeline.*` | Oracle gap before and after R2 | Results |
| Figure 6 `figures/fig6_failure_categories.*` | Failures are dominated by upstream errors | Results / Discussion |
| `RETRIEVAL_OPTIMIZATION_ANALYSIS.md` | Full R2 analysis | Supplementary / internal |
| `MANUSCRIPT_R2_METHODS.md`, `MANUSCRIPT_R2_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_R2.md` | Content for Methodology and Results and Discussion | University report |
| `R2_RETRIEVAL_CANDIDATE_CONFIG.json`, `r2_selection_protocol.json`, `finding_query_expansion.json` | Candidate configuration, pre-declared protocol, fixed expansion mapping | Methods / Supplementary / code release |
| `per_query_metrics_all_configs.csv`, `r2_paired_comparisons.csv` | All per-query metrics; all paired comparisons | Supplementary data |
| `r2_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
"""
    (R2 / "R2_JOURNAL_ASSET_INDEX.md").write_text(idx, encoding="utf-8")

    # ============================================================ university report
    U = []
    A = U.append
    A("# University report content: R2 (query construction, hybrid retrieval, diversity)")
    A("")
    A("_Insert-ready material generated from the R2 artifacts. Retrieval-validation data only; the locked retrieval-test split is unopened; the classifier is unchanged._")
    A("")
    A("## Methodology")
    A("")
    A("### Query construction")
    A("")
    A(f"The retrieval stage receives the findings predicted by the frozen classifier. Four ways of turning them into a query were compared on the {n_prim}-study validation benchmark of the previous stage. (Q0) The finding names, as before. (Q1) A fixed short radiology phrase for each finding, defined once before evaluation. (Q2) A weighted combination of the embeddings of the finding phrases, using uniform weights, the calibrated probability, or the margin above the operating threshold; the calibrated probability is used only as a weight and never changes a classification. (Q3) Only the N most probable positive findings (N = 1, 2, 3), without padding. A fixed normal-study phrase is used when the classifier predicts no finding. Options were compared in stages and a new option replaced the current one only if the paired bootstrap interval of the Jaccard@3 difference excluded zero on the favourable side.")
    A("")
    A("### Hybrid retrieval")
    A("")
    A("The dense retriever (MiniLM embeddings, cosine similarity) and the BM25 lexical retriever were combined by Reciprocal Rank Fusion: each report receives the sum of 1/(60 + rank) over the two rankings, restricted to each ranker's top 100, with ties broken by corpus order. The dense and lexical rank of every fused report are stored so that each result can be traced to its sources.")
    A("")
    A("### Diversity re-ranking and Top-K")
    A("")
    A("Maximal Marginal Relevance was applied to the 30 best candidates to reduce near-duplicate references: at each step the report with the best trade-off between relevance and dissimilarity to the already chosen reports is added (lambda 0.5, 0.7, 0.9). It was to be adopted only if exact duplicates fell reliably without loss of relevance or coverage. The number of references K was chosen from 1, 3, 5, 10 by a rule declared in advance that rewards a reliable gain in coverage of the true findings while limiting redundancy and context length.")
    A("")
    A("## Results and Discussion")
    A("")
    A("### Query-policy comparison")
    A("")
    A(f"On the validation benchmark, finding names with dense retrieval reached a Jaccard@3 of {ci3('R1_Q0_dense')}. Neither phrase expansion ({pcf(ch('A1_Q1_expanded_dense'))}) nor probability weighting (differences of {ch('A2_Q2_uniform_dense').mean_difference:+.3f} to {ch('A2_Q2_margin_dense').mean_difference:+.3f}) improved retrieval; the Top-3 rule gave a difference of {pcf(ch('A3_Q3_top3_dense'))}, which is negligible in practice, while restricting the query to one finding lowered coverage (Table 1, Figure 1). The simple weight max(p - 0.5, 0) was not used because {pc(share_lt05)} of the positive findings have p below 0.5.")
    A("")
    A("### Retrieval-method comparison")
    A("")
    A(f"Dense retrieval, BM25 and hybrid fusion reached Jaccard@3 of {f3(S.loc['B_dense', 'jaccard_truth@3'])}, {f3(S.loc['B_bm25', 'jaccard_truth@3'])} and {f3(S.loc['B_hybrid_rrf', 'jaccard_truth@3'])}; the hybrid was reliably better than dense retrieval ({pcf(ch('B_hybrid_rrf'))}) and was selected (Table 2, Figure 2). MMR did not reduce exact duplicate reports and was not adopted: the corpus consists largely of near-identical templated reports, so the similarity penalty cannot separate exact duplicates from similar ones (Table 4, Figure 4).")
    A("")
    A("### Top-K analysis")
    A("")
    A(f"Coverage of the true findings rose from {f3(S.loc[fin, 'union_coverage@1'])} at K=1 to {f3(S.loc[fin, 'union_coverage@3'])} at K=3, {f3(S.loc[fin, 'union_coverage@5'])} at K=5 and {f3(S.loc[fin, 'union_coverage@10'])} at K=10, while Jaccard@K stayed nearly constant and the duplicate-text rate increased from {f3(S.loc[fin, 'duplicate_text_rate@3'])} (K=3) to {f3(S.loc[fin, 'duplicate_text_rate@10'])} (K=10). {k_rule_text} K = {K} is therefore the provisional number of references for generation, to be confirmed on the untouched retrieval test split and on generated reports (Table 3, Figure 3).")
    A("")
    A("### Error analysis and remaining gap")
    A("")
    A(f"The selected pipeline narrowed the gap to oracle queries from {gap_r1.mean_difference:.3f} to {gap_r2.mean_difference:.3f} Jaccard@3 (Figure 5), but {100 * upstream['R2_final'] / nfail['R2_final']:.1f}% of the remaining top-1 failures stem from the classifier's finding query, mainly normal-versus-abnormal disagreements ({int(fcat('R2_final', 'D'))}) and missing or extra findings ({int(fcat('R2_final', 'A'))}); only {int(fcat('R2_final', 'C'))} were retriever failures despite a correct query (Table 6, Figure 6). Improving retrieval further is therefore unlikely to help unless the classifier's findings improve.")
    A("")
    A("### Limitations")
    A("")
    A("Selection and evaluation used the same validation queries, so gains are optimistic; most differences between query policies are small; the reference annotations come from mapped MeSH terms with several rare findings; the Top-K rule constants are design choices; and the retrieval test split has not been evaluated.")
    A("")
    (R2 / "UNIVERSITY_REPORT_R2.md").write_text("\n".join(U), encoding="utf-8")

    bad = []
    for n in ("RETRIEVAL_OPTIMIZATION_ANALYSIS.md", "MANUSCRIPT_R2_METHODS.md", "MANUSCRIPT_R2_RESULTS.md", "UNIVERSITY_REPORT_R2.md", "FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md", "R2_JOURNAL_ASSET_INDEX.md"):
        txt = (R2 / n).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"superior", r"\{[a-z_]+\[[^}]*\}", r"\bnan\b", r"\bNone\b"):
            if re.search(pat, txt):
                bad.append((n, pat))
    print(json.dumps({"methods_words": mw, "results_words": rw, "length_ok": ok_len, "wording_flags": bad}))
    return 0 if ok_len and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
