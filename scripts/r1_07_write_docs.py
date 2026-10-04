"""R1 documents generated from the stored R1 artifacts (no number is typed by hand):
RETRIEVAL_BASELINE_ANALYSIS.md, MANUSCRIPT_R1_METHODS.md, MANUSCRIPT_R1_RESULTS.md, UNIVERSITY_REPORT_R1.md,
FIGURE_CAPTIONS.md, TABLE_CAPTIONS.md, R1_JOURNAL_ASSET_INDEX.md

    .venv\\Scripts\\python.exe -m scripts.r1_07_write_docs
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.retrieval.findings import ABNORMAL
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
C6 = PROJECT_ROOT / "results/classification/experiments/c6_final_test"
KS = (1, 3, 5, 10)
RN = {"dense_minilm": "dense", "lexical_bm25": "lexical"}
PN = {"oracle": "oracle", "classifier_all_positive": "all-positive classifier", "classifier_gated": "precision-aware gated classifier"}
rd = lambda n: pd.read_csv(OUT / n, float_precision="round_trip")  # noqa: E731
jl = lambda n: json.loads((OUT / n).read_text(encoding="utf-8"))  # noqa: E731
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100 * v:.1f}%"  # noqa: E731
wc = lambda t: len(t.split())  # noqa: E731


def main() -> int:
    inv, split, qs, ms = jl("iu_xray_dataset_inventory.json"), jl("retrieval_split.json"), jl("r1_query_sets.json"), jl("r1_mapping_and_corpus_stats.json")
    emb, lex, gates, dsum = jl("embedding_metadata.json"), jl("lexical_index_metadata.json"), jl("precision_aware_query_gates.json"), jl("iu_domain_shift_summary.json")
    rnd, leak, crun = jl("random_baseline_expectation.json"), jl("self_retrieval_check.json"), jl("iu_classifier_run.json")
    c6 = json.loads((C6 / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    summ, ci, pdiff = rd("retrieval_summary_primary.csv"), rd("retrieval_bootstrap_ci.csv"), rd("retrieval_paired_differences.csv")
    cq, rem, gt = rd("classifier_query_comparison.csv"), rd("gating_removed_terms.csv"), rd("retrieval_gate_table.csv")
    shift, ea = rd("iu_domain_shift_per_finding.csv"), rd("retrieval_error_analysis.csv")
    allf, dupx = rd("retrieval_summary_all_studies_empty_as_failure.csv"), rd("retrieval_summary_excluding_duplicate_text.csv")
    oall = rd("retrieval_summary_oracle_all_eligible.csv")
    S = lambda rt, pol, c: float(summ[(summ.retriever == rt) & (summ.policy == pol)][c].iloc[0])  # noqa: E731
    CI = lambda rt, pol, m: ci[(ci.retriever == rt) & (ci.policy == pol) & (ci.metric == m)].iloc[0]  # noqa: E731
    PD = lambda rt, comp, m: pdiff[(pdiff.retriever == rt) & (pdiff.comparison == comp) & (pdiff.metric == m)].iloc[0]  # noqa: E731
    cij = lambda rt, pol, m: f"{f3(CI(rt, pol, m)['mean'])} (95% CI {f3(CI(rt, pol, m).ci95_low)} to {f3(CI(rt, pol, m).ci95_high)})"  # noqa: E731
    pds = lambda rt, comp, m: f"{PD(rt, comp, m).mean_difference:+.3f} (95% CI {PD(rt, comp, m).ci95_low:+.3f} to {PD(rt, comp, m).ci95_high:+.3f})"  # noqa: E731
    G = "classifier_gated minus classifier_all_positive"
    sz, inc = split["sizes"], split["integrity"]
    n_prim = qs["primary_paired_set_non_empty_classifier_query"]
    gated_ci_all_include_zero = all(PD(rt, G, m).ci95_low <= 0 <= PD(rt, G, m).ci95_high for rt in RN for m in ("jaccard_truth@3", "ndcg@3", "jaccard_truth@5", "ndcg@5"))
    sig = [f"{RN[rt]} {m}" for rt in RN for m in ("jaccard_truth@1", "jaccard_truth@3", "jaccard_truth@5", "ndcg@3", "ndcg@5")
           if not (PD(rt, G, m).ci95_low <= 0 <= PD(rt, G, m).ci95_high)]
    answer = ("No reliable improvement was found" if gated_ci_all_include_zero and not sig else
              "No consistent improvement was found")
    answer_detail = (f"For Jaccard@3 the paired difference (gated minus all-positive) was {pds('dense_minilm', G, 'jaccard_truth@3')} with dense retrieval and "
                     f"{pds('lexical_bm25', G, 'jaccard_truth@3')} with lexical retrieval; for nDCG@3 it was {pds('dense_minilm', G, 'ndcg@3')} and {pds('lexical_bm25', G, 'ndcg@3')}."
                     + (f" The only interval excluding zero was for {', '.join(sig)} (a decrease)." if sig and all(PD(rt, G, m).mean_difference < 0 for rt in RN for m in ("jaccard_truth@1", "jaccard_truth@3", "jaccard_truth@5", "ndcg@3", "ndcg@5") if f"{RN[rt]} {m}" in sig) else ""))
    # top-K facts
    cov = {(rt, pol, k): S(rt, pol, f"union_coverage@{k}") for rt in RN for pol in PN for k in KS}
    dup = {(rt, k): S(rt, "oracle", f"duplicate_text_rate@{k}") for rt in RN for k in KS}
    zero_c = {k: S("dense_minilm", "classifier_all_positive", f"zero_overlap_rate@{k}") for k in KS}
    rng_k = {(rt, pol): max(S(rt, pol, f"jaccard_truth@{k}") for k in KS) - min(S(rt, pol, f"jaccard_truth@{k}") for k in KS) for rt in RN for pol in PN}
    max_rng = max(rng_k.values())
    topk_text = (f"With oracle queries the finding Jaccard changes little across K (range {rng_k[('dense_minilm', 'oracle')]:.3f} for dense; dense {', '.join(f3(S('dense_minilm', 'oracle', f'jaccard_truth@{k}')) for k in KS)} at K = 1, 3, 5, 10), while the union coverage of the true findings rises from "
                 f"{f3(cov[('dense_minilm', 'oracle', 1)])} (K=1) to {f3(cov[('dense_minilm', 'oracle', 3)])} (K=3), {f3(cov[('dense_minilm', 'oracle', 5)])} (K=5) and {f3(cov[('dense_minilm', 'oracle', 10)])} (K=10); "
                 f"most of the gain is reached by K=3. With all-positive classifier queries the Jaccard range is {rng_k[('dense_minilm', 'classifier_all_positive')]:.3f}, but coverage keeps rising ({f3(cov[('dense_minilm', 'classifier_all_positive', 1)])}, {f3(cov[('dense_minilm', 'classifier_all_positive', 3)])}, "
                 f"{f3(cov[('dense_minilm', 'classifier_all_positive', 5)])}, {f3(cov[('dense_minilm', 'classifier_all_positive', 10)])}), and the share of retrieved reports with zero finding overlap stays near {pc(zero_c[3])} at every K. "
                 f"Redundancy grows with K: with dense oracle retrieval {pc(dup[('dense_minilm', 3)])} of the top-3 and {pc(dup[('dense_minilm', 10)])} of the top-10 reports have text identical to another retrieved report (templated normal reports).")
    top3_verdict = ("Top-3 is a defensible default for oracle-quality queries (most of the achievable coverage, no Jaccard loss) but it is not shown to be optimal: for classifier-driven queries coverage still improves at larger K, "
                    "redundancy grows with K while the share of retrieved reports without finding overlap stays roughly constant, and the choice of K is deferred to R2 where it can be judged on generated reports.")
    # error analysis (dense; all-positive classifier)
    e = ea[ea.policy == "classifier_all_positive"]
    oc = e.outcome.value_counts()
    sub = e[e.outcome != "good"].classifier_error_subtype.value_counts()
    eo = ea[ea.policy == "oracle"].outcome.value_counts()
    n_cls_err = int((e.attribution == "classifier_query_error").sum())
    n_ret_mis = int((e.attribution == "retrieval_mismatch_despite_correct_query").sum())
    n_map_flag = int(e[e.outcome != "good"].mapping_uncertainty_flag.sum())
    # domain shift
    sh = shift[(shift.finding != "No Finding") & (shift.iu_truth_positive_studies > 0)].copy()
    sh["ratio"] = sh.predicted_frequency_all_frontal_studies / sh.iu_truth_frequency_eligible_studies
    over = sh.sort_values("ratio", ascending=False).head(3)
    under = sh.sort_values("ratio").head(3)
    nf = shift[shift.finding == "No Finding"].iloc[0]
    ecm = shift[shift.finding == "Enlarged Cardiomediastinum"].iloc[0]
    rare = [l for l in ABNORMAL if bool(rem.set_index("finding").loc[l, "rare_class_c2_definition"])]
    rr = rem.set_index("finding").loc[rare]
    gated_cls = gates["classes_with_gate"]
    no_gate = gates["classes_without_gate"]
    removed_total, pos_total = int(rem.removed_by_gate_low_confidence.sum()), int(rem.frozen_positive_studies.sum())
    gtab = gt.set_index("finding")
    prec_gain = (gtab.precision_after - gtab.precision_before)
    rec_loss = (gtab.recall_before - gtab.recall_after)
    cmp3 = cq.set_index(["retriever", "query_policy"])
    fb = cq[(cq.retriever == "dense_minilm") & (cq.query_policy == "classifier_gated")].iloc[0]
    ex_all = dsum["queries_with_exact_truth_match_all_positive"], dsum["queries_with_exact_truth_match_gated"]

    # ============================================================ analysis report
    R = []
    A = R.append
    A("# R1: retrieval baseline and vision-to-retrieval evaluation")
    A("")
    A("_Generated by `scripts/r1_07_write_docs.py` from the R1 artifacts. Retrieval-validation queries only; the locked retrieval-test split was not opened. The frozen classifier was not modified._")
    A("")
    A("## 1. Objective")
    A("")
    A("Establish a defensible retrieval baseline for `Chest X-ray → frozen classifier → finding query → retrieval → RAG → preliminary report`, and measure how much retrieval quality is lost between a perfect finding query (oracle) and the query produced by the frozen classifier on IU X-Ray, including whether a precision-aware query gate helps. RAG and report generation are not part of R1.")
    A("")
    A("## 2. IU X-Ray dataset preparation")
    A("")
    A(f"Local files were used (nothing downloaded): {inv['reports_studies']:,} reports, {inv['images_listed_in_projections_csv']:,} listed images ({inv['frontal_images']:,} frontal, {inv['lateral_images']:,} lateral; {inv['image_files_on_disk']:,} files on disk, {inv['disk_images_not_listed']} not listed), {inv['studies_with_multiple_images']:,} studies with more than one image and {inv['studies_without_frontal']} studies without a frontal image. After cleaning, {inv['sections_after_cleaning']['with_findings']:,} reports have a Findings section, {inv['sections_after_cleaning']['with_impression']:,} an Impression, {inv['sections_after_cleaning']['with_both']:,} both and {inv['sections_after_cleaning']['with_neither']} neither. Terminology available: MeSH indexing ({inv['terminology_available']['MeSH']['distinct_terms']} distinct terms; {inv['terminology_available']['MeSH']['studies_indexed_normal']:,} studies indexed `normal`) and a Problems field that repeats the MeSH terms. The inventory is `iu_xray_dataset_inventory.json`.")
    A("")
    A("Retrieval text = cleaned Findings + Impression (Impression or Findings alone when only one exists; duplicated headings, markup and the `XXXX` de-identification placeholder removed; Indication and Comparison excluded as metadata). Original text is stored separately in `retrieval_corpus.csv`; no text is rewritten or generated.")
    A("")
    A("## 3. Leakage-safe retrieval split")
    A("")
    A(f"The Phase 1 study-level split is reused, not recreated (project rule: no ad hoc re-splitting). It is the preferred 70/15/15 split (seed {split['provenance']['seed']}), its unit is the report id so every image of a study stays together, and it stratifies on MeSH normal versus other. Reference corpus: {sz['reference_corpus_studies']:,} training studies with text. Retrieval-validation queries: {sz['validation_studies']:,} studies ({sz['validation_images']:,} images). Locked retrieval test: {sz['locked_test_studies']:,} studies ({sz['locked_test_images']:,} images), listed by id only and never read.")
    A("")
    A(f"Checks (status {split['status']}): corpus ∩ validation = {inc['corpus_and_validation']}, corpus ∩ test = {inc['corpus_and_test']}, validation ∩ test = {inc['validation_and_test']}; studies whose images span more than one split = {inc['studies_whose_images_span_more_than_one_split']}; image files in more than one split = {inc['image_files_in_more_than_one_split']}. Self-retrieval check: {leak['retrieved_rows']:,} retrieved rows, {leak['retrieved_equal_to_query_study']} equal to the query study, {leak['retrieved_ids_in_validation']} from validation, {leak['retrieved_ids_in_locked_test']} from the locked test split (status {leak['status']}). Identical report text appears across different studies (generic normal reporting): {leak['validation_queries_whose_report_text_equals_a_corpus_report']} validation reports have an exact text copy in the corpus; this is not self-retrieval, and the primary-set sensitivity analysis below excludes such queries.")
    A("")
    A("## 4. Finding mapping")
    A("")
    A(f"IU MeSH terms were mapped to the frozen 14-class vocabulary with explicit rules (`iu_finding_mapping.json`): direct term mappings from the Phase 1 lexicon, two qualifier-conditional rules (Cardiac Shadow/enlarged → Cardiomegaly; Thickening/pleura → Pleural Other), and an explicit list of ambiguous terms that are **not** mapped (Pulmonary Congestion, Density, Mediastinum, other Cardiac Shadow and Thickening qualifiers). Terms outside the vocabulary (spine, aorta, granuloma, emphysema and others) are stored as unmapped. No IU term maps to Enlarged Cardiomediastinum. A study's evaluation set is its mapped abnormal findings, or {{No Finding}} if it is indexed normal and has none; a study with neither has no usable representation. Of {ms['validation_studies']} validation studies, {ms['validation_with_eval_set']} have an evaluation set ({ms['validation_indexed_normal']} normal, {ms['validation_with_abnormal_mapped_finding']} with a mapped abnormal finding) and {ms['validation_without_eval_set_unusable_as_queries']} are excluded from evaluation; {qs['eligible_and_classified']} of the {ms['validation_with_eval_set']} have a frontal image, and {qs['primary_paired_set_non_empty_classifier_query']} of those have a non-empty classifier query (the primary paired set; {qs['eligible_but_empty_classifier_query']} empty).")
    A("")
    A("## 5. Dense retrieval")
    A("")
    A(f"`{emb['model_identifier']}` (revision `{emb['model_snapshot_revision'][:12]}`, not fine-tuned) embedded the {emb['n_corpus_studies']:,} corpus reports; the embedding dimension was verified programmatically as {emb['embedding_dimension_verified']}; vectors are L2-normalised (norm range {emb['norm_min']:.6f}–{emb['norm_max']:.6f}) and stored in a FAISS `{emb['faiss_index_type']}` (inner product = cosine). {emb['corpus_texts_truncated_by_encoder']} corpus texts exceed the encoder limit of {emb['max_seq_length_tokens']} tokens and are truncated. Index, id mapping and metadata are saved under R1.")
    A("")
    A("## 6. Lexical baseline")
    A("")
    A(f"Okapi BM25 (k1 = {lex['k1']}, b = {lex['b']}), implemented in-house; lowercase alphanumeric tokens, no stop-word removal or stemming; ties broken by corpus order; vocabulary {lex['vocabulary_size']:,} tokens. No hyper-parameter was tuned.")
    A("")
    A("## 7. Query construction")
    A("")
    A("Query phrases are fixed before evaluation (`src/retrieval/queries.py`) and emitted in a fixed class order: oracle = the study's mapped finding names (never the report text); classifier all-positive = every frozen-positive finding among the 13 non-No-Finding classes; if none and No Finding is positive, the fixed query `no acute abnormality`; if neither, the query is recorded as empty (no finding is invented). The classifier query uses the first frontal image of the study, the frozen preprocessing, checkpoint, Platt calibrators, C4 thresholds and No Finding rule; this is an external, domain-shift application, and nothing was tuned on IU X-Ray.")
    A("")
    A("**Precision-aware gating.** Gates were derived on the CheXpert classifier validation set only (`precision_aware_query_gates.json`): candidates are calibrated-probability quantiles of each class's frozen positives; the criterion is F0.5; a gate is adopted only if its cross-fitted (5-fold, patient-level) F0.5 gain has a bootstrap 95% lower bound above zero. A gate is not a classifier threshold: binary decisions are unchanged. Positives failing the gate are kept as low-confidence metadata and excluded from the query; if no abnormal finding survives, the normal query is used when No Finding is positive, otherwise the strongest classifier-positive finding is used and marked as a fallback.")
    A("")
    A(f"Gates were adopted for {len(gated_cls)} of 13 findings; no gate for {', '.join(no_gate)} (no stricter candidate with a reliable out-of-fold gain). Across adopted gates precision rose by {prec_gain[prec_gain.index.isin(gated_cls)].min():+.3f} to {prec_gain[prec_gain.index.isin(gated_cls)].max():+.3f} and recall fell by {rec_loss[rec_loss.index.isin(gated_cls)].min():.3f} to {rec_loss[rec_loss.index.isin(gated_cls)].max():.3f} (absolute, validation set).")
    A("")
    A("**Table 6. Retrieval gates.**")
    A("")
    A((OUT / "tables/table6_precision_aware_gates.md").read_text(encoding="utf-8"))
    A("## 8. Oracle retrieval results")
    A("")
    A(f"Primary paired set: {n_prim} queries (all three policies evaluated on the same studies). Relevance is finding agreement with the study's true finding set; Jaccard@K is the mean over the K retrieved reports; nDCG uses Jaccard as graded gain with the ideal ranking taken over the whole corpus. With oracle queries, dense retrieval reaches Jaccard@3 {cij('dense_minilm', 'oracle', 'jaccard_truth@3')} and nDCG@3 {f3(S('dense_minilm', 'oracle', 'ndcg@3'))}; lexical retrieval {cij('lexical_bm25', 'oracle', 'jaccard_truth@3')} and {f3(S('lexical_bm25', 'oracle', 'ndcg@3'))}. Both are far above the random expectation ({f3(rnd['jaccard_truth@3'])}). Dense minus lexical Jaccard@3 is {pds('dense minus lexical', 'oracle', 'jaccard_truth@3')}. Hit@3 (retrieved report has exactly the true finding set) is {f3(S('dense_minilm', 'oracle', 'hit@3'))} (dense) and {f3(S('lexical_bm25', 'oracle', 'hit@3'))} (lexical). Binary relevance is defined only as an exact finding-set match, a parameter-free criterion; Recall@K is omitted because the number of relevant reports varies from a handful to over a thousand between queries, which makes it uninformative.")
    A("")
    A("**Table 2. Dense versus lexical baseline.**")
    A("")
    A((OUT / "tables/table2_dense_vs_lexical_baseline.md").read_text(encoding="utf-8"))
    A("## 9. Frozen-classifier retrieval results")
    A("")
    A(f"With queries built from the frozen classifier, Jaccard@3 falls to {cij('dense_minilm', 'classifier_all_positive', 'jaccard_truth@3')} (dense) and {cij('lexical_bm25', 'classifier_all_positive', 'jaccard_truth@3')} (lexical), i.e. {f3(PD('dense_minilm', 'oracle minus classifier_all_positive', 'jaccard_truth@3').mean_difference)} and {f3(PD('lexical_bm25', 'oracle minus classifier_all_positive', 'jaccard_truth@3').mean_difference)} below the oracle (paired 95% intervals exclude zero). The retriever is therefore not the main bottleneck; the upstream query is. Query-faithfulness (agreement with the query's own findings) is higher than agreement with the truth (dense all-positive Jaccard@3 {f3(S('dense_minilm', 'classifier_all_positive', 'jaccard_query@3'))} against the query versus {f3(S('dense_minilm', 'classifier_all_positive', 'jaccard_truth@3'))} against the truth), consistent with retrieval following the query while the query is wrong about the study.")
    A("")
    A("**Main question: does precision-aware finding selection improve retrieval compared with passing every classifier-positive finding?** " + f"{answer}. {answer_detail} "
      f"The gate shortened queries from {cmp3.loc[('dense_minilm', 'classifier_all_positive'), 'mean_findings_per_query']:.2f} to {cmp3.loc[('dense_minilm', 'classifier_gated'), 'mean_findings_per_query']:.2f} findings on average and removed {removed_total} of {pos_total} frozen-positive finding instances across the {qs['eligible_and_classified'] and dsum['n_validation_studies_classified']} classified validation studies, but {pc(dsum['fallback_frequency'])} of studies then relied on the strongest-positive fallback. The share of queries whose finding set exactly equals the truth changed from {pc(ex_all[0])} to {pc(ex_all[1])} (eligible studies). This is a downstream systems comparison; it does not change classifier performance.")
    A("")
    A("**Table 3. Oracle versus classifier queries.**")
    A("")
    A((OUT / "tables/table3_oracle_vs_classifier_queries.md").read_text(encoding="utf-8"))
    A(f"Rare-class terms removed by gating (C2 definition): " + "; ".join(f"{l} {int(rr.loc[l, 'removed_by_gate_low_confidence'])} of {int(rr.loc[l, 'frozen_positive_studies'])}" for l in rare) + ". Sensitivity: when studies whose classifier query is empty are scored as retrieval failures the Jaccard@3 values are " + "; ".join(f"{RN[r.retriever]} {PN[r.policy]} {f3(r['jaccard_truth@3'])}" for _, r in allf.iterrows()) + f". Excluding queries whose own report text has an exact copy in the corpus ({int(dupx.n_queries.iloc[0])} of {n_prim} queries remain) lowers all values but keeps the ordering (dense oracle {f3(dupx[(dupx.retriever == 'dense_minilm') & (dupx.policy == 'oracle')]['jaccard_truth@3'].iloc[0])}, dense all-positive {f3(dupx[(dupx.retriever == 'dense_minilm') & (dupx.policy == 'classifier_all_positive')]['jaccard_truth@3'].iloc[0])}). Oracle retrieval over all {int(oall.n_queries.iloc[0])} eligible studies gives Jaccard@3 {f3(oall[oall.retriever == 'dense_minilm']['jaccard_truth@3'].iloc[0])} (dense) and {f3(oall[oall.retriever == 'lexical_bm25']['jaccard_truth@3'].iloc[0])} (lexical).")
    A("")
    A("## 10. Top-K analysis")
    A("")
    A(topk_text)
    A("")
    A(top3_verdict)
    A("")
    A("**Table 4. Performance across K.**")
    A("")
    A((OUT / "tables/table4_performance_across_k.md").read_text(encoding="utf-8"))
    A("## 11. Error analysis")
    A("")
    A(f"Using the dense retriever and the top-1 report (`retrieval_error_analysis.csv`; outcome = exact finding-set match → good, partial overlap → partial, none → poor): oracle queries give {int(eo.get('good', 0))} good / {int(eo.get('partial', 0))} partial / {int(eo.get('poor', 0))} poor; all-positive classifier queries give {int(oc.get('good', 0))} / {int(oc.get('partial', 0))} / {int(oc.get('poor', 0))}. Of the {int((e.outcome != 'good').sum())} non-good classifier-query cases, {n_cls_err} trace to a classifier query that differs from the true finding set (normal-versus-abnormal {int(sub.get('normal_vs_abnormal', 0))}, missing and extra findings {int(sub.get('missing_and_extra', 0))}, extra findings only {int(sub.get('extra_only', 0))}, missing findings only {int(sub.get('missing_only', 0))}) and {n_ret_mis} to retrieval mismatch despite a correct query; {n_map_flag} of the non-good cases involve a study with an ambiguous (unmapped) MeSH term, so the truth may be incomplete there. The failures are therefore dominated by the upstream query. These are descriptive; nothing was changed on the basis of examples.")
    A("")
    A("## 12. Domain-shift observations")
    A("")
    A(f"On {dsum['n_validation_studies_classified']} IU validation frontal images (no image failed validation) the frozen classifier predicted a mean of {dsum['mean_predicted_abnormal_findings_per_study_all']:.2f} abnormal findings per study; the mapped IU truth averages {dsum['mean_mapped_true_abnormal_findings_eligible']:.2f} on the eligible studies (predicted {dsum['mean_predicted_abnormal_findings_eligible']:.2f} on the same studies). No Finding was predicted for {pc(dsum['no_finding_positive_frequency'])} of studies against {pc(dsum['iu_indexed_normal_frequency_eligible'])} indexed normal (precision {f3(nf.precision)}, recall {f3(nf.recall)}, F1 {f3(nf.f1)}); {dsum['n_empty']} studies ({pc(dsum['empty_query_frequency'])}) produced an empty query. Largest over-prediction relative to IU truth: " + "; ".join(f"{r.finding} ({pc(r.predicted_frequency_all_frontal_studies)} predicted vs {pc(r.iu_truth_frequency_eligible_studies)})" for _, r in over.iterrows()) + ". Largest under-prediction: " + "; ".join(f"{r.finding} ({pc(r.predicted_frequency_all_frontal_studies)} vs {pc(r.iu_truth_frequency_eligible_studies)})" for _, r in under.iterrows()) + f". Enlarged Cardiomediastinum was predicted for {pc(ecm.predicted_frequency_all_frontal_studies)} of studies but has no IU mapping to compare against. Macro agreement over the 13 abnormal findings is precision {f3(dsum['descriptive_agreement_macro_over_13_abnormal']['precision'])}, recall {f3(dsum['descriptive_agreement_macro_over_13_abnormal']['recall'])}, F1 {f3(dsum['descriptive_agreement_macro_over_13_abnormal']['f1'])}; this is descriptive, the IU truth treats unmapped as absent (precision is a lower bound), and nothing was tuned.")
    A("")
    A("**Table 5. Classifier query and domain-shift summary.**")
    A("")
    A((OUT / "tables/table5_classifier_query_domain_shift.md").read_text(encoding="utf-8"))
    A("## 13. Limitations")
    A("")
    A(f"- IU truth is derived from MeSH indexing mapped to 14 classes: {ms['validation_without_eval_set_unusable_as_queries']} of {ms['validation_studies']} validation studies have no usable representation, Enlarged Cardiomediastinum cannot be evaluated, and several classes have very few IU positives (e.g. Edema {qs['primary_truth_class_frequency']['Edema']}, Consolidation {qs['primary_truth_class_frequency']['Consolidation']}, Pneumothorax {qs['primary_truth_class_frequency']['Pneumothorax']} in the primary set), so per-class conclusions are not supported.")
    A("- The primary evaluation set is dominated by normal studies (" + f"{qs['primary_truth_class_frequency']['No Finding']} of {n_prim}), for which finding agreement is easy, and templated reports are common (see the duplicate-text rate).")
    A("- Finding agreement is a proxy for usefulness as a RAG reference; it does not measure whether a retrieved report is clinically appropriate or whether a generated report would be correct.")
    A("- Query phrases are minimal finding names; no query expansion was tried, which may handicap lexical matching for terms such as support devices. Retrieval hyper-parameters were not tuned.")
    A("- IU has no patient identifier, so split independence is at study level; the same patient may appear in several studies.")
    A("- The classifier was applied to IU as an external domain; its behaviour there was not tuned and its precision is limited (see C6). Gates were derived on CheXpert, not IU.")
    A("- Confidence intervals resample queries (studies); they do not include classifier, embedding-model or mapping uncertainty.")
    A("- Only retrieval-validation queries were used; locked-test results are not available.")
    A("")
    A("## 14. R1 conclusion")
    A("")
    A(f"A leakage-safe, reproducible retrieval baseline is in place. Both retrievers recover the true findings well when given oracle finding queries (Jaccard@3 {f3(S('dense_minilm', 'oracle', 'jaccard_truth@3'))} dense, {f3(S('lexical_bm25', 'oracle', 'jaccard_truth@3'))} lexical, with no reliable difference), so retrieval itself is not the limiting component on this corpus. With queries from the frozen classifier the agreement drops to about {f3(S('dense_minilm', 'classifier_all_positive', 'jaccard_truth@3'))}–{f3(S('lexical_bm25', 'classifier_all_positive', 'jaccard_truth@3'))}, identifying the upstream finding query as the bottleneck. Precision-aware gating, derived on validation data without changing the classifier, did not give a reliable improvement over passing all frozen positives. {top3_verdict} R2 has not been started.")
    A("")
    (OUT / "RETRIEVAL_BASELINE_ANALYSIS.md").write_text("\n".join(R), encoding="utf-8")

    # ============================================================ manuscript Methods
    meth = f"""# Methods: retrieval baseline and vision-to-retrieval evaluation (R1)

## Data and leakage-safe split

Reports and images came from the locally available IU X-Ray (Open-I) collection: {inv['reports_studies']:,} studies, {inv['images_listed_in_projections_csv']:,} images ({inv['frontal_images']:,} frontal). The retrieval unit was the study, so every image of a report remained in the same partition. We reused the study-level split fixed before any retrieval work (seed {split['provenance']['seed']}; 70% reference corpus, 15% retrieval-validation queries, 15% locked test), and verified that corpus, validation and test studies are disjoint and that no study's images span two partitions. The corpus held {sz['reference_corpus_studies']:,} training studies; {sz['validation_studies']:,} validation studies served as queries. The locked retrieval-test partition was not used. Retrieval text was the cleaned Findings plus Impression of each corpus report (markup, repeated headings and de-identification placeholders removed; indication and comparison excluded); no text was rewritten.

## Finding representation

Independent of the retrieval text, each study's MeSH indexing was mapped to the 14-class vocabulary of the frozen classifier using explicit rules; ambiguous terms (for example pulmonary congestion) and terms outside the vocabulary were not mapped. A study's finding set was its mapped abnormal findings, or {{No Finding}} if indexed normal without a mapped abnormality; {ms['validation_with_eval_set']} of {ms['validation_studies']} validation studies had a usable set.

## Retrievers

Dense retrieval used the unmodified {emb['model_identifier']} encoder (embedding dimension verified as {emb['embedding_dimension_verified']}) with L2-normalised vectors in a FAISS flat inner-product index, equivalent to cosine similarity. The lexical baseline was Okapi BM25 (k1 = {lex['k1']}, b = {lex['b']}) without stop-word removal or stemming. Neither retriever was tuned. The top ten reports were retrieved per query, and we confirmed that no query study could be retrieved.

## Query sources

Three query policies were compared on the same studies. The oracle query listed the study's true mapped findings. Classifier queries were built from the frozen classifier applied, unchanged, to the first frontal image of each study (frozen preprocessing, checkpoint, Platt calibration, operating thresholds and No Finding rule), which is an external application to a different dataset. The all-positive policy used every positive finding; the precision-aware policy used only positives that passed a per-class gate derived on the CheXpert validation set (candidate gates from calibrated-probability quantiles, F0.5 criterion, adopted only when the cross-fitted gain was reliably above zero), without changing any classifier decision. If nothing remained, the fixed normal query was used when No Finding was positive; otherwise the policy fell back to the strongest positive finding, flagged as such, and studies with no output at all were recorded as empty.

## Evaluation

Relevance was judged by independent finding agreement rather than the retriever's similarity score. For each retrieved report we computed the Jaccard similarity between its finding set and the study's true set (No Finding is a single element that never overlaps abnormal findings), reported as the mean over the top K = 1, 3, 5 and 10, and graded nDCG@K using the Jaccard as gain. Hit@K and reciprocal rank were computed with an exact finding-set match as the only binary criterion. We also report the coverage of true findings by the union of the top K, the share of retrieved reports without overlap, and the share duplicating another retrieved report's text. Confidence intervals came from {jl('r1_query_sets.json')['bootstrap']['n']} bootstrap resamples of queries (seed {jl('r1_query_sets.json')['bootstrap']['seed']}), with paired differences between policies and retrievers. The primary set contained the {n_prim} studies with a usable truth, a frontal image and a non-empty classifier query.
"""
    mw = wc(meth)
    (OUT / "MANUSCRIPT_R1_METHODS.md").write_text(meth, encoding="utf-8")

    # ============================================================ manuscript Results
    gcls = ", ".join(gated_cls)
    resu = f"""# Results: retrieval baseline and vision-to-retrieval evaluation (R1)

## Dataset, split and mapping

The collection contained {inv['reports_studies']:,} studies and {inv['images_listed_in_projections_csv']:,} images. The reference corpus comprised {sz['reference_corpus_studies']:,} studies and the retrieval-validation partition {sz['validation_studies']:,}; no study was shared between partitions and no retrieved report was the query study (Table 1). Of the validation studies, {ms['validation_with_eval_set']} had a usable finding set ({ms['validation_indexed_normal']} normal), {qs['eligible_and_classified']} of them had a frontal image, and {n_prim} had a non-empty classifier query and formed the primary set. Embeddings had dimension {emb['embedding_dimension_verified']}.

## Oracle retrieval and the retriever comparison

With oracle finding queries, dense retrieval reached a Jaccard@3 of {cij('dense_minilm', 'oracle', 'jaccard_truth@3')} and nDCG@3 of {f3(S('dense_minilm', 'oracle', 'ndcg@3'))}; lexical retrieval reached {cij('lexical_bm25', 'oracle', 'jaccard_truth@3')} and {f3(S('lexical_bm25', 'oracle', 'ndcg@3'))}. Random retrieval would give about {f3(rnd['jaccard_truth@3'])}. The paired dense-minus-lexical difference was {PD('dense minus lexical', 'oracle', 'jaccard_truth@3').mean_difference:+.3f} (95% CI {PD('dense minus lexical', 'oracle', 'jaccard_truth@3').ci95_low:+.3f} to {PD('dense minus lexical', 'oracle', 'jaccard_truth@3').ci95_high:+.3f}), so the two retrievers were not reliably different (Table 2, Figure 1). Hit@3 was {f3(S('dense_minilm', 'oracle', 'hit@3'))} for dense and {f3(S('lexical_bm25', 'oracle', 'hit@3'))} for lexical retrieval.

## Frozen-classifier queries

Using queries from the frozen classifier lowered agreement substantially: Jaccard@3 was {cij('dense_minilm', 'classifier_all_positive', 'jaccard_truth@3')} (dense) and {cij('lexical_bm25', 'classifier_all_positive', 'jaccard_truth@3')} (lexical), which is {PD('dense_minilm', 'oracle minus classifier_all_positive', 'jaccard_truth@3').mean_difference:.3f} and {PD('lexical_bm25', 'oracle minus classifier_all_positive', 'jaccard_truth@3').mean_difference:.3f} below the oracle (Table 3, Figure 2). With these queries lexical retrieval was slightly better than dense retrieval (paired Jaccard@3 difference {PD('dense minus lexical', 'classifier_all_positive', 'jaccard_truth@3').mean_difference:+.3f}, 95% CI {PD('dense minus lexical', 'classifier_all_positive', 'jaccard_truth@3').ci95_low:+.3f} to {PD('dense minus lexical', 'classifier_all_positive', 'jaccard_truth@3').ci95_high:+.3f}). Retrieval followed the query: the Jaccard@3 against the query's own findings was {f3(S('dense_minilm', 'classifier_all_positive', 'jaccard_query@3'))} (dense), higher than against the truth. Among the {int((e.outcome != 'good').sum())} dense top-1 failures, {n_cls_err} arose from a classifier query that differed from the true finding set and {n_ret_mis} from retrieval despite a correct query; the commonest subtype was normal-versus-abnormal disagreement ({int(sub.get('normal_vs_abnormal', 0))}).

## Precision-aware gating

Gates were derived on CheXpert validation data for {len(gated_cls)} of 13 findings ({gcls}); {', '.join(no_gate)} received no gate. They raised class-level precision at a cost in recall (Table 6) and removed {removed_total} of {pos_total} frozen-positive finding instances from the IU validation studies, shortening queries from {cmp3.loc[('dense_minilm', 'classifier_all_positive'), 'mean_findings_per_query']:.2f} to {cmp3.loc[('dense_minilm', 'classifier_gated'), 'mean_findings_per_query']:.2f} findings. {pc(dsum['fallback_frequency'])} of studies then used the strongest-positive fallback. {answer}: Jaccard@3 differed by {pds('dense_minilm', G, 'jaccard_truth@3')} (dense) and {pds('lexical_bm25', G, 'jaccard_truth@3')} (lexical), and nDCG@3 by {pds('dense_minilm', G, 'ndcg@3')} and {pds('lexical_bm25', G, 'ndcg@3')}.{(' For lexical Jaccard@1 the gated policy was lower: ' + pds('lexical_bm25', G, 'jaccard_truth@1') + '.') if 'lexical jaccard_truth@1' in sig else ''} Precision-aware gating therefore did not improve retrieval over passing all frozen positives, and it does not alter classifier performance.

## Performance across K

Jaccard@K varied by at most {max_rng:.3f} between K = 1 and K = 10 for any policy and retriever (Table 4, Figure 3). Union coverage of the true findings with oracle dense retrieval increased from {f3(cov[('dense_minilm', 'oracle', 1)])} (K=1) to {f3(cov[('dense_minilm', 'oracle', 3)])} (K=3) and {f3(cov[('dense_minilm', 'oracle', 10)])} (K=10), whereas with all-positive classifier queries it increased from {f3(cov[('dense_minilm', 'classifier_all_positive', 1)])} to {f3(cov[('dense_minilm', 'classifier_all_positive', 3)])} and {f3(cov[('dense_minilm', 'classifier_all_positive', 10)])}. Duplicate-text rates rose with K ({pc(dup[('dense_minilm', 3)])} at K=3 and {pc(dup[('dense_minilm', 10)])} at K=10 for dense oracle retrieval). The per-query distribution of top-3 agreement was bimodal for classifier queries, with {pc(zero_c[3])} of retrieved reports sharing no finding (Figure 4).

## Classifier behaviour on IU X-Ray

The frozen classifier produced no output for {dsum['n_empty']} of {dsum['n_validation_studies_classified']} studies ({pc(dsum['empty_query_frequency'])}) and predicted No Finding for {pc(dsum['no_finding_positive_frequency'])} against {pc(dsum['iu_indexed_normal_frequency_eligible'])} indexed normal. It over-predicted, for example, {over.iloc[0].finding} ({pc(over.iloc[0].predicted_frequency_all_frontal_studies)} predicted versus {pc(over.iloc[0].iu_truth_frequency_eligible_studies)} in the mapped truth), and under-predicted {under.iloc[0].finding} ({pc(under.iloc[0].predicted_frequency_all_frontal_studies)} versus {pc(under.iloc[0].iu_truth_frequency_eligible_studies)}). Macro precision, recall and F1 over the 13 abnormal findings were {f3(dsum['descriptive_agreement_macro_over_13_abnormal']['precision'])}, {f3(dsum['descriptive_agreement_macro_over_13_abnormal']['recall'])} and {f3(dsum['descriptive_agreement_macro_over_13_abnormal']['f1'])} (descriptive only; Table 5, Figure 5).

## Interpretation

Both retrievers recover the true findings well when the query is correct, and most of the loss in the end-to-end pipeline arises before retrieval. {top3_verdict} The locked retrieval-test partition has not been evaluated.
"""
    rw = wc(resu)
    (OUT / "MANUSCRIPT_R1_RESULTS.md").write_text(resu, encoding="utf-8")
    ok_len = 450 <= mw <= 650 and 650 <= rw <= 900

    # ============================================================ captions / index
    fc = f"""# Figure captions (R1)

Retrieval-validation queries only ({n_prim} primary paired studies); the locked retrieval-test split was not used. Relevance = finding agreement with the study's true (MeSH-mapped) finding set. Classifier queries come from the frozen classifier applied to the IU frontal image without any tuning.

**Figure 1. Dense versus lexical retrieval across K.** Mean finding Jaccard@K for dense (MiniLM + FAISS, solid) and lexical (BM25, dashed) retrieval, with oracle finding queries (left) and all-positive frozen-classifier queries (right). Error bars: 95% bootstrap intervals over queries. Dotted line: expected value for random retrieval.

**Figure 2. Oracle versus classifier queries.** Jaccard@3 (left) and nDCG@3 (right) for oracle, all-positive classifier and precision-aware gated classifier queries with both retrievers; error bars are 95% bootstrap intervals. The comparison between gated and all-positive queries is a downstream query-policy comparison and does not change classifier performance.

**Figure 3. Retrieval performance versus K for the three query policies.** Finding Jaccard@K (left), union coverage of the true findings by the top K (centre) and the share of retrieved reports whose text duplicates another retrieved report (right); colours: query policy; solid, dense; dashed, lexical.

**Figure 4. Distribution of Top-3 finding agreement.** Per-query mean Jaccard of the three highest-ranked reports (dense retrieval) for each query policy, as a percentage of queries; annotations give the mean and the share of queries with zero overlap.

**Figure 5. Frozen-classifier behaviour on IU X-Ray validation studies.** Left: predicted frequency of each finding against its frequency in the mapped IU truth. Right: composition of the query outcomes (pathology findings, No Finding query, strongest-positive fallback, empty) for the all-positive and precision-aware policies. External application of the classifier; nothing was tuned on IU X-Ray.
"""
    tc = """# Table captions (R1)

**Table 1. IU X-Ray dataset and retrieval split summary.** Dataset inventory, study-level split sizes (reference corpus, retrieval-validation queries, locked test), usable query sets, split-overlap checks and embedding model.

**Table 2. Dense versus lexical baseline.** Oracle-query retrieval quality of MiniLM dense retrieval, BM25 and the random expectation on the primary paired set (finding Jaccard@K, nDCG, Hit@3, MRR@10).

**Table 3. Oracle versus classifier queries.** Retrieval quality for the oracle query, the all-positive classifier query and the precision-aware gated classifier query with both retrievers, with mean query length, findings per query, and empty and fallback query rates (over the eligible studies with classifier output; rates over all frontal studies are in Table 5 and the domain-shift summary).

**Table 4. Performance across K.** Jaccard@K, nDCG@K, Hit@K, union coverage, zero-overlap rate and duplicate-text rate at K = 1, 3, 5 and 10 for each retriever and query policy.

**Table 5. Classifier query and domain-shift summary.** Per-finding frequency in the mapped IU truth and in the frozen classifier's predictions, descriptive precision, recall and F1, retrieval gate, and positives removed by the gate.

**Table 6. Precision-aware retrieval gates.** Validation-derived gates on the calibrated probability, with precision and recall before and after gating, and the cross-fitted F0.5 gain with its bootstrap interval. Gates are query-building rules, not classifier thresholds.
"""
    (OUT / "FIGURE_CAPTIONS.md").write_text(fc, encoding="utf-8")
    (OUT / "TABLE_CAPTIONS.md").write_text(tc, encoding="utf-8")
    idx = f"""# R1 journal asset index

| Asset | Main finding | Suggested manuscript section |
|---|---|---|
| Table 1 `tables/table1_dataset_and_split_summary.*` | {sz['reference_corpus_studies']:,} / {sz['validation_studies']} / {sz['locked_test_studies']} corpus / validation / locked-test studies, zero overlaps | Methods |
| Table 2 `tables/table2_dense_vs_lexical_baseline.*` | Oracle Jaccard@3 {f3(S('dense_minilm', 'oracle', 'jaccard_truth@3'))} dense, {f3(S('lexical_bm25', 'oracle', 'jaccard_truth@3'))} lexical, random {f3(rnd['jaccard_truth@3'])} | Results |
| Table 3 `tables/table3_oracle_vs_classifier_queries.*` | Classifier queries Jaccard@3 {f3(S('dense_minilm', 'classifier_all_positive', 'jaccard_truth@3'))}-{f3(S('lexical_bm25', 'classifier_all_positive', 'jaccard_truth@3'))} versus oracle about {f3(S('dense_minilm', 'oracle', 'jaccard_truth@3'))} | Results |
| Table 4 `tables/table4_performance_across_k.*` | Jaccard flat across K; coverage rises with K | Results / Discussion |
| Table 5 `tables/table5_classifier_query_domain_shift.*` | Per-finding over- and under-prediction on IU | Results / Supplementary |
| Table 6 `tables/table6_precision_aware_gates.*` | Gates for {len(gated_cls)} of 13 findings | Methods / Supplementary |
| Figure 1 `figures/fig1_dense_vs_lexical_across_k.*` | No reliable dense-lexical difference with oracle queries | Results |
| Figure 2 `figures/fig2_oracle_vs_classifier_queries.*` | Large oracle-to-classifier drop; gating does not close it | Results |
| Figure 3 `figures/fig3_performance_vs_k.*` | Top-K behaviour of agreement, coverage and redundancy | Results / Discussion |
| Figure 4 `figures/fig4_top3_agreement_distribution.*` | Bimodal top-3 agreement for classifier queries | Results |
| Figure 5 `figures/fig5_domain_shift_query_quality.*` | Classifier domain shift and query-status composition | Results / Discussion |
| `RETRIEVAL_BASELINE_ANALYSIS.md` | Full R1 analysis | Supplementary / internal |
| `MANUSCRIPT_R1_METHODS.md`, `MANUSCRIPT_R1_RESULTS.md` | Manuscript text drafts | Methods, Results |
| `UNIVERSITY_REPORT_R1.md` | Content for Chapters 3, 4 and Results and Discussion | University report |
| `precision_aware_query_gates.json`, `classifier_query_comparison.csv` | Gate definitions; policy comparison | Methods / Results |
| `retrieval_results_top10.csv.gz`, `per_query_metrics.csv` | Full retrieval outputs and per-query metrics | Supplementary data |
| `r1_integrity_report.json` | Proof that nothing frozen changed | Supplementary |
"""
    (OUT / "R1_JOURNAL_ASSET_INDEX.md").write_text(idx, encoding="utf-8")

    # ============================================================ university report
    U = []
    A = U.append
    A("# University report content: R1 (retrieval baseline)")
    A("")
    A(f"_Insert-ready material generated from the R1 artifacts. Upstream classifier frozen at commit `{c6['commits']['C5'][:7]}` (C5) / C6 freeze manifest `{c6['created_utc']}`; R1 uses retrieval-validation data only._")
    A("")
    A("## Chapter 3 — Materials and Methods")
    A("")
    A("### 3.x IU X-Ray (Open-I) dataset")
    A("")
    A(f"The retrieval component uses the Indiana University chest X-ray collection (Open-I) in the form distributed as CSV files and normalised PNG images, held locally and treated as read-only. It contains {inv['reports_studies']:,} radiology reports, each linked to one or more images ({inv['images_listed_in_projections_csv']:,} images listed: {inv['frontal_images']:,} frontal and {inv['lateral_images']:,} lateral; {inv['studies_with_multiple_images']:,} studies have more than one image and {inv['studies_without_frontal']} have no frontal image). Each report may contain Findings and Impression sections ({inv['sections_after_cleaning']['with_both']:,} contain both, {inv['sections_after_cleaning']['with_neither']} neither), together with Indication and Comparison fields, which are metadata and are not used for retrieval. The data also include MeSH and Problems annotations ({inv['terminology_available']['MeSH']['distinct_terms']} distinct MeSH terms; {inv['terminology_available']['MeSH']['studies_indexed_normal']:,} studies indexed as normal). The files carry no patient identifier, so independence can only be guaranteed at the level of the report.")
    A("")
    A("### 3.x Report preprocessing and study-level split")
    A("")
    A(f"The retrieval unit is the report (study), never a single image, so that the frontal and lateral images of one report cannot fall in different partitions. The pre-defined split (seed {split['provenance']['seed']}; 70% / 15% / 15%, stratified on normal versus abnormal indexing) gives a reference corpus of {sz['reference_corpus_studies']:,} studies, {sz['validation_studies']:,} retrieval-validation studies used as queries during development, and {sz['locked_test_studies']:,} locked test studies that remain unused. The checks confirm no overlap between corpus, validation and test studies and no study with images in more than one partition. For retrieval, each corpus report is represented by its cleaned Findings followed by its Impression (or the single available section); markup, repeated headings and the de-identification placeholder are removed, while the original text is stored unchanged. No report text is rewritten or generated.")
    A("")
    A("### 3.x Finding representation of IU studies")
    A("")
    A(f"To evaluate retrieval independently of the text being searched, the MeSH annotations of each study were mapped to the 14 observation names of the frozen chest X-ray classifier. Direct mappings follow the project's concept lexicon; two rules depend on the MeSH qualifier (an enlarged cardiac shadow is cardiomegaly; pleural thickening is the 'pleural other' observation); ambiguous terms (for example pulmonary congestion) are kept apart and not mapped; terms outside the vocabulary are not mapped. A study's finding set is its mapped abnormal findings, or 'No Finding' for normal studies; {ms['validation_with_eval_set']} of the {ms['validation_studies']} validation studies have a usable set. No IU term corresponds to enlarged cardiomediastinum.")
    A("")
    A("## Chapter 4 — Methodology")
    A("")
    A("### 4.x Retrieval architecture")
    A("")
    A(f"The retrieval stage sits between the classifier and the (future) generation stage. Two retrievers were implemented over the reference corpus. The dense retriever encodes each report with the pre-trained all-MiniLM-L6-v2 sentence encoder (verified embedding dimension {emb['embedding_dimension_verified']}, not fine-tuned), normalises the vectors to unit length and searches a FAISS flat inner-product index, which equals cosine similarity for unit vectors. The lexical baseline is Okapi BM25 (k1 = {lex['k1']}, b = {lex['b']}) over lowercase tokens. Both return the ten highest-ranked reports; no hyper-parameter was tuned.")
    A("")
    A("### 4.x Query generation from the classifier")
    A("")
    A("The classifier output for an image is a set of independent findings. Queries are built deterministically from finding names in a fixed order. Three policies are compared: (1) an oracle query from the true mapped findings, which measures retrieval independently of vision errors; (2) an all-positive query from every finding the frozen classifier calls positive; and (3) a precision-aware query that keeps only positives passing a per-class gate on the calibrated probability. If no abnormal finding remains, a fixed normal-study query is used when 'No Finding' is positive; otherwise the strongest positive finding is used as a flagged fallback, and an output with no positive finding is recorded as an empty query. The classifier, its thresholds, its calibration and its 'No Finding' rule are not changed.")
    A("")
    A(f"The gates are derived from the classifier's validation outputs: candidate gates are quantiles of each class's positive predictions, the criterion is the precision-weighted F0.5 score, and a gate is adopted only if its cross-validated improvement is reliably above zero. Gates were adopted for {len(gated_cls)} of 13 findings and not for {', '.join(no_gate)}.")
    A("")
    A("### 4.x Evaluation protocol")
    A("")
    A("Retrieval is judged by agreement between the finding set of each retrieved report and the true finding set of the query study (Jaccard similarity, graded nDCG, and Hit@K / MRR with an exact finding-set match as the only binary criterion), not by the retriever's own similarity score. Confidence intervals use bootstrap resampling of queries.")
    A("")
    A("## Results and Discussion")
    A("")
    A("### Retrieval baseline")
    A("")
    A(f"With oracle queries both retrievers recover the true findings well (Jaccard@3 {f3(S('dense_minilm', 'oracle', 'jaccard_truth@3'))} dense and {f3(S('lexical_bm25', 'oracle', 'jaccard_truth@3'))} lexical, against {f3(rnd['jaccard_truth@3'])} for random retrieval) and the difference between them is not reliable (paired difference {PD('dense minus lexical', 'oracle', 'jaccard_truth@3').mean_difference:+.3f}, 95% interval {PD('dense minus lexical', 'oracle', 'jaccard_truth@3').ci95_low:+.3f} to {PD('dense minus lexical', 'oracle', 'jaccard_truth@3').ci95_high:+.3f}). A simple lexical baseline is therefore competitive with a dense encoder on this short, templated corpus (Table 2, Figure 1).")
    A("")
    A("### Vision-to-retrieval and query gating")
    A("")
    A(f"When the query comes from the frozen classifier applied to the IU frontal image, Jaccard@3 falls to {f3(S('dense_minilm', 'classifier_all_positive', 'jaccard_truth@3'))} (dense) and {f3(S('lexical_bm25', 'classifier_all_positive', 'jaccard_truth@3'))} (lexical); about {pc(zero_c[3])} of retrieved reports share no finding with the study. The error analysis attributes the non-good cases mainly to the classifier query (n = {n_cls_err} of {int((e.outcome != 'good').sum())}), so the bottleneck is upstream of retrieval (Table 3, Figures 2 and 4). The classifier is applied here to a different dataset without any tuning, and its behaviour shifts: it predicts 'No Finding' for {pc(dsum['no_finding_positive_frequency'])} of studies, over-predicts {over.iloc[0].finding} and under-predicts {under.iloc[0].finding} relative to the mapped truth (Table 5, Figure 5). Precision-aware gating shortened the queries and removed {removed_total} of {pos_total} positive findings, yet {answer.lower()} (Jaccard@3 difference {pds('dense_minilm', G, 'jaccard_truth@3')} dense, {pds('lexical_bm25', G, 'jaccard_truth@3')} lexical), and {pc(dsum['fallback_frequency'])} of studies needed the fallback; gating changes only the query, not classifier performance.")
    A("")
    A("### Choice of the number of references")
    A("")
    A(topk_text)
    A("")
    A(top3_verdict)
    A("")
    A("### Limitations")
    A("")
    A("The reference annotations come from mapped MeSH terms, so some studies cannot be evaluated and several findings have very few positives; finding agreement is a proxy for usefulness, not a clinical judgement; the corpus contains many near-identical normal reports; the classifier was not adapted to IU X-Ray; and only validation queries were used, with the locked retrieval-test partition reserved for later.")
    A("")
    (OUT / "UNIVERSITY_REPORT_R1.md").write_text("\n".join(U), encoding="utf-8")

    # ============================================================ wording discipline
    bad = []
    for n in ("RETRIEVAL_BASELINE_ANALYSIS.md", "MANUSCRIPT_R1_METHODS.md", "MANUSCRIPT_R1_RESULTS.md", "UNIVERSITY_REPORT_R1.md", "FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md", "R1_JOURNAL_ASSET_INDEX.md"):
        txt = (OUT / n).read_text(encoding="utf-8")
        for pat in (r"clinically (optimal|safe|ready)", r"\bproves?\b", r"superior", r"\{[a-z_]+\[[^}]*\}", r"\bnan\b", r"\bNone\b"):
            if re.search(pat, txt):
                bad.append((n, pat))
    print(json.dumps({"methods_words": mw, "results_words": rw, "length_ok": ok_len, "wording_flags": bad}))
    return 0 if ok_len and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
