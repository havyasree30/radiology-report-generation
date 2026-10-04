"""R2 failure analysis (categories A-F) for the selected pipeline versus the R1 Q0 dense baseline, and the retrieval
CANDIDATE configuration (not final: the locked retrieval test remains unopened).

Failure definitions (both reported):  top-1 failure = top-1 report is not an exact finding-set match (R1 convention);
coverage failure = the union of the selected top-K reports misses at least one true finding.
Exclusive categories (priority order):
  D normal/abnormal disagreement   the query is normal while the truth is abnormal, or vice versa
  A upstream classification/query error   the query finding set differs from the truth (missing and/or extra findings)
  E corpus limitation              query == truth but no corpus report has exactly the true finding set
  F duplicate/template issue       query == truth, exact match exists, and the top-1 report text is repeated within the top-K
  B terminology mismatch           query == truth, none of the above, and the query study or the top-1 report has an ambiguous
                                   (unmapped) MeSH term, so the finding vocabularies may not align
  C retriever failure despite a good query   everything else
Non-exclusive flags: any duplicate text in the top-K; ambiguous terms in the query study.

    .venv\\Scripts\\python.exe -m scripts.r2_03_error_analysis_and_config
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.classification.final_test import sha256_file
from src.classification.labels import NO_FINDING
from src.retrieval.r2_context import R1, R2, R2Context, FS
from scripts.r2_02_experiments import Engine

CATS = {"D": "Normal/abnormal disagreement", "A": "Upstream classification/query error", "E": "Corpus limitation", "F": "Duplicate/template issue",
        "B": "Terminology mismatch", "C": "Retriever failure despite good query", "none": "No failure"}


def categorize(ctx: R2Context, uid: str, qset: frozenset, ranked: list[str], k: int, failed: bool, doc_unc: dict) -> str:
    if not failed:
        return "none"
    truth = ctx.truth[uid]
    if (NO_FINDING in qset) != (NO_FINDING in truth):
        return "D"
    if qset != truth:
        return "A"
    if not ctx.exact_possible[uid]:
        return "E"
    norms = [ctx.doc_norm[d] for d in ranked[:k]]
    if norms.count(norms[0]) > 1:
        return "F"
    if ctx.val.loc[uid, "uncertain_terms"] or doc_unc.get(ranked[0]):
        return "B"
    return "C"


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def main() -> int:
    dec = json.loads((R2 / "r2_stage_decisions.json").read_text(encoding="utf-8"))
    K = int(dec["selected_K"])
    eng = Engine()
    ctx = eng.ctx
    sf = pd.read_csv(R1 / "iu_study_findings.csv", dtype={"uid": str}).fillna("")
    doc_unc = {r.uid: r.uncertain_terms for r in sf[sf.partition == "reference_corpus"].itertuples()}
    top = pd.read_csv(R2 / "retrieval_top10_all_configs.csv.gz", dtype={"uid": str, "retrieved_uid": str})
    sel = dec["selected_query_policy"]
    spec_final = {"phrases": sel["phrases"], "normal": sel["normal"], "top_n": sel["top_n"], "weighting": sel["weighting"], "oracle": False}
    spec_base = {"phrases": "names", "normal": "no acute abnormality", "oracle": False}
    rows = []
    for tag, cfg, spec in (("R1_Q0_dense", "R1_Q0_dense", spec_base), ("R2_final", "FINAL_pipeline", spec_final)):
        g = top[top.config == cfg].sort_values(["uid", "rank"])
        lists = g.groupby("uid").retrieved_uid.apply(list)
        pm = pd.read_csv(R2 / "per_query_metrics_all_configs.csv", dtype={"uid": str})
        pm = pm[pm.config == cfg].set_index("uid")
        for u in ctx.primary:
            q = eng.query(u, spec)
            ranked = lists[u]
            top1_exact = ctx.doc_set[ranked[0]] == ctx.truth[u]
            cov_fail = pm.loc[u, f"union_coverage@{K}"] < 1
            dup_any = pm.loc[u, f"duplicate_text_rate@{K}"] > 0
            row = {"system": tag, "uid": u, "truth": ";".join(sorted(ctx.truth[u])), "query": q["text"], "query_findings": ";".join(sorted(q["qset"])),
                   "top1_uid": ranked[0], "top1_findings": ";".join(sorted(ctx.doc_set[ranked[0]])), "top1_exact": bool(top1_exact), "coverage_fail_at_K": bool(cov_fail),
                   "duplicate_text_in_topK": bool(dup_any), "query_study_has_ambiguous_terms": bool(ctx.val.loc[u, "uncertain_terms"]), "K": K,
                   "exact_match_possible": bool(ctx.exact_possible[u]), "query_equals_truth": bool(q["qset"] == ctx.truth[u])}
            row["category_top1_failure"] = categorize(ctx, u, q["qset"], ranked, K, not top1_exact, doc_unc)
            row["category_coverage_failure"] = categorize(ctx, u, q["qset"], ranked, K, bool(cov_fail), doc_unc)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(R2 / "r2_error_analysis.csv", index=False)
    summ = []
    for sysn, g in df.groupby("system"):
        for definition, col, fcol in (("top-1 not an exact finding-set match", "category_top1_failure", "top1_exact"), (f"true findings not fully covered by top-{K}", "category_coverage_failure", None)):
            fails = g[g[col] != "none"]
            for cat in ("D", "A", "E", "F", "B", "C"):
                n = int((fails[col] == cat).sum())
                summ.append({"system": sysn, "failure_definition": definition, "category": cat, "category_name": CATS[cat], "n": n,
                             "pct_of_failures": 100 * n / max(len(fails), 1), "pct_of_all_queries": 100 * n / len(g), "n_failures": len(fails), "n_queries": len(g)})
    pd.DataFrame(summ).to_csv(R2 / "r2_failure_categories.csv", index=False, float_format="%.17g")
    flags = {sysn: {"duplicate_text_in_topK": float(g.duplicate_text_in_topK.mean()), "query_study_has_ambiguous_terms": float(g.query_study_has_ambiguous_terms.mean())} for sysn, g in df.groupby("system")}
    (R2 / "r2_failure_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- candidate configuration
    emb = json.loads((R1 / "embedding_metadata.json").read_text(encoding="utf-8"))
    lex = json.loads((R1 / "lexical_index_metadata.json").read_text(encoding="utf-8"))
    exp = json.loads((R2 / "finding_query_expansion.json").read_text(encoding="utf-8"))
    prot = json.loads((R2 / "r2_selection_protocol.json").read_text(encoding="utf-8"))
    summ_cfg = pd.read_csv(R2 / "r2_summary_all_configs.csv").set_index("config")
    f = summ_cfg.loc["FINAL_pipeline"]
    o = summ_cfg.loc["O_Q0_dense"]
    cfg = {
        "status": "CANDIDATE - not the frozen retrieval system; the locked retrieval-test split remains unopened and unevaluated",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_branch": git("branch", "--show-current"), "git_head_at_creation": git("rev-parse", "HEAD"),
        "selection_protocol_sha256": sha256_file(R2 / "r2_selection_protocol.json"), "selection_decisions": dec["decisions"],
        "evaluation_population": f"R1 primary paired validation set, {len(ctx.primary)} studies",
        "embedding_model": {"identifier": emb["model_identifier"], "snapshot_revision": emb["model_snapshot_revision"], "dimension": emb["embedding_dimension_verified"],
                            "normalised": True, "fine_tuned": False, "weights_sha256": emb["model_weights_sha256"]},
        "query_construction": {"policy": "Q0_finding_names + Q3_topN" if sel["top_n"] else "Q0_finding_names", "phrase_set": sel["phrases"],
                               "top_n_findings": sel["top_n"], "top_n_order": "calibrated probability descending, ties by class order; never pads",
                               "probability_weighting": sel["weighting"], "weighting_selected": sel["weighting"] is not None,
                               "clinical_phrase_expansion_selected": sel["phrases"] == "expanded",
                               "clinical_expansion_mapping_file": "finding_query_expansion.json", "clinical_expansion_mapping_sha256": sha256_file(R2 / "finding_query_expansion.json"),
                               "clinical_expansion_mapping": exp["expansions"], "finding_name_phrases": "src/retrieval/queries.py PHRASE",
                               "no_finding_query_phrase": sel["normal"], "normal_phrase_candidates_evaluated": exp["normal_phrase_candidates"],
                               "empty_query_policy": "recorded as empty; no finding invented"},
        "lexical": {"method": lex["method"], "k1": lex["k1"], "b": lex["b"], "tokenizer": lex["tokenizer"], "tie_break": lex["tie_break"]},
        "retriever": {"selected": dec["selected_retriever"], "hybrid": {"method": "Reciprocal Rank Fusion", "rrf_k": 60, "depth_per_ranker": 100, "rankers": ["dense_minilm", "bm25"], "tie_break": "corpus order",
                                                                             "provenance_saved": "dense_rank, bm25_rank, rrf_score per retrieved report (final_pipeline_top10.csv.gz)"},
                      "dense_tie_break": "score descending then corpus order (independent of FAISS depth)"},
        "diversity_reranking": {"mmr_selected": dec["mmr"]["lambda"] is not None, "mmr_lambda": dec["mmr"]["lambda"], "mmr_pool": 30, "lambdas_evaluated": prot["stage_C"]["lambdas"],
                                "reason": "not adopted: no lambda reliably reduced the exact-duplicate rate at both K=3 and K=5 (see r2_mmr_adoption_checks.csv)" if dec["mmr"]["lambda"] is None else "adopted"},
        "provisional_top_k": K, "top_k_rule": prot["k_selection"],
        "validation_performance_of_candidate": {"jaccard@K": float(f[f"jaccard_truth@{K}"]), "ndcg@K": float(f[f"ndcg@{K}"]), "union_coverage@K": float(f[f"union_coverage@{K}"]),
                                                "duplicate_text_rate@K": float(f[f"duplicate_text_rate@{K}"]), "jaccard@3": float(f["jaccard_truth@3"]),
                                                "oracle_dense_q0_jaccard@K_for_reference": float(o[f"jaccard_truth@{K}"])},
        "classifier": "frozen (C2-B checkpoint, C4 thresholds and No Finding rule, C5 Platt calibration); unchanged",
        "caveats": ["selected and evaluated on the same retrieval-validation queries (the locked test is unopened); paired intervals do not account for selection",
                    "differences between several challengers were small; the Top-3 query rule gained +0.003 Jaccard@3 and is not practically meaningful on its own"]}
    (R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").write_text(json.dumps(cfg, indent=2, allow_nan=False), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summ).pipe(lambda d: d[d.failure_definition.str.startswith("top-1")]).pivot(index="category", columns="system", values="n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
