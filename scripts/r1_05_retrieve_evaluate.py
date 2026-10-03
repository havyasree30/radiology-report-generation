"""R1 steps 8-16 + gating comparison: retrieve Top-10 with dense (MiniLM + FAISS) and lexical (BM25) retrievers for three
query policies (oracle, classifier all-positive, classifier precision-aware gated) over the retrieval-VALIDATION
studies, and evaluate by independent finding agreement. The locked retrieval-test studies are never opened.

Relevance of a retrieved report = finding agreement (Jaccard of the mapped IU finding sets) with the study's TRUE
finding set (primary, end-to-end) and with the QUERY's finding set (secondary, query faithfulness). Jaccard@K is the
mean Jaccard over the K retrieved reports.

    .venv\\Scripts\\python.exe -m scripts.r1_05_retrieve_evaluate
"""

from __future__ import annotations

import json
import re

import faiss
import numpy as np
import pandas as pd

from src.classification.labels import LABELS, NO_FINDING
from src.retrieval.bm25 import BM25
from src.retrieval.dense import embed, load_encoder, search
from src.retrieval.findings import ABNORMAL
from src.retrieval.metrics import KS, hit_and_rr, jaccard, ndcg_at_k
from src.retrieval.queries import NORMAL_QUERY, oracle_query
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
N_BOOT, SEED, TOPK = 1000, 42, 10
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]   # C2 definition
POLICIES = ("oracle", "classifier_all_positive", "classifier_gated")
RETRIEVERS = ("dense_minilm", "lexical_bm25")
FS = lambda s: frozenset(x for x in str(s).split(";") if x) if isinstance(s, str) and s else frozenset()  # noqa: E731


def norm_text(t: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()) if isinstance(t, str) else ""


def main() -> int:
    split = json.loads((OUT / "retrieval_split.json").read_text(encoding="utf-8"))
    corpus = pd.read_csv(OUT / "retrieval_corpus.csv", dtype={"uid": str})
    sf = pd.read_csv(OUT / "iu_study_findings.csv", dtype={"uid": str}).fillna({"eval_set": "", "mapped_findings": "", "uncertain_terms": "", "unmapped_terms": ""})
    cls = pd.read_csv(OUT / "iu_validation_classifier_outputs.csv", dtype={"uid": str}).fillna("")
    vtext = pd.read_csv(OUT / "validation_report_text_for_error_analysis.csv", dtype={"uid": str})
    ids = json.loads((OUT / "corpus_study_ids.json").read_text(encoding="utf-8"))
    assert ids == corpus["uid"].tolist()
    test_ids = set(split["study_ids"]["locked_test"])
    val = sf[sf.partition == "validation"].set_index("uid")
    assert not (set(val.index) & set(ids)) and not (set(val.index) & test_ids) and not (set(ids) & test_ids)
    doc_set = {u: FS(e) for u, e in zip(corpus.uid, corpus.eval_set)}
    doc_text = dict(zip(corpus.uid, corpus.retrieval_text))
    doc_norm = {u: norm_text(t) for u, t in doc_text.items()}
    corpus_norms = set(doc_norm.values()) - {""}
    emb = np.load(OUT / "corpus_embeddings.npy")
    row_of = {u: i for i, u in enumerate(ids)}
    index = faiss.read_index(str(OUT / "dense_minilm_flatip.faiss"))
    assert index.ntotal == len(ids)
    bm25 = BM25(ids, corpus.retrieval_text.tolist())

    # ------------------------------------------------------------------ query sets
    cl = cls.set_index("uid")
    eligible = [u for u in val.index if val.loc[u, "has_eval_set"]]                    # usable ground truth (403)
    with_cls = [u for u in eligible if u in cl.index]
    primary = [u for u in with_cls if cl.loc[u, "query_all_positive_status"] != "empty"]
    assert primary == [u for u in with_cls if cl.loc[u, "query_gated_status"] != "empty"]
    truth = {u: FS(val.loc[u, "eval_set"]) for u in eligible}
    q_text, q_set, q_status = {}, {}, {}
    for u in eligible:
        t, f = oracle_query(truth[u])
        q_text[("oracle", u)], q_set[("oracle", u)], q_status[("oracle", u)] = t, frozenset(f), "oracle"
    for u in with_cls:
        for pol, col in (("classifier_all_positive", "query_all_positive"), ("classifier_gated", "query_gated")):
            q_text[(pol, u)] = cl.loc[u, col]
            q_set[(pol, u)] = FS(cl.loc[u, col + "_findings"])
            q_status[(pol, u)] = cl.loc[u, col + "_status"]

    # ------------------------------------------------------------------ retrieval
    texts = sorted({t for t in q_text.values() if t})
    model = load_encoder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    qemb = dict(zip(texts, embed(model, texts)))
    dense_hits = dict(zip(texts, search(index, ids, np.stack([qemb[t] for t in texts]), TOPK)))

    def retrieve(retriever: str, text: str) -> list[tuple[str, float]]:
        return dense_hits[text] if retriever == "dense_minilm" else bm25.search(text, TOPK)

    all_gain = {u: np.array([jaccard(truth[u], doc_set[d]) if doc_set[d] else 0.0 for d in ids]) for u in eligible}
    exact_possible = {u: any(doc_set[d] == truth[u] for d in ids) for u in eligible}
    rows, out_rows, self_hits = [], [], 0
    for retriever in RETRIEVERS:
        for pol in POLICIES:
            for u in (eligible if pol == "oracle" else with_cls):
                text, status = q_text[(pol, u)], q_status[(pol, u)]
                base = {"retriever": retriever, "policy": pol, "uid": u, "query": text, "query_status": status,
                        "query_n_findings": len(q_set[(pol, u)]), "query_words": len(text.split()), "in_primary_set": u in primary,
                        "exact_match_possible": exact_possible[u]}
                if not text:
                    rows.append({**base, "retrieved": 0})
                    continue
                hits = retrieve(retriever, text)
                assert len(hits) == TOPK and len({d for d, _ in hits}) == TOPK
                self_hits += sum(d == u for d, _ in hits)
                dids = [d for d, _ in hits]
                g_truth = np.array([jaccard(truth[u], doc_set[d]) if doc_set[d] else 0.0 for d in dids])
                g_query = np.array([jaccard(q_set[(pol, u)], doc_set[d]) if doc_set[d] else 0.0 for d in dids])
                exact = np.array([doc_set[d] == truth[u] for d in dids])
                m = {**base, "retrieved": TOPK}
                for k in KS:
                    h, rr = hit_and_rr(exact, k)
                    sets = [doc_set[d] for d in dids[:k]]
                    union = frozenset().union(*sets)
                    norms = [doc_norm[d] for d in dids[:k]]
                    pair = np.nan
                    if k > 1:
                        e = emb[[row_of[d] for d in dids[:k]]]
                        s = e @ e.T
                        pair = float((s.sum() - np.trace(s)) / (k * (k - 1)))
                    m.update({f"jaccard_truth@{k}": float(g_truth[:k].mean()), f"jaccard_query@{k}": float(g_query[:k].mean()),
                              f"ndcg@{k}": ndcg_at_k(g_truth, all_gain[u], k), f"hit@{k}": h if exact_possible[u] else np.nan,
                              f"rr@{k}": rr if exact_possible[u] else np.nan,
                              f"union_coverage@{k}": float(len(truth[u] & union) / len(truth[u])),
                              f"zero_overlap_rate@{k}": float((g_truth[:k] == 0).mean()),
                              f"duplicate_text_rate@{k}": float(1 - len(set(norms)) / k), f"mean_pairwise_cosine@{k}": pair})
                rows.append(m)
                for rank, (d, sc) in enumerate(hits, 1):
                    out_rows.append({"retriever": retriever, "policy": pol, "query_uid": u, "query": text, "query_status": status, "rank": rank,
                                     "retrieved_uid": d, "score": sc, "retrieved_mapped_findings": ";".join(sorted(doc_set[d], key=LABELS.index)),
                                     "jaccard_vs_study_truth": g_truth[rank - 1], "jaccard_vs_query": g_query[rank - 1], "retrieved_report_text": doc_text[d]})
    pq = pd.DataFrame(rows)
    pq.to_csv(OUT / "per_query_metrics.csv", index=False, float_format="%.17g")
    res = pd.DataFrame(out_rows)
    res.to_csv(OUT / "retrieval_results_top10.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    # self-retrieval / leakage verification
    leak = {"retrieved_rows": int(len(res)), "retrieved_equal_to_query_study": int((res.retrieved_uid == res.query_uid).sum()),
            "self_hits_counted_in_loop": int(self_hits), "retrieved_ids_not_in_corpus": int((~res.retrieved_uid.isin(set(ids))).sum()),
            "retrieved_ids_in_validation": int(res.retrieved_uid.isin(set(val.index)).sum()), "retrieved_ids_in_locked_test": int(res.retrieved_uid.isin(test_ids).sum()),
            "query_ids_in_corpus": len(set(res.query_uid) & set(ids)), "query_ids_in_locked_test": len(set(res.query_uid) & test_ids),
            "corpus_validation_overlap": len(set(ids) & set(val.index)),
            "validation_queries_whose_report_text_equals_a_corpus_report": int(sum(norm_text(t) in corpus_norms for t in vtext[vtext.uid.isin(eligible)].retrieval_text)),
            }
    vt = vtext.set_index("uid").retrieval_text
    dup_pairs = res[(res["rank"] == 1)].apply(lambda r: norm_text(vt.get(r.query_uid)) != "" and norm_text(doc_text[r.retrieved_uid]) == norm_text(vt.get(r.query_uid)), axis=1)
    leak["top1_retrieved_report_text_identical_to_query_study_report"] = int(dup_pairs.sum())
    leak["status"] = "PASS" if (leak["retrieved_equal_to_query_study"] == 0 and leak["retrieved_ids_in_validation"] == 0 and leak["retrieved_ids_in_locked_test"] == 0
                                and leak["query_ids_in_corpus"] == 0 and leak["retrieved_ids_not_in_corpus"] == 0) else "FAIL"
    (OUT / "self_retrieval_check.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
    if leak["status"] != "PASS":
        raise SystemExit(f"STOP: leakage check failed {leak}")

    # ------------------------------------------------------------------ aggregation + bootstrap (queries are the unit; IU has no patient id)
    mcols = [f"{m}@{k}" for m in ("jaccard_truth", "jaccard_query", "ndcg", "hit", "union_coverage", "zero_overlap_rate", "duplicate_text_rate", "mean_pairwise_cosine") for k in KS]
    mcols += [f"rr@{k}" for k in KS]
    prim = pq[pq.in_primary_set & (pq.retrieved > 0)]

    def agg(d: pd.DataFrame, extra: dict) -> dict:
        r = {**extra, "n_queries": len(d)}
        for c in mcols:
            r[c] = float(np.nanmean(d[c])) if c in d and d[c].notna().any() else np.nan
        r["mrr@10"] = r.get("rr@10", np.nan)
        r["n_with_exact_match_possible"] = int(d["exact_match_possible"].sum())
        r["mean_query_words"] = float(d.query_words.mean())
        r["mean_findings_per_query"] = float(d.query_n_findings.mean())
        return r

    summary = pd.DataFrame([agg(prim[(prim.retriever == rt) & (prim.policy == pol)], {"retriever": rt, "policy": pol, "set": "primary_paired"})
                            for rt in RETRIEVERS for pol in POLICIES])
    summary.to_csv(OUT / "retrieval_summary_primary.csv", index=False, float_format="%.17g")
    orc_all = pq[(pq.policy == "oracle") & pq.retrieved.gt(0)]
    pd.DataFrame([agg(orc_all[orc_all.retriever == rt], {"retriever": rt, "policy": "oracle", "set": "all_eligible_validation_studies"}) for rt in RETRIEVERS]
                 ).to_csv(OUT / "retrieval_summary_oracle_all_eligible.csv", index=False, float_format="%.17g")
    # end-to-end including studies whose classifier query is empty (scored as failure = 0)
    allq = pq[pq.uid.isin(with_cls) & pq.policy.isin(POLICIES)].copy()
    for c in [c for c in mcols if c.startswith(("jaccard", "ndcg", "hit", "rr", "union"))]:
        allq.loc[allq.retrieved == 0, c] = np.where(allq.loc[allq.retrieved == 0, "exact_match_possible"] | (not c.startswith(("hit", "rr"))), 0.0, np.nan)
    pd.DataFrame([agg(allq[(allq.retriever == rt) & (allq.policy == pol)], {"retriever": rt, "policy": pol, "set": "all_frontal_studies_empty_query_scored_as_failure"})
                  for rt in RETRIEVERS for pol in POLICIES]).to_csv(OUT / "retrieval_summary_all_studies_empty_as_failure.csv", index=False, float_format="%.17g")
    # random expectation (exact, no sampling): mean over corpus documents
    rnd = {f"jaccard_truth@{k}": float(np.mean([all_gain[u].mean() for u in primary])) for k in KS}
    rnd.update({f"ndcg@{k}": float(np.nanmean([ndcg_at_k(np.full(k, all_gain[u].mean()), all_gain[u], k) for u in primary])) for k in KS})
    (OUT / "random_baseline_expectation.json").write_text(json.dumps({"description": "expected value when retrieving uniformly at random from the corpus (exact expectation for Jaccard; nDCG uses the expected gain at every rank)",
                                                                     "n_queries": len(primary), **rnd}, indent=2), encoding="utf-8")

    rng = np.random.default_rng(SEED)
    pos_u = {u: i for i, u in enumerate(primary)}
    draws = [rng.integers(0, len(primary), len(primary)) for _ in range(N_BOOT)]
    mat = {}
    for rt in RETRIEVERS:
        for pol in POLICIES:
            d = prim[(prim.retriever == rt) & (prim.policy == pol)].set_index("uid").loc[primary]
            mat[(rt, pol)] = {c: d[c].to_numpy(float) for c in ("jaccard_truth@1", "jaccard_truth@3", "jaccard_truth@5", "jaccard_truth@10", "ndcg@3", "ndcg@5", "jaccard_query@3")}

    def ci(v):
        b = np.array([np.nanmean(v[i]) for i in draws])
        return float(np.nanmean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    crow = []
    for rt in RETRIEVERS:
        for pol in POLICIES:
            for c, v in mat[(rt, pol)].items():
                m, lo, hi = ci(v)
                crow.append({"retriever": rt, "policy": pol, "metric": c, "mean": m, "ci95_low": lo, "ci95_high": hi, "n_queries": len(primary)})
    pd.DataFrame(crow).to_csv(OUT / "retrieval_bootstrap_ci.csv", index=False, float_format="%.17g")
    drow = []
    pairs = [("classifier_gated", "classifier_all_positive"), ("oracle", "classifier_all_positive"), ("oracle", "classifier_gated")]
    for rt in RETRIEVERS:
        for a, b in pairs:
            for c in ("jaccard_truth@1", "jaccard_truth@3", "jaccard_truth@5", "ndcg@3", "ndcg@5"):
                diff = mat[(rt, a)][c] - mat[(rt, b)][c]
                m, lo, hi = ci(diff)
                drow.append({"retriever": rt, "comparison": f"{a} minus {b}", "metric": c, "mean_difference": m, "ci95_low": lo, "ci95_high": hi, "n_queries": len(primary)})
    for pol in POLICIES:
        for c in ("jaccard_truth@3", "ndcg@3"):
            diff = mat[("dense_minilm", pol)][c] - mat[("lexical_bm25", pol)][c]
            m, lo, hi = ci(diff)
            drow.append({"retriever": "dense minus lexical", "comparison": pol, "metric": c, "mean_difference": m, "ci95_low": lo, "ci95_high": hi, "n_queries": len(primary)})
    pd.DataFrame(drow).to_csv(OUT / "retrieval_paired_differences.csv", index=False, float_format="%.17g")

    # sensitivity: drop queries whose own report text exactly duplicates a corpus report
    dupset = {u for u in primary if norm_text(vt.get(u)) in corpus_norms}
    sens = pd.DataFrame([agg(prim[(prim.retriever == rt) & (prim.policy == pol) & ~prim.uid.isin(dupset)], {"retriever": rt, "policy": pol, "set": "primary_excluding_exact_duplicate_report_text"})
                         for rt in RETRIEVERS for pol in POLICIES])
    sens.to_csv(OUT / "retrieval_summary_excluding_duplicate_text.csv", index=False, float_format="%.17g")

    # ------------------------------------------------------------------ classifier_query_comparison.csv (+ query statistics)
    gates = json.loads((OUT / "precision_aware_query_gates.json").read_text(encoding="utf-8"))["gates"]
    rows_c = []
    for rt in RETRIEVERS:
        for pol in POLICIES:
            r = summary[(summary.retriever == rt) & (summary.policy == pol)].iloc[0]
            qs = [(u, q_status[(pol, u)]) for u in primary]
            allcl = [q_status[(pol, u)] for u in with_cls] if pol != "oracle" else []
            rows_c.append({"retriever": rt, "query_policy": pol, "n_queries": int(r.n_queries),
                           **{f"jaccard@{k}": r[f"jaccard_truth@{k}"] for k in KS}, **{f"ndcg@{k}": r[f"ndcg@{k}"] for k in KS},
                           **{f"hit@{k}": r[f"hit@{k}"] for k in KS}, "mrr@10": r["mrr@10"],
                           **{f"jaccard_query_faithfulness@{k}": r[f"jaccard_query@{k}"] for k in KS},
                           "mean_query_words": r.mean_query_words, "mean_findings_per_query": r.mean_findings_per_query,
                           "empty_query_rate_eligible_classified_studies": (allcl.count("empty") / len(allcl)) if allcl else 0.0,
                           "no_finding_query_rate_primary": float(np.mean([s == "no_finding" or (pol == "oracle" and q_set[(pol, u)] == {NO_FINDING}) for u, s in qs])),
                           "fallback_query_rate_primary": float(np.mean([s == "fallback_strongest_positive" for _, s in qs])),
                           "fallback_query_rate_eligible_classified_studies": (allcl.count("fallback_strongest_positive") / len(allcl)) if allcl else 0.0})
    cq = pd.DataFrame(rows_c)
    cq.to_csv(OUT / "classifier_query_comparison.csv", index=False, float_format="%.17g")

    rem = []
    for l in ABNORMAL:
        pos = cl[f"{l}__pred_final"] == 1
        low = pos & (cl[f"{l}__state"] == "low_confidence")
        rem.append({"finding": l, "rare_class_c2_definition": l in RARE, "gate": gates[l]["gate_calibrated_probability"], "frozen_positive_studies": int(pos.sum()),
                    "removed_by_gate_low_confidence": int(low.sum()), "kept_high_confidence": int((pos & ~low).sum()),
                    "fraction_removed": float(low.sum() / pos.sum()) if pos.sum() else np.nan})
    rem = pd.DataFrame(rem)
    rem.to_csv(OUT / "gating_removed_terms.csv", index=False, float_format="%.17g")

    # ------------------------------------------------------------------ error analysis (dense retriever; classifier policies + oracle)
    ea = []
    dn = pq[(pq.retriever == "dense_minilm") & pq.in_primary_set & (pq.retrieved > 0)]
    for _, r in dn.iterrows():
        u, pol = r.uid, r.policy
        h = res[(res.retriever == "dense_minilm") & (res.policy == pol) & (res.query_uid == u) & (res["rank"] == 1)].iloc[0]
        j1 = h.jaccard_vs_study_truth
        cat = "good" if j1 == 1 else ("partial" if j1 > 0 else "poor")
        qs_, ts = q_set[(pol, u)], truth[u]
        if cat == "good":
            attr = "none"
        elif pol == "oracle":
            attr = "retrieval_mismatch_despite_correct_query"
        elif qs_ != ts:
            attr = "classifier_query_error"
        else:
            attr = "retrieval_mismatch_despite_correct_query"
        sub = ""
        if pol != "oracle" and qs_ != ts:
            miss, extra = ts - qs_, qs_ - ts
            sub = "normal_vs_abnormal" if (NO_FINDING in qs_) != (NO_FINDING in ts) else ("missing_and_extra" if miss and extra else "missing_only" if miss else "extra_only")
        ea.append({"policy": pol, "uid": u, "truth_findings": ";".join(sorted(ts, key=LABELS.index)), "query": r["query"], "query_findings": ";".join(sorted(qs_, key=LABELS.index)),
                   "top1_uid": h.retrieved_uid, "top1_findings": h.retrieved_mapped_findings, "top1_jaccard_vs_truth": j1, "outcome": cat, "attribution": attr,
                   "classifier_error_subtype": sub, "mapping_uncertainty_flag": bool(val.loc[u, "uncertain_terms"]),
                   "unmapped_terms_present": bool(val.loc[u, "unmapped_terms"]), "top1_report_text": h.retrieved_report_text,
                   "query_study_report_text": vt.get(u, "")})
    ea = pd.DataFrame(ea)
    ea["example_rank"] = ea.sort_values("uid", key=lambda s: s.astype(int)).groupby(["policy", "outcome"]).cumcount() + 1
    ea.to_csv(OUT / "retrieval_error_analysis.csv", index=False)

    # ------------------------------------------------------------------ classifier domain shift (descriptive; NOT a tuning experiment)
    cn = cl.loc[with_cls]
    shift = []
    for l in LABELS:
        tr = np.array([l in truth[u] for u in with_cls])
        pr = cn[f"{l}__pred_final"].to_numpy().astype(bool)
        tp, fp, fn = int((tr & pr).sum()), int((~tr & pr).sum()), int((tr & ~pr).sum())
        prec, rec = (tp / (tp + fp) if tp + fp else np.nan), (tp / (tp + fn) if tp + fn else np.nan)
        pcol = cn[f"{l}__prob"].to_numpy()
        shift.append({"finding": l, "iu_truth_positive_studies": int(tr.sum()), "predicted_positive_studies": int(pr.sum()),
                      "predicted_frequency_all_frontal_studies": float((cl[f"{l}__pred_final"] == 1).mean()), "iu_truth_frequency_eligible_studies": float(tr.mean()),
                      "precision": prec, "recall": rec, "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan, "tp": tp, "fp": fp, "fn": fn,
                      "calibrated_probability_median": float(np.median(pcol)), "calibrated_probability_p25": float(np.percentile(pcol, 25)),
                      "calibrated_probability_p75": float(np.percentile(pcol, 75)), "calibrated_probability_p95": float(np.percentile(pcol, 95))})
    shift = pd.DataFrame(shift)
    shift.to_csv(OUT / "iu_domain_shift_per_finding.csv", index=False, float_format="%.17g")
    n_pos_pred = cl[[f"{l}__pred_final" for l in ABNORMAL]].sum(1)
    n_pos_truth = np.array([len(truth[u] - {NO_FINDING}) for u in with_cls])
    dshift = {"n_validation_studies_classified": int(len(cl)), "n_eligible_with_truth": len(with_cls),
              "mean_predicted_abnormal_findings_per_study_all": float(n_pos_pred.mean()), "median": float(n_pos_pred.median()),
              "mean_predicted_abnormal_findings_eligible": float(cn[[f"{l}__pred_final" for l in ABNORMAL]].sum(1).mean()),
              "mean_mapped_true_abnormal_findings_eligible": float(n_pos_truth.mean()),
              "no_finding_positive_frequency": float((cl[f"{NO_FINDING}__pred_final"] == 1).mean()),
              "iu_indexed_normal_frequency_eligible": float(np.mean([truth[u] == {NO_FINDING} for u in with_cls])),
              "empty_query_frequency": float((cl.query_all_positive_status == "empty").mean()),
              "n_empty": int((cl.query_all_positive_status == "empty").sum()),
              "status_all_positive": cl.query_all_positive_status.value_counts().to_dict(), "status_gated": cl.query_gated_status.value_counts().to_dict(),
              "fallback_frequency": float((cl.query_gated_status == "fallback_strongest_positive").mean()),
              "queries_with_exact_truth_match_all_positive": float(np.mean([FS(cl.loc[u, "query_all_positive_findings"]) == truth[u] for u in with_cls])),
              "queries_with_exact_truth_match_gated": float(np.mean([FS(cl.loc[u, "query_gated_findings"]) == truth[u] for u in with_cls])),
              "descriptive_agreement_macro_over_13_abnormal": {m: float(np.nanmean(shift[shift.finding != NO_FINDING][m])) for m in ("precision", "recall", "f1")},
              "caveat": "IU truth = MeSH-mapped findings; 'not mapped' is not a verified negative, so precision is a lower bound; descriptive only, nothing was tuned"}
    (OUT / "iu_domain_shift_summary.json").write_text(json.dumps(dshift, indent=2, allow_nan=False, default=float), encoding="utf-8")
    pc_all = pd.DataFrame({l: cl[f"{l}__prob"] for l in LABELS})
    pc_all.to_csv(OUT / "iu_calibrated_probabilities_validation_studies.csv", index=False, float_format="%.6g")

    set_info = {"validation_studies": int(len(val)), "with_usable_truth_eval_set": len(eligible), "with_frontal_image_and_classifier_output": int(len(cl)),
                "eligible_and_classified": len(with_cls), "primary_paired_set_non_empty_classifier_query": len(primary),
                "eligible_but_empty_classifier_query": len(with_cls) - len(primary), "exact_match_possible_in_primary": int(sum(exact_possible[u] for u in primary)),
                "primary_queries_with_exact_duplicate_report_text_in_corpus": len(dupset), "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "query (study)"},
                "primary_truth_class_frequency": {l: int(sum(l in truth[u] for u in primary)) for l in LABELS}}
    (OUT / "r1_query_sets.json").write_text(json.dumps(set_info, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps(set_info, indent=1)[:900])
    print(cq[["retriever", "query_policy", "n_queries", "jaccard@1", "jaccard@3", "ndcg@3", "ndcg@5", "hit@3", "mean_findings_per_query", "fallback_query_rate_primary"]].round(3).to_string())
    print(leak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
