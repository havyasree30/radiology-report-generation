"""R2 staged experiments on the R1 primary validation set (358 studies); locked retrieval test never opened.

Follows r2_selection_protocol.json (written before any evaluation):
  normal phrase -> Stage A (query representation) -> Stage B (retriever) -> Stage C (MMR) -> Top-K selection.

    .venv\\Scripts\\python.exe -m scripts.r2_02_experiments
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.classification.final_test import sha256_file
from src.classification.labels import NO_FINDING
from src.retrieval.diversity import mmr_rerank
from src.retrieval.evaluation import bootstrap_draws, evaluate_ranking, mean_ci, paired_ci
from src.retrieval.expansion import finding_weights, load_expansion, phrase_text, select_top_n, weighted_query_vector
from src.retrieval.fusion import RRF_K, rrf_fuse
from src.retrieval.metrics import KS
from src.retrieval.queries import ABNORMAL, NORMAL_QUERY, PHRASE
from src.retrieval.r2_context import R1, R2, FS, R2Context

log = logging.getLogger("r2")
PRIMARY_METRIC = "jaccard_truth@3"
DEPTH, POOL = 100, 30
Q0_PHRASES = dict(PHRASE)


class Engine:
    def __init__(self):
        self.ctx = R2Context()
        self.exp = load_expansion(R2 / "finding_query_expansion.json")
        self.protocol = json.loads((R2 / "r2_selection_protocol.json").read_text(encoding="utf-8"))
        assert self.protocol["expansion_file_sha256"] == sha256_file(R2 / "finding_query_expansion.json"), "expansion mapping changed after the protocol was written"
        self.draws = bootstrap_draws(len(self.ctx.primary), 1000, 42)
        self.rows: list[dict] = []
        self.top10: list[dict] = []
        self.q_stats: dict[str, dict] = {}
        self.mat: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ queries
    def query(self, uid: str, spec: dict) -> dict | None:
        c = self.ctx
        phrases = Q0_PHRASES if spec["phrases"] == "names" else self.exp["expansions"]
        normal = spec["normal"]
        if spec.get("oracle"):
            findings = [l for l in ABNORMAL if l in c.truth[uid]]
            probs = None
            nf = c.truth[uid] == {NO_FINDING}
        else:
            findings, probs, nf = c.positives(uid), c.probs(uid), c.no_finding_positive(uid)
            findings = select_top_n(findings, probs, spec.get("top_n"))
        if findings:
            text = phrase_text(findings, phrases, normal)
            if spec.get("weighting") and probs is not None:
                w = finding_weights(spec["weighting"], findings, probs, c.cal_thr)
                vec = weighted_query_vector({l: c.embed_text(phrases[l]) for l in findings}, w)
            else:
                vec = c.embed_text(text)
            return {"text": text, "vec": vec, "qset": frozenset(findings), "status": "findings"}
        if nf:
            return {"text": normal, "vec": c.embed_text(normal), "qset": frozenset({NO_FINDING}), "status": "no_finding"}
        return None

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, q: dict, retriever: str, mmr_lambda: float | None = None, bm25_text: str | None = None) -> tuple[list[str], dict]:
        c = self.ctx
        d_ids, d_sc = c.dense_ranking(q["vec"], DEPTH)
        prov: dict = {}
        if retriever == "dense":
            ranked, rel = d_ids, dict(zip(d_ids, d_sc))
        elif retriever == "bm25":
            ranked, sc = c.bm25_ranking(bm25_text or q["text"], DEPTH)
            rel = dict(zip(ranked, sc))
        elif retriever == "hybrid":
            b_ids, _ = c.bm25_ranking(bm25_text or q["text"], DEPTH)
            fused = rrf_fuse({"dense": d_ids, "bm25": b_ids}, c.corpus_index, k=RRF_K, depth=DEPTH)
            ranked, rel = [f["doc"] for f in fused], {f["doc"]: f["rrf_score"] for f in fused}
            prov = {f["doc"]: (f["dense_rank"], f["bm25_rank"], f["rrf_score"]) for f in fused[:POOL]}
        else:
            raise ValueError(retriever)
        if mmr_lambda is not None:
            pool = ranked[:POOL]
            ranked = mmr_rerank(pool, np.array([rel[d] for d in pool]), c.emb[[c.corpus_index[d] for d in pool]], mmr_lambda, 10)
        return ranked[:10], {"rel": rel, "prov": prov}

    def run(self, name: str, spec: dict, retriever: str = "dense", mmr_lambda: float | None = None, oracle: bool = False, population: list[str] | None = None,
            keep_text: bool = False, bm25_spec: dict | None = None, store: bool = True) -> pd.DataFrame:
        c = self.ctx
        rows, words, nfind, changed = [], [], [], []
        for u in population or c.primary:
            q = self.query(u, {**spec, "oracle": oracle})
            bt = self.query(u, {**bm25_spec, "oracle": oracle})["text"] if bm25_spec else None
            ranked, info = self.retrieve(q, retriever, mmr_lambda, bt)
            m = evaluate_ranking(ranked, c.truth[u], q["qset"], c.doc_set, c.doc_norm, c.doc_words, c.emb, c.corpus_index, c.all_gain[u], c.exact_possible[u])
            rows.append({"config": name, "uid": u, **m})
            words.append(len((bt or q["text"]).split()))
            nfind.append(len(q["qset"]))
            if store:
                for r, d in enumerate(ranked, 1):
                    p = info["prov"].get(d, (None, None, None))
                    self.top10.append({"config": name, "uid": u, "rank": r, "retrieved_uid": d, "score": info["rel"].get(d), "dense_rank": p[0], "bm25_rank": p[1], "rrf_score": p[2],
                                       "query": q["text"], "retrieved_report_text": c.doc_text[d] if keep_text else None})
        df = pd.DataFrame(rows)
        self.mat[name] = df.set_index("uid")
        self.q_stats[name] = {"mean_query_words": float(np.mean(words)), "mean_findings_per_query": float(np.mean(nfind)), "n_queries": len(df)}
        if store:
            self.rows += rows
        return df

    def arr(self, name: str, metric: str) -> np.ndarray:
        return self.mat[name].loc[self.ctx.primary, metric].to_numpy(float)

    def challenge(self, incumbent: str, challenger: str, metric: str = PRIMARY_METRIC) -> dict:
        r = paired_ci(self.arr(challenger, metric), self.arr(incumbent, metric), self.draws)
        r.update(incumbent=incumbent, challenger=challenger, metric=metric, replaces=bool(r["ci95_low"] > 0),
                 ndcg3_same_sign=bool(np.sign(np.nanmean(self.arr(challenger, "ndcg@3") - self.arr(incumbent, "ndcg@3"))) == np.sign(r["mean_difference"])))
        return r


def pick(eng: Engine, incumbent: str, challengers: list[str], log_: list[dict], stage: str) -> str:
    """Protocol replacement rule: among challengers whose paired CI lower bound > 0 vs the incumbent pick the highest mean."""
    res = [eng.challenge(incumbent, ch) for ch in challengers]
    for r in res:
        log_.append({"stage": stage, **r})
    ok = [r for r in res if r["replaces"]]
    return max(ok, key=lambda r: eng.arr(r["challenger"], PRIMARY_METRIC).mean())["challenger"] if ok else incumbent


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    eng = Engine()
    c = eng.ctx
    decisions: list[dict] = []
    # ---------------------------------------------------------------- normal-query phrase (oracle-normal studies, dense)
    normal_pop = [u for u in c.eligible if c.truth[u] == {NO_FINDING}]
    cand = eng.exp["normal_phrase_candidates"]
    nres = {}
    for ph in cand:
        d = eng.run(f"N_{ph}", {"phrases": "names", "normal": ph}, population=normal_pop, oracle=True, store=False)
        nres[ph] = {f"{m}@{k}": float(d[f"{m}@{k}"].mean()) for m in ("jaccard_truth", "ndcg") for k in (3, 5)}
    nd = bootstrap_draws(len(normal_pop), 1000, 42)
    base = NORMAL_QUERY
    chosen, nrows = base, []
    for ph in cand:
        a, b = eng.mat[f"N_{ph}"].loc[normal_pop, PRIMARY_METRIC].to_numpy(float), eng.mat[f"N_{base}"].loc[normal_pop, PRIMARY_METRIC].to_numpy(float)
        ci = paired_ci(a, b, nd)
        nrows.append({"phrase": ph, **nres[ph], **{f"vs_r1_phrase_{k}": v for k, v in ci.items()}})
    best = max([r for r in nrows if r["phrase"] != base and r["vs_r1_phrase_ci95_low"] > 0], key=lambda r: r["jaccard_truth@3"], default=None)
    chosen = best["phrase"] if best else base
    (R2 / "normal_query_selection.json").write_text(json.dumps({
        "population": f"{len(normal_pop)} eligible validation studies whose true finding set is {{No Finding}} (oracle-normal; dense retrieval)", "candidates": nrows,
        "chosen": chosen, "r1_phrase": base,
        "note": "every normal query uses one fixed phrase, so all studies share one retrieved list; the paired interval is degenerate and the rule reduces to 'strictly better than the R1 phrase'. Selection used retrieval-validation data only."}, indent=2), encoding="utf-8")
    log.info("normal phrase: %s", chosen)

    # ---------------------------------------------------------------- baselines (R1 definitions, R2 tie rule)
    names = {"phrases": "names", "normal": NORMAL_QUERY}
    eng.run("R1_Q0_dense", names, "dense")
    eng.run("R1_Q0_bm25", names, "bm25")
    # R1 gated query reconstructed from the stored R1 gated texts / findings
    for rt, tag in (("dense", "dense"), ("bm25", "bm25")):
        rows = []
        for u in c.primary:
            findings = FS(c.cls.loc[u, "query_gated_findings"])
            text = c.cls.loc[u, "query_gated"]
            q = {"text": text, "vec": c.embed_text(text), "qset": findings}
            ranked, info = eng.retrieve(q, rt)
            m = evaluate_ranking(ranked, c.truth[u], findings, c.doc_set, c.doc_norm, c.doc_words, c.emb, c.corpus_index, c.all_gain[u], c.exact_possible[u])
            rows.append({"config": f"R1_gated_{rt}", "uid": u, **m})
            for r, d in enumerate(ranked, 1):
                eng.top10.append({"config": f"R1_gated_{rt}", "uid": u, "rank": r, "retrieved_uid": d, "score": info["rel"].get(d), "query": text})
        eng.rows += rows
        eng.mat[f"R1_gated_{rt}"] = pd.DataFrame(rows).set_index("uid")
        eng.q_stats[f"R1_gated_{rt}"] = {"mean_query_words": float(np.mean([len(c.cls.loc[u, 'query_gated'].split()) for u in c.primary])), "mean_findings_per_query": float(np.mean([len(FS(c.cls.loc[u, 'query_gated_findings'])) for u in c.primary])), "n_queries": len(c.primary)}
    eng.run("O_Q0_dense", names, "dense", oracle=True)
    eng.run("O_Q0_bm25", names, "bm25", oracle=True)

    # ---------------------------------------------------------------- Stage A
    log_: list[dict] = []
    exp_spec = {"phrases": "expanded", "normal": chosen}
    eng.run("A1_Q1_expanded_dense", exp_spec, "dense")
    base_name = pick(eng, "R1_Q0_dense", ["A1_Q1_expanded_dense"], log_, "A1_phrases")
    base_phr = "expanded" if base_name == "A1_Q1_expanded_dense" else "names"
    base_spec = exp_spec if base_phr == "expanded" else names
    decisions.append({"stage": "A1", "winner": base_name, "base_phrases": base_phr})
    w_names = []
    for w in ("uniform", "probability", "margin"):
        n = f"A2_Q2_{w}_dense"
        eng.run(n, {**base_spec, "weighting": w}, "dense")
        w_names.append(n)
    rep_name = pick(eng, base_name, w_names, log_, "A2_weighting")
    rep_spec = {**base_spec, "weighting": rep_name.split("_")[2]} if rep_name in w_names else base_spec
    decisions.append({"stage": "A2", "winner": rep_name, "weighting": rep_spec.get("weighting")})
    # share of frozen positives with p < 0.5 (why max(p-0.5,0) is rejected by design)
    allp = [c.probs(u)[l] for u in c.primary for l in c.positives(u)]
    decisions.append({"stage": "A2_note", "share_of_frozen_positive_findings_with_p_below_0.5": float(np.mean(np.array(allp) < 0.5)), "n_positive_findings": len(allp)})
    tn_names = []
    for n_ in (1, 2, 3):
        n = f"A3_Q3_top{n_}_dense"
        eng.run(n, {**rep_spec, "top_n": n_}, "dense")
        tn_names.append(n)
    q_name = pick(eng, rep_name, tn_names, log_, "A3_topN")
    q_spec = {**rep_spec, "top_n": int(q_name.split("top")[1].split("_")[0])} if q_name in tn_names else rep_spec
    decisions.append({"stage": "A3", "winner": q_name, "top_n": q_spec.get("top_n")})
    sel_q_desc = {"phrases": q_spec["phrases"], "weighting": q_spec.get("weighting"), "top_n": q_spec.get("top_n"), "normal": chosen}
    # text spec for BM25 (weights apply only to the dense vector)
    bm_spec = {k: v for k, v in q_spec.items() if k != "weighting"}

    # ---------------------------------------------------------------- Stage B
    eng.run("B_dense", q_spec, "dense")
    eng.run("B_bm25", q_spec, "bm25", bm25_spec=bm_spec)
    eng.run("B_hybrid_rrf", q_spec, "hybrid", bm25_spec=bm_spec)
    ret_name = pick(eng, "B_dense", ["B_bm25", "B_hybrid_rrf"], log_, "B_retriever")
    ret = {"B_dense": "dense", "B_bm25": "bm25", "B_hybrid_rrf": "hybrid"}[ret_name]
    decisions.append({"stage": "B", "winner": ret_name, "retriever": ret})

    # ---------------------------------------------------------------- Stage C (MMR on dense or hybrid)
    mmr_base = ret if ret in ("dense", "hybrid") else "dense"
    mmr_base_name = {"dense": "B_dense", "hybrid": "B_hybrid_rrf"}[mmr_base]
    eligible_l, c_rows = [], []
    for lam in eng.protocol["stage_C"]["lambdas"]:
        n = f"C_{mmr_base}_mmr{lam}"
        eng.run(n, q_spec, mmr_base, mmr_lambda=lam, bm25_spec=bm_spec if mmr_base == "hybrid" else None)
        ok = True
        for k in (3, 5):
            dup = paired_ci(eng.arr(n, f"duplicate_text_rate@{k}"), eng.arr(mmr_base_name, f"duplicate_text_rate@{k}"), eng.draws)
            jac = paired_ci(eng.arr(n, f"jaccard_truth@{k}"), eng.arr(mmr_base_name, f"jaccard_truth@{k}"), eng.draws)
            ndc = paired_ci(eng.arr(n, f"ndcg@{k}"), eng.arr(mmr_base_name, f"ndcg@{k}"), eng.draws)
            cov = paired_ci(eng.arr(n, f"union_coverage@{k}"), eng.arr(mmr_base_name, f"union_coverage@{k}"), eng.draws)
            cond = {"dup_reliably_lower": dup["ci95_high"] < 0, "jaccard_no_reliable_loss": not (jac["ci95_high"] < 0), "ndcg_no_reliable_loss": not (ndc["ci95_high"] < 0), "coverage_point_not_lower": cov["mean_difference"] >= 0}
            ok &= all(cond.values())
            c_rows.append({"lambda": lam, "K": k, **{f"dup_{a}": b for a, b in dup.items()}, **{f"jaccard_{a}": b for a, b in jac.items()}, **{f"ndcg_{a}": b for a, b in ndc.items()},
                           **{f"coverage_{a}": b for a, b in cov.items()}, **cond})
        if ok:
            eligible_l.append(lam)
    pd.DataFrame(c_rows).to_csv(R2 / "r2_mmr_adoption_checks.csv", index=False, float_format="%.17g")
    lam_sel = max(eligible_l) if eligible_l else None
    final_name = f"C_{mmr_base}_mmr{lam_sel}" if lam_sel is not None else mmr_base_name
    decisions.append({"stage": "C", "mmr_base": mmr_base, "eligible_lambdas": eligible_l, "selected_lambda": lam_sel, "final_pipeline_config": final_name})

    # ---------------------------------------------------------------- oracle diagnostic with the selected representation / pipeline
    o_spec = {"phrases": q_spec["phrases"], "normal": chosen}
    o_bm = o_spec
    eng.run("O_final_representation_dense", o_spec, "dense", oracle=True)
    eng.run("O_final_pipeline", o_spec, ret if lam_sel is None else mmr_base, mmr_lambda=lam_sel, oracle=True, bm25_spec=o_bm if (ret == "hybrid" or (lam_sel is not None and mmr_base == "hybrid")) else None, keep_text=True)
    # final classifier pipeline with texts for error analysis
    eng.run("FINAL_pipeline", q_spec, ret if lam_sel is None else mmr_base, mmr_lambda=lam_sel, bm25_spec=bm_spec if (ret == "hybrid" or (lam_sel is not None and mmr_base == "hybrid")) else None, keep_text=True)

    # ---------------------------------------------------------------- Top-K selection on the final classifier pipeline
    k_log, cur = [], 1
    cfg = eng.protocol["k_selection"]
    while True:
        passed = []
        for kn in [k for k in KS if k > cur]:
            cov = paired_ci(eng.arr("FINAL_pipeline", f"union_coverage@{kn}"), eng.arr("FINAL_pipeline", f"union_coverage@{cur}"), eng.draws)
            jac_drop = float(np.nanmean(eng.arr("FINAL_pipeline", f"jaccard_truth@{cur}")) - np.nanmean(eng.arr("FINAL_pipeline", f"jaccard_truth@{kn}")))
            dup = float(np.nanmean(eng.arr("FINAL_pipeline", f"duplicate_text_rate@{kn}")))
            uq = float(np.nanmean(eng.arr("FINAL_pipeline", f"unique_templates@{kn}")) - np.nanmean(eng.arr("FINAL_pipeline", f"unique_templates@{cur}")))
            ctxw = float(np.nanmean(eng.arr("FINAL_pipeline", f"context_words@{kn}")))
            cond = {"coverage_gain_ci_lower_gt_0": cov["ci95_low"] > 0, "coverage_gain_ge_0.05": cov["mean_difference"] >= 0.05, "jaccard_fall_le_0.03": jac_drop <= 0.03,
                    "duplicate_rate_le_0.50": dup <= 0.50, "unique_templates_increase": uq > 0, "context_le_1000_words": ctxw <= 1000}
            k_log.append({"from_K": cur, "to_K": kn, "coverage_gain": cov["mean_difference"], "coverage_ci_low": cov["ci95_low"], "coverage_ci_high": cov["ci95_high"], "jaccard_fall": jac_drop,
                          "duplicate_rate_at_to_K": dup, "unique_templates_added": uq, "context_words_at_to_K": ctxw, **cond, "advance": all(cond.values())})
            if all(cond.values()):
                passed.append(kn)
        if not passed:
            break
        cur = min(passed)
    decisions.append({"stage": "K", "selected_K": cur})
    pd.DataFrame(k_log).to_csv(R2 / "r2_k_selection_log.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- paired comparison table (protocol section 15)
    pc_rows = []

    def pcomp(group: str, a: str, b: str, metrics: tuple, ks: tuple, label: str = "") -> None:
        for m in metrics:
            for k in ks:
                col = f"{m}@{k}"
                r = paired_ci(eng.arr(a, col), eng.arr(b, col), eng.draws)
                pc_rows.append({"group": group, "comparison": label or f"{a} minus {b}", "a": a, "b": b, "metric": col, **r, "n_queries": len(c.primary), "ci_excludes_zero": bool(r["ci95_low"] > 0 or r["ci95_high"] < 0)})

    core, div = ("jaccard_truth", "ndcg", "union_coverage"), ("duplicate_text_rate", "unique_templates", "mean_pairwise_cosine", "zero_overlap_rate")
    for ch in ("A1_Q1_expanded_dense", "A2_Q2_uniform_dense", "A2_Q2_probability_dense", "A2_Q2_margin_dense", "A3_Q3_top1_dense", "A3_Q3_top2_dense", "A3_Q3_top3_dense"):
        pcomp("query_policy_vs_Q0", ch, "R1_Q0_dense", core, KS)
    pcomp("query_policy_vs_Q0", "FINAL_pipeline", "R1_Q0_dense", core, KS, "final R2 pipeline minus R1 Q0 dense")
    pcomp("retriever", "B_bm25", "B_dense", core, KS)
    pcomp("retriever", "B_hybrid_rrf", "B_dense", core, KS)
    pcomp("retriever", "B_hybrid_rrf", "B_bm25", core, KS)
    for lam in eng.protocol["stage_C"]["lambdas"]:
        pcomp("mmr_vs_standard", f"C_{mmr_base}_mmr{lam}", mmr_base_name, core + div, (3, 5, 10))
    for a_, b_ in ((3, 1), (5, 3), (10, 5), (10, 3), (5, 1)):
        for m in core + div:
            r = paired_ci(eng.arr("FINAL_pipeline", f"{m}@{a_}"), eng.arr("FINAL_pipeline", f"{m}@{b_}"), eng.draws)
            pc_rows.append({"group": "top_k", "comparison": f"K={a_} minus K={b_}", "a": "FINAL_pipeline", "b": "FINAL_pipeline", "metric": f"{m}@{a_} - {m}@{b_}", **r, "n_queries": len(c.primary), "ci_excludes_zero": bool(r["ci95_low"] > 0 or r["ci95_high"] < 0)})
    pcomp("oracle_gap", "O_Q0_dense", "R1_Q0_dense", core, KS, "oracle dense minus R1 classifier Q0 dense (R1 gap)")
    pcomp("oracle_gap", "O_Q0_dense", "FINAL_pipeline", core, KS, "oracle dense Q0 minus final R2 pipeline")
    pcomp("oracle_gap", "O_final_pipeline", "FINAL_pipeline", core, KS, "oracle with the final pipeline minus final R2 pipeline")
    pd.DataFrame(pc_rows).to_csv(R2 / "r2_paired_comparisons.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- outputs
    pd.DataFrame(eng.rows).to_csv(R2 / "per_query_metrics_all_configs.csv", index=False, float_format="%.17g")
    t = pd.DataFrame(eng.top10)
    t[t.config.isin(["FINAL_pipeline", "O_final_pipeline"])].to_csv(R2 / "final_pipeline_top10.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    t.drop(columns=["retrieved_report_text"]).to_csv(R2 / "retrieval_top10_all_configs.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(log_).to_csv(R2 / "r2_stage_comparisons.csv", index=False, float_format="%.17g")
    c.save_cache()
    summ = []
    mcols = [f"{m}@{k}" for m in ("jaccard_truth", "ndcg", "union_coverage", "hit", "zero_overlap_rate", "duplicate_text_rate", "unique_templates", "mean_pairwise_cosine", "context_words", "jaccard_query") for k in KS if not (m == "mean_pairwise_cosine" and k == 1)]
    mcols += ["rr@10"]
    sub = [u for u in c.primary if u not in c.dupset]
    for name in eng.mat:
        if name.startswith("N_"):
            continue
        d = eng.mat[name]
        if len(d) != len(c.primary):
            continue
        row = {"config": name, "n_queries": len(c.primary), **eng.q_stats[name]}
        for m in mcols:
            v = d.loc[c.primary, m].to_numpy(float)
            row[m] = float(np.nanmean(v))
            if m.split("@")[0] in ("jaccard_truth", "ndcg", "union_coverage") or m in ("duplicate_text_rate@3", "duplicate_text_rate@5", "hit@3"):
                ci = mean_ci(v, eng.draws)
                row[f"{m}__ci95_low"], row[f"{m}__ci95_high"] = ci["ci95_low"], ci["ci95_high"]
        row["jaccard_truth@3_excluding_exact_duplicate_text_queries"] = float(d.loc[sub, "jaccard_truth@3"].mean())
        row["ndcg@3_excluding_exact_duplicate_text_queries"] = float(d.loc[sub, "ndcg@3"].mean())
        row["n_queries_excluding_duplicates"] = len(sub)
        summ.append(row)
    pd.DataFrame(summ).to_csv(R2 / "r2_summary_all_configs.csv", index=False, float_format="%.17g")
    # random reference for union coverage (expected value of K random reports; fixed seed)
    rng = np.random.default_rng(42)
    rand = {}
    for k in KS:
        vals = []
        for u in c.primary:
            cov = []
            for _ in range(200):
                pick_ = rng.choice(len(c.ids), size=k, replace=False)
                un = frozenset().union(*[c.doc_set[c.ids[i]] for i in pick_])
                cov.append(len(c.truth[u] & un) / len(c.truth[u]))
            vals.append(np.mean(cov))
        rand[f"union_coverage@{k}"] = float(np.mean(vals))
    rand.update({f"jaccard_truth@{k}": float(np.mean([c.all_gain[u].mean() for u in c.primary])) for k in KS})
    (R2 / "random_reference.json").write_text(json.dumps({"description": "expected value for K reports drawn uniformly at random (200 draws per query, seed 42); Jaccard is the exact expectation", **rand}, indent=2), encoding="utf-8")
    res = {"decisions": decisions, "selected_query_policy": sel_q_desc, "selected_retriever": ret, "mmr": {"base": mmr_base, "lambda": lam_sel}, "final_config": final_name, "selected_K": cur,
           "primary_set_size": len(c.primary), "protocol_sha256": sha256_file(R2 / "r2_selection_protocol.json"), "expansion_sha256": sha256_file(R2 / "finding_query_expansion.json"), "normal_phrase": chosen}
    (R2 / "r2_stage_decisions.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
