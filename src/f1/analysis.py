"""F1 analysis for one partition ("validation" dry run or "test"). Uses the frozen metric definitions only: C6 evaluate (classifier), R2 evaluate_ranking (retrieval),
G1 study_row (report generation, provenance, copying), G1F routing / abstention definitions. Bootstrap: 1,000 resamples, seed 42, unit = study (IU has no patient id).
No value is tuned; nothing here feeds back into any component.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from scripts.c6_03_evaluate import evaluate as c6_evaluate, macro_cal
from src.classification.calibration import adaptive_bins, brier, ece_adaptive, log_loss
from src.classification.final_test import bootstrap_weights, load_frozen, macro_binary, micro_binary, weighted_macro_metrics
from src.classification.labels import LABELS, NO_FINDING
from src.f1.pipeline import F1, out_dir
from src.generation.extraction import ABNORMAL, ReportExtractor
from src.generation.metrics import per_class_prf, prf
from src.generation.parse import parse_report
from src.generation.study_eval import FS, study_row
from src.retrieval.evaluation import bootstrap_draws, mean_ci, paired_ci
from src.system import guard
from src.utils.config import PROJECT_ROOT

EXP = PROJECT_ROOT / "results/classification/experiments"
G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]     # C2 definition
N_BOOT, SEED = 1000, 42
KS = (1, 3, 5, 10)
RET_METRICS = ("jaccard_truth", "ndcg", "union_coverage", "hit", "rr", "duplicate_text_rate", "zero_overlap_rate")
TEXT_KEYS = ("rouge_l", "bleu1", "bleu4", "meteor_exact", "mean_words")
ELLIPSIS = None


def sets_of(v):
    if isinstance(v, (list, tuple, set, frozenset)):
        return set(v)
    return set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()


def q(v, p) -> float:
    return float(np.nanpercentile(v, p))


# ------------------------------------------------------------------------------------------------------------------ classification
def classification_eval(partition: str, pop: pd.DataFrame, cls_df: pd.DataFrame, od) -> dict:
    policy, cal = load_frozen(EXP / "c4_operating_policy/final_operating_policy.json", EXP / "c5_calibration/final_calibrators.json")
    cl = cls_df.set_index("uid")
    uids = [u for u in pop.index if bool(pop.loc[u, "has_eval_set"]) and u in cl.index]
    scores = np.column_stack([cl.loc[uids, f"{l}__score"].astype(float).to_numpy() for l in LABELS])
    truth = {u: set(str(pop.loc[u, "eval_set"]).split(";")) - {""} for u in uids}
    raw = np.array([[1 if (l in truth[u]) else 0 for l in LABELS] for u in uids], dtype=int)         # No Finding positive iff the mapped truth is exactly {No Finding}
    valid = np.ones_like(raw, dtype=bool)
    R = c6_evaluate(partition, scores, raw, valid, np.array(uids), policy, cal)
    pc, bt, calt, nf = R["ranking_pc"], R["binary"], R["calibration"], R["no_finding"]
    per = pc[["observation", "n_valid", "n_positive", "n_negative", "prevalence", "auroc", "auprc", "undefined_reason"]].copy()
    per = per.merge(bt[["observation", "tp", "fp", "tn", "fn", "precision", "recall", "specificity", "f1", "balanced_accuracy"]], on="observation").merge(calt[["observation", "brier", "log_loss", "ece"]], on="observation")
    per = per.rename(columns={"n_positive": "support"})
    per.to_csv(od / "classification_per_class.csv", index=False, float_format="%.17g")
    macro, micro, mcal = macro_binary(bt), micro_binary(bt) if bt.tp.sum() + bt.fp.sum() and bt.tp.sum() + bt.fn.sum() else {}, macro_cal(calt)
    point = {**R["ranking_sum"], **macro, **micro, "macro_brier": mcal["brier"], "macro_log_loss": mcal["log_loss"], "macro_ece": mcal["ece"]}
    # study-level bootstrap (same convention as C6, unit = study)
    prob, pred = R["pipeline"]["prob"], R["pipeline"]["pred_final"]
    boots = {k: [] for k in ("macro_auroc", "macro_auprc", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "macro_balanced_accuracy", "macro_brier", "macro_log_loss", "macro_ece")}
    y = raw == 1
    draws = bootstrap_draws(len(uids), N_BOOT, SEED)
    for w, idx in zip(bootstrap_weights(np.array(uids), N_BOOT, SEED), draws):
        m = weighted_macro_metrics(scores, prob, pred, raw, valid, w)
        for k in ("macro_auroc", "macro_auprc", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "macro_balanced_accuracy"):
            boots[k].append(m[k])
        ll, ec, br = [], [], []
        for j in range(len(LABELS)):
            pj, yj = prob[idx, j], y[idx, j]
            br.append(brier(pj, yj)), ll.append(log_loss(pj, yj)), ec.append(ece_adaptive(pj, yj, 10))
        boots["macro_brier"].append(float(np.mean(br))), boots["macro_log_loss"].append(float(np.mean(ll))), boots["macro_ece"].append(float(np.mean(ec)))
    ci = {k: {"point": point[k], "ci95_low": q(v, 2.5), "ci95_high": q(v, 97.5)} for k, v in boots.items()}
    pool_p, pool_y = prob.ravel(), y.ravel()
    rel = pd.DataFrame(adaptive_bins(pool_p, pool_y, 10))
    rel.to_csv(od / "classification_reliability_pooled_bins.csv", index=False, float_format="%.17g")
    rel_cls = pd.concat([pd.DataFrame(adaptive_bins(prob[:, j], y[:, j], 10)).assign(observation=l) for j, l in enumerate(LABELS) if y[:, j].sum() > 0 and (~y[:, j]).sum() > 0])
    rel_cls.to_csv(od / "classification_reliability_per_class_bins.csv", index=False, float_format="%.17g")
    out = {"n_studies": len(uids), "point": point, "ci95": ci, "n_classes_defined": R["ranking_sum"]["n_classes_defined"], "undefined_classes": R["ranking_sum"]["undefined_classes"], "no_finding": nf, "pooled_ece": float(ece_adaptive(pool_p, pool_y, 10)),
           "labels_note": "IU truth = MeSH-mapped findings; not mapped is not a verified negative; the classifier is applied out of domain (CheXpert -> IU X-Ray); metrics use the frozen C6 definitions"}
    (od / "classification_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.DataFrame({"uid": uids, **{f"{l}__label": raw[:, j] for j, l in enumerate(LABELS)}}).to_csv(od / "classification_labels.csv", index=False)
    return out


# ------------------------------------------------------------------------------------------------------------------ retrieval
def retrieval_eval(od) -> dict:
    cls_q = pd.read_csv(od / "retrieval_classifier_query_per_study.csv", dtype={"uid": str}).set_index("uid")
    orc = pd.read_csv(od / "retrieval_oracle_query_per_study.csv", dtype={"uid": str}).set_index("uid").loc[cls_q.index]
    draws = bootstrap_draws(len(cls_q), N_BOOT, SEED)
    rows = []
    for k in KS:
        for m in RET_METRICS:
            col = f"{m}@{k}"
            a, b = cls_q[col].to_numpy(float), orc[col].to_numpy(float)
            c1, c2, gap = mean_ci(a, draws), mean_ci(b, draws), paired_ci(a, b, draws)
            rows.append({"K": k, "metric": m, "n_queries": len(cls_q), "classifier_query": c1["mean"], "classifier_ci95_low": c1["ci95_low"], "classifier_ci95_high": c1["ci95_high"], "oracle_query": c2["mean"], "oracle_ci95_low": c2["ci95_low"],
                         "oracle_ci95_high": c2["ci95_high"], "gap_classifier_minus_oracle": gap["mean_difference"], "gap_ci95_low": gap["ci95_low"], "gap_ci95_high": gap["ci95_high"], "n_defined": int(np.isfinite(a).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(od / "retrieval_metrics_by_k.csv", index=False, float_format="%.17g")
    return {"n_queries": len(cls_q), "primary_K": 5, "table": rows}


# ------------------------------------------------------------------------------------------------------------------ report generation and the end-to-end system
def final_reports(cases: list[dict], raw: dict) -> tuple[dict, list[str]]:
    out, failed = {}, []
    for c in cases:
        u = c["uid"]
        if c["system_interpretation_state"] == guard.STATE_INDETERMINATE:
            out[u] = {"findings": guard.INDETERMINATE_FINDINGS, "impression": guard.INDETERMINATE_IMPRESSION, "format_ok": True}
        elif u in raw:
            p = parse_report(raw[u]["text"])
            out[u] = {"findings": p["findings"], "impression": p["impression"], "format_ok": p["format_ok"]}
        else:
            failed.append(u)
    return out, failed


def report_eval(partition: str, cases: list[dict], raw: dict, od) -> dict:
    ext = ReportExtractor()
    final, failed = final_reports(cases, raw)
    rows = []
    for c in cases:
        if c["uid"] in final:
            rows.append({**study_row(c, "final_system", final[c["uid"]], ext), "routing_state": c["system_interpretation_state"], "unmapped_terms": c["unmapped_terms"], "uncertain_terms": c["uncertain_terms"]})
    df = pd.DataFrame(rows)
    df.to_csv(od / "report_per_study_results.csv", index=False)
    d = df.set_index("uid")
    ids = list(d.index)
    clin = [u for u in ids if bool(d.loc[u, "in_clinical"])]
    cl = d.loc[clin]
    state = d.routing_state
    case_by = {c["uid"]: c for c in cases}
    cls_idx = {c: i for i, c in enumerate(ABNORMAL)}
    rare_mask = np.array([c in RARE for c in ABNORMAL])
    truth_n = np.zeros(len(ABNORMAL))
    for t in cl.truth:
        for f in sets_of(t):
            truth_n[cls_idx[f]] += 1
    macro_classes = truth_n > 0
    n = len(cl)
    A = cl[["tp", "fp", "fn", "hall", "omit"]].to_numpy(float)
    P = cl[["clf_fp", "clf_fp_mentioned", "clf_tp", "clf_tp_retained"]].to_numpy(float)
    C, Rr = np.zeros((n, len(ABNORMAL), 3)), np.zeros((n, len(ABNORMAL), 2))
    for i, (st, tr, cp) in enumerate(zip(cl.stated, cl.truth, cl.clf_pos)):
        st, tr, cp = sets_of(st), sets_of(tr), sets_of(cp)
        for f in ABNORMAL:
            j = cls_idx[f]
            C[i, j] = (f in st and f in tr, f in st and f not in tr, f not in st and f in tr)
            Rr[i, j] = (f in cp and f in tr, f in cp and f in tr and f in st)
    txt = d[["rouge_l", "bleu1", "bleu4", "meteor_exact", "words"]].to_numpy(float)

    def metrics(idx: np.ndarray) -> dict:
        a, p, c, r = A[idx], P[idx], C[idx].sum(0), Rr[idx].sum(0)
        pr = prf(a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum())
        tp, fp, fn = c[:, 0], c[:, 1], c[:, 2]
        f1c = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
        rt, rr_ = r[rare_mask, 0].sum(), r[rare_mask, 1].sum()
        return {"precision": pr["precision"], "recall": pr["recall"], "f1": pr["f1"], "macro_f1": float(np.nanmean(f1c[macro_classes])), "hallucination_rate": a[:, 3].mean(), "omission_rate": a[:, 4].mean(),
                "clf_fp_propagation": p[:, 1].sum() / p[:, 0].sum() if p[:, 0].sum() else np.nan, "tp_retention": p[:, 3].sum() / p[:, 2].sum() if p[:, 2].sum() else np.nan, "rare_tp_retention": rr_ / rt if rt else np.nan}

    def tmetrics(idx: np.ndarray) -> dict:
        m = txt[idx].mean(0)
        return {"rouge_l": m[0], "bleu1": m[1], "bleu4": m[2], "meteor_exact": m[3], "mean_words": m[4]}

    dc, dp = bootstrap_draws(len(clin), N_BOOT, SEED), bootstrap_draws(len(ids), N_BOOT, SEED)
    pt = {**metrics(np.arange(n)), **tmetrics(np.arange(len(ids)))}
    bs = {k: [] for k in pt}
    for i in range(N_BOOT):
        for k, v in {**metrics(dc[i]), **tmetrics(dp[i])}.items():
            bs[k].append(v)
    rep = pd.DataFrame([{"metric": k, "n_studies": len(ids) if k in TEXT_KEYS else n, "value": pt[k], "ci95_low": q(bs[k], 2.5), "ci95_high": q(bs[k], 97.5)} for k in pt])
    rep.to_csv(od / "report_generation_metrics.csv", index=False, float_format="%.17g")

    # ---- per-finding results and rare findings
    gen_sets, tr_sets = [FS(sets_of(x)) for x in cl.stated], [FS(sets_of(x)) for x in cl.truth]
    res = {r["finding"]: r for r in per_class_prf(gen_sets, tr_sets, ABNORMAL)}
    pf = []
    for f in ABNORMAL:
        j = cls_idx[f]
        r = Rr[:, j].sum(0)
        pf.append({"finding": f, "rare_c2": f in RARE, "support_test_truth_positive": int(res[f]["n_truth"]), "n_generated": int(res[f]["n_generated"]), "tp": int(res[f]["tp"]), "fp": int(res[f]["fp"]), "fn": int(res[f]["fn"]),
                   "precision": res[f]["precision"], "recall": res[f]["recall"], "f1": res[f]["f1"], "classifier_true_positives": int(r[0]), "classifier_tp_retained": int(r[1]), "tp_retention": r[1] / r[0] if r[0] else np.nan})
    pd.DataFrame(pf).to_csv(od / "report_per_finding_results.csv", index=False, float_format="%.17g")
    rr_ = Rr[:, rare_mask].sum((0, 1))
    rare_pool = {"classifier_true_positives": int(rr_[0]), "retained": int(rr_[1]), "tp_retention": float(rr_[1] / rr_[0]) if rr_[0] else None}

    # ---- three-state analysis (P4)
    ref = cl.reference_state
    st_c = state.loc[clin]
    sc = {s: int((state == s).sum()) for s in guard.STATES}
    sc_c = {s: int((st_c == s).sum()) for s in guard.STATES}
    refn, refa = (ref == "normal").to_numpy(), (ref == "abnormal").to_numpy()
    rn, ra = (st_c == guard.STATE_NORMAL).to_numpy(), (st_c == guard.STATE_ABNORMAL).to_numpy()
    dec = rn | ra

    def decided(idx: np.ndarray) -> dict:
        nd, ad = (refn & dec)[idx].sum(), (refa & dec)[idx].sum()
        return {"normal_recall_among_decided": (refn & rn)[idx].sum() / nd if nd else np.nan, "abnormal_recall_among_decided": (refa & ra)[idx].sum() / ad if ad else np.nan, "decision_coverage": dec[idx].mean()}

    dpt = decided(np.arange(n))
    db = {k: [decided(dc[i])[k] for i in range(N_BOOT)] for k in dpt}
    strat = []
    for r_ in ("normal", "abnormal"):
        m = ref == r_
        nn = int(m.sum())
        strat.append({"reference_state": r_, "n": nn, **{f"routed_{s}_n": int((st_c[m] == s).sum()) for s in guard.STATES}, **{f"routed_{s}_pct": 100 * int((st_c[m] == s).sum()) / nn for s in guard.STATES}})
    three = {"state_counts_P3": sc, "state_counts_P4": sc_c, "reference_stratified": strat, "decided": {k: {"value": dpt[k], "ci95_low": q(db[k], 2.5), "ci95_high": q(db[k], 97.5)} for k in dpt}, "decided_n": int(dec.sum()), "eligible_n": n,
             "decision_coverage_P3": float((state != guard.STATE_INDETERMINATE).mean())}

    # ---- abstention analysis (descriptive)
    abst_c = [u for u in clin if state[u] == guard.STATE_INDETERMINATE]
    ref_abn = [u for u in clin if d.loc[u, "reference_state"] == "abnormal"]
    ref_abn_abst = [u for u in abst_c if d.loc[u, "reference_state"] == "abnormal"]
    ref_abn_dec = [u for u in ref_abn if state[u] != guard.STATE_INDETERMINATE]
    truth = {u: sets_of(d.loc[u, "truth"]) for u in clin}
    inst_abst, inst_all = [f for u in ref_abn_abst for f in truth[u]], [f for u in ref_abn for f in truth[u]]
    a_, b_ = sum(bool(truth[u] & set(RARE)) for u in ref_abn_abst), sum(bool(truth[u] & set(RARE)) for u in ref_abn_dec)
    odds = fisher_exact([[a_, len(ref_abn_abst) - a_], [b_, len(ref_abn_dec) - b_]]) if ref_abn_abst and ref_abn_dec else (np.nan, np.nan)
    fd = pd.DataFrame([{"finding": f, "rare_c2": f in RARE, "reference_positive_in_abstained_reference_abnormal": inst_abst.count(f), "reference_positive_in_all_reference_abnormal": inst_all.count(f)} for f in ABNORMAL])
    fd.to_csv(od / "abstention_finding_distribution.csv", index=False)
    abst = {"n_abstained_P3": int((state == guard.STATE_INDETERMINATE).sum()), "pct_abstained_P3": float(100 * (state == guard.STATE_INDETERMINATE).mean()), "n_abstained_P4": len(abst_c), "pct_abstained_P4": float(100 * len(abst_c) / n),
            "reference_normal_among_abstained": int(sum(d.loc[u, "reference_state"] == "normal" for u in abst_c)), "reference_abnormal_among_abstained": len(ref_abn_abst),
            "reference_normal_fraction": float(np.mean([d.loc[u, "reference_state"] == "normal" for u in abst_c])) if abst_c else None, "reference_abnormal_fraction": len(ref_abn_abst) / len(abst_c) if abst_c else None,
            "abstained_share_of_reference_abnormal": len(ref_abn_abst) / max(len(ref_abn), 1), "abstained_share_of_reference_normal": float(np.mean([state[u] == guard.STATE_INDETERMINATE for u in clin if d.loc[u, "reference_state"] == "normal"])),
            "findings_in_abstained_reference_abnormal": {f: inst_abst.count(f) for f in ABNORMAL if inst_abst.count(f)}, "rare_instances_in_abstained": int(sum(f in RARE for f in inst_abst)), "total_instances_in_abstained": len(inst_abst),
            "rare_instances_in_all_reference_abnormal": int(sum(f in RARE for f in inst_all)), "total_instances_in_all_reference_abnormal": len(inst_all), "abstained_reference_abnormal_with_a_rare_finding": a_, "decided_reference_abnormal_with_a_rare_finding": b_,
            "decided_reference_abnormal_n": len(ref_abn_dec), "fisher_exact_p_two_sided": float(odds[1]), "fisher_exact_odds_ratio": float(odds[0])}

    # ---- routing / prose consistency
    def prose(mask_ids):
        s = d.loc[mask_ids].report_state.value_counts()
        nn = len(mask_ids)
        return {"n": nn, **{f"prose_{k}": int(s.get(k, 0)) for k in ("normal", "abnormal", "indeterminate")}, **{f"prose_{k}_pct": float(100 * s.get(k, 0) / nn) if nn else None for k in ("normal", "abnormal", "indeterminate")}}
    cons = {"abnormal_routed_P3": prose([u for u in ids if state[u] == guard.STATE_ABNORMAL]), "abnormal_routed_P4": prose([u for u in clin if state[u] == guard.STATE_ABNORMAL]),
            "normal_routed_P3": prose([u for u in ids if state[u] == guard.STATE_NORMAL]), "indeterminate_routed_P3": prose([u for u in ids if state[u] == guard.STATE_INDETERMINATE])}
    cons["abnormal_routing_to_normal_prose_rate_P3"] = cons["abnormal_routed_P3"]["prose_normal_pct"]
    ab_ids = [u for u in ids if state[u] == guard.STATE_ABNORMAL]
    idx_ab = np.array([i for i, u in enumerate(ids) if state[u] == guard.STATE_ABNORMAL])
    dpr = bootstrap_draws(len(ab_ids), N_BOOT, SEED)
    flag = (d.loc[ab_ids].report_state == "normal").to_numpy(float)
    bt = [flag[i].mean() for i in dpr]
    cons["abnormal_routing_to_normal_prose_rate_ci95"] = [q(bt, 2.5), q(bt, 97.5)]

    # ---- provenance (A classifier only, B retrieval only, C both, D neither) with reference support, clinical subset
    prov = {k: {"n": 0, "supported": 0, "unsupported": 0} for k in "ABCD"}
    for u in clin:
        st, tr, cp = sets_of(d.loc[u, "stated"]), sets_of(d.loc[u, "truth"]), sets_of(d.loc[u, "clf_pos"])
        union = set().union(*[set(r["mapped_findings"]) for r in case_by[u]["retrieved"]]) if case_by[u]["retrieved"] else set()
        for f in st:
            k = "C" if (f in cp and f in union) else "A" if f in cp else "B" if f in union else "D"
            prov[k]["n"] += 1
            prov[k]["supported" if f in tr else "unsupported"] += 1
    names = {"A": "classifier only", "B": "retrieval only", "C": "classifier and retrieval", "D": "neither"}
    pd.DataFrame([{"provenance": names[k], "stated_findings": v["n"], "reference_supported": v["supported"], "reference_unsupported": v["unsupported"], "share_supported": v["supported"] / v["n"] if v["n"] else np.nan} for k, v in prov.items()]).to_csv(od / "provenance.csv", index=False)

    # ---- copying (G1 definitions; studies with context)
    ctx_ids = [u for u in ids if case_by[u]["retrieved"]]
    cop = {"n_studies_with_context": len(ctx_ids), "copied_sentence_rate_mean": float(d.loc[ctx_ids, "copy_copied_sentence_rate"].mean()), "whole_report_copy_rate": float(d.loc[ctx_ids, "copy_whole_report_copy"].astype(float).mean()),
           "repeated_sentence_rate_mean_P3": float(d.repeated_sentence_rate.mean()), "repeated_sentence_rate_mean_with_context": float(d.loc[ctx_ids, "repeated_sentence_rate"].mean())}
    dcx = bootstrap_draws(len(ctx_ids), N_BOOT, SEED)
    for nm, col in (("copied_sentence_rate", "copy_copied_sentence_rate"), ("whole_report_copy_rate", "copy_whole_report_copy"), ("repeated_sentence_rate", "repeated_sentence_rate")):
        v = d.loc[ctx_ids, col].astype(float).to_numpy()
        bb = [v[i].mean() for i in dcx]
        cop[f"{nm}_ci95"] = [q(bb, 2.5), q(bb, 97.5)]

    # ---- frozen failure taxonomy (P4; multi-label)
    cats = {k: [] for k in ("1_classifier_error", "2_normal_abnormal_classifier_mismatch", "3_retrieval_query_mismatch", "4_retrieval_only_unsupported_finding", "5_report_generator_omission", "6_report_generator_hallucination",
                            "7_abnormal_routing_normal_prose", "8_indeterminate_abstention", "9_corpus_limitation", "10_reference_label_limitation")}
    exact_possible = {}
    corpus = pd.read_csv(PROJECT_ROOT / "results/retrieval/experiments/r1_baseline/retrieval_corpus.csv", dtype={"uid": str}).fillna("")
    corpus_sets = {frozenset(x for x in str(e).split(";") if x) for e in corpus.eval_set}
    for u in clin:
        c = case_by[u]
        tr = sets_of(d.loc[u, "truth"])
        cp = set(c["classifier_positive_findings"])
        st, cps = sets_of(d.loc[u, "stated"]), sets_of(d.loc[u, "clf_pos"])
        union = set().union(*[set(r["mapped_findings"]) for r in c["retrieved"]]) if c["retrieved"] else set()
        full_truth = set(c["reference"]["truth_findings"])
        flags = {"1_classifier_error": cp != tr, "2_normal_abnormal_classifier_mismatch": state[u] != guard.STATE_INDETERMINATE and ((state[u] == guard.STATE_NORMAL) != (d.loc[u, "reference_state"] == "normal")),
                 "3_retrieval_query_mismatch": bool(c["retrieved"]) and len(full_truth & union) == 0, "4_retrieval_only_unsupported_finding": any((f not in cps and f in union and f not in tr) for f in st),
                 "5_report_generator_omission": bool(d.loc[u, "omit"]), "6_report_generator_hallucination": bool(d.loc[u, "hall"]), "7_abnormal_routing_normal_prose": state[u] == guard.STATE_ABNORMAL and d.loc[u, "report_state"] == "normal",
                 "8_indeterminate_abstention": state[u] == guard.STATE_INDETERMINATE, "9_corpus_limitation": frozenset(full_truth) not in corpus_sets, "10_reference_label_limitation": bool(c["unmapped_terms"] or c["uncertain_terms"])}
        for k, v in flags.items():
            if v:
                cats[k].append(u)
    anyfail = {u for k in list(cats)[:8] for u in cats[k]}
    fa = pd.DataFrame([{"category": k, "n_studies": len(v), "pct_of_clinical_subset": 100 * len(v) / n} for k, v in cats.items()] + [{"category": "no_failure_in_categories_1_to_8", "n_studies": n - len(anyfail), "pct_of_clinical_subset": 100 * (n - len(anyfail)) / n}])
    fa.to_csv(od / "failure_analysis.csv", index=False)
    pd.DataFrame([{"uid": u, "categories": ";".join(k for k, v in cats.items() if u in v)} for u in clin]).to_csv(od / "failure_categories_per_study.csv", index=False)

    out = {"n_primary_P3_with_final_report": len(ids), "n_clinical_P4": n, "generation_failures": failed, "n_generation_failures": len(failed), "metrics": {r["metric"]: {"value": r["value"], "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"]} for r in rep.to_dict("records")},
           "rare_pooled": rare_pool, "three_state": three, "abstention": abst, "routing_prose_consistency": cons, "provenance": {names[k]: v for k, v in prov.items()}, "copying": cop, "failure_counts": {k: len(v) for k, v in cats.items()},
           "no_failure_categories_1_to_8": n - len(anyfail), "format_ok_rate": float(d.format_ok.astype(float).mean())}
    (od / "report_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out
