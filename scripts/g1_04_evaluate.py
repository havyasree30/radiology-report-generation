"""G1 step 4: evaluate B0_rule_based versus G1_single_agent_rag on validation data only.

    .venv\\Scripts\\python.exe -m scripts.g1_04_evaluate [--dry-run]

--dry-run uses the B0 report as a stand-in for G1 and writes to dry_run/ (pipeline test only; never reported).
Primary: finding precision/recall/F1, hallucination, omission, classifier false-positive propagation, true-positive retention.
Secondary: BLEU-1/4, ROUGE-L, METEOR (exact-match variant). Study-level paired bootstrap, 1,000 resamples, seed 42.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.classification.labels import LABELS, NO_FINDING
from src.generation.extraction import ABNORMAL, ReportExtractor
from src.generation.metrics import (classifier_propagation, copy_stats, per_class_prf, prf, provenance, repeated_sentence_rate, study_counts, text_metrics, tokens)
from src.generation.parse import combined, parse_report
from src.retrieval.evaluation import bootstrap_draws
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
SYSTEMS = ("B0_rule_based", "G1_single_agent_rag")
N_BOOT, SEED = 1000, 42
FS = lambda x=(): frozenset(x)  # noqa: E731


def state_of_truth(truth: list[str]) -> str:
    return "normal" if set(truth) == {NO_FINDING} else "abnormal"


def classifier_state(c: dict) -> str:
    if c["classifier_positive_findings"]:
        return "abnormal"
    return "normal" if c["no_finding_positive"] else "indeterminate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = G1 / "dry_run" if args.dry_run else G1
    out.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    raw = {}
    if not args.dry_run:
        raw = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")}
    ext = ReportExtractor()
    pop = json.loads((G1 / "g1_populations.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- per-study, per-system tables
    rows, reports = [], []
    for c in cases:
        u = c["uid"]
        gen = {"B0_rule_based": {"findings": c["b0"]["findings"], "impression": c["b0"]["impression"], "format_ok": True, "format_flags": []}}
        if args.dry_run:
            gen["G1_single_agent_rag"] = dict(gen["B0_rule_based"])
        elif u in raw:
            gen["G1_single_agent_rag"] = parse_report(raw[u]["text"])
        if len(gen) < 2:
            continue                                                   # no G1 report: excluded from the paired set and reported as missing
        ref = c["reference"]
        truth = FS(ref["truth_findings"]) - {NO_FINDING} if ref["truth_findings"] is not None else None
        ref_text_set = ext.stated(ref["combined"])
        retr_sets = [FS(r["mapped_findings"]) for r in c["retrieved"]]
        retr_union = FS().union(*retr_sets) if retr_sets else FS()
        clf = FS(c["classifier_positive_findings"])
        for sysn, g in gen.items():
            text = combined(g)
            stated, definite = ext.stated(text), ext.definite(text)
            row = {"uid": u, "anon_id": c["anon_id"], "system": sysn, "in_clinical": truth is not None, "words": len(tokens(text)), "format_ok": g["format_ok"],
                   "report_state": ext.report_state(text), "stated": sorted(stated), "definite": sorted(definite), "clf_pos": sorted(clf), "truth": sorted(truth) if truth is not None else None,
                   "classifier_state": classifier_state(c), "reference_state": state_of_truth(ref["truth_findings"]) if truth is not None else None,
                   "repeated_sentence_rate": repeated_sentence_rate(text), **text_metrics(text, ref["combined"])}
            cs = copy_stats(text, [r["findings"] + " " + r["impression"] for r in c["retrieved"]])
            row.update({f"copy_{k}": v for k, v in cs.items()})
            n_unmapped = len(ext.unmappable_statements(text))
            row.update({"sentences": cs["n_sentences"], "unmappable_sentences": n_unmapped})
            if truth is not None:
                for tag, gs in (("", stated), ("def_", definite), ("reftext_", stated)):
                    tr = truth if tag != "reftext_" else ref_text_set
                    row.update({f"{tag}{k}": v for k, v in study_counts(gs, tr).items()})
                    row[f"{tag}hall"], row[f"{tag}omit"] = int(row[f"{tag}fp"] > 0), int(row[f"{tag}fn"] > 0)
                for tag, gs in (("", stated), ("def_", definite)):
                    row.update({f"{tag}{k}": v for k, v in classifier_propagation(clf, truth, gs).items()})
                row.update({f"prov_{k}": v for k, v in provenance(stated, clf, retr_union).items()})
            rows.append(row)
            if sysn == "G1_single_agent_rag":
                supp = {}
                for f in stated:
                    supp[f] = [r["rank"] for r, s in zip(c["retrieved"], retr_sets) if f in s]
                row["support_ranks"] = json.dumps(supp)
        reports.append({"study_id": u, "image_id": c["image_id"], "anon_id": c["anon_id"], "classifier_positive_findings": ";".join(c["classifier_positive_findings"]),
                        "calibrated_probabilities": json.dumps({l: round(c["classifier_probabilities"][l], 4) for l in LABELS if l in c["classifier_positive_findings"] or l == NO_FINDING}),
                        "no_finding_positive": c["no_finding_positive"], "query": c["query"], "query_status": c["query_status"],
                        "retrieved_top5_study_ids": ";".join(r["study_id"] for r in c["retrieved"]),
                        "evidence_support": ";".join(f"{e['finding']}={e['n_supporting']}/{e['n_retrieved']}" for e in c["evidence_rows"]),
                        "B0_report": f"FINDINGS: {gen['B0_rule_based']['findings']} IMPRESSION: {gen['B0_rule_based']['impression']}",
                        "G1_report": f"FINDINGS: {gen['G1_single_agent_rag']['findings']} IMPRESSION: {gen['G1_single_agent_rag']['impression']}",
                        "reference_findings": ref["findings"], "reference_impression": ref["impression"]})
    df = pd.DataFrame(rows)
    pd.DataFrame(reports).to_csv(out / "generated_reports.csv", index=False)
    df.to_csv(out / "g1_per_study_results.csv", index=False)
    paired_ids = sorted(set(df[df.system == SYSTEMS[0]].uid) & set(df[df.system == SYSTEMS[1]].uid), key=int)
    clin_ids = [u for u in paired_ids if df[(df.uid == u) & (df.system == SYSTEMS[0])].in_clinical.iloc[0]]
    sub = {s: df[(df.system == s) & df.uid.isin(paired_ids)].set_index("uid").loc[paired_ids] for s in SYSTEMS}
    clin = {s: sub[s].loc[clin_ids] for s in SYSTEMS}
    pops = {**pop, "paired_model_comparison_set": len(paired_ids), "paired_clinical_subset": len(clin_ids), "studies_without_g1_report": len(cases) - len(paired_ids),
            "g1_format_deviations": int((~sub[SYSTEMS[1]].format_ok).sum()) if not args.dry_run else 0, "dry_run": args.dry_run}
    (out / "g1_evaluation_populations.json").write_text(json.dumps(pops, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- point estimates
    def finding_summary(d: pd.DataFrame, pre: str = "") -> dict:
        tp, fp, fn = d[f"{pre}tp"].sum(), d[f"{pre}fp"].sum(), d[f"{pre}fn"].sum()
        return {**prf(tp, fp, fn), "tp": int(tp), "fp": int(fp), "fn": int(fn), "hallucination_rate_reports_with_ge1": float(d[f"{pre}hall"].mean()), "mean_hallucinated_per_report": float(d[f"{pre}fp"].mean()),
                "omission_rate_reports_with_ge1": float(d[f"{pre}omit"].mean()), "mean_omitted_per_report": float(d[f"{pre}fn"].mean())}

    def prop_summary(d: pd.DataFrame, pre: str = "") -> dict:
        return {"clf_fp_propagation": float(d[f"{pre}clf_fp_mentioned"].sum() / d[f"{pre}clf_fp"].sum()), "tp_retention": float(d[f"{pre}clf_tp_retained"].sum() / d[f"{pre}clf_tp"].sum()),
                "n_clf_fp": int(d[f"{pre}clf_fp"].sum()), "n_clf_tp": int(d[f"{pre}clf_tp"].sum())}

    summary = {"populations": pops}
    for s in SYSTEMS:
        d, p = clin[s], sub[s]
        summary[s] = {"finding_any_mention": finding_summary(d), "finding_definite_only": finding_summary(d, "def_"), "finding_vs_reference_text_extraction": finding_summary(d, "reftext_"),
                      "propagation_any_mention": prop_summary(d), "propagation_definite_only": prop_summary(d, "def_"),
                      "text": {k: float(p[k].mean()) for k in ("bleu1", "bleu4", "rouge_l", "meteor_exact")}, "words_mean": float(p.words.mean()), "words_median": float(p.words.median()),
                      "repeated_sentence_rate_mean": float(p.repeated_sentence_rate.mean()), "format_ok_rate": float(p.format_ok.mean()),
                      "unmappable_sentence_share": float(p.unmappable_sentences.sum() / max(p.sentences.sum(), 1))}
    # per-finding table
    pf = []
    for s in SYSTEMS:
        gen_sets = [FS(x) for x in clin[s].stated]
        tr_sets = [FS(x) for x in clin[s].truth]
        for r in per_class_prf(gen_sets, tr_sets, ABNORMAL):
            clf_pos_n = int(sum(r["finding"] in x for x in clin[s].clf_pos))
            pf.append({"system": s, **r, "classifier_positive_studies": clf_pos_n})
    pfd = pd.DataFrame(pf)
    pfd.to_csv(out / "g1_per_finding_results.csv", index=False, float_format="%.17g")
    macro = {}
    for s in SYSTEMS:
        g = pfd[(pfd.system == s) & (pfd.n_truth > 0)]
        macro[s] = {"classes_with_truth": int(len(g)), "macro_precision": float(g.precision.mean()), "macro_recall": float(g.recall.mean()), "macro_f1": float(g.f1.mean())}
    summary["macro_over_classes_with_truth"] = macro

    # ---------------------------------------------------------------- bootstrap (paired, studies as the unit)
    dc = bootstrap_draws(len(clin_ids), N_BOOT, SEED)
    dp = bootstrap_draws(len(paired_ids), N_BOOT, SEED)

    def fm(d, idx, pre=""):
        a = d[[f"{pre}tp", f"{pre}fp", f"{pre}fn", f"{pre}hall", f"{pre}omit"]].to_numpy(float)[idx]
        tp, fp, fn = a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum()
        r = prf(tp, fp, fn)
        return {"precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "hallucination_rate": a[:, 3].mean(), "omission_rate": a[:, 4].mean(), "mean_hallucinated": a[:, 1].mean(), "mean_omitted": a[:, 2].mean()}

    def pm(d, idx, pre=""):
        a = d[[f"{pre}clf_fp", f"{pre}clf_fp_mentioned", f"{pre}clf_tp", f"{pre}clf_tp_retained"]].to_numpy(float)[idx]
        return {"clf_fp_propagation": a[:, 1].sum() / a[:, 0].sum(), "tp_retention": a[:, 3].sum() / a[:, 2].sum()}

    def collect(draws, d_b0, d_g1, fn_):
        full = np.arange(len(d_b0))
        pt = {k: (fn_(d_g1, full)[k], fn_(d_b0, full)[k]) for k in fn_(d_b0, full)}
        boots = {k: [] for k in pt}
        bg, bb = {k: [] for k in pt}, {k: [] for k in pt}
        for idx in draws:
            g, b = fn_(d_g1, idx), fn_(d_b0, idx)
            for k in pt:
                boots[k].append(g[k] - b[k])
                bg[k].append(g[k])
                bb[k].append(b[k])
        pc = lambda v, q: float(np.nanpercentile(v, q))  # noqa: E731
        return {k: {"b0": pt[k][1], "g1": pt[k][0], "diff": pt[k][0] - pt[k][1], "ci95_low": pc(boots[k], 2.5), "ci95_high": pc(boots[k], 97.5),
                    "b0_ci95_low": pc(bb[k], 2.5), "b0_ci95_high": pc(bb[k], 97.5), "g1_ci95_low": pc(bg[k], 2.5), "g1_ci95_high": pc(bg[k], 97.5)} for k in pt}

    boot_rows = []
    for name, res in (("finding_any_mention", collect(dc, clin[SYSTEMS[0]], clin[SYSTEMS[1]], fm)), ("finding_definite_only", collect(dc, clin[SYSTEMS[0]], clin[SYSTEMS[1]], lambda d, i: fm(d, i, "def_"))),
                      ("propagation_any_mention", collect(dc, clin[SYSTEMS[0]], clin[SYSTEMS[1]], pm)), ("propagation_definite_only", collect(dc, clin[SYSTEMS[0]], clin[SYSTEMS[1]], lambda d, i: pm(d, i, "def_"))),
                      ("text_primary_set", collect(dp, sub[SYSTEMS[0]], sub[SYSTEMS[1]], lambda d, i: {k: d[k].to_numpy(float)[i].mean() for k in ("rouge_l", "bleu1", "bleu4", "meteor_exact")}))):
        for k, v in res.items():
            boot_rows.append({"analysis": name, "metric": k, "n_studies": len(clin_ids) if name != "text_primary_set" else len(paired_ids), **v})
    bt = pd.DataFrame(boot_rows)
    bt.to_csv(out / "g1_paired_bootstrap.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- hallucination / omission, provenance, support vs truth, context use
    ho = []
    for s in SYSTEMS:
        d = clin[s]
        for lab, pre in (("any mention (primary)", ""), ("definite assertions only", "def_")):
            ho.append({"system": s, "definition": lab, "hallucinated_per_report": d[f"{pre}fp"].mean(), "omitted_per_report": d[f"{pre}fn"].mean(), "pct_reports_ge1_hallucinated": 100 * d[f"{pre}hall"].mean(),
                       "pct_reports_ge1_omission": 100 * d[f"{pre}omit"].mean(), "n_reports": len(d)})
    pd.DataFrame(ho).to_csv(out / "g1_hallucination_omission.csv", index=False, float_format="%.17g")
    pr_rows = []
    for s in SYSTEMS:
        for lab, pre in (("any mention (primary)", ""), ("definite assertions only", "def_")):
            d = clin[s]
            pr_rows.append({"system": s, "definition": lab, **prop_summary(d, pre), "fp_mentioned": int(d[f"{pre}clf_fp_mentioned"].sum()), "tp_retained": int(d[f"{pre}clf_tp_retained"].sum())})
    pd.DataFrame(pr_rows).to_csv(out / "g1_fp_propagation.csv", index=False, float_format="%.17g")
    g1c, g1p = clin[SYSTEMS[1]], sub[SYSTEMS[1]]
    prov = g1c[[f"prov_{k}" for k in "ABCD"]].sum()
    provp = {"A_classifier_only": int(prov["prov_A"]), "B_retrieval_only": int(prov["prov_B"]), "C_both": int(prov["prov_C"]), "D_neither": int(prov["prov_D"])}
    tot = max(sum(provp.values()), 1)
    pd.DataFrame([{"category": k, "n_findings": v, "share": v / tot} for k, v in provp.items()]).to_csv(out / "g1_grounding_provenance.csv", index=False, float_format="%.17g")
    summary["grounding_provenance_g1_clinical_subset"] = {**provp, "unsupported_rate_D": provp["D_neither"] / tot, "n_stated_findings": tot}
    # support vs truth for classifier positives, and G1 mention by support bucket
    sv = []
    case_by = {c["uid"]: c for c in cases}
    for u in clin_ids:
        c = case_by[u]
        truth = FS(c["reference"]["truth_findings"]) - {NO_FINDING}
        stated = FS(clin[SYSTEMS[1]].loc[u, "stated"])
        for e in c["evidence_rows"]:
            sv.append({"uid": u, "finding": e["finding"], "probability": e["classifier_probability"], "n_supporting": e["n_supporting"], "n_retrieved": e["n_retrieved"],
                       "is_true_positive": e["finding"] in truth, "g1_mentions": e["finding"] in stated})
    svd = pd.DataFrame(sv)
    svd.to_csv(out / "g1_classifier_positive_instances.csv", index=False)
    svd["support_bucket"] = pd.cut(svd.n_supporting, [-1, 0, 2, 5], labels=["0/5", "1-2/5", "3-5/5"])
    sb = svd.groupby(["support_bucket", "is_true_positive"], observed=False).agg(n=("uid", "size"), g1_mention_rate=("g1_mentions", "mean")).reset_index()
    sb.to_csv(out / "g1_support_bucket_analysis.csv", index=False, float_format="%.17g")
    sup = {"mean_support_fraction_true_positives": float((svd[svd.is_true_positive].n_supporting / svd[svd.is_true_positive].n_retrieved).mean()),
           "mean_support_fraction_false_positives": float((svd[~svd.is_true_positive].n_supporting / svd[~svd.is_true_positive].n_retrieved).mean()),
           "fp_with_zero_support": int(((~svd.is_true_positive) & (svd.n_supporting == 0)).sum()), "n_fp": int((~svd.is_true_positive).sum()),
           "tp_with_zero_support": int(((svd.is_true_positive) & (svd.n_supporting == 0)).sum()), "n_tp": int(svd.is_true_positive.sum()),
           "fp_suppressed_by_g1": int(((~svd.is_true_positive) & (~svd.g1_mentions)).sum()), "fp_suppressed_with_zero_support": int(((~svd.is_true_positive) & (~svd.g1_mentions) & (svd.n_supporting == 0)).sum()),
           "tp_omitted_by_g1": int(((svd.is_true_positive) & (~svd.g1_mentions)).sum()), "tp_omitted_with_zero_support": int(((svd.is_true_positive) & (~svd.g1_mentions) & (svd.n_supporting == 0)).sum())}
    summary["support_vs_truth"] = sup
    # context utilisation (G1 primary set with retrieval)
    ctxu, rank_hits, n_rep = [], np.zeros(5), 0
    for u in paired_ids:
        c = case_by[u]
        if not c["retrieved"]:
            continue
        st = json.loads(sub[SYSTEMS[1]].loc[u, "support_ranks"]) if isinstance(sub[SYSTEMS[1]].loc[u, "support_ranks"], str) else {}
        supporting = sorted({r for v in st.values() for r in v})
        for v in st.values():
            for r in v:
                rank_hits[r - 1] += 1
        ctxu.append({"uid": u, "n_stated": len(st), "n_reports_supporting_any_stated_finding": len(supporting), "supporting_ranks": supporting})
        n_rep += 1
    cu = pd.DataFrame(ctxu)
    cu.to_csv(out / "g1_context_utilization_per_study.csv", index=False)
    withf = cu[cu.n_stated > 0]
    summary["context_utilization"] = {"studies_with_retrieval": int(len(cu)), "studies_with_stated_findings": int(len(withf)),
                                      "mean_supporting_reports_among_top5": float(withf.n_reports_supporting_any_stated_finding.mean()) if len(withf) else float("nan"),
                                      "support_pairs_by_rank_1_to_5": [int(x) for x in rank_hits], "share_by_rank": [float(x / max(rank_hits.sum(), 1)) for x in rank_hits],
                                      "studies_supported_only_by_rank1": int(sum(r == [1] for r in withf.supporting_ranks))}
    # normal / abnormal consistency
    conf = []
    for who, col in (("classifier", "classifier_state"), ("B0_rule_based", None), ("G1_single_agent_rag", None)):
        d = clin[SYSTEMS[0]] if who == "classifier" else clin[who]
        pred = d.classifier_state if who == "classifier" else d.report_state
        ct = pd.crosstab(d.reference_state, pred).reindex(index=["normal", "abnormal"], columns=["normal", "abnormal", "indeterminate"], fill_value=0)
        for ref_state in ct.index:
            conf.append({"source": who, "reference": ref_state, **{f"predicted_{k}": int(v) for k, v in ct.loc[ref_state].items()}})
    cdf = pd.DataFrame(conf)
    cdf.to_csv(out / "g1_normal_abnormal_confusion.csv", index=False)
    na = {}
    for who in ("classifier", "B0_rule_based", "G1_single_agent_rag"):
        c_ = cdf[cdf.source == who].set_index("reference").drop(columns="source")
        n_n, n_a = c_.loc["normal"].sum(), c_.loc["abnormal"].sum()
        na[who] = {"normal_recall": float(c_.loc["normal", "predicted_normal"] / n_n), "abnormal_recall": float(c_.loc["abnormal", "predicted_abnormal"] / n_a),
                   "accuracy": float((c_.loc["normal", "predicted_normal"] + c_.loc["abnormal", "predicted_abnormal"]) / (n_n + n_a)), "indeterminate_share": float((c_.loc["normal", "predicted_indeterminate"] + c_.loc["abnormal", "predicted_indeterminate"]) / (n_n + n_a))}
    summary["normal_abnormal_consistency"] = na
    # length / redundancy / copying
    lr = []
    for s in SYSTEMS:
        p = sub[s]
        lr.append({"system": s, "mean_words": p.words.mean(), "median_words": p.words.median(), "repeated_sentence_rate": p.repeated_sentence_rate.mean(), "copied_sentence_rate": p.copy_copied_sentence_rate.mean(),
                   "ngram8_overlap_with_retrieved": p.copy_ngram_overlap_rate.mean(), "whole_report_copy_rate": p.copy_whole_report_copy.mean(), "n": len(p), "format_ok_rate": p.format_ok.mean()})
    pd.DataFrame(lr).to_csv(out / "g1_length_redundancy.csv", index=False, float_format="%.17g")
    text_df = pd.DataFrame([{"system": s, **{k: sub[s][k].mean() for k in ("bleu1", "bleu4", "rouge_l", "meteor_exact")}, "n": len(sub[s])} for s in SYSTEMS])
    text_df.to_csv(out / "g1_text_metrics.csv", index=False, float_format="%.17g")
    # extractor validity: reference text extraction versus MeSH truth (descriptive; measurement noise ceiling)
    ev = []
    seen = set()
    for u in clin_ids:
        c = case_by[u]
        if u in seen:
            continue
        seen.add(u)
        ev.append((ext.stated(c["reference"]["combined"]), FS(c["reference"]["truth_findings"]) - {NO_FINDING}))
    evd = pd.DataFrame(per_class_prf([a for a, _ in ev], [b for _, b in ev], ABNORMAL))
    evd.to_csv(out / "g1_extractor_vs_mesh_agreement.csv", index=False, float_format="%.17g")
    tp_, fp_, fn_ = evd.tp.sum(), evd.fp.sum(), evd.fn.sum()
    summary["extractor_vs_mesh_on_reference_text"] = {**prf(tp_, fp_, fn_), "n_studies": len(ev)}
    (out / "g1_summary_metrics.json").write_text(json.dumps(summary, indent=2, default=float, allow_nan=True), encoding="utf-8")
    print(json.dumps({"n_paired": len(paired_ids), "n_clin": len(clin_ids), **{s: summary[s]["finding_any_mention"] for s in SYSTEMS}}, indent=1, default=float)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
