"""G1A step 4: three-way comparison B0_rule_based / G1A_llm_no_retrieval / G1_single_agent_rag on the exact same studies, the controlled
G1 - G1A comparison (what retrieval adds with classifier and language model held constant), retrieval-induced findings,
normalisation (over-normalisation) analysis, rare-finding retention and copying. Same extractor, metrics, populations and bootstrap
convention as G1 (study-level paired bootstrap, 1,000 resamples, seed 42). Validation data only.

    .venv\\Scripts\\python.exe -m scripts.g1a_04_evaluate
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.classification.labels import NO_FINDING
from src.generation.extraction import ABNORMAL, ReportExtractor
from src.generation.metrics import copy_stats, norm_sentence, per_class_prf, prf, sentences
from src.generation.parse import combined, parse_report
from src.generation.study_eval import FS, study_row
from src.retrieval.evaluation import bootstrap_draws
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
B0S, GAS, G1S = "B0_rule_based", "G1A_llm_no_retrieval", "G1_single_agent_rag"
SYSTEMS = (B0S, GAS, G1S)
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]     # C2 definition
N_BOOT, SEED = 1000, 42
PAIRS = ((G1S, GAS), (GAS, B0S), (G1S, B0S))


def sets_of(v):
    if isinstance(v, (list, tuple, set, frozenset)):          # in-memory per-study rows hold lists; CSV round trips hold their string form
        return set(v)
    return set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()


def main() -> int:
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    raw = {"G1": {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")},
           "G1A": {json.loads(l)["uid"]: json.loads(l) for l in open(G1A / "g1a_reports_raw.jsonl", encoding="utf-8")}}
    ext = ReportExtractor()
    rows = []
    for c in cases:
        u = c["uid"]
        gens = {B0S: {"findings": c["b0"]["findings"], "impression": c["b0"]["impression"], "format_ok": True}}
        if u in raw["G1A"]:
            gens[GAS] = parse_report(raw["G1A"][u]["text"])
        if u in raw["G1"]:
            gens[G1S] = parse_report(raw["G1"][u]["text"])
        if len(gens) < 3:
            continue
        for s, g in gens.items():
            rows.append(study_row(c, s, g, ext))
    df = pd.DataFrame(rows)
    df.to_csv(G1A / "g1a_per_study_results.csv", index=False)
    ids = sorted(set(df[df.system == B0S].uid) & set(df[df.system == GAS].uid) & set(df[df.system == G1S].uid), key=int)
    by = {s: df[(df.system == s) & df.uid.isin(ids)].set_index("uid").loc[ids] for s in SYSTEMS}
    clin_ids = [u for u in ids if bool(by[B0S].loc[u, "in_clinical"])]
    cl = {s: by[s].loc[clin_ids] for s in SYSTEMS}
    case_by = {c["uid"]: c for c in cases}

    # ---------------------------------------------------------------- G1 reproduction (B0 and G1 re-derived through the shared function)
    old = pd.read_csv(G1 / "g1_per_study_results.csv", dtype={"uid": str}).fillna("")
    new = df.fillna("")
    repro = {}
    for s in (B0S, G1S):
        a = old[old.system == s].set_index("uid").loc[ids]
        b = new[new.system == s].set_index("uid").loc[ids]
        num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
        diff = {c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) for c in num if len(a)}
        strs = ["stated", "definite", "clf_pos", "truth", "report_state", "classifier_state", "reference_state"]
        repro[s] = {"n_studies": len(a), "max_abs_numeric_difference": max(diff.values()), "numeric_columns_compared": len(num),
                    "string_columns_identical": {c: bool((a[c].astype(str) == b[c].astype(str)).all()) for c in strs}}
    repro["g1_reproduced"] = bool(all(repro[s]["max_abs_numeric_difference"] < 1e-9 and all(repro[s]["string_columns_identical"].values()) for s in (B0S, G1S)))
    (G1A / "g1_reproduction_check.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    if not repro["g1_reproduced"]:
        raise SystemExit(f"STOP: G1 did not reproduce: {repro}")

    # ---------------------------------------------------------------- arrays for the bootstrap
    cls_idx = {c: i for i, c in enumerate(ABNORMAL)}
    rare_mask = np.array([c in RARE for c in ABNORMAL])
    truth_n = np.zeros(len(ABNORMAL))
    for t in cl[B0S].truth:
        for f in sets_of(t):
            truth_n[cls_idx[f]] += 1
    macro_classes = truth_n > 0

    def arrays(d: pd.DataFrame) -> dict:
        n = len(d)
        A = d[["tp", "fp", "fn", "hall", "omit"]].to_numpy(float)
        P = d[["clf_fp", "clf_fp_mentioned", "clf_tp", "clf_tp_retained"]].to_numpy(float)
        refn, refa = (d.reference_state == "normal").to_numpy(), (d.reference_state == "abnormal").to_numpy()
        repn, repa, clfa = (d.report_state == "normal").to_numpy(), (d.report_state == "abnormal").to_numpy(), (d.classifier_state == "abnormal").to_numpy()
        C = np.zeros((n, len(ABNORMAL), 3))
        R = np.zeros((n, len(ABNORMAL), 2))
        for i, (st, tr, cp) in enumerate(zip(d.stated, d.truth, d.clf_pos)):
            st, tr, cp = sets_of(st), sets_of(tr), sets_of(cp)
            for f in ABNORMAL:
                j = cls_idx[f]
                C[i, j] = (f in st and f in tr, f in st and f not in tr, f not in st and f in tr)
                R[i, j] = (f in cp and f in tr, f in cp and f in tr and f in st)
        return {"A": A, "P": P, "N": np.column_stack([refn & repn, refn, refa & repa, refa, clfa & repn, clfa]).astype(float), "C": C, "R": R}

    arr = {s: arrays(cl[s]) for s in SYSTEMS}
    txt = {s: by[s][["rouge_l", "bleu1", "bleu4", "meteor_exact", "words"]].to_numpy(float) for s in SYSTEMS}

    def clinical_metrics(a: dict, idx: np.ndarray) -> dict:
        A, P, N, C, R = a["A"][idx], a["P"][idx], a["N"][idx].sum(0), a["C"][idx].sum(0), a["R"][idx].sum(0)
        r = prf(A[:, 0].sum(), A[:, 1].sum(), A[:, 2].sum())
        tp, fp, fn = C[:, 0], C[:, 1], C[:, 2]
        f1c = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
        rare_tp, rare_ret = R[rare_mask, 0].sum(), R[rare_mask, 1].sum()
        com_tp, com_ret = R[~rare_mask, 0].sum(), R[~rare_mask, 1].sum()
        out = {"precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "macro_f1": float(np.nanmean(f1c[macro_classes])), "hallucination_rate": A[:, 3].mean(), "omission_rate": A[:, 4].mean(),
               "mean_hallucinated": A[:, 1].mean(), "mean_omitted": A[:, 2].mean(), "clf_fp_propagation": P[:, 1].sum() / P[:, 0].sum(), "tp_retention": P[:, 3].sum() / P[:, 2].sum(),
               "normal_recall": N[0] / N[1], "abnormal_recall": N[2] / N[3], "over_normalisation_given_classifier_abnormal": N[4] / N[5],
               "rare_tp_retention": rare_ret / rare_tp if rare_tp else np.nan, "common_tp_retention": com_ret / com_tp if com_tp else np.nan}
        out["rare_minus_common_retention"] = out["rare_tp_retention"] - out["common_tp_retention"]
        return out

    def text_metrics_fn(t: np.ndarray, idx: np.ndarray) -> dict:
        m = t[idx].mean(0)
        return {"rouge_l": m[0], "bleu1": m[1], "bleu4": m[2], "meteor_exact": m[3], "mean_words": m[4]}

    dc, dp = bootstrap_draws(len(clin_ids), N_BOOT, SEED), bootstrap_draws(len(ids), N_BOOT, SEED)
    full_c, full_p = np.arange(len(clin_ids)), np.arange(len(ids))
    pt = {s: {**clinical_metrics(arr[s], full_c), **text_metrics_fn(txt[s], full_p)} for s in SYSTEMS}
    bs = {s: {k: [] for k in pt[s]} for s in SYSTEMS}
    for i in range(N_BOOT):
        for s in SYSTEMS:
            m = {**clinical_metrics(arr[s], dc[i]), **text_metrics_fn(txt[s], dp[i])}
            for k, v in m.items():
                bs[s][k].append(v)
    q = lambda v, p: float(np.nanpercentile(v, p))  # noqa: E731
    main_rows = []
    for k in pt[B0S]:
        row = {"metric": k, "n_studies": len(ids) if k in ("rouge_l", "bleu1", "bleu4", "meteor_exact", "mean_words") else len(clin_ids)}
        for s in SYSTEMS:
            row.update({f"{s}": pt[s][k], f"{s}_ci95_low": q(bs[s][k], 2.5), f"{s}_ci95_high": q(bs[s][k], 97.5)})
        main_rows.append(row)
    pd.DataFrame(main_rows).to_csv(G1A / "g1a_main_comparison.csv", index=False, float_format="%.17g")
    pair_rows = []
    for a, b in PAIRS:
        for k in pt[a]:
            d = np.array(bs[a][k]) - np.array(bs[b][k])
            pair_rows.append({"comparison": f"{a} minus {b}", "metric": k, "n_studies": len(ids) if k in ("rouge_l", "bleu1", "bleu4", "meteor_exact", "mean_words") else len(clin_ids),
                              "a": pt[a][k], "b": pt[b][k], "diff": pt[a][k] - pt[b][k], "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5), "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pdf = pd.DataFrame(pair_rows)
    pdf.to_csv(G1A / "g1a_paired_differences.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- per-finding results and rare-finding retention
    pf = []
    for s in SYSTEMS:
        gen_sets, tr_sets = [FS(sets_of(x)) for x in cl[s].stated], [FS(sets_of(x)) for x in cl[s].truth]
        res = {r["finding"]: r for r in per_class_prf(gen_sets, tr_sets, ABNORMAL)}
        for f in ABNORMAL:
            j = cls_idx[f]
            R = arr[s]["R"][:, j].sum(0)
            fpm = sum((f in sets_of(cp)) and (f not in sets_of(tr)) and (f in sets_of(st)) for st, tr, cp in zip(cl[s].stated, cl[s].truth, cl[s].clf_pos))
            fp_clf = sum((f in sets_of(cp)) and (f not in sets_of(tr)) for tr, cp in zip(cl[s].truth, cl[s].clf_pos))
            pf.append({"system": s, **res[f], "rare_c2": f in RARE, "classifier_true_positives": int(R[0]), "classifier_tp_retained": int(R[1]), "tp_retention": R[1] / R[0] if R[0] else np.nan,
                       "classifier_false_positives": int(fp_clf), "classifier_fp_mentioned": int(fpm), "fp_propagation": fpm / fp_clf if fp_clf else np.nan})
    pfd = pd.DataFrame(pf)
    pfd.to_csv(G1A / "g1a_per_finding_results.csv", index=False, float_format="%.17g")
    rare_rows = []
    for s in SYSTEMS:
        for grp, mask in (("rare (C2 definition)", rare_mask), ("other findings", ~rare_mask)):
            R = arr[s]["R"][:, mask].sum((0, 1))
            rare_rows.append({"system": s, "group": grp, "classifier_true_positives": int(R[0]), "retained": int(R[1]), "tp_retention": R[1] / R[0] if R[0] else np.nan})
    pd.DataFrame(rare_rows).to_csv(G1A / "g1a_rare_finding_retention.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- retrieval-induced findings (G1 versus G1A)
    inst = []
    for u in clin_ids:
        c = case_by[u]
        retr = [FS(r["mapped_findings"]) for r in c["retrieved"]]
        sg1, sga, tr, cp = sets_of(cl[G1S].loc[u, "stated"]), sets_of(cl[GAS].loc[u, "stated"]), sets_of(cl[G1S].loc[u, "truth"]), sets_of(cl[G1S].loc[u, "clf_pos"])
        for f in ABNORMAL:
            if (f in sg1) == (f in sga):
                continue
            inst.append({"uid": u, "anon_id": c["anon_id"], "finding": f, "direction": "added_by_retrieval_context (in G1, not in G1A)" if f in sg1 else "removed_by_retrieval_context (in G1A, not in G1)",
                         "n_retrieved_reports_containing_finding": sum(f in r for r in retr), "in_reference": f in tr, "classifier_positive": f in cp})
    ind = pd.DataFrame(inst)
    ind.to_csv(G1A / "g1a_retrieval_induced_instances.csv", index=False)
    ri = []
    add, rem = ind[ind.direction.str.startswith("added")], ind[ind.direction.str.startswith("removed")]
    n_g1 = int(sum(len(sets_of(x)) for x in cl[G1S].stated))
    n_ga = int(sum(len(sets_of(x)) for x in cl[GAS].stated))
    n_both = int(sum(len(sets_of(a) & sets_of(b)) for a, b in zip(cl[G1S].stated, cl[GAS].stated)))

    def block(d, name):
        return {"group": name, "n_findings": len(d), "appear_in_retrieved_reports": int((d.n_retrieved_reports_containing_finding > 0).sum()), "not_in_any_retrieved_report": int((d.n_retrieved_reports_containing_finding == 0).sum()),
                "match_reference": int(d.in_reference.sum()), "unsupported_by_reference": int((~d.in_reference).sum()), "also_classifier_positive": int(d.classifier_positive.sum()), "not_classifier_positive": int((~d.classifier_positive).sum()),
                "match_reference_and_in_retrieved": int((d.in_reference & (d.n_retrieved_reports_containing_finding > 0)).sum()), "unsupported_and_in_retrieved": int((~d.in_reference & (d.n_retrieved_reports_containing_finding > 0)).sum()),
                "match_reference_and_not_classifier_positive": int((d.in_reference & ~d.classifier_positive).sum()), "unsupported_and_not_classifier_positive": int((~d.in_reference & ~d.classifier_positive).sum())}

    ri += [block(add, "added: in G1, absent in G1A"), block(rem, "removed: in G1A, absent in G1")]
    ri.append({"group": "net effect of the retrieval context", "net_correct_findings (added matching reference - removed matching reference)": int(add.in_reference.sum() - rem.in_reference.sum()),
               "net_unsupported_findings (added unsupported - removed unsupported)": int((~add.in_reference).sum() - (~rem.in_reference).sum()), "stated_findings_G1": n_g1, "stated_findings_G1A": n_ga, "stated_in_both": n_both,
               "n_studies": len(clin_ids)})
    rid = pd.DataFrame(ri)
    rid.to_csv(G1A / "g1a_retrieval_induced_findings.csv", index=False)

    # ---------------------------------------------------------------- normal / abnormal analysis
    na = []
    for who in ("classifier", B0S, GAS, G1S):
        d = cl[B0S] if who == "classifier" else cl[who]
        pred = d.classifier_state if who == "classifier" else d.report_state
        for ref in ("normal", "abnormal"):
            sub = d[d.reference_state == ref]
            p = (sub.classifier_state if who == "classifier" else sub.report_state).value_counts()
            na.append({"source": who, "reference": ref, "n": len(sub), "predicted_normal": int(p.get("normal", 0)), "predicted_abnormal": int(p.get("abnormal", 0)), "predicted_indeterminate": int(p.get("indeterminate", 0))})
    nad = pd.DataFrame(na)
    nad.to_csv(G1A / "g1a_normal_abnormal.csv", index=False)
    dec = []
    for s in (GAS, G1S):
        d = cl[s]
        for cs in ("abnormal", "normal", "indeterminate"):
            sub = d[d.classifier_state == cs]
            p = sub.report_state.value_counts()
            dec.append({"system": s, "classifier_state": cs, "n": len(sub), "report_normal": int(p.get("normal", 0)), "report_abnormal": int(p.get("abnormal", 0)), "report_indeterminate": int(p.get("indeterminate", 0))})
    pd.DataFrame(dec).to_csv(G1A / "g1a_report_state_given_classifier_state.csv", index=False)

    # ---------------------------------------------------------------- copying analysis (same definitions as G1 + corpus-wide)
    corpus = pd.read_csv(R1 / "retrieval_corpus.csv", dtype={"uid": str}).fillna("")
    dfreq: dict[str, int] = {}
    corpus_norm = set()
    for t in corpus.retrieval_text:
        corpus_norm.add(norm_sentence(t))
        for s_ in {norm_sentence(x) for x in sentences(t)}:
            dfreq[s_] = dfreq.get(s_, 0) + 1
    TEMPLATE_MIN_DF = 10
    cp_rows = []
    for s in SYSTEMS:
        d = by[s]
        out = {"system": s, "n_reports": len(d)}
        long_n = copied_n = templ_n = 0
        whole_corp = whole_exact = 0
        g1_retr_copy, g1_retr_whole = [], []
        for u in ids:
            c = case_by[u]
            gen = {B0S: {"findings": c["b0"]["findings"], "impression": c["b0"]["impression"]}, GAS: parse_report(raw["G1A"][u]["text"]), G1S: parse_report(raw["G1"][u]["text"])}[s]
            text = combined(gen)
            ss = [norm_sentence(x) for x in sentences(text)]
            lg = [x for x in ss if len(x.split()) >= 6]
            long_n += len(lg)
            inc = [x for x in lg if x in dfreq]
            copied_n += len(inc)
            templ_n += sum(dfreq[x] >= TEMPLATE_MIN_DF for x in inc)
            whole_corp += int(bool(lg) and len(inc) == len(lg))
            whole_exact += int(norm_sentence(text) in corpus_norm)
            cs = copy_stats(text, [r["findings"] + " " + r["impression"] for r in c["retrieved"]])         # G1 definition against the G1 Top-5 texts
            g1_retr_copy.append(cs["copied_sentence_rate"])
            g1_retr_whole.append(float(cs["whole_report_copy"]))
        out.update({"mean_words": d.words.mean(), "long_sentences_total": long_n, "corpus_copied_sentence_rate": copied_n / max(long_n, 1), "corpus_template_sentence_share_of_long (df>=%d)" % TEMPLATE_MIN_DF: templ_n / max(long_n, 1),
                    "corpus_template_sentence_share_of_copied": templ_n / max(copied_n, 1), "whole_report_all_long_sentences_in_corpus_rate": whole_corp / len(ids), "exact_whole_report_duplicate_of_a_corpus_report_rate": whole_exact / len(ids),
                    "g1_top5_copied_sentence_rate (G1 definition)": float(np.mean(g1_retr_copy)), "g1_top5_whole_report_copy_rate (G1 definition)": float(np.mean(g1_retr_whole)),
                    "repeated_sentence_rate": d.repeated_sentence_rate.mean()})
        cp_rows.append(out)
    cpd = pd.DataFrame(cp_rows)
    cpd.to_csv(G1A / "g1a_copying_analysis.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- supplementary: degenerate-output diagnostics (post hoc, descriptive)
    import collections
    import re as _re
    diag = {}
    for s, src_raw in ((GAS, raw["G1A"]), (G1S, raw["G1"])):
        d_all = df[df.system == s].set_index("uid").loc[ids]
        ecm_by_state, tot_by_state = collections.Counter(), collections.Counter()
        for u in ids:
            st = d_all.loc[u, "classifier_state"]
            tot_by_state[st] += 1
            if "Enlarged Cardiomediastinum" in sets_of(d_all.loc[u, "stated"]) and "Enlarged Cardiomediastinum" not in sets_of(d_all.loc[u, "clf_pos"]):
                ecm_by_state[st] += 1
        texts = collections.Counter(_re.sub(r"\s+", " ", src_raw[u]["text"]).strip() for u in ids)
        top_text, top_n = texts.most_common(1)[0]
        diag[s] = {"studies_by_classifier_state": dict(tot_by_state), "enlarged_cardiomediastinum_stated_without_classifier_positive_by_classifier_state": dict(ecm_by_state),
                   "distinct_report_texts": len(texts), "most_common_text_share": top_n / len(ids), "most_common_text": top_text[:200],
                   "reports_with_leaked_reasoning_tokens": int(sum("<unused" in src_raw[u]["text"] for u in ids)), "reports_stopped_by_token_limit": int(sum(src_raw[u]["done_reason"] == "length" for u in ids)),
                   "format_ok_rate": float(d_all.format_ok.mean())}
    diag["note"] = ("post-hoc descriptive diagnostics; the G1A prompt was frozen before generation and is NOT changed. The most common G1A report is the first standard term of the prompt's term list "
                    "(Enlarged Cardiomediastinum), written for 'No Finding: positive' inputs; G1 never does this.")
    (G1A / "g1a_degenerate_output_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- supplementary: stratified by classifier state (post hoc, descriptive; same bootstrap convention)
    strat_rows = []
    pos_state = np.array([cl[B0S].loc[u, "classifier_state"] for u in clin_ids])
    for state in ("abnormal", "normal", "indeterminate"):
        pos = np.where(pos_state == state)[0]
        if len(pos) < 10:
            continue
        sub_arr = {s: {k: v[pos] for k, v in arr[s].items()} for s in SYSTEMS}
        dsub = bootstrap_draws(len(pos), N_BOOT, SEED)
        full = np.arange(len(pos))
        pts = {s: clinical_metrics(sub_arr[s], full) for s in SYSTEMS}
        bsub = {s: {k: [] for k in pts[s]} for s in SYSTEMS}
        for i in range(N_BOOT):
            for s in SYSTEMS:
                for k, v in clinical_metrics(sub_arr[s], dsub[i]).items():
                    bsub[s][k].append(v)
        for k in ("precision", "recall", "f1", "hallucination_rate", "omission_rate", "mean_hallucinated", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall"):
            d = np.array(bsub[G1S][k]) - np.array(bsub[GAS][k])
            if np.all(np.isnan(d)):
                continue
            strat_rows.append({"classifier_state": state, "n_studies": len(pos), "metric": k, B0S: pts[B0S][k], GAS: pts[GAS][k], G1S: pts[G1S][k], "g1_minus_g1a": pts[G1S][k] - pts[GAS][k],
                               "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5), "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pd.DataFrame(strat_rows).to_csv(G1A / "g1a_stratified_by_classifier_state.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- format / length / failures
    fmt = {s: {"format_ok_rate": float(by[s].format_ok.mean()), "mean_words": float(by[s].words.mean()), "median_words": float(by[s].words.median())} for s in SYSTEMS}
    summary = {"n_paired_studies": len(ids), "n_clinical_subset": len(clin_ids), "g1_reproduction": repro, "format_length": fmt, "point_estimates": pt,
               "macro_classes_with_truth": int(macro_classes.sum()), "template_min_df": TEMPLATE_MIN_DF, "systems": list(SYSTEMS)}
    (G1A / "g1a_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(main_rows)[["metric", B0S, GAS, G1S]].round(3).to_string())
    print(pdf[pdf.comparison.str.startswith(G1S + " minus " + GAS)][["metric", "diff", "ci95_low", "ci95_high"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
