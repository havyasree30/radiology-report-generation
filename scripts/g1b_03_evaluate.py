"""G1B step 3: four-system comparison B0 / G1A (no retrieval) / G1B (sham retrieval) / G1 (relevant retrieval) on identical studies, the
primary controlled comparison G1 - G1B (relevant versus structurally matched sham context; same prompt, model, decoding, classifier, extractor,
metrics), context-introduced and removed findings, copying, rare-finding retention and normal/abnormal behaviour. Same populations, extractor,
metrics and bootstrap convention as G1/G1A (study-level paired bootstrap, 1,000 resamples, seed 42). Validation data only.

    .venv\\Scripts\\python.exe -m scripts.g1b_03_evaluate
"""

from __future__ import annotations

import collections
import json
import re

import numpy as np
import pandas as pd

from src.classification.labels import NO_FINDING
from src.generation.extraction import ABNORMAL, ReportExtractor
from src.generation.metrics import copy_stats, norm_sentence, per_class_prf, prf, sentences
from src.generation.parse import combined, parse_report
from src.generation.study_eval import FS, study_row
from src.retrieval.evaluation import bootstrap_draws
from src.retrieval.metrics import jaccard
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
B0S, GAS, GBS, G1S = "B0_rule_based", "G1A_llm_no_retrieval", "G1B_llm_sham_retrieval", "G1_single_agent_rag"
SYSTEMS = (B0S, GAS, GBS, G1S)
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]     # C2 definition
N_BOOT, SEED = 1000, 42
PAIRS = ((G1S, GBS), (GBS, GAS), (GBS, B0S), (G1S, GAS))                  # the first pair is the primary controlled comparison
TEXT_KEYS = ("rouge_l", "bleu1", "bleu4", "meteor_exact", "mean_words")
TEMPLATE_MIN_DF = 10


def sets_of(v):
    if isinstance(v, (list, tuple, set, frozenset)):
        return set(v)
    return set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()


def main() -> int:
    g1_cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    gb_cases = {c["uid"]: c for c in (json.loads(l) for l in open(G1B / "g1b_cases.jsonl", encoding="utf-8"))}
    case_by = {c["uid"]: c for c in g1_cases}
    rd_raw = lambda p: {json.loads(l)["uid"]: json.loads(l) for l in open(p, encoding="utf-8")}  # noqa: E731
    raw = {"G1": rd_raw(G1 / "g1_reports_raw.jsonl"), "G1A": rd_raw(G1A / "g1a_reports_raw.jsonl"), "G1B": rd_raw(G1B / "g1b_reports_raw.jsonl")}
    ext = ReportExtractor()
    rows = []
    for c in g1_cases:
        u = c["uid"]
        if not (u in raw["G1"] and u in raw["G1A"] and u in raw["G1B"]):
            continue
        gens = {B0S: ({"findings": c["b0"]["findings"], "impression": c["b0"]["impression"], "format_ok": True}, c), GAS: (parse_report(raw["G1A"][u]["text"]), c),
                GBS: (parse_report(raw["G1B"][u]["text"]), gb_cases[u]), G1S: (parse_report(raw["G1"][u]["text"]), c)}
        for s, (g, cc) in gens.items():
            rows.append(study_row(cc, s, g, ext))
    df = pd.DataFrame(rows)
    df.to_csv(G1B / "g1b_per_study_results.csv", index=False)
    ids = sorted(set.intersection(*(set(df[df.system == s].uid) for s in SYSTEMS)), key=int)
    by = {s: df[(df.system == s) & df.uid.isin(ids)].set_index("uid").loc[ids] for s in SYSTEMS}
    clin_ids = [u for u in ids if bool(by[B0S].loc[u, "in_clinical"])]
    cl = {s: by[s].loc[clin_ids] for s in SYSTEMS}

    # ---------------------------------------------------------------- reproduction of the stored G1 (and G1A) results through the shared function
    repro = {}
    for s, path, name in ((B0S, G1 / "g1_per_study_results.csv", B0S), (G1S, G1 / "g1_per_study_results.csv", G1S), (GAS, G1A / "g1a_per_study_results.csv", GAS)):
        old = pd.read_csv(path, dtype={"uid": str}).fillna("")
        a = old[old.system == name].set_index("uid").loc[ids]
        b = df.fillna("")[df.system == s].set_index("uid").loc[ids]
        num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
        diff = {c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) for c in num}
        strs = ["stated", "definite", "clf_pos", "truth", "report_state", "classifier_state", "reference_state"]
        repro[s] = {"n_studies": len(a), "max_abs_numeric_difference": max(diff.values()), "numeric_columns_compared": len(num), "string_columns_identical": {c: bool((a[c].astype(str) == b[c].astype(str)).all()) for c in strs}}
    repro["g1_and_g1a_reproduced"] = bool(all(repro[s]["max_abs_numeric_difference"] < 1e-9 and all(repro[s]["string_columns_identical"].values()) for s in (B0S, G1S, GAS)))
    # empty-query studies: the G1B request is byte-identical to G1's; identical output indicates run-to-run determinism of the local generator
    empty = [c["uid"] for c in g1_cases if not c["retrieved"]]
    same_out = sum(raw["G1B"][u]["text"] == raw["G1"][u]["text"] for u in empty)
    repro["empty_query_studies"] = {"n": len(empty), "g1b_text_identical_to_g1_text": int(same_out)}
    (G1B / "g1_reproduction_check.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    if not repro["g1_and_g1a_reproduced"]:
        raise SystemExit(f"STOP: stored G1/G1A results did not reproduce: {repro}")

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
            for k, v in {**clinical_metrics(arr[s], dc[i]), **text_metrics_fn(txt[s], dp[i])}.items():
                bs[s][k].append(v)
    q = lambda v, p: float(np.nanpercentile(v, p))  # noqa: E731
    main_rows = []
    for k in pt[B0S]:
        row = {"metric": k, "n_studies": len(ids) if k in TEXT_KEYS else len(clin_ids)}
        for s in SYSTEMS:
            row.update({f"{s}": pt[s][k], f"{s}_ci95_low": q(bs[s][k], 2.5), f"{s}_ci95_high": q(bs[s][k], 97.5)})
        main_rows.append(row)
    pd.DataFrame(main_rows).to_csv(G1B / "g1b_main_comparison.csv", index=False, float_format="%.17g")
    pair_rows = []
    for a, b in PAIRS:
        for k in pt[a]:
            d = np.array(bs[a][k]) - np.array(bs[b][k])
            pair_rows.append({"comparison": f"{a} minus {b}", "metric": k, "n_studies": len(ids) if k in TEXT_KEYS else len(clin_ids), "a": pt[a][k], "b": pt[b][k], "diff": pt[a][k] - pt[b][k],
                              "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5), "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pdf = pd.DataFrame(pair_rows)
    pdf.to_csv(G1B / "g1b_paired_differences.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- per-finding results and rare-finding retention
    pf = []
    for s in SYSTEMS:
        gen_sets, tr_sets = [FS(sets_of(x)) for x in cl[s].stated], [FS(sets_of(x)) for x in cl[s].truth]
        res = {r["finding"]: r for r in per_class_prf(gen_sets, tr_sets, ABNORMAL)}
        for f in ABNORMAL:
            R = arr[s]["R"][:, cls_idx[f]].sum(0)
            pf.append({"system": s, **res[f], "rare_c2": f in RARE, "classifier_true_positives": int(R[0]), "classifier_tp_retained": int(R[1]), "tp_retention": R[1] / R[0] if R[0] else np.nan})
    pd.DataFrame(pf).to_csv(G1B / "g1b_per_finding_results.csv", index=False, float_format="%.17g")
    rare_rows = []
    for s in SYSTEMS:
        for grp, mask in (("rare (C2 definition)", rare_mask), ("other findings", ~rare_mask)):
            R = arr[s]["R"][:, mask].sum((0, 1))
            rare_rows.append({"system": s, "group": grp, "classifier_true_positives": int(R[0]), "retained": int(R[1]), "tp_retention": R[1] / R[0] if R[0] else np.nan})
    pd.DataFrame(rare_rows).to_csv(G1B / "g1b_rare_finding_retention.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- context-introduced findings and findings removed relative to the classifier output
    union = lambda c: set().union(*[set(r["mapped_findings"]) for r in c["retrieved"]]) if c["retrieved"] else set()  # noqa: E731
    ctx_of = {G1S: lambda u: union(case_by[u]), GBS: lambda u: union(gb_cases[u]), GAS: lambda u: set(), B0S: lambda u: set()}
    intro, rem = [], []
    for s in SYSTEMS:
        n_int = n_int_ctx = n_int_ok = n_int_bad = n_int_bad_ctx = n_int_ok_ctx = 0
        n_removed = n_rem_ok = n_rem_bad = 0
        n_stated = 0
        for u in clin_ids:
            st, tr, cp = sets_of(cl[s].loc[u, "stated"]), sets_of(cl[s].loc[u, "truth"]), sets_of(cl[s].loc[u, "clf_pos"])
            cu = ctx_of[s](u)
            n_stated += len(st)
            for f in st - cp:
                n_int += 1
                n_int_ctx += f in cu
                n_int_ok += f in tr
                n_int_bad += f not in tr
                n_int_ok_ctx += (f in tr) and (f in cu)
                n_int_bad_ctx += (f not in tr) and (f in cu)
            for f in cp - st:
                n_removed += 1
                n_rem_ok += f in tr
                n_rem_bad += f not in tr
        intro.append({"system": s, "stated_findings": n_stated, "introduced_not_classifier_positive": n_int, "introduced_in_own_context_reports": n_int_ctx, "introduced_not_in_own_context": n_int - n_int_ctx,
                      "introduced_match_reference": n_int_ok, "introduced_unsupported_by_reference": n_int_bad, "share_introduced_supported_by_reference": n_int_ok / n_int if n_int else np.nan,
                      "introduced_match_reference_and_in_context": n_int_ok_ctx, "introduced_unsupported_and_in_context": n_int_bad_ctx,
                      "classifier_positives_removed": n_removed, "removed_were_correct_(reference_positive)": n_rem_ok, "removed_were_unsupported_(classifier_false_positive)": n_rem_bad})
    pd.DataFrame(intro).to_csv(G1B / "g1b_context_introduced_and_removed_findings.csv", index=False, float_format="%.17g")

    # G1 versus G1B, finding by finding
    inst = []
    for u in clin_ids:
        sg1, sgb, tr, cp = sets_of(cl[G1S].loc[u, "stated"]), sets_of(cl[GBS].loc[u, "stated"]), sets_of(cl[G1S].loc[u, "truth"]), sets_of(cl[G1S].loc[u, "clf_pos"])
        c1, cb = ctx_of[G1S](u), ctx_of[GBS](u)
        for f in ABNORMAL:
            if (f in sg1) == (f in sgb):
                continue
            inst.append({"uid": u, "anon_id": case_by[u]["anon_id"], "finding": f, "direction": "relevant_only (in G1, not in G1B)" if f in sg1 else "sham_only (in G1B, not in G1)", "in_true_top5_context": f in c1,
                         "in_sham_context": f in cb, "in_reference": f in tr, "classifier_positive": f in cp})
    ind = pd.DataFrame(inst)
    ind.to_csv(G1B / "g1b_g1_vs_g1b_instances.csv", index=False)

    def block(d, name):
        return {"group": name, "n_findings": len(d), "match_reference": int(d.in_reference.sum()), "unsupported_by_reference": int((~d.in_reference).sum()), "share_supported_by_reference": float(d.in_reference.mean()) if len(d) else np.nan,
                "also_classifier_positive": int(d.classifier_positive.sum()), "not_classifier_positive": int((~d.classifier_positive).sum()), "in_true_top5_context": int(d.in_true_top5_context.sum()),
                "in_sham_context": int(d.in_sham_context.sum()), "match_reference_and_not_classifier_positive": int((d.in_reference & ~d.classifier_positive).sum()),
                "unsupported_and_not_classifier_positive": int((~d.in_reference & ~d.classifier_positive).sum())}

    a1, a2 = ind[ind.direction.str.startswith("relevant")], ind[ind.direction.str.startswith("sham")]
    dd = [block(a1, "stated with relevant context only (G1 not G1B)"), block(a2, "stated with sham context only (G1B not G1)")]
    dd.append({"group": "net (relevant-only minus sham-only)", "match_reference": int(a1.in_reference.sum() - a2.in_reference.sum()), "unsupported_by_reference": int((~a1.in_reference).sum() - (~a2.in_reference).sum())})
    pd.DataFrame(dd).to_csv(G1B / "g1b_g1_vs_g1b_findings.csv", index=False)

    # ---------------------------------------------------------------- context manipulation check (relevance of the supplied context to the reference and to the classifier output)
    manip = []
    for s, cases_ in ((G1S, case_by), (GBS, gb_cases)):
        jac, exact, supp_clf, share_any = [], [], [], []
        for u in clin_ids:
            c = cases_[u]
            if not c["retrieved"]:
                continue
            tr = set(c["reference"]["truth_findings"])
            js = [jaccard(tr, set(r["mapped_findings"])) if r["mapped_findings"] else 0.0 for r in c["retrieved"]]
            jac.append(np.mean(js))
            exact.append(np.mean([set(r["mapped_findings"]) == tr for r in c["retrieved"]]))
            share_any.append(np.mean([bool(tr & set(r["mapped_findings"])) for r in c["retrieved"]]))
            cp = set(c["classifier_positive_findings"])
            if cp:
                supp_clf.append(np.mean([f in union(c) for f in cp]))
        manip.append({"system": s, "n_clinical_studies_with_context": len(jac), "mean_jaccard_context_vs_reference_findings": float(np.mean(jac)), "mean_share_context_reports_with_identical_finding_set_as_reference": float(np.mean(exact)),
                      "mean_share_context_reports_sharing_a_reference_finding": float(np.mean(share_any)), "mean_share_of_classifier_positives_present_in_context": float(np.mean(supp_clf)) if supp_clf else np.nan,
                      "mean_context_words": float(np.mean([np.mean([len((r["findings"] + " " + r["impression"]).split()) for r in cases_[u]["retrieved"]]) for u in ids if cases_[u]["retrieved"]]))})
    pd.DataFrame(manip).to_csv(G1B / "g1b_context_manipulation_check.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- normal / abnormal behaviour
    na = []
    for who in ("classifier", B0S, GAS, GBS, G1S):
        d = cl[B0S] if who == "classifier" else cl[who]
        for ref in ("normal", "abnormal"):
            sub = d[d.reference_state == ref]
            p = (sub.classifier_state if who == "classifier" else sub.report_state).value_counts()
            na.append({"source": who, "reference": ref, "n": len(sub), "predicted_normal": int(p.get("normal", 0)), "predicted_abnormal": int(p.get("abnormal", 0)), "predicted_indeterminate": int(p.get("indeterminate", 0))})
    pd.DataFrame(na).to_csv(G1B / "g1b_normal_abnormal.csv", index=False)
    dec = []
    for s in (GAS, GBS, G1S):
        for cs in ("abnormal", "normal", "indeterminate"):
            sub = cl[s][cl[s].classifier_state == cs]
            p = sub.report_state.value_counts()
            dec.append({"system": s, "classifier_state": cs, "n": len(sub), "report_normal": int(p.get("normal", 0)), "report_abnormal": int(p.get("abnormal", 0)), "report_indeterminate": int(p.get("indeterminate", 0))})
    pd.DataFrame(dec).to_csv(G1B / "g1b_report_state_given_classifier_state.csv", index=False)

    # ---------------------------------------------------------------- copying (G1 definition against each system's OWN context; cross-checks; corpus-wide)
    corpus = pd.read_csv(R1 / "retrieval_corpus.csv", dtype={"uid": str}).fillna("")
    dfreq: dict[str, int] = {}
    corpus_norm = set()
    for t in corpus.retrieval_text:
        corpus_norm.add(norm_sentence(t))
        for s_ in {norm_sentence(x) for x in sentences(t)}:
            dfreq[s_] = dfreq.get(s_, 0) + 1
    ctx_texts = lambda c: [r["findings"] + " " + r["impression"] for r in c["retrieved"]]  # noqa: E731
    with_ctx = [u for u in ids if case_by[u]["retrieved"]]
    gen_text = {B0S: lambda u: combined({"findings": case_by[u]["b0"]["findings"], "impression": case_by[u]["b0"]["impression"]}), GAS: lambda u: combined(parse_report(raw["G1A"][u]["text"])),
                GBS: lambda u: combined(parse_report(raw["G1B"][u]["text"])), G1S: lambda u: combined(parse_report(raw["G1"][u]["text"]))}
    own = {G1S: case_by, GBS: gb_cases}
    cp_rows = []
    for s in SYSTEMS:
        long_n = copied_n = templ_n = whole_corp = whole_exact = 0
        own_copy, own_whole, true_copy, true_whole, sham_copy, sham_whole = [], [], [], [], [], []
        for u in ids:
            text = gen_text[s](u)
            lg = [x for x in (norm_sentence(y) for y in sentences(text)) if len(x.split()) >= 6]
            long_n += len(lg)
            inc = [x for x in lg if x in dfreq]
            copied_n += len(inc)
            templ_n += sum(dfreq[x] >= TEMPLATE_MIN_DF for x in inc)
            whole_corp += int(bool(lg) and len(inc) == len(lg))
            whole_exact += int(norm_sentence(text) in corpus_norm)
            if u in with_ctx:
                for lst_c, lst_w, texts in ((true_copy, true_whole, ctx_texts(case_by[u])), (sham_copy, sham_whole, ctx_texts(gb_cases[u]))):
                    cs = copy_stats(text, texts)
                    lst_c.append(cs["copied_sentence_rate"])
                    lst_w.append(float(cs["whole_report_copy"]))
        out = {"system": s, "n_reports": len(ids), "n_reports_with_context": len(with_ctx), "mean_words": float(by[s].words.mean()), "corpus_copied_sentence_rate": copied_n / max(long_n, 1),
               f"corpus_template_sentence_share_of_long (df>={TEMPLATE_MIN_DF})": templ_n / max(long_n, 1), "whole_report_all_long_sentences_in_corpus_rate": whole_corp / len(ids),
               "exact_whole_report_duplicate_of_a_corpus_report_rate": whole_exact / len(ids), "copied_sentence_rate_from_g1_true_top5 (studies with context)": float(np.mean(true_copy)),
               "whole_report_copy_rate_from_g1_true_top5": float(np.mean(true_whole)), "copied_sentence_rate_from_sham_context (studies with context)": float(np.mean(sham_copy)),
               "whole_report_copy_rate_from_sham_context": float(np.mean(sham_whole)), "repeated_sentence_rate": float(by[s].repeated_sentence_rate.mean())}
        out["copied_sentence_rate_from_own_context (G1 definition)"] = {G1S: out["copied_sentence_rate_from_g1_true_top5 (studies with context)"], GBS: out["copied_sentence_rate_from_sham_context (studies with context)"]}.get(s, np.nan)
        out["whole_report_copy_rate_from_own_context (G1 definition)"] = {G1S: out["whole_report_copy_rate_from_g1_true_top5"], GBS: out["whole_report_copy_rate_from_sham_context"]}.get(s, np.nan)
        cp_rows.append(out)
    pd.DataFrame(cp_rows).to_csv(G1B / "g1b_copying_analysis.csv", index=False, float_format="%.17g")
    # paired study-level bootstrap of own-context copying (G1 vs G1B, studies with context)
    cs_g1 = np.array([copy_stats(gen_text[G1S](u), ctx_texts(case_by[u]))["copied_sentence_rate"] for u in with_ctx])
    cs_gb = np.array([copy_stats(gen_text[GBS](u), ctx_texts(gb_cases[u]))["copied_sentence_rate"] for u in with_ctx])
    wh_g1 = np.array([float(copy_stats(gen_text[G1S](u), ctx_texts(case_by[u]))["whole_report_copy"]) for u in with_ctx])
    wh_gb = np.array([float(copy_stats(gen_text[GBS](u), ctx_texts(gb_cases[u]))["whole_report_copy"]) for u in with_ctx])
    dbw = bootstrap_draws(len(with_ctx), N_BOOT, SEED)
    cop = []
    for name, a, b in (("copied_sentence_rate_from_own_context", cs_g1, cs_gb), ("whole_report_copy_rate_from_own_context", wh_g1, wh_gb)):
        d = np.array([a[i].mean() - b[i].mean() for i in dbw])
        cop.append({"metric": name, "n_studies_with_context": len(with_ctx), "G1": a.mean(), "G1B": b.mean(), "diff_g1_minus_g1b": a.mean() - b.mean(), "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5), "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pd.DataFrame(cop).to_csv(G1B / "g1b_copying_paired_difference.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- stability diagnostics (descriptive)
    diag = {}
    for s, key in ((GAS, "G1A"), (GBS, "G1B"), (G1S, "G1")):
        d_all = df[df.system == s].set_index("uid").loc[ids]
        texts = collections.Counter(re.sub(r"\s+", " ", raw[key][u]["text"]).strip() for u in ids)
        top_text, top_n = texts.most_common(1)[0]
        ecm = sum(("Enlarged Cardiomediastinum" in sets_of(d_all.loc[u, "stated"])) and ("Enlarged Cardiomediastinum" not in sets_of(d_all.loc[u, "clf_pos"])) and d_all.loc[u, "classifier_state"] == "normal" for u in ids)
        diag[s] = {"distinct_report_texts": len(texts), "most_common_text_share": top_n / len(ids), "most_common_text": top_text[:200], "enlarged_cardiomediastinum_stated_without_classifier_positive_in_classifier_normal_studies": int(ecm),
                   "classifier_normal_studies": int((d_all.classifier_state == "normal").sum()), "reports_with_leaked_reasoning_tokens": int(sum("<unused" in raw[key][u]["text"] for u in ids)),
                   "reports_stopped_by_token_limit": int(sum(raw[key][u]["done_reason"] == "length" for u in ids)), "format_ok_rate": float(d_all.format_ok.mean()), "mean_output_tokens": float(np.mean([raw[key][u]["output_tokens"] for u in ids]))}
    (G1B / "g1b_stability_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- stratified by classifier state (G1 - G1B and G1B - G1A, same bootstrap convention; descriptive)
    strat_rows = []
    pos_state = np.array([cl[B0S].loc[u, "classifier_state"] for u in clin_ids])
    for state in ("abnormal", "normal", "indeterminate"):
        pos = np.where(pos_state == state)[0]
        if len(pos) < 10:
            continue
        sub_arr = {s: {k: v[pos] for k, v in arr[s].items()} for s in SYSTEMS}
        dsub = bootstrap_draws(len(pos), N_BOOT, SEED)
        pts = {s: clinical_metrics(sub_arr[s], np.arange(len(pos))) for s in SYSTEMS}
        bsub = {s: {k: [] for k in pts[s]} for s in SYSTEMS}
        for i in range(N_BOOT):
            for s in SYSTEMS:
                for k, v in clinical_metrics(sub_arr[s], dsub[i]).items():
                    bsub[s][k].append(v)
        for k in ("precision", "recall", "f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall"):
            d = np.array(bsub[G1S][k]) - np.array(bsub[GBS][k])
            if np.all(np.isnan(d)):
                continue
            strat_rows.append({"classifier_state": state, "n_studies": len(pos), "metric": k, B0S: pts[B0S][k], GAS: pts[GAS][k], GBS: pts[GBS][k], G1S: pts[G1S][k], "g1_minus_g1b": pts[G1S][k] - pts[GBS][k],
                               "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5), "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pd.DataFrame(strat_rows).to_csv(G1B / "g1b_stratified_by_classifier_state.csv", index=False, float_format="%.17g")

    fmt = {s: {"format_ok_rate": float(by[s].format_ok.mean()), "mean_words": float(by[s].words.mean()), "median_words": float(by[s].words.median())} for s in SYSTEMS}
    summary = {"n_paired_studies": len(ids), "n_clinical_subset": len(clin_ids), "n_studies_with_context": len(with_ctx), "n_empty_query_studies": len(ids) - len(with_ctx), "reproduction": repro, "format_length": fmt,
               "point_estimates": pt, "macro_classes_with_truth": int(macro_classes.sum()), "template_min_df": TEMPLATE_MIN_DF, "systems": list(SYSTEMS), "primary_comparison": f"{G1S} minus {GBS}"}
    (G1B / "g1b_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 240)
    print(pd.DataFrame(main_rows)[["metric", B0S, GAS, GBS, G1S]].round(3).to_string())
    print(pdf[pdf.comparison == f"{G1S} minus {GBS}"][["metric", "diff", "ci95_low", "ci95_high", "excludes_zero"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
