"""G2 step 4: evaluation of the structured multi-agent RAG (G2) against the single-agent RAG (G1) on identical validation studies; B0, G1A, G1B and the
Agent 2 draft (before critique) are secondary context. Same populations, extractor, metrics, copying definitions and bootstrap convention as G1
(study-level paired bootstrap, 1,000 resamples, seed 42). Agent 1 / Agent 2 / Agent 3 contribution analyses, retrieval-only findings, leakage checks,
latency. Validation data only.

    .venv\\Scripts\\python.exe -m scripts.g2_04_evaluate
"""

from __future__ import annotations

import collections
import json
import re

import numpy as np
import pandas as pd

from src.generation.extraction import ABNORMAL, ReportExtractor
from src.generation.g2_agents import (PROMPTS, build_agent1_message, build_agent2_message, build_agent3_message, evidence_for_downstream, message_sha256, shares_ngram)
from src.generation.g2_pipeline import AGENTS, case_inputs
from src.generation.metrics import copy_stats, norm_sentence, per_class_prf, prf, sentences
from src.generation.parse import combined, parse_report
from src.generation.study_eval import FS, study_row
from src.retrieval.evaluation import bootstrap_draws
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = PROJECT_ROOT / "results/report_generation/experiments/g1a_no_retrieval"
G1B = PROJECT_ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
G2 = PROJECT_ROOT / "results/report_generation/experiments/g2_multi_agent"
R1 = PROJECT_ROOT / "results/retrieval/experiments/r1_baseline"
B0S, GAS, GBS, G1S, G2DS, G2S = "B0_rule_based", "G1A_llm_no_retrieval", "G1B_llm_sham_retrieval", "G1_single_agent_rag", "G2_agent2_draft_pre_critic", "G2_multi_agent_rag"
SYSTEMS = (B0S, GAS, GBS, G1S, G2DS, G2S)
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]     # C2 definition
N_BOOT, SEED, TEMPLATE_MIN_DF = 1000, 42, 10
PAIRS = ((G2S, G1S), (G2DS, G1S), (G2S, G2DS), (G2S, B0S), (G2S, GBS), (G2S, GAS))          # the first pair is the primary comparison
TEXT_KEYS = ("rouge_l", "bleu1", "bleu4", "meteor_exact", "mean_words")


def sets_of(v):
    if isinstance(v, (list, tuple, set, frozenset)):
        return set(v)
    return set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()


def main() -> int:
    g1_cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    g2_cases = {c["uid"]: c for c in (json.loads(l) for l in open(G2 / "g2_cases.jsonl", encoding="utf-8"))}
    gb_cases = {c["uid"]: c for c in (json.loads(l) for l in open(G1B / "g1b_cases.jsonl", encoding="utf-8"))}
    case_by = {c["uid"]: c for c in g1_cases}
    rd_raw = lambda p: {json.loads(l)["uid"]: json.loads(l) for l in open(p, encoding="utf-8")}  # noqa: E731
    raw = {"G1": rd_raw(G1 / "g1_reports_raw.jsonl"), "G1A": rd_raw(G1A / "g1a_reports_raw.jsonl"), "G1B": rd_raw(G1B / "g1b_reports_raw.jsonl")}
    ag = {c["uid"]: json.loads((G2 / "agent_outputs" / f"{c['uid']}.json").read_text(encoding="utf-8")) for c in g1_cases if (G2 / "agent_outputs" / f"{c['uid']}.json").exists()}
    ext = ReportExtractor()
    rows = []
    for c in g1_cases:
        u = c["uid"]
        if u not in ag:
            continue
        gens = {B0S: ({"findings": c["b0"]["findings"], "impression": c["b0"]["impression"], "format_ok": True}, c), GAS: (parse_report(raw["G1A"][u]["text"]), c), GBS: (parse_report(raw["G1B"][u]["text"]), gb_cases[u]),
                G1S: (parse_report(raw["G1"][u]["text"]), c), G2DS: (ag[u]["agent2_draft"], c), G2S: (dict(ag[u]["final_report"], format_ok=bool(ag[u]["final_report"]["findings"] and ag[u]["final_report"]["impression"])), c)}
        for s, (g, cc) in gens.items():
            rows.append(study_row(cc, s, g, ext))
    df = pd.DataFrame(rows)
    df.to_csv(G2 / "g2_per_study_results.csv", index=False)
    ids = sorted(set.intersection(*(set(df[df.system == s].uid) for s in SYSTEMS)), key=int)
    by = {s: df[(df.system == s) & df.uid.isin(ids)].set_index("uid").loc[ids] for s in SYSTEMS}
    clin_ids = [u for u in ids if bool(by[B0S].loc[u, "in_clinical"])]
    cl = {s: by[s].loc[clin_ids] for s in SYSTEMS}

    # ---------------------------------------------------------------- reproduction of the stored B0 / G1 / G1A / G1B results through the shared function
    repro = {}
    for s, path in ((B0S, G1 / "g1_per_study_results.csv"), (G1S, G1 / "g1_per_study_results.csv"), (GAS, G1A / "g1a_per_study_results.csv"), (GBS, G1B / "g1b_per_study_results.csv")):
        old = pd.read_csv(path, dtype={"uid": str}).fillna("")
        a = old[old.system == s].set_index("uid").loc[ids]
        b = df.fillna("")[df.system == s].set_index("uid").loc[ids]
        num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
        diff = {c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) for c in num}
        strs = ["stated", "definite", "clf_pos", "truth", "report_state", "classifier_state", "reference_state"]
        repro[s] = {"n_studies": len(a), "max_abs_numeric_difference": max(diff.values()), "numeric_columns_compared": len(num), "string_columns_identical": {c: bool((a[c].astype(str) == b[c].astype(str)).all()) for c in strs}}
    repro["earlier_results_reproduced"] = bool(all(repro[s]["max_abs_numeric_difference"] < 1e-9 and all(repro[s]["string_columns_identical"].values()) for s in (B0S, G1S, GAS, GBS)))
    (G2 / "g1_reproduction_check.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    if not repro["earlier_results_reproduced"]:
        raise SystemExit(f"STOP: stored earlier results did not reproduce: {repro}")

    # ---------------------------------------------------------------- arrays for the bootstrap (same definitions as G1)
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
    pd.DataFrame(main_rows).to_csv(G2 / "g2_main_comparison.csv", index=False, float_format="%.17g")
    pair_rows = []
    for a, b in PAIRS:
        for k in pt[a]:
            d = np.array(bs[a][k]) - np.array(bs[b][k])
            pair_rows.append({"comparison": f"{a} minus {b}", "metric": k, "n_studies": len(ids) if k in TEXT_KEYS else len(clin_ids), "a": pt[a][k], "b": pt[b][k], "diff": pt[a][k] - pt[b][k],
                              "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5), "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pdf = pd.DataFrame(pair_rows)
    pdf.to_csv(G2 / "g2_paired_differences.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- post hoc sensitivity: studies whose Agent 1 JSON was parsed (Agent 1 hit its output budget in the others)
    ok_pos = np.array([i for i, u in enumerate(clin_ids) if "agent1_unparseable" not in ag[u]["agent1_validation_flags"]])
    dsens = bootstrap_draws(len(ok_pos), N_BOOT, SEED)
    sens_pts = {s: clinical_metrics({k: v[ok_pos] for k, v in arr[s].items()}, np.arange(len(ok_pos))) for s in (G1S, G2S)}
    sens_bs = {s: {k: [] for k in sens_pts[s]} for s in (G1S, G2S)}
    for i in range(N_BOOT):
        for s in (G1S, G2S):
            for k, v in clinical_metrics({kk: vv[ok_pos] for kk, vv in arr[s].items()}, dsens[i]).items():
                sens_bs[s][k].append(v)
    sens_rows = []
    for k in ("precision", "recall", "f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall"):
        d = np.array(sens_bs[G2S][k]) - np.array(sens_bs[G1S][k])
        sens_rows.append({"metric": k, "n_clinical_studies": len(ok_pos), "n_excluded_agent1_unparseable": len(clin_ids) - len(ok_pos), G1S: sens_pts[G1S][k], G2S: sens_pts[G2S][k], "diff": sens_pts[G2S][k] - sens_pts[G1S][k],
                          "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5), "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pd.DataFrame(sens_rows).to_csv(G2 / "g2_sensitivity_agent1_parsed_only.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- per-finding results and rare-finding retention
    pf = []
    for s in SYSTEMS:
        gen_sets, tr_sets = [FS(sets_of(x)) for x in cl[s].stated], [FS(sets_of(x)) for x in cl[s].truth]
        res = {r["finding"]: r for r in per_class_prf(gen_sets, tr_sets, ABNORMAL)}
        for f in ABNORMAL:
            R = arr[s]["R"][:, cls_idx[f]].sum(0)
            pf.append({"system": s, **res[f], "rare_c2": f in RARE, "classifier_true_positives": int(R[0]), "classifier_tp_retained": int(R[1]), "tp_retention": R[1] / R[0] if R[0] else np.nan})
    pd.DataFrame(pf).to_csv(G2 / "g2_per_finding_results.csv", index=False, float_format="%.17g")
    rare_rows = []
    for s in SYSTEMS:
        for grp, mask in (("rare (C2 definition)", rare_mask), ("other findings", ~rare_mask)):
            R = arr[s]["R"][:, mask].sum((0, 1))
            rare_rows.append({"system": s, "group": grp, "classifier_true_positives": int(R[0]), "retained": int(R[1]), "tp_retention": R[1] / R[0] if R[0] else np.nan})
    pd.DataFrame(rare_rows).to_csv(G2 / "g2_rare_finding_retention.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- Agent 1: evidence quality (clinical subset), candidates, citation agreement with the IU labels of the retrieved reports
    st_rows, cite_n, cite_ok, labelled_n, labelled_cited, det_agree, det_n = [], 0, 0, 0, 0, 0, 0
    flag_count = collections.Counter()
    cand_rows = []
    for u in ids:
        flag_count.update(ag[u]["agent1_validation_flags"])
        ev, gcase = ag[u]["evidence"], case_by[u]
        retr_sets = [set(r["mapped_findings"]) for r in gcase["retrieved"]]
        truth = sets_of(cl[B0S].loc[u, "truth"]) if u in cl[B0S].index else None
        for it in ev["classifier_findings"]:
            f, ranks = it["finding"], it["supporting_retrieved_report_ranks"]
            cite_n += len(ranks)
            cite_ok += sum(1 for r in ranks if f in retr_sets[r - 1])
            lab = [r for r in range(1, len(retr_sets) + 1) if f in retr_sets[r - 1]]
            labelled_n += len(lab)
            labelled_cited += len([r for r in lab if r in ranks])
            det_n += 1
            det_agree += int(it["support_status"] == ("supported" if len(lab) >= 2 else "partially_supported" if len(lab) == 1 else "unsupported"))
            if truth is not None:
                st_rows.append({"uid": u, "finding": f, "status": it["support_status"], "agent_stated_status": it["agent_stated_status"], "in_reference": f in truth, "count": it["retrieval_support_count"],
                                "probability": it["classifier_probability"]})
        for cnd in ev["retrieval_only_candidates"]:
            cand_rows.append({"uid": u, "finding": cnd["finding"], "count": cnd["retrieval_support_count"], "in_clinical": truth is not None, "in_reference": (cnd["finding"] in truth) if truth is not None else None,
                              "promoted_to_draft": cnd["finding"] in sets_of(by[G2DS].loc[u, "stated"]), "survives_critic": cnd["finding"] in sets_of(by[G2S].loc[u, "stated"])})
    sdf, cdf = pd.DataFrame(st_rows), pd.DataFrame(cand_rows)
    sdf.to_csv(G2 / "g2_agent1_classifier_finding_status.csv", index=False)
    cdf.to_csv(G2 / "g2_agent1_retrieval_only_candidates.csv", index=False)
    a1 = []
    for col in ("status", "agent_stated_status"):
        for stt in ("supported", "partially_supported", "unsupported"):
            sub = sdf[sdf[col] == stt]
            a1.append({"status_source": "validated (used downstream)" if col == "status" else "model-stated (before validation)", "support_status": stt, "n_classifier_positives": len(sub), "n_reference_positive": int(sub.in_reference.sum()),
                       "share_matching_reference": float(sub.in_reference.mean()) if len(sub) else np.nan})
    a1.append({"status_source": "all classifier positives (clinical subset)", "support_status": "all", "n_classifier_positives": len(sdf), "n_reference_positive": int(sdf.in_reference.sum()), "share_matching_reference": float(sdf.in_reference.mean())})
    pd.DataFrame(a1).to_csv(G2 / "g2_agent1_status_vs_reference.csv", index=False, float_format="%.17g")
    clin_c = cdf[cdf.in_clinical]
    a1s = {"n_studies": len(ids), "n_classifier_positives_all_studies": det_n, "validation_flag_counts": dict(flag_count), "n_retrieval_only_candidates_all_studies": len(cdf), "n_retrieval_only_candidates_clinical": len(clin_c),
           "retrieval_only_candidates_reference_supported": int(clin_c.in_reference.sum()), "retrieval_only_candidates_unsupported": int((~clin_c.in_reference.astype(bool)).sum()),
           "candidate_ge_promotion_threshold_clinical": int((clin_c["count"] >= 3).sum()), "candidate_ge_threshold_reference_supported": int(clin_c[clin_c["count"] >= 3].in_reference.sum()),
           "cited_ranks_total": cite_n, "cited_ranks_whose_report_carries_the_finding_in_iu_labels": cite_ok, "citation_agreement_with_iu_labels": cite_ok / cite_n if cite_n else np.nan,
           "iu_labelled_supporting_reports_total": labelled_n, "iu_labelled_supporting_reports_cited": labelled_cited, "recall_of_iu_labelled_support": labelled_cited / labelled_n if labelled_n else np.nan,
           "status_agreement_with_deterministic_g1_support_table": det_agree / det_n if det_n else np.nan,
           "note": "IU labels of the retrieved reports are an incomplete MeSH-derived approximation of what those reports state; agreement is descriptive"}
    # support by count for classifier positives (a diagnostic of how informative the count is)
    byc = sdf.groupby("count").agg(n=("in_reference", "size"), reference_positive=("in_reference", "sum")).reset_index()
    byc["share_matching_reference"] = byc.reference_positive / byc.n
    byc.to_csv(G2 / "g2_agent1_support_count_vs_reference.csv", index=False, float_format="%.17g")
    (G2 / "g2_agent1_summary.json").write_text(json.dumps(a1s, indent=2, default=float), encoding="utf-8")

    # ---------------------------------------------------------------- Agent 3 interventions (critic): draft versus final
    act = collections.Counter(ag[u]["agent3"]["action"] for u in ids)
    iss = collections.Counter(i for u in ids for i in ag[u]["agent3"]["issues"])
    flg = collections.Counter(f for u in ids for f in ag[u]["agent3"]["flags"])
    modified = [u for u in ids if norm_sentence(combined(ag[u]["agent2_draft"])) != norm_sentence(combined(ag[u]["final_report"]))]
    strip_labels = lambda t: norm_sentence(re.sub(r"(?i)(findings|impression)\s*:", " ", t))  # noqa: E731
    modified_sub = [u for u in ids if strip_labels(combined(ag[u]["agent2_draft"])) != strip_labels(combined(ag[u]["final_report"]))]
    inst = []
    for u in ids:
        sd, sf = sets_of(by[G2DS].loc[u, "stated"]), sets_of(by[G2S].loc[u, "stated"])
        truth = sets_of(by[G2S].loc[u, "truth"]) if u in cl[B0S].index else None
        cp = sets_of(by[G2S].loc[u, "clf_pos"])
        for f in sorted(sd - sf):
            inst.append({"uid": u, "finding": f, "direction": "removed_by_critic", "in_clinical": truth is not None, "in_reference": (f in truth) if truth is not None else None, "classifier_positive": f in cp})
        for f in sorted(sf - sd):
            inst.append({"uid": u, "finding": f, "direction": "added_by_critic", "in_clinical": truth is not None, "in_reference": (f in truth) if truth is not None else None, "classifier_positive": f in cp})
    idf = pd.DataFrame(inst, columns=["uid", "finding", "direction", "in_clinical", "in_reference", "classifier_positive"])
    idf.to_csv(G2 / "g2_agent3_finding_changes.csv", index=False)
    ci = idf[idf.in_clinical.astype(bool)] if len(idf) else idf
    rem, add = (ci[ci.direction == "removed_by_critic"], ci[ci.direction == "added_by_critic"]) if len(ci) else (ci, ci)
    a3 = {"n_studies": len(ids), "actions": dict(act), "approved_unchanged_pct": 100 * (len(ids) - len(modified)) / len(ids), "modified_pct": 100 * len(modified) / len(ids), "n_modified": len(modified), "issue_counts": dict(iss), "flag_counts": dict(flg),
          "approve_model_copy_identical_to_draft_pct": 100 * float(np.mean([bool(ag[u]["agent3"]["model_copy_identical_to_draft"]) for u in ids if ag[u]["agent3"]["action"] == "approve"])) if act.get("approve") else np.nan,
          "findings_removed_total_all_studies": int((idf.direction == "removed_by_critic").sum()) if len(idf) else 0, "findings_added_total_all_studies": int((idf.direction == "added_by_critic").sum()) if len(idf) else 0,
          "clinical_subset": {"findings_removed": len(rem), "correct_removals_(not_in_reference)": int((~rem.in_reference.astype(bool)).sum()) if len(rem) else 0, "incorrect_removals_(in_reference)": int(rem.in_reference.astype(bool).sum()) if len(rem) else 0,
                              "findings_added": len(add), "correct_additions_(in_reference)": int(add.in_reference.astype(bool).sum()) if len(add) else 0, "unsupported_additions_(not_in_reference)": int((~add.in_reference.astype(bool)).sum()) if len(add) else 0,
                              "added_were_classifier_positive": int(add.classifier_positive.astype(bool).sum()) if len(add) else 0},
          "studies_with_removed_or_added_finding": int(idf.uid.nunique()) if len(idf) else 0, "modified_beyond_section_labels_n": len(modified_sub), "modified_beyond_section_labels_pct": 100 * len(modified_sub) / len(ids),
          "revise_actions": int(act.get("revise", 0))}
    (G2 / "g2_agent3_summary.json").write_text(json.dumps(a3, indent=2, default=float), encoding="utf-8")

    # ---------------------------------------------------------------- retrieval-only findings: G1 versus G2 (clinical subset)
    def intro_stats(system: str, retr_sets_of=None) -> dict:
        n = ok = ctxn = ctxok = 0
        for u in clin_ids:
            st, tr, cp = sets_of(cl[system].loc[u, "stated"]), sets_of(cl[system].loc[u, "truth"]), sets_of(cl[system].loc[u, "clf_pos"])
            union = set().union(*[set(r["mapped_findings"]) for r in case_by[u]["retrieved"]]) if case_by[u]["retrieved"] else set()
            for f in st - cp:
                n += 1
                ok += f in tr
                if f in union:
                    ctxn += 1
                    ctxok += f in tr
        return {"system": system, "introduced_findings_(stated, not classifier positive)": n, "introduced_reference_supported": ok, "introduced_unsupported": n - ok,
                "share_supported": ok / n if n else np.nan, "introduced_also_in_retrieved_reports_(retrieval-derived)": ctxn, "retrieval_derived_reference_supported": ctxok,
                "retrieval_derived_unsupported": ctxn - ctxok, "retrieval_derived_share_supported": ctxok / ctxn if ctxn else np.nan, "stated_findings_total": int(sum(len(sets_of(x)) for x in cl[system].stated))}
    ro = pd.DataFrame([intro_stats(s) for s in (B0S, GAS, G1S, G2DS, G2S)])
    ro.to_csv(G2 / "g2_introduced_findings.csv", index=False, float_format="%.17g")
    stage = []
    cc = clin_c.copy()
    cc["in_reference"] = cc.in_reference.astype(bool)
    for name, sub in (("proposed by Agent 1 (retrieval_only candidates)", cc), ("promoted into the Agent 2 draft", cc[cc.promoted_to_draft]), ("surviving Agent 3 (in final G2 report)", cc[cc.survives_critic])):
        stage.append({"stage": name, "n_candidates": len(sub), "reference_supported": int(sub.in_reference.sum()), "reference_unsupported": int((~sub.in_reference).sum()), "share_supported": float(sub.in_reference.mean()) if len(sub) else np.nan})
    pd.DataFrame(stage).to_csv(G2 / "g2_retrieval_only_funnel.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- normal / abnormal behaviour
    na = []
    for who in ("classifier", B0S, G1S, G2DS, G2S):
        d = cl[B0S] if who == "classifier" else cl[who]
        for ref in ("normal", "abnormal"):
            sub = d[d.reference_state == ref]
            p = (sub.classifier_state if who == "classifier" else sub.report_state).value_counts()
            na.append({"source": who, "reference": ref, "n": len(sub), "predicted_normal": int(p.get("normal", 0)), "predicted_abnormal": int(p.get("abnormal", 0)), "predicted_indeterminate": int(p.get("indeterminate", 0))})
    pd.DataFrame(na).to_csv(G2 / "g2_normal_abnormal.csv", index=False)
    dec = []
    for s in (G1S, G2DS, G2S):
        for cs in ("abnormal", "normal", "indeterminate"):
            sub = cl[s][cl[s].classifier_state == cs]
            p = sub.report_state.value_counts()
            dec.append({"system": s, "classifier_state": cs, "n": len(sub), "report_normal": int(p.get("normal", 0)), "report_abnormal": int(p.get("abnormal", 0)), "report_indeterminate": int(p.get("indeterminate", 0))})
    pd.DataFrame(dec).to_csv(G2 / "g2_report_state_given_classifier_state.csv", index=False)
    # empty-classifier-output studies (all 547 primary set): explicit handling
    emp = [u for u in ids if not g2_cases[u]["retrieved"]]
    emp_rows = [{"system": s, "n": len(emp), **{f"report_{k}": int((by[s].loc[emp, "report_state"] == k).sum()) for k in ("normal", "abnormal", "indeterminate")}} for s in (B0S, G1S, G2DS, G2S)]
    pd.DataFrame(emp_rows).to_csv(G2 / "g2_empty_classifier_output_studies.csv", index=False)

    # ---------------------------------------------------------------- copying (G1 definitions against the true Top-5) and corpus-wide
    corpus = pd.read_csv(R1 / "retrieval_corpus.csv", dtype={"uid": str}).fillna("")
    dfreq: dict[str, int] = {}
    corpus_norm = set()
    for t in corpus.retrieval_text:
        corpus_norm.add(norm_sentence(t))
        for s_ in {norm_sentence(x) for x in sentences(t)}:
            dfreq[s_] = dfreq.get(s_, 0) + 1
    with_ctx = [u for u in ids if case_by[u]["retrieved"]]
    gen_text = {B0S: lambda u: combined({"findings": case_by[u]["b0"]["findings"], "impression": case_by[u]["b0"]["impression"]}), GAS: lambda u: combined(parse_report(raw["G1A"][u]["text"])),
                GBS: lambda u: combined(parse_report(raw["G1B"][u]["text"])), G1S: lambda u: combined(parse_report(raw["G1"][u]["text"])), G2DS: lambda u: combined(ag[u]["agent2_draft"]), G2S: lambda u: combined(ag[u]["final_report"])}
    ctx_texts = lambda u: [r["findings"] + " " + r["impression"] for r in case_by[u]["retrieved"]]  # noqa: E731
    cp_rows, cs_all = [], {}
    for s in SYSTEMS:
        long_n = copied_n = templ_n = whole_corp = whole_exact = short_n = sent_n = 0
        cs_s, wh_s = [], []
        for u in ids:
            text = gen_text[s](u)
            lg = [x for x in (norm_sentence(y) for y in sentences(text)) if len(x.split()) >= 6]
            long_n += len(lg)
            inc = [x for x in lg if x in dfreq]
            copied_n += len(inc)
            templ_n += sum(dfreq[x] >= TEMPLATE_MIN_DF for x in inc)
            whole_corp += int(bool(lg) and len(inc) == len(lg))
            whole_exact += int(norm_sentence(text) in corpus_norm)
            sl = [len(norm_sentence(y).split()) for y in sentences(text)]
            short_n += sum(x < 6 for x in sl)
            sent_n += len(sl)
        for u in with_ctx:
            st_ = copy_stats(gen_text[s](u), ctx_texts(u))
            cs_s.append(st_["copied_sentence_rate"])
            wh_s.append(float(st_["whole_report_copy"]))
        cs_all[s] = (np.array(cs_s), np.array(wh_s))
        cp_rows.append({"system": s, "n_reports": len(ids), "n_reports_with_context": len(with_ctx), "mean_words": float(by[s].words.mean()), "copied_sentence_rate_from_top5 (G1 definition)": float(np.mean(cs_s)),
                        "whole_report_copy_rate_from_top5 (G1 definition)": float(np.mean(wh_s)), "repeated_sentence_rate": float(by[s].repeated_sentence_rate.mean()), "share_of_sentences_under_6_words_(cannot_register_as_copied)": short_n / max(sent_n, 1), "corpus_copied_sentence_rate": copied_n / max(long_n, 1),
                        f"corpus_template_sentence_share_of_long (df>={TEMPLATE_MIN_DF})": templ_n / max(long_n, 1), "whole_report_all_long_sentences_in_corpus_rate": whole_corp / len(ids),
                        "exact_whole_report_duplicate_of_a_corpus_report_rate": whole_exact / len(ids)})
    pd.DataFrame(cp_rows).to_csv(G2 / "g2_copying_analysis.csv", index=False, float_format="%.17g")
    dbw = bootstrap_draws(len(with_ctx), N_BOOT, SEED)
    rep_arr = {s: by[s].loc[with_ctx, "repeated_sentence_rate"].to_numpy(float) for s in SYSTEMS}
    cop = []
    for a, b in ((G2S, G1S), (G2DS, G1S), (G2S, G2DS)):
        for name, xa, xb in (("copied_sentence_rate_from_top5", cs_all[a][0], cs_all[b][0]), ("whole_report_copy_rate_from_top5", cs_all[a][1], cs_all[b][1]), ("repeated_sentence_rate", rep_arr[a], rep_arr[b])):
            d = np.array([xa[i].mean() - xb[i].mean() for i in dbw])
            cop.append({"comparison": f"{a} minus {b}", "metric": name, "n_studies_with_context": len(with_ctx), "a": xa.mean(), "b": xb.mean(), "diff": xa.mean() - xb.mean(), "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5),
                        "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pd.DataFrame(cop).to_csv(G2 / "g2_copying_paired_difference.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- latency and compute
    sec = {a: np.array([ag[u]["seconds"][a] for u in ids if a in ag[u]["seconds"]], float) for a in AGENTS}
    tok_out = {a: np.array([ag[u]["output_tokens"][a] for u in ids if a in ag[u]["output_tokens"]], float) for a in AGENTS}
    tok_in = {a: np.array([ag[u]["prompt_tokens"][a] for u in ids if a in ag[u]["prompt_tokens"]], float) for a in AGENTS}
    per_study = np.array([sum(ag[u]["seconds"].values()) for u in ids], float)
    run = json.loads((G2 / "generation_run_log.json").read_text(encoding="utf-8"))
    g1_sec = np.array([raw["G1"][u]["wall_seconds"] for u in ids], float)
    g1b_sec = np.array([raw["G1B"][u]["wall_seconds"] for u in ids], float)
    lat = {"n_studies": len(ids), "agents": {a: {"n_calls": len(sec[a]), "mean_seconds": float(sec[a].mean()), "median_seconds": float(np.median(sec[a])), "mean_prompt_tokens": float(tok_in[a].mean()), "mean_output_tokens": float(tok_out[a].mean()),
                                                 "total_output_tokens": int(tok_out[a].sum())} for a in AGENTS},
           "g2_mean_seconds_per_study": float(per_study.mean()), "g2_median_seconds_per_study": float(np.median(per_study)), "g2_total_recorded_call_seconds": float(per_study.sum()), "g2_wall_seconds_this_run": run["wall_seconds_this_run"],
           "g1_mean_seconds_per_study": float(g1_sec.mean()), "g1_total_recorded_call_seconds": float(g1_sec.sum()), "g1b_mean_seconds_per_study": float(g1b_sec.mean()), "g2_over_g1_time_ratio": float(per_study.mean() / g1_sec.mean()),
           "g2_calls_per_study_mean": float(np.mean([len(ag[u]["seconds"]) for u in ids])), "gpu_memory_used_mib_peak_sampled": run["gpu_memory_used_mib_peak_sampled"], "gpu_memory_samples": run["gpu_memory_samples"],
           "note": "seconds are the wall time of each local Ollama call as recorded at generation time (cached responses keep their original timing); G1 ran earlier on the same machine; GPU memory is the nvidia-smi total used memory sampled every 5 studies"}
    (G2 / "g2_latency_compute.json").write_text(json.dumps(lat, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- leakage checks and message hashes
    leak = {"n_studies": len(ids), "agent2_message_shares_6gram_with_retrieved_text": 0, "agent3_message_shares_6gram_with_retrieved_text": 0, "agent1_message_reference_sentences_outside_retrieved_text": 0,
            "agent2_message_reference_sentences": 0, "agent3_message_reference_sentences": 0, "messages_rebuilt_identically": 0, "messages_checked": 0, "agent1_hash_equals_case_file": 0, "retrieved_study_ids_in_messages": 0,
            "reference_report_in_any_case_or_agent_output_input_field": 0}
    sh_rows = []
    for u in ids:
        c2, c1, a = g2_cases[u], case_by[u], ag[u]
        pos, nf, retr = case_inputs(c2)
        texts = [r["findings"] + " " + r["impression"] for r in c2["retrieved"]]
        refs = [norm_sentence(s) for s in sentences(c1["reference"]["combined"]) if len(norm_sentence(s).split()) >= 6]
        m = a["messages"]
        rebuilt = {AGENTS[1]: build_agent2_message(pos, nf, a["evidence"]), AGENTS[2]: build_agent3_message(pos, nf, a["evidence"], a["agent2_draft"])}
        if retr:
            rebuilt[AGENTS[0]] = build_agent1_message(pos, nf, retr)
            leak["agent1_hash_equals_case_file"] += int(__import__("hashlib").sha256(m[AGENTS[0]].encode()).hexdigest() == c2["agent1_message_sha256"])
            nm = norm_sentence(m[AGENTS[0]])
            ctx = norm_sentence(" ".join(texts))
            leak["agent1_message_reference_sentences_outside_retrieved_text"] += sum(s in nm and s not in ctx for s in refs)
        for ag_name, msg in m.items():
            leak["messages_checked"] += 1
            leak["messages_rebuilt_identically"] += int(msg == rebuilt[ag_name])
            leak["retrieved_study_ids_in_messages"] += sum(str(i) in msg.replace("\n", " ").split() for i in c2["retrieved_study_ids"])
            sh_rows.append({"uid": u, "agent": ag_name, "message_sha256": a["messages_sha256"][ag_name], "recomputed_message_sha256": message_sha256(PROMPTS[ag_name], msg), "request_hash": a["request_hashes"][ag_name]})
        leak["agent2_message_shares_6gram_with_retrieved_text"] += int(shares_ngram(m[AGENTS[1]], texts, 6))
        leak["agent3_message_shares_6gram_with_retrieved_text"] += int(shares_ngram(m[AGENTS[2]], texts, 6))
        nm2, nm3 = norm_sentence(m[AGENTS[1]]), norm_sentence(m[AGENTS[2]])
        leak["agent2_message_reference_sentences"] += sum(s in nm2 for s in refs)
        leak["agent3_message_reference_sentences"] += sum(s in nm3 for s in refs)
    shd = pd.DataFrame(sh_rows)
    shd["hash_matches_recomputed"] = shd.message_sha256 == shd.recomputed_message_sha256
    shd.to_csv(G2 / "g2_sent_messages_sha256.csv", index=False)
    leak["all_stored_hashes_match_recomputed"] = bool(shd.hash_matches_recomputed.all())
    leak["pass"] = bool(leak["agent2_message_shares_6gram_with_retrieved_text"] == 0 and leak["agent3_message_shares_6gram_with_retrieved_text"] == 0 and leak["agent1_message_reference_sentences_outside_retrieved_text"] == 0
                        and leak["agent2_message_reference_sentences"] == 0 and leak["agent3_message_reference_sentences"] == 0 and leak["messages_rebuilt_identically"] == leak["messages_checked"]
                        and leak["agent1_hash_equals_case_file"] == len(with_ctx) and leak["retrieved_study_ids_in_messages"] == 0 and leak["all_stored_hashes_match_recomputed"])
    (G2 / "g2_leakage_checks.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- generated reports table (reference column for evaluation only)
    gr = []
    for u in ids:
        c1, a = case_by[u], ag[u]
        gr.append({"study_id": u, "anon_id": c1["anon_id"], "classifier_positive_findings": "; ".join(c1["classifier_positive_findings"]), "classifier_probabilities_of_positives": json.dumps({f: round(c1["classifier_probabilities"][f], 4) for f in c1["classifier_positive_findings"]}),
                   "no_finding_positive": c1["no_finding_positive"], "retrieved_top5_ids": "; ".join(r["study_id"] for r in c1["retrieved"]), "agent1_evidence_json": json.dumps(evidence_for_downstream(a["evidence"]), ensure_ascii=False),
                   "agent1_validation_flags": "; ".join(a["agent1_validation_flags"]), "agent2_report": combined(a["agent2_draft"]), "agent3_action": a["agent3"]["action"], "agent3_issues": "; ".join(a["agent3"]["issues"]),
                   "final_g2_report": combined(a["final_report"]), "g1_report": gen_text[G1S](u), "reference_report_for_evaluation_only": c1["reference"]["combined"]})
    pd.DataFrame(gr).to_csv(G2 / "g2_generated_reports.csv", index=False)

    # ---------------------------------------------------------------- stability diagnostics, stratification, failures
    diag = {}
    for s, texts_ in ((G1S, {u: combined(parse_report(raw["G1"][u]["text"])) for u in ids}), (G2DS, {u: combined(ag[u]["agent2_draft"]) for u in ids}), (G2S, {u: combined(ag[u]["final_report"]) for u in ids})):
        cnt = collections.Counter(re.sub(r"\s+", " ", t).strip() for t in texts_.values())
        top, n_top = cnt.most_common(1)[0]
        diag[s] = {"distinct_report_texts": len(cnt), "most_common_text_share": n_top / len(ids), "most_common_text": top[:200], "format_ok_rate": float(by[s].format_ok.mean())}
    diag["agents"] = {"reasoning_token_leaks": int(sum("<unused" in (ag[u].get("agent1_raw_text") or "") + ag[u]["agent2_raw_text"] + ag[u]["agent3_raw_text"] for u in ids)),
                      "calls_stopped_by_token_limit": int(sum(v == "length" for u in ids for v in ag[u]["done_reasons"].values())), "calls_with_possibly_truncated_prompt": int(sum(any(ag[u]["prompt_may_be_truncated"].values()) for u in ids))}
    (G2 / "g2_stability_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    strat_rows = []
    pos_state = np.array([cl[B0S].loc[u, "classifier_state"] for u in clin_ids])
    for state in ("abnormal", "normal", "indeterminate"):
        pos = np.where(pos_state == state)[0]
        if len(pos) < 10:
            continue
        sub_arr = {s: {k: v[pos] for k, v in arr[s].items()} for s in SYSTEMS}
        dsub = bootstrap_draws(len(pos), N_BOOT, SEED)
        pts = {s: clinical_metrics(sub_arr[s], np.arange(len(pos))) for s in (G1S, G2S)}
        bsub = {s: {k: [] for k in pts[s]} for s in (G1S, G2S)}
        for i in range(N_BOOT):
            for s in (G1S, G2S):
                for k, v in clinical_metrics(sub_arr[s], dsub[i]).items():
                    bsub[s][k].append(v)
        for k in ("precision", "recall", "f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall"):
            d = np.array(bsub[G2S][k]) - np.array(bsub[G1S][k])
            if np.all(np.isnan(d)):
                continue
            strat_rows.append({"classifier_state": state, "n_studies": len(pos), "metric": k, G1S: pts[G1S][k], G2S: pts[G2S][k], "g2_minus_g1": pts[G2S][k] - pts[G1S][k], "ci95_low": q(d, 2.5), "ci95_high": q(d, 97.5),
                               "excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0)})
    pd.DataFrame(strat_rows).to_csv(G2 / "g2_stratified_by_classifier_state.csv", index=False, float_format="%.17g")

    fmt = {s: {"format_ok_rate": float(by[s].format_ok.mean()), "mean_words": float(by[s].words.mean()), "median_words": float(by[s].words.median())} for s in SYSTEMS}
    summary = {"n_paired_studies": len(ids), "n_clinical_subset": len(clin_ids), "n_studies_with_context": len(with_ctx), "n_empty_query_studies": len(ids) - len(with_ctx), "n_failed_generation": run["n_failed"], "reproduction": repro,
               "format_length": fmt, "point_estimates": pt, "systems": list(SYSTEMS), "primary_comparison": f"{G2S} minus {G1S}", "macro_classes_with_truth": int(macro_classes.sum()), "template_min_df": TEMPLATE_MIN_DF}
    (G2 / "g2_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 240)
    print(pd.DataFrame(main_rows)[["metric", B0S, G1S, G2DS, G2S]].round(3).to_string())
    print(pdf[pdf.comparison == f"{G2S} minus {G1S}"][["metric", "diff", "ci95_low", "ci95_high", "excludes_zero"]].round(3).to_string())
    print(json.dumps({"leak_pass": leak["pass"], "agent3": {k: a3[k] for k in ("actions", "approved_unchanged_pct", "clinical_subset")}}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
