"""G1F step 2: validation metrics of the guarded final system (G1F) against the original G1, the three-state (normal / abnormal / indeterminate) routing
analysis with decided-case recalls and decision coverage, and the descriptive abstention analysis. Same populations, extractor, metrics and bootstrap convention
as G1 (study-level paired bootstrap, 1,000 resamples, seed 42). Validation data only; nothing here feeds back into any rule.

    .venv\\Scripts\\python.exe -m scripts.g1f_02_evaluate
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.generation.extraction import ABNORMAL, ReportExtractor
from src.generation.metrics import per_class_prf, prf
from src.generation.parse import parse_report
from src.generation.study_eval import FS, study_row
from src.retrieval.evaluation import bootstrap_draws
from src.system.guard import INDETERMINATE_FINDINGS, INDETERMINATE_IMPRESSION, STATE_ABNORMAL, STATE_INDETERMINATE, STATE_NORMAL
from src.utils.config import PROJECT_ROOT

G1 = PROJECT_ROOT / "results/report_generation/experiments/g1_single_agent"
G1F = PROJECT_ROOT / "results/report_generation/experiments/g1f_final_system"
G1S, G1FS = "G1_original", "G1F_final_system"
SYSTEMS = (G1S, G1FS)
RARE = ["Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"]     # C2 definition
N_BOOT, SEED = 1000, 42
TEXT_KEYS = ("rouge_l", "bleu1", "bleu4", "meteor_exact", "mean_words")


def sets_of(v):
    if isinstance(v, (list, tuple, set, frozenset)):
        return set(v)
    return set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()


def main() -> int:
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    raw = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")}
    out = pd.read_csv(G1F / "g1f_final_validation_outputs.csv", dtype={"study_id": str}).fillna("").set_index("study_id")
    ext = ReportExtractor()
    rows = []
    for c in cases:
        u = c["uid"]
        g1 = parse_report(raw[u]["text"])
        g1f = {"findings": out.loc[u, "final_findings"], "impression": out.loc[u, "final_impression"], "format_ok": True}
        rows.append({**study_row(c, G1S, g1, ext), "routing_state": out.loc[u, "system_interpretation_state"]})
        rows.append({**study_row(c, G1FS, g1f, ext), "routing_state": out.loc[u, "system_interpretation_state"]})
    df = pd.DataFrame(rows)
    df.to_csv(G1F / "g1f_per_study_results.csv", index=False)
    ids = [c["uid"] for c in cases]
    by = {s: df[df.system == s].set_index("uid").loc[ids] for s in SYSTEMS}
    state = by[G1S].routing_state
    clin_ids = [u for u in ids if bool(by[G1S].loc[u, "in_clinical"])]
    cl = {s: by[s].loc[clin_ids] for s in SYSTEMS}

    # ---------------------------------------------------------------- unchanged-output check: Path A / B rows must be identical
    cmp_cols = [c for c in by[G1S].columns if c not in ("system",)]
    ab = [u for u in ids if state[u] != STATE_INDETERMINATE]
    same_rows = int(sum(by[G1S].loc[u, cmp_cols].astype(str).equals(by[G1FS].loc[u, cmp_cols].astype(str)) for u in ab))
    unchanged = {"path_A_B_studies": len(ab), "per_study_evaluation_rows_identical": same_rows, "all_identical": same_rows == len(ab),
                 "changed_studies_are_exactly_path_C": sorted(u for u in ids if not by[G1S].loc[u, cmp_cols].astype(str).equals(by[G1FS].loc[u, cmp_cols].astype(str))) == sorted(u for u in ids if state[u] == STATE_INDETERMINATE)}
    (G1F / "g1f_unchanged_outputs_check.json").write_text(json.dumps(unchanged, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- bootstrap arrays (same definitions as G1)
    cls_idx = {c: i for i, c in enumerate(ABNORMAL)}
    rare_mask = np.array([c in RARE for c in ABNORMAL])
    truth_n = np.zeros(len(ABNORMAL))
    for t in cl[G1S].truth:
        for f in sets_of(t):
            truth_n[cls_idx[f]] += 1
    macro_classes = truth_n > 0

    def arrays(d: pd.DataFrame) -> dict:
        n = len(d)
        A = d[["tp", "fp", "fn", "hall", "omit"]].to_numpy(float)
        P = d[["clf_fp", "clf_fp_mentioned", "clf_tp", "clf_tp_retained"]].to_numpy(float)
        C = np.zeros((n, len(ABNORMAL), 3))
        R = np.zeros((n, len(ABNORMAL), 2))
        for i, (st, tr, cp) in enumerate(zip(d.stated, d.truth, d.clf_pos)):
            st, tr, cp = sets_of(st), sets_of(tr), sets_of(cp)
            for f in ABNORMAL:
                j = cls_idx[f]
                C[i, j] = (f in st and f in tr, f in st and f not in tr, f not in st and f in tr)
                R[i, j] = (f in cp and f in tr, f in cp and f in tr and f in st)
        return {"A": A, "P": P, "C": C, "R": R}

    arr = {s: arrays(cl[s]) for s in SYSTEMS}
    txt = {s: by[s][["rouge_l", "bleu1", "bleu4", "meteor_exact", "words"]].to_numpy(float) for s in SYSTEMS}

    def clinical_metrics(a: dict, idx: np.ndarray) -> dict:
        A, P, C, R = a["A"][idx], a["P"][idx], a["C"][idx].sum(0), a["R"][idx].sum(0)
        r = prf(A[:, 0].sum(), A[:, 1].sum(), A[:, 2].sum())
        tp, fp, fn = C[:, 0], C[:, 1], C[:, 2]
        f1c = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
        rare_tp, rare_ret = R[rare_mask, 0].sum(), R[rare_mask, 1].sum()
        return {"precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "macro_f1": float(np.nanmean(f1c[macro_classes])), "hallucination_rate": A[:, 3].mean(), "omission_rate": A[:, 4].mean(),
                "clf_fp_propagation": P[:, 1].sum() / P[:, 0].sum(), "tp_retention": P[:, 3].sum() / P[:, 2].sum(), "rare_tp_retention": rare_ret / rare_tp if rare_tp else np.nan}

    def text_metrics_fn(t: np.ndarray, idx: np.ndarray) -> dict:
        m = t[idx].mean(0)
        return {"rouge_l": m[0], "bleu1": m[1], "bleu4": m[2], "meteor_exact": m[3], "mean_words": m[4]}

    dc, dp = bootstrap_draws(len(clin_ids), N_BOOT, SEED), bootstrap_draws(len(ids), N_BOOT, SEED)
    pt = {s: {**clinical_metrics(arr[s], np.arange(len(clin_ids))), **text_metrics_fn(txt[s], np.arange(len(ids)))} for s in SYSTEMS}
    bs = {s: {k: [] for k in pt[s]} for s in SYSTEMS}
    for i in range(N_BOOT):
        for s in SYSTEMS:
            for k, v in {**clinical_metrics(arr[s], dc[i]), **text_metrics_fn(txt[s], dp[i])}.items():
                bs[s][k].append(v)
    q = lambda v, p: float(np.nanpercentile(v, p))  # noqa: E731
    pos_c = np.array([i for i, u in enumerate(clin_ids) if state[u] == STATE_INDETERMINATE])
    rows_m = []
    for k in pt[G1S]:
        d = np.array(bs[G1FS][k]) - np.array(bs[G1S][k])
        rows_m.append({"metric": k, "n_studies": len(ids) if k in TEXT_KEYS else len(clin_ids), G1S: pt[G1S][k], f"{G1S}_ci95_low": q(bs[G1S][k], 2.5), f"{G1S}_ci95_high": q(bs[G1S][k], 97.5), G1FS: pt[G1FS][k],
                       f"{G1FS}_ci95_low": q(bs[G1FS][k], 2.5), f"{G1FS}_ci95_high": q(bs[G1FS][k], 97.5), "diff_g1f_minus_g1": pt[G1FS][k] - pt[G1S][k], "diff_ci95_low": q(d, 2.5), "diff_ci95_high": q(d, 97.5),
                       "diff_excludes_zero": bool(q(d, 2.5) > 0 or q(d, 97.5) < 0), "change_arises_only_from_path_C_studies": True})
    pd.DataFrame(rows_m).to_csv(G1F / "g1f_validation_metrics.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- what changed: Path C studies only
    pc_rows = []
    for s in SYSTEMS:
        sub = cl[s].loc[[clin_ids[i] for i in pos_c]]
        pc_rows.append({"system": s, "n_path_C_studies_in_clinical_subset": len(sub), "stated_findings": int(sum(len(sets_of(x)) for x in sub.stated)), "tp": int(sub.tp.sum()), "fp": int(sub.fp.sum()), "fn": int(sub.fn.sum()),
                        "reports_with_hallucination": int(sub.hall.sum()), "reports_with_omission": int(sub.omit.sum()), "reports_stating_normal": int((sub.report_state == "normal").sum()), "reports_stating_abnormal": int((sub.report_state == "abnormal").sum()),
                        "reports_indeterminate_by_extractor": int((sub.report_state == "indeterminate").sum()), "mean_words": float(sub.words.mean())})
    pd.DataFrame(pc_rows).to_csv(G1F / "g1f_path_C_only_effect.csv", index=False)

    # ---------------------------------------------------------------- three-state analysis (clinical subset = studies with a reference state)
    ref = cl[G1S].reference_state
    st_c = state.loc[clin_ids]
    all_counts = {s: int((state == s).sum()) for s in (STATE_NORMAL, STATE_ABNORMAL, STATE_INDETERMINATE)}
    clin_counts = {s: int((st_c == s).sum()) for s in (STATE_NORMAL, STATE_ABNORMAL, STATE_INDETERMINATE)}
    three = [{"population": "all 547 validation studies (primary set)", "n": len(ids), **{f"{s}_n": all_counts[s] for s in all_counts}, **{f"{s}_pct": 100 * all_counts[s] / len(ids) for s in all_counts}},
             {"population": f"{len(clin_ids)} clinical-subset studies (reference available)", "n": len(clin_ids), **{f"{s}_n": clin_counts[s] for s in clin_counts}, **{f"{s}_pct": 100 * clin_counts[s] / len(clin_ids) for s in clin_counts}}]
    pd.DataFrame(three).to_csv(G1F / "g1f_system_state_counts.csv", index=False, float_format="%.17g")
    strat = []
    for r in ("normal", "abnormal"):
        m = ref == r
        n = int(m.sum())
        strat.append({"reference_state": r, "n": n, **{f"routed_{s}_n": int((st_c[m] == s).sum()) for s in clin_counts}, **{f"routed_{s}_pct": 100 * int((st_c[m] == s).sum()) / n for s in clin_counts}})
    pd.DataFrame(strat).to_csv(G1F / "g1f_reference_stratified_routing.csv", index=False, float_format="%.17g")
    refn, refa = (ref == "normal").to_numpy(), (ref == "abnormal").to_numpy()
    rn, ra, ri = (st_c == STATE_NORMAL).to_numpy(), (st_c == STATE_ABNORMAL).to_numpy(), (st_c == STATE_INDETERMINATE).to_numpy()
    dec = rn | ra

    def decided(idx: np.ndarray) -> dict:
        nd, ad = (refn & dec)[idx].sum(), (refa & dec)[idx].sum()
        return {"normal_recall_among_decided": (refn & rn)[idx].sum() / nd if nd else np.nan, "abnormal_recall_among_decided": (refa & ra)[idx].sum() / ad if ad else np.nan, "decision_coverage": dec[idx].mean(),
                "normal_recall_abstention_counted_as_miss": (refn & rn)[idx].sum() / refn[idx].sum(), "abnormal_recall_abstention_counted_as_miss": (refa & ra)[idx].sum() / refa[idx].sum()}

    pt3 = decided(np.arange(len(clin_ids)))
    b3 = {k: [] for k in pt3}
    for i in range(N_BOOT):
        for k, v in decided(dc[i]).items():
            b3[k].append(v)
    # original G1 prose-derived states on the same studies for context
    g1_prose = {"normal_recall_(G1 report prose)": float(((by[G1S].loc[clin_ids].report_state == "normal") & refn).sum() / refn.sum()), "abnormal_recall_(G1 report prose)": float(((by[G1S].loc[clin_ids].report_state == "abnormal") & refa).sum() / refa.sum())}
    dec_rows = [{"metric": k, "value": pt3[k], "ci95_low": q(b3[k], 2.5), "ci95_high": q(b3[k], 97.5)} for k in pt3]
    dec_rows += [{"metric": k, "value": v, "ci95_low": np.nan, "ci95_high": np.nan} for k, v in g1_prose.items()]
    dec_rows += [{"metric": "decided_studies_n", "value": int(dec.sum()), "ci95_low": np.nan, "ci95_high": np.nan}, {"metric": "eligible_studies_n", "value": len(clin_ids), "ci95_low": np.nan, "ci95_high": np.nan},
                 {"metric": "decision_coverage_all_547_studies", "value": float((state != STATE_INDETERMINATE).mean()), "ci95_low": np.nan, "ci95_high": np.nan}]
    pd.DataFrame(dec_rows).to_csv(G1F / "g1f_decided_case_metrics.csv", index=False, float_format="%.17g")

    # ---------------------------------------------------------------- routing state versus the prose of the reused G1 report (descriptive consistency check)
    pv = []
    for s in (STATE_NORMAL, STATE_ABNORMAL, STATE_INDETERMINATE):
        ids_s = [u for u in ids if state[u] == s]
        prose = by[G1FS].loc[ids_s].report_state.value_counts()
        g1prose = by[G1S].loc[ids_s].report_state.value_counts()
        pv.append({"routing_state": s, "n": len(ids_s), **{f"final_report_prose_{k}": int(prose.get(k, 0)) for k in ("normal", "abnormal", "indeterminate")}, **{f"original_g1_prose_{k}": int(g1prose.get(k, 0)) for k in ("normal", "abnormal", "indeterminate")}})
    pd.DataFrame(pv).to_csv(G1F / "g1f_routing_state_vs_report_prose.csv", index=False)

    # ---------------------------------------------------------------- abstention analysis (descriptive)
    abst_all = [u for u in ids if state[u] == STATE_INDETERMINATE]
    abst_c = [u for u in clin_ids if state[u] == STATE_INDETERMINATE]
    ref_abn_all = [u for u in clin_ids if cl[G1S].loc[u, "reference_state"] == "abnormal"]
    ref_abn_abst = [u for u in abst_c if cl[G1S].loc[u, "reference_state"] == "abnormal"]
    ref_abn_dec = [u for u in ref_abn_all if state[u] != STATE_INDETERMINATE]
    truth = {u: sets_of(cl[G1S].loc[u, "truth"]) for u in clin_ids}
    has_rare = lambda u: bool(truth[u] & set(RARE))  # noqa: E731
    inst_abst = [f for u in ref_abn_abst for f in truth[u]]
    inst_all = [f for u in ref_abn_all for f in truth[u]]
    fr = []
    for f in ABNORMAL:
        na, nt = inst_abst.count(f), inst_all.count(f)
        fr.append({"finding": f, "rare_c2": f in RARE, "reference_positive_in_abstained_reference_abnormal": na, "share_of_abstained_findings": na / max(len(inst_abst), 1), "reference_positive_in_all_reference_abnormal_clinical": nt,
                   "share_of_all_findings": nt / max(len(inst_all), 1), "abstained_fraction_of_this_finding": na / nt if nt else np.nan})
    pd.DataFrame(fr).to_csv(G1F / "g1f_abstention_finding_distribution.csv", index=False, float_format="%.17g")
    a, b = sum(has_rare(u) for u in ref_abn_abst), sum(has_rare(u) for u in ref_abn_dec)
    odds = fisher_exact([[a, len(ref_abn_abst) - a], [b, len(ref_abn_dec) - b]]) if ref_abn_abst and ref_abn_dec else (np.nan, np.nan)
    abst = {"n_abstained_all_studies": len(abst_all), "pct_abstained_all_studies": 100 * len(abst_all) / len(ids), "n_abstained_with_reference": len(abst_c), "pct_of_clinical_subset": 100 * len(abst_c) / len(clin_ids),
            "n_abstained_without_reference_(not_in_clinical_subset)": len(abst_all) - len(abst_c), "reference_normal_among_abstained": int(sum(cl[G1S].loc[u, "reference_state"] == "normal" for u in abst_c)),
            "reference_abnormal_among_abstained": len(ref_abn_abst), "reference_normal_fraction": float(np.mean([cl[G1S].loc[u, "reference_state"] == "normal" for u in abst_c])), "reference_abnormal_fraction": len(ref_abn_abst) / len(abst_c),
            "reference_abnormal_studies_total_clinical": len(ref_abn_all), "abstained_share_of_all_reference_abnormal": len(ref_abn_abst) / len(ref_abn_all), "reference_normal_studies_total_clinical": int(refn.sum()),
            "abstained_share_of_all_reference_normal": float(np.mean([st_c[u] == STATE_INDETERMINATE for u in clin_ids if cl[G1S].loc[u, "reference_state"] == "normal"])),
            "abstained_reference_abnormal_with_a_rare_finding": a, "decided_reference_abnormal_with_a_rare_finding": b, "decided_reference_abnormal_n": len(ref_abn_dec),
            "share_rare_among_abstained_ref_abnormal": a / max(len(ref_abn_abst), 1), "share_rare_among_decided_ref_abnormal": b / max(len(ref_abn_dec), 1), "fisher_exact_odds_ratio": float(odds[0]), "fisher_exact_p_two_sided": float(odds[1]),
            "rare_instances_among_abstained_findings": int(sum(f in RARE for f in inst_abst)), "total_instances_among_abstained_findings": len(inst_abst), "rare_instances_among_all_reference_abnormal_findings": int(sum(f in RARE for f in inst_all)),
            "total_instances_among_all_reference_abnormal_findings": len(inst_all), "note": "descriptive only; counts are small and the routing rule is not changed in response to them"}
    (G1F / "g1f_abstention_analysis.json").write_text(json.dumps(abst, indent=2, default=float), encoding="utf-8")
    summary = {"n_studies": len(ids), "n_clinical_subset": len(clin_ids), "path_counts": state.map({STATE_NORMAL: "A", STATE_ABNORMAL: "B", STATE_INDETERMINATE: "C"}).value_counts().sort_index().to_dict(), "unchanged_outputs_check": unchanged,
               "metrics": {r["metric"]: {"g1": r[G1S], "g1f": r[G1FS], "diff": r["diff_g1f_minus_g1"], "ci": [r["diff_ci95_low"], r["diff_ci95_high"]]} for r in rows_m}, "decided": {r["metric"]: r["value"] for r in dec_rows}}
    (G1F / "g1f_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows_m)[["metric", G1S, G1FS, "diff_g1f_minus_g1", "diff_ci95_low", "diff_ci95_high"]].round(3).to_string())
    print(json.dumps(unchanged), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in pt3.items()}))
    return 0 if unchanged["all_identical"] and unchanged["changed_studies_are_exactly_path_C"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
