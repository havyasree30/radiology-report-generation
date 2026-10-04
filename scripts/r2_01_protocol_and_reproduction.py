"""R2 step 0-2: write the FIXED query-expansion mapping and the selection protocol BEFORE any R2 evaluation, then reproduce
the R1 baselines exactly (R1 tie order = FAISS k=10) and quantify the effect of R2's deterministic tie rule.

    .venv\\Scripts\\python.exe -m scripts.r2_01_protocol_and_reproduction
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.classification.final_test import sha256_file, verify_freeze
from src.retrieval.evaluation import evaluate_ranking
from src.retrieval.metrics import KS
from src.retrieval.queries import ABNORMAL
from src.retrieval.r2_context import R1, R2, R2Context

EXPANSION = {
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinal silhouette widened mediastinum",
    "Cardiomegaly": "cardiomegaly enlarged cardiac silhouette",
    "Lung Opacity": "lung opacity airspace opacity",
    "Lung Lesion": "lung nodule mass lesion",
    "Edema": "pulmonary edema interstitial edema",
    "Consolidation": "consolidation airspace consolidation",
    "Pneumonia": "pneumonia focal airspace disease",
    "Atelectasis": "atelectasis volume loss",
    "Pneumothorax": "pneumothorax pleural air",
    "Pleural Effusion": "pleural effusion pleural fluid",
    "Pleural Other": "pleural thickening pleural scarring",
    "Fracture": "fracture rib fracture",
    "Support Devices": "support device catheter tube pacemaker lead"}
NORMAL_CANDIDATES = ["no acute abnormality", "normal chest radiograph", "no acute cardiopulmonary abnormality"]


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def main() -> int:
    R2.mkdir(parents=True, exist_ok=True)
    if (R2 / "per_query_metrics_all_configs.csv").exists():
        raise SystemExit("R2 results already exist; the protocol must be written BEFORE evaluation and is not rewritten afterwards")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    exp = {"version": "r2-v1", "created_utc": now, "evaluated_before_creation": False,
           "purpose": "ONE fixed mapping from each frozen-classifier finding to a short radiology phrase (finding names + common report wording). Never generated per query; no patient-specific content.",
           "rules": ["<= 8 words", "only terms that name the finding or a standard synonym / closely tied sign", "no negation, laterality, severity or patient-specific observation"],
           "expansions": EXPANSION, "normal_phrase_candidates": NORMAL_CANDIDATES,
           "normal_phrase_selection": "see normal_query_selection.json (chosen on retrieval-validation data only, rule in r2_selection_protocol.json)"}
    (R2 / "finding_query_expansion.json").write_text(json.dumps(exp, indent=2), encoding="utf-8")
    protocol = {
        "version": "r2-v1", "created_utc": now, "written_before_any_r2_evaluation": True, "r1_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "expansion_file_sha256": sha256_file(R2 / "finding_query_expansion.json"),
        "population": "the exact R1 primary paired validation set (358 studies); no re-split, no removal, locked retrieval test never opened",
        "primary_selection_metric": "dense Jaccard@3 versus the study truth, mean over primary queries; nDCG@3 must not disagree in sign (reported)",
        "replacement_rule": "a challenger replaces the incumbent only if the lower bound of the paired bootstrap 95% CI of (challenger - incumbent) in the primary selection metric is > 0; otherwise the incumbent (simpler / earlier) is kept",
        "bootstrap": {"n_resamples": 1000, "seed": 42, "unit": "query (study); one shared set of resampling indices for every comparison"},
        "normal_phrase_rule": {"candidates": NORMAL_CANDIDATES, "data": "dense retrieval on ALL eligible validation studies whose TRUE finding set is {No Finding} (oracle-normal queries; independent of classifier errors)",
                               "metric": "Jaccard@3", "rule": "replace the R1 phrase 'no acute abnormality' only if a candidate's paired CI lower bound > 0"},
        "stage_A": {"A1_phrases": "incumbent Q0_finding_names (R1); challenger Q1_expanded_findings (text query of the fixed expansions)",
                    "A2_weighting": {"representation": "dense query vector = L2-normalised sum_i w_i * embedding(phrase_i) using the phrases of the A1 winner; challengers: uniform, probability, margin; incumbent = the A1 winner's text query",
                                     "weight_definitions": {"uniform": "w=1", "probability": "w=calibrated probability p", "margin": "w=(p - t)/(1 - t), t = frozen calibrated-equivalent threshold"},
                                     "rejected_by_design": "max(p - 0.5, 0): most frozen positives have p < 0.5, so it would zero most findings"},
                    "A3_top_n": {"N": [1, 2, 3], "incumbent": "all positives", "ordering": "calibrated probability descending, ties by class order", "rule": "among N with a reliable gain over 'all' choose the highest mean Jaccard@3; else keep all"}},
        "stage_B": {"retrievers": ["dense", "bm25", "hybrid_rrf"], "incumbent": "dense", "rrf": {"k": 60, "depth": 100, "ties": "corpus order"}, "rule": "same replacement rule (challenger CI lower bound > 0 vs dense)",
                    "bm25_text_when_q2_selected": "text query of the A1 winner (weights apply to the dense vector only)"},
        "stage_C": {"method": "MMR over the top-30 candidates of the Stage-B pipeline", "pool": 30, "lambdas": [0.5, 0.7, 0.9], "relevance": "min-max normalised retriever score inside the pool", "similarity": "cosine of MiniLM report embeddings",
                    "adoption_rule": "a lambda is eligible if, at BOTH K=3 and K=5: duplicate-text-rate difference (MMR - standard) has CI upper bound < 0; Jaccard@K and nDCG@K differences do not have CI upper bound < 0 (no reliable loss); union-coverage@K point difference >= 0. Adopt the LARGEST eligible lambda (least intervention); otherwise reject MMR"},
        "k_selection": {"candidates": [1, 3, 5, 10], "start": 1,
                        "advance_to_larger_K_if_all": ["paired union-coverage gain has CI lower bound > 0", "mean coverage gain >= 0.05", "Jaccard@K does not fall by more than 0.03 (point estimate)",
                                                       "duplicate-text rate@K <= 0.50 and unique-template count increases", "mean context <= 1000 words"],
                        "procedure": "from the current K test every larger candidate in ascending order and move to the SMALLEST candidate that satisfies all conditions; repeat; stop when none does (prefer the smaller K)"},
        "oracle_references": "R1 oracle dense and BM25 (Q0) reproduced exactly; oracle with the selected R2 representation/pipeline reported as a diagnostic",
        "no_finding_query": "the selected normal phrase is used by every R2 policy for studies whose classifier query is normal; Q0 keeps the R1 phrase to remain the exact R1 baseline",
        "secondary_sensitivity": "primary set excluding queries whose own report text has an exact copy in the corpus; official primary results are unchanged"}
    (R2 / "r2_selection_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- R1 reproduction (R1 tie order) and the R2 deterministic tie rule
    verify_freeze(R1.parent.parent.parent / "classification/experiments/c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json", R1.parents[3])
    ctx = R2Context()
    r1 = pd.read_csv(R1 / "per_query_metrics.csv", dtype={"uid": str})
    r1 = r1[r1.retrieved > 0]
    assert set(r1[r1.in_primary_set].uid) == set(ctx.primary) and len(ctx.primary) == 358
    metric_cols = [f"{m}@{k}" for m in ("jaccard_truth", "jaccard_query", "ndcg", "hit", "rr", "union_coverage", "zero_overlap_rate", "duplicate_text_rate", "mean_pairwise_cosine") for k in KS if not (m == "mean_pairwise_cosine" and k == 1)]
    rows, work = [], []
    for rt in ("dense_minilm", "lexical_bm25"):
        for pol in ("oracle", "classifier_all_positive", "classifier_gated"):
            d = r1[(r1.retriever == rt) & (r1.policy == pol)]
            for _, r in d.iterrows():
                u, text = r.uid, r["query"]
                qset = ctx.truth[u] if pol == "oracle" else frozenset(x for x in str(ctx.cls.loc[u, "query_all_positive_findings" if pol == "classifier_all_positive" else "query_gated_findings"]).split(";") if x)
                if rt == "dense_minilm":
                    q = ctx.embed_text(text)
                    sims, rws = ctx.index.search(q[None, :].astype(np.float32), 10)           # R1 order (FAISS k=10)
                    r1_list = [ctx.ids[i] for i in rws[0]]
                    w_list, _ = ctx.dense_ranking(q, 10)                                      # R2 deterministic tie rule
                else:
                    r1_list = [ctx.ids[i] for i in np.argsort(-ctx.bm25.scores(text), kind="stable")[:10]]
                    w_list = r1_list
                for tag, lst in (("r1_order", r1_list), ("r2_order", w_list)):
                    m = evaluate_ranking(lst, ctx.truth[u], qset, ctx.doc_set, ctx.doc_norm, ctx.doc_words, ctx.emb, ctx.corpus_index, ctx.all_gain[u], ctx.exact_possible[u])
                    rows.append({"tag": tag, "retriever": rt, "policy": pol, "uid": u, "list_equal_to_r1": lst == r1_list, **{c: m[c] for c in metric_cols}, **{f"r1_{c}": r[c] for c in metric_cols}})
    df = pd.DataFrame(rows)
    out = {}
    for tag in ("r1_order", "r2_order"):
        d = df[df.tag == tag]
        diffs = np.array([np.nanmax(np.abs(d[c].to_numpy(float) - d[f"r1_{c}"].to_numpy(float))) if d[c].notna().any() else 0.0 for c in metric_cols])
        prim = d[d.uid.isin(ctx.primary)]
        agg = {f"{rt}|{pol}": float(abs(g["jaccard_truth@3"].mean() - g["r1_jaccard_truth@3"].mean())) for (rt, pol), g in prim.groupby(["retriever", "policy"])}
        out[tag] = {"n_rows": int(len(d)), "n_lists_not_identical_to_r1": int((~d.list_equal_to_r1).sum()), "max_abs_metric_difference_any_query": float(diffs.max()),
                    "primary_set_abs_difference_in_mean_jaccard@3": agg}
    out["r1_reproduced_exactly"] = bool(out["r1_order"]["n_lists_not_identical_to_r1"] == 0 and out["r1_order"]["max_abs_metric_difference_any_query"] < 1e-12)
    out["note"] = ("r1_order = FAISS k=10 order used by R1 (reproduced exactly). r2_order = the deterministic tie rule used for every R2 comparison; it differs from R1 only in the order of "
                   "exactly tied (identical-embedding) reports, which affects a handful of lists.")
    out["primary_set_size"] = len(ctx.primary)
    (R2 / "r1_reproduction_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ctx.save_cache()
    print(json.dumps({"reproduced": out["r1_reproduced_exactly"], "r1_order": {k: v for k, v in out["r1_order"].items() if k != "primary_set_abs_difference_in_mean_jaccard@3"},
                      "r2_order": out["r2_order"]}, indent=1))
    return 0 if out["r1_reproduced_exactly"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
