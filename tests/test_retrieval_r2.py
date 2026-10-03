"""R2: query expansion/weighting/Top-N, normal queries, RRF, MMR, determinism, leakage exclusion, metric consistency and
frozen-classifier integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS, NO_FINDING
from src.retrieval.diversity import mmr_rerank
from src.retrieval.evaluation import bootstrap_draws, evaluate_ranking, paired_ci
from src.retrieval.expansion import finding_weights, load_expansion, phrase_text, select_top_n, weighted_query_vector
from src.retrieval.fusion import rrf_fuse
from src.retrieval.queries import ABNORMAL, NORMAL_QUERY, PHRASE, classifier_query
from src.retrieval.r2_context import R2Context, norm_text

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "results/retrieval/experiments/r2_optimization"
R1 = ROOT / "results/retrieval/experiments/r1_baseline"
EXP = ROOT / "results/classification/experiments"


@pytest.fixture(scope="module")
def ctx():
    return R2Context()


# ---------------------------------------------------------------- expansion mapping
def test_expansion_mapping_is_fixed_deterministic_and_short():
    a, b = load_expansion(R2 / "finding_query_expansion.json"), load_expansion(R2 / "finding_query_expansion.json")
    assert a == b and set(a["expansions"]) == set(ABNORMAL)
    assert all(0 < len(p.split()) <= 8 for p in a["expansions"].values())
    assert a["expansions"]["Pleural Effusion"] == "pleural effusion pleural fluid"
    prot = json.loads((R2 / "r2_selection_protocol.json").read_text(encoding="utf-8"))
    assert prot["expansion_file_sha256"] == sha256_file(R2 / "finding_query_expansion.json") and prot["written_before_any_r2_evaluation"] is True
    ph = a["expansions"]
    assert phrase_text(["Pleural Effusion", "Cardiomegaly"], ph, "n") == phrase_text(["Cardiomegaly", "Pleural Effusion"], ph, "n")   # fixed class order
    assert phrase_text([NO_FINDING], ph, "normal chest") == "normal chest" and phrase_text([], ph, "x") == ""
    bad = {"expansions": {**ph, "Fracture": "a b c d e f g h i"}}
    p = R2 / "_tmp_bad_expansion.json"
    try:
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            load_expansion(p)
    finally:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------- weights / weighted vector / Top-N
def test_finding_weights_definitions_and_fallback():
    thr = {l: 0.2 for l in ABNORMAL}
    probs = {l: 0.6 for l in ABNORMAL} | {"Fracture": 0.2, "Edema": 1.0}
    f = ["Cardiomegaly", "Fracture", "Edema"]
    assert finding_weights("uniform", f, probs, thr) == {l: 1.0 for l in f}
    assert finding_weights("probability", f, probs, thr)["Cardiomegaly"] == 0.6
    m = finding_weights("margin", f, probs, thr)
    assert m["Fracture"] == 0.0 and m["Edema"] == pytest.approx(1.0) and m["Cardiomegaly"] == pytest.approx(0.5)
    assert finding_weights("margin", ["Fracture"], probs, thr) == {"Fracture": 1.0}           # all-zero -> uniform fallback
    with pytest.raises(ValueError):
        finding_weights("max_p_minus_half", f, probs, thr)


def test_weighted_query_vector_is_normalised_and_consistent():
    rng = np.random.default_rng(0)
    vecs = {l: (lambda v: v / np.linalg.norm(v))(rng.normal(size=16)) for l in ("Cardiomegaly", "Edema")}
    q = weighted_query_vector(vecs, {"Cardiomegaly": 3.0, "Edema": 1.0})
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6) and q.dtype == np.float32
    assert np.allclose(weighted_query_vector(vecs, {"Cardiomegaly": 1.0}), vecs["Cardiomegaly"], atol=1e-6)     # single finding == its phrase
    assert np.allclose(weighted_query_vector(vecs, {"Cardiomegaly": 5.0, "Edema": 5.0}), weighted_query_vector(vecs, {"Cardiomegaly": 1.0, "Edema": 1.0}), atol=1e-6)   # scale invariant
    a, b = np.dot(q, vecs["Cardiomegaly"]), np.dot(q, vecs["Edema"])
    assert a > b                                                                                                  # heavier weight pulls the query closer
    with pytest.raises(ValueError):
        weighted_query_vector(vecs, {})


def test_top_n_selection_orders_by_probability_and_never_pads():
    probs = {l: 0.1 for l in LABELS} | {"Edema": 0.9, "Fracture": 0.5, "Cardiomegaly": 0.5, "Pneumonia": 0.7}
    pos = ["Cardiomegaly", "Edema", "Fracture", "Pneumonia"]
    assert select_top_n(pos, probs, 1) == ["Edema"]
    assert select_top_n(pos, probs, 2) == ["Edema", "Pneumonia"]
    assert select_top_n(pos, probs, 3) == ["Cardiomegaly", "Edema", "Pneumonia"] and "Fracture" not in select_top_n(pos, probs, 3)   # tie 0.5: class order (Cardiomegaly first)
    assert select_top_n(pos, probs, None) == [l for l in ABNORMAL if l in pos]
    assert select_top_n(["Edema"], probs, 3) == ["Edema"] and select_top_n([], probs, 2) == []


def test_probability_rejected_formula_would_zero_most_positives():
    dec = json.loads((R2 / "r2_stage_decisions.json").read_text(encoding="utf-8"))
    note = [d for d in dec["decisions"] if d["stage"] == "A2_note"][0]
    assert note["share_of_frozen_positive_findings_with_p_below_0.5"] > 0.5


# ---------------------------------------------------------------- normal queries
def test_normal_query_handling():
    pred = {l: False for l in LABELS} | {NO_FINDING: True}
    assert classifier_query(pred)["query"] == NORMAL_QUERY and classifier_query({l: False for l in LABELS})["status"] == "empty"
    sel = json.loads((R2 / "normal_query_selection.json").read_text(encoding="utf-8"))
    exp = load_expansion(R2 / "finding_query_expansion.json")
    assert sel["chosen"] in exp["normal_phrase_candidates"] and len(exp["normal_phrase_candidates"]) == 3
    cfg = json.loads((R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    assert cfg["query_construction"]["no_finding_query_phrase"] == sel["chosen"]
    assert phrase_text([NO_FINDING], exp["expansions"], sel["chosen"]) == sel["chosen"]


# ---------------------------------------------------------------- RRF
def test_rrf_scores_ranks_ties_and_provenance():
    idx = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    out = rrf_fuse({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]}, idx, k=60, depth=100)
    assert [o["doc"] for o in out[:2]] == ["a", "b"]                                # exact tie broken by corpus order
    assert out[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62) == pytest.approx(out[1]["rrf_score"])
    assert (out[0]["dense_rank"], out[0]["bm25_rank"]) == (1, 2) and (out[1]["dense_rank"], out[1]["bm25_rank"]) == (2, 1)
    c = next(o for o in out if o["doc"] == "c")
    d = next(o for o in out if o["doc"] == "d")
    assert c["rrf_score"] == pytest.approx(1 / 63) and c["bm25_rank"] is None and d["dense_rank"] is None
    assert [o["doc"] for o in out] == ["a", "b", "c", "d"] and [o["rank"] for o in out] == [1, 2, 3, 4]
    cut = rrf_fuse({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]}, idx, k=60, depth=2)
    assert {o["doc"] for o in cut} == {"a", "b"}                                      # depth cut-off
    assert rrf_fuse({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]}, idx) == out   # deterministic
    with pytest.raises(ValueError):
        rrf_fuse({"x": ["a"]}, idx, k=0)


# ---------------------------------------------------------------- MMR
def test_mmr_known_behaviour():
    e = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)         # doc0 and doc1 are exact duplicates
    cands, rel = ["d0", "d1", "d2", "d3"], np.array([1.0, 0.95, 0.9, 0.5])
    assert mmr_rerank(cands, rel, e, 1.0, 4) == ["d0", "d1", "d2", "d3"]            # lambda=1 -> pure relevance order
    out = mmr_rerank(cands, rel, e, 0.5, 4)
    assert out[0] == "d0" and out[1] == "d2" and out.index("d1") > out.index("d2")  # the duplicate is pushed down
    assert mmr_rerank(cands, rel, e, 0.5, 2) == out[:2]                              # prefix-consistent
    assert mmr_rerank(cands, rel, e, 0.0, 2)[0] == "d0"                              # first pick is always the most relevant
    assert sorted(mmr_rerank(cands, rel, e, 0.7, 4)) == sorted(cands)
    assert mmr_rerank(cands, np.ones(4), e, 0.5, 4)[0] == "d0"                       # constant relevance: ties by original rank
    with pytest.raises(ValueError):
        mmr_rerank(cands, rel[:3], e, 0.5, 2)
    with pytest.raises(ValueError):
        mmr_rerank(cands, rel, e, 1.5, 2)


# ---------------------------------------------------------------- determinism, leakage, duplicates
def test_rankings_are_deterministic_with_corpus_order_ties(ctx):
    q = ctx.emb[10].copy()
    a, sa = ctx.dense_ranking(q, 50)
    b, sb = ctx.dense_ranking(q, 50)
    assert a == b and np.array_equal(sa, sb) and np.all(np.diff(sa) <= 0)
    tied = [i for i in range(len(sa) - 1) if sa[i] == sa[i + 1]]
    assert tied and all(ctx.corpus_index[a[i]] < ctx.corpus_index[a[i + 1]] for i in tied)
    t1, t2 = ctx.bm25_ranking("pleural effusion cardiomegaly", 30), ctx.bm25_ranking("pleural effusion cardiomegaly", 30)
    assert t1[0] == t2[0] and np.array_equal(t1[1], t2[1])


def test_no_self_retrieval_and_locked_test_exclusion(ctx):
    top = pd.read_csv(R2 / "retrieval_top10_all_configs.csv.gz", dtype={"uid": str, "retrieved_uid": str})
    assert len(top) > 50000 and (top.uid == top.retrieved_uid).sum() == 0
    assert set(top.retrieved_uid) <= set(ctx.ids) and not (set(top.retrieved_uid) & ctx.test_ids)
    assert not (set(top.uid) & ctx.test_ids) and not (set(top.uid) & set(ctx.ids))
    assert not (set(ctx.primary) & ctx.test_ids)
    for g in list(top.groupby(["config", "uid"]))[:300]:
        assert g[1].retrieved_uid.is_unique and len(g[1]) == 10


def test_duplicate_detection_and_rate():
    assert norm_text("No acute  abnormality.") == norm_text("no acute abnormality") and norm_text(float("nan")) == ""
    sets = {f"d{i}": frozenset({"Edema"}) for i in range(10)}
    norms = {f"d{i}": ("same text" if i < 2 else f"text {i}") for i in range(10)}
    words = {d: 3 for d in sets}
    emb = np.random.default_rng(0).normal(size=(10, 8))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    m = evaluate_ranking([f"d{i}" for i in range(10)], frozenset({"Edema"}), frozenset({"Edema"}), sets, norms, words, emb, {f"d{i}": i for i in range(10)}, np.ones(10), True)
    assert m["duplicate_text_rate@1"] == 0 and m["duplicate_text_rate@3"] == pytest.approx(1 / 3) and m["unique_templates@3"] == 2
    assert m["jaccard_truth@10"] == 1.0 and m["ndcg@5"] == pytest.approx(1.0) and m["context_words@5"] == 15 and m["union_coverage@1"] == 1.0
    with pytest.raises(ValueError):
        evaluate_ranking(["d0"] * 5, frozenset({"Edema"}), frozenset(), sets, norms, words, emb, {"d0": 0}, np.ones(10), True)


# ---------------------------------------------------------------- metric consistency / reproduction
def test_r1_baselines_reproduced_exactly_and_summary_is_consistent():
    rep = json.loads((R2 / "r1_reproduction_check.json").read_text(encoding="utf-8"))
    assert rep["r1_reproduced_exactly"] is True and rep["r1_order"]["n_lists_not_identical_to_r1"] == 0 and rep["primary_set_size"] == 358
    assert rep["r1_order"]["max_abs_metric_difference_any_query"] < 1e-12
    r1 = pd.read_csv(R1 / "retrieval_summary_primary.csv").set_index(["retriever", "policy"])
    s = pd.read_csv(R2 / "r2_summary_all_configs.csv").set_index("config")
    assert abs(s.loc["O_Q0_dense", "jaccard_truth@3"] - r1.loc[("dense_minilm", "oracle"), "jaccard_truth@3"]) < 1e-9
    assert abs(s.loc["R1_Q0_bm25", "ndcg@3"] - r1.loc[("lexical_bm25", "classifier_all_positive"), "ndcg@3"]) < 1e-9
    assert abs(s.loc["R1_gated_bm25", "jaccard_truth@5"] - r1.loc[("lexical_bm25", "classifier_gated"), "jaccard_truth@5"]) < 1e-9
    pq = pd.read_csv(R2 / "per_query_metrics_all_configs.csv", dtype={"uid": str})
    for cfg in ("FINAL_pipeline", "B_dense", "A3_Q3_top3_dense"):
        d = pq[pq.config == cfg]
        assert len(d) == 358 and d.uid.is_unique and abs(d["jaccard_truth@3"].mean() - s.loc[cfg, "jaccard_truth@3"]) < 1e-12


def test_paired_bootstrap_is_deterministic_and_paired():
    d1, d2 = bootstrap_draws(50, 100, 42), bootstrap_draws(50, 100, 42)
    assert np.array_equal(d1, d2) and not np.array_equal(d1, bootstrap_draws(50, 100, 43))
    a = np.random.default_rng(1).random(50)
    r = paired_ci(a + 0.1, a, d1)
    assert r["mean_difference"] == pytest.approx(0.1) and r["ci95_low"] == pytest.approx(0.1) and r["ci95_high"] == pytest.approx(0.1)   # constant paired shift


def test_protocol_decisions_and_candidate_config_are_consistent():
    dec = json.loads((R2 / "r2_stage_decisions.json").read_text(encoding="utf-8"))
    cfg = json.loads((R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    assert dec["protocol_sha256"] == sha256_file(R2 / "r2_selection_protocol.json") == cfg["selection_protocol_sha256"]
    assert cfg["provisional_top_k"] == dec["selected_K"] in (1, 3, 5, 10) and "unopened" in cfg["status"]
    assert cfg["retriever"]["selected"] == dec["selected_retriever"] and cfg["embedding_model"]["dimension"] == 384
    log = pd.read_csv(R2 / "r2_k_selection_log.csv")
    assert not log[(log.from_K == dec["selected_K"]) & log.advance].shape[0]          # no further advance is possible from the selected K


# ---------------------------------------------------------------- frozen classifier
def test_frozen_classifier_unchanged_by_r2():
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    assert all(verify_freeze(man_path, ROOT).values())
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert sha256_file(ROOT / man["checkpoint"]) == man["checkpoint_sha256"]
    r1 = json.loads((R1 / "iu_classifier_run.json").read_text(encoding="utf-8"))
    assert r1["classifier_tuned_on_iu"] is False and r1["checkpoint_sha256"] == man["checkpoint_sha256"]
