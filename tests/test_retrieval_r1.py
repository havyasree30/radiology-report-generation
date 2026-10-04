"""R1: leakage-safe split, embeddings/FAISS mapping, queries, finding mapping, metrics, gating and frozen-classifier integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS, NO_FINDING
from src.classification.operating_policy import apply_final_policy, load_final_policy
from src.retrieval.bm25 import BM25
from src.retrieval.dense import build_index, embedding_dimension, l2_normalize, search
from src.retrieval.findings import eval_set, map_study, parse_mesh
from src.retrieval.gating import cross_fitted_gain, fbeta, select_gate
from src.retrieval.metrics import dcg, hit_and_rr, jaccard, ndcg_at_k
from src.retrieval.queries import NORMAL_QUERY, PHRASE, classifier_query, gated_query, oracle_query, phrase_query
from src.retrieval.split_checks import images_in_multiple_splits, pairwise_overlaps, studies_spanning_splits
from src.retrieval.text import build_retrieval_text, clean_section

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results/retrieval/experiments/r1_baseline"
EXP = ROOT / "results/classification/experiments"
ABN = [l for l in LABELS if l != NO_FINDING]


@pytest.fixture(scope="module")
def mapping():
    return json.loads((R1 / "iu_finding_mapping.json").read_text(encoding="utf-8"))


def _neg(**kw):
    return {l: kw.get(l, False) for l in LABELS}


# ---------------------------------------------------------------- split / leakage
def test_study_level_split_is_disjoint_and_corpus_is_train_only():
    sp = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    ids = sp["study_ids"]
    assert pairwise_overlaps(ids["reference_corpus"], ids["validation"], ids["locked_test"]) == {"corpus_and_validation": 0, "corpus_and_test": 0, "validation_and_test": 0}
    table = pd.read_csv(ROOT / "data/splits/iu_xray/iu_report_splits.csv", dtype={"uid": str}).set_index("uid")["split"]
    assert set(table[ids["reference_corpus"]]) == {"train"} and set(table[ids["validation"]]) == {"val"} and set(table[ids["locked_test"]]) == {"test"}
    assert sp["status"] == "PASS"


def test_paired_images_of_a_study_never_cross_splits():
    m = pd.read_csv(ROOT / "data/splits/iu_xray/iu_image_manifest.csv", dtype={"uid": str})
    assert studies_spanning_splits(m) == [] and images_in_multiple_splits(m) == []
    bad = pd.DataFrame({"uid": ["1", "1", "2"], "filename": ["a", "b", "c"], "split": ["train", "test", "val"]})   # frontal in train, lateral in test
    assert studies_spanning_splits(bad) == ["1"]
    assert images_in_multiple_splits(pd.DataFrame({"filename": ["a", "a"], "split": ["train", "val"]})) == ["a"]


def test_no_locked_test_study_appears_in_any_r1_output():
    test_ids = set(json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))["study_ids"]["locked_test"])
    for name in ("iu_study_findings.csv", "retrieval_corpus.csv", "iu_validation_classifier_outputs.csv", "per_query_metrics.csv", "retrieval_error_analysis.csv"):
        col = "uid"
        assert not (set(pd.read_csv(R1 / name, dtype={col: str})[col]) & test_ids), name
    res = pd.read_csv(R1 / "retrieval_results_top10.csv.gz", dtype={"query_uid": str, "retrieved_uid": str})
    assert not (set(res.query_uid) | set(res.retrieved_uid)) & test_ids


# ---------------------------------------------------------------- embeddings / FAISS
def test_embeddings_are_l2_normalised():
    x = l2_normalize(np.random.default_rng(0).normal(size=(50, 12)))
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6) and x.dtype == np.float32
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((2, 4)))
    stored = np.load(R1 / "corpus_embeddings.npy")
    assert np.allclose(np.linalg.norm(stored, axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        build_index(np.random.default_rng(1).normal(size=(5, 8)) * 3)       # un-normalised vectors are rejected


def test_embedding_dimension_is_detected_not_assumed():
    class Fake:
        def __init__(self, reported, actual): self.r, self.a = reported, actual
        def get_embedding_dimension(self): return self.r
        def encode(self, texts, **kw): return np.zeros((len(texts), self.a), dtype=np.float32)
    assert embedding_dimension(Fake(7, 7)) == 7
    with pytest.raises(RuntimeError):
        embedding_dimension(Fake(384, 128))
    meta = json.loads((R1 / "embedding_metadata.json").read_text(encoding="utf-8"))
    assert meta["embedding_dimension_verified"] == np.load(R1 / "corpus_embeddings.npy").shape[1]


def test_faiss_inner_product_equals_cosine_and_ids_map_correctly():
    rng = np.random.default_rng(2)
    emb = l2_normalize(rng.normal(size=(30, 16)))
    ids = [f"s{i}" for i in range(30)]
    index = build_index(emb)
    q = emb[[4, 17]]
    hits = search(index, ids, q, 5)
    assert [h[0][0] for h in hits] == ["s4", "s17"]
    assert hits[0][0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in hits[0]] == sorted([s for _, s in hits[0]], reverse=True)
    brute = emb @ q[0]
    assert [i for i, _ in hits[0]] == [ids[k] for k in np.argsort(-brute)[:5]]
    with pytest.raises(ValueError):
        search(index, ids[:-1], q, 3)
    ids_real = json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8"))
    import faiss
    assert faiss.read_index(str(R1 / "dense_minilm_flatip.faiss")).ntotal == len(ids_real) == len(set(ids_real))


# ---------------------------------------------------------------- self-retrieval / ordering
def test_self_retrieval_is_impossible_and_top_k_is_ordered():
    res = pd.read_csv(R1 / "retrieval_results_top10.csv.gz", dtype={"query_uid": str, "retrieved_uid": str})
    assert (res.retrieved_uid == res.query_uid).sum() == 0
    corpus = set(json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8")))
    assert set(res.retrieved_uid) <= corpus and not (set(res.query_uid) & corpus)
    for _, g in list(res.groupby(["retriever", "policy", "query_uid"]))[:400]:
        g = g.sort_values("rank")
        assert g["rank"].tolist() == list(range(1, 11)) and g.retrieved_uid.is_unique
        assert np.all(np.diff(g.score.to_numpy()) <= 1e-9)
    chk = json.loads((R1 / "self_retrieval_check.json").read_text(encoding="utf-8"))
    assert chk["status"] == "PASS" and chk["retrieved_equal_to_query_study"] == 0


def test_bm25_ranking_is_deterministic_with_stable_ties():
    texts = ["no pleural effusion", "pleural effusion is present", "cardiomegaly", "pleural effusion is present"]
    bm = BM25(["a", "b", "c", "d"], texts)
    r1, r2 = bm.search("pleural effusion", 4), bm.search("pleural effusion", 4)
    assert r1 == r2 and [i for i, _ in r1] == ["a", "b", "d", "c"]      # shorter doc first; identical docs b, d keep corpus order
    scores = [s for _, s in r1]
    assert scores == sorted(scores, reverse=True) and bm.search("cardiomegaly", 1)[0][0] == "c"
    assert r1[1][1] == r1[2][1]


# ---------------------------------------------------------------- query building
def test_query_building_is_deterministic_and_fixed():
    pred = _neg(**{"Pleural Effusion": True, "Cardiomegaly": True})
    assert classifier_query(pred) == classifier_query(dict(reversed(list(pred.items()))))
    assert classifier_query(pred)["query"] == "cardiomegaly pleural effusion"
    assert classifier_query(_neg(**{NO_FINDING: True}))["query"] == NORMAL_QUERY
    e = classifier_query(_neg())
    assert e["query"] == "" and e["status"] == "empty"
    assert phrase_query({"Pleural Effusion", "Cardiomegaly"}) == "cardiomegaly pleural effusion"
    assert oracle_query(frozenset({NO_FINDING}))[0] == NORMAL_QUERY and set(PHRASE) == set(ABN)


def test_gated_query_states_and_fallback():
    gates = {l: None for l in ABN}
    gates["Fracture"] = 0.5
    pred = _neg(Fracture=True, Cardiomegaly=True)
    prob = {l: 0.0 for l in LABELS}
    prob.update(Fracture=0.3, Cardiomegaly=0.2)
    q = gated_query(pred, prob, gates)
    assert q["states"]["Fracture"] == "low_confidence" and q["states"]["Cardiomegaly"] == "high_confidence" and q["query"] == "cardiomegaly"
    q = gated_query(_neg(Fracture=True, Edema=True), {**prob, "Edema": 0.4}, {**gates, "Edema": 0.9})
    assert q["status"] == "fallback_strongest_positive" and q["findings"] == ["Edema"]          # strongest positive, never invented
    q = gated_query(_neg(Fracture=True, **{NO_FINDING: True}), prob, gates)
    assert q["status"] == "no_finding" and q["query"] == NORMAL_QUERY and q["low_confidence"] == ["Fracture"]
    assert gated_query(_neg(), prob, gates)["status"] == "empty"


# ---------------------------------------------------------------- finding mapping
def test_finding_mapping_rules(mapping):
    assert parse_mesh("Opacity/lung/base;Cardiomegaly/mild")[1] == ("Cardiomegaly", ["mild"])
    assert map_study("Cardiomegaly/borderline", mapping)["findings"] == ["Cardiomegaly"]
    assert map_study("Cardiac Shadow/enlarged/mild", mapping)["findings"] == ["Cardiomegaly"]
    m = map_study("Cardiac Shadow/borderline", mapping)
    assert m["findings"] == [] and m["uncertain_terms"] == ["Cardiac Shadow"]
    m = map_study("Pulmonary Congestion/mild", mapping)                       # ambiguous: stored separately, never mapped to Edema
    assert m["findings"] == [] and m["uncertain_terms"] == ["Pulmonary Congestion"]
    assert map_study("Thickening/pleura/apex", mapping)["findings"] == ["Pleural Other"]
    assert map_study("Thickening/bronchovascular", mapping)["findings"] == []
    m = map_study("Spine/degenerative;Aorta/tortuous", mapping)
    assert m["findings"] == [] and set(m["unmapped_terms"]) == {"Spine", "Aorta"}
    assert map_study("normal", mapping)["normal"] is True and map_study("No Indexing", mapping)["no_indexing"]
    assert eval_set([], True) == frozenset({NO_FINDING}) and eval_set([], False) is None
    assert eval_set(["Edema", "Fracture"], False) == frozenset({"Edema", "Fracture"})
    assert "Pulmonary Congestion" not in mapping["direct"] and set(mapping["direct"].values()) <= set(ABN)


def test_text_cleaning_preserves_content_and_removes_artifacts():
    assert clean_section("<p>FINDINGS: The heart is normal. XXXX</p>") == "The heart is normal."
    assert clean_section(float("nan")) == "" and clean_section("None.") == ""
    t, src = build_retrieval_text("No effusion.", "Normal chest.")
    assert t == "No effusion. Normal chest." and src == "findings+impression"
    assert build_retrieval_text("Same text.", "same text") == ("Same text.", "both_identical")
    assert build_retrieval_text(float("nan"), "Only impression.") == ("Only impression.", "impression_only")


# ---------------------------------------------------------------- metrics
def test_jaccard_values_and_normal_handling():
    assert jaccard({"A", "B"}, {"B", "C"}) == pytest.approx(1 / 3)
    assert jaccard({"A"}, {"A"}) == 1.0 and jaccard({"A"}, {"B"}) == 0.0
    assert jaccard({NO_FINDING}, {NO_FINDING}) == 1.0 and jaccard({NO_FINDING}, {"Edema"}) == 0.0
    assert np.isnan(jaccard(set(), set()))


def test_ndcg_graded_gain_known_values():
    assert dcg([1, 0.5]) == pytest.approx(1 + 0.5 / np.log2(3))
    allg = [1.0, 0.5, 0.0, 0.0]
    assert ndcg_at_k([1.0, 0.5], allg, 2) == pytest.approx(1.0)
    assert ndcg_at_k([0.5, 1.0], allg, 2) == pytest.approx((0.5 + 1 / np.log2(3)) / (1 + 0.5 / np.log2(3)))
    assert ndcg_at_k([0.0, 0.0], allg, 2) == 0.0 and np.isnan(ndcg_at_k([0, 0], [0, 0, 0], 2))
    assert hit_and_rr([False, False, True], 2) == (0.0, 0.0) and hit_and_rr([False, False, True], 3) == (1.0, pytest.approx(1 / 3))


# ---------------------------------------------------------------- gating
def test_gate_selection_finds_a_useful_gate_and_rejects_noise():
    rng = np.random.default_rng(3)
    n = 6000
    p = np.clip(rng.beta(2, 5, n), 1e-4, 1 - 1e-4)
    y = rng.random(n) < 1 / (1 + np.exp(-14 * (p - 0.40)))          # strongly informative: higher probability -> more likely positive
    pos = p >= np.quantile(p, 0.7)
    g, scores = select_gate(p, y, pos)
    assert g > p[pos].min() and scores.max() >= scores[0]
    folds, pidx = rng.integers(0, 5, n), np.arange(n)
    cv = cross_fitted_gain(p, y, pos, folds, pidx, 200, 1)
    assert cv["oof_gain_f05"] > 0 and cv["ci95_low"] > 0
    y_noise = rng.random(n) < 0.3                                  # scores carry no information about y
    cv2 = cross_fitted_gain(p, y_noise, pos, folds, pidx, 200, 1)
    assert cv2["ci95_low"] <= 0
    assert float(fbeta(10, 0, 0)) == pytest.approx(1.0) and float(fbeta(0, 0, 0)) == 0.0


def test_gate_file_is_validation_derived_and_leaves_classifier_untouched():
    g = json.loads((R1 / "precision_aware_query_gates.json").read_text(encoding="utf-8"))
    assert "locked classifier test set not used" in g["derived_from"] and set(g["gates"]) == set(ABN)
    pol = load_final_policy(EXP / "c4_operating_policy/final_operating_policy.json")
    for j, l in enumerate(LABELS):
        if l in g["gates"]:
            assert g["gates"][l]["frozen_raw_threshold"] == pol["thresholds"][j]
    for l, d in g["gates"].items():
        assert (d["gate_calibrated_probability"] is not None) == d["gate_adopted"]
        if d["gate_adopted"]:
            assert d["gate_calibrated_probability"] > d["frozen_calibrated_equivalent_threshold"] - 1e-9


# ---------------------------------------------------------------- frozen classifier integrity
def test_frozen_classifier_artifacts_are_intact_and_iu_decisions_follow_the_frozen_policy():
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    assert all(verify_freeze(man_path, ROOT).values())
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert sha256_file(ROOT / man["checkpoint"]) == man["checkpoint_sha256"]
    pol = load_final_policy(EXP / "c4_operating_policy/final_operating_policy.json")
    out = pd.read_csv(R1 / "iu_validation_classifier_outputs.csv", dtype={"uid": str}, float_precision="round_trip")
    scores = np.column_stack([out[f"{l}__score"] for l in LABELS])
    _, final = apply_final_policy(scores, pol)
    assert np.array_equal(final.astype(int), np.column_stack([out[f"{l}__pred_final"] for l in LABELS]))
    prob = np.column_stack([out[f"{l}__prob"] for l in LABELS])
    assert prob.min() >= 0 and prob.max() <= 1 and not np.allclose(prob.sum(1), 1.0)
    run = json.loads((R1 / "iu_classifier_run.json").read_text(encoding="utf-8"))
    assert run["classifier_tuned_on_iu"] is False and run["checkpoint_sha256"] == man["checkpoint_sha256"]
