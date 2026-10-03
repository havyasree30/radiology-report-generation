"""G1B: sham-retrieval control. The generator input differs from G1 only in the identity of the five context reports."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from scripts.g1b_01_build_cases import K, SEED, sham_ids
from src.classification.final_test import sha256_file
from src.generation.client import GenerationConfig, request_hash, request_params
from src.generation.prompts import SYSTEM_PROMPT, assert_payload_clean, prompt_hash, render_user_message

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "results/report_generation/experiments/g1_single_agent"
G1B = ROOT / "results/report_generation/experiments/g1b_sham_retrieval"
R1 = ROOT / "results/retrieval/experiments/r1_baseline"
SKELETON = re.compile(r"^(A\. |B\. |C\. |Final positive|No Finding:|Write the|\[\d\] study |None: no retrieval|\| |\(no classifier)")


@pytest.fixture(scope="module")
def g1_cases():
    return {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}


@pytest.fixture(scope="module")
def g1b_cases():
    return [json.loads(l) for l in open(G1B / "g1b_cases.jsonl", encoding="utf-8")]


@pytest.fixture(scope="module")
def pool():
    return sorted(json.loads((R1 / "corpus_study_ids.json").read_text(encoding="utf-8")), key=int)


def test_sham_selection_is_deterministic_and_respects_exclusions(pool):
    ex = {"7", "8", "9", "10", "11"}
    a, b = sham_ids("123", pool, ex), sham_ids("123", pool, ex)
    assert a == b and len(a) == len(set(a)) == K and not set(a) & ex and "123" not in a
    assert sham_ids("124", pool, ex) != a                                   # different study, different draw

    assert SEED == 42


def test_sham_selection_uses_only_the_study_id(pool):
    import inspect
    src = inspect.getsource(sham_ids)
    assert "reference" not in src and "classifier" not in src and "truth" not in src


def test_g1b_covers_the_same_studies_and_empty_query_is_identical(g1_cases, g1b_cases):
    assert [c["uid"] for c in g1b_cases] == list(g1_cases)
    for c in g1b_cases:
        g = g1_cases[c["uid"]]
        assert c["classifier_positive_findings"] == g["classifier_positive_findings"] and c["classifier_probabilities"] == g["classifier_probabilities"]
        if not g["retrieved"]:
            assert c["user_message"] == g["user_message"] and not c["retrieved"]


def test_g1b_message_skeleton_equals_g1_and_context_ids_differ(g1_cases, g1b_cases):
    skel = lambda m: [re.sub(r"study \d+", "study N", l) for l in m.splitlines() if SKELETON.match(l) and not l.startswith("| ")]  # noqa: E731
    for c in g1b_cases:
        g = g1_cases[c["uid"]]
        if not g["retrieved"]:
            continue
        assert skel(c["user_message"]) == skel(g["user_message"])
        assert [r["rank"] for r in c["retrieved"]] == [1, 2, 3, 4, 5] and len(c["retrieved"]) == K
        assert not {r["study_id"] for r in c["retrieved"]} & {r["study_id"] for r in g["retrieved"]}
        assert c["uid"] not in {r["study_id"] for r in c["retrieved"]}
        assert render_user_message(c["payload"]) == c["user_message"]


def test_sham_context_is_training_corpus_only_and_unlabelled(g1b_cases, pool):
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    test_ids, corpus = set(split["study_ids"]["locked_test"]), set(pool)
    for c in g1b_cases:
        assert_payload_clean(c["payload"], c["uid"], corpus, test_ids)
        for ln in c["user_message"].splitlines():
            if SKELETON.match(ln):
                assert not re.search(r"sham|random|unrelated|control", ln, re.IGNORECASE)
    assert not re.search(r"sham|random|unrelated", SYSTEM_PROMPT, re.IGNORECASE)


def test_no_reference_sentence_outside_context(g1b_cases):
    from src.generation.metrics import norm_sentence, sentences
    g1 = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}
    for c in g1b_cases[:150]:
        ctx = norm_sentence(" ".join(r["findings"] + " " + r["impression"] for r in c["retrieved"]))
        non_ctx = norm_sentence(c["user_message"])
        for s in sentences(g1[c["uid"]]["reference"]["combined"]):
            n = norm_sentence(s)
            if len(n.split()) >= 6 and n in non_ctx:
                assert n in ctx


def test_frozen_mapping_hash_and_prompt_equal_g1():
    fz = json.loads((G1B / "g1b_sham_mapping_frozen.json").read_text(encoding="utf-8"))
    assert fz["mapping_sha256"] == sha256_file(G1B / "g1b_sham_mapping.csv")
    assert fz["g1_prompt_sha256"] == prompt_hash() == json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))["prompt_sha256"]
    m = pd.read_csv(G1B / "g1b_sham_mapping.csv", dtype=str)
    assert m.groupby("uid").size().eq(K).all() and not (m.sham_study_id == m.replaced_true_study_id).any()


def test_g1b_requests_use_the_g1_generator(g1b_cases):
    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    cfg = GenerationConfig(**{k: v for k, v in json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    assert cfg.options() == g1_gen["generation_options"]
    meta = json.loads((G1B / "g1b_generator_metadata.json").read_text(encoding="utf-8"))
    assert all(meta["same_generator_as_g1"].values()) and meta["external_api"] is False
    for c in g1b_cases[:20]:
        rec = json.loads((G1B / "generation_cache" / f"{c['uid']}.json").read_text(encoding="utf-8"))
        assert rec["request_hash"] == request_hash(request_params(cfg, SYSTEM_PROMPT, c["user_message"]), g1_gen["model"]["digest"])


def test_g1b_code_has_no_external_api_or_fitting():
    for p in sorted((ROOT / "scripts").glob("g1b_*.py")) + [ROOT / "tests/test_g1b.py"]:
        t = p.read_text(encoding="utf-8")
        if p.name not in ("test_g1b.py", "g1b_06_integrity.py"):          # the integrity script names the banned calls in order to scan for them (AST-checked by that script)
            assert not re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai", t)
            assert "allow_locked=True" not in t and "fit_platt" not in t and "fit_calibrator" not in t


def test_primary_comparison_is_g1_minus_g1b():
    s = json.loads((G1B / "g1b_summary.json").read_text(encoding="utf-8"))
    assert s["primary_comparison"] == "G1_single_agent_rag minus G1B_llm_sham_retrieval" and s["n_paired_studies"] == 547
    pd_ = pd.read_csv(G1B / "g1b_paired_differences.csv")
    first = pd_.iloc[0].comparison
    assert first == s["primary_comparison"]
    need = {"precision", "recall", "f1", "macro_f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall", "rouge_l"}
    assert need <= set(pd_[pd_.comparison == first].metric)


def test_stored_g1_results_reproduced():
    r = json.loads((G1B / "g1_reproduction_check.json").read_text(encoding="utf-8"))
    assert r["g1_and_g1a_reproduced"] is True
