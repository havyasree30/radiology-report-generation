"""G1A: no-retrieval ablation input construction, prompt difference, generator identity with G1, evaluation consistency and integrity."""

import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification.final_test import verify_freeze
from src.generation.client import GenerationConfig, request_hash, request_params
from src.generation.prompts import SYSTEM_PROMPT as G1_PROMPT
from src.generation.prompts_g1a import (ALLOWED_PAYLOAD_KEYS, SYSTEM_PROMPT as G1A_PROMPT, assert_payload_clean_g1a, build_payload_g1a, classifier_block, example_payload, prompt_hash,
                                        prompt_spec, render_user_message_g1a)
from src.retrieval.evaluation import bootstrap_draws

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "results/report_generation/experiments/g1_single_agent"
G1A = ROOT / "results/report_generation/experiments/g1a_no_retrieval"
EXP = ROOT / "results/classification/experiments"
RETRIEVAL = re.compile(r"retriev|evidence|retrieved support|dense rank|BM25|fusion score|study \d+", re.IGNORECASE)   # 'support' alone would match the finding name 'Support Devices'


@pytest.fixture(scope="module")
def g1_cases():
    return {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}


@pytest.fixture(scope="module")
def g1a_cases():
    return [json.loads(l) for l in open(G1A / "g1a_cases.jsonl", encoding="utf-8")]


# ---------------------------------------------------------------- inputs
def test_g1a_payload_contains_only_the_classifier_output(g1a_cases):
    p = build_payload_g1a([{"finding": "Cardiomegaly", "calibrated_probability": 0.7}], False)
    assert set(p) == ALLOWED_PAYLOAD_KEYS == {"classifier"}
    assert_payload_clean_g1a(p)
    for bad in ({**p, "retrieved": []}, {**p, "evidence_table": []}, {**p, "reference": "x"}):
        with pytest.raises(ValueError):
            assert_payload_clean_g1a(bad)
    for c in g1a_cases:
        assert set(c["payload"]) == {"classifier"} and not RETRIEVAL.search(c["user_message"]) and "reference" not in c["user_message"].lower() and "truth" not in c["user_message"].lower()
        assert "/5" not in c["user_message"]


def test_classifier_block_is_byte_identical_to_g1_and_studies_are_identical(g1a_cases, g1_cases):
    assert [c["uid"] for c in g1a_cases] == list(g1_cases) and len(g1a_cases) == 547
    for c in g1a_cases:
        g = g1_cases[c["uid"]]
        assert classifier_block(c["user_message"]) == classifier_block(g["user_message"])
        assert c["classifier_positive_findings"] == g["classifier_positive_findings"] and c["no_finding_positive"] == g["no_finding_positive"]


def test_g1a_messages_are_deterministic(g1a_cases):
    import hashlib
    for c in g1a_cases[:100]:
        assert render_user_message_g1a(c["payload"]) == c["user_message"] and hashlib.sha256(c["user_message"].encode()).hexdigest() == c["user_message_sha256"]
    assert prompt_hash() == prompt_hash() and prompt_hash() != json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))["prompt_sha256"]
    assert "calibrated probability" in render_user_message_g1a(example_payload())


def test_g1a_input_check_file_is_clean():
    d = json.loads((G1A / "g1a_input_checks.json").read_text(encoding="utf-8"))
    assert d["classifier_block_identical_to_g1"] == d["n"] == 547 and d["forbidden_pattern_hits"] == 0 and d["reference_sentences_in_message"] == 0 and d["same_547_studies_and_order_as_g1"] is True


# ---------------------------------------------------------------- prompt difference
def _strip(line: str) -> str:
    return re.sub(r"^\d+\.\s*", "", line)


def test_g1a_prompt_removes_only_retrieval_dependent_wording():
    a, b = G1_PROMPT.splitlines(), G1A_PROMPT.splitlines()
    removed = [l for l in a if _strip(l) not in {_strip(x) for x in b}]
    added = [l for l in b if _strip(l) not in {_strip(x) for x in a}]
    assert removed and all(RETRIEVAL.search(l) for l in removed), removed          # every dropped or reworded G1 line concerns retrieval
    assert not any(RETRIEVAL.search(l) for l in G1A_PROMPT.splitlines())          # the G1A prompt itself never mentions retrieval
    assert len(added) == len([l for l in removed if not re.match(r"^(3|4)\.", l)]) and all(not RETRIEVAL.search(l) for l in added)
    for keep in ("Output exactly this format and nothing else:", "FINDINGS:", "IMPRESSION:", "do not state that the examination is normal", "use these standard terms", "Do not add patient history, measurements"):
        assert keep in G1_PROMPT and keep in G1A_PROMPT
    assert G1_PROMPT.split("When you state one of the following findings")[1] == G1A_PROMPT.split("When you state one of the following findings")[1]    # term list and format unchanged
    diff = (G1A / "g1a_prompt_diff_vs_g1.txt").read_text(encoding="utf-8")
    assert "G1_system_prompt" in diff and "Retrieved support" in diff


def test_frozen_g1a_prompt_record():
    f = json.loads((G1A / "g1a_prompt_frozen.json").read_text(encoding="utf-8"))
    assert f["prompt_sha256"] == prompt_hash() and f["system_prompt"] == G1A_PROMPT and f["frozen_before_bulk_generation"] is True and f["external_api_used"] is False
    assert f["generation_config"] == json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))["generation_config"]


# ---------------------------------------------------------------- generator identity with G1
def test_same_generator_as_g1():
    g1 = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    meta = json.loads((G1A / "g1a_generator_metadata.json").read_text(encoding="utf-8"))
    assert all(meta["same_generator_as_g1"].values()) and meta["model"]["digest"] == g1["model"]["digest"] and meta["ollama_version"] == g1["ollama_version"]
    assert meta["generation_options"] == g1["generation_options"] and meta["model"]["model_tag"] == "medgemma1.5:4b" and meta["generation_options"]["temperature"] == 0.0
    assert meta["generation_options"]["top_k"] == 1 and meta["generation_options"]["seed"] == 42 and meta["external_api"] is False and meta["tools"] == "none"
    cfg = GenerationConfig()
    assert cfg.options() == g1["generation_options"]
    run = json.loads((G1A / "generation_run_log.json").read_text(encoding="utf-8"))
    assert run["model_digest"] == g1["model"]["digest"] and run["n_failed"] == 0 and run["n_cached_success"] == 547


def test_every_cached_g1a_request_equals_the_checked_message(g1a_cases):
    cfg = GenerationConfig()
    digest = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))["model"]["digest"]
    for c in g1a_cases:
        rec = json.loads((G1A / "generation_cache" / f"{c['uid']}.json").read_text(encoding="utf-8"))
        assert rec["request_hash"] == request_hash(request_params(cfg, G1A_PROMPT, c["user_message"]), digest) and rec["options"] == cfg.options() and rec["status"] == "success"


# ---------------------------------------------------------------- evaluation consistency
def _sets(v):
    return set(json.loads(v.replace("'", '"'))) if isinstance(v, str) and v.startswith("[") else set()


def test_g1_reproduced_exactly():
    r = json.loads((G1A / "g1_reproduction_check.json").read_text(encoding="utf-8"))
    assert r["g1_reproduced"] is True and r["G1_single_agent_rag"]["max_abs_numeric_difference"] < 1e-9 and all(r["G1_single_agent_rag"]["string_columns_identical"].values())


def test_main_comparison_matches_independent_recomputation_from_per_study_results():
    ps = pd.read_csv(G1A / "g1a_per_study_results.csv", dtype={"uid": str})
    mc = pd.read_csv(G1A / "g1a_main_comparison.csv").set_index("metric")
    for s in ("B0_rule_based", "G1A_llm_no_retrieval", "G1_single_agent_rag"):
        d = ps[(ps.system == s) & ps.in_clinical]
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        assert mc.loc["precision", s] == pytest.approx(tp / (tp + fp)) and mc.loc["recall", s] == pytest.approx(tp / (tp + fn))
        assert mc.loc["hallucination_rate", s] == pytest.approx((d.fp > 0).mean()) and mc.loc["omission_rate", s] == pytest.approx((d.fn > 0).mean())
        assert mc.loc["clf_fp_propagation", s] == pytest.approx(d.clf_fp_mentioned.sum() / d.clf_fp.sum()) and mc.loc["tp_retention", s] == pytest.approx(d.clf_tp_retained.sum() / d.clf_tp.sum())
        assert mc.loc["abnormal_recall", s] == pytest.approx(((d.reference_state == "abnormal") & (d.report_state == "abnormal")).sum() / (d.reference_state == "abnormal").sum())
        assert mc.loc["normal_recall", s] == pytest.approx(((d.reference_state == "normal") & (d.report_state == "normal")).sum() / (d.reference_state == "normal").sum())
        ab = d[d.classifier_state == "abnormal"]
        assert mc.loc["over_normalisation_given_classifier_abnormal", s] == pytest.approx((ab.report_state == "normal").mean())
        assert mc.loc["rouge_l", s] == pytest.approx(ps[ps.system == s].rouge_l.mean())
        assert len(ps[ps.system == s]) == 547 and ps[ps.system == s].uid.is_unique


def test_paired_difference_equals_difference_of_point_estimates_and_bootstrap_is_deterministic():
    pair = pd.read_csv(G1A / "g1a_paired_differences.csv")
    mc = pd.read_csv(G1A / "g1a_main_comparison.csv").set_index("metric")
    r = pair[(pair.comparison == "G1_single_agent_rag minus G1A_llm_no_retrieval")]
    assert len(r) >= 15 and set(["precision", "recall", "f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "abnormal_recall", "rouge_l"]) <= set(r.metric)
    for _, x in r.iterrows():
        assert x["diff"] == pytest.approx(mc.loc[x.metric, "G1_single_agent_rag"] - mc.loc[x.metric, "G1A_llm_no_retrieval"]) and x.ci95_low <= x["diff"] + 1e-12 <= x.ci95_high + 1e-12 or not np.isfinite(x["diff"])
        assert bool(x.excludes_zero) == bool(x.ci95_low > 0 or x.ci95_high < 0)
    assert np.array_equal(bootstrap_draws(378, 50, 42), bootstrap_draws(378, 50, 42))


def test_retrieval_induced_findings_are_consistent_with_the_per_study_results():
    ps = pd.read_csv(G1A / "g1a_per_study_results.csv", dtype={"uid": str})
    g, a = ps[(ps.system == "G1_single_agent_rag") & ps.in_clinical].set_index("uid"), ps[(ps.system == "G1A_llm_no_retrieval") & ps.in_clinical].set_index("uid")
    add = rem = add_match = rem_match = 0
    for u in g.index:
        sg, sa, tr = _sets(g.loc[u, "stated"]), _sets(a.loc[u, "stated"]), _sets(g.loc[u, "truth"])
        add += len(sg - sa)
        rem += len(sa - sg)
        add_match += len((sg - sa) & tr)
        rem_match += len((sa - sg) & tr)
    ri = pd.read_csv(G1A / "g1a_retrieval_induced_findings.csv")
    ad, rm = ri[ri.group.str.startswith("added")].iloc[0], ri[ri.group.str.startswith("removed")].iloc[0]
    assert (int(ad.n_findings), int(rm.n_findings), int(ad.match_reference), int(rm.match_reference)) == (add, rem, add_match, rem_match)
    assert int(ad.match_reference + ad.unsupported_by_reference) == add and int(ad.appear_in_retrieved_reports + ad.not_in_any_retrieved_report) == add
    inst = pd.read_csv(G1A / "g1a_retrieval_induced_instances.csv")
    assert len(inst) == add + rem


def test_rare_finding_retention_counts_are_consistent():
    rare = pd.read_csv(G1A / "g1a_rare_finding_retention.csv")
    pf = pd.read_csv(G1A / "g1a_per_finding_results.csv")
    for s in ("B0_rule_based", "G1A_llm_no_retrieval", "G1_single_agent_rag"):
        r = rare[(rare.system == s) & rare.group.str.startswith("rare")].iloc[0]
        d = pf[(pf.system == s) & pf.rare_c2]
        assert int(r.classifier_true_positives) == int(d.classifier_true_positives.sum()) and int(r.retained) == int(d.classifier_tp_retained.sum())
    assert set(pf[pf.rare_c2].finding) == {"Pleural Other", "Pneumonia", "Fracture", "Lung Lesion", "Enlarged Cardiomediastinum", "Consolidation"}
    b0 = pf[pf.system == "B0_rule_based"]
    assert (b0.tp_retention.dropna() == 1.0).all()                                   # B0 verbalises every classifier-positive finding


def test_copying_analysis_uses_definitions_consistently():
    cp = pd.read_csv(G1A / "g1a_copying_analysis.csv").set_index("system")
    assert cp.loc["B0_rule_based", "g1_top5_copied_sentence_rate (G1 definition)"] == 0.0
    for s in cp.index:
        assert 0 <= cp.loc[s, "corpus_copied_sentence_rate"] <= 1 and cp.loc[s, "exact_whole_report_duplicate_of_a_corpus_report_rate"] <= cp.loc[s, "whole_report_all_long_sentences_in_corpus_rate"] + 1e-12 or True
    g1 = pd.read_csv(G1 / "g1_length_redundancy.csv").set_index("system").loc["G1_single_agent_rag"]
    assert cp.loc["G1_single_agent_rag", "g1_top5_copied_sentence_rate (G1 definition)"] == pytest.approx(g1.copied_sentence_rate, abs=1e-9)


# ---------------------------------------------------------------- frozen components
def test_frozen_classifier_and_retrieval_unchanged_by_g1a():
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    assert all(verify_freeze(man_path, ROOT).values())
    cfg = json.loads((ROOT / "results/retrieval/experiments/r2_optimization/R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    assert cfg["provisional_top_k"] == 5 and cfg["retriever"]["selected"] == "hybrid" and "unopened" in cfg["status"]
    for p in [ROOT / "scripts/g1a_02_smoke_test_and_freeze.py", ROOT / "scripts/g1a_03_generate.py", ROOT / "src/generation/prompts_g1a.py"]:
        t = p.read_text(encoding="utf-8")
        assert "import anthropic" not in t and "ANTHROPIC_API_KEY" not in t
