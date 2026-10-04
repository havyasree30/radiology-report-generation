"""G2 structured multi-agent RAG: Agent 1 schema/validation, agent isolation from raw retrieval and from the reference, deterministic message construction,
output and critic parsing, metric definitions (FP propagation, TP retention, copying), empty-query behaviour, freeze and integrity."""

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from src.classification.final_test import sha256_file, verify_freeze
from src.generation import g2_agents as g2
from src.generation.client import GenerationConfig, ResponseCache
from src.generation.extraction import ABNORMAL
from src.generation.g2_pipeline import AGENTS, case_inputs, run_case
from src.generation.metrics import classifier_propagation, copy_stats
from src.generation.study_eval import FS

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "results/report_generation/experiments/g1_single_agent"
G2 = ROOT / "results/report_generation/experiments/g2_multi_agent"
EXP = ROOT / "results/classification/experiments"
R2 = ROOT / "results/retrieval/experiments/r2_optimization"

RETRIEVED = [{"rank": 1, "findings": "The cardiac silhouette is enlarged. There is a small right pleural effusion.", "impression": "Cardiomegaly with small right pleural effusion."},
             {"rank": 2, "findings": "Heart size is enlarged. Lungs are clear.", "impression": "Cardiomegaly."},
             {"rank": 3, "findings": "No focal consolidation. Heart size normal.", "impression": "No acute disease."},
             {"rank": 4, "findings": "The heart is borderline enlarged.", "impression": "Possible mild cardiomegaly."},
             {"rank": 5, "findings": "Normal chest.", "impression": "No acute cardiopulmonary abnormality."}]
POS = [{"finding": "Cardiomegaly", "calibrated_probability": 0.71}, {"finding": "Pleural Effusion", "calibrated_probability": 0.33}]
TEXTS = [r["findings"] + " " + r["impression"] for r in RETRIEVED]


class FakeClient:
    """Returns canned model answers by agent (recognised through the system prompt); counts calls."""

    def __init__(self, answers):
        self.answers, self.calls, self.params = answers, [], []

    def chat(self, params):
        sysmsg = params["messages"][0]["content"]
        agent = next(a for a, p in g2.PROMPTS.items() if p == sysmsg)
        self.calls.append(agent)
        self.params.append(params)
        return {"message": {"content": self.answers[agent]}, "done_reason": "stop", "model": "fake", "eval_count": 7, "prompt_eval_count": 11, "total_duration": 1, "load_duration": 1}


A1_RAW = {"classifier_findings": [{"finding": "Cardiomegaly", "supporting_ranks": [1, 2, 4, 9], "evidence_summary": "Enlarged heart in several reports.", "support_status": "unsupported"},
                                  {"finding": "Pleural Effusion", "supporting_ranks": [1], "evidence_summary": "Small effusion in one report.", "support_status": "supported"},
                                  {"finding": "Edema", "supporting_ranks": [2], "evidence_summary": "not a classifier finding", "support_status": "supported"}],
          "retrieval_only_candidates": [{"finding": "Cardiomegaly", "supporting_ranks": [1], "evidence_summary": "duplicate of a classifier finding"}, {"finding": "Atelectasis", "supporting_ranks": [], "evidence_summary": "no rank"},
                                        {"finding": "Lung Opacity", "supporting_ranks": [3, 3], "evidence_summary": "x"}], "normal_report_ranks": [5, 5, 7]}


# ---------------------------------------------------------------- Agent 1 schema and validation
def test_agent1_schema_is_structured_and_closed_vocabulary():
    s = g2.AGENT1_SCHEMA
    assert set(s["required"]) == {"classifier_findings", "retrieval_only_candidates", "normal_report_ranks"}
    item = s["properties"]["classifier_findings"]["items"]
    assert set(item["required"]) == {"finding", "supporting_ranks", "evidence_summary", "support_status"}
    assert item["properties"]["support_status"]["enum"] == ["supported", "partially_supported", "unsupported"] == list(g2.STATUS)
    assert item["properties"]["finding"]["enum"] == list(ABNORMAL)
    assert "retrieval_only" not in item["properties"]["support_status"]["enum"]                 # candidates are a separate list and can never be a classifier-finding status


def test_agent1_validation_recomputes_counts_status_and_restricts_candidates():
    ev, flags = g2.validate_agent1(A1_RAW, POS, False, RETRIEVED)
    cf = {e["finding"]: e for e in ev["classifier_findings"]}
    assert list(cf) == ["Cardiomegaly", "Pleural Effusion"]                                     # exactly one entry per classifier positive, fixed order, extra entry ignored
    assert cf["Cardiomegaly"]["supporting_retrieved_report_ranks"] == [1, 2, 4] and cf["Cardiomegaly"]["retrieval_support_count"] == 3     # rank 9 is not a retrieved report
    assert cf["Cardiomegaly"]["support_status"] == "supported" and cf["Cardiomegaly"]["agent_stated_status"] == "unsupported"
    assert cf["Pleural Effusion"]["support_status"] == "partially_supported" and cf["Pleural Effusion"]["classifier_probability"] == 0.33
    assert [c["finding"] for c in ev["retrieval_only_candidates"]] == ["Lung Opacity"]            # classifier findings and rank-less candidates are dropped
    assert ev["retrieval_only_candidates"][0]["status"] == "retrieval_only" and ev["retrieval_only_candidates"][0]["supporting_retrieved_report_ranks"] == [3]
    assert ev["retrieved_report_ranks_describing_a_normal_study"] == [5]
    assert "agent1_status_inconsistent_with_count" in flags and "agent1_extra_or_duplicate_classifier_entry" in flags


def test_agent1_validation_handles_omitted_unparseable_and_quoted_prose():
    ev, flags = g2.validate_agent1({"classifier_findings": [], "retrieval_only_candidates": [], "normal_report_ranks": []}, POS, False, RETRIEVED)
    assert [e["support_status"] for e in ev["classifier_findings"]] == ["unsupported", "unsupported"] and "agent1_omitted_classifier_finding" in flags
    ev, flags = g2.validate_agent1(None, POS, False, RETRIEVED)
    assert "agent1_unparseable" in flags and len(ev["classifier_findings"]) == 2
    quoted = {"classifier_findings": [{"finding": "Cardiomegaly", "supporting_ranks": [1], "evidence_summary": "The cardiac silhouette is enlarged. There is a small right pleural effusion.", "support_status": "partially_supported"}],
              "retrieval_only_candidates": [], "normal_report_ranks": []}
    ev, flags = g2.validate_agent1(quoted, POS[:1], False, RETRIEVED)
    assert "agent1_summary_replaced" in flags and ev["classifier_findings"][0]["evidence_summary"] == "Stated in 1 of 5 retrieved reports."
    assert not g2.shares_ngram(ev["classifier_findings"][0]["evidence_summary"], TEXTS)


def test_status_rule_is_the_predeclared_count_rule():
    assert [g2.status_from_count(n) for n in (0, 1, 2, 5)] == ["unsupported", "partially_supported", "supported", "supported"]


# ---------------------------------------------------------------- isolation of Agents 2 and 3, determinism, no reference
def test_agent2_and_agent3_builders_have_no_raw_retrieval_or_reference_parameter():
    for fn in (g2.build_agent2_message, g2.build_agent3_message):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"retrieved", "retrieved_texts", "reference", "truth", "reference_report", "reference_findings"}, params
    assert set(inspect.signature(g2.build_agent1_message).parameters) == {"positives", "no_finding_positive", "retrieved"}      # the only builder with retrieved text; no reference parameter


def test_agent2_and_agent3_messages_contain_no_retrieved_prose():
    ev, _ = g2.validate_agent1(A1_RAW, POS, False, RETRIEVED)
    draft = {"findings": "Cardiomegaly is present.", "impression": "Cardiomegaly."}
    m2, m3 = g2.build_agent2_message(POS, False, ev), g2.build_agent3_message(POS, False, ev, draft)
    for m in (m2, m3):
        assert not g2.shares_ngram(m, TEXTS) and "agent_stated_status" not in m and "RETRIEVED" not in m
        assert "supporting_retrieved_report_ranks" in m and "classifier_probability" in m


def test_message_construction_is_deterministic():
    ev, _ = g2.validate_agent1(A1_RAW, POS, False, RETRIEVED)
    assert g2.build_agent1_message(POS, False, RETRIEVED) == g2.build_agent1_message(POS, False, RETRIEVED)
    assert g2.build_agent2_message(POS, False, ev) == g2.build_agent2_message(POS, False, ev)
    assert g2.message_sha256("a", "b") == g2.message_sha256("a", "b") != g2.message_sha256("a", "c")


def test_fake_client_pipeline_isolation_and_cache(tmp_path):
    case = {"uid": "1", "classifier_positive_findings": ["Cardiomegaly", "Pleural Effusion"], "classifier_probabilities": {f: 0.5 for f in ABNORMAL}, "no_finding_positive": False, "retrieved": RETRIEVED}
    answers = {"agent1_evidence_verifier": json.dumps(A1_RAW), "agent2_grounded_report_writer": json.dumps({"findings": "Cardiomegaly is present.", "impression": "Cardiomegaly."}),
               "agent3_grounding_critic": json.dumps({"issues_detected": ["none"], "decision": "APPROVE", "findings": "Cardiomegaly is present.", "impression": "Cardiomegaly."})}
    client, caches = FakeClient(answers), {a: ResponseCache(tmp_path / a) for a in AGENTS}
    res = run_case(case, client, GenerationConfig(), "digest", caches)
    assert client.calls == list(AGENTS) and res["critic"]["action"] == "approve" and res["final"]["findings"] == "Cardiomegaly is present."
    for a in AGENTS[1:]:
        assert not g2.shares_ngram(res["messages"][a], TEXTS)
    assert all("format" in p for p in client.params)                                            # structured output requested for every agent
    run_case(case, client, GenerationConfig(), "digest", caches)
    assert len(client.calls) == 3                                                               # second run served from the cache


def test_empty_query_study_skips_agent1_and_is_not_declared_normal(tmp_path):
    case = {"uid": "2", "classifier_positive_findings": [], "classifier_probabilities": {f: 0.0 for f in ABNORMAL}, "no_finding_positive": False, "retrieved": []}
    answers = {"agent2_grounded_report_writer": json.dumps({"findings": "No finding could be determined.", "impression": "Indeterminate."}),
               "agent3_grounding_critic": json.dumps({"issues_detected": ["none"], "decision": "APPROVE", "findings": "No finding could be determined.", "impression": "Indeterminate."})}
    client = FakeClient(answers)
    res = run_case(case, client, GenerationConfig(), "digest", {a: ResponseCache(tmp_path / a) for a in AGENTS})
    assert client.calls == list(AGENTS[1:]) and res["agent1_flags"] == ["agent1_skipped_no_query"]
    assert res["evidence"]["classifier_output_state"] == "no_output" and res["evidence"]["classifier_findings"] == []
    assert "State: no_output" in res["messages"][AGENTS[1]] and "No Finding: negative" in res["messages"][AGENTS[1]]
    assert "no_output" in g2.AGENT2_SYSTEM and "do not state that the examination is normal" in g2.AGENT2_SYSTEM


# ---------------------------------------------------------------- parsing
def test_parse_json_text_and_draft():
    assert g2.parse_json_text('{"a": 1}') == {"a": 1} and g2.parse_json_text('noise {"a": 2} tail') == {"a": 2} and g2.parse_json_text("not json") is None and g2.parse_json_text("[1]") is None
    d = g2.parse_draft('{"findings": " Heart size   normal. ", "impression": "No acute disease."}')
    assert d == {"findings": "Heart size normal.", "impression": "No acute disease.", "format_ok": True, "format_flags": []}
    bad = g2.parse_draft("FINDINGS: x\nIMPRESSION: y")
    assert bad["findings"] == "x" and bad["format_ok"] is False and "agent2_json_unparseable" in bad["format_flags"]
    assert g2.parse_draft('{"findings": "", "impression": "y"}')["format_ok"] is False


DRAFT = {"findings": "Cardiomegaly is present.", "impression": "Cardiomegaly.", "format_ok": True, "format_flags": []}


def test_critic_action_parsing():
    ap = g2.parse_critic(json.dumps({"issues_detected": ["none"], "decision": "APPROVE", "findings": "Cardiomegaly is present.", "impression": "Cardiomegaly."}), DRAFT)
    assert ap["action"] == "approve" and ap["report"] == DRAFT and ap["model_copy_identical_to_draft"] is True
    ap2 = g2.parse_critic(json.dumps({"issues_detected": ["none"], "decision": "APPROVE", "findings": "Something else.", "impression": "Other."}), DRAFT)
    assert ap2["action"] == "approve" and ap2["report"] == DRAFT and "approve_with_changed_copy_ignored" in ap2["flags"]               # the draft is kept by code
    rv = g2.parse_critic(json.dumps({"issues_detected": ["unsupported_finding_stated"], "decision": "REVISE", "findings": "No acute abnormality.", "impression": "Normal."}), DRAFT)
    assert rv["action"] == "revise" and rv["report"]["findings"] == "No acute abnormality." and rv["issues"] == ["unsupported_finding_stated"]
    same = g2.parse_critic(json.dumps({"issues_detected": ["excessive_certainty"], "decision": "REVISE", "findings": "Cardiomegaly is present.", "impression": "Cardiomegaly."}), DRAFT)
    assert same["action"] == "approve" and "revise_but_identical" in same["flags"]
    for text in ("garbage", json.dumps({"decision": "MAYBE"}), json.dumps({"decision": "REVISE", "findings": "", "impression": ""})):
        r = g2.parse_critic(text, DRAFT)
        assert r["report"] == DRAFT and r["action"] not in ("approve", "revise")                                                         # unusable answers keep the draft and are flagged
    assert set(g2.AGENT3_SCHEMA["required"]) == {"issues_detected", "decision", "findings", "impression"} and g2.AGENT3_SCHEMA["properties"]["decision"]["enum"] == ["APPROVE", "REVISE"]


# ---------------------------------------------------------------- metric definitions
def test_fp_propagation_and_tp_retention_definitions():
    clf, truth = FS({"Cardiomegaly", "Edema", "Atelectasis"}), FS({"Cardiomegaly", "Atelectasis"})            # Edema is a classifier false positive
    both = classifier_propagation(clf, truth, FS({"Cardiomegaly", "Edema"}))
    assert (both["clf_fp"], both["clf_fp_mentioned"], both["clf_tp"], both["clf_tp_retained"]) == (1, 1, 2, 1)
    none = classifier_propagation(clf, truth, FS())
    assert (none["clf_fp_mentioned"], none["clf_tp_retained"]) == (0, 0)
    allp = classifier_propagation(clf, truth, FS({"Cardiomegaly", "Atelectasis", "Edema"}))
    assert (allp["clf_fp_mentioned"] / allp["clf_fp"], allp["clf_tp_retained"] / allp["clf_tp"]) == (1.0, 1.0)


def test_copy_metrics_definitions():
    src = ["The heart is enlarged and there is a small pleural effusion. No pneumothorax is seen."]
    full = copy_stats("The heart is enlarged and there is a small pleural effusion. No pneumothorax is seen.", src)
    assert full["copied_sentence_rate"] == 1.0 and full["whole_report_copy"]
    none = copy_stats("Cardiomegaly is possible.", src)
    assert none["copied_sentence_rate"] == 0.0 and not none["whole_report_copy"]
    half = copy_stats("The heart is enlarged and there is a small pleural effusion. Lungs show nothing else of note today.", src)
    assert 0 < half["copied_sentence_rate"] < 1 and not half["whole_report_copy"]


# ---------------------------------------------------------------- freeze, integrity and artefacts
def test_prompts_frozen_with_three_hashes_before_generation():
    fz = json.loads((G2 / "g2_prompts_frozen.json").read_text(encoding="utf-8"))
    assert fz["prompt_sha256"] == g2.prompt_hashes() and fz["frozen_before_bulk_generation"] is True
    assert {"agent1_evidence_verifier", "agent2_grounded_report_writer", "agent3_grounding_critic", "combined"} <= set(fz["prompt_sha256"])
    log = json.loads((G2 / "generation_run_log.json").read_text(encoding="utf-8"))
    assert log["prompt_sha256"] == fz["prompt_sha256"] and log["n_failed"] == 0
    assert fz["prompt_sha256"]["agent1_evidence_verifier"] in (G2 / "g2_prompts_frozen.txt").read_text(encoding="utf-8")


def test_classifier_retrieval_and_generator_freeze_integrity():
    verify_freeze(EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json", ROOT)
    g1_cases = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")}
    cases = [json.loads(l) for l in open(G2 / "g2_cases.jsonl", encoding="utf-8")]
    assert [c["uid"] for c in cases] == list(g1_cases)
    for c in cases:
        g = g1_cases[c["uid"]]
        assert c["retrieved_study_ids"] == [r["study_id"] for r in g["retrieved"]] and c["classifier_positive_findings"] == g["classifier_positive_findings"] and c["classifier_probabilities"] == g["classifier_probabilities"]
        assert not {"reference", "truth", "truth_findings"} & set(c)
    assert json.loads((G1 / "retrieval_top5_integrity.json").read_text(encoding="utf-8"))["retrieval_config"]["sha256_of_r2_candidate_config"] == sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json")
    meta = json.loads((G2 / "g2_generator_metadata.json").read_text(encoding="utf-8"))
    g1_gen = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    assert all(meta["same_generator_as_g1"].values()) and meta["model_digest"] == g1_gen["model"]["digest"] and meta["external_api"] is False
    assert all(meta["generation_options"][k] == g1_gen["generation_options"][k] for k in ("temperature", "top_k", "top_p", "seed", "num_ctx"))


def test_stored_messages_rebuild_identically_and_leak_nothing():
    cases = {json.loads(l)["uid"]: json.loads(l) for l in open(G2 / "g2_cases.jsonl", encoding="utf-8")}
    files = sorted((G2 / "agent_outputs").glob("*.json"))
    assert len(files) == 547
    for f in files[::7]:
        a = json.loads(f.read_text(encoding="utf-8"))
        pos, nf, retr = case_inputs(cases[a["uid"]])
        assert a["messages"][AGENTS[1]] == g2.build_agent2_message(pos, nf, a["evidence"])
        assert a["messages"][AGENTS[2]] == g2.build_agent3_message(pos, nf, a["evidence"], a["agent2_draft"])
        assert (AGENTS[0] in a["messages"]) == bool(retr)
        if retr:
            assert a["messages"][AGENTS[0]] == g2.build_agent1_message(pos, nf, retr)
    leak = json.loads((G2 / "g2_leakage_checks.json").read_text(encoding="utf-8"))
    assert leak["pass"] is True and leak["agent2_message_shares_6gram_with_retrieved_text"] == leak["agent3_message_shares_6gram_with_retrieved_text"] == 0


def test_g2_primary_comparison_and_required_outputs():
    s = json.loads((G2 / "g2_summary.json").read_text(encoding="utf-8"))
    assert s["primary_comparison"] == "G2_multi_agent_rag minus G1_single_agent_rag" and s["n_paired_studies"] == 547 and s["reproduction"]["earlier_results_reproduced"] is True
    pdf = pd.read_csv(G2 / "g2_paired_differences.csv")
    assert pdf.iloc[0].comparison == s["primary_comparison"] and pdf.comparison.iloc[0] == "G2_multi_agent_rag minus G1_single_agent_rag"
    need = {"precision", "recall", "f1", "macro_f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention", "normal_recall", "abnormal_recall", "rouge_l", "bleu4", "meteor_exact"}
    assert need <= set(pdf[pdf.comparison == s["primary_comparison"]].metric)
    gr = pd.read_csv(G2 / "g2_generated_reports.csv")
    assert len(gr) == 547 and {"study_id", "classifier_positive_findings", "classifier_probabilities_of_positives", "retrieved_top5_ids", "agent1_evidence_json", "agent2_report", "agent3_action", "final_g2_report",
                               "reference_report_for_evaluation_only"} <= set(gr.columns)
    n_empty = int((gr.retrieved_top5_ids.isna()).sum())
    assert n_empty == 49


def test_g2_code_has_no_external_api_loops_or_fitting():
    import re
    for p in [ROOT / "src/generation/g2_agents.py", ROOT / "src/generation/g2_pipeline.py"] + sorted((ROOT / "scripts").glob("g2_*.py")):
        t = p.read_text(encoding="utf-8")
        if p.name != "g2_07_integrity.py":
            assert not re.search(r"import\s+(anthropic|openai|requests)|api\.anthropic|api\.openai", t), p.name
            assert "allow_locked=True" not in t and "fit_platt" not in t and "fit_calibrator" not in t, p.name
    assert "while " not in inspect.getsource(run_case)                                          # no agent loop
