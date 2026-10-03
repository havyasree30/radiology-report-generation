"""G1: leakage prevention, deterministic prompts, Top-5 integrity, evidence support, normal cases, parser, finding metrics,
propagation, schema, API client behaviour (fake client) and frozen classifier / retriever integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS, NO_FINDING
from src.generation.baseline import b0_report
from src.generation.client import (GenerationConfig, OllamaClient, OllamaRequestError, OllamaTransientError, ResponseCache, generate_single, request_hash, request_params)
from src.generation.evidence import format_table, support_table
from src.generation.extraction import ABNORMAL, ReportExtractor
from src.generation.metrics import (bleu, classifier_propagation, copy_stats, meteor_exact, per_class_prf, prf, provenance, repeated_sentence_rate, rouge_l, study_counts)
from src.generation.parse import parse_report
from src.generation.prompts import (ALLOWED_PAYLOAD_KEYS, SYSTEM_PROMPT, assert_payload_clean, build_payload, prompt_hash, render_user_message)
from src.generation.schema import build_record, output_schema, validate_record

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "results/report_generation/experiments/g1_single_agent"
R2 = ROOT / "results/retrieval/experiments/r2_optimization"
EXP = ROOT / "results/classification/experiments"


@pytest.fixture(scope="module")
def cases():
    return [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]


@pytest.fixture(scope="module")
def ext():
    return ReportExtractor()


def _ret(rank, sid, mapped, findings="F text.", impression="I text."):
    return {"rank": rank, "study_id": sid, "dense_rank": rank, "bm25_rank": rank + 1, "rrf_score": 0.03, "findings": findings, "impression": impression, "mapped_findings": mapped}


# ---------------------------------------------------------------- leakage prevention
def test_payload_has_no_path_for_reference_or_truth_and_validates_study_ids(cases):
    payload = build_payload([{"finding": "Cardiomegaly", "calibrated_probability": 0.7}], False, [_ret(1, "10", ["Cardiomegaly"])])
    assert set(payload) == ALLOWED_PAYLOAD_KEYS
    assert_payload_clean(payload, "5", {"10"}, {"99"})
    for bad_ctx in (("10", {"10"}, {"99"}), ("5", {"11"}, {"99"}), ("5", {"10"}, {"10"})):
        with pytest.raises(ValueError):
            assert_payload_clean(payload, *bad_ctx)
    with pytest.raises(ValueError):
        assert_payload_clean({**payload, "reference": "secret"}, "5", {"10"}, set())
    sentinel = "SENTINEL-REFERENCE-SENTENCE-XYZ"
    msg = render_user_message(payload)
    assert sentinel not in msg and "reference" not in msg.lower().replace("reference corpus", "") and "truth" not in msg.lower()
    for c in cases:
        assert set(c["payload"]) == ALLOWED_PAYLOAD_KEYS
        assert "truth" not in c["user_message"].lower() and c["reference"]["combined"] not in c["user_message"].replace(" ", " ") or not c["reference"]["combined"]


def test_payload_leakage_report_is_clean():
    lk = json.loads((G1 / "payload_leakage_check.json").read_text(encoding="utf-8"))
    assert lk["cases_with_reference_sentence_outside_retrieved_reports"] == 0 and lk["cases_with_own_study_in_retrieved"] == 0
    assert lk["truth_findings_in_payload"] is False and lk["payload_whitelist_enforced"] is True


# ---------------------------------------------------------------- deterministic prompt
def test_prompt_and_message_construction_are_deterministic(cases):
    assert prompt_hash() == prompt_hash() and len(prompt_hash()) == 64
    p1 = build_payload([{"finding": "Edema", "calibrated_probability": 0.5}, {"finding": "Cardiomegaly", "calibrated_probability": 0.9}], False, [_ret(1, "10", ["Edema"])])
    p2 = build_payload([{"finding": "Cardiomegaly", "calibrated_probability": 0.9}, {"finding": "Edema", "calibrated_probability": 0.5}], False, [_ret(1, "10", ["Edema"])])
    assert render_user_message(p1) == render_user_message(p2) and p1["classifier"]["positive_findings"][0]["finding"] == "Cardiomegaly"   # fixed class order
    import hashlib
    for c in cases[:50]:
        assert hashlib.sha256(c["user_message"].encode()).hexdigest() == c["user_message_sha256"] == hashlib.sha256(render_user_message(c["payload"]).encode()).hexdigest()
    frozen = G1 / "g1_prompt_frozen.json"
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding="utf-8"))["prompt_sha256"] == prompt_hash()
    assert "Do not copy retrieved reports verbatim" in SYSTEM_PROMPT and "cautiously" in SYSTEM_PROMPT and "No Finding: positive" in SYSTEM_PROMPT


# ---------------------------------------------------------------- retrieval Top-5 integrity
def test_top5_retrieval_integrity(cases):
    integ = json.loads((G1 / "retrieval_top5_integrity.json").read_text(encoding="utf-8"))
    assert integ["top5_integrity_pass"] and integ["top5_identical_to_r2"] == integ["r2_primary_studies_checked"] > 300
    sp = json.loads((ROOT / "results/retrieval/experiments/r1_baseline/retrieval_split.json").read_text(encoding="utf-8"))["study_ids"]
    corpus, val, test = set(sp["reference_corpus"]), set(sp["validation"]), set(sp["locked_test"])
    assert integ["retrieval_config"]["sha256_of_r2_candidate_config"] == sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json")
    for c in cases:
        ids = [r["study_id"] for r in c["retrieved"]]
        assert len(ids) in (0, 5) and len(set(ids)) == len(ids) and [r["rank"] for r in c["retrieved"]] == list(range(1, len(ids) + 1))
        assert set(ids) <= corpus and not (set(ids) & val) and not (set(ids) & test) and c["uid"] not in ids
        assert (len(ids) == 0) == (c["query_status"] == "empty")


# ---------------------------------------------------------------- evidence support
def test_evidence_support_counts():
    sets = [frozenset({"Cardiomegaly", "Edema"}), frozenset({"Cardiomegaly"}), frozenset(), frozenset({"Edema"}), frozenset({"Cardiomegaly"})]
    rows = support_table(["Cardiomegaly", "Fracture"], {"Cardiomegaly": 0.78, "Fracture": 0.31}, sets)
    assert rows[0] == {"finding": "Cardiomegaly", "classifier_probability": 0.78, "n_supporting": 3, "n_retrieved": 5, "support_fraction": 0.6}
    assert rows[1]["n_supporting"] == 0 and rows[1]["support_fraction"] == 0.0
    t = format_table(rows)
    assert "| Cardiomegaly | 0.78 | 3/5 |" in t and "| Fracture | 0.31 | 0/5 |" in t and format_table([]).startswith("(no")
    assert support_table(["Edema"], {"Edema": 0.5}, []) == [{"finding": "Edema", "classifier_probability": 0.5, "n_supporting": 0, "n_retrieved": 0, "support_fraction": 0.0}]


def test_cases_evidence_matches_mapped_findings(cases):
    for c in cases[:200]:
        for e in c["evidence_rows"]:
            assert e["n_supporting"] == sum(e["finding"] in r["mapped_findings"] for r in c["retrieved"]) and e["finding"] in c["classifier_positive_findings"]


# ---------------------------------------------------------------- normal / empty cases
def test_normal_and_empty_case_handling(cases, ext):
    r = b0_report([], True)
    assert r["state"] == "normal" and ext.report_state(r["findings"] + " " + r["impression"]) == "normal" and not ext.stated(r["findings"])
    r = b0_report([], False)
    assert r["state"] == "indeterminate" and ext.report_state(r["findings"] + " " + r["impression"]) == "indeterminate"
    r = b0_report(["Support Devices"], True)
    assert r["state"] == "abnormal" and ext.stated(r["findings"]) == {"Support Devices"} and "No acute" in r["findings"]
    normal = [c for c in cases if c["query_status"] == "no_finding" and not c["classifier_positive_findings"]]
    empty = [c for c in cases if c["query_status"] == "empty"]
    assert normal and empty
    assert all(c["payload"]["classifier"]["output_state"] == "no_finding" and c["b0"]["state"] == "normal" for c in normal)
    assert all(c["payload"]["retrieval_status"] != "available" and not c["retrieved"] and c["b0"]["state"] == "indeterminate" and "None: no retrieval query" in c["user_message"] for c in empty)
    assert all(c["payload"]["retrieval_status"] == "available" for c in normal)
    assert "classifier state decides" in SYSTEM_PROMPT


# ---------------------------------------------------------------- parser
def test_report_parser_formats():
    p = parse_report("FINDINGS:\nMild cardiomegaly. No effusion.\nIMPRESSION:\nCardiomegaly.")
    assert p["findings"] == "Mild cardiomegaly. No effusion." and p["impression"] == "Cardiomegaly." and p["format_ok"]
    p = parse_report("**Findings:** Clear lungs.\n**Impression:** Normal chest.")
    assert p["findings"] == "Clear lungs." and p["impression"] == "Normal chest." and p["format_ok"]
    p = parse_report("Findings: A\nB\nImpression: C")
    assert p["findings"] == "A B" and p["impression"] == "C"
    p = parse_report("No headings at all.")
    assert p["findings"] == "No headings at all." and not p["format_ok"] and "no_headings_whole_text_as_findings" in p["format_flags"]
    p = parse_report("FINDINGS:\nText only.")
    assert p["impression"] == "" and "empty_impression" in p["format_flags"]
    p = parse_report("")
    assert p["findings"] == "" and not p["format_ok"]


def test_report_parser_removes_echoed_input_and_repeated_sections():
    echoed = ("FINDINGS:\nThe lungs are clear. The heart size is normal.\nIMPRESSION:\nNo acute pulmonary abnormality.\n\nA. AUTOMATED CLASSIFIER OUTPUT (may contain errors)\n"
              "Final positive findings: none\n\nB. RETRIEVED EVIDENCE (reports of other patients)\n[1] study 1176 (dense rank 12)\nFINDINGS: Other patient text.\nIMPRESSION: 1. Other impression.")
    p = parse_report(echoed)
    assert p["findings"] == "The lungs are clear. The heart size is normal." and p["impression"] == "No acute pulmonary abnormality."
    assert "echoed_input_removed" in p["format_flags"] and p["format_ok"] and "1176" not in p["findings"] + p["impression"]
    repeated = "FINDINGS:\nA.\nIMPRESSION:\nB.\nFINDINGS:\nC.\nIMPRESSION:\nD."
    p = parse_report(repeated)
    assert p["findings"] == "A." and p["impression"] == "B." and "repeated_section_ignored" in p["format_flags"]
    cfg = GenerationConfig()
    assert set(cfg.options()["stop"]) == {"A. AUTOMATED CLASSIFIER OUTPUT", "B. RETRIEVED EVIDENCE", "C. EVIDENCE SUMMARY"}
    from src.generation.prompts import example_payload
    msg = render_user_message(example_payload())
    assert all(m in msg for m in cfg.options()["stop"])                                # the stop markers are exactly the input section titles


# ---------------------------------------------------------------- finding extraction / evaluation
def test_extraction_round_trip_and_assertions(ext):
    for l in ABNORMAL:
        r = b0_report([l], False)
        assert ext.stated(r["findings"] + " " + r["impression"]) == frozenset({l}), l
    allr = b0_report(list(ABNORMAL), False)
    assert ext.stated(allr["findings"] + " " + allr["impression"]) == frozenset(ABNORMAL)
    st = ext.status("No pleural effusion. Possible small pneumothorax. Cannot exclude pneumonia. Cardiomegaly is present.")
    assert st["Pleural Effusion"] == "negated" and st["Pneumothorax"] == "uncertain" and st["Pneumonia"] == "uncertain" and st["Cardiomegaly"] == "positive"
    t = "No pleural effusion. Possible small pneumothorax. Cardiomegaly is present."
    assert ext.stated(t) == {"Pneumothorax", "Cardiomegaly"} and ext.definite(t) == {"Cardiomegaly"}
    assert ext.status("Pulmonary vascular congestion is present.")["Edema"] == "not_mentioned"          # aligned with the R1 truth mapping
    assert ext.report_state("Heart size normal. Lungs clear.") == "normal" and ext.report_state("Lungs are clear.") == "indeterminate"


def test_finding_prf_and_per_class():
    assert prf(3, 1, 1) == {"precision": 0.75, "recall": 0.75, "f1": 0.75}
    assert np.isnan(prf(0, 0, 2)["precision"]) and prf(0, 0, 2)["recall"] == 0.0
    gen = [frozenset({"Edema", "Fracture"}), frozenset(), frozenset({"Edema"})]
    tr = [frozenset({"Edema"}), frozenset({"Fracture"}), frozenset({"Edema", "Pneumonia"})]
    rows = {r["finding"]: r for r in per_class_prf(gen, tr, ["Edema", "Fracture", "Pneumonia"])}
    assert (rows["Edema"]["tp"], rows["Edema"]["fp"], rows["Edema"]["fn"]) == (2, 0, 0) and rows["Fracture"]["fp"] == 1 and rows["Fracture"]["fn"] == 1 and rows["Pneumonia"]["fn"] == 1


def test_hallucination_and_omission_counts():
    assert study_counts(frozenset({"A", "B"}), frozenset({"B", "C"})) == {"tp": 1, "fp": 1, "fn": 1}
    assert study_counts(frozenset(), frozenset({"A"})) == {"tp": 0, "fp": 0, "fn": 1}
    assert study_counts(frozenset({"A"}), frozenset()) == {"tp": 0, "fp": 1, "fn": 0}
    assert study_counts(frozenset(), frozenset()) == {"tp": 0, "fp": 0, "fn": 0}


def test_classifier_false_positive_propagation_and_tp_retention():
    clf, truth = frozenset({"Edema", "Fracture", "Pneumonia"}), frozenset({"Edema", "Atelectasis"})
    assert classifier_propagation(clf, truth, frozenset({"Edema", "Fracture", "Atelectasis"})) == {"clf_fp": 2, "clf_fp_mentioned": 1, "clf_tp": 1, "clf_tp_retained": 1}
    assert classifier_propagation(clf, truth, frozenset()) == {"clf_fp": 2, "clf_fp_mentioned": 0, "clf_tp": 1, "clf_tp_retained": 0}      # deleting everything: no FP, but no TP retained
    assert classifier_propagation(clf, truth, clf) == {"clf_fp": 2, "clf_fp_mentioned": 2, "clf_tp": 1, "clf_tp_retained": 1}              # verbalising the classifier = B0


def test_grounding_provenance_categories():
    r = provenance(frozenset({"Edema", "Fracture", "Pneumonia", "Atelectasis"}), frozenset({"Edema", "Fracture"}), frozenset({"Edema", "Pneumonia"}))
    assert r == {"A": 1, "B": 1, "C": 1, "D": 1}
    assert provenance(frozenset(), frozenset({"Edema"}), frozenset()) == {"A": 0, "B": 0, "C": 0, "D": 0}


def test_b0_stated_findings_equal_classifier_positives(cases, ext):
    for c in cases[:300]:
        assert ext.stated(c["b0"]["findings"] + " " + c["b0"]["impression"]) == frozenset(c["classifier_positive_findings"])


# ---------------------------------------------------------------- text metrics, copying
def test_text_metrics_known_values():
    s = "the heart is mildly enlarged with clear lungs"
    assert bleu(s, s) == pytest.approx(1.0) and rouge_l(s, s) == pytest.approx(1.0) and meteor_exact(s, s) == pytest.approx(1.0 * (1 - 0.5 * (1 / 8) ** 3))
    assert bleu("alpha beta", "gamma delta", 1) == 0.0 or bleu("alpha beta", "gamma delta", 1) < 0.1
    assert rouge_l("a b c d", "a x c d") == pytest.approx(2 * 0.75 * 0.75 / 1.5) and rouge_l("", "a") == 0.0
    assert bleu("a b c d e f", "a b c d e f g h i j", 4) < 1.0                                      # brevity penalty
    assert repeated_sentence_rate("Normal heart. Normal heart. Clear lungs.") == pytest.approx(1 / 3)
    cs = copy_stats("The cardiac silhouette is mildly enlarged today. Short one.", ["The cardiac silhouette is mildly enlarged today."])
    assert cs["n_copied_sentences"] == 1 and cs["n_long_sentences"] == 1 and cs["whole_report_copy"] is True and cs["copied_sentence_rate"] == 1.0
    cs0 = copy_stats("No acute abnormality.", ["No acute abnormality."])
    assert cs0["n_long_sentences"] == 0 and cs0["copied_sentence_rate"] == 0.0                         # generic short phrases are not counted as copying


# ---------------------------------------------------------------- output schema
def test_output_schema_and_record(cases):
    c = [x for x in cases if x["retrieved"] and x["classifier_positive_findings"]][0]
    rec = build_record(c, {"findings": c["b0"]["findings"], "impression": c["b0"]["impression"]}, "claude-sonnet-5-5", prompt_hash(), 0.0)
    validate_record(rec)
    json.dumps(rec, allow_nan=False)
    assert "reference" not in json.dumps(rec).lower().replace("reference corpus", "") and list(rec["classifier_findings"]["calibrated_probabilities_all_classes"]) == list(LABELS)
    assert not np.isclose(sum(rec["classifier_findings"]["calibrated_probabilities_all_classes"].values()), 1.0)
    bad = json.loads(json.dumps(rec))
    bad["classifier_findings"]["calibrated_probabilities_all_classes"]["Edema"] = 1.5
    with pytest.raises(ValueError):
        validate_record(bad)
    bad = json.loads(json.dumps(rec))
    bad["reference"] = {"findings": "x"}
    with pytest.raises(ValueError):
        validate_record(bad)
    assert set(output_schema()["required"]) == {"image_id", "classifier_findings", "retrieved_evidence", "evidence_summary", "preliminary_report", "generation", "disclaimer"}


# ---------------------------------------------------------------- local Ollama client behaviour (fake client; no network)
class _FakeOllama:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, params):
        self.calls.append(params)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"message": {"role": "assistant", "content": step}, "done_reason": "stop", "model": params["model"], "prompt_eval_count": 1200, "eval_count": 90, "total_duration": 5, "load_duration": 1}


def test_client_options_retry_cache_and_no_fabrication(tmp_path):
    cfg = GenerationConfig()
    o = cfg.options()
    assert cfg.model == "medgemma1.5:4b" and o["temperature"] == 0.0 and o["top_k"] == 1 and o["seed"] == 42 and o["num_ctx"] == 8192 and 0 < o["num_predict"] <= 600
    assert cfg.tools == "none" and cfg.web_access is False and cfg.external_api is False and cfg.stream is False
    params = request_params(cfg, "sys", "user")
    assert "tools" not in params and "think" not in params and params["stream"] is False and params["messages"][0]["role"] == "system"
    fake = _FakeOllama([OllamaTransientError("ConnectionError"), OllamaTransientError("timeout"), "FINDINGS:\nA.\nIMPRESSION:\nB."])
    rec = generate_single(fake, cfg, "sys", "user", "digest-1", sleep=lambda s: None)
    assert rec["status"] == "success" and rec["attempts"] == 3 and rec["text"].startswith("FINDINGS") and rec["options"] == o and rec["prompt_may_be_truncated"] is False
    f = generate_single(_FakeOllama([OllamaTransientError("x")] * 5), cfg, "sys", "user", sleep=lambda s: None)
    assert f["status"] == "failed" and "text" not in f and f["error_class"] == "OllamaTransientError"
    nr = _FakeOllama([OllamaRequestError("HTTP 404: model not found")])
    assert generate_single(nr, cfg, "s", "u", sleep=lambda s: None)["status"] == "failed" and len(nr.calls) == 1
    assert generate_single(_FakeOllama(["   "]), cfg, "s", "u", sleep=lambda s: None)["status"] == "failed"          # empty output is never accepted
    long = _FakeOllama(["FINDINGS:\nA.\nIMPRESSION:\nB."])
    long.chat = lambda p: {"message": {"content": "x"}, "prompt_eval_count": 8192, "eval_count": 1}
    assert generate_single(long, cfg, "s", "u")["prompt_may_be_truncated"] is True
    cache = ResponseCache(tmp_path / "c")
    h = request_hash(params, "digest-1")
    cache.put("42", {**rec, "request_hash": h})
    assert cache.get("42", h)["text"] == rec["text"] and cache.get("42", request_hash(params, "digest-2")) is None and cache.completed({"42": h, "43": h}) == {"42"}
    with pytest.raises(ValueError):
        cache.put("44", f)
    assert request_hash(params, "d") == request_hash(request_params(cfg, "sys", "user"), "d") != request_hash(request_params(cfg, "sys", "other"), "d")


def test_client_is_local_only():
    OllamaClient("http://localhost:11434")
    OllamaClient("http://127.0.0.1:11434")
    for bad in ("http://example.com:11434", "https://api.anthropic.com", "http://192.168.1.5:11434"):
        with pytest.raises(ValueError):
            OllamaClient(bad)
    src_text = (ROOT / "src/generation/client.py").read_text(encoding="utf-8").lower()
    assert "anthropic" not in src_text.replace("no external api", "") or "api key" in src_text            # no Anthropic code path


def test_no_anthropic_dependency_in_g1_path():
    for f in ["scripts/g1_02_smoke_test_and_freeze.py", "scripts/g1_03_generate.py", "src/generation/client.py", "requirements-generation.txt"]:
        t = (ROOT / f).read_text(encoding="utf-8")
        assert "import anthropic" not in t and "anthropic==" not in t and "ANTHROPIC_API_KEY" not in t.replace("no API key", ""), f


def test_generation_config_and_frozen_records():
    f = G1 / "g1_prompt_frozen.json"
    g = G1 / "GENERATOR_FREEZE.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d["external_api_used"] is False and d["frozen_before_bulk_generation"] is True and d["generation_config"]["model"] == "medgemma1.5:4b" and d["prompt_sha256"] == prompt_hash()
    if g.exists():
        d = json.loads(g.read_text(encoding="utf-8"))
        assert d["applies_to"] == ["G1_single_agent_rag", "G2_multi_agent_rag"] and d["model"]["model_tag"] == "medgemma1.5:4b" and d["model"]["digest"] and d["ollama_version"]
        assert d["generation_options"]["temperature"] == 0.0 and d["tools"] == "none" and d["web_access"] is False and d["external_api"] is False and d["endpoint"] == "http://localhost:11434/api/chat"


# ---------------------------------------------------------------- frozen classifier / retriever integrity
def test_frozen_classifier_and_retrieval_configuration_unchanged():
    man_path = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"
    assert all(verify_freeze(man_path, ROOT).values())
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert sha256_file(ROOT / man["checkpoint"]) == man["checkpoint_sha256"]
    cfg = json.loads((R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    assert cfg["provisional_top_k"] == 5 and cfg["retriever"]["selected"] == "hybrid" and cfg["query_construction"]["top_n_findings"] == 3
    assert not cfg["query_construction"]["weighting_selected"] and not cfg["query_construction"]["clinical_phrase_expansion_selected"] and not cfg["diversity_reranking"]["mmr_selected"]
    assert cfg["retriever"]["hybrid"]["rrf_k"] == 60 and cfg["retriever"]["hybrid"]["depth_per_ranker"] == 100 and cfg["embedding_model"]["dimension"] == 384
    assert "unopened" in cfg["status"]
