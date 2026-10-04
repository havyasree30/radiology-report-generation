"""G1F: deterministic routing guard, exact indeterminate wording, no retrieval / no model call for indeterminate studies, byte-identical reuse of the frozen
G1 outputs, final configuration serialisation and hash, locked-test exclusion, and classifier / retrieval / generator integrity."""

import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src.classification.final_test import sha256_file, verify_freeze
from src.classification.labels import LABELS, NO_FINDING
from src.generation.parse import parse_report
from src.generation.prompts import prompt_hash
from src.system import guard

ROOT = Path(__file__).resolve().parents[1]
RG = ROOT / "results/report_generation/experiments"
G1, G1F = RG / "g1_single_agent", RG / "g1f_final_system"
EXP = ROOT / "results/classification/experiments"
R1 = ROOT / "results/retrieval/experiments/r1_baseline"
R2 = ROOT / "results/retrieval/experiments/r2_optimization"
CONFIG = G1F / "FINAL_SYSTEM_CONFIG.json"
MANIFEST = EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json"


class Spy:
    def __init__(self, report=None):
        self.retrieve_calls = self.generate_calls = 0
        self.report = report or {"findings": "Cardiomegaly is present.", "impression": "Cardiomegaly."}

    def retrieve(self):
        self.retrieve_calls += 1
        return ["retrieved reports"]

    def generate(self, retrieved):
        self.generate_calls += 1
        assert retrieved == ["retrieved reports"]
        return dict(self.report)


# ---------------------------------------------------------------- routing
def test_explicit_no_finding_routes_to_the_normal_path():
    assert guard.route([], True) == "normal" and guard.PATH_OF_STATE["normal"] == "A"


def test_no_finding_with_only_a_support_device_is_still_normal_as_in_the_frozen_c4_rule():
    assert guard.route(["Support Devices"], True) == "normal"
    assert json.loads(MANIFEST.read_text(encoding="utf-8"))["support_devices_suppresses_no_finding"] is False


def test_one_or_several_pathology_positives_route_to_the_abnormal_g1_path():
    assert guard.route(["Cardiomegaly"], False) == "abnormal"
    assert guard.route(["Cardiomegaly", "Pleural Effusion", "Support Devices"], False) == "abnormal" and guard.PATH_OF_STATE["abnormal"] == "B"
    assert guard.route(["Support Devices"], False) == "abnormal"                       # a detected device without No Finding is a non-empty output and keeps its G1 report


def test_empty_classifier_output_is_indeterminate_not_normal():
    assert guard.route([], False) == "indeterminate" and guard.PATH_OF_STATE["indeterminate"] == "C"


def test_routing_is_deterministic_and_validates_labels():
    cases = [([], True), ([], False), (["Edema"], False), (["Support Devices"], True), (["Fracture", "Pneumonia"], False)]
    assert [guard.route(p, n) for p, n in cases] * 3 == [guard.route(p, n) for _ in range(3) for p, n in cases]
    assert set(guard.STATES) == {"normal", "abnormal", "indeterminate"}
    with pytest.raises(ValueError):
        guard.route(["Not A Label"], False)
    with pytest.raises(ValueError):
        guard.route([NO_FINDING], False)                                                # No Finding is a state flag, not a positive pathology list entry


def test_pathology_vocabulary_equals_the_frozen_c4_no_finding_rule_labels():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert list(guard.PATHOLOGY_LABELS) == m["no_finding_rule_abnormal_labels"] and list(guard.ABNORMAL_LABELS) == [l for l in m["label_order"] if l != NO_FINDING]
    assert set(guard.ABNORMAL_LABELS) | {NO_FINDING} == set(LABELS)


# ---------------------------------------------------------------- no retrieval, no model call for indeterminate; exact wording
def test_indeterminate_makes_no_retrieval_call_and_no_model_call():
    spy = Spy()
    res = guard.run_guarded([], False, spy.retrieve, spy.generate)
    assert (spy.retrieve_calls, spy.generate_calls) == (0, 0)
    assert (res.system_interpretation_state, res.routing_path, res.retrieval_invoked, res.llm_invoked) == ("indeterminate", "C", False, False)


def test_normal_and_abnormal_paths_call_the_frozen_pipeline_once_and_return_its_output_unchanged():
    for pos, nf, state in (([], True, "normal"), (["Cardiomegaly"], False, "abnormal"), (["Cardiomegaly", "Edema"], False, "abnormal")):
        spy = Spy({"findings": "F text.", "impression": "I text."})
        res = guard.run_guarded(pos, nf, spy.retrieve, spy.generate)
        assert (spy.retrieve_calls, spy.generate_calls) == (1, 1) and res.system_interpretation_state == state and res.retrieval_invoked and res.llm_invoked
        assert (res.findings, res.impression) == ("F text.", "I text.")


def test_exact_frozen_indeterminate_wording():
    assert guard.INDETERMINATE_FINDINGS == "Model output is indeterminate for this study."
    assert guard.INDETERMINATE_IMPRESSION == "Automated preliminary interpretation could not be established. Radiologist review is required."
    res = guard.run_guarded([], False, Spy().retrieve, Spy().generate)
    assert res.report_text == "FINDINGS: Model output is indeterminate for this study.\nIMPRESSION: Automated preliminary interpretation could not be established. Radiologist review is required."
    assert guard.UI_MESSAGE_INDETERMINATE == "Automated interpretation unavailable — radiologist review required."
    assert not any(bad.lower() in guard.INDETERMINATE_REPORT_TEXT.lower() for bad in ("no abnormality", "normal"))


def test_guard_module_is_pure_logic_without_model_or_retrieval_imports():
    tree = ast.parse((ROOT / "src/system/guard.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and re.search(r"generation|retrieval|ollama|torch|faiss|requests|urllib", m) for m in mods), mods


# ---------------------------------------------------------------- artefacts: byte-identical G1 outputs, counts, no new generation
@pytest.fixture(scope="module")
def outputs():
    return pd.read_csv(G1F / "g1f_final_validation_outputs.csv", dtype={"study_id": str}).fillna("")


def test_state_field_values_and_counts(outputs):
    assert set(outputs.system_interpretation_state) == {"normal", "abnormal", "indeterminate"}
    assert outputs.routing_path.value_counts().sort_index().to_dict() == {"A": 269, "B": 229, "C": 49}
    cases = [json.loads(l) for l in open(G1 / "g1_cases.jsonl", encoding="utf-8")]
    empty = {c["uid"] for c in cases if not c["retrieved"]}
    assert set(outputs[outputs.routing_path == "C"].study_id) == empty and len(empty) == 49
    for r, c in zip(outputs.itertuples(), cases):
        assert r.system_interpretation_state == guard.route(c["classifier_positive_findings"], c["no_finding_positive"])


def test_non_indeterminate_outputs_are_byte_identical_to_the_stored_g1_reports(outputs):
    raw = {json.loads(l)["uid"]: json.loads(l) for l in open(G1 / "g1_reports_raw.jsonl", encoding="utf-8")}
    ab = outputs[outputs.routing_path.isin(["A", "B"])]
    assert len(ab) == 498
    for r in ab.itertuples():
        p = parse_report(raw[r.study_id]["text"])
        assert (r.final_findings, r.final_impression) == (p["findings"], p["impression"])
        assert r.source_g1_raw_report_sha256 == hashlib.sha256(raw[r.study_id]["text"].encode()).hexdigest() and r.retrieval_invoked_in_final_pipeline and r.llm_invoked_in_final_pipeline


def test_indeterminate_rows_carry_the_fixed_message_and_no_calls(outputs):
    c = outputs[outputs.routing_path == "C"]
    assert (c.final_findings == guard.INDETERMINATE_FINDINGS).all() and (c.final_impression == guard.INDETERMINATE_IMPRESSION).all()
    assert not c.retrieval_invoked_in_final_pipeline.astype(bool).any() and not c.llm_invoked_in_final_pipeline.astype(bool).any() and (c.source_g1_raw_report_sha256 == "").all()


def test_no_new_generation_or_retrieval_in_the_g1f_run(outputs):
    assert int(outputs.new_llm_calls_in_g1f_run.sum()) == 0 and int(outputs.new_retrieval_calls_in_g1f_run.sum()) == 0
    for p in sorted((ROOT / "scripts").glob("g1f_*.py")):
        if p.name not in ("g1f_05_integrity.py",):
            assert not re.search(r"OllamaClient|generate_single|generate_agent|faiss|SentenceTransformer", p.read_text(encoding="utf-8")), p.name
    assert json.loads((G1F / "g1f_unchanged_outputs_check.json").read_text(encoding="utf-8"))["all_identical"] is True


def test_original_g1_files_are_untouched_and_g1f_metrics_change_only_through_path_c():
    summ = json.loads((G1F / "g1f_summary.json").read_text(encoding="utf-8"))
    for k in ("precision", "recall", "f1", "macro_f1", "hallucination_rate", "omission_rate", "clf_fp_propagation", "tp_retention"):
        assert summ["metrics"][k]["diff"] == 0 and summ["metrics"][k]["ci"] == [0, 0]
    assert summ["unchanged_outputs_check"]["changed_studies_are_exactly_path_C"] is True


# ---------------------------------------------------------------- locked test, classifier, retrieval and generator integrity
def test_locked_test_studies_are_excluded(outputs):
    split = json.loads((R1 / "retrieval_split.json").read_text(encoding="utf-8"))
    assert not set(outputs.study_id) & set(split["study_ids"]["locked_test"])
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["settings"]["evaluation"]["locked_test_split"]
    assert cfg["opened"] is False and cfg["n_studies"] == 578 and "results" not in json.dumps(cfg).lower().replace("no retrieval or end-to-end test result exists", "")


def test_classifier_integrity_and_configuration_match_the_frozen_artifacts():
    verify_freeze(MANIFEST, ROOT)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    s = json.loads(CONFIG.read_text(encoding="utf-8"))["settings"]["classification"]
    assert s["checkpoint_sha256"] == man["checkpoint_sha256"] == sha256_file(ROOT / man["checkpoint"]) and s["f1_thresholds_raw_score"] == man["f1_thresholds_raw_score"] and s["label_order"] == man["label_order"]
    assert s["calibration"]["file_sha256"] == sha256_file(EXP / "c5_calibration/final_calibrators.json") and s["no_finding_rule"] == man["no_finding_rule"] == "suppress_if_any_abnormal_positive"
    assert s["loss"] == "sqrt_weighted_bce"


def test_retrieval_integrity_and_configuration():
    s = json.loads(CONFIG.read_text(encoding="utf-8"))["settings"]["retrieval"]
    assert s["r2_candidate_config_sha256"] == sha256_file(R2 / "R2_RETRIEVAL_CANDIDATE_CONFIG.json") == json.loads((G1 / "retrieval_top5_integrity.json").read_text(encoding="utf-8"))["retrieval_config"]["sha256_of_r2_candidate_config"]
    assert (s["top_k"], s["fusion"]["rrf_k"], s["fusion"]["candidate_depth_per_ranker"], s["embedding_dimension"], s["mmr"], s["phrase_expansion"], s["confidence_weighting"]) == (5, 60, 100, 384, False, False, False)
    assert s["query_construction"]["top_n_findings"] == 3 and s["bm25"]["k1"] == 1.5 and s["bm25"]["b"] == 0.75 and s["corpus"]["n_reports"] == 2675


def test_g1_generator_and_prompt_integrity():
    g = json.loads(CONFIG.read_text(encoding="utf-8"))["settings"]["generation"]
    gf = json.loads((G1 / "GENERATOR_FREEZE.json").read_text(encoding="utf-8"))
    gp = json.loads((G1 / "g1_prompt_frozen.json").read_text(encoding="utf-8"))
    assert g["selected_architecture"] == "G1 Single-Agent RAG" and g["model_tag"] == "medgemma1.5:4b" and g["model_digest"] == gf["model"]["digest"] and g["ollama_version"] == gf["ollama_version"]
    assert g["generation_options"] == gf["generation_options"] and g["g1_prompt_sha256"] == gp["prompt_sha256"] == prompt_hash() and g["external_api"] is False and g["tools"] == "none"


# ---------------------------------------------------------------- final configuration serialisation and hash
def test_final_configuration_serialisation_hash_and_content():
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    recomputed = hashlib.sha256(json.dumps(doc["settings"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == doc["config_sha256"] and doc["locked_test_results_included"] is False
    assert json.loads(json.dumps(doc, ensure_ascii=False)) == doc                       # round trip
    assert set(doc["settings"]) == {"classification", "retrieval", "generation", "guard", "evaluation"}
    g = doc["settings"]["guard"]
    assert g["guard_module_sha256"] == sha256_file(ROOT / "src/system/guard.py") and g["indeterminate_findings"] == guard.INDETERMINATE_FINDINGS and g["indeterminate_impression"] == guard.INDETERMINATE_IMPRESSION
    assert g["support_devices_is_pathology"] is False and g["state_field"] == "system_interpretation_state" and g["validation_counts"] == {"A_normal": 269, "B_abnormal": 229, "C_indeterminate": 49}
    e = doc["settings"]["evaluation"]
    assert e["bootstrap"] == {"resamples": 1000, "seed": 42, "unit": "study"} and e["primary_set_n"] == 547 and e["clinical_subset_n"] == 378
    assert not re.search(r"test_metric|test_result|auroc_test", json.dumps(doc["settings"]), re.IGNORECASE)          # the settings hold no test result of any kind


def test_freeze_document_states_the_required_freeze_sentences_and_hashes():
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    txt = (G1F / "FINAL_SYSTEM_FREEZE.md").read_text(encoding="utf-8")
    assert "frozen before opening the locked test set" in txt and "No parameter, prompt, threshold, routing rule or model selection may be changed after inspecting the locked-test outputs." in txt
    assert doc["config_sha256"] in txt and doc["settings"]["generation"]["g1_prompt_sha256"] in txt and doc["settings"]["guard"]["guard_module_sha256"] in txt and doc["frozen_utc"] in txt
    assert "Automated interpretation unavailable — radiologist review required." in txt and "G1 Single-Agent RAG" in txt
