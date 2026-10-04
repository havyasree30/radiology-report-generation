"""F1: one-shot locked end-to-end evaluation. Tests the gate, the pre-registration, the frozen-component integrity, the routing and prompt reconstruction, the metric definitions
against the stored per-study outputs, and the generated assets. Code correctness only: no test checks a metric VALUE against a target."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification.final_test import sha256_file, verify_freeze
from src.f1 import pipeline
from src.generation.client import GenerationConfig, request_hash, request_params
from src.generation.prompts import SYSTEM_PROMPT, build_payload, prompt_hash, render_user_message
from src.system import guard

ROOT = Path(__file__).resolve().parents[1]
F1 = ROOT / "results/final_test/f1_locked_end_to_end"
T, V = F1 / "test", F1 / "validation_dry_run"
R1 = ROOT / "results/retrieval/experiments/r1_baseline"
G1 = ROOT / "results/report_generation/experiments/g1_single_agent"
G1F = ROOT / "results/report_generation/experiments/g1f_final_system"
EXP = ROOT / "results/classification/experiments"
CONFIG_HASH = "cc04996cf8e8f6441505eedf5b41068b6af63384bf3a0a51bfde2a37c503eff5"


def js(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- gate, pre-registration, one-shot
def test_locked_partition_is_closed_without_the_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "F1", tmp_path)
    with pytest.raises(PermissionError):
        pipeline.require_open("test")
    pipeline.require_open("validation")                                  # the dry-run partition is never gated


def test_protocol_registered_hashed_and_precedes_the_opening():
    rec = {ln.split(" *")[1]: ln.split(" *")[0] for ln in (F1 / "F1_LOCKED_TEST_PROTOCOL.sha256").read_text(encoding="utf-8").splitlines() if ln.strip()}
    assert sha256_file(F1 / "F1_LOCKED_TEST_PROTOCOL.json") == rec["F1_LOCKED_TEST_PROTOCOL.json"] and sha256_file(F1 / "F1_LOCKED_TEST_PROTOCOL.md") == rec["F1_LOCKED_TEST_PROTOCOL.md"]
    proto, marker, cf = js(F1 / "F1_LOCKED_TEST_PROTOCOL.json"), js(T / "test_opened.json"), js(F1 / "f1_code_freeze.json")
    assert proto["created_utc"] < cf["frozen_utc"] <= marker["opened_utc"] and marker["protocol_json_sha256"] == rec["F1_LOCKED_TEST_PROTOCOL.json"]
    assert proto["final_system_config_sha256"] == CONFIG_HASH and proto["locked_test"]["opened_before_this_protocol"] is False and proto["bootstrap"]["resamples"] == 1000 and proto["bootstrap"]["seed"] == 42
    assert {"figures_planned", "tables_planned", "failure_taxonomy_frozen", "no_post_test_tuning", "evaluation_populations"} <= set(proto) and len(proto["figures_planned"]) == 8 and len(proto["tables_planned"]) == 10


def test_opening_gate_is_one_shot_and_code_is_frozen():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "scripts.f1_02_open_test"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "already opened" in r.stderr
    cf = js(F1 / "f1_code_freeze.json")
    assert all(sha256_file(ROOT / p) == h for p, h in cf["sha256"].items()) and len(cf["sha256"]) == 9


def test_freeze_verification_and_dry_run_reproduction_passed_before_opening():
    assert js(F1 / "f1_freeze_verification.json")["all_identifiers_verified"] is True
    assert js(V / "reproduction_checks.json")["all_reproduced"] is True and js(V / "evaluation_reproduction_checks.json")["all_reproduced"] is True
    assert js(V / "reproduction_checks.json")["cases_user_message_sha256_identical"] == "547 of 547"


# ---------------------------------------------------------------- frozen components
def test_frozen_components_unchanged():
    cfg = js(G1F / "FINAL_SYSTEM_CONFIG.json")
    h = hashlib.sha256(json.dumps(cfg["settings"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert h == cfg["config_sha256"] == CONFIG_HASH
    assert prompt_hash() == "476b05326f85652b0cdb2faf79a4868a6e839660eb43e34d46cdae625e4b5688" and sha256_file(ROOT / "src/system/guard.py") == "1f0b340cc71cf261de84f4b19f290c99568a69760dc757d57c62d2c68c9397ee"
    verify_freeze(EXP / "c6_final_test/FINAL_CLASSIFIER_FREEZE_MANIFEST.json", ROOT)
    assert pipeline.SPEC == {"phrases": "names", "normal": "no acute abnormality", "top_n": 3, "weighting": None, "oracle": False} and pipeline.TOPK == 5


# ---------------------------------------------------------------- locked-test population, routing, prompts, generation
@pytest.fixture(scope="module")
def cases():
    return [json.loads(l) for l in open(T / "cases.jsonl", encoding="utf-8")]


def test_population_is_exactly_the_locked_ids_and_flow_is_consistent(cases):
    split = js(R1 / "retrieval_split.json")
    ids = sorted(str(i) for i in split["study_ids"]["locked_test"])
    pop = pd.read_csv(T / "population.csv", dtype={"uid": str})
    assert sorted(pop.uid) == ids and len(ids) == 578 and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == js(G1F / "FINAL_SYSTEM_CONFIG.json")["settings"]["evaluation"]["locked_test_split"]["study_ids_sha256"]
    fl = js(T / "population_flow.json")
    assert fl["locked_or_validation_studies"] == 578 and fl["with_frontal_image"] + fl["without_frontal_image"] == 578 and fl["classified_P2"] + fl["failing_image_validator"] == fl["with_frontal_image"]
    assert fl["primary_end_to_end_set_P3_(classified_and_usable_reference_report)"] == len(cases) and sum(fl["routing_counts_on_P3"].values()) == len(cases)
    assert fl["clinical_finding_subset_P4"] == sum(c["in_clinical_subset"] for c in cases) and not set(c["uid"] for c in cases) - set(ids)


def test_routing_is_the_frozen_guard_and_path_c_has_no_query_or_generation(cases):
    for c in cases:
        assert c["system_interpretation_state"] == guard.route(c["classifier_positive_findings"], c["no_finding_positive"]) and c["routing_path"] == guard.PATH_OF_STATE[c["system_interpretation_state"]]
        assert (c["system_interpretation_state"] == "indeterminate") == (not c["retrieved"]) == (c["query_status"] == "empty")
        if c["retrieved"]:
            assert len(c["retrieved"]) == 5 and [r["rank"] for r in c["retrieved"]] == [1, 2, 3, 4, 5]
        else:
            assert not (T / "generation_cache" / f"{c['uid']}.json").exists()


def test_every_actual_prompt_equals_its_rebuild_and_requests_match(cases):
    cfgf = js(G1 / "g1_prompt_frozen.json")
    cfg = GenerationConfig(**{k: v for k, v in cfgf["generation_config"].items() if k in GenerationConfig.__dataclass_fields__})
    digest = js(G1 / "GENERATOR_FREEZE.json")["model"]["digest"]
    for c in cases:
        msg = render_user_message(build_payload([{"finding": f, "calibrated_probability": c["classifier_probabilities"][f]} for f in c["classifier_positive_findings"]], c["no_finding_positive"], c["retrieved"]))
        assert msg == c["user_message"] and hashlib.sha256(msg.encode()).hexdigest() == c["user_message_sha256"]
        if c["retrieved"]:
            rec = js(T / "generation_cache" / f"{c['uid']}.json")
            assert rec["request_hash"] == request_hash(request_params(cfg, SYSTEM_PROMPT, msg), digest) and rec["options"] == cfg.options() and rec["status"] == "success"


def test_retrieval_never_returns_query_validation_or_test_studies(cases):
    split = js(R1 / "retrieval_split.json")
    corpus, test_ids, val_ids = set(split["study_ids"]["reference_corpus"]), set(split["study_ids"]["locked_test"]), set(split["study_ids"]["validation"])
    for c in cases:
        for r in c["retrieved"]:
            assert r["study_id"] in corpus and r["study_id"] not in test_ids | val_ids and r["study_id"] != c["uid"]


def test_generation_log_and_single_call_per_study(cases):
    log = js(T / "generation_run_log.json")
    n_ab = sum(c["system_interpretation_state"] != "indeterminate" for c in cases)
    assert log["n_failed"] == 0 and log["n_cached_success"] == n_ab == len(list((T / "generation_cache").glob("*.json"))) and log["path_C_studies_not_called"] == len(cases) - n_ab
    raw = [json.loads(l)["uid"] for l in open(T / "g1_reports_raw.jsonl", encoding="utf-8")]
    assert len(raw) == len(set(raw)) == n_ab


# ---------------------------------------------------------------- metric definitions against stored per-study outputs
def test_report_metrics_equal_recomputation_from_the_per_study_table():
    d = pd.read_csv(T / "report_per_study_results.csv", dtype={"uid": str})
    s = js(T / "report_summary.json")
    c = d[d.in_clinical.astype(bool)]
    tp, fp, fn = c.tp.sum(), c.fp.sum(), c.fn.sum()
    m = s["metrics"]
    assert m["precision"]["value"] == pytest.approx(tp / (tp + fp)) and m["recall"]["value"] == pytest.approx(tp / (tp + fn)) and m["f1"]["value"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert m["hallucination_rate"]["value"] == pytest.approx(c.hall.mean()) and m["omission_rate"]["value"] == pytest.approx(c.omit.mean())
    assert m["clf_fp_propagation"]["value"] == pytest.approx(c.clf_fp_mentioned.sum() / c.clf_fp.sum()) and m["tp_retention"]["value"] == pytest.approx(c.clf_tp_retained.sum() / c.clf_tp.sum())
    assert m["rouge_l"]["value"] == pytest.approx(d.rouge_l.mean()) and s["n_clinical_P4"] == len(c) and s["n_primary_P3_with_final_report"] == len(d)
    for k, e in m.items():
        assert e["ci95_low"] <= e["value"] + 1e-9 or np.isnan(e["ci95_low"])


def test_three_state_counts_and_decision_coverage_definition():
    s = js(T / "report_summary.json")["three_state"]
    assert sum(s["state_counts_P3"].values()) == js(T / "report_summary.json")["n_primary_P3_with_final_report"] and sum(s["state_counts_P4"].values()) == s["eligible_n"]
    assert s["decided"]["decision_coverage"]["value"] == pytest.approx(s["decided_n"] / s["eligible_n"]) and s["decided_n"] == s["state_counts_P4"]["normal"] + s["state_counts_P4"]["abnormal"]
    for r in s["reference_stratified"]:
        assert r["routed_normal_n"] + r["routed_abnormal_n"] + r["routed_indeterminate_n"] == r["n"]
    ab = js(T / "report_summary.json")["abstention"]
    assert ab["n_abstained_P3"] == s["state_counts_P3"]["indeterminate"] and ab["n_abstained_P4"] == s["state_counts_P4"]["indeterminate"]
    assert ab["reference_normal_among_abstained"] + ab["reference_abnormal_among_abstained"] == ab["n_abstained_P4"]


def test_classification_metrics_use_c6_definitions_and_undefined_classes_are_not_zero():
    pc = pd.read_csv(T / "classification_per_class.csv")
    s = js(T / "f1_summary.json")["classification"]
    assert s["point"]["macro_auroc"] == pytest.approx(np.nanmean(pc.auroc)) and s["point"]["macro_auprc"] == pytest.approx(np.nanmean(pc.auprc))
    assert s["n_classes_defined"] == int(pc.auroc.notna().sum()) and sorted(s["undefined_classes"]) == sorted(pc[pc.auroc.isna()].observation)
    assert (pc.loc[pc.auroc.isna(), "undefined_reason"] != "").all() and len(pc) == 14 and (pc.support == pc.tp + pc.fn).all()
    assert s["ci95"]["macro_auroc"]["ci95_low"] <= s["point"]["macro_auroc"] <= s["ci95"]["macro_auroc"]["ci95_high"]
    saved = pd.read_csv(T / "classifier_outputs.csv")
    assert all(f"{l}__score" in saved and f"{l}__prob" in saved and f"{l}__pred_final" in saved and f"{l}__pred_raw" in saved for l in pipeline.LABELS) and saved.system_interpretation_state.isin(guard.STATES).all()


def test_retrieval_table_equals_per_study_means():
    per = pd.read_csv(T / "retrieval_classifier_query_per_study.csv")
    tab = pd.read_csv(T / "retrieval_metrics_by_k.csv")
    for k in (1, 3, 5, 10):
        for m in ("jaccard_truth", "ndcg", "union_coverage", "duplicate_text_rate"):
            assert tab[(tab.K == k) & (tab.metric == m)].classifier_query.iloc[0] == pytest.approx(np.nanmean(per[f"{m}@{k}"]))
    assert set(tab.K) == {1, 3, 5, 10} and (tab.n_queries == len(per)).all()


def test_failure_taxonomy_is_the_frozen_one():
    proto = js(F1 / "F1_LOCKED_TEST_PROTOCOL.json")["failure_taxonomy_frozen"]["categories"]
    fa = pd.read_csv(T / "failure_analysis.csv")
    assert set(fa.category) - {"no_failure_in_categories_1_to_8"} == set(proto) and (fa.n_studies <= js(T / "report_summary.json")["n_clinical_P4"]).all()


# ---------------------------------------------------------------- assets
def test_final_assets_exist_in_all_formats():
    figs = sorted((F1 / "figures").glob("fig*.png"))
    assert len(figs) == 8 and all(p.with_suffix(e).exists() for p in figs for e in (".pdf", ".svg")) and js(F1 / "figures/asset_checks.json")["min_dpi"] == 300
    assert len(list((F1 / "figures/source_data").glob("*.csv"))) >= 8
    tabs = sorted((F1 / "tables").glob("table*.csv"))
    assert len(tabs) >= 10 and all(p.with_suffix(e).exists() for p in tabs for e in (".md", ".tex"))
    for n in ("FINAL_LOCKED_TEST_ANALYSIS.md", "MANUSCRIPT_FINAL_TEST_METHODS.md", "MANUSCRIPT_FINAL_TEST_RESULTS.md", "MANUSCRIPT_FINAL_DISCUSSION_POINTS.md", "FINAL_FIGURE_CAPTIONS.md", "FINAL_TABLE_CAPTIONS.md", "FINAL_JOURNAL_ASSET_INDEX.md", "UNIVERSITY_REPORT_FINAL_TEST.md"):
        assert (F1 / n).exists() and len((F1 / n).read_text(encoding="utf-8")) > 500, n


def test_final_integrity_report_passes():
    r = js(F1 / "f1_final_integrity_report.json")
    assert r["status"] == "PASS" and all(r["checks"].values())
