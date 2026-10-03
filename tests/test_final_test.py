"""C6: frozen pipeline, split independence, schema, deterministic bootstrap and freeze-manifest integrity."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classification.final_test import (IDX12, NF, apply_frozen_pipeline, binary_table, bootstrap_weights, count_stats,
                                           load_frozen, output_record, output_schema, sha256_file, validate_record,
                                           verify_freeze, weighted_macro_metrics)
from src.classification.labels import LABELS
from src.classification.operating_policy import apply_no_finding_rule

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results/classification/experiments"
C6 = EXP / "c6_final_test"
POLICY, CAL = EXP / "c4_operating_policy/final_operating_policy.json", EXP / "c5_calibration/final_calibrators.json"
MANIFEST = C6 / "FINAL_CLASSIFIER_FREEZE_MANIFEST.json"


@pytest.fixture(scope="module")
def frozen():
    return load_frozen(POLICY, CAL)


def _scores(n=500, seed=0):
    return np.clip(np.random.default_rng(seed).beta(0.8, 3.0, (n, 14)), 1e-6, 1 - 1e-6)


def test_splits_are_patient_disjoint():
    d = json.loads((C6 / "test_split_integrity.json").read_text(encoding="utf-8"))
    assert d["status"] == "PASS"
    for pair, c in d["overlap_counts"].items():
        assert c == {"patients": 0, "split_groups": 0, "image_paths": 0}, pair
    m = pd.read_csv(ROOT / "data/splits/chexpert/chexpert_image_manifest.csv.gz")
    sets = {s: set(m.loc[m.split == s, "patient_id"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])


def test_frozen_thresholds_equal_c4_and_manifest(frozen):
    policy, _ = frozen
    c4 = json.loads(POLICY.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for j, lab in enumerate(LABELS):
        assert policy["thresholds"][j] == c4["classes"][lab]["threshold"] == man["f1_thresholds_raw_score"][lab]


def test_frozen_calibrators_are_platt_and_match_manifest(frozen):
    _, cal = frozen
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for lab in LABELS:
        c = cal["classes"][lab]
        assert c["method"] == "platt" and c["a"] > 0
        assert (c["a"], c["b"]) == (man["calibration"]["parameters"][lab]["a"], man["calibration"]["parameters"][lab]["b"])


def test_pipeline_preserves_raw_scores_and_probability_range(frozen):
    policy, cal = frozen
    s = _scores()
    s[0], s[1] = 1.0, 0.0                                     # saturated float32 values must not crash or leave [0, 1]
    before = s.copy()
    out = apply_frozen_pipeline(s, policy, cal)
    assert np.array_equal(s, before) and np.array_equal(out["scores"], before)
    assert np.all(np.isfinite(out["prob"])) and out["prob"].min() >= 0 and out["prob"].max() <= 1
    assert not np.allclose(out["prob"].sum(1), 1.0)           # multi-label: never normalised across classes


def test_calibrated_equivalent_thresholds_reproduce_raw_decisions(frozen):
    policy, cal = frozen
    s = _scores(4000, seed=3)
    out = apply_frozen_pipeline(s, policy, cal)
    assert np.array_equal(out["prob"] >= out["calibrated_thresholds"], s >= policy["thresholds"])


def test_no_finding_rule_on_frozen_pipeline(frozen):
    policy, cal = frozen
    t = policy["thresholds"]
    s = np.full((4, 14), 0.0001)
    s[:, NF] = 0.99                                            # No Finding eligible in every row
    s[1, LABELS.index("Edema")] = 0.99                         # pathology positive -> suppress
    s[2, LABELS.index("Support Devices")] = 0.99               # Support Devices alone -> does NOT suppress
    s[3, LABELS.index("Fracture")] = t[LABELS.index("Fracture")] - 1e-9   # just below its threshold -> no suppression
    out = apply_frozen_pipeline(s, policy, cal)
    assert out["pred_raw"][:, NF].all()
    assert out["pred_final"][:, NF].tolist() == [True, False, True, True]
    assert np.array_equal(np.delete(out["pred_raw"], NF, 1), np.delete(out["pred_final"], NF, 1))
    assert np.array_equal(out["pred_final"], apply_no_finding_rule(out["pred_raw"]))
    assert len(IDX12) == 12 and LABELS.index("Support Devices") not in IDX12


def test_output_schema_and_record(frozen):
    policy, cal = frozen
    out = apply_frozen_pipeline(_scores(3, seed=5), policy, cal)
    rec = output_record("img_1", out["scores"][0], out["prob"][0], out["thresholds"], out["calibrated_thresholds"], out["pred_final"][0])
    validate_record(rec)
    json.dumps(rec, allow_nan=False)
    assert list(rec["findings"]) == list(LABELS)
    sch = output_schema()
    assert set(sch["properties"]["findings"]["required"]) == set(LABELS)
    bad = json.loads(json.dumps(rec))
    bad["findings"]["Edema"]["calibrated_probability"] = 1.2
    with pytest.raises(ValueError):
        validate_record(bad)
    bad = json.loads(json.dumps(rec))
    bad["positive_findings"] = [] if rec["positive_findings"] else ["Edema"]
    with pytest.raises(ValueError):
        validate_record(bad)
    f = C6 / "FINAL_CLASSIFIER_OUTPUT_SCHEMA.json"
    if f.exists():
        assert json.loads(f.read_text(encoding="utf-8"))["schema"]["required"] == sch["required"]


def test_c6_code_never_fits_or_tunes():
    banned = {"fit_calibrator", "fit_platt", "fit_temperature", "fit_isotonic", "f1_threshold", "youden_threshold",
              "youden_thresholds", "candidate_table", "patient_bootstrap_youden", "save_calibrators", "save_threshold_file",
              "save_thresholds", "save_final_policy"}
    files = list((ROOT / "scripts").glob("c6_*.py")) + [ROOT / "src/classification/final_test.py"]
    assert len(files) >= 2
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        called = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not (banned & (called | imported)), (p.name, banned & (called | imported))
    hashed = json.loads(MANIFEST.read_text(encoding="utf-8"))["hashed_files"]
    assert sha256_file(CAL) == hashed["results/classification/experiments/c5_calibration/final_calibrators.json"]
    assert sha256_file(POLICY) == hashed["results/classification/experiments/c4_operating_policy/final_operating_policy.json"]


def test_bootstrap_is_deterministic_and_patient_level():
    pids = np.repeat([f"p{i}" for i in range(60)], np.random.default_rng(1).integers(1, 5, 60))
    a = [w.copy() for w in bootstrap_weights(pids, 5, seed=7)]
    b = [w.copy() for w in bootstrap_weights(pids, 5, seed=7)]
    c = [w.copy() for w in bootstrap_weights(pids, 5, seed=8)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and not all(np.array_equal(x, y) for x, y in zip(a, c))
    for w in a:
        for p in np.unique(pids):                               # all images of one patient share one weight
            assert len(set(w[pids == p])) == 1


def test_weighted_metrics_equal_explicit_duplication():
    r = np.random.default_rng(2)
    n = 300
    labels = (r.random((n, 14)) < 0.3).astype(float)
    valid = r.random((n, 14)) > 0.1
    s = np.clip(r.random((n, 14)) * 0.5 + 0.5 * labels * r.random((n, 14)), 1e-4, 1 - 1e-4)
    prob, pred = s ** 1.3, s > 0.4
    w = r.integers(0, 4, n)
    got = weighted_macro_metrics(s, prob, pred, labels, valid, w)
    idx = np.repeat(np.arange(n), w)
    ref = weighted_macro_metrics(s[idx], prob[idx], pred[idx], labels[idx], valid[idx], np.ones(len(idx), dtype=int))
    for k in got:
        assert got[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12), k


def test_binary_table_counts_and_count_stats():
    labels = np.array([[1.0] * 14, [0.0] * 14, [1.0] * 14, [0.0] * 14])
    valid = np.ones((4, 14), bool)
    valid[3, 0] = False
    pred = np.array([[True] * 14, [True] * 14, [False] * 14, [False] * 14])
    t = binary_table(pred, labels, valid, np.full(14, 0.5))
    r = t.iloc[0]
    assert (r.tp, r.fp, r.tn, r.fn) == (1, 1, 0, 1) and t.iloc[1].tn == 1
    s = count_stats(np.array([0, 1, 2, 3, 4, 6]))
    assert s["pct_0"] == pytest.approx(100 / 6) and s["pct_gt5"] == pytest.approx(100 / 6) and s["max"] == 6


def test_freeze_manifest_integrity_and_ordering(tmp_path):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert all(verify_freeze(MANIFEST, ROOT).values())
    for key in ("checkpoint_sha256", "f1_thresholds_raw_score", "calibration", "commits", "environment", "created_utc", "label_order"):
        assert key in man
    assert man["label_order"] == list(LABELS) and len(man["f1_thresholds_raw_score"]) == 14
    assert man["hashed_files"][man["checkpoint"]] == man["checkpoint_sha256"]
    run = C6 / "test_inference_run.json"
    if run.exists():                                            # the manifest must pre-date the test inference
        r = json.loads(run.read_text(encoding="utf-8"))
        assert man["created_utc"] < r["finished_utc"] and r["freeze_manifest_sha256"] == sha256_file(MANIFEST)
        assert r["n_inference_passes"] == 1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(dict(man, hashed_files={**man["hashed_files"], man["checkpoint"]: "0" * 64})), encoding="utf-8")
    with pytest.raises(RuntimeError):
        verify_freeze(tampered, ROOT)
