"""C5 calibration: fitting, metrics, patient-level folds, threshold equivalence and files."""

import numpy as np
import pytest

from src.classification.calibration import (CalibrationError, adaptive_bins, apply_calibrator, brier, calibrated_threshold,
                                            class_metrics, decision_mismatches, ece_adaptive, fit_calibrator, fit_isotonic,
                                            fit_platt, fit_temperature, is_strictly_increasing, load_calibrators,
                                            log_loss, patient_folds, safe_logit, save_calibrators, sigmoid)
from src.classification.labels import LABELS


def _synthetic(n=6000, true_T=2.0, seed=0, prevalence_shift=0.0):
    """Overconfident scores: true logits z0, model reports z = true_T * z0 (+ shift)."""
    r = np.random.default_rng(seed)
    z0 = r.normal(-1.0, 1.5, n)
    y = r.random(n) < sigmoid(z0)
    return true_T * z0 + prevalence_shift, y


def test_temperature_recovers_known_overconfidence():
    z, y = _synthetic(n=40000, true_T=2.0)
    fit = fit_temperature(z, y)
    assert fit["T"] == pytest.approx(2.0, rel=0.08) and not fit["pathological"]
    assert fit["T"] > 0
    assert log_loss(apply_calibrator(fit, z), y) < log_loss(sigmoid(z), y)


def test_platt_recovers_slope_and_intercept():
    r = np.random.default_rng(1)
    z = r.normal(0, 2, 40000)
    y = r.random(40000) < sigmoid(0.6 * z - 1.5)
    fit = fit_platt(z, y)
    assert fit["a"] == pytest.approx(0.6, abs=0.04) and fit["b"] == pytest.approx(-1.5, abs=0.08) and fit["flags"] == []
    assert is_strictly_increasing(fit)


def test_platt_flags_non_positive_slope_instead_of_accepting_it():
    r = np.random.default_rng(2)
    z = r.normal(0, 2, 5000)
    y = r.random(5000) < sigmoid(-1.0 * z)          # anti-correlated scores
    fit = fit_platt(z, y)
    assert fit["a"] < 0 and "non_positive_slope" in fit["flags"] and fit["pathological"]
    assert not is_strictly_increasing(fit)


def test_isotonic_is_monotone_bounded_and_flags_few_levels():
    z, y = _synthetic()
    fit = fit_isotonic(z, y)
    grid = np.linspace(z.min() - 1, z.max() + 1, 500)
    out = apply_calibrator(fit, grid)
    assert np.all(np.diff(out) >= -1e-15) and out.min() > 0 and out.max() < 1
    assert not is_strictly_increasing(fit)
    r = np.random.default_rng(3)
    zz = r.normal(size=300)
    assert "very_few_output_levels" in fit_isotonic(zz, np.zeros(300, bool) | (zz > 5) | (np.arange(300) < 2))["flags"]


def test_calibrated_values_stay_in_unit_interval_for_extreme_logits():
    z = np.array([-700.0, -40.0, 0.0, 40.0, 700.0])
    for spec in ({"method": "raw"}, {"method": "temperature", "T": 0.05}, {"method": "platt", "a": 2.9, "b": -4.0},
                 {"method": "isotonic", "x": [-10.0, 0.0, 10.0], "y": [0.0, 0.3, 1.0]}):
        out = apply_calibrator(spec, z)
        assert np.all(np.isfinite(out)) and out.min() >= 0 and out.max() <= 1


def test_monotonic_mapping_preserves_ordering_for_parametric_methods():
    z = np.sort(np.random.default_rng(4).normal(0, 3, 400))
    for spec in ({"method": "temperature", "T": 1.7}, {"method": "platt", "a": 0.4, "b": -2.0}):
        assert np.all(np.diff(apply_calibrator(spec, z)) > 0)


def test_threshold_mapping_gives_identical_binary_decisions():
    r = np.random.default_rng(5)
    p = np.clip(r.beta(0.7, 4, 5000), 1e-6, 1 - 1e-6)
    thr = float(np.sort(p)[3500])                    # threshold ON an observed score, as in C4
    for spec in ({"method": "temperature", "T": 1.6}, {"method": "platt", "a": 0.55, "b": -1.1}):
        assert decision_mismatches(p, thr, spec) == 0
        assert 0 < calibrated_threshold(spec, thr) < 1
    plateau = {"method": "isotonic", "x": [-20.0, -2.0, 20.0], "y": [0.0, 0.2, 0.2]}   # flat for z >= -2, which contains logit(thr)
    assert decision_mismatches(p, thr, plateau) > 0  # plateaus cannot preserve decisions: flagged by the runner


def test_patient_folds_are_leak_free_deterministic_and_balanced():
    r = np.random.default_rng(6)
    n_pat = 800
    pats = np.repeat([f"p{i}" for i in range(n_pat)], r.integers(1, 5, n_pat))
    labels = (r.random((len(pats), 14)) < np.linspace(0.03, 0.4, 14)).astype(float)
    valid = r.random(labels.shape) > 0.05
    a = patient_folds(pats, valid, labels, 5, seed=11)
    b = patient_folds(pats, valid, labels, 5, seed=11)
    c = patient_folds(pats, valid, labels, 5, seed=12)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    for k in range(5):
        assert not (set(pats[a == k]) & set(pats[a != k]))             # no patient on both sides of any fold
    assert set(a) == set(range(5))
    pos = ((labels == 1) & valid).sum(0)
    for j in range(14):
        per_fold = np.array([((labels[a == k, j] == 1) & valid[a == k, j]).sum() for k in range(5)])
        assert per_fold.min() > 0 and per_fold.max() < 2.0 * pos[j] / 5 + 5     # roughly even positives per fold


def test_metrics_known_values_and_adaptive_bins():
    p, y = np.array([0.0, 0.5, 1.0, 0.5]), np.array([0, 1, 1, 0])
    assert brier(p, y) == pytest.approx((0 + 0.25 + 0 + 0.25) / 4)
    assert log_loss(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(np.log(2))
    perfect = np.linspace(0.01, 0.99, 1000)
    yy = np.random.default_rng(7).random(1000) < perfect
    assert ece_adaptive(perfect, yy, 10) < 0.06
    bins = adaptive_bins(np.random.default_rng(8).random(95), np.zeros(95, bool), 10)
    assert sum(b["n"] for b in bins) == 95 and all(b["n"] > 0 for b in bins)       # no empty bins
    assert ece_adaptive(np.full(1000, 0.9), np.zeros(1000, bool)) == pytest.approx(0.9)


def test_nan_inf_and_degenerate_labels_are_rejected():
    for bad in (np.array([0.2, np.nan]), np.array([0.2, np.inf]), np.array([-0.1, 0.5]), np.array([0.5, 1.5])):
        with pytest.raises(CalibrationError):
            safe_logit(bad)
    z = np.linspace(-3, 3, 50)
    with pytest.raises(CalibrationError):
        fit_platt(np.r_[z, np.nan], np.r_[np.arange(50) % 2, 1])
    for fitter in (fit_temperature, fit_platt, fit_isotonic):
        with pytest.raises(CalibrationError):
            fitter(z, np.zeros(50, bool))
        with pytest.raises(CalibrationError):
            fitter(z, np.ones(50, bool))
    with pytest.raises(CalibrationError):
        fit_calibrator("spline", z, np.arange(50) % 2)


def test_calibrator_file_round_trip_and_validation(tmp_path):
    z, y = _synthetic(n=3000)
    specs = {lab: fit_calibrator(m, z, y) for lab, m in zip(LABELS, ["raw", "temperature", "platt", "isotonic"] * 4)}
    path = tmp_path / "c.json"
    save_calibrators(path, specs, {"fitting_split": "validation"})
    back = load_calibrators(path)
    assert list(back["classes"]) == list(LABELS)
    for lab in LABELS:
        assert np.array_equal(apply_calibrator(back["classes"][lab], z[:200]), apply_calibrator(specs[lab], z[:200]))
    bad = dict(back)
    bad["classes"] = {k: v for k, v in back["classes"].items() if k != "Edema"}
    import json
    path.write_text(json.dumps(bad))
    with pytest.raises(Exception):
        load_calibrators(path)


def test_class_metrics_reports_overconfidence_in_slope_and_intercept():
    z, y = _synthetic(n=30000, true_T=2.0)
    m = class_metrics(sigmoid(z), y)
    assert m["cal_slope"] == pytest.approx(0.5, rel=0.1)          # overconfident scores -> slope < 1
    shifted = class_metrics(sigmoid(z + 1.5), y)
    assert shifted["cal_intercept_slope1"] < -0.5                 # over-prediction -> negative intercept
