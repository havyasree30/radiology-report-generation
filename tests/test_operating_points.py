"""C3 threshold code: Youden J, F1, tie rules, 0.50 baseline, edge cases, determinism, files."""

import json

import numpy as np
import pytest

from src.classification.labels import LABELS
from src.classification.operating_points import (ScoreError, apply_thresholds, candidate_table, check_scores,
                                                 confusion_counts, f1_threshold, load_threshold_file, macro_summary,
                                                 metrics_from_counts, operating_point, patient_bootstrap_youden,
                                                 per_class_table, save_threshold_file, youden_threshold)

# 3 positives, 3 negatives; hand-computed in the comments below
Y = np.array([1, 1, 1, 0, 0, 0])
P = np.array([0.9, 0.8, 0.4, 0.6, 0.3, 0.1])


def test_youden_on_known_toy_with_exact_tie_prefers_higher_sensitivity():
    # thresholds desc: .9 .8 .6 .4 .3 .1 ; tp 1 2 2 3 3 3 ; fp 0 0 1 1 2 3 ; P=N=3
    # J = tp/3 - fp/3 = .333 .667 .333 .667 .333 0  -> exact tie at 0.8 (sens 2/3) and 0.4 (sens 1)
    r = youden_threshold(Y, P)
    assert r["n_tied_at_max_j"] == 2
    assert r["threshold"] == pytest.approx(0.4)  # tie rule: higher sensitivity
    op = operating_point(Y, P, r["threshold"])
    assert (op["tp"], op["fp"], op["tn"], op["fn"]) == (3, 1, 2, 0)
    assert op["youden_j"] == pytest.approx(2 / 3)


def test_f1_optimal_on_known_toy():
    # F1 = 2tp/(tp+fp+P): .5 .8 .667 .857 .75 .667  -> best at 0.4
    r = f1_threshold(Y, P)
    assert r["threshold"] == pytest.approx(0.4)
    assert operating_point(Y, P, 0.4)["f1"] == pytest.approx(6 / 7)


def test_f1_tie_prefers_higher_recall_then_higher_threshold():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6])
    # thr .9: tp1 fp0 -> 2/3 ; thr .6: tp2 fp2 -> 4/6 = 2/3 (exact tie, recall .5 vs 1.0)
    assert f1_threshold(y, p)["threshold"] == pytest.approx(0.6)


def test_duplicate_scores_collapse_into_one_candidate_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.8, 0.1])  # 0.8 is shared by one positive and one negative
    c = candidate_table(y, p)
    assert list(c.thresholds) == [0.9, 0.8, 0.1]            # duplicates collapse into one candidate (>= rule)
    assert list(c.tp) == [1, 2, 2] and list(c.fp) == [0, 1, 2]


def test_duplicate_scores_are_treated_as_one_candidate_with_ge_rule():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    r = youden_threshold(y, p)
    assert r["threshold"] is None and "J > 0" in r["undefined_reason"]  # constant scores carry no information


def test_baseline_050_confusion_and_metrics_from_counts():
    c = confusion_counts(Y, P, 0.5)
    assert c == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    m = metrics_from_counts(**c)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3) and m["youden_j"] == pytest.approx(1 / 3)


def test_metrics_division_by_zero_gives_nan_not_zero():
    m = metrics_from_counts(tp=0, fp=0, tn=5, fn=3)   # nothing predicted positive
    assert np.isnan(m["precision"]) and m["recall"] == 0.0 and m["f1"] == 0.0
    m = metrics_from_counts(tp=0, fp=0, tn=5, fn=0)   # no positives, no predictions
    assert np.isnan(m["recall"]) and np.isnan(m["precision"]) and np.isnan(m["f1"])


def test_thresholds_applied_per_class_without_normalisation():
    scores = np.array([[0.9, 0.2, 0.8], [0.4, 0.6, 0.1]])
    thr = np.array([0.5, 0.15, 0.7])
    pred = apply_thresholds(scores, thr)
    assert pred.tolist() == [[True, True, True], [False, True, False]]
    assert pred[0].sum() == 3                       # several findings may be positive at once
    assert scores.sum(axis=1).tolist() == pytest.approx([1.9, 1.1])  # sums are arbitrary, never forced to 1
    assert not np.isclose(scores.sum(axis=1), 1.0).all()


def test_nan_inf_and_out_of_range_scores_are_rejected():
    for bad in (np.array([0.1, np.nan]), np.array([0.1, np.inf]), np.array([-0.1, 0.5]), np.array([0.5, 1.2])):
        with pytest.raises(ScoreError):
            check_scores(bad)
        with pytest.raises(ScoreError):
            youden_threshold(np.array([1, 0]), bad)
    with pytest.raises(ScoreError):
        apply_thresholds(np.array([[0.1, 0.2]]), np.array([0.5, np.nan]))
    with pytest.raises(ScoreError):
        youden_threshold(np.array([0, 2]), np.array([0.1, 0.2]))  # non-binary labels


def test_degenerate_label_sets_return_reason_not_a_threshold():
    p = np.array([0.1, 0.4, 0.7])
    assert youden_threshold(np.zeros(3), p)["threshold"] is None
    assert "no valid positives" in youden_threshold(np.zeros(3), p)["undefined_reason"]
    assert "no valid negatives" in f1_threshold(np.ones(3), p)["undefined_reason"]
    # inverted scores: J <= 0 everywhere -> undefined, not invented
    assert youden_threshold(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9]))["threshold"] is None


def test_weighted_candidates_equal_replicated_rows():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.9, 0.7, 0.6, 0.4, 0.2])
    w = np.array([2, 1, 0, 3, 1])
    rep = np.repeat(np.arange(5), w)
    a, b = candidate_table(y, p, weights=w), candidate_table(y[rep], p[rep])
    assert np.array_equal(a.thresholds[np.isin(a.thresholds, b.thresholds)], b.thresholds)
    assert (a.P, a.N) == (b.P, b.N)
    assert youden_threshold(y, p, weights=w)["threshold"] == youden_threshold(y[rep], p[rep])["threshold"]


def _synthetic(n=600, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random((n, 14)) < np.linspace(0.05, 0.5, 14)).astype(float)
    s = np.clip(0.55 * y + 0.5 * r.random((n, 14)), 0, 1)
    valid = r.random((n, 14)) > 0.05
    patients = np.array([f"p{i // 3}" for i in range(n)])
    return s, y, valid, patients


def test_per_class_table_preserves_class_order_and_marks_undefined():
    s, y, valid, _ = _synthetic()
    y[:, 3] = 0  # class 3 has no positives
    rows = per_class_table(s, y, valid, [0.5] * 3 + [None] + [0.5] * 10)
    assert [r["observation"] for r in rows] == list(LABELS)
    assert rows[3]["status"] == "undefined" and rows[3]["threshold"] is None
    macro = macro_summary(rows)
    assert macro["n_classes_defined_recall"] == 13


def test_threshold_json_round_trip_keeps_full_precision_and_order(tmp_path):
    thr = np.random.default_rng(1).random(14)
    classes = {lab: {"threshold": float(t), "youden_j": 0.5} for lab, t in zip(LABELS, thr)}
    path = tmp_path / "t.json"
    save_threshold_file(path, "youden_j", "validation", "ckpt.pt", "abc", classes)
    assert np.array_equal(load_threshold_file(path), thr)           # bit-exact
    assert list(json.loads(path.read_text())["classes"]) == list(LABELS)
    classes["Edema"]["threshold"] = None
    save_threshold_file(path, "youden_j", "validation", "ckpt.pt", "abc", classes)
    with pytest.raises(ValueError):
        load_threshold_file(path)
    save_threshold_file(path, "youden_j", "test", "ckpt.pt", "abc", {lab: {"threshold": 0.5} for lab in LABELS})
    with pytest.raises(ValueError):
        load_threshold_file(path)                                    # only validation-derived thresholds load


def test_patient_bootstrap_is_deterministic_and_seed_sensitive():
    s, y, valid, pat = _synthetic()
    a = patient_bootstrap_youden(s, y, valid, pat, n_boot=25, seed=7)
    b = patient_bootstrap_youden(s, y, valid, pat, n_boot=25, seed=7)
    c = patient_bootstrap_youden(s, y, valid, pat, n_boot=25, seed=8)
    assert a.shape == (25, 14) and np.array_equal(a, b, equal_nan=True) and not np.array_equal(a, c, equal_nan=True)
    for j in range(14):  # every bootstrap threshold is an observed valid score of that class
        ok = a[:, j][np.isfinite(a[:, j])]
        assert np.isin(ok, s[valid[:, j], j]).all()


def test_bootstrap_resamples_patients_not_images():
    # one patient owns all positives; any resample that omits that patient has NO positives -> NaN
    n = 120
    pat = np.array(["pos_patient"] * 10 + [f"n{i}" for i in range(n - 10)])
    y = np.zeros((n, 14))
    y[:10, 0] = 1
    s = np.where(y == 1, 0.9, 0.1)
    valid = np.ones((n, 14), bool)
    out = patient_bootstrap_youden(s, y, valid, pat, n_boot=300, seed=1)[:, 0]
    frac_nan = np.isnan(out).mean()
    expected = (1 - 1 / 111) ** 111  # P(a specific patient is never drawn), 111 unique patients
    assert abs(frac_nan - expected) < 0.06
    assert np.isnan(out).any() and np.isfinite(out).any()
