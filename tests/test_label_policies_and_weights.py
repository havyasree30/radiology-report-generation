"""Uncertainty / missing-label policies and training-only pos_weight computation."""

import numpy as np
import pytest

from src.data.label_policies import LabelPolicy, apply_label_policy, compute_pos_weight

NAN = np.nan
RAW = np.array([[1.0, 0.0, -1.0, NAN],
                [0.0, -1.0, NAN, 1.0]])


def test_policies_never_mutate_input():
    before = RAW.copy()
    for u in ("zero", "one", "mask"):
        for m in ("zero", "mask"):
            apply_label_policy(RAW, LabelPolicy(u, m))
    np.testing.assert_array_equal(np.isnan(RAW), np.isnan(before))
    np.testing.assert_array_equal(RAW[~np.isnan(RAW)], before[~np.isnan(before)])


@pytest.mark.parametrize("u, m, targets, mask", [
    ("zero", "zero", [[1, 0, 0, 0], [0, 0, 0, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]),
    ("one", "zero", [[1, 0, 1, 0], [0, 1, 0, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]),
    ("mask", "zero", [[1, 0, 0, 0], [0, 0, 0, 1]], [[1, 1, 0, 1], [1, 0, 1, 1]]),
    ("mask", "mask", [[1, 0, 0, 0], [0, 0, 0, 1]], [[1, 1, 0, 0], [1, 0, 0, 1]]),
])
def test_policy_mapping(u, m, targets, mask):
    t, k = apply_label_policy(RAW, LabelPolicy(u, m))
    np.testing.assert_array_equal(t, np.array(targets, dtype=np.float32))
    np.testing.assert_array_equal(k, np.array(mask, dtype=bool))


def test_unexpected_label_value_raises():
    with pytest.raises(ValueError):
        apply_label_policy(np.array([[2.0]]), LabelPolicy("zero", "zero"))
    with pytest.raises(ValueError):
        LabelPolicy("maybe", "zero")


def test_pos_weight_is_negatives_over_positives():
    t = np.array([[1], [0], [0], [0]], dtype=np.float32)
    r = compute_pos_weight(t, np.ones_like(t, dtype=bool))
    assert r["pos_weight"][0] == pytest.approx(3.0)
    assert r["n_positive"][0] == 1 and r["n_negative"][0] == 3


def test_pos_weight_excludes_masked_uncertain_and_missing():
    # 1 positive, 1 explicit negative, 2 uncertain, 2 unmentioned.
    raw = np.array([[1.0], [0.0], [-1.0], [-1.0], [NAN], [NAN]])
    t, m = apply_label_policy(raw, LabelPolicy("mask", "mask"))
    assert compute_pos_weight(t, m)["pos_weight"][0] == pytest.approx(1.0)  # 1 neg / 1 pos
    t, m = apply_label_policy(raw, LabelPolicy("mask", "zero"))
    assert compute_pos_weight(t, m)["pos_weight"][0] == pytest.approx(3.0)  # (1 + 2 NaN) / 1
    t, m = apply_label_policy(raw, LabelPolicy("one", "zero"))
    assert compute_pos_weight(t, m)["pos_weight"][0] == pytest.approx(1.0)  # 3 neg / 3 pos


def test_zero_positive_class_is_safe_and_flagged():
    t = np.array([[0, 1], [0, 0]], dtype=np.float32)
    r = compute_pos_weight(t, np.ones_like(t, dtype=bool))
    assert np.isfinite(r["pos_weight"]).all()
    assert r["pos_weight"][0] == 1.0 and bool(r["degenerate"][0])
    assert r["pos_weight"][1] == pytest.approx(1.0) and not bool(r["degenerate"][1])


def test_zero_negative_class_does_not_zero_out_positive_term():
    t = np.array([[1], [1]], dtype=np.float32)
    r = compute_pos_weight(t, np.ones_like(t, dtype=bool))
    assert r["pos_weight"][0] == 1.0 and bool(r["degenerate"][0])


def test_fully_masked_class_is_degenerate():
    t = np.zeros((3, 1), dtype=np.float32)
    r = compute_pos_weight(t, np.zeros_like(t, dtype=bool))
    assert bool(r["degenerate"][0]) and r["n_masked"][0] == 3
