"""Reference multi-label losses: known reductions and numerical stability."""

import numpy as np
import pytest

from src.imbalance.losses import asymmetric_loss, bce_with_logits, focal_loss, sigmoid

rng = np.random.default_rng(0)
X = rng.normal(0, 2, size=(32, 14))
Y = (rng.random((32, 14)) < 0.2).astype(float)


def test_bce_matches_closed_form():
    p = sigmoid(X)
    expected = -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean()
    assert bce_with_logits(X, Y) == pytest.approx(expected, rel=1e-9)


def test_unit_pos_weight_equals_bce():
    assert bce_with_logits(X, Y, pos_weight=np.ones(14)) == pytest.approx(bce_with_logits(X, Y))


def test_pos_weight_scales_only_positive_term():
    w = np.full(14, 3.0)
    p = sigmoid(X)
    expected = -(3.0 * Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean()
    assert bce_with_logits(X, Y, pos_weight=w) == pytest.approx(expected, rel=1e-9)


def test_focal_gamma_zero_equals_bce():
    assert focal_loss(X, Y, gamma=0.0) == pytest.approx(bce_with_logits(X, Y), rel=1e-9)


def test_focal_downweights_easy_examples():
    x = np.array([[6.0, -6.0]])  # confident and correct
    y = np.array([[1.0, 0.0]])
    assert focal_loss(x, y, gamma=2.0) < bce_with_logits(x, y) * 1e-3


def test_asl_reduces_to_bce():
    assert asymmetric_loss(X, Y, gamma_pos=0, gamma_neg=0, clip=0) == pytest.approx(bce_with_logits(X, Y), rel=1e-9)


def test_asl_clip_discards_very_easy_negatives():
    x = np.array([[-5.0]])  # p ~ 0.0067 < clip
    y = np.array([[0.0]])
    assert asymmetric_loss(x, y, gamma_pos=0, gamma_neg=4, clip=0.05) == 0.0


def test_mask_excludes_entries():
    mask = np.zeros_like(Y, dtype=bool)
    mask[:, 0] = True
    assert bce_with_logits(X, Y, mask=mask) == pytest.approx(bce_with_logits(X[:, :1], Y[:, :1]))
    assert bce_with_logits(X, Y, mask=np.zeros_like(mask)) == 0.0


def test_labels_are_independent_no_softmax_coupling():
    # Changing one class's logit must not change another class's loss contribution.
    x2 = X.copy()
    x2[:, 5] += 10
    keep = np.ones_like(Y, dtype=bool)
    keep[:, 5] = False
    assert bce_with_logits(X, Y, mask=keep) == pytest.approx(bce_with_logits(x2, Y, mask=keep))


@pytest.mark.parametrize("fn", [bce_with_logits, lambda x, y: focal_loss(x, y, 2.0),
                                lambda x, y: asymmetric_loss(x, y)])
def test_stable_for_extreme_logits(fn):
    x = np.array([[1000.0, -1000.0, 1000.0, -1000.0]])
    y = np.array([[1.0, 0.0, 0.0, 1.0]])
    v = fn(x, y)
    assert np.isfinite(v)
