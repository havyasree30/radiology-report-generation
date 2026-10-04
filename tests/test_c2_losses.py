"""C2 loss study: loss behaviour on edge cases, and proof that the four arms differ only in the loss."""

import copy

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from src.classification.losses import (MaskedAsymmetricLoss, MaskedBCEWithLogits, MaskedFocalLoss,  # noqa: E402
                                       build_loss, temper_pos_weight)
from src.imbalance import losses as ref  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402

C2 = PROJECT_ROOT / "configs/classifier/c2"
ARMS = ["c2a_bce", "c2b_sqrt_weighted_bce", "c2c_focal_gamma2", "c2d_asl"]
POS_W = torch.linspace(0.8, 75.0, 14)


def all_losses():
    return {
        "bce": MaskedBCEWithLogits(),
        "weighted_bce": MaskedBCEWithLogits(temper_pos_weight(POS_W, "sqrt")),
        "focal": MaskedFocalLoss(2.0),
        "asl": MaskedAsymmetricLoss(0.0, 4.0, 0.05),
    }


def _batch(kind: str, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 14, generator=g) * 3
    y = {"mixed": (torch.rand(n, 14, generator=g) < 0.3).float(),
         "all_negative": torch.zeros(n, 14), "all_positive": torch.ones(n, 14)}[kind]
    m = torch.rand(n, 14, generator=g) > 0.25
    return x.requires_grad_(True), y, m


@pytest.mark.parametrize("name", ["bce", "weighted_bce", "focal", "asl"])
@pytest.mark.parametrize("kind", ["mixed", "all_negative", "all_positive"])
def test_finite_loss_finite_gradients_and_masking(name, kind):
    fn = all_losses()[name]
    x, y, m = _batch(kind)
    loss = fn(x, y, m)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad[~m] == 0)            # ignored labels: zero gradient
    assert torch.any(x.grad[m] != 0)


@pytest.mark.parametrize("name", ["bce", "weighted_bce", "focal", "asl"])
def test_ignored_labels_have_zero_contribution(name):
    fn = all_losses()[name]
    x, y, m = _batch("mixed")
    x2, y2 = x.detach().clone(), y.clone()
    x2[~m] = torch.where(torch.rand_like(x2[~m]) > 0.5, 1e4, -1e4)  # garbage at masked positions
    y2[~m] = 1 - y2[~m]
    assert torch.allclose(fn(x, y, m), fn(x2, y2, m))


@pytest.mark.parametrize("name", ["bce", "weighted_bce", "focal", "asl"])
def test_extreme_logits_are_stable(name):
    fn = all_losses()[name]
    x = torch.tensor([[60.0, -60.0, 1e4, -1e4] * 3 + [0.0, 0.0]], requires_grad=True)
    for y in (torch.zeros(1, 14), torch.ones(1, 14)):
        loss = fn(x, y, torch.ones(1, 14, dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
        x.grad = None


def test_fully_masked_batch_gives_zero_loss_and_gradient():
    for fn in all_losses().values():
        x, y, _ = _batch("mixed")
        loss = fn(x, y, torch.zeros(16, 14, dtype=torch.bool))
        loss.backward()
        assert loss.item() == 0.0 and torch.all(x.grad == 0)


def test_correct_values_against_numpy_reference():
    x, y, m = _batch("mixed", n=64, seed=3)
    X, Y, M = x.detach().numpy().astype(np.float64), y.numpy(), m.numpy()
    w = temper_pos_weight(POS_W, "sqrt").numpy()
    L = all_losses()
    assert L["bce"](x, y, m).item() == pytest.approx(ref.bce_with_logits(X, Y, M), rel=1e-5)
    assert L["weighted_bce"](x, y, m).item() == pytest.approx(ref.bce_with_logits(X, Y, M, pos_weight=w), rel=1e-5)
    assert L["focal"](x, y, m).item() == pytest.approx(ref.focal_loss(X, Y, 2.0, mask=M), rel=1e-5)
    assert L["asl"](x, y, m).item() == pytest.approx(ref.asymmetric_loss(X, Y, 0.0, 4.0, 0.05, mask=M), rel=1e-4)


def test_focal_and_asl_downweight_easy_negatives_relative_to_bce():
    x = torch.full((1, 14), -3.0)  # confident, correct negatives
    y, m = torch.zeros(1, 14), torch.ones(1, 14, dtype=torch.bool)
    bce = MaskedBCEWithLogits()(x, y, m)
    assert MaskedFocalLoss(2.0)(x, y, m) < 0.01 * bce
    assert MaskedAsymmetricLoss(0.0, 4.0, 0.05)(x, y, m) == 0  # p < clip -> discarded


def test_weighted_bce_amplifies_only_positive_term():
    x = torch.zeros(1, 14)
    m = torch.ones(1, 14, dtype=torch.bool)
    w = temper_pos_weight(POS_W, "sqrt")
    neg = MaskedBCEWithLogits(w)(x, torch.zeros(1, 14), m)
    pos = MaskedBCEWithLogits(w)(x, torch.ones(1, 14), m)
    assert neg.item() == pytest.approx(np.log(2), rel=1e-6)
    assert pos.item() == pytest.approx(np.log(2) * w.mean().item(), rel=1e-6)


# ---------------- experimental control ----------------
def _load(name):
    return yaml.safe_load((C2 / f"{name}.yaml").read_text(encoding="utf-8"))


def test_c2_arms_differ_only_in_loss():
    cfgs = {a: _load(a) for a in ARMS}
    strip = lambda c: {k: v for k, v in copy.deepcopy(c).items() if k not in ("loss", "experiment_name", "output_dir")}  # noqa: E731
    base = strip(cfgs[ARMS[0]])
    for a in ARMS[1:]:
        assert strip(cfgs[a]) == base, f"{a} differs from {ARMS[0]} outside the loss"
    assert len({str(c["loss"]) for c in cfgs.values()}) == 4
    assert base["training"]["monitor"] == base["scheduler"]["monitor"] == "val_macro_auprc"
    assert base["inference"]["policy"] == "fp32_strict"


def test_c2_arms_match_c1_except_documented_protocol_changes():
    c1 = yaml.safe_load((PROJECT_ROOT / "configs/classifier/densenet121.yaml").read_text(encoding="utf-8"))
    c2 = _load(ARMS[0])
    for k in ("seed", "data", "label_policy", "preprocessing", "augmentation", "model", "optimizer", "thresholds"):
        assert c2[k] == c1[k], k
    assert {k: v for k, v in c2["training"].items() if k != "monitor"} == \
           {k: v for k, v in c1["training"].items() if k != "monitor"}
    assert {k: v for k, v in c2["scheduler"].items() if k != "monitor"} == \
           {k: v for k, v in c1["scheduler"].items() if k != "monitor"}


def test_build_loss_from_each_arm_config():
    for a in ARMS:
        fn = build_loss(_load(a)["loss"], POS_W if _load(a)["loss"]["type"] == "weighted_bce" else None)
        x, y, m = _batch("mixed")
        assert torch.isfinite(fn(x, y, m))
