"""Classifier invariants: label order, model outputs, masked losses, checkpoints, thresholds, findings."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.classification.checkpointing import REQUIRED_META, load_checkpoint, save_checkpoint  # noqa: E402
from src.classification.dataset import LockedSplitError, label_policy_from_config, load_split_frame  # noqa: E402
from src.classification.findings import (build_query, build_structured_findings,  # noqa: E402
                                         consistency_warnings, label_predictions)
from src.classification.labels import LABELS, NUM_LABELS, assert_matches  # noqa: E402
from src.classification.losses import (MaskedAsymmetricLoss, MaskedBCEWithLogits,  # noqa: E402
                                       MaskedFocalLoss, build_loss)
from src.classification.metrics import per_class_metrics, summary_metrics  # noqa: E402
from src.classification.model import build_densenet121, predict_proba  # noqa: E402
from src.classification.thresholds import load_thresholds, save_thresholds, youden_thresholds  # noqa: E402
from src.imbalance import losses as ref  # noqa: E402
from src.utils.config import SPLITS_DIR, load_paths  # noqa: E402

EXPECTED = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]


# ---------------- label order ----------------
def test_exact_label_order():
    assert list(LABELS) == EXPECTED and NUM_LABELS == 14


def test_label_order_matches_phase1_artifacts():
    w = json.loads((SPLITS_DIR / "chexpert" / "training_pos_weights.json").read_text(encoding="utf-8"))
    assert_matches(w["targets"])
    with pytest.raises(ValueError):
        assert_matches(EXPECTED[::-1])


def test_label_order_matches_source_csv_header():
    try:
        paths = load_paths()
    except FileNotFoundError:
        pytest.skip("no local dataset config")
    header = pd.read_csv(paths.chexpert_train_csv, nrows=0).columns.tolist()
    assert header[5:] == EXPECTED


# ---------------- split isolation ----------------
def test_patient_split_isolation_frontal_manifest():
    m = pd.read_csv(SPLITS_DIR / "chexpert" / "chexpert_image_manifest.csv.gz")
    f = m[m["Frontal/Lateral"] == "Frontal"]
    sets = {s: set(g["patient_id"]) for s, g in f.groupby("split")}
    for a in sets:
        for b in sets:
            if a < b:
                assert not sets[a] & sets[b], f"{a} and {b} share patients"


def test_locked_test_split_refuses_to_load():
    cfg = {"manifest": "data/splits/chexpert/chexpert_image_manifest.csv.gz", "locked_splits": ["test"], "view": "Frontal"}
    with pytest.raises(LockedSplitError):
        load_split_frame(None, cfg, "test")


def test_label_policy_names():
    assert label_policy_from_config({"uncertain": "ignore", "blank": "zero"}).name == "U-Mask_NaN-Zero"
    assert label_policy_from_config({"uncertain": "u_one", "blank": "ignore"}).name == "U-One_NaN-Mask"
    with pytest.raises(ValueError):
        label_policy_from_config({"uncertain": "maybe", "blank": "zero"})


# ---------------- model ----------------
@pytest.fixture(scope="module")
def model():
    return build_densenet121(pretrained=None).eval()


def test_output_shape_is_logits_without_softmax(model):
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (3, 14)
    assert not isinstance(model.classifier, torch.nn.Sequential)  # plain Linear head, no activation
    assert (logits < 0).any() or (logits > 1).any()  # raw logits, not probabilities
    assert not torch.allclose(logits.sum(1), torch.ones(3))  # not a normalised distribution


def test_independent_sigmoid_probabilities_can_exceed_one_in_sum(model):
    with torch.no_grad():
        model.classifier.bias.fill_(3.0)  # every label strongly positive
        p = predict_proba(model, torch.randn(2, 3, 224, 224))
        model.classifier.bias.zero_()
    assert ((p > 0.5).sum(1) == 14).all()
    assert (p.sum(1) > 1.0).all()  # not normalised
    # each probability is exactly sigmoid of its own logit
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        assert torch.allclose(predict_proba(model, x), torch.sigmoid(model(x)), atol=1e-6)


# ---------------- losses ----------------
def _batch(n=8):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(n, 14, generator=g, requires_grad=True)
    y = (torch.rand(n, 14, generator=g) < 0.3).float()
    m = torch.rand(n, 14, generator=g) > 0.3
    return logits, y, m


@pytest.mark.parametrize("fn", [MaskedBCEWithLogits(), MaskedBCEWithLogits(torch.full((14,), 5.0)),
                                MaskedFocalLoss(2.0), MaskedAsymmetricLoss()])
def test_masked_entries_get_zero_gradient_and_loss_is_finite(fn):
    logits, y, m = _batch()
    loss = fn(logits, y, m)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.all(logits.grad[~m] == 0)
    assert torch.any(logits.grad[m] != 0)


def test_masked_value_changes_do_not_change_loss():
    logits, y, m = _batch()
    fn = MaskedBCEWithLogits()
    y2 = y.clone()
    y2[~m] = 1 - y2[~m]
    l2 = logits.detach().clone()
    l2[~m] = 1e4  # even extreme logits at masked positions are irrelevant
    assert torch.allclose(fn(logits, y, m), fn(l2, y2, m))


def test_torch_losses_match_numpy_reference():
    logits, y, m = _batch(32)
    x, yy, mm = logits.detach().numpy(), y.numpy(), m.numpy()
    w = np.linspace(0.5, 9, 14)
    assert MaskedBCEWithLogits()(logits, y, m).item() == pytest.approx(ref.bce_with_logits(x, yy, mm), rel=1e-5)
    assert MaskedBCEWithLogits(torch.tensor(w, dtype=torch.float32))(logits, y, m).item() == pytest.approx(
        ref.bce_with_logits(x, yy, mm, pos_weight=w), rel=1e-5)
    assert MaskedFocalLoss(2.0)(logits, y, m).item() == pytest.approx(ref.focal_loss(x, yy, 2.0, mask=mm), rel=1e-5)
    assert MaskedAsymmetricLoss()(logits, y, m).item() == pytest.approx(ref.asymmetric_loss(x, yy, mask=mm), rel=1e-4)


def test_weighted_bce_requires_train_weights_and_tempers():
    with pytest.raises(ValueError):
        build_loss({"type": "weighted_bce"}, None)
    fn = build_loss({"type": "weighted_bce", "pos_weight_tempering": "sqrt"}, torch.tensor([4.0] * 14))
    assert torch.allclose(fn.pos_weight, torch.tensor([2.0] * 14))


# ---------------- CUDA ----------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward_pass_amp(model):
    m = build_densenet121(pretrained=None).cuda().eval()
    x = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = m(x)
    assert out.device.type == "cuda" and out.shape == (2, 14) and torch.isfinite(out.float()).all()


# ---------------- checkpoint ----------------
def test_checkpoint_roundtrip_and_missing_fails(tmp_path, model):
    meta = {k: None for k in REQUIRED_META}
    meta.update(architecture="densenet121", num_labels=14, label_order=list(LABELS), epoch=1)
    save_checkpoint(tmp_path / "c.pt", model, meta)
    loaded, m2 = load_checkpoint(tmp_path / "c.pt")
    assert m2["label_order"] == list(LABELS)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "bad.pt", model, {"architecture": "densenet121"})


# ---------------- metrics / thresholds ----------------
def _val_data(n=400, seed=1):
    r = np.random.default_rng(seed)
    y = (r.random((n, 14)) < 0.2).astype(np.float32)
    p = np.clip(0.6 * y + 0.4 * r.random((n, 14)), 0, 1)
    m = np.ones_like(y, dtype=bool)
    return y, p, m


def test_youden_threshold_maximises_j():
    y, p, m = _val_data()
    thr = youden_thresholds(y, p, m, source_split="val")
    for j, e in enumerate(thr):
        pred = p[:, j] >= e["threshold"]
        sens = (pred & (y[:, j] == 1)).sum() / (y[:, j] == 1).sum()
        spec = (~pred & (y[:, j] == 0)).sum() / (y[:, j] == 0).sum()
        assert sens == pytest.approx(e["sensitivity"]) and spec == pytest.approx(e["specificity"])
        grid = np.unique(p[:, j])
        best = max(((p[:, j] >= t)[y[:, j] == 1].mean() - (p[:, j] >= t)[y[:, j] == 0].mean()) for t in grid)
        assert e["youden_j"] == pytest.approx(best)


def test_thresholds_only_from_validation_and_undefined_not_invented(tmp_path):
    y, p, m = _val_data()
    with pytest.raises(ValueError):
        youden_thresholds(y, p, m, source_split="test")
    y[:, 11] = 0  # no positives for one class
    thr = youden_thresholds(y, p, m, source_split="val")
    assert thr[11]["threshold"] is None and "no valid positives" in thr[11]["undefined_reason"]
    save_thresholds(thr, tmp_path / "t.json", {"source_split": "val"})
    with pytest.raises(ValueError):
        load_thresholds(tmp_path / "t.json")
    save_thresholds(thr, tmp_path / "t2.json", {"source_split": "test"})
    with pytest.raises(ValueError):
        load_thresholds(tmp_path / "t2.json")


def test_undefined_metrics_are_nan_not_zero():
    y, p, m = _val_data()
    y[:, 0] = 0
    pc = per_class_metrics(y, p, m)
    assert np.isnan(pc.loc[0, "auroc"]) and pc.loc[0, "undefined_reason"] == "no valid positives"
    s = summary_metrics(y, p, m)
    assert s["n_classes_defined"] == 13 and "No Finding" in s["undefined_classes"]


# ---------------- findings / query ----------------
def _preds(probs, thr=None):
    return label_predictions(np.asarray(probs), np.full(14, 0.5) if thr is None else np.asarray(thr))


def test_all_14_returned_and_unrounded():
    probs = np.linspace(0.01, 0.99, 14) + 1e-7
    out = _preds(probs)
    assert [d["observation"] for d in out] == list(LABELS)
    assert out[0]["probability"] == pytest.approx(probs[0], abs=0) and out[0]["status"] == "negative"


def test_structured_findings_partition_and_borderline():
    probs = np.full(14, 0.1)
    probs[LABELS.index("Pleural Effusion")] = 0.8
    probs[LABELS.index("Edema")] = 0.52
    probs[LABELS.index("Cardiomegaly")] = 0.47
    s = build_structured_findings(_preds(probs), borderline_margin=0.05)
    pos = [d["observation"] for d in s["positive_findings"]]
    assert pos == ["Pleural Effusion", "Edema"]  # ordered by margin over threshold
    assert len(s["positive_findings"]) + len(s["negative_findings"]) == 14
    assert {d["observation"] for d in s["borderline_findings"]} == {"Edema", "Cardiomegaly"}
    q = build_query(s)
    assert q["query_text"] == "Chest radiograph findings: pleural effusion and edema."
    assert q["borderline_negative_observations"] == ["Cardiomegaly"]
    assert "0.8" not in q["query_text"]


def test_no_positive_case_is_not_forced_and_not_called_normal():
    s = build_structured_findings(_preds(np.full(14, 0.2)), 0.05)
    assert s["positive_findings"] == []
    q = build_query(s)
    assert q["query_mode"] == "no_positive_findings" and "normal" not in q["query_text"].lower()


def test_no_finding_contradiction_warning_without_altering_outputs():
    probs = np.full(14, 0.1)
    probs[LABELS.index("No Finding")] = 0.9
    probs[LABELS.index("Pneumothorax")] = 0.9
    preds = _preds(probs)
    w = consistency_warnings(preds)
    assert w and w[0]["code"] == "no_finding_with_pathology" and w[0]["observations"] == ["Pneumothorax"]
    assert [d["probability"] for d in preds] == list(probs)
    probs[LABELS.index("Pneumothorax")] = 0.1
    probs[LABELS.index("Support Devices")] = 0.9  # devices are compatible with No Finding
    assert consistency_warnings(_preds(probs)) == []


def test_query_no_finding_positive():
    probs = np.full(14, 0.1)
    probs[LABELS.index("No Finding")] = 0.9
    q = build_query(build_structured_findings(_preds(probs), 0.05))
    assert q["query_mode"] == "no_finding_positive" and q["query_text"] == "Chest radiograph: no finding."


def test_invalid_prediction_inputs_rejected():
    with pytest.raises(ValueError):
        label_predictions(np.full(13, 0.5), np.full(13, 0.5))
    with pytest.raises(ValueError):
        label_predictions(np.full(14, 1.5), np.full(14, 0.5))
