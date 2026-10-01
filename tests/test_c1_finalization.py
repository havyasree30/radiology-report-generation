"""Regression tests from the C1 finalization: lossless serialisation, canonical inference
precision, threshold/status consistency, and integrity of the frozen C1 artifacts."""

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.classification import inference_policy  # noqa: E402
from src.classification.evaluator import predict  # noqa: E402
from src.classification.labels import LABELS  # noqa: E402
from src.classification.serialization import (load_predictions_csv, load_predictions_npz,  # noqa: E402
                                              roundtrip_report, save_predictions)
from src.classification.thresholds import load_thresholds, save_thresholds, youden_thresholds  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402

C1 = PROJECT_ROOT / "results/classification"
CANON = C1 / "densenet121_baseline/evaluation/fp32_strict"


def _tied_predictions(n=600, seed=0):
    """Probabilities with many exact ties (like fp16 outputs) and values 1 ULP around a threshold."""
    r = np.random.default_rng(seed)
    grid = np.round(r.random((n, 14)), 3)  # heavy ties
    p = grid + r.random((n, 14)) * 1e-9 * (r.random((n, 14)) < 0.3)
    y = (r.random((n, 14)) < 0.25 + 0.5 * p).astype(np.float64)
    raw = y.copy()
    raw[r.random((n, 14)) < 0.05] = -1.0
    return p, raw, raw != -1.0


def test_17_digit_csv_and_npz_are_lossless_and_status_preserving(tmp_path):
    p, raw, valid = _tied_predictions()
    thr = np.array([np.sort(p[:, j])[len(p) // 2] for j in range(14)])  # thresholds ON observed values
    p[0] = np.nextafter(thr, 0)  # 1 ULP below every threshold
    p[1] = thr                     # exactly on every threshold
    files = save_predictions(tmp_path, "pred", [f"img{i}" for i in range(len(p))], ["pat"] * len(p), p, raw, valid)
    for loader in (load_predictions_npz, load_predictions_csv):
        back = loader(files["npz"] if loader is load_predictions_npz else files["csv"])
        rep = roundtrip_report(p, back["probs"], thr)
        assert rep["bit_identical"] and rep["max_abs_diff"] == 0.0 and rep["status_mismatches"] == 0
        assert np.array_equal(back["valid"], valid)
        assert np.array_equal(np.nan_to_num(back["raw_labels"], nan=9), np.nan_to_num(raw, nan=9))


def test_default_pandas_parser_is_not_exact_but_loader_is(tmp_path):
    # Documents the actual C1 root cause: values were written exactly, but pandas' default
    # float parser is not correctly rounded; the project loader must use round_trip parsing.
    import pandas as pd
    vals = np.random.default_rng(1).random((5_000, 14))
    files = save_predictions(tmp_path, "p", [str(i) for i in range(len(vals))], ["x"] * len(vals), vals,
                             np.zeros_like(vals), np.ones_like(vals, dtype=bool))
    default = pd.read_csv(files["csv"])[[f"prob::{l}" for l in LABELS]].to_numpy()
    assert np.any(default != vals)                                    # default reader loses the last bit
    assert np.array_equal(load_predictions_csv(files["csv"])["probs"], vals)  # project reader is exact
    assert np.all(np.array([float(f"{v:.17g}") for v in vals.ravel()]) == vals.ravel())


def test_thresholds_and_statuses_consistent_after_reload(tmp_path):
    p, raw, valid = _tied_predictions(seed=3)
    y = (raw == 1).astype(np.float64)
    entries = youden_thresholds(y, p, valid, source_split="val")
    save_thresholds(entries, tmp_path / "t.json", {"source_split": "val"})
    thr = load_thresholds(tmp_path / "t.json")
    files = save_predictions(tmp_path, "pred", [str(i) for i in range(len(p))], ["x"] * len(p), p, raw, valid)
    p_back = load_predictions_csv(files["csv"])["probs"]
    assert np.array_equal(p >= thr, p_back >= thr)
    for j, e in enumerate(entries):
        v = valid[:, j]
        pred, yj = p_back[v, j] >= thr[j], y[v, j] == 1
        assert (pred & yj).sum() / yj.sum() == e["sensitivity"]  # exact, not approximate


def test_canonical_policy_definition_and_apply():
    c = inference_policy.CANONICAL
    assert c.name == "fp32_strict" and not c.autocast_fp16 and not c.allow_tf32 and c.cudnn_deterministic
    inference_policy.apply(c)
    f = inference_policy.current_flags()
    assert f == {"cudnn_allow_tf32": False, "matmul_allow_tf32": False, "cudnn_deterministic": True, "cudnn_benchmark": False}
    with pytest.raises(ValueError):
        inference_policy.get("fp8_fast")


class _Toy(Dataset):
    def __init__(self, n=40):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 32, 32, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], torch.zeros(14), torch.ones(14, dtype=torch.bool), i


def _toy_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                               torch.nn.Flatten(), torch.nn.Linear(8, 14))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_canonical_predict_is_fp32_sigmoid_and_deterministic(device):
    model = _toy_model().to(device)
    loader = DataLoader(_Toy(), batch_size=16, shuffle=False)
    a = predict(model, loader, torch.device(device))
    b = predict(model, loader, torch.device(device))
    assert a["policy"] == "fp32_strict" and np.array_equal(a["probs"], b["probs"])
    with torch.no_grad():  # same batching as the loader (cuDNN kernels depend on batch shape)
        ref = np.concatenate([torch.sigmoid(model(xb.to(device)).float()).cpu().numpy() for xb, *_ in loader])
    assert np.array_equal(a["probs"], ref.astype(np.float64))  # no autocast, no extra transformation
    assert not np.allclose(a["probs"].sum(1), 1.0)  # independent sigmoids, not a softmax


def test_single_image_path_uses_canonical_policy():
    from src.classification.model import predict_proba
    torch.backends.cudnn.allow_tf32 = True  # simulate the PyTorch default
    predict_proba(_toy_model(), _Toy().x[:1])
    assert torch.backends.cudnn.allow_tf32 is False


# ---------------- frozen C1 artifacts ----------------
needs_c1 = pytest.mark.skipif(not (C1 / "C1_PROVENANCE.json").exists(), reason="C1 not finalized yet")


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@needs_c1
def test_c1_checkpoint_is_byte_identical_to_provenance():
    prov = json.loads((C1 / "C1_PROVENANCE.json").read_text(encoding="utf-8"))
    ckpt = PROJECT_ROOT / prov["checkpoint"]["path"]
    if not ckpt.exists():
        pytest.skip("checkpoint is git-ignored and not present on this machine")
    assert _sha(ckpt) == prov["checkpoint"]["sha256"]


@needs_c1
def test_c1_canonical_predictions_reproduce_stored_operating_points_exactly():
    d = load_predictions_npz(CANON / "validation_predictions.npz")
    thr = load_thresholds(CANON / "thresholds/youden_j_thresholds.json")
    stored = json.loads((CANON / "thresholds/youden_j_thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    csv = load_predictions_csv(CANON / "validation_predictions.csv.gz")
    assert np.array_equal(d["probs"], csv["probs"])  # 17-digit CSV == npz, bit for bit
    y, v = d["raw_labels"] == 1, d["valid"]
    for j, e in enumerate(stored):
        pred = d["probs"][v[:, j], j] >= thr[j]
        yj = y[v[:, j], j]
        # stored values are TPR and 1 - FPR; identical counts, last-bit float rounding may differ
        assert (pred & yj).sum() / yj.sum() == pytest.approx(e["sensitivity"], rel=0, abs=1e-15)
        assert (~pred & ~yj).sum() / (~yj).sum() == pytest.approx(e["specificity"], rel=0, abs=1e-15)
        assert int((pred & yj).sum()) == round(e["sensitivity"] * yj.sum())
    assert list(LABELS) == json.loads((CANON / "thresholds/youden_j_thresholds.json").read_text(encoding="utf-8"))["labels"]


@needs_c1
def test_c1_canonical_thresholds_declare_canonical_policy():
    prov = json.loads((CANON / "thresholds/youden_j_thresholds.json").read_text(encoding="utf-8"))["provenance"]
    assert prov["source_split"] == "val" and prov["inference_policy"]["name"] == "fp32_strict"
    assert prov["backend_flags"]["cudnn_allow_tf32"] is False
