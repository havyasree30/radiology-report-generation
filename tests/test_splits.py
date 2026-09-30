"""Deterministic group-level splitting and leakage invariants.

The last tests check the ACTUAL split manifests committed under data/splits/.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (LeakageError, assert_no_group_overlap, max_relative_deviation, random_group_split,
                             stratified_group_split, stratified_simple_group_split)
from src.utils.config import SPLITS_DIR

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _synthetic(n_groups=3000, n_labels=6, seed=1):
    rng = np.random.default_rng(seed)
    ids = np.array([f"p{i:05d}" for i in range(n_groups)])
    sizes = rng.integers(1, 6, n_groups)
    prev = np.array([0.4, 0.2, 0.1, 0.05, 0.02, 0.01])[:n_labels]
    labels = rng.binomial(sizes[:, None], prev[None, :])
    return ids, labels, sizes


def test_random_split_is_deterministic_and_seed_sensitive():
    ids, _, sizes = _synthetic()
    a = random_group_split(ids, sizes, RATIOS, seed=42)
    assert a == random_group_split(ids, sizes, RATIOS, seed=42)
    assert a != random_group_split(ids, sizes, RATIOS, seed=43)


def test_stratified_split_is_deterministic_and_input_order_invariant():
    ids, lab, sizes = _synthetic()
    a = stratified_group_split(ids, lab, sizes, RATIOS, seed=42)
    perm = np.random.default_rng(9).permutation(len(ids))
    b = stratified_group_split(ids[perm], lab[perm], sizes[perm], RATIOS, seed=42)
    assert a == b


def test_every_group_assigned_exactly_once_with_target_proportions():
    ids, lab, sizes = _synthetic()
    for assign in (random_group_split(ids, sizes, RATIOS, 42), stratified_group_split(ids, lab, sizes, RATIOS, 42)):
        assert set(assign) == set(ids)
        img = pd.Series(sizes, index=ids).groupby(pd.Series(assign)).sum() / sizes.sum()
        for k, r in RATIOS.items():
            assert img[k] == pytest.approx(r, abs=0.02)


def test_stratification_stabilises_rare_label_prevalence():
    ids, lab, sizes = _synthetic()
    # expand to image level
    img_group = np.repeat(ids, sizes)
    Y = np.zeros((sizes.sum(), lab.shape[1]), bool)
    start = 0
    for g in range(len(ids)):
        for j in range(lab.shape[1]):
            Y[start:start + lab[g, j], j] = True
        start += sizes[g]
    strat = stratified_group_split(ids, lab, sizes, RATIOS, 42)
    s = np.array([strat[g] for g in img_group])
    assert max_relative_deviation(Y, s) < 0.05
    for j in range(lab.shape[1]):
        for k in RATIOS:
            assert Y[s == k, j].any(), "rare label missing from a split"


def test_bad_ratios_rejected():
    ids, lab, sizes = _synthetic(100)
    with pytest.raises(ValueError):
        random_group_split(ids, sizes, {"train": 0.8, "val": 0.3}, 1)


def test_leakage_detector_catches_overlap_and_missing_ids():
    ok = pd.DataFrame({"pid": ["a", "a", "b", "c"], "split": ["train", "train", "val", "test"]})
    assert_no_group_overlap(ok, "pid")
    with pytest.raises(LeakageError):
        assert_no_group_overlap(pd.DataFrame({"pid": ["a", "a"], "split": ["train", "test"]}), "pid")
    with pytest.raises(LeakageError):
        assert_no_group_overlap(pd.DataFrame({"pid": ["a", None], "split": ["train", "val"]}), "pid")


def test_simple_stratified_split_keeps_strata_proportions():
    ids = [str(i) for i in range(1000)]
    strata = ["normal" if i % 3 == 0 else "abnormal" for i in range(1000)]
    a = stratified_simple_group_split(ids, strata, RATIOS, 7)
    assert a == stratified_simple_group_split(ids, strata, RATIOS, 7)
    df = pd.DataFrame({"id": ids, "stratum": strata, "split": [a[i] for i in ids]})
    share = df.groupby("split")["stratum"].apply(lambda s: (s == "normal").mean())
    assert share.max() - share.min() < 0.01
    with pytest.raises(ValueError):
        stratified_simple_group_split(["1", "1"], ["a", "a"], RATIOS, 7)


# ---------------- the real CheXpert manifests ----------------
CX = SPLITS_DIR / "chexpert"
needs_cx = pytest.mark.skipif(not (CX / "chexpert_image_manifest.csv.gz").exists(),
                              reason="CheXpert split manifest not generated yet")


@needs_cx
def test_real_chexpert_zero_patient_leakage():
    m = pd.read_csv(CX / "chexpert_image_manifest.csv.gz")
    sets = {s: set(g["patient_id"]) for s, g in m.groupby("split")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    for s in ("train", "val", "test"):
        assert not sets[s] & sets["official_valid"]
    assert_no_group_overlap(m, "patient_id")
    assert_no_group_overlap(m, "split_group_id")


@needs_cx
def test_real_chexpert_duplicate_identities_share_a_split():
    links = pd.read_csv(SPLITS_DIR.parent.parent / "results/eda/tables/chexpert_cross_patient_duplicate_links.csv")
    pats = pd.read_csv(CX / "chexpert_patient_splits.csv").set_index("patient_id")["split"]
    for group in links["patients"]:
        assert pats.loc[group.split(";")].nunique() == 1


@needs_cx
def test_real_chexpert_split_is_reproducible_metadata():
    meta = json.loads((CX / "split_metadata.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    assert meta["leakage_checks"] == {"patient_overlap": 0, "split_group_overlap": 0}
