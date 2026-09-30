"""Deterministic GROUP-level splitting and leakage checks.

The unit of assignment is always a group (CheXpert: patient, IU X-Ray:
report uid). A group is never divided between partitions.

Two CheXpert strategies are provided so their prevalence stability can be
compared empirically:
    random_group_split       shuffle groups, fill partitions by image count
    stratified_group_split   iterative multi-label stratification (Sechidis
                             et al., 2011) lifted to groups: each group carries
                             the count of positive images per label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


class LeakageError(AssertionError):
    """Raised when a group appears in more than one partition."""


def _ratios(ratios: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    names = list(ratios)
    r = np.array([ratios[n] for n in names], dtype=float)
    if np.any(r < 0) or not np.isclose(r.sum(), 1.0):
        raise ValueError(f"split ratios must be non-negative and sum to 1: {dict(ratios)}")
    return names, r


def random_group_split(group_ids: Sequence[str], group_sizes: Sequence[int],
                       ratios: Mapping[str, float], seed: int) -> dict[str, str]:
    """Shuffle groups deterministically, then fill partitions in order until
    each reaches its target share of IMAGES."""
    names, r = _ratios(ratios)
    ids = np.asarray(group_ids)
    sizes = np.asarray(group_sizes, dtype=float)
    order = np.argsort(ids, kind="stable")  # canonical order before shuffling
    ids, sizes = ids[order], sizes[order]
    perm = np.random.default_rng(seed).permutation(len(ids))
    cum = np.cumsum(sizes[perm]) / sizes.sum()
    bounds = np.cumsum(r)
    fold = np.searchsorted(bounds, cum, side="left")
    fold = np.minimum(fold, len(names) - 1)
    return {str(ids[p]): names[f] for p, f in zip(perm, fold)}


def stratified_group_split(group_ids: Sequence[str], group_label_counts: np.ndarray,
                           group_sizes: Sequence[int], ratios: Mapping[str, float],
                           seed: int) -> dict[str, str]:
    """Iterative multi-label stratification over groups.

    Repeatedly takes the label with the fewest remaining positive images among
    unassigned groups and distributes the groups carrying it to the partition
    with the largest remaining demand for that label (ties: largest remaining
    demand for images, then a seeded random choice). Groups without any
    positive label are finally assigned by remaining image demand.
    """
    names, r = _ratios(ratios)
    ids = np.asarray(group_ids)
    L = np.asarray(group_label_counts, dtype=float)
    sizes = np.asarray(group_sizes, dtype=float)
    if L.shape[0] != len(ids) or len(sizes) != len(ids):
        raise ValueError("group_ids, group_label_counts and group_sizes must align")
    canon = np.argsort(ids, kind="stable")
    ids, L, sizes = ids[canon], L[canon], sizes[canon]

    rng = np.random.default_rng(seed)
    visit_rank = np.empty(len(ids), dtype=np.int64)
    visit_rank[rng.permutation(len(ids))] = np.arange(len(ids))

    need_label = r[:, None] * L.sum(axis=0)[None, :]
    need_size = r * sizes.sum()
    fold = np.full(len(ids), -1, dtype=np.int64)

    def choose(candidates_need: np.ndarray) -> int:
        best = np.flatnonzero(candidates_need == candidates_need.max())
        if len(best) > 1:
            by_size = need_size[best]
            best = best[by_size == by_size.max()]
        return int(best[0] if len(best) == 1 else rng.choice(best))

    while True:
        unassigned = fold < 0
        remaining = L[unassigned].sum(axis=0)
        active = np.flatnonzero(remaining > 0)
        if len(active) == 0:
            break
        lab = active[np.argmin(remaining[active])]
        members = np.flatnonzero(unassigned & (L[:, lab] > 0))
        for g in members[np.argsort(visit_rank[members])]:
            k = choose(need_label[:, lab])
            fold[g] = k
            need_label[k] -= L[g]
            need_size[k] -= sizes[g]

    for g in np.flatnonzero(fold < 0)[np.argsort(visit_rank[fold < 0])]:
        k = choose(need_size)
        fold[g] = k
        need_size[k] -= sizes[g]
    return {str(i): names[f] for i, f in zip(ids, fold)}


def stratified_simple_group_split(group_ids: Sequence[str], strata: Sequence[str],
                                  ratios: Mapping[str, float], seed: int) -> dict[str, str]:
    """Per-stratum deterministic shuffle-and-cut (one row per group)."""
    names, r = _ratios(ratios)
    df = pd.DataFrame({"gid": list(map(str, group_ids)), "stratum": list(map(str, strata))})
    if df["gid"].duplicated().any():
        raise ValueError("group ids must be unique for stratified_simple_group_split")
    out: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    for _, g in df.sort_values(["stratum", "gid"]).groupby("stratum", sort=True):
        ids = g["gid"].to_numpy()[rng.permutation(len(g))]
        bounds = np.round(np.cumsum(r) * len(ids)).astype(int)
        start = 0
        for name, end in zip(names, bounds):
            for gid in ids[start:end]:
                out[gid] = name
            start = end
    return out


def assert_no_group_overlap(df: pd.DataFrame, group_col: str, split_col: str = "split") -> None:
    """Fail loudly if any group occurs in more than one partition."""
    if df[group_col].isna().any():
        raise LeakageError(f"{int(df[group_col].isna().sum())} rows have no {group_col}; cannot verify leakage")
    per_group = df.groupby(group_col)[split_col].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked):
        raise LeakageError(f"{len(leaked)} {group_col} values span multiple splits, e.g. {leaked.index[:5].tolist()}")
    sets = {s: set(g[group_col]) for s, g in df.groupby(split_col)}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                raise LeakageError(f"{a} ∩ {b} share {len(inter)} {group_col} values")


def prevalence_by_split(Y: np.ndarray, split: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Positive prevalence per label for the full cohort and each split."""
    rows = []
    split = np.asarray(split)
    full = Y.mean(axis=0)
    for s in ["all", *[x for x in SPLIT_NAMES if x in set(split)]]:
        sel = np.ones(len(split), bool) if s == "all" else split == s
        prev = Y[sel].mean(axis=0) if sel.any() else np.full(Y.shape[1], np.nan)
        pos = Y[sel].sum(axis=0)
        for j, lab in enumerate(labels):
            rows.append({"split": s, "observation": lab, "n_images": int(sel.sum()), "n_positive": int(pos[j]),
                         "prevalence": prev[j],
                         "relative_deviation": (prev[j] - full[j]) / full[j] if full[j] > 0 else np.nan})
    return pd.DataFrame(rows)


def max_relative_deviation(Y: np.ndarray, split: np.ndarray, min_positive: int = 0) -> float:
    full_pos = Y.sum(axis=0)
    keep = full_pos >= min_positive
    full = Y.mean(axis=0)
    worst = 0.0
    for s in np.unique(split):
        sel = split == s
        dev = np.abs(Y[sel].mean(axis=0) - full) / np.where(full > 0, full, np.nan)
        worst = max(worst, float(np.nanmax(dev[keep])) if keep.any() else 0.0)
    return worst
