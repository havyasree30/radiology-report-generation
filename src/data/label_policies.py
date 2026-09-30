"""Training-time label policies for CheXpert uncertain (-1) and unmentioned (NaN) labels.

These functions never modify their input. They return a NEW float target
array plus a boolean mask telling the loss which entries contribute.

Uncertain policies (candidates for controlled comparison; none is the default):
    "zero"  U-Zero    -1 -> 0
    "one"   U-One     -1 -> 1
    "mask"  U-Masked  -1 excluded from the loss

Unmentioned (NaN) policies:
    "zero"  NaN -> 0 (treated as absent, contributes to the loss)
    "mask"  NaN excluded from the loss
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNCERTAIN_POLICIES = ("zero", "one", "mask")
MISSING_POLICIES = ("zero", "mask")


@dataclass(frozen=True)
class LabelPolicy:
    uncertain: str
    missing: str

    def __post_init__(self) -> None:
        if self.uncertain not in UNCERTAIN_POLICIES:
            raise ValueError(f"uncertain policy must be one of {UNCERTAIN_POLICIES}, got {self.uncertain!r}")
        if self.missing not in MISSING_POLICIES:
            raise ValueError(f"missing policy must be one of {MISSING_POLICIES}, got {self.missing!r}")

    @property
    def name(self) -> str:
        return f"U-{self.uncertain.capitalize()}_NaN-{self.missing.capitalize()}"


def apply_label_policy(labels: np.ndarray, policy: LabelPolicy) -> tuple[np.ndarray, np.ndarray]:
    """Map raw labels {1, 0, -1, NaN} to (targets, mask) under ``policy``.

    Raises on any value outside {1, 0, -1, NaN} rather than guessing.
    """
    raw = np.asarray(labels, dtype=float)
    finite = raw[~np.isnan(raw)]
    bad = ~np.isin(finite, (1.0, 0.0, -1.0))
    if bad.any():
        raise ValueError(f"unexpected label values: {np.unique(finite[bad])[:10]}")

    targets = np.zeros_like(raw, dtype=np.float32)
    mask = np.ones_like(raw, dtype=bool)
    targets[raw == 1.0] = 1.0

    unc = raw == -1.0
    if policy.uncertain == "one":
        targets[unc] = 1.0
    elif policy.uncertain == "mask":
        mask[unc] = False

    nan = np.isnan(raw)
    if policy.missing == "mask":
        mask[nan] = False
    return targets, mask


def compute_pos_weight(
    targets: np.ndarray, mask: np.ndarray, zero_fallback: float = 1.0
) -> dict[str, np.ndarray]:
    """Per-class ``pos_weight = N_negative / N_positive`` over UNMASKED entries.

    Only entries that actually enter the loss are counted, so uncertain or
    unmentioned labels are handled exactly as the chosen policy handles them.

    Degenerate classes get ``zero_fallback`` (1.0 = no reweighting) and are
    flagged, instead of producing inf (no positives) or 0 (no negatives, which
    would silently delete the positive term from the loss).
    """
    t = np.asarray(targets, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if t.shape != m.shape:
        raise ValueError("targets and mask must have the same shape")
    n_pos = ((t == 1.0) & m).sum(axis=0).astype(np.int64)
    n_neg = ((t == 0.0) & m).sum(axis=0).astype(np.int64)
    degenerate = (n_pos == 0) | (n_neg == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(degenerate, zero_fallback, n_neg / np.maximum(n_pos, 1)).astype(np.float64)
    return {"n_positive": n_pos, "n_negative": n_neg, "n_masked": (~m).sum(axis=0).astype(np.int64),
            "pos_weight": w, "degenerate": degenerate}
