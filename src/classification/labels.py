"""The ONE authoritative label order for the classifier.

Index i of every logit / probability / threshold / mask vector refers to
LABELS[i]. The order is the CheXpert CSV column order, verified at runtime
against the CSV header (``assert_matches``) and stored in every checkpoint.
"""

from __future__ import annotations

from src.data.chexpert import EXPECTED_OBSERVATIONS

LABELS: tuple[str, ...] = tuple(EXPECTED_OBSERVATIONS)
NUM_LABELS = len(LABELS)
NO_FINDING = "No Finding"
SUPPORT_DEVICES = "Support Devices"
# Observations that describe pathology (used for the No Finding consistency check).
PATHOLOGY_LABELS: tuple[str, ...] = tuple(l for l in LABELS if l not in (NO_FINDING, SUPPORT_DEVICES))

assert NUM_LABELS == 14


def assert_matches(order) -> None:
    """Fail loudly if an external label order (CSV header, checkpoint) differs."""
    if tuple(order) != LABELS:
        raise ValueError(f"label order mismatch:\n expected {list(LABELS)}\n got      {list(order)}")
