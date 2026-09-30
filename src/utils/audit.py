"""Collect data-quality warnings as structured records instead of crashing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """Accumulates per-record issues so one bad sample never aborts an analysis."""

    dataset: str
    records: list[dict] = field(default_factory=list)

    def add(self, issue: str, record_id: str | None = None, detail: str = "", severity: str = "warning") -> None:
        self.records.append(
            {"dataset": self.dataset, "severity": severity, "issue": issue,
             "record_id": record_id, "detail": detail}
        )
        if severity == "error":
            log.error("[%s] %s %s %s", self.dataset, issue, record_id or "", detail)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["dataset", "severity", "issue", "record_id", "detail"])

    def summary(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return pd.DataFrame(columns=["dataset", "severity", "issue", "count"])
        return df.groupby(["dataset", "severity", "issue"]).size().rename("count").reset_index()

    def save(self, path: Path) -> None:
        self.to_frame().to_csv(path, index=False)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
