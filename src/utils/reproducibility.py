"""Capture the computational environment alongside every analysis output."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from src.utils.config import PROJECT_ROOT

_TRACKED_PACKAGES = (
    "numpy", "pandas", "matplotlib", "pillow", "scikit-learn", "scipy", "pyyaml", "tqdm", "pytest",
)


def git_commit() -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("branch", "--show-current"),
        "dirty_working_tree": bool(status) if status is not None else None,
    }


def environment_snapshot(seed: int | None = None) -> dict[str, Any]:
    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "git": git_commit(),
    }
