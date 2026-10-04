"""Checkpoint save/load with mandatory metadata; missing checkpoints fail loudly."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.classification.labels import LABELS, NUM_LABELS, assert_matches
from src.classification.model import build_densenet121

REQUIRED_META = ("architecture", "num_labels", "label_order", "epoch", "val_metric_name", "val_metric",
                 "preprocessing", "label_policy", "loss", "optimizer", "scheduler", "seed", "git_commit")


def save_checkpoint(path: Path, model: torch.nn.Module, meta: dict, optimizer=None, scheduler=None) -> None:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"checkpoint metadata missing {missing}")
    assert_matches(meta["label_order"])
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state": model.state_dict(), "meta": meta}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}. Train a model first; no fallback weights are used.")
    state = torch.load(path, map_location=device, weights_only=False)
    meta = state["meta"]
    assert_matches(meta["label_order"])
    if meta["num_labels"] != NUM_LABELS or meta["architecture"] != "densenet121":
        raise ValueError(f"incompatible checkpoint: {meta['architecture']} / {meta['num_labels']} labels")
    model = build_densenet121(NUM_LABELS, pretrained=None)
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), meta


__all__ = ["save_checkpoint", "load_checkpoint", "LABELS"]
