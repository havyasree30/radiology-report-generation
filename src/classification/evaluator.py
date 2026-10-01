"""Batched prediction over a loader: logits -> independent sigmoid probabilities.

Numerical behaviour is fixed by an InferencePolicy (default: the canonical
fp32_strict policy); see src/classification/inference_policy.py.
"""

from __future__ import annotations

import numpy as np
import torch

from src.classification.inference_policy import CANONICAL, InferencePolicy, apply


@torch.no_grad()
def predict(model, loader, device, loss_fn=None, policy: InferencePolicy = CANONICAL) -> dict:
    apply(policy)
    model.eval()
    probs, targets, masks, idx = [], [], [], []
    loss_sum, n_valid = 0.0, 0.0
    use_autocast = policy.autocast_fp16 and device.type == "cuda"
    for x, y, m, i in loader:
        x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            logits = model(x)
        logits = logits.float()
        if loss_fn is not None:
            nv = m.sum().item()
            loss_sum += loss_fn(logits, y, m).item() * nv
            n_valid += nv
        probs.append(torch.sigmoid(logits).cpu())
        targets.append(y.cpu())
        masks.append(m.cpu())
        idx.append(i)
    return {"probs": torch.cat(probs).numpy().astype(np.float64), "targets": torch.cat(targets).numpy(),
            "mask": torch.cat(masks).numpy().astype(bool), "index": torch.cat(idx).numpy(),
            "loss": loss_sum / n_valid if n_valid else float("nan"), "policy": policy.name}
