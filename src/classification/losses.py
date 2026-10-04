"""Masked multi-label losses on LOGITS (PyTorch). Numerical specification:
src/imbalance/losses.py (NumPy reference); tests assert agreement.

All losses compute element-wise terms (reduction='none'), multiply by the
validity mask, and normalise by the number of VALID entries, so masked labels
contribute exactly zero loss and zero gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(elem.dtype)
    # where(): a masked entry's value (even inf/nan) cannot leak into the result or its gradient
    elem = torch.where(mask, elem, torch.zeros_like(elem))
    return (elem * m).sum() / m.sum().clamp_min(1.0)


class MaskedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, targets, mask):
        elem = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none",
                                                  pos_weight=self.pos_weight)
        return _masked_mean(elem, mask)


class MaskedFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets, mask):
        x, y = logits.float(), targets.float()
        log_pt = y * F.logsigmoid(x) + (1 - y) * F.logsigmoid(-x)
        elem = -((1 - log_pt.exp()) ** self.gamma) * log_pt
        if self.alpha is not None:
            elem = elem * (y * self.alpha + (1 - y) * (1 - self.alpha))
        return _masked_mean(elem, mask)


class MaskedAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gp, self.gn, self.clip = gamma_pos, gamma_neg, clip

    def forward(self, logits, targets, mask):
        x, y = logits.float(), targets.float()
        p = torch.sigmoid(x)
        loss_pos = -((1 - p) ** self.gp) * F.logsigmoid(x)
        if self.clip > 0:
            p_m = (p - self.clip).clamp_min(0)
            loss_neg = -(p_m ** self.gn) * torch.log((1 - p_m).clamp_min(1e-12))
        else:
            loss_neg = -(p ** self.gn) * F.logsigmoid(-x)
        return _masked_mean(y * loss_pos + (1 - y) * loss_neg, mask)


def temper_pos_weight(raw: torch.Tensor, how: str) -> torch.Tensor:
    if how == "none":
        return raw
    if how == "sqrt":
        return raw.sqrt()
    raise ValueError(f"unknown tempering {how!r}")


def build_loss(cfg: dict, pos_weight: torch.Tensor | None) -> nn.Module:
    t = cfg["type"]
    if t == "bce":
        return MaskedBCEWithLogits()
    if t == "weighted_bce":
        if pos_weight is None:
            raise ValueError("weighted_bce requires training-split pos_weight")
        return MaskedBCEWithLogits(temper_pos_weight(pos_weight, cfg.get("pos_weight_tempering", "none")))
    if t == "focal":
        return MaskedFocalLoss(cfg.get("gamma", 2.0), cfg.get("alpha"))
    if t == "asl":
        return MaskedAsymmetricLoss(cfg.get("gamma_pos", 0.0), cfg.get("gamma_neg", 4.0), cfg.get("clip", 0.05))
    raise ValueError(f"unknown loss type {t!r}")
