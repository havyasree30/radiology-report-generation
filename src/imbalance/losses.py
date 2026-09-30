"""Reference (NumPy) implementations of the candidate multi-label losses.

Phase 1 deliberately has no deep-learning framework installed. These exact,
framework-free formulas are the specification that the Phase 2 PyTorch
implementations must match numerically (tests will compare them).

All losses are MULTI-LABEL: every class is an independent binary problem on
its own sigmoid output. There is no softmax and no coupling between classes.

Shapes: logits, targets, mask are (N, C). ``mask`` marks entries that
contribute to the loss (e.g. uncertain labels under U-Masked are False).
Reduction: mean over contributing entries.

Candidates (no winner is chosen here):
    A  bce                 standard BCE-with-logits
    B  bce + pos_weight    positive-weighted BCE, weights from TRAINING split only
    C  focal               Lin et al. (2017), per-label binary focal loss
    D  asymmetric (ASL)    Ridnik et al. (2021), asymmetric focusing + probability margin
"""

from __future__ import annotations

import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    # log(sigmoid(x)) = -softplus(-x), computed stably.
    return -np.logaddexp(0.0, -x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(_log_sigmoid(x))


def _reduce(loss: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        return float(loss.mean())
    m = np.asarray(mask, dtype=bool)
    n = m.sum()
    return float(loss[m].sum() / n) if n else 0.0


def bce_with_logits(logits, targets, mask=None, pos_weight=None) -> float:
    """-(w_c * y * log p + (1 - y) * log(1 - p)), w_c = pos_weight (default 1)."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = 1.0 if pos_weight is None else np.asarray(pos_weight, dtype=np.float64)[None, :]
    loss = -(w * y * _log_sigmoid(x) + (1.0 - y) * _log_sigmoid(-x))
    return _reduce(loss, mask)


def focal_loss(logits, targets, gamma: float = 2.0, alpha: float | None = None, mask=None) -> float:
    """Binary focal loss applied independently per label.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t), p_t = p if y=1 else 1-p.
    gamma=0 and alpha=None reduces exactly to BCE.
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    log_p, log_1mp = _log_sigmoid(x), _log_sigmoid(-x)
    log_pt = y * log_p + (1.0 - y) * log_1mp
    pt = np.exp(log_pt)
    loss = -((1.0 - pt) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * (y * alpha + (1.0 - y) * (1.0 - alpha))
    return _reduce(loss, mask)


def asymmetric_loss(logits, targets, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                    clip: float = 0.05, mask=None) -> float:
    """Asymmetric Loss (Ridnik et al., ICCV 2021).

    L+ = -(1 - p)^gamma_pos * log(p)                 for y = 1
    L- = -(p_m)^gamma_neg * log(1 - p_m),  p_m = max(p - clip, 0)   for y = 0
    With gamma_pos = gamma_neg = 0 and clip = 0 it reduces exactly to BCE.
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    p = sigmoid(x)
    loss_pos = -((1.0 - p) ** gamma_pos) * _log_sigmoid(x)
    if clip > 0:
        p_m = np.maximum(p - clip, 0.0)
        log_1m_pm = np.log(np.clip(1.0 - p_m, 1e-12, 1.0))
    else:
        p_m = p
        log_1m_pm = _log_sigmoid(-x)
    loss_neg = -(p_m ** gamma_neg) * log_1m_pm
    loss = y * loss_pos + (1.0 - y) * loss_neg
    return _reduce(loss, mask)
