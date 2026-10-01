"""The ONE canonical numerical policy for classifier inference.

Applies to validation probability generation, AUROC/AUPRC evaluation, threshold
fitting, single-image inference, structured findings and (eventually) the
locked-test evaluation. Training may still use AMP; inference may not.

Canonical policy ("fp32_strict"):
  * no autocast: weights and activations in IEEE float32
  * TF32 disabled for cuDNN convolutions and CUDA matmuls (PyTorch enables TF32
    for cuDNN by default on Ampere GPUs, which silently lowers precision)
  * cuDNN deterministic algorithms, benchmark off
  * probabilities = sigmoid(float32 logits), stored as float64 without rounding

Rationale: it is the only precision that is identical in meaning on every
device (CPU, any GPU), so validation, thresholds and deployed inference use the
same numbers. It was chosen for consistency, not for its metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

CANONICAL_POLICY_NAME = "fp32_strict"


@dataclass(frozen=True)
class InferencePolicy:
    name: str
    autocast_fp16: bool
    allow_tf32: bool
    cudnn_deterministic: bool

    def as_dict(self) -> dict:
        return asdict(self)


CANONICAL = InferencePolicy(CANONICAL_POLICY_NAME, autocast_fp16=False, allow_tf32=False, cudnn_deterministic=True)
# Historical policy of the original C1 evaluation (kept only to reproduce those numbers).
LEGACY_C1_AMP = InferencePolicy("amp_fp16_legacy_c1", autocast_fp16=True, allow_tf32=True, cudnn_deterministic=True)

POLICIES = {p.name: p for p in (CANONICAL, LEGACY_C1_AMP)}


def apply(policy: InferencePolicy = CANONICAL) -> InferencePolicy:
    """Set the global backend flags for this policy and return it."""
    torch.backends.cudnn.allow_tf32 = policy.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = policy.allow_tf32
    torch.backends.cudnn.deterministic = policy.cudnn_deterministic
    torch.backends.cudnn.benchmark = not policy.cudnn_deterministic
    return policy


def current_flags() -> dict:
    return {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}


def get(name: str) -> InferencePolicy:
    if name not in POLICIES:
        raise ValueError(f"unknown inference policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name]
