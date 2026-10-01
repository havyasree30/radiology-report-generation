"""DenseNet121 with a 14-logit head. The forward pass returns LOGITS only.

No sigmoid and no softmax inside the model: losses consume logits, and
inference applies an element-wise sigmoid (``predict_proba``), giving 14
independent probabilities that need not sum to 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.classification.labels import NUM_LABELS


def build_densenet121(num_labels: int = NUM_LABELS, pretrained: str | None = "IMAGENET1K_V1") -> nn.Module:
    weights = DenseNet121_Weights[pretrained] if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_labels)
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Independent per-label probabilities: sigmoid(logits). Never softmax, never renormalised.

    Runs under the canonical inference policy (fp32, TF32 off), the same policy
    used to generate validation probabilities and fit thresholds.
    """
    from src.classification.inference_policy import CANONICAL, apply
    apply(CANONICAL)
    model.eval()
    return torch.sigmoid(model(x).float())
