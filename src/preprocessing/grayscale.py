"""Torch-free grayscale conversion shared by validation and preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image


def to_grayscale(img: Image.Image) -> Image.Image:
    """Single-channel 8-bit image. 16-bit / RGB(A) inputs are converted, not rejected."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(img, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
        return Image.fromarray(a.astype(np.uint8), mode="L")
    return img.convert("L")
