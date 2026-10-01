"""Deterministic, aspect-ratio-preserving preprocessing to a 3x224x224 ImageNet-normalised tensor.

Pipeline (eval):  PIL image -> grayscale 'L' -> resize longest side to S (bilinear)
                  -> center-pad to SxS with pad_value -> [0,1] float -> replicate to 3 channels
                  -> ImageNet mean/std normalisation
Train adds a mild random affine (rotation / translation / scale) AFTER letterboxing,
filling with the same pad value. No flips, no colour jitter, no CLAHE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from src.preprocessing.grayscale import to_grayscale

_INTERP = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
           "lanczos": Image.Resampling.LANCZOS}


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    interpolation: str = "bilinear"
    pad_value: int = 0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(image_size=int(d["image_size"]), interpolation=d["interpolation"], pad_value=int(d["pad_value"]),
                   mean=tuple(d["mean"]), std=tuple(d["std"]))


@dataclass(frozen=True)
class AugmentConfig:
    rotation_degrees: float = 5.0
    translate_fraction: float = 0.05
    scale_range: tuple[float, float] = (0.95, 1.05)

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentConfig":
        for forbidden in ("horizontal_flip", "vertical_flip", "color_jitter", "clahe"):
            if d.get(forbidden):
                raise ValueError(f"{forbidden} is not allowed in the primary baseline")
        return cls(float(d["rotation_degrees"]), float(d["translate_fraction"]), tuple(d["scale_range"]))


def letterbox(img: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    """Resize so the LONGEST side equals image_size (aspect preserved), then center-pad."""
    s = cfg.image_size
    w, h = img.size
    scale = s / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), _INTERP[cfg.interpolation])
    canvas = Image.new("L", (s, s), color=cfg.pad_value)
    canvas.paste(resized, ((s - nw) // 2, (s - nh) // 2))
    return canvas


def random_affine(img: Image.Image, aug: AugmentConfig, rng: np.random.Generator, pad_value: int) -> Image.Image:
    """Mild rotation + translation + isotropic scale about the image centre."""
    angle = rng.uniform(-aug.rotation_degrees, aug.rotation_degrees)
    s = img.size[0]
    tx = rng.uniform(-aug.translate_fraction, aug.translate_fraction) * s
    ty = rng.uniform(-aug.translate_fraction, aug.translate_fraction) * s
    scale = rng.uniform(*aug.scale_range)
    import torchvision.transforms.functional as TF
    return TF.affine(img, angle=angle, translate=[int(round(tx)), int(round(ty))], scale=scale, shear=[0.0],
                     interpolation=TF.InterpolationMode.BILINEAR, fill=pad_value)


def to_tensor(img: Image.Image, cfg: PreprocessConfig) -> torch.Tensor:
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)       # [H, W]
    x = a.unsqueeze(0).repeat(3, 1, 1)                                     # [3, H, W]
    mean = torch.tensor(cfg.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg.std, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


class EvalTransform:
    """Deterministic preprocessing for validation, test and inference."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return to_tensor(letterbox(to_grayscale(img), self.cfg), self.cfg)


class TrainTransform:
    """Letterbox + mild random affine. Randomness comes from the worker-seeded numpy RNG."""

    def __init__(self, cfg: PreprocessConfig, aug: AugmentConfig):
        self.cfg, self.aug = cfg, aug

    def __call__(self, img: Image.Image) -> torch.Tensor:
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        boxed = letterbox(to_grayscale(img), self.cfg)
        return to_tensor(random_affine(boxed, self.aug, rng, self.cfg.pad_value), self.cfg)
