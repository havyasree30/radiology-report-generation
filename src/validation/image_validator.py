"""Structured plausibility checks for an input image, run BEFORE preprocessing.

Result: {"valid": bool, "errors": [...], "warnings": [...], "metrics": {...}}
    errors   -> HARD rejection (file unusable, or image content cannot support inference)
    warnings -> image accepted but atypical; downstream components should surface them

These are heuristics. Passing validation does not establish that an image is a
chest radiograph, and failing a warning check does not make an image invalid.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageFile, UnidentifiedImageError

from src.preprocessing.grayscale import to_grayscale
from src.utils.config import CONFIG_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass
class ValidationResult:
    valid: bool
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    image: Image.Image | None = None  # decoded image when readable (not serialised)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings, "metrics": self.metrics}


def load_validator_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG_DIR / "validation" / "image_validator.yaml").read_text(encoding="utf-8"))


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness; ~0 for grayscale images."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


class ImageValidator:
    def __init__(self, config: dict | None = None):
        self.cfg = config or load_validator_config()

    def _fail(self, res: ValidationResult, code: str, msg: str) -> ValidationResult:
        res.valid = False
        res.errors.append({"code": code, "message": msg})
        return res

    def validate(self, source: str | Path | bytes) -> ValidationResult:
        res = ValidationResult(valid=True)
        hard, warn = self.cfg["hard"], self.cfg["warn"]

        # ---- file level ----
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                return self._fail(res, "missing_file", f"{p} does not exist")
            data = p.read_bytes()
        else:
            data = bytes(source)
        res.metrics["file_bytes"] = len(data)
        if len(data) == 0:
            return self._fail(res, "zero_byte_file", "file is empty")

        try:
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            img.load()  # full decode: truncated/corrupt data raises here
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            return self._fail(res, "unreadable_image", f"cannot decode image: {type(e).__name__}")
        res.metrics.update(format=fmt, mode=img.mode, width=img.width, height=img.height,
                           channels=len(img.getbands()))
        if fmt not in self.cfg["allowed_formats"]:
            return self._fail(res, "unsupported_format", f"format {fmt} not in {self.cfg['allowed_formats']}")

        # ---- geometry ----
        w, h = img.size
        short, long_ = min(w, h), max(w, h)
        ar = w / h if h else float("inf")
        res.metrics["aspect_ratio"] = ar
        if short < hard["min_side_px"] or long_ > hard["max_side_px"]:
            return self._fail(res, "invalid_dimensions", f"{w}x{h} outside [{hard['min_side_px']}, {hard['max_side_px']}] px")
        if not hard["aspect_ratio_min"] <= ar <= hard["aspect_ratio_max"]:
            return self._fail(res, "extreme_aspect_ratio", f"aspect ratio {ar:.2f}")
        if short < warn["tiny_side_px"]:
            res.warnings.append({"code": "small_image", "message": f"shortest side {short}px < {warn['tiny_side_px']}px"})
        if not warn["aspect_ratio_min"] <= ar <= warn["aspect_ratio_max"]:
            res.warnings.append({"code": "unusual_aspect_ratio", "message": f"aspect ratio {ar:.2f}"})

        # ---- channels / colour ----
        bands = img.getbands()
        if len(bands) not in (1, 2, 3, 4):
            return self._fail(res, "unexpected_channel_count", f"{len(bands)} channels")
        if "A" in bands:
            alpha = np.asarray(img.getchannel("A"))
            if (alpha < 255).mean() > 0.01:
                res.warnings.append({"code": "transparency", "message": "image has a non-trivial alpha channel"})
        if img.mode in ("RGB", "RGBA", "P", "CMYK", "YCbCr", "LAB", "HSV"):
            rgb = np.asarray(img.convert("RGB"))
            c = colorfulness(rgb[::4, ::4])
            res.metrics["colorfulness"] = c
            if c >= hard["colorfulness"]:
                return self._fail(res, "strongly_colored_image",
                                  f"colourfulness {c:.1f} >= {hard['colorfulness']}: not consistent with a radiograph")
            if c >= warn["colorfulness"]:
                res.warnings.append({"code": "color_content", "message": f"colourfulness {c:.1f}"})
        else:
            res.metrics["colorfulness"] = 0.0

        # ---- intensity content ----
        g = to_grayscale(img)
        g.thumbnail((512, 512))
        a = np.asarray(g, dtype=np.float32)
        mean, std = float(a.mean()), float(a.std())
        white, black = float((a >= 250).mean()), float((a <= 5).mean())
        dominant = float(np.bincount(a.astype(np.uint8).ravel(), minlength=256).max() / a.size)
        res.metrics.update(mean_intensity=mean, std_intensity=std, frac_near_white=white,
                           frac_near_black=black, dominant_value_fraction=dominant)
        if std < hard["min_pixel_std"]:
            code = "fully_black_image" if mean <= 5 else "fully_white_image" if mean >= 250 else "constant_image"
            return self._fail(res, code, f"pixel std {std:.2f} < {hard['min_pixel_std']}")
        if white >= hard["saturated_fraction"]:
            return self._fail(res, "nearly_white_image", f"{100 * white:.1f}% near-white pixels")
        if black >= hard["saturated_fraction"]:
            return self._fail(res, "nearly_black_image", f"{100 * black:.1f}% near-black pixels")
        if mean < warn["dark_mean"]:
            res.warnings.append({"code": "very_dark_image", "message": f"mean intensity {mean:.1f}"})
        if mean > warn["bright_mean"]:
            res.warnings.append({"code": "very_bright_image", "message": f"mean intensity {mean:.1f}"})
        if dominant >= warn["dominant_value_fraction"]:
            res.warnings.append({"code": "dominant_flat_region",
                                 "message": f"{100 * dominant:.1f}% of pixels share one grey level (screenshot/synthetic?)"})
        res.image = img
        return res
