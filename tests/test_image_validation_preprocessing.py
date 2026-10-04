"""Image validator decisions and preprocessing geometry."""

import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.transforms import EvalTransform, PreprocessConfig, letterbox
from src.validation.image_validator import ImageValidator

rng = np.random.default_rng(0)


def _png(arr: np.ndarray, mode: str | None = None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _radiograph_like(h=320, w=390) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    base = 120 + 60 * np.sin(xx / 25.0) * np.cos(yy / 30.0)
    return np.clip(base + rng.normal(0, 15, (h, w)), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def v():
    return ImageValidator()


def test_plausible_grayscale_image_is_valid(v):
    r = v.validate(_png(_radiograph_like()))
    assert r.valid and not r.errors
    assert r.metrics["width"] == 390 and r.metrics["channels"] == 1


def test_missing_zero_byte_and_corrupt(v, tmp_path):
    assert v.validate(tmp_path / "nope.png").errors[0]["code"] == "missing_file"
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert v.validate(empty).errors[0]["code"] == "zero_byte_file"
    good = _png(_radiograph_like())
    assert v.validate(good[: len(good) // 2]).errors[0]["code"] == "unreadable_image"
    assert v.validate(b"not an image at all").errors[0]["code"] == "unreadable_image"


@pytest.mark.parametrize("value, code", [(0, "fully_black_image"), (255, "fully_white_image"), (128, "constant_image")])
def test_constant_images_rejected(v, value, code):
    r = v.validate(_png(np.full((300, 300), value, np.uint8)))
    assert not r.valid and r.errors[0]["code"] == code


def test_nearly_blank_image_rejected(v):
    a = np.full((320, 390), 255, np.uint8)
    idx = rng.random(a.shape) < 0.03
    a[idx] = 0  # sparse speckle, like the confirmed blank CheXpert files
    r = v.validate(_png(a))
    assert not r.valid and r.errors[0]["code"] == "nearly_white_image"


def test_tiny_and_extreme_aspect_rejected_but_wide_crop_only_warned(v):
    assert v.validate(_png(_radiograph_like(40, 40))).errors[0]["code"] == "invalid_dimensions"
    assert v.validate(_png(_radiograph_like(100, 900))).errors[0]["code"] == "extreme_aspect_ratio"
    wide = v.validate(_png(_radiograph_like(320, 800)))  # like the valid collimated CheXpert crops
    assert wide.valid and any(w["code"] == "unusual_aspect_ratio" for w in wide.warnings)


def test_strongly_coloured_photo_rejected_grey_rgb_accepted(v):
    photo = np.zeros((300, 300, 3), np.uint8)
    photo[..., 0] = 200
    photo[:150, :, 1] = 180
    photo[:, 150:, 2] = 220
    photo = np.clip(photo + rng.integers(0, 40, photo.shape), 0, 255).astype(np.uint8)
    r = v.validate(_png(photo))
    assert not r.valid and r.errors[0]["code"] == "strongly_colored_image"
    grey = np.repeat(_radiograph_like()[..., None], 3, axis=2)
    assert v.validate(_png(grey)).valid  # an RGB-encoded radiograph is not rejected


def test_screenshot_like_flat_region_warns(v):
    a = _radiograph_like()
    a[:, :200] = 240  # large perfectly flat panel
    r = v.validate(_png(a))
    assert r.valid and any(w["code"] == "dominant_flat_region" for w in r.warnings)


def test_unsupported_format(v):
    buf = io.BytesIO()
    Image.fromarray(_radiograph_like()).save(buf, format="GIF")
    assert v.validate(buf.getvalue()).errors[0]["code"] == "unsupported_format"


def test_preprocessing_shape_and_normalisation():
    x = EvalTransform(PreprocessConfig())(Image.fromarray(_radiograph_like()))
    assert tuple(x.shape) == (3, 224, 224)
    assert np.allclose(x[0].numpy() * 0.229 + 0.485, x[1].numpy() * 0.224 + 0.456, atol=1e-5)  # replicated grey


@pytest.mark.parametrize("w, h", [(390, 320), (320, 390), (930, 320), (320, 320)])
def test_letterbox_preserves_aspect_ratio(w, h):
    img = Image.new("L", (w, h), color=200)
    out = np.asarray(letterbox(img, PreprocessConfig()))
    assert out.shape == (224, 224)
    rows, cols = np.nonzero(out > 100)  # content region (pad is 0)
    cw, ch = cols.max() - cols.min() + 1, rows.max() - rows.min() + 1
    assert max(cw, ch) == 224
    assert abs(cw / ch - w / h) < 0.02 * (w / h) + 1 / 224 * 2


def test_eval_transform_is_deterministic():
    img = Image.fromarray(_radiograph_like())
    t = EvalTransform(PreprocessConfig())
    assert (t(img) == t(img)).all()
