"""Read-only image integrity scanning (never writes to source files).

Each file is fully decoded to detect truncation/corruption; per-file failures
are recorded and never abort the scan.
"""

from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Truncated images must raise rather than be silently padded.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None  # large radiographs are legitimate

AUDIT_COLUMNS = [
    "path", "exists", "file_bytes", "readable", "error", "format", "mode",
    "width", "height", "sha256", "dhash", "dhash256", "pixel_mean", "pixel_std",
    "frac_near_white", "frac_near_black",
]


def dhash(img: Image.Image, size: int = 8) -> str:
    """Difference hash with size*size bits as a hex string (8 -> 64 bit, 16 -> 256 bit)."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def dhash64(img: Image.Image) -> str:
    return dhash(img, 8)


def audit_one(path: str) -> dict:
    rec = dict.fromkeys(AUDIT_COLUMNS)
    rec.update(path=path, exists=False, readable=False, file_bytes=0)
    try:
        st = os.stat(path)
    except OSError as e:
        rec["error"] = f"stat: {type(e).__name__}"
        return rec
    rec["exists"] = True
    rec["file_bytes"] = st.st_size
    if st.st_size == 0:
        rec["error"] = "zero_byte"
        return rec
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        from io import BytesIO
        with Image.open(BytesIO(data)) as img:
            rec["format"], rec["mode"] = img.format, img.mode
            rec["width"], rec["height"] = img.size
            img.load()  # full decode: raises on truncated / corrupt data
            rec["dhash"] = dhash(img, 8)
            rec["dhash256"] = dhash(img, 16)
            thumb = img.convert("L")
            thumb.thumbnail((256, 256))
            arr = np.asarray(thumb, dtype=np.float32)
            rec["pixel_mean"] = float(arr.mean())
            rec["pixel_std"] = float(arr.std())
            # Share of (almost) saturated pixels: near-blank images are dominated by one extreme.
            rec["frac_near_white"] = float((arr >= 250).mean())
            rec["frac_near_black"] = float((arr <= 5).mean())
        rec["readable"] = True
    except Exception as e:  # noqa: BLE001 - any decoder failure is a finding, not a crash
        rec["error"] = stable_error(f"{type(e).__name__}: {str(e)[:200]}")
    return rec


def stable_error(message: object) -> object:
    """Drop run-specific memory addresses so audit outputs are reproducible."""
    return re.sub(r" at 0x[0-9A-Fa-f]+", "", message) if isinstance(message, str) else message


def audit_many(paths: list[str], workers: int | None = None, chunksize: int = 256, progress: bool = True) -> list[dict]:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = ex.map(audit_one, paths, chunksize=chunksize)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(paths), desc="image audit", mininterval=5)
        out.extend(it)
    return out


def walk_files(root: Path) -> list[Path]:
    """All regular files under ``root`` (fast os.scandir recursion)."""
    found: list[Path] = []
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        found.append(Path(e.path))
        except OSError:
            continue
    return found


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")
