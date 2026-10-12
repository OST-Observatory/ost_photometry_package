"""Write colour images: 16-bit TIFF, PNG and JPEG (north up as in DS9)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .composite import ColorResult

#: Values of ``formats``.
FORMATS = ("tiff", "png", "jpeg")


def _display(rgb: np.ndarray) -> np.ndarray:
    # FITS row 0 is the bottom row; image files start at the top.
    return np.flipud(np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0))


def write_tiff16(rgb: np.ndarray, path: str | Path, description: str = "") -> Path:
    import tifffile

    path = Path(path)
    data = np.round(_display(rgb) * 65535.0).astype(np.uint16)
    tifffile.imwrite(path, data, photometric="rgb", description=description or None)
    return path


def write_png(rgb: np.ndarray, path: str | Path) -> Path:
    from PIL import Image

    path = Path(path)
    Image.fromarray(np.round(_display(rgb) * 255.0).astype(np.uint8)).save(path)
    return path


def write_jpeg(rgb: np.ndarray, path: str | Path, quality: int = 95) -> Path:
    from PIL import Image

    path = Path(path)
    Image.fromarray(np.round(_display(rgb) * 255.0).astype(np.uint8)).save(
        path, quality=int(quality))
    return path


def write_images(result: ColorResult, base: str | Path,
                 formats: Sequence[str] = FORMATS) -> list[Path]:
    """Write ``<base>.tif`` / ``.png`` / ``.jpg``; the TIFF description holds
    the processing log as JSON."""
    unknown = sorted(set(formats) - set(FORMATS))
    if unknown:
        raise ValueError(f"Unknown format(s) {unknown}; allowed: {FORMATS}")
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if "tiff" in formats:
        written.append(write_tiff16(result.rgb, base.with_suffix(".tif"),
                                    json.dumps(result.log, default=str)))
    if "png" in formats:
        written.append(write_png(result.rgb, base.with_suffix(".png")))
    if "jpeg" in formats:
        written.append(write_jpeg(result.rgb, base.with_suffix(".jpg")))
    return written


__all__ = ["FORMATS", "write_images", "write_jpeg", "write_png", "write_tiff16"]
