"""Weighted average of images that respects masked and clipped pixels.

Workaround for ccdproc: ``Combiner.average_combine`` with ``weights``
divides the weighted sum of a pixel by the sum of *all* weights,
including those of frames whose value at that pixel is masked or
sigma-clipped (``Combiner._weighted_sum``). Pixels masked in some frames
come out too dark by the masked weight fraction: holes and dark columns
in the background, and stars too faint where clipping hits their cores.
Unweighted averages and the median are not affected.

Checked up to ccdproc :data:`CCDPROC_CHECKED_VERSION`. Once ccdproc fixes
it, :func:`weighted_average_combine` can go back to ``ccdproc.combine``
(see ``docs/TODO.md``, "ccdproc weighted average"); the regression test
``tests/test_weighted_combine.py::test_ccdproc_weighted_average_bug``
fails as soon as an installed ccdproc behaves correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clip as astropy_sigma_clip

#: Newest ccdproc version in which the weighted-average bug was confirmed.
CCDPROC_CHECKED_VERSION = "2.5.1"

#: Default memory budget per call (bytes). Larger blocks are not faster.
DEFAULT_MEM_LIMIT = 2e9

#: Bytes per value held during a block (measured): data and products in
#: float64 plus the copies made by the sigma clipping and the masks.
_BYTES_PER_VALUE = 8 * 14


def masked_weighted_mean(
    data: np.ma.MaskedArray,
    weights: np.ndarray,
    *,
    sigma_clip_thresholds: tuple[float, float] | None = (5.0, 5.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted mean along axis 0 over the unmasked values only.

    ``data`` has shape ``(n_images, ...)``; ``weights`` has length
    ``n_images``. With ``sigma_clip_thresholds = (low, high)`` values more
    than ``low`` / ``high`` robust standard deviations (MAD) below / above
    the median are rejected first (one pass, as ``ccdproc.combine`` does).

    Returns ``(mean, uncertainty, mask)``: the uncertainty is the scatter of
    the remaining values divided by the square root of their number (as in
    ccdproc); ``mask`` marks pixels without any remaining value (mean and
    uncertainty NaN there).
    """
    data = np.ma.masked_invalid(data)
    if sigma_clip_thresholds is not None:
        low, high = sigma_clip_thresholds
        clipped = astropy_sigma_clip(data, sigma_lower=low, sigma_upper=high, axis=0, maxiters=1,
                             cenfunc="median", stdfunc="mad_std", masked=True, copy=True)
        data = np.ma.masked_array(data.data, mask=np.ma.getmaskarray(clipped))
    valid = ~np.ma.getmaskarray(data)
    w = np.asarray(weights, dtype=np.float64).reshape((-1,) + (1,) * (data.ndim - 1))
    w_valid = np.where(valid, w, 0.0)
    w_sum = w_valid.sum(axis=0)
    values = np.where(valid, data.data, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (values * w_valid).sum(axis=0) / w_sum
        n = valid.sum(axis=0)
        uncertainty = np.ma.std(data, axis=0).filled(np.nan) / np.sqrt(n)
    empty = (n == 0) | (w_sum <= 0)
    mean[empty] = np.nan
    uncertainty[empty] = np.nan
    return mean, uncertainty, empty


def _block_rows(n_images: int, n_columns: int, mem_limit: float) -> int:
    per_row = n_images * n_columns * _BYTES_PER_VALUE
    return max(1, int(mem_limit // max(per_row, 1)))


def weighted_average_combine(
    images: Sequence[str | Path | CCDData],
    weights: Sequence[float] | np.ndarray,
    *,
    sigma_clip: bool = True,
    sigma_clip_low_thresh: float = 5.0,
    sigma_clip_high_thresh: float = 5.0,
    mem_limit: float = DEFAULT_MEM_LIMIT,
    dtype: str | np.dtype | None = None,
) -> CCDData:
    """Mask-aware weighted average of images (files or CCDData).

    Replacement for ``ccdproc.combine(method="average", weights=...)``
    with its sigma clipping (median / MAD). The images are read in blocks
    of rows to stay within ``mem_limit``. Header and unit come from the
    first image; the result carries a mask (pixels without any valid
    value) and a ``StdDevUncertainty``.
    """
    n = len(images)
    if n == 0:
        raise ValueError("no images to combine")
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.shape != (n,):
        raise ValueError(f"{weights.size} weights for {n} images")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and >= 0")
    out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
    first = images[0] if isinstance(images[0], CCDData) else CCDData.read(images[0])
    shape = first.shape
    meta = first.meta.copy()
    unit = first.unit
    del first
    thresholds = (sigma_clip_low_thresh, sigma_clip_high_thresh) if sigma_clip else None

    mean = np.empty(shape, dtype=out_dtype)
    uncertainty = np.empty(shape, dtype=out_dtype)
    mask = np.zeros(shape, dtype=bool)
    rows = _block_rows(n, int(np.prod(shape[1:])), mem_limit)
    with ExitStack() as stack:
        sources = []
        for image in images:
            if isinstance(image, CCDData):
                sources.append((image.data, image.mask))
                continue
            hdul = stack.enter_context(fits.open(image, memmap=True))
            data_hdu = next(h for h in hdul if h.is_image and h.header.get("NAXIS", 0) >= 2)
            mask_hdu = hdul["MASK"] if "MASK" in hdul else None
            if tuple(data_hdu.shape) != tuple(shape):
                raise ValueError(f"{image}: shape {data_hdu.shape} differs from {shape}")
            sources.append((data_hdu.section, None if mask_hdu is None else mask_hdu.section))
        for start in range(0, shape[0], rows):
            stop = min(start + rows, shape[0])
            block = np.empty((n, stop - start) + tuple(shape[1:]), dtype=np.float64)
            block_mask = np.zeros(block.shape, dtype=bool)
            for i, (data_src, mask_src) in enumerate(sources):
                block[i] = data_src[start:stop]
                if mask_src is not None:
                    block_mask[i] = np.asarray(mask_src[start:stop], dtype=bool)
            m, u, e = masked_weighted_mean(np.ma.masked_array(block, mask=block_mask), weights,
                                           sigma_clip_thresholds=thresholds)
            mean[start:stop], uncertainty[start:stop], mask[start:stop] = m, u, e
    return CCDData(mean, unit=unit, meta=meta, mask=mask,
                   uncertainty=StdDevUncertainty(uncertainty))


__all__ = [
    "CCDPROC_CHECKED_VERSION",
    "masked_weighted_mean",
    "weighted_average_combine",
]
