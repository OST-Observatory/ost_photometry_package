"""Combination of images with masks, clipping and error propagation.

:func:`combine_images` combines frames by (weighted) mean, median or sum and
propagates the frames' own uncertainties (``UNCERT``)::

    mean:   σ² = Σ w_i² σ_i² / (Σ w_i)²
    median: σ² = k(n) · n / (Σ 1/σ_i)²   (k = Var(median)/Var(mean), → π/2)
    sum:    the mean scaled to all n frames

over the values that survive masking and sigma clipping. Only frames
without any uncertainty (raw bias frames: read noise only) fall back to the
scatter between the frames, ``std(ddof=1)/√n``. The earlier stacks used
that scatter for every frame: zero where one frame covered a pixel, biased
low for few frames, inflated on stars by seeing changes between frames.

The weighted mean also works around a ccdproc bug:

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


#: Var(median) / Var(mean) of n Gaussian values (n = 1..30, simulated with
#: 4e5 draws; n > 30: π/2). Even n average the two middle values and are
#: more efficient.
_MEDIAN_VARIANCE_FACTOR = np.array(
    [1.0, 1.0, 1.346, 1.192, 1.434, 1.283, 1.478, 1.347, 1.497, 1.381, 1.502,
     1.414, 1.52, 1.435, 1.525, 1.451, 1.533, 1.463, 1.529, 1.467, 1.54, 1.474,
     1.538, 1.488, 1.547, 1.49, 1.546, 1.499, 1.545, 1.507]
)


def median_variance_factor(n: np.ndarray) -> np.ndarray:
    """Var(median)/Var(mean) for ``n`` Gaussian values (1 for n <= 2)."""
    n = np.asarray(n, dtype=int)
    idx = np.clip(n, 1, _MEDIAN_VARIANCE_FACTOR.size) - 1
    return np.where(n > _MEDIAN_VARIANCE_FACTOR.size, np.pi / 2, _MEDIAN_VARIANCE_FACTOR[idx])


#: Fewest valid values of a pixel that are clipped against their own σ: with
#: two values the median lies halfway, and both or neither would go.
MIN_CLIP_FRAMES = 3


def clip_against_uncertainty(
    data: np.ma.MaskedArray,
    sigma: np.ndarray,
    low: float,
    high: float,
    *,
    psf_noise: np.ndarray | None = None,
    min_frames: int = MIN_CLIP_FRAMES,
) -> np.ndarray:
    """Mask of values to reject: ``x_i - med`` beyond ``low`` / ``high`` ×
    ``sqrt(σ_i² + psf_noise²)`` (ccdclip with the frames' own errors).

    ``psf_noise`` (shape of one frame) is the tolerated difference between
    frames that is not noise: stars look different in frames of different
    seeing. Pixels with fewer than ``min_frames`` valid values are not
    clipped.
    """
    valid = ~np.ma.getmaskarray(data)
    n = valid.sum(axis=0)
    med = np.ma.median(data, axis=0).filled(np.nan)
    var = np.asarray(sigma, dtype=np.float64) ** 2
    if psf_noise is not None:
        var = var + np.asarray(psf_noise, dtype=np.float64) ** 2
    with np.errstate(invalid="ignore"):
        tol = np.sqrt(var)
        dev = data.data - med
        reject = (dev < -low * tol) | (dev > high * tol)
    return valid & (n >= min_frames) & reject


def combine_stack(
    data: np.ma.MaskedArray,
    sigma: np.ndarray | None,
    *,
    method: str = "average",
    weights: np.ndarray | None = None,
    sigma_clip_thresholds: tuple[float, float] | None = (5.0, 5.0),
    clip_against_sigma: bool = False,
    psf_noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine along axis 0; returns ``(value, uncertainty, empty)``.

    ``sigma`` (same shape as ``data``) are the frames' 1σ errors; ``None``
    uses the scatter of the frames (``std(ddof=1)/√n``, NaN for one frame).
    Clipping (one pass, before combining): values more than ``low`` /
    ``high`` MAD-σ below / above the median, or with
    ``clip_against_sigma`` (needs ``sigma``) more than that many of their
    own σ (and ``psf_noise``), see :func:`clip_against_uncertainty`.
    Clipped and masked values are left out of the value and of the error
    alike.
    """
    data = np.ma.masked_invalid(data)
    if sigma is not None:
        data = np.ma.masked_array(
            data.data, mask=np.ma.getmaskarray(data) | ~np.isfinite(sigma)
        )
    if sigma_clip_thresholds is not None:
        low, high = sigma_clip_thresholds
        if clip_against_sigma and sigma is not None:
            reject = clip_against_uncertainty(data, sigma, low, high, psf_noise=psf_noise)
            data = np.ma.masked_array(data.data, mask=np.ma.getmaskarray(data) | reject)
        else:
            clipped = astropy_sigma_clip(data, sigma_lower=low, sigma_upper=high, axis=0, maxiters=1,
                                         cenfunc="median", stdfunc="mad_std", masked=True, copy=True)
            data = np.ma.masked_array(data.data, mask=np.ma.getmaskarray(clipped))
    valid = ~np.ma.getmaskarray(data)
    n_total = data.shape[0]
    n = valid.sum(axis=0)
    w = np.ones(n_total) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w.reshape((-1,) + (1,) * (data.ndim - 1))
    w_valid = np.where(valid, w, 0.0)
    w_sum = w_valid.sum(axis=0)
    values = np.where(valid, data.data, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "median":
            value = np.ma.median(data, axis=0).filled(np.nan)
        else:
            value = (values * w_valid).sum(axis=0) / w_sum
        if sigma is not None:
            var = np.where(valid, np.asarray(sigma, dtype=np.float64) ** 2, 0.0)
            if method == "median":
                # Var(median) = 1/(4 n f(m)²) with the density of the mixture
                # at the median, f = Σ φ(0)/σ_i / n: k(n)·n / (Σ 1/σ_i)²,
                # which is k(n)·σ²/n for equal errors.
                inv_sigma = np.where(valid, 1.0 / np.sqrt(var), 0.0)
                err2 = median_variance_factor(n) * n / inv_sigma.sum(axis=0) ** 2
            else:
                err2 = (w_valid**2 * var).sum(axis=0) / w_sum**2
            uncertainty = np.sqrt(err2)
        else:
            scatter = np.ma.std(data, axis=0, ddof=1).filled(np.nan)
            uncertainty = scatter / np.sqrt(n)
            if method == "median":
                uncertainty = uncertainty * np.sqrt(median_variance_factor(n))
        if method == "sum":
            value = value * n_total
            uncertainty = uncertainty * n_total
    empty = (n == 0) | (w_sum <= 0)
    value[empty] = np.nan
    uncertainty[empty] = np.nan
    return value, uncertainty, empty


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

    Returns ``(mean, uncertainty, mask)`` from the scatter of the remaining
    values (no input errors); see :func:`combine_stack` for propagated errors.
    """
    return combine_stack(
        data, None, method="average", weights=weights,
        sigma_clip_thresholds=sigma_clip_thresholds,
    )


def _block_rows(n_images: int, n_columns: int, mem_limit: float) -> int:
    per_row = n_images * n_columns * _BYTES_PER_VALUE
    return max(1, int(mem_limit // max(per_row, 1)))


def _hdu_with_data(hdul):
    return next(h for h in hdul if h.is_image and h.header.get("NAXIS", 0) >= 2)


class _ScaledSection:
    """Row blocks of a memory-mapped HDU with ``BSCALE`` / ``BZERO`` applied.

    Integer raw frames (``BZERO`` 32768) cannot be memory-mapped with
    astropy's automatic scaling, so the file is opened unscaled and the
    scaling is applied per block.
    """

    def __init__(self, hdu):
        self._section = hdu.section
        self._scale = float(hdu.header.get("BSCALE", 1.0))
        self._zero = float(hdu.header.get("BZERO", 0.0))

    def __getitem__(self, item):
        raw = np.asarray(self._section[item], dtype=np.float64)
        return raw * self._scale + self._zero


def combine_images(
    images: Sequence[str | Path | CCDData],
    *,
    method: str = "average",
    weights: Sequence[float] | np.ndarray | None = None,
    scale: Sequence[float] | np.ndarray | None = None,
    sigma_clip: bool = True,
    sigma_clip_low_thresh: float = 5.0,
    sigma_clip_high_thresh: float = 5.0,
    mem_limit: float = DEFAULT_MEM_LIMIT,
    dtype: str | np.dtype | None = None,
    unit=None,
    match_background: bool = False,
    clip_snoise: float | None = None,
    clip_snoise_radius: int = 0,
) -> CCDData:
    """Combine images (files or CCDData) with propagated uncertainties.

    Clipping: by default against the MAD of the frames at each pixel. With
    ``clip_snoise`` (needs uncertainties) a value is clipped when it is more
    than ``low`` / ``high`` × ``sqrt(σ_i² + (clip_snoise·P)²)`` from the
    median, and only where at least :data:`MIN_CLIP_FRAMES` values remain.
    ``P`` is the local peak above the sky of the mean frame within
    ``clip_snoise_radius`` pixels: a star in a frame of different seeing
    differs from the median by a fraction of its peak, in the core and in
    the wings alike, and is kept. Clipping against the MAD of a few frames
    (or relative to the pixel value) cuts star cores and wings of the frames
    with the best / worst seeing, which loses flux of bright stars.

    ``match_background=True`` shifts every frame to the common (weighted
    mean) sky level before clipping and combining: frames from different
    nights differ in sky brightness, and the per-pixel clipping would keep
    the brighter frames in some pixels and reject them in others, which
    turns the sky difference into pixel-to-pixel noise. The shift is a
    constant per frame (clipped median of the unmasked pixels); the
    uncertainty is unchanged.

    ``method`` is ``average``, ``median`` or ``sum``; ``weights`` (not for
    the median) and ``scale`` (multiplies data and σ of each frame, e.g. the
    inverse median of flats) have one value per image. The uncertainty comes
    from the frames' ``UNCERT`` / ``CCDData.uncertainty``; frames without
    any fall back to the scatter between frames (see :func:`combine_stack`).
    Images are read in blocks of rows within ``mem_limit``. Header and unit
    come from the first image (``unit`` overrides). The result carries a
    mask (pixels without any valid value) and a ``StdDevUncertainty``.
    """
    if method not in ("average", "median", "sum"):
        raise ValueError(f"unknown combine method {method!r}")
    n = len(images)
    if n == 0:
        raise ValueError("no images to combine")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.shape != (n,):
            raise ValueError(f"{weights.size} weights for {n} images")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and >= 0")
        if method == "median":
            raise ValueError("weights are not defined for method='median'")
    factors = np.ones(n) if scale is None else np.asarray(scale, dtype=np.float64).ravel()
    if factors.shape != (n,):
        raise ValueError(f"{factors.size} scale factors for {n} images")
    out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
    first = images[0] if isinstance(images[0], CCDData) else CCDData.read(images[0], unit=unit)
    shape = first.shape
    meta = first.meta.copy()
    out_unit = unit if unit is not None else first.unit
    del first
    thresholds = (sigma_clip_low_thresh, sigma_clip_high_thresh) if sigma_clip else None
    clip_sigma = sigma_clip and clip_snoise is not None
    offsets = np.zeros(n)
    peak = None
    if match_background or clip_sigma:
        offsets, sky_ref, peak = _frame_pass(
            images, factors, weights, unit, shape,
            match_background=match_background,
            peak_radius=int(clip_snoise_radius) if clip_sigma else None,
        )
        if not match_background:
            offsets = np.zeros(n)

    value = np.empty(shape, dtype=out_dtype)
    uncertainty = np.empty(shape, dtype=out_dtype)
    mask = np.zeros(shape, dtype=bool)
    rows = _block_rows(n, int(np.prod(shape[1:])), mem_limit)
    with ExitStack() as stack:
        sources = []
        for image in images:
            if isinstance(image, CCDData):
                unc = None if image.uncertainty is None else image.uncertainty.array
                sources.append((image.data, image.mask, unc))
                continue
            hdul = stack.enter_context(
                fits.open(image, memmap=True, do_not_scale_image_data=True)
            )
            data_hdu = _hdu_with_data(hdul)
            if tuple(data_hdu.shape) != tuple(shape):
                raise ValueError(f"{image}: shape {data_hdu.shape} differs from {shape}")
            mask_hdu = hdul["MASK"] if "MASK" in hdul else None
            unc_hdu = hdul["UNCERT"] if "UNCERT" in hdul else None
            sources.append((
                _ScaledSection(data_hdu),
                None if mask_hdu is None else _ScaledSection(mask_hdu),
                None if unc_hdu is None else _ScaledSection(unc_hdu),
            ))
        have_sigma = [src[2] is not None for src in sources]
        if any(have_sigma) and not all(have_sigma):
            raise ValueError(
                "combine_images: some frames have an uncertainty and some do not; "
                "they cannot be combined consistently"
            )
        use_sigma = all(have_sigma)
        if clip_sigma and not use_sigma:
            raise ValueError("clip_snoise needs frames with an uncertainty (UNCERT)")
        for start in range(0, shape[0], rows):
            stop = min(start + rows, shape[0])
            block = np.empty((n, stop - start) + tuple(shape[1:]), dtype=np.float64)
            block_mask = np.zeros(block.shape, dtype=bool)
            block_sigma = np.empty(block.shape, dtype=np.float64) if use_sigma else None
            for i, (data_src, mask_src, unc_src) in enumerate(sources):
                block[i] = np.asarray(data_src[start:stop], dtype=np.float64) * factors[i] + offsets[i]
                if mask_src is not None:
                    block_mask[i] = np.asarray(mask_src[start:stop], dtype=bool)
                if use_sigma:
                    block_sigma[i] = np.abs(np.asarray(unc_src[start:stop], dtype=np.float64) * factors[i])
            v, u, e = combine_stack(
                np.ma.masked_array(block, mask=block_mask),
                block_sigma,
                method=method,
                weights=weights,
                sigma_clip_thresholds=thresholds,
                clip_against_sigma=clip_sigma,
                psf_noise=None if peak is None else float(clip_snoise) * peak[start:stop],
            )
            value[start:stop], uncertainty[start:stop], mask[start:stop] = v, u, e
    return CCDData(value, unit=out_unit, meta=meta, mask=mask,
                   uncertainty=StdDevUncertainty(uncertainty))


def _frame_pass(
    images, factors, weights, unit, shape, *, match_background: bool, peak_radius: int | None
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """One read of every frame: sky levels and, optionally, the peak map.

    Returns ``(offsets, sky_ref, peak)``: the shift of every frame to the
    weighted mean sky level ``sky_ref`` (see ``match_background``) and,
    with ``peak_radius``, the maximum of the weighted mean frame minus
    ``sky_ref`` within that radius (``float32``, ≥ 0).
    """
    from astropy.stats import sigma_clipped_stats
    from scipy.ndimage import maximum_filter

    n = len(images)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    sky = np.empty(n)
    total = wsum = None
    if peak_radius is not None:
        total = np.zeros(shape, dtype=np.float32)
        wsum = np.zeros(shape, dtype=np.float32)
    for i, image in enumerate(images):
        ccd = image if isinstance(image, CCDData) else CCDData.read(image, unit=unit)
        data = np.asarray(ccd.data, dtype=np.float64) * factors[i]
        good = np.isfinite(data)
        if ccd.mask is not None:
            good &= ~np.asarray(ccd.mask, dtype=bool)
        sub = good[::4, ::4]
        sky[i] = sigma_clipped_stats(data[::4, ::4][sub], sigma=3.0, maxiters=5)[1] if np.any(sub) else 0.0
        if total is not None:
            wi = w[i] if w[i] > 0 else 0.0
            total += np.where(good, data - sky[i], 0.0) * wi
            wsum += good * wi
        del ccd, data, good
    reference = float(np.sum(w * sky) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(sky))
    offsets = reference - sky if match_background else np.zeros(n)
    peak = None
    if total is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(wsum > 0, total / wsum, 0.0)
        del total, wsum
        size = 2 * max(int(peak_radius), 0) + 1
        peak = maximum_filter(np.clip(mean, 0.0, None).astype(np.float32), size=size)
    return offsets, reference, peak


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
    """Mask-aware weighted average with propagated errors (:func:`combine_images`)."""
    return combine_images(
        images,
        method="average",
        weights=weights,
        sigma_clip=sigma_clip,
        sigma_clip_low_thresh=sigma_clip_low_thresh,
        sigma_clip_high_thresh=sigma_clip_high_thresh,
        mem_limit=mem_limit,
        dtype=dtype,
    )


__all__ = [
    "CCDPROC_CHECKED_VERSION",
    "MIN_CLIP_FRAMES",
    "clip_against_uncertainty",
    "combine_images",
    "combine_stack",
    "masked_weighted_mean",
    "median_variance_factor",
    "weighted_average_combine",
]
