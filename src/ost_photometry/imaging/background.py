"""Sky level, background neutralisation and gradient removal."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats

#: Pixels used for the sky statistics (subsampled for speed).
_MAX_SAMPLES = 2_000_000


def _sample(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    good = np.isfinite(values)
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool)
    values = values[good]
    step = max(1, values.size // _MAX_SAMPLES)
    return values[::step]


def sky_level(data: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Sky median and robust standard deviation (sigma-clipped)."""
    values = _sample(data, mask)
    if values.size == 0:
        return 0.0, 1.0
    _mean, median, std = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    return float(median), float(std) if std > 0 else 1.0


def neutralize_background(
    channels: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Subtract each channel's sky level, so the sky is grey (zero) in all.

    Returns the channels and the subtracted levels.
    """
    out, levels = {}, {}
    for name, data in channels.items():
        level, _std = sky_level(data)
        out[name] = np.asarray(data, dtype=np.float32) - np.float32(level)
        levels[name] = level
    return out, levels


def source_mask(data: np.ndarray, *, nsigma: float = 1.5, npixels: int = 5,
                block: int = 4, dilate: int = 3) -> np.ndarray:
    """Pixels of stars and extended objects (to keep them out of the sky model).

    Detection runs on a ``block`` x ``block`` binned image, so faint halos of
    galaxies and nebulae (below the noise of single pixels) are found too.
    """
    from photutils.segmentation import detect_sources
    from scipy.ndimage import binary_dilation

    values = np.asarray(data, dtype=float)
    level, _std = sky_level(values)
    values = np.nan_to_num(values, nan=level) - level
    h, w = values.shape
    hb, wb = h // block, w // block
    binned = values[: hb * block, : wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3))
    _level, std = sky_level(binned)
    segments = detect_sources(binned, nsigma * std, npixels=npixels)
    small = np.zeros(binned.shape, dtype=bool) if segments is None else segments.data > 0
    if dilate > 0 and small.any():
        small = binary_dilation(small, iterations=int(dilate))
    if segments is not None:
        # Extended objects (galaxies, nebulae, bright stars) have halos far
        # below the detection limit: grow them by half their size.
        from scipy.ndimage import distance_transform_edt

        areas = np.asarray(segments.areas, dtype=float)
        for label, area in zip(segments.labels, areas, strict=True):
            if area < 100:
                continue
            grow = int(np.ceil(0.5 * np.sqrt(area)))
            ys, xs = np.nonzero(segments.data == label)
            r0, r1 = max(ys.min() - grow, 0), min(ys.max() + grow + 1, binned.shape[0])
            c0, c1 = max(xs.min() - grow, 0), min(xs.max() + grow + 1, binned.shape[1])
            region = segments.data[r0:r1, c0:c1] != label
            small[r0:r1, c0:c1] |= distance_transform_edt(region) <= grow
    mask = np.zeros((h, w), dtype=bool)
    mask[: hb * block, : wb * block] = np.repeat(np.repeat(small, block, axis=0), block, axis=1)
    return mask


def cosmetic_correction(data: np.ndarray, *, nsigma: float = 5.0) -> tuple[np.ndarray, int]:
    """Replace cold pixels (far below their 3x3 median and the sky) by that median.

    Dead or cold pixels at the same detector position in every frame survive
    an undithered stack and show as black dots once stretched. Bright
    outliers are left alone (undersampled star cores look alike).
    """
    from scipy.ndimage import median_filter

    values = np.asarray(data, dtype=np.float32)
    level, std = sky_level(values)
    local = median_filter(values, size=3, mode="nearest")
    # far below the neighbours AND below the sky: star flanks are above it
    cold = ((local - values) > nsigma * std) & (values < level - 3.0 * std)
    if cold.any():
        values = np.where(cold, local, values)
    return values, int(np.count_nonzero(cold))


#: Values of ``method`` in :func:`remove_gradient`.
GRADIENT_METHODS = ("polynomial", "background2d")


def _tile_samples(values: np.ndarray, mask: np.ndarray, tiles: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma-clipped sky medians of a ``tiles``-wide grid (well covered tiles only)."""
    h, w = values.shape
    ny = max(2, int(round(tiles * h / max(h, w))))
    nx = max(2, int(round(tiles * w / max(h, w))))
    ys, xs, levels = [], [], []
    for i in range(ny):
        for j in range(nx):
            r0, r1 = i * h // ny, (i + 1) * h // ny
            c0, c1 = j * w // nx, (j + 1) * w // nx
            tile, tile_mask = values[r0:r1, c0:c1], mask[r0:r1, c0:c1]
            good = np.isfinite(tile) & ~tile_mask
            if good.mean() < 0.5:
                continue
            _m, median, _s = sigma_clipped_stats(tile[good], sigma=2.5, maxiters=5)
            ys.append(0.5 * (r0 + r1))
            xs.append(0.5 * (c0 + c1))
            levels.append(median)
    return np.asarray(ys), np.asarray(xs), np.asarray(levels)


def _poly_terms(y: np.ndarray, x: np.ndarray, degree: int) -> np.ndarray:
    return np.stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)],
                    axis=-1)


def remove_gradient(
    data: np.ndarray,
    *,
    method: str = "polynomial",
    degree: int = 2,
    tiles: int = 16,
    box: int | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a smooth sky model, keeping the mean level.

    ``polynomial`` (default): a 2-D polynomial of ``degree`` fitted to the
    sigma-clipped sky of a ``tiles`` grid, with tiles that deviate from the
    fit rejected iteratively. Smooth gradients (moon, light pollution) are
    removed; galaxies and nebulae are too local to bend it.
    ``background2d``: photutils ``Background2D`` with ``box`` (default an
    eighth of the shorter side); follows smaller structures, but extended
    objects partly end up in the model (dark halos).
    Stars and objects are masked first (:func:`source_mask`). Returns the
    corrected image and the subtracted model (minus its median).
    """
    if method not in GRADIENT_METHODS:
        raise ValueError(f"method must be one of {GRADIENT_METHODS}, got {method!r}")
    values = np.asarray(data, dtype=float)
    if mask is None:
        mask = source_mask(values)
    if method == "background2d":
        from photutils.background import Background2D, MedianBackground

        box = int(box) if box else max(32, min(values.shape) // 8)
        background = Background2D(
            np.nan_to_num(values, nan=float(np.nanmedian(values))), box,
            mask=mask, filter_size=3, sigma_clip=SigmaClip(sigma=3.0),
            bkg_estimator=MedianBackground(), exclude_percentile=50.0,
        )
        model = background.background - float(np.median(background.background))
        return (values - model).astype(np.float32), model.astype(np.float32)

    h, w = values.shape
    ys, xs, levels = _tile_samples(values, np.asarray(mask, dtype=bool), tiles)
    n_terms = (degree + 1) * (degree + 2) // 2
    if levels.size < 2 * n_terms:
        return values.astype(np.float32), np.zeros_like(values, dtype=np.float32)
    scale = float(max(h, w))
    use = np.ones(levels.size, dtype=bool)
    for _ in range(5):
        terms = _poly_terms(ys[use] / scale, xs[use] / scale, degree)
        coeffs, *_ = np.linalg.lstsq(terms, levels[use], rcond=None)
        residual = levels - _poly_terms(ys / scale, xs / scale, degree) @ coeffs
        spread = 1.4826 * np.median(np.abs(residual[use] - np.median(residual[use])))
        new_use = np.abs(residual) <= 3.0 * max(spread, 1e-12)
        if new_use.sum() < 2 * n_terms or np.array_equal(new_use, use):
            break
        use = new_use
    yy, xx = np.mgrid[0:h, 0:w]
    model = (_poly_terms(yy / scale, xx / scale, degree) @ coeffs).astype(np.float32)
    model -= np.float32(np.median(model))
    return (values - model).astype(np.float32), model


__all__ = ["GRADIENT_METHODS", "cosmetic_correction", "neutralize_background",
           "remove_gradient", "sky_level", "source_mask"]
