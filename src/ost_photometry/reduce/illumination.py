"""Illumination correction from the science frames (sky flat).

Flats whose light source is not uniform (daylight in the dome, a panel
that is not even) leave their own large-scale pattern in every frame they
calibrate: rings and blotches of a few per cent of the sky, the same in all
filters. Normalised to their sky and with stars and objects masked, the
reduced lights of one flat master show that pattern; their median, filled
under the objects and smoothed, is a multiplicative correction map.

The map is built per group of lights calibrated with the same flat master
(and filter, camera, binning) and divides those lights and their
uncertainty before registration. It acts like a flat: star fluxes change by
the same factor as the sky, so the photometry improves when the pattern is a
flat error. Additive sky structure (moonlight, scattered light) does not
belong in it; the median over many frames, ideally of several nights and
pointings, suppresses it. Extended emission that sits at the same detector
position in every frame and is too faint to be masked ends up in the map
too: without dithering or other fields, faint nebulosity around the target
is flattened as well.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clipped_stats

#: Header keywords of a corrected frame.
ILLUMINATION_KEYWORD = "ILLUMCOR"
ILLUMINATION_FRAMES_KEYWORD = "ILLUMN"


@dataclass
class IlluminationMap:
    """Correction map of one group of frames (median 1, full resolution)."""

    key: Hashable
    data: np.ndarray
    n_frames: int
    masked_fraction: float
    #: 2nd and 98th percentile of the map minus 1
    spread: tuple[float, float]


def _binned_median(values: np.ndarray, factor: int) -> np.ndarray:
    h, w = values.shape
    hb, wb = h // factor, w // factor
    blocks = values[: hb * factor, : wb * factor].reshape(hb, factor, wb, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
        return np.nanmedian(blocks, axis=(1, 3))


def _normalised_binned(path: str | Path, bin_factor: int) -> np.ndarray | None:
    """Frame / its sigma-clipped sky, median-binned; masked pixels NaN."""
    ccd = CCDData.read(path, unit="adu") if _no_unit(path) else CCDData.read(path)
    data = np.asarray(ccd.data, dtype=float)
    if ccd.mask is not None:
        data = np.where(np.asarray(ccd.mask, dtype=bool), np.nan, data)
    finite = np.isfinite(data)
    if finite.mean() < 0.5:
        return None
    _mean, sky, _std = sigma_clipped_stats(data[::4, ::4], sigma=3.0, maxiters=5)
    if not (np.isfinite(sky) and sky > 0):
        return None
    return _binned_median(data / sky, bin_factor)


def _no_unit(path: str | Path) -> bool:
    return "BUNIT" not in fits.getheader(path)


#: An extended object must stand out from its surroundings (scale
#: ``OBJECT_SCALE`` of the frame) by at least this fraction of the sky.
#: Flat errors vary over a large part of the frame, so they barely show at
#: that scale; nebulae and galaxies have edges.
OBJECT_MIN_EXCESS = 0.005
OBJECT_SCALE = 1.0 / 16.0


def _grow_components(region: np.ndarray, fraction: float = 0.2, minimum: int = 4) -> np.ndarray:
    """Grow every connected part by ``fraction`` of its size (faint halos
    below the detection threshold)."""
    from scipy.ndimage import distance_transform_edt, label

    labels, n = label(region)
    if n == 0:
        return region
    grown = region.copy()
    sizes = np.bincount(labels.ravel())
    for k in range(1, n + 1):
        grow = max(minimum, int(np.ceil(fraction * np.sqrt(sizes[k]))))
        ys, xs = np.nonzero(labels == k)
        r0, r1 = max(ys.min() - grow, 0), min(ys.max() + grow + 1, region.shape[0])
        c0, c1 = max(xs.min() - grow, 0), min(xs.max() + grow + 1, region.shape[1])
        outside = labels[r0:r1, c0:c1] != k
        grown[r0:r1, c0:c1] |= distance_transform_edt(outside) <= grow
    return grown


def _broad_background(values: np.ndarray, excluded: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian average of ``values`` over the pixels not ``excluded``."""
    from scipy.ndimage import gaussian_filter

    weight = (~excluded & np.isfinite(values)).astype(float)
    total = gaussian_filter(np.where(weight > 0, values, 0.0), sigma, mode="nearest")
    norm = gaussian_filter(weight, sigma, mode="nearest")
    return total / np.maximum(norm, 1e-6)


def _object_mask(deep: np.ndarray, min_excess: float = OBJECT_MIN_EXCESS) -> np.ndarray:
    """Stars and objects of the deep (median, sky = 1) image.

    Compact sources: detected on the image minus a smoothed version.
    Extended objects: the smoothed image more than ``min_excess`` of the sky
    (and 5 sigma) above its surroundings, a Gaussian average over
    ``OBJECT_SCALE`` of the frame that leaves out what was found before
    (two passes); holes filled, each object grown by a fifth of its size. The
    illumination pattern changes over a large part of the frame and stays
    unmasked, also where it is brighter than a faint object.
    """
    from photutils.segmentation import detect_sources
    from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter

    finite = np.isfinite(deep)
    _mean, level, std = sigma_clipped_stats(deep[finite], sigma=3.0)
    filled = np.where(finite, deep, level)
    compact = np.zeros(deep.shape, dtype=bool)
    high_pass = filled - gaussian_filter(filled, 8)
    segments = detect_sources(high_pass, 2.5 * std, n_pixels=3)
    if segments is not None:
        compact = binary_dilation(segments.data > 0, iterations=2)
    # 4-pixel Gaussian (16 px of the frame): noise / 8, faint shells found
    smooth = gaussian_filter(filled, 4)
    threshold = max(5.0 * std / 8.0, float(min_excess) * level)
    sigma = max(deep.shape) * OBJECT_SCALE
    extended = np.zeros(deep.shape, dtype=bool)
    for _ in range(2):
        surroundings = _broad_background(smooth, compact | extended | ~finite, sigma)
        extended = (smooth - surroundings) > threshold
    if extended.any():
        from scipy.ndimage import binary_opening

        extended = binary_opening(extended, iterations=2)  # noise specks
        extended = _grow_components(binary_fill_holes(extended))
    return compact | extended | ~finite


def _fill_and_smooth(grid: np.ndarray, smooth: float) -> np.ndarray:
    """Holes filled from their surroundings (growing Gaussian), then smoothed."""
    from scipy.ndimage import gaussian_filter

    filled = grid.copy()
    for sigma in (2, 4, 8, 16, 32, 64):
        if np.all(np.isfinite(filled)):
            break
        weight = gaussian_filter(np.isfinite(filled).astype(float), sigma)
        estimate = gaussian_filter(np.nan_to_num(filled), sigma) / np.maximum(weight, 1e-12)
        filled = np.where(np.isfinite(filled), filled, np.where(weight > 0.05, estimate, np.nan))
    if not np.all(np.isfinite(filled)):
        filled = np.where(np.isfinite(filled), filled, np.nanmedian(filled))
    return gaussian_filter(filled, smooth) if smooth > 0 else filled


def _to_full_resolution(grid: np.ndarray, shape: tuple[int, int], cell: int) -> np.ndarray:
    """Bilinear interpolation of a grid of ``cell`` x ``cell`` pixel cells."""
    from scipy.ndimage import map_coordinates

    h, w = shape
    gy = (np.arange(h) + 0.5) / cell - 0.5
    gx = (np.arange(w) + 0.5) / cell - 0.5
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    return map_coordinates(grid, [yy, xx], order=1, mode="nearest").astype(np.float32)


def illumination_map(
    paths: Sequence[str | Path],
    *,
    key: Hashable = "",
    bin_factor: int = 4,
    cell_bins: int = 4,
    smooth_cells: float = 1.5,
    min_frames: int = 10,
) -> IlluminationMap | None:
    """Correction map from reduced, not yet registered frames of one group.

    Frames are divided by their sky and median-binned by ``bin_factor``;
    their median is the deep image on which stars and objects are masked.
    The masked per-frame median is reduced to cells of ``bin_factor`` x
    ``cell_bins`` pixels, filled under the masks and smoothed by
    ``smooth_cells`` cells (16 px cells, 24 px Gaussian by default: rings,
    vignetting and dust shadows, not stars). Under masked objects the map is
    interpolated from their surroundings. ``None`` with fewer than
    ``min_frames`` usable frames.
    """
    frames = [f for f in (_normalised_binned(p, bin_factor) for p in paths) if f is not None]
    if len(frames) < max(int(min_frames), 1):
        return None
    shape = fits.getdata(paths[0]).shape
    cube = np.array(frames, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        deep = np.nanmedian(cube, axis=0)
        mask = _object_mask(deep)
        level = np.nanmedian(np.where(mask[None], np.nan, cube), axis=0)
    grid = _binned_median(level, cell_bins)
    smooth = _fill_and_smooth(grid, smooth_cells)
    smooth = smooth / float(np.median(smooth))
    full = _to_full_resolution(smooth, shape, bin_factor * cell_bins)
    full /= np.float32(np.median(full))
    p2, p98 = np.percentile(full, [2, 98])
    return IlluminationMap(key, full, len(frames), float(mask.mean()), (float(p2 - 1), float(p98 - 1)))


def apply_illumination(path: str | Path, illumination: IlluminationMap, label: str) -> None:
    """Divide a reduced frame and its uncertainty by the map (in place)."""
    from .storage import cast_like, file_float_dtype

    dtype = file_float_dtype(path)
    ccd = CCDData.read(path, unit="adu") if _no_unit(path) else CCDData.read(path)
    factor = illumination.data.astype(float)
    ccd.data = np.asarray(ccd.data, dtype=float) / factor
    if ccd.uncertainty is not None:
        ccd.uncertainty = StdDevUncertainty(np.asarray(ccd.uncertainty.array, dtype=float) / factor)
    ccd.meta[ILLUMINATION_KEYWORD] = (label[:68], "Illumination (sky flat) correction map")
    ccd.meta[ILLUMINATION_FRAMES_KEYWORD] = (int(illumination.n_frames), "Frames in that map")
    if dtype is not None:
        cast_like(ccd, dtype)
    ccd.write(path, overwrite=True)


def write_map(illumination: IlluminationMap, path: str | Path) -> Path:
    """The map as FITS (float32) plus a PNG preview next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["ILLUMKEY"] = (str(illumination.key)[:68], "Group (flat master, filter, ...)")
    header["NFRAMES"] = (illumination.n_frames, "Frames combined")
    header["MASKFRAC"] = (round(illumination.masked_fraction, 4), "Masked fraction (objects)")
    fits.writeto(path, illumination.data.astype(np.float32), header, overwrite=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rel = illumination.data[::8, ::8] - 1.0
        lim = max(float(np.percentile(np.abs(rel), 99)), 1e-4)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        im = ax.imshow(rel * 100, origin="lower", cmap="RdBu_r", vmin=-100 * lim, vmax=100 * lim)
        fig.colorbar(im, ax=ax, label="correction [%]")
        ax.set_title(f"{illumination.key}\n{illumination.n_frames} frames", fontsize=9)
        fig.savefig(path.with_suffix(".png"), dpi=80, bbox_inches="tight")
        plt.close(fig)
    except Exception:  # preview only
        pass
    return path


def correct_groups(
    paths: Sequence[str | Path],
    group_of: Callable[[fits.Header], Hashable],
    map_dir: str | Path,
    *,
    min_frames: int = 10,
    log: Callable[[str], None] | None = None,
) -> dict[Hashable, IlluminationMap | None]:
    """Build one map per group (``group_of(header)``) and divide its frames.

    Groups with fewer than ``min_frames`` frames stay uncorrected. Maps are
    written to ``map_dir`` (``ILLUM_<group>.fits`` and ``.png``).
    """
    say = log or (lambda _text: None)
    from ..archive.cache import safe_name

    groups: dict[Hashable, list[Path]] = {}
    for p in paths:
        header = fits.getheader(p)
        if ILLUMINATION_KEYWORD in header:
            say(f"{Path(p).name}: already illumination-corrected, left out")
            continue
        groups.setdefault(group_of(header), []).append(Path(p))
    result: dict[Hashable, IlluminationMap | None] = {}
    for key, members in groups.items():
        label = safe_name("ILLUM_" + "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,))))
        illumination = illumination_map(members, key=key, min_frames=min_frames)
        result[key] = illumination
        if illumination is None:
            say(f"Illumination correction {key}: {len(members)} frame(s), fewer than "
                f"{min_frames} usable; not corrected")
            continue
        write_map(illumination, Path(map_dir) / f"{label}.fits")
        for p in members:
            apply_illumination(p, illumination, label)
        lo, hi = illumination.spread
        say(f"Illumination correction {key}: {illumination.n_frames} frames, map "
            f"{100 * lo:+.1f} … {100 * hi:+.1f} % (p2-p98), objects masked "
            f"{100 * illumination.masked_fraction:.0f} %")
    return result


__all__ = [
    "ILLUMINATION_FRAMES_KEYWORD",
    "ILLUMINATION_KEYWORD",
    "IlluminationMap",
    "apply_illumination",
    "correct_groups",
    "illumination_map",
    "write_map",
]
