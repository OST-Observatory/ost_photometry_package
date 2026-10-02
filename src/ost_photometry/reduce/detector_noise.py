"""Gain and read noise for the reduction: catalog values or measured ones.

Two sources, chosen with ``camera_noise_source``:

``catalog``
    Gain and read noise from the header / camera catalog. Catalog read
    noise is quoted per native pixel; for cameras that bin digitally after
    the readout (CMOS: QHY, ZWO) a binned pixel is the sum of
    ``xbin * ybin`` readouts, so its read noise is ``sqrt(xbin * ybin)``
    times larger. CCDs that bin the charge on the chip read each binned
    pixel once (:func:`binned_read_noise`).

``measured``
    Read noise from the scatter of bias-frame pairs (darks of the shortest
    exposure if no bias exists) and gain from the photon transfer of
    flat-field pairs (:func:`measure_detector_noise`). The values hold for
    the binning, readout mode and gain setting of the frames themselves.

:func:`add_signal_uncertainty` builds the per-pixel uncertainty from the
bias-free signal (Poisson + read noise), so the bias pedestal is not
counted as photon noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clip

from ..camera_specs import binning_mode as catalog_binning_mode

#: Values of ``camera_noise_source``.
NOISE_SOURCES = ("catalog", "measured")

#: Values of ``binning_mode``; ``auto`` asks the camera catalog.
BINNING_MODES = ("auto", "digital", "charge")

#: Side length of the central crop used for the noise statistics.
DEFAULT_CROP_SIZE = 1024

#: Maximum number of frame pairs evaluated per statistic.
DEFAULT_MAX_PAIRS = 8

#: Flats are used for the gain only between these fractions of saturation.
FLAT_LEVEL_RANGE = (0.05, 0.7)

#: Plausible gain range (e-/ADU); measurements outside are discarded.
GAIN_RANGE = (0.05, 50.0)

#: Relative level difference up to which two flats form a pair.
FLAT_PAIR_LEVEL_TOLERANCE = 0.1


def _check_choice(name: str, value: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}, got {value!r}")
    return value


def check_noise_source(value: str) -> str:
    return _check_choice("camera_noise_source", value, NOISE_SOURCES)


def check_binning_mode(value: str) -> str:
    return _check_choice("binning_mode", value, BINNING_MODES)


def resolve_binning_mode(camera: str, binning_mode: str = "auto") -> str | None:
    """``digital``, ``charge`` or ``None`` (unknown camera with ``auto``)."""
    check_binning_mode(binning_mode)
    if binning_mode != "auto":
        return binning_mode
    return catalog_binning_mode(camera)


def binned_read_noise(
    read_noise: float,
    camera: str,
    xbin: int,
    ybin: int,
    binning_mode: str = "auto",
) -> tuple[float, str | None]:
    """Read noise per binned pixel from the read noise per native pixel.

    Returns the scaled value and the binning mode that was applied. Unknown
    cameras (mode ``None``) are not scaled.
    """
    mode = resolve_binning_mode(camera, binning_mode)
    n = max(int(xbin), 1) * max(int(ybin), 1)
    if mode == "digital" and n > 1:
        return float(read_noise) * math.sqrt(n), mode
    return float(read_noise), mode


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseMeasurement:
    """Read noise and gain measured on frames of one electronic setup."""

    read_noise_adu: float
    gain: float | None
    n_zero_pairs: int
    n_flat_pairs: int
    zero_source: str

    def read_noise(self, gain: float | None = None) -> float | None:
        """Read noise in electrons, with the measured or the given gain."""
        g = gain if gain is not None else self.gain
        return None if g is None else self.read_noise_adu * float(g)

    def describe(self) -> str:
        gain = "n/a" if self.gain is None else f"{self.gain:.3f} e-/ADU"
        return (f"read noise {self.read_noise_adu:.2f} ADU from {self.n_zero_pairs} "
                f"{self.zero_source} pair(s), gain {gain} from {self.n_flat_pairs} flat pair(s)")


def robust_std(values: np.ndarray, sigma: float = 4.0) -> float:
    """Standard deviation after iterative clipping of outliers.

    Not the MAD: on integer ADU data the MAD moves in whole-ADU steps and
    biases a noise of ~10 ADU by up to 5 %.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    clipped = sigma_clip(values, sigma=sigma, maxiters=10, cenfunc="median", stdfunc="std")
    return float(np.ma.std(clipped))


def central_crop(path: str | Path, size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """Central ``size x size`` region of the primary image as float."""
    with fits.open(path) as hdul:
        hdu = next((h for h in hdul if h.is_image and h.header.get("NAXIS") == 2), None)
        if hdu is None:
            raise ValueError(f"{path}: no 2-D image")
        ny, nx = hdu.shape
        hy, hx = min(size, ny) // 2, min(size, nx) // 2
        cy, cx = ny // 2, nx // 2
        return np.array(hdu.section[cy - hy: cy + hy, cx - hx: cx + hx], dtype=float)


def _header_values(path: str | Path) -> tuple[float, str, str]:
    header = fits.getheader(path)
    exptime = header.get("EXPTIME", header.get("EXPOSURE", float("nan")))
    try:
        exptime = float(exptime)
    except (TypeError, ValueError):
        exptime = float("nan")
    return exptime, str(header.get("FILTER", "")).strip(), str(header.get("DATE-OBS", ""))


def pair_noise_adu(
    paths: Sequence[str | Path],
    *,
    crop_size: int = DEFAULT_CROP_SIZE,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> tuple[float, int, float]:
    """Noise per frame from disjoint frame pairs: ``std(a - b) / sqrt(2)``.

    Returns ``(median noise in ADU, number of pairs, median level in ADU)``.
    Fixed patterns (bias structure, hot pixels, offsets) cancel in the
    difference; the robust scatter ignores the remaining outliers.
    """
    paths = list(paths)
    noises: list[float] = []
    levels: list[float] = []
    for a_path, b_path in zip(paths[0::2], paths[1::2], strict=False):
        if len(noises) >= max_pairs:
            break
        a, b = central_crop(a_path, crop_size), central_crop(b_path, crop_size)
        if a.shape != b.shape:
            continue
        noises.append(robust_std(a - b) / math.sqrt(2.0))
        levels.extend((float(np.median(a)), float(np.median(b))))
    if not noises:
        return float("nan"), 0, float("nan")
    return float(np.median(noises)), len(noises), float(np.median(levels))


def flat_pair_gains(
    flat_paths: Sequence[str | Path],
    zero_level,
    read_noise_adu: float,
    *,
    saturation_level: float | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> list[float]:
    """Gain (e-/ADU) from pairs of flats with equal filter and exposure time.

    For two flats with the zero level removed, the difference cancels the
    illumination pattern; its variance per frame is ``S / g + RN^2`` (ADU),
    so ``g = S / (var - RN^2)``. ``zero_level(exptime)`` returns the bias
    (or dark) level in ADU for a flat of that exposure time.
    """
    groups: dict[tuple[str, float], list[tuple[str, str]]] = {}
    for path in flat_paths:
        exptime, filt, date = _header_values(path)
        groups.setdefault((filt, round(exptime, 3)), []).append((date, str(path)))
    saturation = float(saturation_level) if saturation_level else 65535.0
    gains: list[float] = []
    for (_filt, exptime), members in groups.items():
        members.sort()
        files = [p for _d, p in members]
        zero = zero_level(exptime)
        if zero is None or not math.isfinite(zero):
            continue
        for a_path, b_path in zip(files[0::2], files[1::2], strict=False):
            if len(gains) >= max_pairs:
                return gains
            a = central_crop(a_path, crop_size) - zero
            b = central_crop(b_path, crop_size) - zero
            if a.shape != b.shape:
                continue
            level_a, level_b = float(np.median(a)), float(np.median(b))
            if min(level_a, level_b) <= 0:
                continue
            if abs(level_a / level_b - 1.0) > FLAT_PAIR_LEVEL_TOLERANCE:
                continue
            level = 0.5 * (level_a + level_b)
            if not (FLAT_LEVEL_RANGE[0] * saturation <= level + zero
                    <= FLAT_LEVEL_RANGE[1] * saturation):
                continue
            variance = robust_std(a - b * (level_a / level_b)) ** 2 / 2.0 - read_noise_adu**2
            if variance <= read_noise_adu**2:
                continue  # read noise dominates: no usable photon transfer
            gain = level / variance
            if GAIN_RANGE[0] <= gain <= GAIN_RANGE[1]:
                gains.append(gain)
    return gains


def _dark_levels(dark_paths: Sequence[str | Path], crop_size: int) -> dict[float, float]:
    by_exptime: dict[float, list[str]] = {}
    for path in dark_paths:
        exptime, _filt, _date = _header_values(path)
        if math.isfinite(exptime):
            by_exptime.setdefault(round(exptime, 3), []).append(str(path))
    return {t: float(np.median(central_crop(sorted(files)[0], crop_size)))
            for t, files in by_exptime.items()}


def measure_detector_noise(
    bias_files: Sequence[str | Path],
    flat_files: Sequence[str | Path] = (),
    dark_files: Sequence[str | Path] = (),
    *,
    saturation_level: float | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> NoiseMeasurement | None:
    """Read noise (ADU) and gain (e-/ADU) of one electronic setup.

    The frames must share camera, binning, readout mode and gain setting.
    Read noise comes from bias pairs; without at least two bias frames,
    from pairs of the darks with the shortest exposure (their small dark
    current adds a little shot noise). The flats' zero level is the bias
    level, otherwise the level of the dark closest in exposure time.
    Returns ``None`` if no read noise can be measured; ``gain`` is ``None``
    if no usable flat pair exists.
    """
    bias_files = sorted(str(p) for p in bias_files)
    dark_files = sorted(str(p) for p in dark_files)
    zero_source = "bias"
    rn_adu, n_zero, bias_level = pair_noise_adu(bias_files, crop_size=crop_size,
                                                max_pairs=max_pairs)
    dark_levels: dict[float, float] = {}
    if n_zero == 0 and dark_files:
        dark_levels = _dark_levels(dark_files, crop_size)
        shortest = min(dark_levels) if dark_levels else None
        shortest_files = [p for p in dark_files
                          if shortest is not None and round(_header_values(p)[0], 3) == shortest]
        rn_adu, n_zero, _level = pair_noise_adu(shortest_files, crop_size=crop_size,
                                                max_pairs=max_pairs)
        zero_source = "dark"
    if n_zero == 0 or not math.isfinite(rn_adu) or rn_adu <= 0:
        return None

    def zero_level(exptime: float) -> float | None:
        if zero_source == "bias":
            return bias_level
        if not dark_levels:
            return None
        nearest = min(dark_levels, key=lambda t: abs(t - exptime))
        return dark_levels[nearest]

    gains = flat_pair_gains(flat_files, zero_level, rn_adu, saturation_level=saturation_level,
                            crop_size=crop_size, max_pairs=max_pairs)
    gain = float(np.median(gains)) if gains else None
    return NoiseMeasurement(rn_adu, gain, n_zero, len(gains), zero_source)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def add_signal_uncertainty(ccd: CCDData, *, gain: float, read_noise: float) -> CCDData:
    """Per-pixel uncertainty (ADU) of a bias-free frame: Poisson + read noise.

    ``ccd`` must already have the bias removed (or bias and dark when the
    darks carry the bias), so that only collected charge counts as Poisson
    noise. An uncertainty that is already attached (from subtracted
    masters) is added in quadrature.
    """
    import ccdproc as ccdp

    previous = None if ccd.uncertainty is None else np.asarray(ccd.uncertainty.array)
    out = ccdp.create_deviation(
        ccd,
        gain=gain * u.electron / u.adu,
        readnoise=read_noise * u.electron,
        disregard_nan=True,
    )
    if previous is not None and previous.shape == out.shape:
        out.uncertainty = StdDevUncertainty(np.hypot(out.uncertainty.array, previous))
    return out


__all__ = [
    "BINNING_MODES",
    "NOISE_SOURCES",
    "NoiseMeasurement",
    "add_signal_uncertainty",
    "binned_read_noise",
    "check_binning_mode",
    "check_noise_source",
    "flat_pair_gains",
    "measure_detector_noise",
    "pair_noise_adu",
    "resolve_binning_mode",
    "robust_std",
]
