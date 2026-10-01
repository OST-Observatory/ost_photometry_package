"""Frame-type classification from image statistics, header and archive.

Header ``IMAGETYP`` values are often wrong in the archive (e.g. night-time
light frames labelled ``Flat Field``). Downloaded frames are therefore
classified from cheap image statistics first: star count, signal level
above the bias level of their electronic setup, saturation, exposure time.
Header, archive ML and archive user types confirm or settle ambiguous
cases. Frames without data (archive context rows) use the archive types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import maximum_filter, uniform_filter, uniform_filter1d

from .setup_keys import electronic_key

BIAS, DARK, FLAT, LIGHT = "bias", "dark", "flat", "light"
SPECTROSCOPY, UNKNOWN = "spectroscopy", "unknown"
SATURATED = "saturated"
FRAME_TYPES = (BIAS, DARK, FLAT, LIGHT, SPECTROSCOPY, SATURATED, UNKNOWN)

#: File / folder name fragments of spectrograph frames (local data has no
#: archive spectrograph flag). Substrings, and whole tokens for short words.
_SPECTRO_SUBSTRINGS = ("spectr", "spektr", "thar", "dados", "baches", "autoguider",
                       "einstein")
_SPECTRO_TOKENS = {"near", "neon", "argon", "arc", "lamp", "wave", "wavecal"}


def spectroscopy_name_hint(*names: object) -> str:
    """The spectroscopy keyword found in file / folder names, else ``""``."""
    import re

    for name in names:
        text = str(name or "").lower()
        for fragment in _SPECTRO_SUBSTRINGS:
            if fragment in text:
                return fragment
        tokens = set(re.split(r"[^a-z0-9]+", text))
        hit = tokens & _SPECTRO_TOKENS
        if hit:
            return sorted(hit)[0]
    return ""

_ARCHIVE_CODES = {"BI": BIAS, "DA": DARK, "FL": FLAT, "LI": LIGHT, "WA": SPECTROSCOPY}

#: Thresholds of the statistics rules (all tunable through ``classify_frames``).
DEFAULT_THRESHOLDS = {
    "bias_max_exptime": 0.01,  # s
    "light_min_stars": 10,
    "flat_max_stars": 4,
    "flat_min_signal_adu": 1000.0,  # median above bias level
    "bright_fill": 0.15,  # median / saturation: twilight / dome flat level
    "saturated_fill": 0.9,  # median / saturation
    "detection_sigma": 8.0,
    # Spectra: structure along one axis only (calibrated on archive test data:
    # imaging frames < 3 / < 0.2, spectra and spectrograph flats >= 15 / >= 0.4,
    # arc lamps >= 4 / >= 1).
    "spectrum_anisotropy": 15.0,
    "spectrum_amplitude": 0.4,
    "arc_anisotropy": 4.0,
    "arc_amplitude": 1.0,
}


def header_frame_type(imagetyp: object) -> str:
    """Frame type from an ``IMAGETYP`` string (case-insensitive substrings)."""
    text = str(imagetyp or "").strip().lower()
    if not text:
        return UNKNOWN
    for key, value in (("bias", BIAS), ("zero", BIAS), ("dark", DARK), ("flat", FLAT),
                       ("light", LIGHT), ("object", LIGHT), ("science", LIGHT)):
        if key in text:
            return value
    return UNKNOWN


def archive_frame_type(code: object) -> str:
    return _ARCHIVE_CODES.get(str(code or "").strip().upper(), UNKNOWN)


# ---------------------------------------------------------------------------
# Image statistics
# ---------------------------------------------------------------------------


def _central_crop(data: np.ndarray, max_size: int) -> np.ndarray:
    ny, nx = data.shape
    y0 = max(0, (ny - max_size) // 2)
    x0 = max(0, (nx - max_size) // 2)
    return data[y0 : y0 + max_size, x0 : x0 + max_size]


def count_stars(
    data: np.ndarray,
    *,
    detection_sigma: float = 8.0,
    max_size: int = 2048,
    min_ring_fraction: float = 0.15,
) -> int:
    """Rough number of extended point sources in the central part of a frame.

    Candidates are local maxima of a high-pass image above
    ``detection_sigma`` times the robust noise. A star spreads light into
    its eight neighbours; hot pixels and most cosmics do not. Candidates
    whose neighbour mean is below ``min_ring_fraction`` of the peak are
    therefore rejected, which keeps darks with many hot pixels from being
    mistaken for light frames.
    """
    image = _central_crop(np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0), max_size)
    if image.size < 64 * 64:
        return 0
    highpass = image - uniform_filter(image, size=25)
    noise = 1.4826 * float(np.median(np.abs(highpass - np.median(highpass))))
    if not math.isfinite(noise) or noise <= 0:
        return 0
    peaks = (highpass == maximum_filter(highpass, size=5)) & (highpass > detection_sigma * noise)
    ring = (uniform_filter(highpass, size=3) * 9.0 - highpass) / 8.0
    stars = peaks & (ring > min_ring_fraction * highpass) & (ring > 2.0 * noise)
    border = 12
    stars[:border] = stars[-border:] = False
    stars[:, :border] = stars[:, -border:] = False
    return int(np.count_nonzero(stars))


def profile_anisotropy(data: np.ndarray) -> tuple[float, float]:
    """``(ratio, amplitude)`` of the row / column median profiles.

    The high-pass scatter of the column profile and of the row profile is
    compared: spectra (traces, orders, arc lines) are structured along one
    axis only and give a large ratio; star fields and calibration frames do
    not. ``amplitude`` is the larger scatter in units of the pixel noise.
    """
    image = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0)
    step = max(1, max(image.shape) // 2048)
    image = image[::step, ::step]
    if min(image.shape) < 32:
        return float("nan"), float("nan")

    def scatter(profile: np.ndarray) -> float:
        n = max(5, profile.size // 20)
        residual = profile - uniform_filter1d(profile, n, mode="nearest")
        return float(np.std(residual[n:-n]))

    sx = scatter(np.median(image, axis=0))
    sy = scatter(np.median(image, axis=1))
    sample = image[::2, ::2]
    noise = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
    ratio = max(sx, sy) / (min(sx, sy) + 1e-9)
    amplitude = max(sx, sy) / noise if noise > 0 else float("nan")
    return ratio, amplitude


def frame_statistics(path: str | Path, *, detection_sigma: float = 8.0) -> dict[str, float]:
    """Median, noise, saturation fill and star count of a raw frame."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    if data is None or np.ndim(data) != 2:
        return {"median": float("nan"), "noise": float("nan"), "fill": float("nan"),
                "n_stars": 0.0, "anisotropy": float("nan"), "amplitude": float("nan")}
    # astropy applies BZERO, so unsigned 16-bit cameras arrive as uint16.
    if np.issubdtype(data.dtype, np.integer):
        saturation = float(np.iinfo(data.dtype).max)
    else:
        saturation = 65535.0
    data = np.asarray(data, dtype=np.float32)
    sample = data[:: max(1, data.shape[0] // 512), :: max(1, data.shape[1] // 512)]
    median = float(np.nanmedian(sample))
    noise = 1.4826 * float(np.nanmedian(np.abs(sample - median)))
    ratio, amplitude = profile_anisotropy(data)
    return {
        "median": median,
        "noise": noise,
        "fill": median / saturation if saturation > 0 else float("nan"),
        "n_stars": float(count_stars(data, detection_sigma=detection_sigma)),
        "anisotropy": ratio,
        "amplitude": amplitude,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _num(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _bias_levels(rows: Sequence[Mapping[str, object]], stats: Sequence[Mapping[str, float]],
                 thresholds: Mapping[str, float]) -> dict[str, float]:
    """Median level of zero-second frames per electronic setup."""
    levels: dict[str, list[float]] = {}
    for row, st in zip(rows, stats, strict=True):
        exptime = _num(row.get("exptime"))
        if math.isfinite(exptime) and exptime <= thresholds["bias_max_exptime"]:
            if math.isfinite(st.get("median", float("nan"))):
                levels.setdefault(electronic_key(row), []).append(st["median"])
    return {key: float(np.median(v)) for key, v in levels.items()}


def classify_frame(
    row: Mapping[str, object],
    stats: Mapping[str, float] | None,
    *,
    bias_level: float | None = None,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> tuple[str, float, str]:
    """``(frame_type, confidence, note)`` for one frame."""
    header_type = header_frame_type(row.get("imagetyp"))
    archive_header = archive_frame_type(row.get("exposure_type"))
    ml_type = archive_frame_type(row.get("exposure_type_ml"))
    user_type = archive_frame_type(row.get("exposure_type_user"))
    claimed = header_type if header_type != UNKNOWN else archive_header

    if str(row.get("exposure_type_user") or "").strip():
        return user_type, 1.0, "archive user classification"
    if str(row.get("spectrograph") or "").strip().upper() not in {"", "N"}:
        return SPECTROSCOPY, 1.0, f"archive spectrograph {row.get('spectrograph')}"
    if ml_type == SPECTROSCOPY or archive_header == SPECTROSCOPY:
        return SPECTROSCOPY, 0.9, "archive wavelength-calibration frame"
    hint = spectroscopy_name_hint(row.get("file_name"), row.get("source_path"),
                                  row.get("object"))
    if hint:
        return SPECTROSCOPY, 0.6, f"spectroscopy keyword {hint!r} in file / folder name"

    if not stats or not math.isfinite(stats.get("median", float("nan"))):
        # No pixel data: fall back on the archive / header types.
        for candidate, source in ((claimed, "header"), (ml_type, "archive ML")):
            if candidate != UNKNOWN:
                return candidate, 0.6, f"no pixel data; {source} type"
        return UNKNOWN, 0.0, "no pixel data and no type information"

    exptime = _num(row.get("exptime"))
    n_stars = int(stats.get("n_stars", 0))
    median = float(stats["median"])
    fill = float(stats.get("fill", float("nan")))

    ratio = float(stats.get("anisotropy", float("nan")))
    amplitude = float(stats.get("amplitude", float("nan")))
    if math.isfinite(ratio) and math.isfinite(amplitude) and (
        (ratio >= thresholds["spectrum_anisotropy"] and amplitude >= thresholds["spectrum_amplitude"])
        or (ratio >= thresholds["arc_anisotropy"] and amplitude >= thresholds["arc_amplitude"])
    ):
        return SPECTROSCOPY, 0.8, f"spectral structure along one axis (ratio {ratio:.0f})"

    def note(kind: str, reason: str) -> str:
        if claimed not in (UNKNOWN, kind):
            return f"{reason}; header says {claimed}"
        return reason

    if math.isfinite(exptime) and exptime <= thresholds["bias_max_exptime"]:
        return BIAS, 0.95, note(BIAS, f"exposure {exptime:g} s")
    if math.isfinite(fill) and fill >= thresholds["saturated_fill"]:
        # Over-exposed (twilight, dome light): neither a usable light nor a flat.
        return SATURATED, 0.9, f"saturated (median at {fill:.0%} of full scale)"
    if n_stars >= thresholds["light_min_stars"]:
        bright = math.isfinite(fill) and fill >= thresholds["bright_fill"]
        if bright and FLAT in (claimed, ml_type):
            # Spectrograph flats and twilight flats with a few stars are bright
            # and structured; the header / ML type settles them.
            return FLAT, 0.7, f"bright ({fill:.0%} of full scale) with structure; flat by type"
        return LIGHT, 0.95, note(LIGHT, f"{n_stars} stars")


    if n_stars <= thresholds["flat_max_stars"] and bias_level is not None:
        signal = median - bias_level
        if signal >= thresholds["flat_min_signal_adu"]:
            return FLAT, 0.9, note(FLAT, f"{signal:.0f} ADU above bias, no stars")
        return DARK, 0.9, note(DARK, f"{signal:.0f} ADU above bias, no stars")

    # Ambiguous without a bias reference: trust header / ML for dark vs flat.
    for candidate, source in ((claimed, "header"), (ml_type, "archive ML")):
        if candidate in (DARK, FLAT):
            return candidate, 0.6, f"{n_stars} stars, no bias reference; {source} type"
        if candidate == LIGHT:
            return LIGHT, 0.4, f"only {n_stars} stars (clouds?); {source} type"
    return UNKNOWN, 0.2, f"{n_stars} stars, no bias reference, no type information"


def _statistics_task(index: int, path: str, detection_sigma: float):
    """Worker for parallel statistics: ``(index, stats or None)``."""
    try:
        return index, frame_statistics(path, detection_sigma=detection_sigma)
    except (OSError, ValueError):
        return index, None


def compute_frame_statistics(
    paths: Sequence[str],
    *,
    detection_sigma: float = 8.0,
    n_cores_multiprocessing: int | None = None,
) -> list[dict[str, float] | None]:
    """:func:`frame_statistics` for many files (parallel above 8 files)."""
    results: list[dict[str, float] | None] = [None] * len(paths)
    todo = [(i, p) for i, p in enumerate(paths) if p and Path(p).is_file()]
    if len(todo) <= 8 or n_cores_multiprocessing == 1:
        for i, path in todo:
            results[i] = _statistics_task(i, path, detection_sigma)[1]
        return results
    from ...core.parallel import Executor

    executor = Executor(n_cores_multiprocessing, n_tasks=len(todo), add_progress_bar=True)
    for i, path in todo:
        executor.schedule(_statistics_task, args=(i, path, detection_sigma))
    if executor.err is not None:
        raise RuntimeError(f"Frame statistics failed: {executor.err}")
    executor.wait()
    for i, stat in executor.res:
        results[i] = stat
    return results


def classify_frames(
    manifest: Table,
    *,
    stats: Sequence[Mapping[str, float] | None] | None = None,
    thresholds: Mapping[str, float] | None = None,
    compute_statistics: bool = True,
    n_cores_multiprocessing: int | None = None,
) -> Table:
    """Add ``frame_type``, ``type_confidence``, ``type_note`` and statistics.

    ``stats`` (one dict per row, or ``None``) can be passed in; otherwise
    :func:`frame_statistics` runs on every row with a readable
    ``local_path``. Returns a copy of the manifest.
    """
    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    table = manifest.copy()
    rows = [dict(zip(table.colnames, r, strict=True)) for r in table]
    if stats is None:
        paths = [str(row.get("local_path") or "") if compute_statistics else "" for row in rows]
        stats = compute_frame_statistics(
            paths, detection_sigma=limits["detection_sigma"],
            n_cores_multiprocessing=n_cores_multiprocessing,
        )
    stats = list(stats)
    usable = [st or {} for st in stats]
    levels = _bias_levels(rows, usable, limits)

    types, confidences, notes = [], [], []
    for row, st in zip(rows, stats, strict=True):
        kind, confidence, note = classify_frame(
            row, st, bias_level=levels.get(electronic_key(row)), thresholds=limits
        )
        types.append(kind)
        confidences.append(confidence)
        notes.append(note)
    table["frame_type"] = np.array(types, dtype=str)
    table["type_confidence"] = np.array(confidences, dtype=float)
    table["type_note"] = np.array(notes, dtype=str)
    for key in ("median", "noise", "fill", "n_stars", "anisotropy", "amplitude"):
        table[f"stat_{key}"] = np.array(
            [float(st.get(key, np.nan)) if st else np.nan for st in stats], dtype=float
        )
    return table


def type_disagreements(table: Table) -> Table:
    """Rows whose classified type differs from the header ``IMAGETYP``."""
    header = np.array([header_frame_type(v) for v in table["imagetyp"]])
    kind = np.asarray(table["frame_type"])
    mask = (header != UNKNOWN) & (kind != UNKNOWN) & (header != kind)
    return table[mask]


__all__ = [
    "BIAS",
    "DARK",
    "DEFAULT_THRESHOLDS",
    "FLAT",
    "FRAME_TYPES",
    "LIGHT",
    "SATURATED",
    "SPECTROSCOPY",
    "UNKNOWN",
    "archive_frame_type",
    "classify_frame",
    "classify_frames",
    "compute_frame_statistics",
    "count_stars",
    "frame_statistics",
    "header_frame_type",
    "profile_anisotropy",
    "spectroscopy_name_hint",
    "type_disagreements",
]
