"""Fetch a HiPS archival cutout and subtract it from the science image."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astroquery.hips2fits import hips2fitsClass

from ... import checks, terminal_output
from ...utilities import Image, get_basename
from ...wcs import find_wcs_for_image
from .. import plots, subtraction
from ..models import ImageSeries

DEFAULT_HIPS_SERVER = (
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
)
DEFAULT_HIPS_FALLBACK_SERVERS: tuple[str, ...] = (
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
)

# Optional fixed trim matching an older one-off workflow (cropped 1599×2501).
LEGACY_SUBTRACT_TRIM: tuple[int, int, int, int] = (0, 1599, 0, 2501)

# Bessell → DSS2 photographic plates; SDSS-like → PanSTARRS (closer bandpass).
_HIPS_SOURCE_FOR_FILTER: dict[str, str] = {
    "U": "CDS/P/DSS2/blue",
    "B": "CDS/P/DSS2/blue",
    "V": "CDS/P/DSS2/red",
    "R": "CDS/P/DSS2/red",
    "I": "CDS/P/DSS2/red",
    "u": "CDS/P/PanSTARRS/DR1/g",
    "g": "CDS/P/PanSTARRS/DR1/g",
    "r": "CDS/P/PanSTARRS/DR1/r",
    "i": "CDS/P/PanSTARRS/DR1/i",
    "z": "CDS/P/PanSTARRS/DR1/z",
    "y": "CDS/P/PanSTARRS/DR1/y",
}
_HIPS_SOURCE_FALLBACK = "CDS/P/DSS2/red"


@dataclass(frozen=True)
class HipsReferenceSubtractResult:
    work_dir: Path
    difference_fits: Path
    hips_fits: Path
    science_fits_path: Path
    hips_source: str
    hips_from_cache: bool
    subtract_backend: str


def hips_source_for_filter(
    filter_: str,
    explicit: str | None = None,
) -> str:
    """HiPS survey id: explicit string wins; otherwise a bandpass-matched default."""
    if explicit is not None:
        text = str(explicit).strip()
        if text and text.lower() not in ("auto", "?", "none"):
            return text
    key = str(filter_).strip()
    return _HIPS_SOURCE_FOR_FILTER.get(key, _HIPS_SOURCE_FALLBACK)


def hips_timeout_seconds(timeout_ms: int | float) -> float:
    """``hips2fitsClass.timeout`` is seconds; config stores milliseconds."""
    ms = float(timeout_ms)
    if ms <= 0:
        return 120.0
    if ms >= 1000.0:
        return ms / 1000.0
    return ms


def _wcs_cache_fingerprint(wcs_obj, shape: tuple[int, ...]) -> str:
    parts = [str(tuple(int(n) for n in shape))]
    try:
        hdr = wcs_obj.to_header()
        for key in (
            "CTYPE1",
            "CTYPE2",
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CDELT1",
            "CDELT2",
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "NAXIS1",
            "NAXIS2",
        ):
            if key in hdr:
                parts.append(f"{key}={hdr[key]}")
    except Exception:
        parts.append(repr(wcs_obj))
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest


def hips_cache_filename(hips_source: str, wcs_obj, shape: tuple[int, ...]) -> str:
    safe = (
        str(hips_source)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )
    return f"hips_{safe}_{_wcs_cache_fingerprint(wcs_obj, shape)}.fits"


def _server_list(primary: str, fallbacks: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for url in (primary, *(fallbacks or ())):
        text = str(url).strip()
        if text and text not in out:
            out.append(text)
    return out or [DEFAULT_HIPS_SERVER]


def fetch_hips_cutout(
    wcs_obj,
    hips_source: str,
    *,
    workdir: Path,
    shape: tuple[int, ...],
    timeout_ms: int = 120_000,
    server: str = DEFAULT_HIPS_SERVER,
    fallback_servers: Sequence[str] | None = DEFAULT_HIPS_FALLBACK_SERVERS,
    retries: int = 3,
    retry_backoff_s: float = 1.5,
    use_cache: bool = True,
    verbose: bool = False,
) -> tuple[fits.HDUList, Path, bool]:
    """
    Query HiPS2FITS with cache, retries, and fallback servers.

    Returns ``(hdus, path, from_cache)``. Raises ``RuntimeError`` if every
    attempt fails (caller may skip the pipeline step).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cache_path = workdir / hips_cache_filename(hips_source, wcs_obj, shape)
    if use_cache and cache_path.is_file():
        try:
            hdus = fits.open(cache_path)
            terminal_output.print_to_terminal(
                f"HiPS cache hit ({cache_path.name})",
                indent=2,
                style_name="NORMAL",
            )
            return hdus, cache_path, True
        except Exception as exc:
            terminal_output.print_to_terminal(
                f"HiPS cache unreadable ({cache_path.name}): {exc}; re-fetching",
                indent=2,
                style_name="WARNING",
            )

    timeout_s = hips_timeout_seconds(timeout_ms)
    n_try = max(1, int(retries))
    last_err: BaseException | None = None
    for server_url in _server_list(server, fallback_servers):
        for attempt in range(1, n_try + 1):
            try:
                terminal_output.print_to_terminal(
                    f"HiPS2FITS query ({hips_source}) via {server_url} "
                    f"[{attempt}/{n_try}] ...",
                    indent=2,
                    style_name="NORMAL",
                )
                hips_instance = hips2fitsClass()
                hips_instance.timeout = timeout_s
                hips_instance.server = server_url
                hdus = hips_instance.query_with_wcs(
                    hips=hips_source,
                    wcs=wcs_obj,
                    get_query_payload=False,
                    format="fits",
                    verbose=verbose,
                )
                hdus.writeto(str(cache_path), overwrite=True)
                return hdus, cache_path, False
            except Exception as exc:
                last_err = exc
                terminal_output.print_to_terminal(
                    f"HiPS2FITS failed ({server_url}, attempt {attempt}/{n_try}): "
                    f"{exc}",
                    indent=2,
                    style_name="WARNING",
                )
                if attempt < n_try and retry_backoff_s > 0:
                    time.sleep(float(retry_backoff_s) * (2 ** (attempt - 1)))
    msg = f"HiPS2FITS failed for {hips_source!r} after retries and fallback servers."
    raise RuntimeError(msg) from last_err


def run_hips_reference_subtraction(
    filter_: str,
    science_image_path: str,
    work_output_dir: str | Path,
    *,
    wcs_method: str = "astap",
    plot_comp: bool = True,
    hips_source: str | None = None,
    file_type_plots: str = "pdf",
    trim_slice_yx: tuple[int, int, int, int] | None = None,
    reuse_wcs_image_series: ImageSeries | Image | None = None,
    hips_timeout_ms: int = 120_000,
    hips_server: str = DEFAULT_HIPS_SERVER,
    hips_fallback_servers: Sequence[str] | None = DEFAULT_HIPS_FALLBACK_SERVERS,
    hips_retries: int = 3,
    hips_retry_backoff_s: float = 1.5,
    hips_use_cache: bool = True,
    hips_verbose: bool = False,
    subtract_backend: str = "auto",
    hotpants_executable: str | None = None,
    hotpants_extra_args: Sequence[str] | None = None,
    hotpants_output_filename: str = "diff.fits",
    image_gain: float = 1.0,
    template_gain: float | None = None,
) -> HipsReferenceSubtractResult:
    """
    Trim (optional), solve or reuse WCS, query HiPS2FITS, optionally plot, subtract.

    Parameters
    ----------
    trim_slice_yx
        If set, ``(y0, y1, x0, x1)`` slice applied as ``ccd[y0:y1, x0:x1]`` before
        WCS and HiPS. Pipeline default is ``None`` (full frame). Use
        `LEGACY_SUBTRACT_TRIM` for the old fixed crop.
    reuse_wcs_image_series
        When ``trim_slice_yx`` is ``None`` and this object has ``.wcs`` set
        (e.g. an :class:`~ost_photometry.analyze.models.ImageSeries` after the
        pipeline WCS step, or a single :class:`~ost_photometry.utilities.Image`),
        that WCS is reused and WCS solving is skipped. The same WCS is written
        onto the science ``CCDData`` before subtraction.
    hips_source
        HiPS id, or ``None`` / ``"auto"`` to pick from ``filter_``.
    subtract_backend
        ``auto`` (HOTPANTS if present, else Alard–Lupton), ``hotpants``, or
        ``alard_lupton``.
    """
    workdir = Path(work_output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    checks.check_file(science_image_path)
    source = hips_source_for_filter(filter_, hips_source)
    terminal_output.print_to_terminal(
        f"HiPS template survey: {source}",
        indent=2,
        style_name="NORMAL",
    )

    ccd_image = CCDData.read(science_image_path)

    if trim_slice_yx is not None:
        y0, y1, x0, x1 = trim_slice_yx
        ccd_image = ccdp.trim_image(ccd_image[y0:y1, x0:x1])
        trimmed_name = f"{get_basename(science_image_path)}_trimmed.fit"
        path_for_series = workdir / trimmed_name
        path_for_series = str(path_for_series.resolve())
    else:
        path_for_series = str(Path(science_image_path).resolve())

    can_reuse_wcs = (
        trim_slice_yx is None
        and reuse_wcs_image_series is not None
        and getattr(reuse_wcs_image_series, "wcs", None) is not None
    )

    if can_reuse_wcs:
        wcs_obj = reuse_wcs_image_series.wcs
    else:
        image = Image(0, filter_, path_for_series, str(workdir))
        wcs_obj = find_wcs_for_image(
            image,
            method=wcs_method,
            indent=2,
        )

    ccd_image.wcs = wcs_obj
    if trim_slice_yx is not None:
        ccd_image.write(path_for_series, overwrite=True)

    shape = tuple(int(n) for n in np.asarray(ccd_image.data).shape)
    hips_hdus, hips_path, from_cache = fetch_hips_cutout(
        wcs_obj,
        source,
        workdir=workdir,
        shape=shape,
        timeout_ms=hips_timeout_ms,
        server=hips_server,
        fallback_servers=hips_fallback_servers,
        retries=hips_retries,
        retry_backoff_s=hips_retry_backoff_s,
        use_cache=hips_use_cache,
        verbose=hips_verbose,
    )

    if plot_comp:
        plots.compare_images(
            str(workdir),
            ccd_image,
            hips_hdus[0],
            file_type=file_type_plots,
        )

    resolved_backend = subtraction.resolve_subtract_backend(
        subtract_backend, hotpants_executable
    )
    diff_path = subtraction.subtract_science_template(
        ccd_image,
        hips_hdus[0],
        workdir=str(workdir),
        output_filename=hotpants_output_filename,
        backend=resolved_backend,
        template_mask=None,
        image_gain=image_gain,
        template_gain=template_gain,
        hotpants_executable=hotpants_executable,
        extra_args=hotpants_extra_args,
    )
    hips_hdus.close()

    return HipsReferenceSubtractResult(
        work_dir=workdir,
        difference_fits=diff_path,
        hips_fits=hips_path,
        science_fits_path=Path(path_for_series),
        hips_source=source,
        hips_from_cache=from_cache,
        subtract_backend=resolved_backend,
    )
