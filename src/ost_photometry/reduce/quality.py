"""Per-frame image-quality metrics for the reduction workflow.

Measures FWHM, roundness, star count, sky background and the masked fraction
of every reduced science frame, writes the results to the quality table
(``frame_quality.ecsv``) and to FITS headers, applies a
:class:`~ost_photometry.reduce.frame_selection.FrameSelection`, and moves
rejected frames out of the way. The pure selection / weighting logic lives
in :mod:`ost_photometry.reduce.frame_selection`.
"""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

from .. import checks, style, terminal_output
from ..core.parallel import Executor, start_plot_process
from ..core.pixel_masks import interior_mask_fraction
from ..fits_headers import (
    FRAME_REFERENCE_KEY,
    mark_frame_rejected,
    set_frame_weight,
    wcs_from_header,
)
from ..fwhm import (
    _finder_roundness_column,
    estimate_fwhm_from_positions,
    estimate_image_fwhm,
    roundness_range_for_finder,
    select_sources_for_fwhm_fit,
    source_positions_from_table,
)
from .frame_selection import (
    GLOBAL_REFERENCE_KEY,
    FrameSelection,
    empty_quality_table,
    group_indices_by_filter,
    mark_reference_frames,
    mark_selection,
    quality_table_from_rows,
    resolve_reference_frames,
    stack_weights,
    summarize_selection,
    write_quality_table,
)
from .image_collection import image_file_collection
from .image_types import get_image_type

#: Quality-table column -> (FITS keyword, comment) written to every frame.
QUALITY_HEADER_KEYWORDS: dict[str, tuple[str, str]] = {
    "fwhm_px": ("FWHM", "Median stellar FWHM [pixel]"),
    "fwhm_arcsec": ("FWHMAS", "Median stellar FWHM [arcsec]"),
    "pixel_scale": ("PIXSCALE", "Pixel scale [arcsec/pixel]"),
    "roundness": ("ROUNDNES", "Median IRAF roundness of FWHM stars"),
    "sharpness": ("SHARPNES", "Median IRAF sharpness of FWHM stars"),
    "n_stars": ("NSTARS", "Stars detected for frame quality"),
    "background": ("BACKGRND", "Sigma-clipped sky median [image units]"),
    "background_rms": ("BKGRMS", "Sigma-clipped sky RMS [image units]"),
    "masked_fraction": ("MASKFRAC", "Masked fraction of the frame interior"),
    "status": ("QCSTAT", "Frame quality status"),
}

_HEADER_CARD_TEXT_LIMIT = 68


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------


def pixel_scale_from_header(header: Mapping[str, object] | fits.Header) -> float | None:
    """Pixel scale in arcsec/pixel from a celestial WCS or header geometry.

    Order: ``proj_plane_pixel_scales`` of a celestial WCS, else
    ``206.265 * XPIXSZ[µm] / FOCALLEN[mm]`` (``PIXSIZE1`` as fallback for the
    pixel size). ``None`` when neither is available; no default focal length
    is assumed so that FWHM values in arcsec are never silently wrong.
    """
    if isinstance(header, fits.Header):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wcs = wcs_from_header(header.copy())
            if wcs.has_celestial:
                from astropy.wcs.utils import proj_plane_pixel_scales

                scales = np.abs(proj_plane_pixel_scales(wcs.celestial)) * 3600.0
                value = float(np.mean(scales))
                if np.isfinite(value) and value > 0.0:
                    return value
        except Exception:  # noqa: BLE001 — malformed WCS falls through to geometry
            pass

    focal = _finite_positive(header.get("FOCALLEN"))
    pixel = _finite_positive(header.get("XPIXSZ"))
    if pixel is None:
        pixel = _finite_positive(header.get("PIXSIZE1"))
    if focal is None or pixel is None:
        return None
    return 206.265 * pixel / focal


def _finite_positive(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return number


def _header_float(header: Mapping[str, object], key: str) -> float:
    try:
        value = float(header.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _update_primary_header(path: Path, cards: Mapping[str, tuple[object, str]]) -> None:
    """Set ``key = (value, comment)`` cards on the primary header in place."""
    with fits.open(path, mode="update") as hdul:
        header = hdul[0].header
        for key, (value, comment) in cards.items():
            if isinstance(value, str):
                value = value[:_HEADER_CARD_TEXT_LIMIT]
            header[key] = (value, comment)
        hdul.flush()


# ---------------------------------------------------------------------------
# Per-frame measurement
# ---------------------------------------------------------------------------


def _find_stars(
    data: np.ndarray,
    mask: np.ndarray | None,
    *,
    fwhm: float,
    threshold: float,
    sharpness_range: tuple[float, float],
) -> Table | None:
    from photutils.detection import IRAFStarFinder

    finder = IRAFStarFinder(
        threshold=float(threshold),
        fwhm=float(fwhm),
        min_separation=max(2, int(float(fwhm) + 0.5)),
        sharpness_range=sharpness_range,
        roundness_range=roundness_range_for_finder("IRAF", (-1.0, 1.0)),
        exclude_border=True,
    )
    try:
        with warnings.catch_warnings():
            #   Empty frames and low-S/N stamps raise NoDetectionsWarning /
            #   divide warnings; the caller handles ``None`` and empty tables.
            warnings.simplefilter("ignore")
            return finder(data, mask=mask)
    except (ValueError, TypeError):
        return None


def _median_of_column(table: Table, column: str | None) -> float:
    if column is None or column not in table.colnames or len(table) == 0:
        return float("nan")
    values = np.asarray(table[column], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def measure_frame_quality(
    file_path: str | Path,
    *,
    threshold_sigma: float = 5.0,
    initial_fwhm: float = 3.0,
    n_select: int = 25,
    min_fwhm: float = 1.0,
    max_fwhm: float = 20.0,
    sharpness_range: tuple[float, float] = (0.2, 1.0),
) -> dict[str, object]:
    """Measure the quality metrics of one frame.

    Never raises for cloudy or empty frames: the returned row carries
    ``status`` ``no_stars`` / ``fwhm_failed`` and ``nan`` metrics instead.
    """
    path = Path(file_path)
    row: dict[str, object] = {"file": path.name, "status": "ok"}

    try:
        ccd = CCDData.read(path)
    except ValueError:
        ccd = CCDData.read(path, unit="adu")
    header = ccd.header
    row["filter"] = str(header.get("FILTER", ""))
    row["imagetyp"] = str(header.get("IMAGETYP", ""))
    row["date_obs"] = str(header.get("DATE-OBS", ""))
    row["jd"] = _header_float(header, "JD")
    row["exptime"] = _header_float(header, "EXPTIME")
    row["airmass"] = _header_float(header, "AIRMASS")

    data = np.asarray(ccd.data, dtype=float)
    source_mask = None if ccd.mask is None else np.asarray(ccd.mask, dtype=bool)
    masked_fraction = interior_mask_fraction(source_mask)
    row["masked_fraction"] = float("nan") if masked_fraction is None else float(masked_fraction)
    mask = ~np.isfinite(data)
    if source_mask is not None:
        mask |= source_mask

    pixel_scale = pixel_scale_from_header(header)
    row["pixel_scale"] = float("nan") if pixel_scale is None else float(pixel_scale)

    if np.all(mask):
        row["status"] = "no_stars"
        return row

    _mean, median, std = sigma_clipped_stats(data, mask=mask, sigma=3.0, maxiters=5)
    row["background"] = float(median)
    row["background_rms"] = float(std)
    if not np.isfinite(std) or std <= 0.0:
        row["status"] = "no_stars"
        return row

    data_sub = data - float(median)
    data_sub[mask] = 0.0
    error = None if ccd.uncertainty is None else np.asarray(ccd.uncertainty.array, dtype=float)
    threshold = float(threshold_sigma) * float(std)

    sources = _find_stars(
        data_sub, mask, fwhm=initial_fwhm, threshold=threshold, sharpness_range=sharpness_range
    )
    if sources is None or len(sources) == 0:
        row["status"] = "no_stars"
        row["n_stars"] = 0
        return row

    #   The FWHM comes from Gaussian fits of the selected stars, not from the
    #   finder's moment-based ``fwhm`` column: that column is biased by the
    #   finder kernel size (a 4 px star measured with a 3 px kernel reads
    #   ~2 px). The finder only provides detections, roundness and sharpness.
    fit_kwargs = dict(mask=mask, error=error, min_fwhm=min_fwhm, max_fwhm=max_fwhm)
    selected = select_sources_for_fwhm_fit(sources, data_shape=data.shape, n_select=n_select)
    fwhm_value, fwhm_error = estimate_fwhm_from_positions(
        data_sub,
        source_positions_from_table(selected),
        default_fwhm=float(initial_fwhm),
        **fit_kwargs,
    )
    fwhm_source = "psf_fit"

    #   Second finder pass with a matched kernel when the first guess was far
    #   off: detections (n_stars) and roundness depend on the kernel size.
    if fwhm_error is None and abs(fwhm_value - initial_fwhm) > 0.3 * initial_fwhm:
        second = _find_stars(
            data_sub, mask, fwhm=fwhm_value, threshold=threshold, sharpness_range=sharpness_range
        )
        if second is not None and len(second) > 0:
            sources = second
            selected = select_sources_for_fwhm_fit(
                sources, data_shape=data.shape, n_select=n_select
            )
            refit, refit_error = estimate_fwhm_from_positions(
                data_sub,
                source_positions_from_table(selected),
                default_fwhm=float(fwhm_value),
                **fit_kwargs,
            )
            if refit_error is None:
                fwhm_value = refit

    if fwhm_error is not None:
        #   Fall back to the finder column / combined estimator.
        fwhm_value, fwhm_error, meta = estimate_image_fwhm(
            data_sub,
            sources,
            default_fwhm=float(initial_fwhm),
            **fit_kwargs,
        )
        fwhm_source = str(meta.get("source", "default"))

    row["n_stars"] = int(len(sources))
    row["roundness"] = _median_of_column(selected, _finder_roundness_column(selected))
    row["sharpness"] = _median_of_column(selected, "sharpness")
    row["n_fwhm_stars"] = int(len(selected))

    if fwhm_error is not None:
        row["status"] = "fwhm_failed"
        row["fwhm_source"] = "default"
        return row

    row["fwhm_px"] = float(fwhm_value)
    row["fwhm_source"] = fwhm_source
    if pixel_scale is not None:
        row["fwhm_arcsec"] = float(fwhm_value) * float(pixel_scale)
    return row


def measure_directory_quality(
    image_path: str | Path,
    *,
    image_type_list: list[str] | None = None,
    n_cores_multiprocessing: int | None = None,
    indent: int = 2,
    **measure_kwargs: object,
) -> Table:
    """Measure every FITS frame in ``image_path`` and return the quality table.

    ``image_type_list`` restricts the frames to one ``IMAGETYP`` (first entry
    present in the collection, as elsewhere in ``reduce``). Frames are
    measured in parallel through :class:`~ost_photometry.core.parallel.Executor`;
    ``n_cores_multiprocessing=1`` runs sequentially in the current process.
    """
    file_path = checks.check_pathlib_path(image_path)
    collection = image_file_collection(file_path)
    if not collection.files:
        return empty_quality_table()

    if image_type_list is not None:
        image_type = get_image_type(collection, image_type_list)
        if not image_type:
            return empty_quality_table()
        files = list(collection.files_filtered(imagetyp=image_type, include_path=True))
    else:
        files = list(collection.files_filtered(include_path=True))
    if not files:
        return empty_quality_table()

    rows: list[dict[str, object]]
    if n_cores_multiprocessing == 1:
        rows = [measure_frame_quality(f, **measure_kwargs) for f in files]  # type: ignore[arg-type]
    else:
        executor = Executor(
            n_cores_multiprocessing,
            n_tasks=len(files),
            add_progress_bar=True,
        )
        for file_name in files:
            executor.schedule(
                measure_frame_quality,
                args=(file_name,),
                kwargs=dict(measure_kwargs),
            )
        if executor.err is not None:
            raise RuntimeError(
                f"\n{style.Bcolors.FAIL}Frame quality measurement using "
                f"multiprocessing failed :({style.Bcolors.ENDC}"
            )
        executor.wait()
        rows = list(executor.res)

    table = quality_table_from_rows(rows)
    n_bad = int(np.count_nonzero(np.asarray(table["status"]) != "ok"))
    if n_bad:
        terminal_output.print_to_terminal(
            f"{n_bad} of {len(table)} frame(s) without a usable FWHM "
            "(no stars or failed fit).",
            indent=indent,
            style_name="WARNING",
        )
    return table


# ---------------------------------------------------------------------------
# Header / file bookkeeping
# ---------------------------------------------------------------------------


def write_quality_to_headers(table: Table, image_dir: str | Path) -> int:
    """Write the quality metrics of every row to the frame header; returns count."""
    image_dir = Path(image_dir)
    written = 0
    for row in table:
        path = image_dir / str(row["file"])
        if not path.is_file():
            continue
        cards: dict[str, tuple[object, str]] = {}
        for column, (key, comment) in QUALITY_HEADER_KEYWORDS.items():
            value = row[column]
            if isinstance(value, str | np.str_):
                cards[key] = (str(value), comment)
            elif isinstance(value, bool | np.bool_):
                cards[key] = (bool(value), comment)
            elif isinstance(value, int | np.integer):
                cards[key] = (int(value), comment)
            else:
                number = float(value)
                if np.isfinite(number):
                    cards[key] = (number, comment)
        if cards:
            _update_primary_header(path, cards)
            written += 1
    return written


def write_weights_to_headers(table: Table, image_dir: str | Path) -> int:
    """Write finite ``stack_weight`` values as ``FRMWGHT``; returns count."""
    image_dir = Path(image_dir)
    written = 0
    for row in table:
        weight = float(row["stack_weight"])
        if not np.isfinite(weight):
            continue
        path = image_dir / str(row["file"])
        if not path.is_file():
            continue
        with fits.open(path, mode="update") as hdul:
            set_frame_weight(hdul[0].header, weight)
            hdul[0].header.comments["FRMWGHT"] = "Relative stack weight (mean 1 per filter)"
            hdul.flush()
        written += 1
    return written


def move_rejected_frames(
    table: Table,
    image_dir: str | Path,
    rejected_dir: str | Path,
) -> list[Path]:
    """Flag rejected frames in their header and move them to ``rejected_dir``."""
    image_dir = Path(image_dir)
    rejected_dir = Path(rejected_dir)
    moved: list[Path] = []
    rejected = np.asarray(table["rejected"], dtype=bool)
    if not np.any(rejected):
        return moved
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for row, is_rejected in zip(table, rejected, strict=True):
        if not is_rejected:
            continue
        source = image_dir / str(row["file"])
        if not source.is_file():
            continue
        with fits.open(source, mode="update") as hdul:
            mark_frame_rejected(hdul[0].header, reason=str(row["reject_reason"]))
            hdul.flush()
        target = rejected_dir / source.name
        shutil.move(str(source), str(target))
        moved.append(target)
    return moved


def apply_frame_selection(
    table: Table,
    selection: FrameSelection,
    *,
    image_dir: str | Path,
    rejected_dir: str | Path,
    per_filter_reference: bool = True,
    indent: int = 2,
) -> tuple[Table, dict[str, str]]:
    """Mark rejections, pick reference frames, move rejected files.

    Returns the updated table and the reference mapping
    (``{filter: file}`` or ``{GLOBAL_REFERENCE_KEY: file}``).
    """
    image_dir = Path(image_dir)
    mark_selection(table, selection, indent=indent)
    references = resolve_reference_frames(table, per_filter=per_filter_reference, indent=indent)
    mark_reference_frames(table, references)
    for file_name in references.values():
        path = image_dir / file_name
        if path.is_file():
            _update_primary_header(
                path, {FRAME_REFERENCE_KEY: (True, "Alignment reference frame")}
            )
    move_rejected_frames(table, image_dir, rejected_dir)
    return table, references


def assess_frame_quality(
    image_dir: str | Path,
    *,
    image_type_list: list[str] | None,
    selection: FrameSelection | Mapping[str, object] | None,
    stack_weighting: str = "none",
    per_filter_reference: bool = True,
    rejected_dir: str | Path | None = None,
    table_path: str | Path | None = None,
    n_cores_multiprocessing: int | None = None,
    indent: int = 2,
    **measure_kwargs: object,
) -> tuple[Table, dict[str, str]]:
    """Complete quality step: measure, annotate headers, select, weight, summarise.

    ``rejected_dir`` defaults to ``<image_dir>/../rejected_lights`` and
    ``table_path`` to ``<image_dir>/../frame_quality.ecsv``.
    """
    image_dir = Path(image_dir)
    if rejected_dir is None:
        rejected_dir = image_dir.parent / "rejected_lights"
    if table_path is None:
        table_path = image_dir.parent / "frame_quality.ecsv"
    selection = FrameSelection.from_mapping(selection)

    table = measure_directory_quality(
        image_dir,
        image_type_list=image_type_list,
        n_cores_multiprocessing=n_cores_multiprocessing,
        indent=indent,
        **measure_kwargs,
    )
    if len(table) == 0:
        terminal_output.print_to_terminal(
            f"No frames found in {image_dir} for the quality assessment.",
            indent=indent,
            style_name="WARNING",
        )
        return table, {}

    write_quality_to_headers(table, image_dir)
    if selection.is_active():
        terminal_output.print_to_terminal(
            f"Frame selection: {selection.describe()}", indent=indent
        )
    table, references = apply_frame_selection(
        table,
        selection,
        image_dir=image_dir,
        rejected_dir=rejected_dir,
        per_filter_reference=per_filter_reference,
        indent=indent,
    )
    table["stack_weight"] = stack_weights(table, stack_weighting, indent=indent)
    write_weights_to_headers(table, image_dir)
    summarize_selection(table, indent=indent)
    write_quality_table(table, table_path)
    return table, references


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_PLOT_COLUMNS = (
    "file",
    "fwhm_px",
    "fwhm_arcsec",
    "roundness",
    "n_stars",
    "background",
    "rejected",
    "is_reference",
    "aligned",
)


def plot_frame_quality(
    table: Table,
    output_dir: str | Path,
    *,
    selection: FrameSelection | Mapping[str, object] | None = None,
    blocking: bool = False,
) -> list[Path]:
    """Write one frame-quality overview PDF per filter.

    Plots run in a child process (:func:`start_plot_process`) unless
    ``blocking`` is set. Returns the expected PDF paths.
    """
    from . import plots

    if len(table) == 0:
        return []
    selection = FrameSelection.from_mapping(selection)
    alignment_known = bool(np.any(np.asarray(table["aligned"], dtype=bool)))
    fwhm_max = None if selection.fwhm_max is None else float(selection.fwhm_max)
    fwhm_unit = selection.fwhm_unit

    paths: list[Path] = []
    for filt, idx in group_indices_by_filter(table).items():
        rows = {name: np.asarray(table[name][idx]) for name in _PLOT_COLUMNS}
        rows["file"] = np.asarray([str(f) for f in rows["file"]])
        kwargs = {
            "fwhm_max": fwhm_max,
            "fwhm_unit": fwhm_unit,
            "alignment_known": alignment_known,
        }
        if blocking:
            paths.append(plots.frame_quality_overview(rows, output_dir, filt, **kwargs))
        else:
            start_plot_process(
                plots.frame_quality_overview,
                args=(rows, output_dir, filt),
                kwargs=kwargs,
            )
            paths.append(
                Path(output_dir)
                / "diagnostics"
                / "frame_quality"
                / f"frame_quality_{plots._safe_filter_name(filt)}.pdf"
            )
    return paths


__all__ = [
    "GLOBAL_REFERENCE_KEY",
    "QUALITY_HEADER_KEYWORDS",
    "apply_frame_selection",
    "assess_frame_quality",
    "measure_directory_quality",
    "measure_frame_quality",
    "move_rejected_frames",
    "pixel_scale_from_header",
    "plot_frame_quality",
    "write_quality_to_headers",
    "write_weights_to_headers",
]
