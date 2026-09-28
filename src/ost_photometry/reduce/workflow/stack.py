"""Reduction workflow: stack module."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table

from ... import checks, style, terminal_output
from ...core.parallel import Executor
from ...fits_headers import frame_weight
from .. import utilities
from ..frame_selection import SUPPORTED_STACK_WEIGHTING

#: Stack header keywords written from ``stack_meta``: key -> (meta field, comment).
STACK_META_KEYWORDS: dict[str, tuple[str, str]] = {
    "WEIGHTNG": ("weighting", "Stack weighting scheme"),
    "NFRAMES0": ("n_frames_total", "Frames of this filter before selection"),
    "NREJECT": ("n_rejected", "Frames rejected by the quality selection"),
    "NALIGNFL": ("n_align_failed", "Frames that could not be aligned"),
    "FWHMMED": ("fwhm_median", "Median FWHM of stacked frames [pixel]"),
    "FWHMMAX": ("fwhm_max", "Max FWHM of stacked frames [pixel]"),
}


def prepare_stack_weights(
    weights: Sequence[float] | np.ndarray | None,
    stacking_method: str,
    n_images: int,
    *,
    label: str = "",
    indent: int = 2,
) -> np.ndarray | None:
    """Validate per-image weights for :func:`ccdproc.combine`.

    Returns ``None`` when no weighting is needed (``weights`` is ``None``,
    all weights are equal, or the method is ``median``, which ignores
    weights in ccdproc). For ``sum`` the weights are rescaled so that they
    add up to ``n_images`` and the sum keeps its meaning.
    """
    if weights is None:
        return None
    array = np.asarray(weights, dtype=float).ravel()
    if array.shape[0] != n_images:
        raise ValueError(
            f"{len(array)} stack weights for {n_images} images{' (' + label + ')' if label else ''}"
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("stack weights must be finite and >= 0")
    if array.size == 0 or np.allclose(array, array[0]):
        return None
    if stacking_method == "median":
        terminal_output.print_to_terminal(
            f"Stack weights ignored{' for ' + label if label else ''}: "
            "ccdproc's median combine does not support weights. Use "
            "stack_method='average' for weighted stacking.",
            indent=indent,
            style_name="WARNING",
        )
        return None
    if stacking_method == "sum":
        total = float(array.sum())
        if total <= 0.0:
            return None
        array = array * (n_images / total)
    return array


def _write_stack_meta(
    image: ccdp.CCDData,
    stack_meta: Mapping[str, object] | None,
    weights: np.ndarray | None,
) -> None:
    meta = dict(stack_meta or {})
    if "weighting" not in meta:
        meta["weighting"] = "none" if weights is None else "custom"
    for key, (field, comment) in STACK_META_KEYWORDS.items():
        if field not in meta or meta[field] is None:
            continue
        value = meta[field]
        if isinstance(value, str):
            image.meta[key] = (value, comment)
        elif isinstance(value, bool | np.bool_ | int | np.integer):
            image.meta[key] = (int(value), comment)
        else:
            number = float(value)
            if np.isfinite(number):
                image.meta[key] = (number, comment)
    if weights is not None and hasattr(image.meta, "add_history"):
        image.meta.add_history(
            f"Weighted stack ({meta['weighting']}): weights "
            f"{float(np.min(weights)):.3f}-{float(np.max(weights)):.3f}, n={weights.size}"
        )


def stack_filter_images(
    images_to_combine: list[str],
    stacking_method: str,
    dtype: str | np.dtype | None,
    filter_: str,
    out_path: Path,
    new_target_name: str | None,
    weights: Sequence[float] | np.ndarray | None = None,
    stack_meta: Mapping[str, object] | None = None,
) -> str:
    """Combine images for one filter and write the stacked file.

    ``weights`` are per-image (1-D, same order as ``images_to_combine``);
    see :func:`prepare_stack_weights`. ``stack_meta`` fields are written as
    header keywords (:data:`STACK_META_KEYWORDS`). ``N-IMAGES`` is the
    number of images actually combined.
    """
    out_path = Path(out_path)
    weight_array = prepare_stack_weights(
        weights, stacking_method, len(images_to_combine), label=f"filter {filter_}"
    )
    combined_image = ccdp.combine(
        images_to_combine,
        method=stacking_method,
        weights=weight_array,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=15e9,
        dtype=dtype,
    )
    utilities.update_header_information(
        combined_image,
        len(images_to_combine),
        new_target_name,
    )
    _write_stack_meta(combined_image, stack_meta, weight_array)
    file_name = "combined_filter_{}.fit".format(filter_.replace("''", "p"))
    combined_image.write(out_path / file_name, overwrite=True)
    return file_name


def weights_for_files(
    files: Sequence[str],
    *,
    quality_table: Table | None = None,
    label: str = "",
    indent: int = 2,
) -> np.ndarray:
    """Per-file stack weights from the ``FRMWGHT`` header keyword.

    Frames without the keyword fall back to ``stack_weight`` in
    ``quality_table`` (matched by basename), else to ``1.0`` with a warning.
    """
    lookup: dict[str, float] = {}
    if quality_table is not None and len(quality_table):
        for name, weight in zip(quality_table["file"], quality_table["stack_weight"], strict=True):
            value = float(weight)
            if np.isfinite(value):
                lookup[Path(str(name)).name] = value
    weights = np.ones(len(files), dtype=float)
    missing: list[str] = []
    for i, file_name in enumerate(files):
        header_weight = frame_weight(fits.getheader(file_name))
        if header_weight is not None:
            weights[i] = header_weight
            continue
        base = Path(str(file_name)).name
        if base in lookup:
            weights[i] = lookup[base]
        else:
            missing.append(base)
    if missing:
        terminal_output.print_to_terminal(
            f"{len(missing)} frame(s){' in ' + label if label else ''} without a stack "
            f"weight (FRMWGHT) use weight 1: {', '.join(missing[:5])}"
            f"{', …' if len(missing) > 5 else ''}",
            indent=indent,
            style_name="WARNING",
        )
    return weights


def stack_meta_for_filter(
    quality_table: Table | None,
    filter_: str,
    *,
    weighting: str,
    stacked_files: Sequence[str] | None = None,
) -> dict[str, object]:
    """Header bookkeeping for one filter from the quality table."""
    meta: dict[str, object] = {"weighting": weighting}
    if quality_table is None or len(quality_table) == 0:
        return meta
    filters = np.asarray([str(f) for f in quality_table["filter"]])
    idx = np.flatnonzero(filters == str(filter_))
    if idx.size == 0:
        return meta
    rejected = np.asarray(quality_table["rejected"], dtype=bool)[idx]
    aligned = np.asarray(quality_table["aligned"], dtype=bool)[idx]
    fwhm = np.asarray(quality_table["fwhm_px"], dtype=float)[idx]
    meta["n_frames_total"] = int(idx.size)
    meta["n_rejected"] = int(np.count_nonzero(rejected))
    if np.any(aligned):
        meta["n_align_failed"] = int(np.count_nonzero(~rejected & ~aligned))
    if stacked_files is not None:
        names = {Path(str(f)).name for f in stacked_files}
        used = np.asarray([Path(str(f)).name in names for f in quality_table["file"][idx]])
    else:
        used = ~rejected
    finite = used & np.isfinite(fwhm)
    if np.any(finite):
        meta["fwhm_median"] = float(np.median(fwhm[finite]))
        meta["fwhm_max"] = float(np.max(fwhm[finite]))
    return meta


def stack_image(
    image_path: Path,
    output_dir: Path,
    image_type_list: list[str],
    stacking_method: str = "average",
    dtype: str | np.dtype | None = None,
    new_target_name: str | None = None,
    debug: bool = False,
    n_cores_multiprocessing: int | None = None,
    stack_weighting: str = "none",
    quality_table: Table | None = None,
    keep_input_frames: bool = False,
) -> None:
    """
    Combine images

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type_list
        Header keyword characterizing the image type for which the
        shifts shall be determined

    stacking_method
        Method used for combining the images.
        Possibilities: ``median`` or ``average`` or ``sum``
        Default is ``average`.

    dtype
        The dtype that should be used while combining the images.
        Default is ''None''. -> None is equivalent to float64

    new_target_name
        Name of the target. If not None, this target name will be written
        to the FITS header.
        Default is ``None``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    n_cores_multiprocessing
        Worker processes (one per filter). Default is ``None``.

    stack_weighting
        Weighting scheme, see
        :data:`~ost_photometry.reduce.frame_selection.STACK_WEIGHTING`.
        Weights are read from the ``FRMWGHT`` header keyword of every frame
        (fallback: ``quality_table``). Default is ``none``.

    quality_table
        Frame-quality table for header bookkeeping and weight fallback.
        Default is ``None``.

    keep_input_frames
        Keep the aligned input frames after stacking (also implied by
        ``debug``). Default is ``False``.
    """
    terminal_output.print_to_terminal("Stack light images...", indent=2)
    if stack_weighting not in SUPPORTED_STACK_WEIGHTING:
        raise ValueError(
            f"stack_weighting must be one of {SUPPORTED_STACK_WEIGHTING}, got {stack_weighting!r}"
        )

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   New image collection for the images
    image_file_collection = utilities.image_file_collection(file_path)

    #   Check if image_file_collection is not empty
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No FITS files found in {file_path}. "
            f"=> EXIT {style.Bcolors.ENDC}"
        )

    #   Determine filter
    image_type = utilities.get_image_type(
        image_file_collection,
        image_type_list,
    )
    filters: set[str] = set(
        h["filter"] for h in image_file_collection.headers(imagetyp=image_type)
    )

    filter_jobs: list[tuple[str, list[str], np.ndarray | None, dict[str, object]]] = []
    for filter_ in sorted(filters):
        images_to_combine = image_file_collection.files_filtered(
            imagetyp=image_type,
            filter=filter_,
            include_path=True,
        )
        if not images_to_combine:
            continue
        weights = None
        if stack_weighting != "none":
            weights = weights_for_files(
                images_to_combine, quality_table=quality_table, label=f"filter {filter_}"
            )
        meta = stack_meta_for_filter(
            quality_table, filter_, weighting=stack_weighting, stacked_files=images_to_combine
        )
        filter_jobs.append((filter_, list(images_to_combine), weights, meta))

    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(filter_jobs),
        add_progress_bar=True,
    )
    for filter_, images_to_combine, weights, meta in filter_jobs:
        executor.schedule(
            stack_filter_images,
            args=(images_to_combine, stacking_method, dtype, filter_, out_path, new_target_name),
            kwargs={"weights": weights, "stack_meta": meta},
        )

    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Stacking light images using multiprocessing"
            f" failed :({style.Bcolors.ENDC}"
        )
    executor.wait()

    for filter_, images_to_combine, weights, _meta in filter_jobs:
        text = f"Stacked {len(images_to_combine)} frames in filter {filter_}"
        if weights is not None and not np.allclose(weights, weights[0]):
            text += f" ({stack_weighting} weights {weights.min():.2f}-{weights.max():.2f})"
        terminal_output.print_to_terminal(text, indent=2)

    #   Remove individual reduced images
    if not debug and not keep_input_frames:
        shutil.rmtree(file_path, ignore_errors=True)


__all__ = [
    "STACK_META_KEYWORDS",
    "prepare_stack_weights",
    "stack_filter_images",
    "stack_image",
    "stack_meta_for_filter",
    "weights_for_files",
]
