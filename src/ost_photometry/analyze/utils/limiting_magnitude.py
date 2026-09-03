"""Limiting-magnitude determination (Image path and epoch-native table path)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.table import Table
from photutils.utils import ImageDepth

from ... import terminal_output
from ...core.parallel import start_plot_process
from ...output_layout import extraction_plot_dir
from .. import plots
from ..post_processing.adapters import ensure_epoch_native_photometry_table
from ..post_processing.imaging import ImagingPlotContext

if TYPE_CHECKING:
    from ..observation import Observation


def _subset_photometry_by_epoch(tbl: Table, epoch_id: str | None) -> Table:
    """Return rows for one epoch, or the full table if no ``epoch_id`` column."""
    if "epoch_id" not in tbl.colnames:
        return tbl
    col = tbl["epoch_id"]
    ids = np.unique(np.asarray(col).astype(str))
    if epoch_id is not None:
        return tbl[np.asarray(col).astype(str) == str(epoch_id)]
    if len(ids) == 1:
        return tbl
    raise ValueError(
        "Photometry table has multiple epoch_id values; pass epoch_id=... "
        f"(available: {list(ids)!r})."
    )


def _resolve_limiting_mag_column(tbl: Table, filter_: str) -> str:
    """Pick magnitude column (``mag_cal_*``, ``mag_inst_*``, or wide-table names)."""
    mc = f"mag_cal_{filter_}"
    mi = f"mag_inst_{filter_}"
    if mc in tbl.colnames:
        return mc
    if mi in tbl.colnames:
        return mi
    if "mag_cali_trans" in tbl.colnames:
        return "mag_cali_trans"
    if "mag_cali_no-trans" in tbl.colnames:
        return "mag_cali_no-trans"
    raise ValueError(
        f"No magnitude column for filter {filter_!r}: expected {mc!r}, {mi!r}, or "
        "'mag_cali_trans' / 'mag_cali_no-trans'."
    )


def _sort_table_by_magnitude(tbl: Table, magnitude_col: str) -> Table:
    out = tbl.copy()
    out.sort(magnitude_col)
    return out


def _plausible_magnitude_mask(tbl: Table, magnitude_col: str) -> np.ndarray:
    col = tbl[magnitude_col]
    if hasattr(col, "unit") and col.unit is not None:
        return np.asarray(col < 30 * u.mag)
    return np.asarray(np.asarray(col, dtype=float) < 30.0)


def _pixel_indices_for_depth_mask(tbl: Table) -> tuple[np.ndarray, np.ndarray]:
    if "x_fit" in tbl.colnames and "y_fit" in tbl.colnames:
        xs = tbl["x_fit"]
        ys = tbl["y_fit"]
    elif "x" in tbl.colnames and "y" in tbl.colnames:
        xs = tbl["x"]
        ys = tbl["y"]
    else:
        raise ValueError(
            "Table needs ('x', 'y') or ('x_fit', 'y_fit') for limiting magnitude mask."
        )
    if hasattr(xs, "value"):
        xs = xs.value
    if hasattr(ys, "value"):
        ys = ys.value
    return np.rint(np.asarray(xs)).astype(int), np.rint(np.asarray(ys)).astype(int)


def _blank_sky_source_mask(
    image_data: np.ndarray,
    index_x: np.ndarray,
    index_y: np.ndarray,
    *,
    catalog_stamp_radius: float = 3.0,
    detect_nsigma: float = 3.0,
    detect_npixels: int = 5,
) -> np.ndarray:
    """
    Boolean source mask for :class:`~photutils.utils.ImageDepth` (True = avoid).

    Combines circular stamps at known photometry positions with a segmentation
    mask of sources detected on the image. Single centroid pixels alone are not
    enough: blank apertures would otherwise land on PSF wings and on objects
    missing from the photometry table.
    """
    from astropy.convolution import convolve
    from astropy.stats import sigma_clipped_stats
    from photutils.segmentation import SourceFinder, make_2dgaussian_kernel
    from photutils.utils.footprints import circular_footprint
    from scipy.ndimage import binary_dilation

    data = np.asarray(image_data, dtype=float)
    finite = np.isfinite(data)
    stats_mask = ~finite if not np.all(finite) else None
    mask = np.zeros(data.shape, dtype=bool)

    xs = np.asarray(index_x, dtype=int).ravel()
    ys = np.asarray(index_y, dtype=int).ravel()
    if xs.size:
        seed = np.zeros(data.shape, dtype=bool)
        inside = (
            (ys >= 0)
            & (ys < data.shape[0])
            & (xs >= 0)
            & (xs < data.shape[1])
        )
        seed[ys[inside], xs[inside]] = True
        r_stamp = max(1, int(np.ceil(float(catalog_stamp_radius))))
        mask |= binary_dilation(seed, structure=circular_footprint(radius=r_stamp))

    mask |= ~finite

    try:
        _mean, median, std = sigma_clipped_stats(
            data, mask=stats_mask, sigma=3.0, maxiters=5
        )
        if np.isfinite(std) and std > 0.0:
            kernel = make_2dgaussian_kernel(3.0, size=5)
            working = np.where(finite, data - median, 0.0)
            convolved = convolve(working, kernel, normalize_kernel=True)
            finder = SourceFinder(n_pixels=detect_npixels, progress_bar=False)
            segm = finder(convolved, float(detect_nsigma) * float(std))
            if segm is not None:
                mask |= segm.make_source_mask()
    except Exception as exc:  # noqa: BLE001 — fall back to catalog stamps only
        terminal_output.print_to_terminal(
            f"Limiting-mag source detection for blank-sky mask failed ({exc}); "
            "using photometry positions only.",
            style_name="WARNING",
            indent=3,
        )

    return mask


def _image_and_mask_for_depth(
    image_data: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero non-finite pixels and mask them so ImageDepth does not clip NaNs."""
    data = np.asarray(image_data, dtype=float)
    mask_out = np.asarray(mask, dtype=bool)
    if mask_out.shape != data.shape:
        raise ValueError(
            f"ImageDepth mask shape {mask_out.shape} does not match "
            f"image shape {data.shape}."
        )
    finite = np.isfinite(data)
    if np.all(finite):
        return data, mask_out
    data = data.copy()
    data[~finite] = 0.0
    return data, mask_out | ~finite


def _median_zeropoint(zp) -> float:
    if zp is None:
        raise ValueError("Zero point (image.zp or zeropoint=...) is required.")
    z = np.median(np.asarray(zp))
    return float(z.value) if hasattr(z, "value") else float(z)


def _derive_limiting_magnitude_one_epoch(
    *,
    image_data: np.ndarray,
    out_path_stub: str,
    filter_: str,
    wcs_image,
    photo: Table,
    magnitude_col: str,
    pixel_scale: float,
    zeropoint: float,
    aperture_radius: float,
    radii_unit: str,
    file_type_plots: str,
    use_wcs_projection_for_star_maps: bool,
    indent: int,
    rts: str,
    image_depth_mag_offset: float = 0.0,
) -> None:
    tbl_sorted = _sort_table_by_magnitude(photo, magnitude_col)
    mask_mag = _plausible_magnitude_mask(tbl_sorted, magnitude_col)
    tbl_mag = tbl_sorted[mask_mag]

    n_take = min(10, len(tbl_mag))
    tbl_faintest = tbl_mag[-n_take:] if n_take else tbl_mag

    start_plot_process(
        plots.starmap,
        (
            str(extraction_plot_dir(out_path_stub)),
            image_data,
            filter_,
            tbl_faintest,
        ),
        {
            "label": f"{n_take} faintest objects" if n_take else "faintest objects",
            "rts": rts,
            "mode": "mags",
            "magnitude_column": magnitude_col,
            "wcs_image": wcs_image,
            "use_wcs_projection": use_wcs_projection_for_star_maps,
            "file_type": file_type_plots,
        },
    )

    terminal_output.print_to_terminal("")
    terminal_output.print_to_terminal(
        f"Determine limiting magnitude for filter: {filter_}",
        indent=indent,
    )
    terminal_output.print_to_terminal(
        "Based on detected objects:",
        indent=indent * 2,
    )
    if n_take == 0:
        terminal_output.print_to_terminal(
            "No stars passed the magnitude plausibility cut (< 30 mag).",
            indent=indent * 3,
            style_name="WARNING",
        )
    else:
        mcol = tbl_faintest[magnitude_col]
        marr = np.asarray(mcol.value if hasattr(mcol, "value") else mcol, dtype=float)
        median_faintest_objects = np.median(marr)
        mean_faintest_objects = np.mean(marr)
        terminal_output.print_to_terminal(
            f"Median of the {n_take} faintest objects: "
            f"{median_faintest_objects:.1f} mag",
            indent=indent * 3,
            style_name="OKBLUE",
        )
        terminal_output.print_to_terminal(
            f"Mean of the {n_take} faintest objects: "
            f"{mean_faintest_objects:.1f} mag",
            indent=indent * 3,
            style_name="OKBLUE",
        )

    # Mask all known photometry positions (not only mag < 30), then add a
    # segmentation mask so blank apertures avoid objects missing from the table.
    index_x, index_y = _pixel_indices_for_depth_mask(photo)
    radius = aperture_radius
    if radii_unit == "arcsec":
        if pixel_scale is None:
            raise ValueError(
                "radii_unit='arcsec' requires a known pixel_scale (arcsec/pixel)."
            )
        radius = radius / pixel_scale

    stamp_r = max(3.0, 0.5 * float(radius))
    mask = _blank_sky_source_mask(
        image_data,
        index_x,
        index_y,
        catalog_stamp_radius=stamp_r,
    )
    mask_pad = max(5.0, 0.5 * float(radius))

    depth_data, depth_mask = _image_and_mask_for_depth(image_data, mask)
    n_nonfinite = int(np.count_nonzero(~np.isfinite(np.asarray(image_data, dtype=float))))
    if n_nonfinite > 0:
        terminal_output.print_to_terminal(
            f"Masked {n_nonfinite} non-finite pixels before ImageDepth.",
            style_name="WARNING",
            indent=indent * 2,
        )

    depth = ImageDepth(
        radius,
        n_sigma=5.0,
        n_apertures=500,
        n_iters=2,
        overlap=False,
        mask_pad=mask_pad,
        zeropoint=zeropoint,
        progress_bar=False,
    )

    _flux_limit, mag_limit = depth(depth_data, depth_mask)

    start_plot_process(
        plots.plot_limiting_mag_sky_apertures,
        (out_path_stub, depth_data, depth_mask, depth),
        {"file_type": file_type_plots},
    )

    mag_report = float(mag_limit) + image_depth_mag_offset

    terminal_output.print_to_terminal(
        "Based on the ImageDepth (photutils) routine:",
        indent=indent * 2,
    )
    if image_depth_mag_offset != 0.0:
        terminal_output.print_to_terminal(
            f"(limit shifted by {image_depth_mag_offset:+.3f} mag to match calibrated table)",
            indent=indent * 2,
            style_name="INFO",
        )
    terminal_output.print_to_terminal(
        f"500 apertures, 5 sigma, 2 iterations: "
        f"{mag_report:6.2f} mag",
        indent=indent * 3,
        style_name="OKBLUE",
    )


def derive_limiting_magnitude(
    observation: Observation | None = None,
    filter_list: list[str] | None = None,
    reference_image_index: int = 0,
    aperture_radius: float = 4.0,
    radii_unit: str = "arcsec",
    file_type_plots: str = "pdf",
    use_wcs_projection_for_star_maps: bool = True,
    indent: int = 1,
    *,
    photometry_table: Table | None = None,
    epoch_id: str | None = None,
    imaging_context: ImagingPlotContext | None = None,
    pixel_scale: float | None = None,
    zeropoint: float | None = None,
    image_depth_mag_offset: float = 0.0,
) -> None:
    """
    Determine limiting magnitude.

    Two input styles:

    1. **Legacy (per-image photometry on ``Observation``):** pass ``observation``
       and ``filter_list``; uses ``image_series_dict[filter].image_list[id]``.

    2. **Epoch-native table + imaging context:** pass ``photometry_table`` (long
       form with ``mag_cal_<filter>`` and optional ``epoch_id``),
       ``imaging_context``, ``filter_list``, and ``pixel_scale`` / ``zeropoint``.
    """
    if photometry_table is not None:
        if filter_list is None:
            raise TypeError("filter_list is required when photometry_table is set.")
        if imaging_context is None:
            raise TypeError(
                "imaging_context=... is required when photometry_table is set "
                "(needs reference image array, WCS, and out_path_stub)."
            )
        if pixel_scale is None or zeropoint is None:
            raise TypeError(
                "pixel_scale and zeropoint are required when photometry_table is set."
            )
        tbl = ensure_epoch_native_photometry_table(photometry_table)
        photo_epoch = _subset_photometry_by_epoch(tbl, epoch_id)
        image_data = np.asarray(imaging_context.reference_image)
        out_stub = str(imaging_context.out_path_stub)
        wcs_obj = imaging_context.wcs
        for filter_ in filter_list:
            magnitude_col = _resolve_limiting_mag_column(photo_epoch, filter_)
            rts = (
                f"faintest objects, {epoch_id}"
                if epoch_id is not None
                else "faintest objects"
            )
            _derive_limiting_magnitude_one_epoch(
                image_data=image_data,
                out_path_stub=out_stub,
                filter_=filter_,
                wcs_image=wcs_obj,
                photo=photo_epoch,
                magnitude_col=magnitude_col,
                pixel_scale=pixel_scale,
                zeropoint=zeropoint,
                aperture_radius=aperture_radius,
                radii_unit=radii_unit,
                file_type_plots=file_type_plots,
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
                indent=indent,
                rts=rts,
                image_depth_mag_offset=image_depth_mag_offset,
            )
        return

    if observation is None or filter_list is None:
        raise TypeError(
            "Provide observation=... and filter_list=..., or photometry_table=... "
            "with imaging_context=..., pixel_scale=..., zeropoint=...."
        )

    image_series_dict = observation.image_series_dict

    for filter_ in filter_list:
        image_series = image_series_dict[filter_]
        image = image_series.image_list[reference_image_index]
        photo = image.photometry
        if photo is None:
            raise ValueError(
                f"No photometry on reference image for filter {filter_!r}."
            )

        magnitude_col = _resolve_limiting_mag_column(photo, filter_)
        rts = f"faintest objects, image: {reference_image_index}"
        zp_med = _median_zeropoint(image.zp)

        _derive_limiting_magnitude_one_epoch(
            image_data=image.get_data(),
            out_path_stub=image.out_path,
            filter_=filter_,
            wcs_image=image.wcs,
            photo=photo,
            magnitude_col=magnitude_col,
            pixel_scale=image.pixel_scale,
            zeropoint=zp_med,
            aperture_radius=aperture_radius,
            radii_unit=radii_unit,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            indent=indent,
            rts=rts,
            image_depth_mag_offset=image_depth_mag_offset,
        )


__all__ = ["derive_limiting_magnitude"]
