"""Align images by reprojecting onto a shared celestial WCS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.wcs import WCS

from ... import terminal_output
from ... import utilities as base_utilities
from ...fits_headers import wcs_from_header
from ...wcs import find_wcs_for_image, wcs_maps_distinct_sky_positions


def celestial_wcs_from_ccd(ccd: CCDData) -> WCS | None:
    """Return a celestial WCS from ``ccd``, or ``None`` if none is present."""
    wcs_obj = ccd.wcs
    if wcs_obj is None and ccd.meta is not None:
        try:
            wcs_obj = wcs_from_header(ccd.meta)
        except Exception:
            return None
    if wcs_obj is None:
        return None
    try:
        if not wcs_obj.is_celestial:
            return None
    except Exception:
        return None
    return wcs_obj


def pixel_offset_on_reference(
    src_wcs: WCS,
    src_shape: tuple[int, ...],
    dst_wcs: WCS,
) -> tuple[float, float]:
    """
    Pixel of the source-image centre on the destination grid, minus the
    source-centre pixel. Same-shape frames: this is the bulk (dx, dy) shift.

    ``(dx, dy)`` uses FITS/WCS pixel axes (x along NAXIS1, y along NAXIS2):
    destination pixel of the source centre minus the source-centre pixel.
    That is **not** the ``ndi.shift`` convention of “how much to shift the
    other image onto the reference”. A source that sits 3 pixels left of
    the reference reports ``dx = +3`` here, because its centre maps to a
    larger x on the destination grid. YAML ``dx_pix`` / ``dy_pix`` follow
    this definition so they stay consistent with the WCS path; do not flip
    the sign silently.
    """
    ny, nx = int(src_shape[0]), int(src_shape[1])
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0
    sky = src_wcs.pixel_to_world(x0, y0)
    x1, y1 = dst_wcs.world_to_pixel(sky)
    return float(x1 - x0), float(y1 - y0)


def fits_has_celestial_wcs(path: str | Path) -> bool:
    """True if the primary HDU has a celestial WCS that maps corners apart."""
    path = Path(path)
    with fits.open(path) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    try:
        wcs_obj = wcs_from_header(header)
    except Exception:
        return False
    if not wcs_obj.is_celestial:
        return False
    if data is None:
        return False
    ny, nx = data.shape
    return wcs_maps_distinct_sky_positions(wcs_obj, (nx, ny))


def ensure_celestial_wcs_on_fits(
    path: str | Path,
    output_dir: str | Path,
    *,
    wcs_method: str = "astap",
    force: bool = False,
    indent: int = 2,
) -> bool:
    """
    Solve and persist a celestial WCS on ``path`` if missing.

    Returns ``True`` if the file then has a celestial WCS. ASTAP writes the
    solution in place; astrometry/twirl persist via :func:`find_wcs_for_image`.
    """
    path = Path(path)
    if fits_has_celestial_wcs(path) and not force:
        return True
    if wcs_method == "twirl":
        terminal_output.print_to_terminal(
            "shift_method='wcs' cannot use wcs_method='twirl' (needs star "
            f"positions). Skip WCS solve for {path.name}; use 'astap'.",
            indent=indent,
            style_name="WARNING",
        )
        return False
    image = base_utilities.Image(0, "filter", path, output_dir)
    try:
        find_wcs_for_image(
            image,
            method=wcs_method,
            force_wcs_determination=force,
            indent=indent,
        )
    except RuntimeError as exc:
        terminal_output.print_to_terminal(
            f"WCS solve failed for {path.name}: {exc}. Skip this frame.",
            indent=indent,
            style_name="WARNING",
        )
        return False
    return fits_has_celestial_wcs(path)


def reproject_ccd_onto_wcs(
    ccd: CCDData,
    target_wcs: WCS,
    target_shape: tuple[int, int],
) -> CCDData:
    """Reproject ``ccd`` onto ``target_wcs`` / ``target_shape`` (bilinear)."""
    from ccdproc import wcs_project

    src_wcs = celestial_wcs_from_ccd(ccd)
    if src_wcs is None:
        raise ValueError("CCD has no celestial WCS to reproject from.")
    ccd.wcs = src_wcs

    projected = wcs_project(
        ccd,
        target_wcs,
        target_shape=target_shape,
        order="bilinear",
    )

    if ccd.uncertainty is not None:
        from astropy.wcs.utils import proj_plane_pixel_area
        from reproject import reproject_interp

        unc_in = np.asarray(ccd.uncertainty.array, dtype=float)
        unc_raw, _ = reproject_interp(
            (unc_in, src_wcs),
            target_wcs,
            shape_out=target_shape,
            order="bilinear",
        )
        area_ratio = proj_plane_pixel_area(target_wcs) / proj_plane_pixel_area(src_wcs)
        projected.uncertainty = StdDevUncertainty(np.asarray(unc_raw) * float(area_ratio))

    return projected


def apply_wcs_align(
    current_image_name: str,
    reference_image_name: str,
    output_path: Path,
    output_path_transformation: Path,
    *,
    modify_file_name: bool = False,
    rm_enlarged_keyword: bool = False,
    instrument: str | None = None,
    save_only_transformation: bool = False,
    wcs_method: str = "astap",
    force_wcs_determination: bool = False,
) -> None:
    """Reproject one science frame onto the reference image WCS and write it."""
    current_path = Path(current_image_name)
    reference_path = Path(reference_image_name)
    output_path = Path(output_path)
    output_path_transformation = Path(output_path_transformation)

    if not ensure_celestial_wcs_on_fits(
        current_path,
        output_path,
        wcs_method=wcs_method,
        force=force_wcs_determination,
    ):
        return
    if not ensure_celestial_wcs_on_fits(
        reference_path,
        output_path,
        wcs_method=wcs_method,
        force=force_wcs_determination,
    ):
        terminal_output.print_to_terminal(
            f"Reference image has no WCS ({reference_path.name}); cannot "
            "align with shift_method='wcs'.",
            indent=2,
            style_name="WARNING",
        )
        return

    current_ccd = CCDData.read(current_image_name)
    reference_ccd = CCDData.read(reference_image_name)
    src_wcs = celestial_wcs_from_ccd(current_ccd)
    dst_wcs = celestial_wcs_from_ccd(reference_ccd)
    if src_wcs is None or dst_wcs is None:
        terminal_output.print_to_terminal(
            f"Missing celestial WCS after solve ({current_path.name}); skip.",
            indent=2,
            style_name="WARNING",
        )
        return

    same_file = current_path.resolve() == reference_path.resolve()
    if same_file:
        output_image = current_ccd
        dx, dy = 0.0, 0.0
    else:
        try:
            output_image = reproject_ccd_onto_wcs(
                current_ccd,
                dst_wcs,
                tuple(int(n) for n in reference_ccd.data.shape),
            )
        except (ValueError, TypeError, ImportError) as exc:
            terminal_output.print_to_terminal(
                f"WCS reproject failed for {current_path.name}: {exc}. Skip.",
                indent=2,
                style_name="WARNING",
            )
            return
        dx, dy = pixel_offset_on_reference(
            src_wcs,
            current_ccd.data.shape,
            dst_wcs,
        )

    file_name = current_path.name
    if modify_file_name:
        filter_ = output_image.meta["filter"]
        file_name = "combined_trimmed_filter_{}.fit".format(
            str(filter_).replace("''", "p")
        )

    if not save_only_transformation:
        if instrument is not None and instrument != "":
            output_image.meta["INSTRUME"] = instrument
        output_image.meta["trimmed"] = True
        output_image.meta["ALIGNMTH"] = "wcs"
        if rm_enlarged_keyword and "enlarged" in output_image.meta:
            output_image.meta.remove("enlarged")
        output_image.write(output_path / file_name, overwrite=True)

    base_name = base_utilities.get_basename(file_name)
    with open(output_path_transformation / f"{base_name}.yaml", "w") as file:
        yaml.dump(
            {
                "method": "wcs",
                "dx_pix": dx,
                "dy_pix": dy,
                "source": current_path.name,
                "reference": reference_path.name,
            },
            file,
        )
    terminal_output.print_to_terminal(
        f"WCS align {current_path.name}: dx={dx:+.2f} pix, dy={dy:+.2f} pix",
        indent=2,
    )


__all__ = [
    "apply_wcs_align",
    "celestial_wcs_from_ccd",
    "ensure_celestial_wcs_on_fits",
    "fits_has_celestial_wcs",
    "pixel_offset_on_reference",
    "reproject_ccd_onto_wcs",
]
