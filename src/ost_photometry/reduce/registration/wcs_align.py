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
from ...core.pixel_masks import warn_if_mask_too_large
from ...fits_headers import wcs_from_header
from ...wcs import find_wcs_for_image, wcs_maps_distinct_sky_positions
from ..storage import cast_like
from .accounting import ApplyResult, apply_ok, apply_skipped, exception_note


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
        with terminal_output.quiet():  # one frame of many: details only
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


#: Pixel-area ratios closer to 1 than this are plate-solution noise.
AREA_RATIO_TOLERANCE = 1e-3

#: Solved frames tried as WCS donor of an unsolved frame (closest in time
#: first), and the largest scale difference of a star-pattern match between
#: two frames of the same camera setup (more is a wrong match).
WCS_DONOR_ATTEMPTS = 3
WCS_DONOR_MAX_SCALE_DIFFERENCE = 0.01


def _camera_setup(header: fits.Header) -> tuple:
    """Frames of one camera setup share their pixel grid on the sky."""
    return (
        str(header.get("INSTRUME", "")).strip(),
        int(header.get("NAXIS1", 0) or 0),
        int(header.get("NAXIS2", 0) or 0),
        str(header.get("XBINNING", 1)),
        str(header.get("YBINNING", 1)),
    )


def _header_jd(header: fits.Header) -> float:
    try:
        return float(header["JD"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        from astropy.time import Time

        return float(Time(str(header["DATE-OBS"]).strip(), format="fits").jd)
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _finite_image(path: Path) -> np.ndarray:
    data = np.asarray(fits.getdata(path, ext=0), dtype=float)
    finite = np.isfinite(data)
    if not finite.all():
        data = np.where(finite, data, np.nanmedian(data) if finite.any() else 0.0)
    return data


def transfer_wcs(path: str | Path, donor_path: str | Path) -> bool:
    """Give ``path`` the celestial WCS of ``donor_path`` (a solved frame of the
    same camera setup), mapped through the star-pattern similarity transform
    between the two frames (astroalign).

    The transform has no shear, so the axes keep the donor's scale and
    orthogonality; distortion terms (SIP) of the donor are dropped (small
    fields). Returns ``True`` when ``path`` then has a celestial WCS.
    """
    import astroalign as aa

    path, donor_path = Path(path), Path(donor_path)
    donor_header = fits.getheader(donor_path)
    donor = wcs_from_header(donor_header).celestial
    if not donor.has_celestial:
        return False
    try:
        transform, _ = aa.find_transform(_finite_image(path), _finite_image(donor_path),
                                         detection_sigma=3)
    except (aa.MaxIterError, TypeError, ValueError):
        return False
    if abs(float(transform.scale) - 1.0) > WCS_DONOR_MAX_SCALE_DIFFERENCE:
        return False

    #   donor pixel (0-based) q = M p + t of this frame's pixel p
    matrix = np.asarray(transform.params[:2, :2], dtype=float)
    shift = np.asarray(transform.params[:2, 2], dtype=float)
    solved = WCS(naxis=2)
    solved.wcs.ctype = [str(c).replace("-SIP", "") for c in donor.wcs.ctype]
    solved.wcs.cunit = donor.wcs.cunit
    solved.wcs.crval = donor.wcs.crval
    solved.wcs.cd = donor.pixel_scale_matrix @ matrix
    solved.wcs.crpix = 1.0 + np.linalg.solve(matrix, np.asarray(donor.wcs.crpix) - 1.0 - shift)
    if donor.wcs.radesys:
        solved.wcs.radesys = donor.wcs.radesys
    if np.isfinite(donor.wcs.equinox):
        solved.wcs.equinox = donor.wcs.equinox

    with fits.open(path, mode="update") as hdul:
        header = hdul[0].header
        for key in [k for k in header if k.startswith(("CD1_", "CD2_", "PC1_", "PC2_", "CDELT",
                                                        "CROTA", "CRPIX", "CRVAL", "CTYPE",
                                                        "CUNIT", "A_", "B_", "AP_", "BP_"))]:
            header.remove(key, ignore_missing=True, remove_all=True)
        header.update(solved.to_header())
        header["WCSSRC"] = (donor_path.name[:68], "WCS of this frame via star pattern")
    return fits_has_celestial_wcs(path)


def transfer_wcs_from_solved_frames(
    unsolved: list[str],
    candidates: list[str],
    *,
    indent: int = 2,
) -> tuple[list[str], list[str]]:
    """WCS for frames ASTAP could not solve, from solved frames of the same
    camera setup (:func:`transfer_wcs`, the closest in time first). Returns
    ``(rescued, failed)``.
    """
    solved = []
    for name in candidates:
        if fits_has_celestial_wcs(name):
            header = fits.getheader(name)
            solved.append((name, _camera_setup(header), _header_jd(header)))
    rescued: list[str] = []
    failed: list[str] = []
    for name in unsolved:
        header = fits.getheader(name)
        setup, jd = _camera_setup(header), _header_jd(header)
        donors = sorted((s for s in solved if s[1] == setup),
                        key=lambda s: abs(s[2] - jd) if np.isfinite(s[2] - jd) else np.inf)
        if any(transfer_wcs(name, donor) for donor, _setup, _jd in donors[:WCS_DONOR_ATTEMPTS]):
            rescued.append(name)
        else:
            failed.append(name)
    if unsolved:
        terminal_output.print_to_terminal(
            f"{len(unsolved)} frame(s) without a plate solution: {len(rescued)} got the WCS "
            "of a solved frame of the same camera via the star pattern"
            + (f", {len(failed)} skipped (no solved frame of the same camera setup, or no "
               "star match)" if failed else ""),
            indent=indent,
            style_name="WARNING" if failed else "NORMAL",
        )
    return rescued, failed


def reproject_ccd_onto_wcs(
    ccd: CCDData,
    target_wcs: WCS,
    target_shape: tuple[int, int],
) -> CCDData:
    """Reproject ``ccd`` onto ``target_wcs`` / ``target_shape`` (bilinear)."""
    from ccdproc import wcs_project
    from reproject import reproject_interp

    from ...core.pixel_masks import fill_masked_pixels, resample_mask, resample_uncertainty

    src_wcs = celestial_wcs_from_ccd(ccd)
    if src_wcs is None:
        raise ValueError("CCD has no celestial WCS to reproject from.")
    ccd.wcs = src_wcs

    #   Bilinear resampling smears every masked defect into its neighbours,
    #   so fill defects with the local median first. The mask is resampled
    #   separately below; ``wcs_project`` would flag every output pixel with
    #   *any* masked contribution (``> 1e-8``) and grow the mask several-fold.
    source_mask = None if ccd.mask is None else np.asarray(ccd.mask, dtype=bool)
    if source_mask is not None and source_mask.any():
        ccd = ccd.copy()
        ccd.data = fill_masked_pixels(ccd.data, source_mask)
        ccd.mask = None

    projected = wcs_project(
        ccd,
        target_wcs,
        target_shape=target_shape,
        order="bilinear",
    )

    def _nearest(values: np.ndarray) -> np.ndarray:
        out, _ = reproject_interp(
            (values, src_wcs),
            target_wcs,
            shape_out=target_shape,
            order="nearest-neighbor",
        )
        return out

    footprint = ~np.isfinite(np.asarray(projected.data, dtype=float))
    warped_mask = resample_mask(source_mask, _nearest)
    projected.mask = footprint if warped_mask is None else (footprint | warped_mask)

    from astropy.wcs.utils import proj_plane_pixel_area

    #   wcs_project scales the data by the pixel-area ratio k (flux per pixel);
    #   σ follows as √k·σ, and the per-pixel noise keywords with it. Within
    #   AREA_RATIO_TOLERANCE k is the noise of the plate solutions (same
    #   binning): undone, so frames do not get random flux factors.
    area_ratio = float(proj_plane_pixel_area(target_wcs) / proj_plane_pixel_area(src_wcs))
    if abs(area_ratio - 1.0) < AREA_RATIO_TOLERANCE:
        projected.data = np.asarray(projected.data) / area_ratio
        area_ratio = 1.0
    if ccd.uncertainty is not None:

        def _bilinear(values: np.ndarray) -> np.ndarray:
            out, _ = reproject_interp(
                (values, src_wcs),
                target_wcs,
                shape_out=target_shape,
                order="bilinear",
            )
            return out

        projected.uncertainty = StdDevUncertainty(
            resample_uncertainty(
                np.asarray(ccd.uncertainty.array, dtype=float),
                source_mask,
                _bilinear,
                area_ratio=area_ratio,
            )
        )
    scale_noise_keywords(projected.meta, area_ratio)

    return projected


def scale_noise_keywords(meta, area_ratio: float) -> None:
    """``SATLEVEL`` × k and ``RDNOISE`` × √k after resampling by area ratio k.

    Pixel values scale by k, so saturation does; the read noise follows the
    σ map convention (√k, correct for aperture sums).
    """
    k = float(area_ratio)
    if not np.isfinite(k) or k <= 0 or abs(k - 1.0) < 1e-6:
        return
    if "SATLEVEL" in meta:
        meta["SATLEVEL"] = float(meta["SATLEVEL"]) * k
    if "RDNOISE" in meta:
        meta["RDNOISE"] = float(meta["RDNOISE"]) * np.sqrt(k)
    meta["PIXAREAR"] = (k, "Pixel area ratio of the resampling (new/old)")


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
) -> ApplyResult:
    """Reproject one science frame onto the reference image WCS and write it.

    Returns ``(basename, success, note)`` for the alignment accounting.
    """
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
        return apply_skipped(current_path, "no celestial WCS")
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
        return apply_skipped(current_path, "reference has no celestial WCS")

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
        return apply_skipped(current_path, "missing celestial WCS after solve")

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
            return apply_skipped(current_path, exception_note(exc))
        dx, dy = pixel_offset_on_reference(
            src_wcs,
            current_ccd.data.shape,
            dst_wcs,
        )

    warn_if_mask_too_large(
        getattr(output_image, "mask", None),
        label=f"WCS align {current_path.name}",
        footprint=~np.isfinite(np.asarray(output_image.data, dtype=float)),
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
        cast_like(output_image, current_ccd)
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
        style_name="DETAIL",
    )
    return apply_ok(current_path)


__all__ = [
    "WCS_DONOR_ATTEMPTS",
    "apply_wcs_align",
    "celestial_wcs_from_ccd",
    "ensure_celestial_wcs_on_fits",
    "fits_has_celestial_wcs",
    "pixel_offset_on_reference",
    "reproject_ccd_onto_wcs",
    "transfer_wcs",
    "transfer_wcs_from_solved_frames",
]
