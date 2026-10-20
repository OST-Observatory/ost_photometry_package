"""
Photometry extraction: main_extract, extract_multiprocessing, and helpers.

Extracted from analyze.py for modular reuse.
"""

import os
from collections import Counter

import astropy.units as u
import ccdproc as ccdp
import numpy as np
import numpy.ma as ma
from astropy.modeling.fitting import (
    LevMarLSQFitter,
    LMLSQFitter,
    TRFLSQFitter,
)
from astropy.nddata import CCDData, NDData
from astropy.stats import SigmaClip
from astropy.table import Table
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.background import (
    Background2D,
    LocalBackground,
    MADStdBackgroundRMS,
    MedianBackground,
    MMMBackground,
)
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import (
    EPSFBuilder,
    IterativePSFPhotometry,
    PSFPhotometry,
    SourceGrouper,
    extract_stars,
)

from .. import checks, style, terminal_output
from .. import utilities as base_utilities
from ..core.parallel import Executor, start_plot_process
from ..core.pixel_masks import (
    aperture_masked_fraction,
    fill_masked_pixels,
    fill_masked_uncertainty,
)
from ..fits_headers import (
    READ_NOISE_KEY,
    SATURATION_KEY,
    cosmics_identified,
    mark_cosmics_identified,
    normalize_cosmic_ray_removal,
)
from ..fwhm import (
    estimate_image_fwhm,
    filter_finder_table_by_fwhm_scale,
    filter_table_finite_cutouts,
    roundness_range_for_finder,
)
from ..output_layout import extraction_plot_dir, tables_dir
from . import correlate, plots, utilities
from .image import AnalysisImage
from .models import ImageSeries
from .photometry_quality import (
    BLEND_FRACTION,
    annulus_sky,
    aperture_correction,
    aperture_flux_errors,
    background_noise_factor,
    moffat_star_model,
    neighbour_flux_fraction,
    saturated_sources,
    source_disk_mask,
    total_flux_correction,
)
from .utils.epsf_selection import n_epsf_stars_to_select


class ExtractionSkipError(RuntimeError):
    """This frame cannot be extracted; multi-image runs skip it and continue."""


def _skip_extraction_result(
    image: AnalysisImage,
    err: ExtractionSkipError,
    terminal_logger: terminal_output.TerminalLog | None,
    multiprocessing: bool,
) -> tuple[int, None] | None:
    msg = f"Skipping image {image.image_id} ({image.filter_}): {err}"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(msg, style_name="WARNING")
        terminal_logger.print_to_terminal("")
    else:
        terminal_output.print_to_terminal(msg, style_name="WARNING")
    if multiprocessing:
        return image.image_id, None
    raise err


def _merge_extraction_results(
    image_series: ImageSeries,
    res: list[tuple[int, Table | None]],
) -> list[int]:
    """Attach photometry from workers; drop skipped frames. Return skipped ids."""
    by_id: dict[int, Table | None] = {}
    for img_id, tbl in res:
        by_id[int(img_id)] = tbl
    ref_idx = int(image_series.reference_image_index)
    ref_id = int(image_series.image_list[ref_idx].image_id)
    kept: list[AnalysisImage] = []
    skipped: list[int] = []
    for img in image_series.image_list:
        tbl = by_id.get(int(img.image_id))
        if tbl is None:
            skipped.append(int(img.image_id))
            continue
        img.photometry = tbl
        kept.append(img)
    if not kept:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Extraction failed for all "
            f"{image_series.filter_} images.{style.Bcolors.ENDC}"
        )
    image_series.image_list = kept
    for i, img in enumerate(kept):
        if int(img.image_id) == ref_id:
            image_series.reference_image_index = i
            break
    else:
        image_series.reference_image_index = 0
        terminal_output.print_to_terminal(
            f"Reference image {ref_id} was skipped; "
            f"using image {kept[0].image_id} as the new reference.",
            style_name="WARNING",
        )
    image_series.reference_image = kept[image_series.reference_image_index]
    return skipped


def _extraction_qc_dir(image: AnalysisImage, *, gallery: bool) -> str:
    return str(extraction_plot_dir(image.out_path, gallery=gallery))


#: Fallbacks for images without ``RDNOISE`` / ``SATLEVEL`` in the header.
DEFAULT_COSMIC_READ_NOISE = 8.0
DEFAULT_COSMIC_SATURATION = 65535.0


def estimate_pixel_error(data: np.ndarray, unit, exposure_time: float) -> np.ndarray:
    """Per-pixel σ for an image without an uncertainty extension.

    ``σ² = σ_sky² + S/t`` for e-/s images (``S`` for electrons): the sky,
    read and dark noise come from the clipped scatter of the frame itself
    (the analysis frames are background-subtracted, so the pixel values no
    longer contain the sky photons), the source photons from the positive
    signal. ADU or unknown units cannot be converted and raise.
    """
    from astropy.stats import sigma_clipped_stats

    values = np.asarray(data, dtype=float)
    _mean, _median, sky_sigma = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    signal = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    if unit == u.electron / u.s:
        t = float(exposure_time)
        if not np.isfinite(t) or t <= 0:
            raise ValueError("estimate_pixel_error: e-/s image needs a positive EXPTIME.")
        source_var = signal / t
    elif unit == u.electron:
        source_var = signal
    else:
        raise ValueError(
            f"estimate_pixel_error: cannot derive photon noise for unit {unit!r}; "
            "reduce the images (e- or e-/s with an UNCERT extension) first."
        )
    return np.sqrt(float(sky_sigma) ** 2 + source_var)


def _pixel_error(image: AnalysisImage, ccd=None) -> np.ndarray:
    """``UNCERT`` of the image, else :func:`estimate_pixel_error` with a warning."""
    ccd = image.read_image() if ccd is None else ccd
    if ccd.uncertainty is not None:
        return np.asarray(ccd.uncertainty.array, dtype=float)
    terminal_output.print_to_terminal(
        f"No uncertainty extension in {image.path}; pixel errors estimated from "
        "the frame's sky scatter and the source photons.",
        indent=2,
        style_name="WARNING",
    )
    exposure = float(ccd.meta.get("EXPTIME", float("nan")))
    return estimate_pixel_error(ccd.data, ccd.unit, exposure)


def _source_mask_radius(image: AnalysisImage, annulus: CircularAnnulus) -> float:
    """Radius masked around detected stars for sky estimates: 2 FWHM, but
    inside the annulus inner radius so a star never masks its own sky."""
    fwhm = float(getattr(image, "fwhm", 4.0) or 4.0)
    r_in = float(np.atleast_1d(annulus.r_in)[0])
    return max(1.5, min(2.0 * fwhm, 0.9 * r_in))


def _saturated_mask(
    image: AnalysisImage, x: np.ndarray, y: np.ndarray, radius: float, fraction: float
) -> tuple[np.ndarray | None, CCDData]:
    """Saturation on the frame with its sky, in the unit of SATLEVEL
    (electrons at EXPTIME): ``(mask or None without SATLEVEL, raw frame)``."""
    raw_path = getattr(image, "path_before_background", None) or image.path
    raw = CCDData.read(raw_path)
    sat_level = raw.meta.get("SATLEVEL")
    if sat_level is None:
        return None, raw
    counts = np.asarray(raw.data, dtype=float)
    if raw.unit == u.electron / u.s:
        counts = counts * float(raw.meta.get("EXPTIME", 1.0))
    return saturated_sources(counts, x, y, radius, float(sat_level), fraction=fraction), raw


def _extraction_quality(
    image: AnalysisImage,
    radii_pix: tuple[float, float, float],
    *,
    saturation_fraction: float = 0.9,
    terminal_logger: terminal_output.TerminalLog | None = None,
) -> None:
    """Saturation flags and the per-image aperture correction.

    ``radii_pix`` is ``(measurement radius, annulus inner, annulus outer)``;
    for PSF photometry the measurement radius only defines where saturation
    is looked for. The aperture correction scales ``flux_fit`` /
    ``flux_err`` to the flux inside the annulus inner radius, measured on
    bright, isolated, unsaturated, unmasked stars of this image.
    """
    phot = image.photometry
    x = np.asarray(phot["x_fit"], dtype=float)
    y = np.asarray(phot["y_fit"], dtype=float)
    r_meas, r_in, r_out = (float(r) for r in radii_pix)

    def _log(msg: str, style_name: str = "NORMAL") -> None:
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg, indent=3, style_name=style_name)
        else:
            terminal_output.print_to_terminal(msg, indent=3, style_name=style_name)

    saturated, raw = _saturated_mask(image, x, y, r_meas, saturation_fraction)
    if saturated is None:
        _log("No SATLEVEL in the header: saturation not checked.", "WARNING")
        saturated = np.zeros(len(phot), dtype=bool)
    elif np.any(saturated):
        _log(f"{int(np.count_nonzero(saturated))} saturated star(s) flagged.")
    phot["saturated"] = saturated
    flat_state = str(raw.meta.get("FLATCORR", "")).strip().lower()
    phot["no_flat"] = np.full(len(phot), flat_state in ("none", "partial"))
    if flat_state in ("none", "partial"):
        _log(
            "Image reduced without flat field (FLATCORR = "
            f"{flat_state}): magnitudes keep the vignetting; flagged no_flat.",
            "WARNING",
        )

    #   Aperture correction to the annulus inner radius.
    ccd = image.read_image()
    data = np.asarray(ccd.data, dtype=float)
    sigma = _pixel_error(image, ccd)
    no_coverage = ~np.isfinite(data)
    masked = (
        np.asarray(phot["aperture_masked_fraction"], dtype=float) > 0.0
        if "aperture_masked_fraction" in phot.colnames
        else np.zeros(len(phot), dtype=bool)
    )
    flux = np.asarray(getattr(phot["flux_fit"], "value", phot["flux_fit"]), dtype=float)
    err = np.asarray(getattr(phot["flux_err"], "value", phot["flux_err"]), dtype=float)
    usable = ~saturated & ~masked
    coverage_mask = no_coverage if no_coverage.any() else None
    source_mask = getattr(image, "source_mask", None)
    #   To the total flux (growth curve to 6 FWHM, Moffat extrapolation): the
    #   light outside a fixed radius depends on the seeing, so a correction
    #   to r_in leaves seeing-dependent offsets between images that share a
    #   zero point. Too few isolated stars: to r_in, with a warning.
    fwhm = float(getattr(image, "fwhm", float("nan")) or float("nan"))
    total = (
        total_flux_correction(data, sigma, x, y, flux, err, fwhm, usable=usable,
                              mask=coverage_mask, source_mask=source_mask)
        if np.isfinite(fwhm) and fwhm > 0 else {"value": float("nan"), "n": 0}
    )
    if np.isfinite(total["value"]):
        value, error, n_used = total["value"], total["error"], total["n"]
        target = "total flux"
        detail = f", {100 * total['beyond']:.2f} % beyond 6 FWHM (Moffat β = {total['beta']:.2f})"
    else:
        value, error, n_used = aperture_correction(
            data, sigma, x, y, flux, err,
            r_large=r_in, r_in=r_in, r_out=r_out,
            usable=usable, isolation_radius=r_out + r_meas,
            mask=coverage_mask, source_mask=source_mask,
        )
        target = f"r = {r_in:.1f} px"
        detail = ""
        if np.isfinite(value):
            _log(
                f"Only {total['n']} isolated bright stars for the growth curve: aperture "
                f"correction to r = {r_in:.1f} px only (not to the total flux).",
                "WARNING",
            )
    if np.isfinite(value):
        phot["flux_fit"] = phot["flux_fit"] * value
        phot["flux_err"] = phot["flux_err"] * value
        _log(f"Aperture correction to {target}: {value:.4f} ± {error:.4f} ({n_used} stars{detail})")
    else:
        _log(
            f"Aperture correction: only {n_used} suitable stars; not applied.",
            "WARNING",
        )
    phot["aperture_correction"] = np.full(len(phot), value)
    phot["aperture_correction_err"] = np.full(len(phot), error)
    phot["aperture_correction_total"] = np.full(len(phot), target == "total flux")

    #   Light of detected neighbours inside the measurement radius (APER) or
    #   1.5 FWHM (PSF, where the neighbours are fitted, as an indicator).
    beta = total.get("beta", float("nan"))
    fraction = neighbour_flux_fraction(
        x, y, np.asarray(getattr(phot["flux_fit"], "value", phot["flux_fit"]), dtype=float),
        fwhm if np.isfinite(fwhm) and fwhm > 0 else 4.0, r_meas,
        beta=float(beta) if np.isfinite(beta) else 2.5,
    )
    phot["neighbour_flux_fraction"] = fraction
    phot["blended"] = np.isfinite(fraction) & (fraction > BLEND_FRACTION)
    n_blend = int(np.count_nonzero(phot["blended"]))
    if n_blend:
        _log(f"{n_blend} star(s) with > {100 * BLEND_FRACTION:g} % neighbour light flagged blended.")
    image.photometry = phot


def cosmic_ray_noise_parameters(
    header,
    read_noise: float | None = None,
    saturation_level: float | None = None,
) -> tuple[float, float, list[str]]:
    """Read noise and saturation for L.A.Cosmic on a reduced image.

    Explicit values win. Otherwise the header keywords ``RDNOISE`` /
    ``SATLEVEL`` written by the reduction are used (electrons per pixel for
    the image times ``EXPTIME``; for stacks already converted to the
    stacked image). Missing keywords fall back to 8 e- / 65535 with a note.
    """
    notes: list[str] = []
    if read_noise is None:
        value = header.get(READ_NOISE_KEY)
        if value is None:
            read_noise = DEFAULT_COSMIC_READ_NOISE
            notes.append(f"{READ_NOISE_KEY} not in header: read noise {read_noise} e- assumed")
        else:
            read_noise = float(value)
    if saturation_level is None:
        value = header.get(SATURATION_KEY)
        if value is None:
            saturation_level = DEFAULT_COSMIC_SATURATION
            notes.append(f"{SATURATION_KEY} not in header: saturation {saturation_level:.0f} "
                         "assumed")
        else:
            saturation_level = float(value)
    return float(read_noise), float(saturation_level), notes


def rm_cosmic_rays(
    image: AnalysisImage,
    limiting_contrast: float = 5.0,
    read_noise: float | None = None,
    sigma_clipping_value: float = 4.5,
    saturation_level: float | None = None,
    verbose: bool = False,
    add_mask: bool = True,
    terminal_logger: terminal_output.TerminalLog | None = None,
    *,
    force: bool = False,
) -> None:
    """
    Remove cosmic rays unless the FITS header already marks them identified.

    Skips lacosmic when ``CRIDENT`` / ``cosmics_rm`` / ``cosmics_msk`` /
    legacy ``cosmic_mas`` is set, unless ``force=True``.

    ``read_noise`` (e-) and ``saturation_level`` refer to the image in
    electrons (times ``EXPTIME`` for e-/s images); ``None`` takes them from
    the header (:func:`cosmic_ray_noise_parameters`).
    """
    ccd = image.read_image()
    already = cosmics_identified(ccd.meta)

    if already and not force:
        msg = (
            "Cosmic rays already identified in FITS header "
            "(CRIDENT / cosmics_rm / cosmics_msk); skipping lacosmic."
        )
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg)
        else:
            terminal_output.print_to_terminal(msg)
        return

    read_noise, saturation_level, notes = cosmic_ray_noise_parameters(
        ccd.meta, read_noise, saturation_level
    )
    for msg in ("Remove cosmic rays ...", *notes):
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg)
        else:
            terminal_output.print_to_terminal(msg)

    #   Get exposure time
    exposure_time = ccd.meta.get("exptime", 1.0)

    #   Get unit of the image to check if the image was scaled with the
    #   exposure time
    if ccd.unit == u.electron / u.s:
        scaled = True
        reduced = ccd.multiply(exposure_time * u.second)
    else:
        scaled = False
        reduced = ccd

    #   Remove cosmic rays
    reduced = ccdp.cosmicray_lacosmic(
        reduced,
        objlim=limiting_contrast,
        readnoise=read_noise,
        sigclip=sigma_clipping_value,
        satlevel=saturation_level,
        verbose=verbose,
    )
    if not add_mask:
        reduced.mask = np.zeros(reduced.shape, dtype=bool)
    if verbose:
        if terminal_logger is not None:
            terminal_logger.add_to_cache("")
        else:
            terminal_output.print_to_terminal("")

    mark_cosmics_identified(reduced.meta, handling="interpolated")

    #   Reapply scaling if image was scaled with the exposure time
    if scaled:
        reduced = reduced.divide(exposure_time * u.second)

    #   Set file name
    basename = base_utilities.get_basename(image.filename)
    file_name = f"{basename}_cosmic-rm.fit"

    #   Set new file name and path
    image.filename = file_name
    image.path = os.path.join(
        str(image.out_path),
        "cosmics_rm",
        file_name,
    )

    #   Check if the 'cosmics_rm' directory already exits.
    #   If not, create it.
    checks.check_output_directories(os.path.join(str(image.out_path), "cosmics_rm"))

    #   Save image
    reduced.write(image.path, overwrite=True)


def _should_run_cosmic_removal(
    image: AnalysisImage,
    cosmic_ray_removal: bool | str,
) -> tuple[bool, bool]:
    """Return ``(run, force)`` for analysis cosmic-ray handling."""
    mode = normalize_cosmic_ray_removal(cosmic_ray_removal)
    if mode == "never":
        return False, False
    if mode == "always":
        return True, True
    # auto: run only when header does not already mark cosmics
    ccd = image.read_image()
    if cosmics_identified(ccd.meta):
        return False, False
    return True, False


def determine_background(
    image: AnalysisImage,
    sigma_background: float = 5.0,
    two_d_background: bool = True,
    apply_background: bool = True,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Determine background, using photutils

    Parameters
    ----------
    image
        Object with all image specific properties

    sigma_background
        Sigma used for the sigma clipping of the background
        Default is ``5.``.

    two_d_background
        If True a 2D background will be estimated and subtracted.
        Default is ``True``.

    apply_background
        If True path and file name will be set to the background
        subtracted images, so that those will automatically be used in
        further processing steps.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    Returns
    -------
    background_value
        Image background

    rms_background
        Root mean square of the image background (measured on the
        background-subtracted frame used by the finder).
    """
    if verbose:
        terminal_output.print_to_terminal(
            f"Determine background: {image.filter_} filter",
            indent=2,
        )

    #   Load image data
    ccd = image.read_image()

    #   Set up sigma clipping
    sigma_clip = SigmaClip(sigma=sigma_background)

    #   2D background?
    if two_d_background:
        #   Estimate 2D background
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            ccd.data,
            (80, 80),
            mask=ccd.mask,
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            exclude_percentile=20,
        )

        #   Remove background
        image_no_bg = ccd.subtract(bkg.background * ccd.unit)

        #   Put metadata back on the image, because it is lost while
        #   subtracting the background
        image_no_bg.meta = ccd.meta
        image_no_bg.meta["HIERARCH"] = "2D background removed"

        #   Add Header keyword to mark the file as background subtracted
        image_no_bg.meta["NO_BG"] = True

        #   Get median of the background
        background_value = bkg.background_median
    else:
        #   Estimate 1D background
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
        background_value = mmm_bkg.calc_background(
            ma.masked_array(ccd.data, mask=ccd.mask)
        )

        #   Remove background
        image_no_bg = ccd.subtract(background_value * ccd.unit)

        #   Put metadata back on the image, because it is lost while
        #   subtracting the background
        image_no_bg.meta = ccd.meta
        image_no_bg.meta["HIERARCH"] = "1D background removed"

        #   Add Header keyword to mark the file as background subtracted
        image_no_bg.meta["NO_BG"] = True

    #   RMS on the frame the finder will see (masked pixels excluded)
    background_rms = MADStdBackgroundRMS(sigma_clip=sigma_clip)
    rms_background = background_rms(
        ma.masked_array(image_no_bg.data, mask=image_no_bg.mask)
    )

    #   Define name and save image
    file_name = f"{base_utilities.get_basename(image.filename)}_no_bkg.fit"
    output_path = image.out_path / "no_bkg"
    checks.check_output_directories(output_path)
    image_no_bg.write(output_path / file_name, overwrite=True)

    #   Set new path and file
    #   -> Background subtracted image will be used in further processing steps
    if apply_background:
        #   Saturation is judged on the frame with its sky (see
        #   ``_extraction_quality``), so remember where it is.
        image.path_before_background = image.path
        image.path = output_path / file_name
        image.filename = file_name

    return background_value, rms_background


def build_star_finder(
    method: str,
    *,
    threshold: float,
    fwhm: float,
    sharpness_range: tuple[float, float] = (0.2, 1.0),
    roundness_range: tuple[float, float] = (-1.0, 1.0),
    min_separation_fwhm: float = 1.0,
):
    """Build IRAF or DAO star finder with shared quality cuts."""
    min_separation = max(2, int(float(fwhm) * float(min_separation_fwhm) + 0.5))
    roundness_range = roundness_range_for_finder(method, roundness_range)

    if method == "DAO":
        return DAOStarFinder(
            fwhm=fwhm,
            threshold=threshold,
            sharpness_range=sharpness_range,
            roundness_range=roundness_range,
            min_separation=min_separation,
            exclude_border=True,
        )
    if method == "IRAF":
        return IRAFStarFinder(
            threshold=threshold,
            fwhm=fwhm,
            min_separation=min_separation,
            sharpness_range=sharpness_range,
            roundness_range=roundness_range,
            exclude_border=True,
        )
    raise ValueError(
        f"{style.Bcolors.FAIL}\nExtraction method ({method}) not valid: "
        f"use either IRAF or DAO {style.Bcolors.ENDC}"
    )


def _odd_psf_fit_shape(requested: int, fwhm: float) -> int:
    """Odd fit stamp: at least ``requested``, ~2.5×FWHM, minimum 11 px."""
    need = max(int(requested), int(np.ceil(2.5 * float(fwhm))), 11)
    if need % 2 == 0:
        need += 1
    return need


def _format_fwhm_status(
    fwhm: float,
    *,
    source: str,
    n_stars: int | None = None,
) -> str:
    """Terminal FWHM line in plain language (not internal meta keys)."""
    value = f"FWHM = {fwhm:.2f} px"
    n = int(n_stars) if n_stars else 0
    if source == "user":
        if n:
            return (
                f"{value} (using the value you provided; {n} sources found)"
            )
        return f"{value} (using the value you provided)"
    if source == "finder_column":
        if n:
            return (
                f"{value} (estimated from a sample of {n} stars "
                f"in the expected size range; sample is capped at 25)"
            )
        return f"{value} (estimated from the star finder)"
    if source == "psf_fit":
        if n:
            return f"{value} (estimated by fitting {n} stars)"
        return f"{value} (estimated by fitting star profiles)"
    return f"{value} (using the default value)"


def find_stars(
    image: AnalysisImage,
    rms_background: float,
    fwhm_object_psf: float | None = None,
    multiplier_background_rms: float = 5.0,
    method: str = "IRAF",
    terminal_logger: terminal_output.TerminalLog | None = None,
    indent: int = 2,
    fwhm_estimate_min: float = 2.0,
    fwhm_estimate_max: float = 15.0,
    finder_sharpness_range: tuple[float, float] = (0.2, 1.0),
    finder_roundness_range: tuple[float, float] = (-1.0, 1.0),
    finder_min_separation_fwhm: float = 1.0,
    finder_fwhm_scale_range: tuple[float, float] | None = (0.5, 2.0),
) -> None:
    """
    Find the stars on the images, using photutils and search and select
    stars for the ePSF stars

    Parameters
    ----------
    image
        Object with all image specific properties

    rms_background
        Root mean square of the image background

    fwhm_object_psf
        FWHM of the objects PSF, assuming it is a Gaussian
        Default is ``None``.

    multiplier_background_rms
        Multiplier for the background RMS, used to calculate the
        threshold to identify stars
        Default is ``5``.

    method
        Finder method DAO or IRAF
        Default is ``IRAF``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``2``.

    fwhm_estimate_min, fwhm_estimate_max
        Accepted per-star FWHM range (pixels) for automatic estimation.

    finder_fwhm_scale_range
        After detection, keep only sources whose finder ``fwhm`` lies in
        this factor times the image FWHM (e.g. ``(0.5, 2.0)``). ``None``
        disables the cut. IRAF catalogs include ``fwhm``; DAO typically does
        not, so the cut is then a no-op.
    """
    def _log(
        msg: str,
        *,
        style_name: str | None = None,
        extra_indent: int = 0,
    ) -> None:
        log_indent = indent + extra_indent
        if terminal_logger is not None:
            if style_name:
                terminal_logger.add_to_cache(
                    msg, indent=log_indent, style_name=style_name
                )
            else:
                terminal_logger.add_to_cache(msg, indent=log_indent)
        elif style_name:
            terminal_output.print_to_terminal(
                msg, indent=log_indent, style_name=style_name
            )
        else:
            terminal_output.print_to_terminal(msg, indent=log_indent)

    def _keep_stellar_sizes(tbl, fwhm_pix):
        filtered, n_drop = filter_finder_table_by_fwhm_scale(
            tbl, fwhm_pix, scale_range=finder_fwhm_scale_range
        )
        if n_drop == 0:
            return tbl
        n_orig = len(tbl)
        if filtered is None or len(filtered) == 0:
            _log(
                f"FWHM size cut would remove all {n_orig} detections; "
                "keeping them.",
                style_name="WARNING",
                extra_indent=1,
            )
            return tbl
        scale = finder_fwhm_scale_range
        if scale is None:
            return filtered
        lo, hi = scale
        _log(
            f"Kept {len(filtered)} of {n_orig} detections with FWHM "
            f"{lo * fwhm_pix:.1f}–{hi * fwhm_pix:.1f} px "
            f"({lo:.1f}–{hi:.1f} × {fwhm_pix:.2f} px)",
            extra_indent=1,
        )
        return filtered

    _log("Identify stars")

    #   Load image data
    ccd = image.read_image()

    #   Use background RMS as sigma
    sigma = rms_background
    threshold = multiplier_background_rms * sigma

    #   Set default FWHM
    user_fwhm = fwhm_object_psf is not None
    if user_fwhm:
        default_fwhm = float(fwhm_object_psf)
    else:
        default_fwhm = image.fwhm

    finder_kwargs = {
        "sharpness_range": finder_sharpness_range,
        "roundness_range": finder_roundness_range,
        "min_separation_fwhm": finder_min_separation_fwhm,
    }

    #   First run of finder with default FWHM or user provided FWHM
    #   -> needed to have some initial object positions for FWHM determination
    finder = build_star_finder(
        method, threshold=threshold, fwhm=default_fwhm, **finder_kwargs
    )
    tbl_objects = finder(ccd.data, mask=ccd.mask)

    if tbl_objects is None or len(tbl_objects) == 0:
        _log(
            f"[Info] FWHM determination skipped: no sources found. "
            f"Use the default FWHM of {default_fwhm}.",
            style_name="WARNING",
            extra_indent=1,
        )
        image.fwhm = default_fwhm
        return

    if user_fwhm:
        # Keep user FWHM; still refresh detections with the same value/cuts.
        tbl_objects = _keep_stellar_sizes(tbl_objects, default_fwhm)
        image.positions = tbl_objects.copy()
        image.fwhm = default_fwhm
        _log(
            _format_fwhm_status(
                default_fwhm, source="user", n_stars=len(tbl_objects)
            ),
            extra_indent=1,
        )
        return

    median_fwhm, fwhm_error, fwhm_meta = estimate_image_fwhm(
        ccd.data,
        tbl_objects,
        mask=ccd.mask,
        error=ccd.uncertainty.array if ccd.uncertainty is not None else None,
        default_fwhm=default_fwhm,
        min_fwhm=fwhm_estimate_min,
        max_fwhm=fwhm_estimate_max,
    )
    if fwhm_error is not None:
        _log(
            f"[Info] FWHM determination failed with the following error "
            f"{fwhm_error}. Use the default FWHM of {default_fwhm}.",
            style_name="WARNING",
            extra_indent=1,
        )

        #   Keep the full finder table (sharpness, roundness, …) for later merge.
        tbl_objects = _keep_stellar_sizes(tbl_objects, default_fwhm)
        image.positions = tbl_objects.copy()
        image.fwhm = default_fwhm
        return

    fwhm_source = str(fwhm_meta.get("source", "default"))
    n_in_range = int(fwhm_meta.get("n_in_range", 0))

    #   Run finder with new FWHM
    finder = build_star_finder(
        method, threshold=threshold, fwhm=median_fwhm, **finder_kwargs
    )
    tbl_objects = finder(ccd.data, mask=ccd.mask)

    if tbl_objects is None or len(tbl_objects) == 0:
        # Fall back to default-FWHM finder if the refined search finds nothing
        finder = build_star_finder(
            method, threshold=threshold, fwhm=default_fwhm, **finder_kwargs
        )
        tbl_objects = finder(ccd.data, mask=ccd.mask)
        median_fwhm = default_fwhm
        fwhm_source = "default"

    #   Keep the full finder table so quality columns can be copied onto photometry.
    if tbl_objects is not None and len(tbl_objects) > 0:
        tbl_objects = _keep_stellar_sizes(tbl_objects, median_fwhm)
        image.positions = tbl_objects.copy()
    image.fwhm = median_fwhm
    _log(
        _format_fwhm_status(
            median_fwhm, source=fwhm_source, n_stars=n_in_range
        ),
        extra_indent=1,
    )

def check_epsf_stars(
    image: AnalysisImage,
    size_epsf_region: int = 25,
    minimum_n_stars: int = 25,
    fraction_epsf_stars: float = 0.2,
    maximum_n_stars: int | None = 50,
    terminal_logger: terminal_output.TerminalLog | None = None,
    strict_epsf_checks: bool = True,
    indent: int = 2,
    saturation_fraction: float = 0.9,
) -> Table:
    """
    Select ePSF stars and check if there are enough

    Parameters
    ----------
    image
        Object with all image specific properties

    size_epsf_region
        Size of the extraction region in pixel
        Default is ``25``.

    minimum_n_stars
        Minimal number of stars required for the ePSF calculations
        Default is ``25``.

    fraction_epsf_stars
        Fraction of all stars that should be used to calculate the ePSF
        Default is ``0.2``.

    maximum_n_stars
        Upper cap on the number of ePSF stars after applying the fraction
        (and the minimum). ``None`` disables the cap. Default is ``50``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    strict_epsf_checks
        If True a stringent test of the ePSF conditions is applied.
        Default is ``True``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    #   Get object positions
    tbl_positions = image.positions
    identification_string = f"{image.image_id}. {image.filter_}"
    if tbl_positions is None or len(tbl_positions) == 0:
        raise ExtractionSkipError(
            f"No sources identified in the {identification_string} band image."
        )

    #   Number of objects
    n_stars = len(tbl_positions)

    #   Get image data
    image_data = image.get_data()

    #   Useful information
    out_string = (
        f"{n_stars} sources identified in the {identification_string} band image"
    )
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            out_string,
            indent=indent + 1,
            style_name="OK",
        )
    else:
        terminal_output.print_to_terminal(
            out_string,
            indent=indent + 1,
            style_name="OK",
        )

    #   Saturated stars (flat cores) would make the ePSF too broad: left out
    #   before the sample is chosen, so fainter stars take their place.
    fwhm = float(getattr(image, "fwhm", 4.0) or 4.0)
    saturated, _raw = _saturated_mask(
        image, tbl_positions["x_centroid"], tbl_positions["y_centroid"], max(2.0, 1.5 * fwhm),
        saturation_fraction,
    )
    if saturated is not None and np.any(saturated):
        tbl_positions = tbl_positions[~saturated]
        msg = f"{int(np.count_nonzero(saturated))} saturated star(s) left out of the ePSF sample."
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg, indent=indent + 1)
        else:
            terminal_output.print_to_terminal(msg, indent=indent + 1)
        n_stars = len(tbl_positions)

    #  Determine sample of stars used for estimating the ePSF
    #   (rm the brightest 1% of all stars because those are often saturated)
    #   Sort list with star positions according to flux
    tbl_positions_sort = tbl_positions.group_by("flux")
    # Determine the 99 percentile
    percentile_99 = np.percentile(tbl_positions_sort["flux"], 99)
    #   Determine the position of the 99 percentile in the position list
    id_percentile_99 = np.argmin(
        np.absolute(tbl_positions_sort["flux"] - percentile_99)
    )

    available_epsf_stars = n_epsf_stars_to_select(
        n_stars,
        fraction_epsf_stars=fraction_epsf_stars,
        minimum_n_stars=minimum_n_stars,
        maximum_n_stars=maximum_n_stars,
    )

    #   Check if enough stars have been identified
    if (
        id_percentile_99 - available_epsf_stars < minimum_n_stars and strict_epsf_checks
    ) or (id_percentile_99 - available_epsf_stars < 1 and not strict_epsf_checks):
        raise ExtractionSkipError(
            f"Not enough stars ({id_percentile_99 - available_epsf_stars}) "
            f"found to determine the ePSF in the {identification_string} band."
        )

    #   Resize table -> limit it to the suitable stars
    tbl_epsf_stars = tbl_positions_sort[:][
        id_percentile_99 - available_epsf_stars : id_percentile_99
    ]

    #   Exclude stars that are too close to the image boarder
    #   Size of the extraction box around each star
    half_size_epsf_region = (size_epsf_region - 1) / 2

    #   New lists with x and y positions
    x = tbl_epsf_stars["x_centroid"]
    y = tbl_epsf_stars["y_centroid"]

    mask = (
        (x > half_size_epsf_region)
        & (x < (image_data.shape[1] - 1 - half_size_epsf_region))
        & (y > half_size_epsf_region)
        & (y < (image_data.shape[0] - 1 - half_size_epsf_region))
    )

    #   Updated positions table
    tbl_epsf_stars = tbl_epsf_stars[:][mask]
    n_useful_epsf_stars = len(tbl_epsf_stars)

    #   Check if there are still enough stars
    if (n_useful_epsf_stars < minimum_n_stars and strict_epsf_checks) or (
        n_useful_epsf_stars < 1 and not strict_epsf_checks
    ):
        raise ExtractionSkipError(
            f"Not enough stars ({n_useful_epsf_stars}) for the ePSF "
            f"determination in the {identification_string} band image. "
            "Too many potential ePSF stars have been removed because they "
            "are too close to the image border."
        )

    image_mask = image.get_mask()
    tbl_epsf_stars, n_nonfinite = filter_table_finite_cutouts(
        tbl_epsf_stars,
        image_data,
        size_epsf_region,
        extra_mask=image_mask,
    )
    if n_nonfinite:
        msg = (
            f"Rejected {n_nonfinite} ePSF star(s) with non-finite cutout pixels "
            f"in the {identification_string} band image."
        )
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg, indent=indent + 1, style_name="WARNING")
        else:
            terminal_output.print_to_terminal(
                msg, indent=indent + 1, style_name="WARNING"
            )
    n_useful_epsf_stars = len(tbl_epsf_stars)
    if (n_useful_epsf_stars < minimum_n_stars and strict_epsf_checks) or (
        n_useful_epsf_stars < 1 and not strict_epsf_checks
    ):
        raise ExtractionSkipError(
            f"Not enough stars ({n_useful_epsf_stars}) for the ePSF "
            f"determination in the {identification_string} band image. "
            "Too many potential ePSF stars have been removed because their "
            "cutouts contain non-finite pixels."
        )

    #   Find all potential ePSF stars with close neighbors
    x1 = tbl_positions_sort["x_centroid"]
    y1 = tbl_positions_sort["y_centroid"]
    x2 = tbl_epsf_stars["x_centroid"]
    y2 = tbl_epsf_stars["y_centroid"]
    max_objects = np.max((len(x1), len(x2)))
    x_all = np.zeros((max_objects, 2))
    y_all = np.zeros((max_objects, 2))
    x_all[0 : len(x1), 0] = x1
    x_all[0 : len(x2), 1] = x2
    y_all[0 : len(y1), 0] = y1
    y_all[0 : len(y2), 1] = y2

    id_percentile_99 = correlate.correlation_own(
        x_all,
        y_all,
        max_pixel_between_objects=size_epsf_region,
        ooi_correlation_strategy=3,
        silent=True,
    )[1]

    #   Determine multiple entries -> stars that are contaminated
    index_percentile_99_mult = [
        ite for ite, count in Counter(id_percentile_99).items() if count > 1
    ]

    #   Find unique entries -> stars that are not contaminated
    index_percentile_99_unique = [
        ite for ite, count in Counter(id_percentile_99).items() if count == 1
    ]
    n_useful_epsf_stars = len(index_percentile_99_unique)

    #   Remove ePSF stars with close neighbors from the corresponding table
    tbl_epsf_stars.remove_rows(index_percentile_99_mult)

    #   Check if there are still enough stars
    if (n_useful_epsf_stars < minimum_n_stars and strict_epsf_checks) or (
        n_useful_epsf_stars < 1 and not strict_epsf_checks
    ):
        raise ExtractionSkipError(
            f"Not enough stars ({n_useful_epsf_stars}) for the ePSF "
            f"determination in the {identification_string} band image. "
            "Too many potential ePSF stars have been removed because other "
            "stars are in the extraction region."
        )

    #   Return ePSF stars
    return tbl_epsf_stars


def determine_epsf(
    image: AnalysisImage,
    epsf_star_positions: Table,
    size_epsf_region: int = 25,
    oversampling_factor: int = 2,
    max_n_iterations: int = 7,
    minimum_n_stars: int = 25,
    multiprocess_plots: bool = True,
    terminal_logger: terminal_output.TerminalLog | None = None,
    file_type_plots: str = "pdf",
    indent: int = 2,
    plot_output_dir: str | None = None,
) -> None:
    """
    Main function to determine the ePSF, using photutils

    Parameters
    ----------
    image
        Object with all image specific properties

    epsf_star_positions
        Table with position of the ePSF stars

    size_epsf_region
        Size of the extraction region in pixel
        Default is ``25``.

    oversampling_factor
        ePSF oversampling factor
        Default is ``2``.

    max_n_iterations
        Number of ePSF iterations
        Default is ``7``.

    minimum_n_stars
        Minimal number of stars required for the ePSF calculations
        Default is ``25``.

    multiprocess_plots
        If True multiprocessing is used for plotting.
        Default is ``True``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    #   Get image data
    data = image.get_data()
    epsf_star_positions, n_nonfinite = filter_table_finite_cutouts(
        epsf_star_positions,
        data,
        size_epsf_region,
        extra_mask=image.get_mask(),
    )
    if n_nonfinite:
        msg = (
            f"Rejected {n_nonfinite} ePSF star(s) with non-finite cutout pixels "
            "before extract_stars."
        )
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg, indent=indent, style_name="WARNING")
        else:
            terminal_output.print_to_terminal(msg, indent=indent, style_name="WARNING")
    if len(epsf_star_positions) < 1:
        raise ExtractionSkipError(
            f"No ePSF stars remain after rejecting non-finite cutouts "
            f"(image {image.image_id}, filter {image.filter_})."
        )

    #   Number of ePSF stars
    n_epsf = len(epsf_star_positions)

    if n_epsf < minimum_n_stars and terminal_logger is not None:
        terminal_logger.add_to_cache(
            f"The number of ePSF stars is less than required."
            f"{n_epsf} ePSF stars available. {minimum_n_stars} were "
            "requested.",
            indent=indent,
            style_name="WARNING",
        )

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            "Determine the point spread function", indent=indent
        )
        terminal_logger.add_to_cache(
            f"{n_epsf} bright stars used",
            indent=indent + 1,
            style_name="OK",
        )
    else:
        terminal_output.print_to_terminal(
            "Determine the point spread function", indent=indent
        )
        terminal_output.print_to_terminal(
            f"{n_epsf} bright stars used",
            indent=indent + 1,
            style_name="OK",
        )

    #   Create new table with the names required by "extract_stars"
    stars_tbl = Table()
    stars_tbl["x"] = epsf_star_positions["x_centroid"]
    stars_tbl["y"] = epsf_star_positions["y_centroid"]

    #   Put image into NDData container (required by "extract_stars")
    nd_data = NDData(data=data)

    #   Extract cutouts of the selected stars
    stars = extract_stars(nd_data, stars_tbl, size=size_epsf_region)

    #   Combine plot identification string
    string = f"img-{image.image_id}-{image.filter_}"

    #   Get output directory
    output_dir = plot_output_dir or _extraction_qc_dir(image, gallery=False)

    #   Plot the brightest ePSF stars
    if multiprocess_plots:
        start_plot_process(
            plots.plot_cutouts,
            (output_dir, stars, string),
            {"file_type": file_type_plots},
        )
    else:
        plots.plot_cutouts(
            output_dir,
            stars,
            string,
            terminal_logger=terminal_logger,
            file_type=file_type_plots,
        )

    #   Build the ePSF (set oversampling and max. number of iterations)
    epsf_builder = EPSFBuilder(
        oversampling=oversampling_factor,
        maxiters=max_n_iterations,
        progress_bar=False,
    )
    epsf, fitted_stars = epsf_builder(stars)

    #   Add ePSF and fitted stars to image class
    image.epsf = epsf


def _masked_annulus_sky(
    data: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray | None,
    x: np.ndarray,
    y: np.ndarray,
    r_in: float,
    r_out: float,
    fwhm: float,
) -> np.ndarray:
    """``(sky, sky error)`` per star: clipped median in ``r_in``–``r_out``
    outside the disks (2 FWHM, inside ``r_in``) of all given stars."""
    from .photometry_quality import MEDIAN_ERROR_FACTOR, annulus_sky, source_disk_mask

    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    annulus = CircularAnnulus(np.column_stack([xs, ys]), r_in=r_in, r_out=r_out)
    disks = source_disk_mask(np.shape(data), xs, ys, max(1.5, min(2.0 * fwhm, 0.9 * r_in)))
    stats = annulus_sky(np.asarray(data, dtype=float), np.asarray(error, dtype=float), annulus,
                        mask=mask, source_mask=disks)
    sky = stats["sky"]
    #   error of the clipped median: √(π/2)·σ/√n
    sky_err = MEDIAN_ERROR_FACTOR * stats["sigma_map"] / np.sqrt(np.maximum(stats["n_sky"], 1.0))
    return np.where(np.isfinite(sky), sky, 0.0), np.where(np.isfinite(sky_err), sky_err, 0.0)


def _photon_exposure(ccd) -> float | None:
    """Seconds that turn the image unit into photo-electrons (1 for electrons,
    EXPTIME for e-/s); ``None`` for other units (photon noise unknown)."""
    unit = getattr(ccd, "unit", None)
    if unit is None:
        return None
    if unit.is_equivalent(u.electron):
        return 1.0
    if unit.is_equivalent(u.electron / u.s):
        exptime = float(ccd.meta.get("EXPTIME", float("nan")))
        return exptime if np.isfinite(exptime) and exptime > 0 else None
    return None


def _background_only_error(
    data: np.ndarray, error: np.ndarray, exposure: float | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(σ_bkg, source variance)``: the pixel σ without the sources' photon
    noise ``max(data - sky, 0)/t``, at least half the typical sky variance.
    Unknown unit: the total σ and ``None``."""
    if exposure is None:
        return error, None
    values = np.asarray(data, dtype=float)
    sky = float(np.nanmedian(values[::4, ::4]))
    source_var = np.clip(np.nan_to_num(values - sky), 0.0, None) / float(exposure)
    total_var = np.asarray(error, dtype=float) ** 2
    floor = 0.5 * float(np.nanmedian(total_var[::4, ::4]))
    return np.sqrt(np.maximum(total_var - source_var, floor)), source_var


def _psf_photon_factor(psf_model, fit_shape: tuple[int, int]) -> tuple[float, float]:
    """``(ΣP³ / (ΣP²)², ΣP / ΣP²)`` of the unit-flux PSF sampled on the fit
    box: photon-noise variance of a constant-weight PSF flux per unit flux
    (4/3 for a Gaussian), and the flux change per unit sky offset."""
    model = psf_model.copy()
    ny, nx = fit_shape
    yy, xx = np.mgrid[:ny, :nx]
    for name, value in (("x_0", (nx - 1) / 2), ("y_0", (ny - 1) / 2), ("flux", 1.0)):
        if hasattr(model, name):
            setattr(model, name, value)
    p = np.asarray(model(xx, yy), dtype=float)
    p = np.clip(p, 0.0, None)
    s2 = float(np.sum(p**2))
    if s2 <= 0:
        return 1.0, 0.0
    return float(np.sum(p**3) / s2**2), float(np.sum(p) / s2)


def _expected_chi2_factor(
    error_total: np.ndarray,
    error_fit: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fit_shape: tuple[int, int],
) -> np.ndarray:
    """Mean of ``σ_tot² / σ_fit²`` over each star's fit box (≥ 1)."""
    ratio = (np.asarray(error_total, dtype=float) / np.asarray(error_fit, dtype=float)) ** 2
    ny, nx = ratio.shape
    hy, hx = fit_shape[0] // 2, fit_shape[1] // 2
    out = np.ones(len(x))
    for k, (xc, yc) in enumerate(zip(np.asarray(x, dtype=float), np.asarray(y, dtype=float), strict=True)):
        if not (np.isfinite(xc) and np.isfinite(yc)):
            continue
        ix, iy = int(round(xc)), int(round(yc))
        box = ratio[max(0, iy - hy):min(ny, iy + hy + 1), max(0, ix - hx):min(nx, ix + hx + 1)]
        if box.size:
            out[k] = max(1.0, float(np.nanmean(box)))
    return out


def _sanitize_pixel_error(error: np.ndarray) -> np.ndarray:
    """Copy pixel σ; replace non-finite / non-positive values, keep the rest.

    PSF flux errors come from this map (photutils fitter covariance). An earlier
    cleanup inverted the NaN mask and set *every* finite pixel to ``max(σ)``,
    which inflated CMD error bars for ``photometry_extraction_method="PSF"``.
    """
    out = np.array(error, dtype=float, copy=True)
    valid = np.isfinite(out) & (out > 0.0)
    if not np.any(valid):
        out[:] = 1.0
        return out
    out[~valid] = float(np.max(out[valid]))
    return out


def extraction_epsf(
    image: AnalysisImage,
    background_rms: float,
    sigma_background: float = 5.0,
    use_initial_positions: bool = True,
    finder_method: str = "IRAF",
    size_extraction_region: int = 11,
    epsf_fitter: str = "TRFLSQFitter",
    n_iterations_eps_extraction: int = 1,
    multiplier_background_rms: float = 5.0,
    multiplier_grouper: float = 2.0,
    terminal_logger: terminal_output.TerminalLog | None = None,
    rm_background: bool = True,
    indent: int = 2,
    finder_sharpness_range: tuple[float, float] = (0.2, 1.0),
    finder_roundness_range: tuple[float, float] = (-1.0, 1.0),
    finder_min_separation_fwhm: float = 1.0,
    psf_find_in_residuals: bool = False,
    group_wing_fwhm: float | None = 2.5,
    residual_sky: bool = True,
) -> None:
    """
    Main function to perform the eEPSF photometry, using photutils

    ``group_wing_fwhm``: stars are fitted together when one lies within the
    other's fit half-width plus this many FWHM (its wings reach the fit
    box); ``None`` groups by ``multiplier_grouper`` × FWHM only (faster in
    crowded fields, neighbours' wings then bias the fit). ``residual_sky``:
    second pass with the local sky from the residual image.
    """
    output_path = image.out_path

    table_dir = tables_dir(output_path)
    checks.check_output_directories(output_path, table_dir)

    ccd = image.read_image()
    data = ccd.data
    error = _pixel_error(image, ccd)
    image_mask = ccd.mask
    filter_ = image.filter_

    initial_positions = None
    if use_initial_positions:
        try:
            positions_flux = image.positions
            initial_positions = Table(
                names=["x_0", "y_0", "flux_0"],
                data=[
                    positions_flux["x_centroid"],
                    positions_flux["y_centroid"],
                    positions_flux["flux"],
                ],
            )
        except RuntimeError:
            use_initial_positions = False

    identification_str = f"{image.image_id}-{filter_}"
    epsf = image.epsf
    fwhm = image.fwhm

    output_str = f"Performing the actual PSF photometry ({identification_str} image)"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(output_str, indent=indent)
    else:
        terminal_output.print_to_terminal(output_str, indent=indent)

    finder = build_star_finder(
        finder_method,
        threshold=multiplier_background_rms * background_rms,
        fwhm=fwhm,
        sharpness_range=finder_sharpness_range,
        roundness_range=finder_roundness_range,
        min_separation_fwhm=finder_min_separation_fwhm,
    )

    if epsf_fitter == "LevMarLSQFitter":
        fitter = LevMarLSQFitter()
    elif epsf_fitter == "LMLSQFitter":
        fitter = LMLSQFitter()
    elif epsf_fitter == "TRFLSQFitter":
        fitter = TRFLSQFitter()
    else:
        terminal_output.print_to_terminal(
            f"WARNING: Fitter method ({epsf_fitter}) for ePSF "
            f"extraction not known: Switching to LMLSQFitter.",
            style_name="WARNING",
            indent=indent,
        )
        fitter = LMLSQFitter()

    size_extraction_region = _odd_psf_fit_shape(size_extraction_region, fwhm)

    fit_shape = (size_extraction_region, size_extraction_region)
    aperture_radius = (size_extraction_region - 1) / 2

    #   Residual sky left by the 80 px Background2D map biases the PSF flux by
    #   about b·N_eff; it is measured locally in an annulus beyond the PSF
    #   wings. With known positions: clipped median of the annulus pixels
    #   outside the disks of all detected stars (as for APER), passed as
    #   ``local_bkg``; photutils' LocalBackground would take neighbours'
    #   wings into the sky.
    r_in = max(7.0, 2.8 * float(fwhm))
    r_out = max(r_in + 3.0, 4.0 * float(fwhm))
    local_bkg_estimator = None
    sky_err = None
    if rm_background and use_initial_positions:
        initial_positions["local_bkg"], sky_err = _masked_annulus_sky(
            data, error, image_mask, initial_positions["x_0"], initial_positions["y_0"],
            r_in, r_out, float(fwhm),
        )
    elif rm_background:
        sigma_clip = SigmaClip(sigma=sigma_background)
        local_bkg_estimator = LocalBackground(r_in, r_out, bkg_estimator=MMMBackground(sigma_clip=sigma_clip))

    #   Stars are fitted together when a neighbour's wings (2.5 FWHM) reach
    #   into the other's fit box; otherwise its light is fitted as the star's.
    min_separation = float(multiplier_grouper) * float(fwhm)
    if group_wing_fwhm is not None:
        min_separation = max(min_separation, aperture_radius + float(group_wing_fwhm) * float(fwhm))
    source_grouper = SourceGrouper(min_separation=min_separation)

    if psf_find_in_residuals:
        photometry = IterativePSFPhotometry(
            psf_model=epsf,
            fit_shape=fit_shape,
            finder=finder,
            grouper=source_grouper,
            fitter=fitter,
            maxiters=n_iterations_eps_extraction,
            local_bkg_estimator=local_bkg_estimator,
            mode="all",
            aperture_radius=aperture_radius,
            xy_bounds=float(fwhm),
        )
    else:
        # Fit finder positions only; do not seed new sources from residuals.
        photometry = PSFPhotometry(
            psf_model=epsf,
            fit_shape=fit_shape,
            finder=None if use_initial_positions else finder,
            grouper=source_grouper,
            fitter=fitter,
            local_bkg_estimator=local_bkg_estimator,
            aperture_radius=aperture_radius,
            xy_bounds=float(fwhm),
        )

    error = _sanitize_pixel_error(error)
    #   Fit weights from the background noise only: weights from σ that holds
    #   the star's own photon noise (measured from the noisy data) pull the
    #   fit low. The star's photon noise is added to flux_err afterwards.
    exposure = _photon_exposure(ccd)
    error_fit, source_var = _background_only_error(data, error, exposure)

    if use_initial_positions:
        result_tbl = photometry(
            data=data,
            error=error_fit,
            mask=image_mask,
            init_params=initial_positions,
        )
        x_start = np.asarray(initial_positions["x_0"], dtype=float)
        y_start = np.asarray(initial_positions["y_0"], dtype=float)
        if (residual_sky and rm_background and not psf_find_in_residuals
                and len(result_tbl) == len(initial_positions)):
            #   Second pass: the sky from the residual image (all fitted stars
            #   subtracted, wings included) instead of the data: the annulus
            #   at 2.8-4 FWHM still holds the star's own wings and those of
            #   brighter neighbours beyond their masked cores.
            psf_size = 2 * int(np.ceil(r_out)) + 1
            residual = data - photometry.make_model_image(np.shape(data), psf_shape=(psf_size, psf_size))
            x1 = np.asarray(result_tbl["x_fit"], dtype=float)
            y1 = np.asarray(result_tbl["y_fit"], dtype=float)
            f1 = np.asarray(getattr(result_tbl["flux_fit"], "value", result_tbl["flux_fit"]), dtype=float)
            ok = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(f1)
            x1, y1 = np.where(ok, x1, x_start), np.where(ok, y1, y_start)
            sky2, sky_err = _masked_annulus_sky(residual, error, image_mask, x1, y1, r_in, r_out, float(fwhm))
            second = Table({
                "x_0": x1,
                "y_0": y1,
                "flux_0": np.where(ok, f1, np.asarray(initial_positions["flux_0"], dtype=float)),
                "local_bkg": sky2,
            })
            del residual
            result_tbl = photometry(data=data, error=error_fit, mask=image_mask, init_params=second)
    else:
        result_tbl = photometry(data=data, error=error_fit, mask=image_mask)
        x_start = y_start = None

    if "flux_err" not in result_tbl.colnames:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nPSF photometry returned no flux_err "
            f"column (needs an error array) {style.Bcolors.ENDC}"
        )

    #   Fits without a covariance (NaN flux_err) are unusable; negative net
    #   fluxes of faint stars are kept (dropping them biases faint flux
    #   light curves upward); their magnitudes are NaN.
    flux_err_values = np.asarray(getattr(result_tbl["flux_err"], "value", result_tbl["flux_err"]), dtype=float)
    keep = np.isfinite(flux_err_values) & (flux_err_values >= 0.0)
    n_bad_objects = int(np.count_nonzero(~keep))
    result_tbl = result_tbl[keep]

    #   flux_err is the formal fit covariance for the background σ. The
    #   star's photon noise is added (variance K·F/t with K = ΣP³/(ΣP²)² of
    #   the PSF on the fit box, the variance of a fit with constant weights).
    #   reduced_chi2 is relative to the background σ: divided by its expected
    #   value for a perfect model (mean of σ_tot²/σ_bkg² over the fit box)
    #   before it scales the error where the model does not describe the star
    #   (ePSF mismatch, blends).
    #   The error of the local sky enters as Δb·ΣP/ΣP² (a constant offset of
    #   the fitted pixels).
    flux_fit = np.asarray(getattr(result_tbl["flux_fit"], "value", result_tbl["flux_fit"]), dtype=float)
    k_psf, sky_gain = _psf_photon_factor(epsf, fit_shape)
    err = np.asarray(getattr(result_tbl["flux_err"], "value", result_tbl["flux_err"]), dtype=float)
    var = err**2
    if source_var is not None:
        var = var + k_psf * np.maximum(flux_fit, 0.0) / exposure
    if sky_err is not None and len(sky_err) == len(result_tbl):
        var = var + (sky_gain * np.asarray(sky_err, dtype=float)) ** 2
    result_tbl["flux_err"] = np.sqrt(var)
    if "reduced_chi2" in result_tbl.colnames:
        chi2 = np.asarray(result_tbl["reduced_chi2"], dtype=float)
        if source_var is not None:
            chi2 = chi2 / _expected_chi2_factor(
                error, error_fit, result_tbl["x_fit"], result_tbl["y_fit"], fit_shape
            )
            result_tbl["reduced_chi2"] = chi2
        scale = np.sqrt(np.where(np.isfinite(chi2), np.maximum(chi2, 1.0), 1.0))
        result_tbl["flux_err"] = result_tbl["flux_err"] * scale
    #   Shift of the fit from the detected position (bounded by xy_bounds =
    #   1 FWHM): large values mark blends or wrong matches.
    if x_start is not None and len(result_tbl) == x_start.size:
        result_tbl["centroid_shift"] = np.hypot(
            np.asarray(result_tbl["x_fit"], dtype=float) - x_start,
            np.asarray(result_tbl["y_fit"], dtype=float) - y_start,
        )
    elif "x_init" in result_tbl.colnames and "y_init" in result_tbl.colnames:
        result_tbl["centroid_shift"] = np.hypot(
            np.asarray(result_tbl["x_fit"], dtype=float) - np.asarray(result_tbl["x_init"], dtype=float),
            np.asarray(result_tbl["y_fit"], dtype=float) - np.asarray(result_tbl["y_init"], dtype=float),
        )
    result_tbl["nonpositive_flux"] = (
        np.asarray(getattr(result_tbl["flux_fit"], "value", result_tbl["flux_fit"])) <= 0.0
    )

    try:
        bad_results = np.where(result_tbl["x_fit"].data < 0.0)
        n_bad_objects += np.size(bad_results)
        result_tbl.remove_rows(bad_results)
        bad_results = np.where(result_tbl["y_fit"].data < 0.0)
        n_bad_objects += np.size(bad_results)
        result_tbl.remove_rows(bad_results)
    except KeyError as err:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of negative pixel "
            f"coordinates... {style.Bcolors.ENDC}"
        ) from err

    if n_bad_objects != 0:
        out_str = f"{n_bad_objects} objects removed because of poor quality"
        if terminal_logger is not None:
            terminal_logger.add_to_cache(out_str, indent=indent + 1)
        else:
            terminal_output.print_to_terminal(out_str, indent=indent + 1)

    try:
        n_stars = len(result_tbl["flux_fit"].data)
    except KeyError as err:
        raise ExtractionSkipError(
            "PSF photometry table is empty after cleaning objects with "
            "negative pixel coordinates or uncertainties."
        ) from err

    out_str = f"{n_stars} good stars extracted from the image"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            out_str,
            indent=indent + 1,
            style_name="OK",
        )
    else:
        terminal_output.print_to_terminal(out_str, indent=indent + 1, style_name="OK")

    result_tbl = utilities.rm_edge_objects(
        result_tbl,
        data,
        int((size_extraction_region - 1) / 2),
        terminal_logger=terminal_logger,
    )

    filename = f"table_photometry_{identification_str}_PSF.dat"
    result_tbl.write(
        table_dir / filename,
        format="ascii",
        overwrite=True,
    )

    residual_image = photometry.make_residual_image(
        data,
    )

    image.photometry = result_tbl
    image.residual_image = residual_image


#: Default APER radii (aperture, annulus inner, annulus outer) per unit.
#: ``fwhm``: 1 FWHM is about S/N-optimal for Moffat stars and keeps
#: neighbours out; the aperture correction brings the flux to the total.
#: (2 FWHM on the crowded NGC 7789 stacks, FWHM 5.5″: scatter against
#: Stetson's standards 70-100 mmag instead of 28-33.)
APER_RADII_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "fwhm": (1.0, 2.5, 4.0),
    "arcsec": (5.0, 7.0, 10.0),
}

#: Units of the APER radii.
APER_RADII_UNITS = ("fwhm", "arcsec", "pixel")


def resolve_aperture_radii(
    *,
    radius_aperture: float | None,
    inner_annulus_radius: float | None,
    outer_annulus_radius: float | None,
    radii_unit: str,
    fwhm_pix: float | None,
    pixel_scale: float | None,
) -> tuple[float, float, float]:
    """APER radii in pixels.

    ``radii_unit`` ``fwhm`` (multiples of the image FWHM), ``arcsec`` or
    ``pixel``. Radii left at ``None`` take :data:`APER_RADII_DEFAULTS`
    (``pixel`` has none and needs all three).
    """
    if radii_unit not in APER_RADII_UNITS:
        raise ValueError(f"radii_unit must be one of {APER_RADII_UNITS}, got {radii_unit!r}")
    radii = [radius_aperture, inner_annulus_radius, outer_annulus_radius]
    if any(r is None for r in radii):
        if radii_unit not in APER_RADII_DEFAULTS:
            raise ValueError(
                "radii_unit='pixel' needs radius_aperture, inner_annulus_radius and "
                "outer_annulus_radius"
            )
        radii = [d if r is None else r for r, d in zip(radii, APER_RADII_DEFAULTS[radii_unit], strict=True)]
    r_ap, r_in, r_out = (float(r) for r in radii)
    if not 0 < r_ap < r_in < r_out:
        raise ValueError(
            f"APER radii must satisfy 0 < aperture < annulus inner < annulus outer, "
            f"got {r_ap}, {r_in}, {r_out} {radii_unit}"
        )
    return tuple(radius_to_pixels(r, radii_unit, fwhm_pix, pixel_scale) for r in (r_ap, r_in, r_out))


def radius_to_pixels(
    radius: float, radii_unit: str, fwhm_pix: float | None, pixel_scale: float | None
) -> float:
    """``radius`` in ``radii_unit`` (``fwhm`` / ``arcsec`` / ``pixel``) in pixels."""
    if radii_unit == "fwhm":
        if fwhm_pix is None or not np.isfinite(fwhm_pix) or float(fwhm_pix) <= 0:
            raise ValueError("radii_unit='fwhm' needs a positive image FWHM in pixels.")
        return float(radius) * float(fwhm_pix)
    if radii_unit == "arcsec":
        if pixel_scale is None or not np.isfinite(pixel_scale) or float(pixel_scale) <= 0:
            raise ValueError("radii_unit='arcsec' needs the image pixel scale.")
        return float(radius) / float(pixel_scale)
    if radii_unit == "pixel":
        return float(radius)
    raise ValueError(f"radii_unit must be one of {APER_RADII_UNITS}, got {radii_unit!r}")


def define_apertures(
    image: AnalysisImage,
    aperture_radius: float,
    inner_annulus_radius: float,
    outer_annulus_radius: float,
    unit_radii: str,
) -> tuple[CircularAperture, CircularAnnulus]:
    """Define stellar and background apertures."""
    tbl = image.positions
    try:
        x_positions = tbl["x_fit"]
        y_positions = tbl["y_fit"]
    except KeyError:
        x_positions = tbl["x_centroid"]
        y_positions = tbl["y_centroid"]
    positions = list(zip(x_positions, y_positions, strict=True))

    if unit_radii not in ["pixel", "arcsec"]:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nUnit of the aperture radii not valid: "
            f"set it either to pixel or arcsec {style.Bcolors.ENDC}"
        )

    pixel_scale = image.pixel_scale
    if pixel_scale is not None and unit_radii == "arcsec":
        aperture_radius = aperture_radius / pixel_scale
        inner_annulus_radius = inner_annulus_radius / pixel_scale
        outer_annulus_radius = outer_annulus_radius / pixel_scale

    aperture = CircularAperture(positions, r=aperture_radius)
    annulus_aperture = CircularAnnulus(
        positions,
        r_in=inner_annulus_radius,
        r_out=outer_annulus_radius,
    )

    return aperture, annulus_aperture


def apertures_from_xy(
    x,
    y,
    radius_aperture: float,
    inner_annulus_radius: float,
    outer_annulus_radius: float,
) -> tuple[CircularAperture, CircularAnnulus] | None:
    """Circular + annulus apertures at finite ``(x, y)`` (pixel radii)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return None
    positions = list(zip(x[ok], y[ok], strict=True))
    return (
        CircularAperture(positions, r=float(radius_aperture)),
        CircularAnnulus(
            positions,
            r_in=float(inner_annulus_radius),
            r_out=float(outer_annulus_radius),
        ),
    )


def _subtract_neighbours(
    image, data, uncertainty, tbl, aperture, annulus, area, mask, stars_mask, r_ap, terminal_logger, indent,
    n_iter: int = 2,
) -> dict | None:
    """Neighbour-subtracted aperture photometry (DAOPHOT style), in place.

    The stars' total fluxes (first pass / Moffat encircled energy of the
    aperture) give a model image of all detected stars. The sky is the
    clipped annulus median of ``data − model`` (wings of the star and of all
    neighbours removed), the aperture sum is ``Σ(data − model) + own model``
    (only the neighbours removed; the star's own pixels stay data). Two
    passes. In crowded fields with poor seeing, neighbour light in the
    annulus made faint stars too faint by up to 50 mmag (NGC 7789 against
    Stetson's standards). Returns the new sky statistics, ``None`` when the
    FWHM is unknown.
    """
    fwhm = float(getattr(image, "fwhm", float("nan")) or float("nan"))
    if not np.isfinite(fwhm) or fwhm <= 0 or len(tbl) == 0:
        return None
    x = np.asarray(tbl["x_center"], dtype=float)
    y = np.asarray(tbl["y_center"], dtype=float)
    flux = np.asarray(tbl["flux_fit"], dtype=float)
    err = np.asarray(tbl["aperture_sum_err"], dtype=float)
    growth = total_flux_correction(data, uncertainty, x, y, flux, err, fwhm, mask=mask, source_mask=stars_mask)
    beta = float(growth["beta"]) if np.isfinite(growth["beta"]) else 2.5
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    ee_ap = 1.0 - (1.0 + (r_ap / alpha) ** 2) ** (1.0 - beta)
    total = np.clip(np.nan_to_num(flux), 0.0, None) / ee_ap
    sky = None
    for _ in range(max(1, int(n_iter))):
        model, own = moffat_star_model(np.shape(data), x, y, total, fwhm, beta, aperture_radius=r_ap)
        residual = np.asarray(data, dtype=float) - model
        ap_sum = np.asarray(aperture_photometry(residual, aperture, mask=mask)["aperture_sum"], dtype=float) + own
        sky = annulus_sky(residual, uncertainty, annulus, mask=mask, source_mask=stars_mask)
        flux = ap_sum - sky["sky"] * area
        total = np.clip(np.nan_to_num(flux), 0.0, None) / ee_ap
        del model, residual
    tbl["aperture_sum"] = ap_sum
    tbl["annulus_median"] = sky["sky"]
    tbl["aper_bkg"] = sky["sky"] * area
    tbl["flux_fit"] = flux
    msg = f"Neighbours subtracted (Moffat, FWHM {fwhm:.2f} px, β = {beta:.2f})"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(msg, indent=indent)
    else:
        terminal_output.print_to_terminal(msg, indent=indent)
    return sky


def extraction_aperture(
    image: AnalysisImage,
    radius_aperture: float,
    inner_annulus_radius: float,
    outer_annulus_radius: float,
    radii_unit: str = "pixel",
    plot_aperture_positions: bool = False,
    terminal_logger: terminal_output.TerminalLog | None = None,
    file_type_plots: str = "pdf",
    indent: int = 2,
    neighbour_subtraction: bool = True,
) -> None:
    """Perform aperture photometry using the photutils aperture package.

    With ``neighbour_subtraction`` the detected stars are modelled as
    Moffat profiles (image FWHM, β of the growth curve) and the sky and the
    aperture sums are measured again with the neighbours removed (see
    :func:`_subtract_neighbours`).
    """
    ccd = image.read_image()
    data = ccd.data
    uncertainty = _pixel_error(image, ccd)
    filter_ = image.filter_

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            f"Performing aperture photometry ({filter_} image)",
            indent=indent,
        )
    else:
        terminal_output.print_to_terminal(
            f"Performing aperture photometry ({filter_} image)",
            indent=indent,
        )

    aperture, annulus_aperture = define_apertures(
        image,
        radius_aperture,
        inner_annulus_radius,
        outer_annulus_radius,
        radii_unit,
    )

    #   ``aperture_photometry`` silently drops masked pixels from the sum, so
    #   a defect (or a mask grown by resampling) inside the aperture removes
    #   star flux. Fill masked pixels (median in the sky, Moffat fit inside
    #   stars) for the sum and keep the masked fraction of the star's light
    #   per object as a quality flag; pixels that are NaN (no coverage) stay
    #   excluded.
    source_mask = None if ccd.mask is None else np.asarray(ccd.mask, dtype=bool)
    no_coverage = ~np.isfinite(np.asarray(data, dtype=float))
    masked_fraction = aperture_masked_fraction(
        aperture, data, source_mask, fwhm=float(getattr(image, "fwhm", float("nan")) or float("nan"))
    )
    if source_mask is not None and source_mask.any():
        defects = source_mask & ~no_coverage
        uncertainty = fill_masked_uncertainty(data, uncertainty, defects)
        data = fill_masked_pixels(data, defects)
    phot_mask = no_coverage if no_coverage.any() else None

    photometry_tbl = aperture_photometry(
        data,
        aperture,
        mask=phot_mask,
        error=uncertainty,
    )
    photometry_tbl["aperture_masked_fraction"] = masked_fraction

    aperture_area = aperture.area_overlap(data, mask=phot_mask)

    #   Sky: clipped median per annulus (neighbour stars and cosmics in the
    #   annulus do not pull it up). Errors: the map's source part plus the
    #   background part scaled by the scatter measured on empty sky
    #   (correlated noise, faint stars, flat-field structure).
    stars_mask = source_disk_mask(
        np.shape(data),
        np.asarray(photometry_tbl["x_center"]),
        np.asarray(photometry_tbl["y_center"]),
        _source_mask_radius(image, annulus_aperture),
    )
    image.source_mask = stars_mask
    sky = annulus_sky(
        data, uncertainty, annulus_aperture, mask=phot_mask, source_mask=stars_mask
    )
    photometry_tbl["annulus_median"] = sky["sky"]
    photometry_tbl["aper_bkg"] = sky["sky"] * aperture_area
    photometry_tbl["flux_fit"] = (
        photometry_tbl["aperture_sum"] - photometry_tbl["aper_bkg"]
    )
    r_ap_pix = float(np.atleast_1d(aperture.r)[0])
    r_in_pix = float(np.atleast_1d(annulus_aperture.r_in)[0])
    r_out_pix = float(np.atleast_1d(annulus_aperture.r_out)[0])
    if neighbour_subtraction:
        sky = _subtract_neighbours(
            image, data, uncertainty, photometry_tbl, aperture, annulus_aperture,
            aperture_area, phot_mask, stars_mask, r_ap_pix, terminal_logger, indent,
        ) or sky
    detected = np.column_stack(
        [np.asarray(photometry_tbl["x_center"]), np.asarray(photometry_tbl["y_center"])]
    )
    noise_factor, n_empty = background_noise_factor(
        data,
        uncertainty,
        detected,
        r_ap_pix,
        r_in_pix,
        r_out_pix,
        mask=phot_mask,
        source_mask=stars_mask,
        seed=int(getattr(image, "image_id", 0) or 0),
    )
    factor_used = max(1.0, noise_factor) if np.isfinite(noise_factor) else 1.0
    photometry_tbl["sky_sigma_map"] = sky["sigma_map"]
    photometry_tbl["bkg_noise_factor"] = np.full(len(photometry_tbl), noise_factor)
    photometry_tbl["flux_err"] = aperture_flux_errors(
        np.asarray(photometry_tbl["aperture_sum_err"], dtype=float),
        sky["sigma_map"],
        aperture_area,
        sky["n_sky"],
        noise_factor=factor_used,
    )
    msg = (
        f"Empty-sky check ({n_empty} apertures): background noise "
        f"{noise_factor:.2f}x the uncertainty map"
        if np.isfinite(noise_factor)
        else f"Empty-sky check: only {n_empty} empty apertures; background "
        "noise taken from the uncertainty map"
    )
    if terminal_logger is not None:
        terminal_logger.add_to_cache(msg, indent=indent)
    else:
        terminal_output.print_to_terminal(msg, indent=indent)

    photometry_tbl.rename_column("x_center", "x_fit")
    photometry_tbl.rename_column("y_center", "y_fit")

    if masked_fraction.size and float(np.nanmedian(masked_fraction)) > 0.01:
        msg = (
            f"{float(np.nanmedian(masked_fraction)):.1%} of the aperture area is "
            f"masked on the median object ({filter_} image); masked pixels were "
            "filled with the local median. Check the bad-pixel mask of the "
            "reduced frames."
        )
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg, style_name="WARNING", indent=indent)
        else:
            terminal_output.print_to_terminal(
                msg, style_name="WARNING", indent=indent
            )

    if radii_unit == "pixel":
        required_distance_to_edge = int(outer_annulus_radius)
    elif radii_unit == "arcsec":
        pixel_scale = image.pixel_scale
        required_distance_to_edge = int(round(outer_annulus_radius / pixel_scale))
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nException in aperture_extract(): '"
            f"\n'r_unit ({radii_unit}) not known -> Exit {style.Bcolors.ENDC}"
        )

    photometry_tbl = utilities.rm_edge_objects(
        photometry_tbl,
        data,
        required_distance_to_edge,
        terminal_logger=terminal_logger,
    )

    #   Keep faint stars with zero or negative net flux: dropping them biases
    #   flux light curves of faint sources upward. Their magnitudes are NaN.
    flux = np.asarray(photometry_tbl["flux_fit"], dtype=float)
    photometry_tbl = photometry_tbl[np.isfinite(flux)]
    photometry_tbl["nonpositive_flux"] = np.asarray(photometry_tbl["flux_fit"]) <= 0.0

    image.photometry = photometry_tbl

    if plot_aperture_positions:
        r_ap_pix = float(np.atleast_1d(aperture.r)[0])
        r_in_pix = float(np.atleast_1d(annulus_aperture.r_in)[0])
        r_out_pix = float(np.atleast_1d(annulus_aperture.r_out)[0])
        plot_aps = apertures_from_xy(
            photometry_tbl["x_fit"],
            photometry_tbl["y_fit"],
            r_ap_pix,
            r_in_pix,
            r_out_pix,
        )
        if plot_aps is not None:
            start_plot_process(
                plots.plot_apertures,
                (
                    _extraction_qc_dir(image, gallery=False),
                    data,
                    plot_aps[0],
                    plot_aps[1],
                    f"{filter_}_{image.image_id}",
                ),
                {
                    "file_type": file_type_plots,
                    "pixel_scale": image.pixel_scale,
                },
            )

    n_objects = len(flux)
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            f"{n_objects} good objects extracted from the image",
            indent=indent + 1,
        )
    else:
        terminal_output.print_to_terminal(
            f"{n_objects} good objects extracted from the image",
            indent=indent + 1,
        )


def extract_multiprocessing(
    image_series: ImageSeries,
    n_cores_multiprocessing: int | None = None,
    fwhm_object_psf: dict[str, float] | None = None,
    sigma_value_background_clipping: float = 5.0,
    multiplier_background_rms: float = 5.0,
    size_epsf_region: int = 25,
    size_extraction_region_epsf: int = 11,
    epsf_fitter: str = "TRFLSQFitter",
    n_iterations_eps_extraction: int = 1,
    fraction_epsf_stars: float = 0.2,
    maximum_n_eps_stars: int | None = 50,
    oversampling_factor_epsf: int = 4,
    max_n_iterations_epsf_determination: int = 7,
    use_initial_positions_epsf: bool = True,
    object_finder_method: str = "IRAF",
    finder_sharpness_range: tuple[float, float] = (0.2, 1.0),
    finder_roundness_range: tuple[float, float] = (-1.0, 1.0),
    finder_min_separation_fwhm: float = 1.0,
    finder_fwhm_scale_range: tuple[float, float] | None = (0.5, 2.0),
    multiplier_background_rms_epsf: float = 5.0,
    multiplier_grouper_epsf: float = 2.0,
    psf_group_wing_fwhm: float | None = 2.5,
    psf_residual_sky: bool = True,
    minimum_n_eps_stars: int = 15,
    photometry_extraction_method: str = "PSF",
    psf_find_in_residuals: bool = False,
    radius_aperture: float | None = None,
    inner_annulus_radius: float | None = None,
    outer_annulus_radius: float | None = None,
    radii_unit: str = "fwhm",
    aperture_neighbour_subtraction: bool = True,
    strict_epsf_checks: bool = True,
    plots_for_all_images: bool = False,
    use_wcs_projection_for_star_maps: bool = True,
    file_type_plots: str = "pdf",
    fwhm_estimate_min: float = 2.0,
    fwhm_estimate_max: float = 15.0,
    saturation_fraction: float = 0.9,
) -> None:
    """
    Extract flux and object positions using multiprocessing
    """
    filter_ = image_series.filter_

    if fwhm_object_psf is not None:
        fwhm = fwhm_object_psf[filter_]
    else:
        fwhm = None

    executor = Executor(n_cores_multiprocessing)

    for image in image_series.image_list:
        executor.schedule(
            main_extract,
            args=(image,),
            kwargs={
                "fwhm_object_psf": fwhm,
                "multiprocessing": True,
                "sigma_value_background_clipping": sigma_value_background_clipping,
                "multiplier_background_rms": multiplier_background_rms,
                "size_epsf_region": size_epsf_region,
                "size_extraction_region_epsf": size_extraction_region_epsf,
                "epsf_fitter": epsf_fitter,
                "n_iterations_eps_extraction": n_iterations_eps_extraction,
                "fraction_epsf_stars": fraction_epsf_stars,
                "maximum_n_eps_stars": maximum_n_eps_stars,
                "oversampling_factor_epsf": oversampling_factor_epsf,
                "max_n_iterations_epsf_determination": max_n_iterations_epsf_determination,
                "use_initial_positions_epsf": use_initial_positions_epsf,
                "object_finder_method": object_finder_method,
                "finder_sharpness_range": finder_sharpness_range,
                "finder_roundness_range": finder_roundness_range,
                "finder_min_separation_fwhm": finder_min_separation_fwhm,
                "finder_fwhm_scale_range": finder_fwhm_scale_range,
                "multiplier_background_rms_epsf": multiplier_background_rms_epsf,
                "multiplier_grouper_epsf": multiplier_grouper_epsf,
                "psf_group_wing_fwhm": psf_group_wing_fwhm,
                "psf_residual_sky": psf_residual_sky,
                "minimum_n_eps_stars": minimum_n_eps_stars,
                "strict_epsf_checks": strict_epsf_checks,
                "id_reference_image": image_series.reference_image_index,
                "photometry_extraction_method": photometry_extraction_method,
                "psf_find_in_residuals": psf_find_in_residuals,
                "radius_aperture": radius_aperture,
                "inner_annulus_radius": inner_annulus_radius,
                "outer_annulus_radius": outer_annulus_radius,
                "radii_unit": radii_unit,
                "aperture_neighbour_subtraction": aperture_neighbour_subtraction,
                "plots_for_all_images": plots_for_all_images,
                "file_type_plots": file_type_plots,
                "use_wcs_projection_for_star_maps": use_wcs_projection_for_star_maps,
                "fwhm_estimate_min": fwhm_estimate_min,
                "fwhm_estimate_max": fwhm_estimate_max,
                "saturation_fraction": saturation_fraction,
            },
        )

    executor.wait()
    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Extraction using multiprocessing failed "
            f"for {filter_} :({style.Bcolors.ENDC}"
        ) from executor.err

    skipped = _merge_extraction_results(image_series, executor.res)
    if skipped:
        terminal_output.print_to_terminal(
            f"Skipped {len(skipped)} {filter_} image(s) "
            f"(ids: {', '.join(str(i) for i in skipped)}).",
            style_name="WARNING",
        )


def main_extract(
    image: AnalysisImage,
    fwhm_object_psf: float | None = None,
    multiprocessing: bool = False,
    sigma_value_background_clipping: float = 5.0,
    multiplier_background_rms: float = 5.0,
    size_epsf_region: int = 25,
    size_extraction_region_epsf: int = 11,
    epsf_fitter: str = "TRFLSQFitter",
    n_iterations_eps_extraction: int = 1,
    fraction_epsf_stars: float = 0.2,
    maximum_n_eps_stars: int | None = 50,
    oversampling_factor_epsf: int = 4,
    max_n_iterations_epsf_determination: int = 7,
    use_initial_positions_epsf: bool = True,
    object_finder_method: str = "IRAF",
    finder_sharpness_range: tuple[float, float] = (0.2, 1.0),
    finder_roundness_range: tuple[float, float] = (-1.0, 1.0),
    finder_min_separation_fwhm: float = 1.0,
    finder_fwhm_scale_range: tuple[float, float] | None = (0.5, 2.0),
    multiplier_background_rms_epsf: float = 5.0,
    multiplier_grouper_epsf: float = 2.0,
    psf_group_wing_fwhm: float | None = 2.5,
    psf_residual_sky: bool = True,
    minimum_n_eps_stars: int = 15,
    id_reference_image: int = 0,
    photometry_extraction_method: str = "PSF",
    psf_find_in_residuals: bool = False,
    radius_aperture: float | None = None,
    inner_annulus_radius: float | None = None,
    outer_annulus_radius: float | None = None,
    radii_unit: str = "fwhm",
    aperture_neighbour_subtraction: bool = True,
    strict_epsf_checks: bool = True,
    cosmic_ray_removal: bool | str = "auto",
    limiting_contrast_rm_cosmics: float = 5.0,
    read_noise: float | None = None,
    sigma_clipping_value: float = 4.5,
    saturation_level: float | None = None,
    plots_for_all_images: bool = False,
    file_type_plots: str = "pdf",
    use_wcs_projection_for_star_maps: bool = True,
    fwhm_estimate_min: float = 2.0,
    fwhm_estimate_max: float = 15.0,
    saturation_fraction: float = 0.9,
) -> None | tuple[int, Table | None]:
    """
    Main function to extract the information from the individual images
    """
    if multiprocessing:
        terminal_logger = terminal_output.TerminalLog()
        terminal_logger.add_to_cache(
            f"Image: {image.image_id}",
            style_name="UNDERLINE",
        )
    else:
        terminal_output.print_to_terminal(
            f"Image: {image.image_id}",
            indent=2,
            style_name="UNDERLINE",
        )
        terminal_logger = None

    run_cosmics, force_cosmics = _should_run_cosmic_removal(image, cosmic_ray_removal)
    if run_cosmics:
        rm_cosmic_rays(
            image,
            limiting_contrast=limiting_contrast_rm_cosmics,
            read_noise=read_noise,
            sigma_clipping_value=sigma_clipping_value,
            saturation_level=saturation_level,
            force=force_cosmics,
            terminal_logger=terminal_logger,
        )
    elif normalize_cosmic_ray_removal(cosmic_ray_removal) == "auto":
        # Header already marks cosmics; make the skip visible once.
        msg = (
            "Cosmic rays already identified in FITS header; "
            "skipping lacosmic (cosmic_ray_removal=auto)."
        )
        if terminal_logger is not None:
            terminal_logger.add_to_cache(msg, indent=2)
        else:
            terminal_output.print_to_terminal(msg, indent=2)

    _, rms_background = determine_background(
        image,
        sigma_background=sigma_value_background_clipping,
    )

    find_stars(
        image,
        rms_background,
        fwhm_object_psf=fwhm_object_psf,
        multiplier_background_rms=multiplier_background_rms,
        method=object_finder_method,
        terminal_logger=terminal_logger,
        fwhm_estimate_min=fwhm_estimate_min,
        fwhm_estimate_max=fwhm_estimate_max,
        finder_sharpness_range=finder_sharpness_range,
        finder_roundness_range=finder_roundness_range,
        finder_min_separation_fwhm=finder_min_separation_fwhm,
        finder_fwhm_scale_range=finder_fwhm_scale_range,
    )

    gallery = image.image_id != id_reference_image
    plot_dir = _extraction_qc_dir(image, gallery=gallery)

    try:
        if photometry_extraction_method == "PSF":
            if size_epsf_region % 2 == 0:
                size_epsf_region = size_epsf_region + 1

            epsf_stars = check_epsf_stars(
                image,
                size_epsf_region=size_epsf_region,
                minimum_n_stars=minimum_n_eps_stars,
                fraction_epsf_stars=fraction_epsf_stars,
                maximum_n_stars=maximum_n_eps_stars,
                terminal_logger=terminal_logger,
                strict_epsf_checks=strict_epsf_checks,
                saturation_fraction=saturation_fraction,
            )

            if plots_for_all_images or image.image_id == id_reference_image:
                start_plot_process(
                    plots.starmap,
                    (plot_dir, image.get_data(), image.filter_, image.positions),
                    {
                        "tbl_2": epsf_stars,
                        "label": "identified stars",
                        "label_2": "stars used to determine the ePSF",
                        "rts": (
                            f"Initial object identification [Image: {image.image_id}"
                            f" ({image.filename})]"
                        ),
                        "filename_suffix": (
                            f"Initial object identification [Image: {image.image_id}]"
                        ),
                        "wcs_image": image.wcs,
                        "use_wcs_projection": use_wcs_projection_for_star_maps,
                        "file_type": file_type_plots,
                    },
                )

            determine_epsf(
                image,
                epsf_stars,
                size_epsf_region=size_epsf_region,
                oversampling_factor=oversampling_factor_epsf,
                max_n_iterations=max_n_iterations_epsf_determination,
                minimum_n_stars=minimum_n_eps_stars,
                multiprocess_plots=True,
                terminal_logger=terminal_logger,
                file_type_plots=file_type_plots,
                plot_output_dir=plot_dir,
            )

            start_plot_process(
                plots.plot_epsf,
                (plot_dir, {f"img-{image.image_id}-{image.filter_}": [image.epsf]}),
                {
                    "file_type": file_type_plots,
                    "id_image": f"_{image.image_id}_{image.filter_}",
                    "indent": 2,
                },
            )

            extraction_epsf(
                image,
                rms_background,
                sigma_background=sigma_value_background_clipping,
                use_initial_positions=use_initial_positions_epsf,
                finder_method=object_finder_method,
                size_extraction_region=size_extraction_region_epsf,
                epsf_fitter=epsf_fitter,
                n_iterations_eps_extraction=n_iterations_eps_extraction,
                multiplier_background_rms=multiplier_background_rms_epsf,
                multiplier_grouper=multiplier_grouper_epsf,
                terminal_logger=terminal_logger,
                finder_sharpness_range=finder_sharpness_range,
                finder_roundness_range=finder_roundness_range,
                finder_min_separation_fwhm=finder_min_separation_fwhm,
                psf_find_in_residuals=psf_find_in_residuals,
                group_wing_fwhm=psf_group_wing_fwhm,
                residual_sky=psf_residual_sky,
            )

            start_plot_process(
                plots.plot_residual,
                (
                    {f"{image.filter_}, Image ID: {image.image_id}": image.get_data()},
                    {
                        f"{image.filter_}, Image ID: {image.image_id}": (
                            image.residual_image
                        )
                    },
                    plot_dir,
                ),
                {"file_type": file_type_plots, "indent": 2},
            )

        elif photometry_extraction_method == "APER":
            if image.positions is None or len(image.positions) == 0:
                raise ExtractionSkipError(
                    f"No sources identified in the {image.image_id}. "
                    f"{image.filter_} band image."
                )
            if image.image_id == id_reference_image:
                plot_aperture_positions = True
            else:
                plot_aperture_positions = False

            radius_aperture_use, inner_annulus_use, outer_annulus_use = resolve_aperture_radii(
                radius_aperture=radius_aperture,
                inner_annulus_radius=inner_annulus_radius,
                outer_annulus_radius=outer_annulus_radius,
                radii_unit=radii_unit,
                fwhm_pix=getattr(image, "fwhm", None),
                pixel_scale=getattr(image, "pixel_scale", None),
            )
            msg = (
                f"APER radii: r = {radius_aperture_use:.2f} px, annulus "
                f"{inner_annulus_use:.2f}–{outer_annulus_use:.2f} px "
                f"(FWHM {float(getattr(image, 'fwhm', float('nan')) or float('nan')):.2f} px, "
                f"radii_unit={radii_unit})"
            )
            if terminal_logger is not None:
                terminal_logger.add_to_cache(msg, indent=3)
            else:
                terminal_output.print_to_terminal(msg, indent=3)

            extraction_aperture(
                image,
                radius_aperture_use,
                inner_annulus_use,
                outer_annulus_use,
                neighbour_subtraction=aperture_neighbour_subtraction,
                radii_unit="pixel",
                plot_aperture_positions=plot_aperture_positions,
                terminal_logger=terminal_logger,
                file_type_plots=file_type_plots,
                indent=3,
            )

        else:
            raise RuntimeError(
                f"{style.Bcolors.FAIL} \nExtraction method "
                f"({photometry_extraction_method}) not "
                f"valid: use either APER or PSF {style.Bcolors.ENDC}"
            )

        if image.photometry is None or len(image.photometry) == 0:
            raise ExtractionSkipError(
                f"No photometric measurements remain for image {image.image_id} "
                f"({image.filter_})."
            )

        if photometry_extraction_method == "APER":
            radii_pix = (radius_aperture_use, inner_annulus_use, outer_annulus_use)
        else:
            fwhm_px = float(getattr(image, "fwhm", 4.0) or 4.0)
            radii_pix = (
                max(2.0, 1.5 * fwhm_px),
                max(7.0, 2.8 * fwhm_px),
                max(10.0, 4.0 * fwhm_px),
            )
        _extraction_quality(
            image,
            radii_pix,
            saturation_fraction=saturation_fraction,
            terminal_logger=terminal_logger,
        )

        flux = np.asarray(
            getattr(image.photometry["flux_fit"], "value", image.photometry["flux_fit"]),
            dtype=float,
        )
        flux_err = np.asarray(
            getattr(image.photometry["flux_err"], "value", image.photometry["flux_err"]),
            dtype=float,
        )
        positive = flux > 0.0
        magnitudes = np.full(flux.shape, np.nan)
        magnitudes_error = np.full(flux.shape, np.nan)
        magnitudes[positive], magnitudes_error[positive] = utilities.flux_to_magnitudes(
            flux[positive], flux_err[positive]
        )

        image.photometry["mags_fit"] = magnitudes
        image.photometry["mags_unc"] = magnitudes_error
        image.photometry = utilities.attach_finder_quality(
            image.photometry,
            image.positions,
        )
        image.photometry = utilities.attach_sky_coords_from_wcs(
            image.photometry,
            getattr(image, "wcs", None),
        )

        method_label = {
            "APER": "aperture photometry",
            "PSF": "PSF photometry",
        }.get(
            str(photometry_extraction_method).upper(),
            str(photometry_extraction_method),
        )

        if plots_for_all_images or image.image_id == id_reference_image:
            utilities.prepare_and_plot_starmap(
                image,
                terminal_logger=terminal_logger,
                file_type_plots=file_type_plots,
                label=f"Stars with photometric extractions ({method_label})",
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
                gallery=gallery,
            )
    except ExtractionSkipError as err:
        skipped = _skip_extraction_result(
            image, err, terminal_logger, multiprocessing
        )
        if skipped is not None:
            return skipped

    if multiprocessing:
        terminal_logger.print_to_terminal("")
    else:
        terminal_output.print_to_terminal("")

    if multiprocessing:
        return image.image_id, image.photometry


__all__ = [
    "ExtractionSkipError",
    "apertures_from_xy",
    "main_extract",
    "extract_multiprocessing",
    "resolve_aperture_radii",
]
