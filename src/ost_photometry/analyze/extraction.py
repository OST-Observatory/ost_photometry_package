"""
Photometry extraction: main_extract, extract_multiprocessing, and helpers.

Extracted from analyze.py for modular reuse.
"""

import multiprocessing as mp
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
from astropy.nddata import NDData
from astropy.stats import SigmaClip
from astropy.table import Table
from photutils.aperture import (
    ApertureStats,
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
from ..core.parallel import Executor
from ..fits_headers import (
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
from . import correlate, plots, utilities
from .image import AnalysisImage
from .models import ImageSeries
from .utils.epsf_selection import n_epsf_stars_to_select


def rm_cosmic_rays(
    image: AnalysisImage,
    limiting_contrast: float = 5.0,
    read_noise: float = 8.0,
    sigma_clipping_value: float = 4.5,
    saturation_level: float = 65535.0,
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

    if terminal_logger is not None:
        terminal_logger.add_to_cache("Remove cosmic rays ...")
    else:
        terminal_output.print_to_terminal("Remove cosmic rays ...")

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
        image_no_bg = ccd.subtract(bkg.background * u.electron / u.s)

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
        image_no_bg = ccd.subtract(background_value)

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

    #   Number of objects
    n_stars = len(tbl_positions)

    #   Get image data
    image_data = image.get_data()

    #   Combine identification string
    identification_string = f"{image.image_id}. {image.filter_}"

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
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ("
            f"{id_percentile_99 - available_epsf_stars}) found to determine "
            f"the ePSF in the {identification_string} band{style.Bcolors.ENDC}"
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
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({n_useful_epsf_stars}) "
            f"for the ePSF determination in the {identification_string} band "
            "image. Too many potential ePSF stars have been removed, because "
            "they are too close to the image border. Check first that enough "
            "stars have been identified, using the starmap_?.pdf files.\n If "
            "that is the case, shrink extraction region or raise "
            "fraction_epsf_stars / maximum_n_eps_stars "
            f"(size_epsf_region). {style.Bcolors.ENDC}"
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
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({n_useful_epsf_stars}) "
            f"for the ePSF determination in the {identification_string} band "
            "image. Too many potential ePSF stars have been removed because "
            "their cutouts contain non-finite pixels. {style.Bcolors.ENDC}"
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
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({n_useful_epsf_stars}) "
            f" for the ePSF determination in the {identification_string} band "
            "image. Too many potential ePSF stars have been removed, because "
            "other stars are in the extraction region. Check first that enough"
            " stars have been identified, using the starmap_?.pdf files.\n"
            "If that is the case, shrink extraction region or raise "
            "fraction_epsf_stars / maximum_n_eps_stars "
            f"(size_epsf_region). {style.Bcolors.ENDC}"
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
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No ePSF stars remain after rejecting "
            f"non-finite cutouts (image {image.image_id}, filter "
            f"{image.filter_}). {style.Bcolors.ENDC}"
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
    output_dir = image.out_path.name

    #   Plot the brightest ePSF stars
    if multiprocess_plots:
        p = mp.Process(
            target=plots.plot_cutouts,
            args=(output_dir, stars, string),
            kwargs={
                "file_type": file_type_plots,
            },
        )
        p.start()
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
    strict_cleaning_results: bool = True,
    terminal_logger: terminal_output.TerminalLog | None = None,
    rm_background: bool = False,
    indent: int = 2,
    finder_sharpness_range: tuple[float, float] = (0.2, 1.0),
    finder_roundness_range: tuple[float, float] = (-1.0, 1.0),
    finder_min_separation_fwhm: float = 1.0,
    psf_find_in_residuals: bool = False,
) -> None:
    """
    Main function to perform the eEPSF photometry, using photutils
    """
    output_path = image.out_path

    checks.check_output_directories(
        output_path,
        output_path / "tables",
    )

    data = image.get_data()
    error = image.get_error()
    image_mask = image.get_mask()
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

    if rm_background:
        sigma_clip = SigmaClip(sigma=sigma_background)
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
        local_bkg_estimator = LocalBackground(7, 10, mmm_bkg)
    else:
        local_bkg_estimator = None

    source_grouper = SourceGrouper(min_separation=multiplier_grouper * fwhm)

    fit_shape = (size_extraction_region, size_extraction_region)
    aperture_radius = (size_extraction_region - 1) / 2

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
        )

    #   Check if error is finite
    finite_error_mask = np.isfinite(error)
    error[np.invert(finite_error_mask)] = np.max(error[finite_error_mask])
    finite_error_mask = np.isfinite(error)

    #   Check if error is negative
    negative_error_mask = error < 0.0
    error[negative_error_mask] = np.max(error[np.invert(negative_error_mask)])

    #   Check if error is nan
    nan_error_mask = np.isnan(error)
    error[np.invert(nan_error_mask)] = np.max(error[np.invert(nan_error_mask)])

    if use_initial_positions:
        result_tbl = photometry(
            data=data,
            error=error,
            mask=image_mask,
            init_params=initial_positions,
        )
    else:
        result_tbl = photometry(data=data, error=error, mask=image_mask)

    if "flux_err" not in result_tbl.colnames:
        estimated_uncertainty = np.absolute(
            result_tbl["flux_fit"] - result_tbl["flux_init"]
        )
        result_tbl.add_column(estimated_uncertainty, name="flux_err")

    try:
        uncertainty_mask = np.invert(np.isnan(result_tbl["flux_err"].value))
        result_tbl = result_tbl[uncertainty_mask]
    except KeyError as err:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of NANs in "
            f"uncertainties... {style.Bcolors.ENDC}"
        ) from err

    n_bad_objects = 0
    try:
        bad_results = np.where(result_tbl["flux_fit"].data < 0.0)
        result_tbl.remove_rows(bad_results)
        n_bad_objects = np.size(bad_results)
        if strict_cleaning_results:
            bad_results = np.where(result_tbl["flux_err"].data < 0.0)
            n_bad_objects += len(bad_results)
            result_tbl.remove_rows(bad_results)
    except KeyError as err:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of negative "
            f"uncertainties... {style.Bcolors.ENDC}"
        ) from err

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
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nTable produced by "
            "IterativePSFPhotometry is empty after cleaning up "
            "of objects with negative pixel coordinates and negative "
            f"uncertainties {style.Bcolors.ENDC}"
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
        output_path / "tables" / filename,
        format="ascii",
        overwrite=True,
    )

    residual_image = photometry.make_residual_image(
        data,
    )

    image.photometry = result_tbl
    image.residual_image = residual_image


def compute_aperture_photometry_uncertainties(
    flux_variance: np.ndarray,
    aperture_area: float,
    annulus_area: float,
    uncertainty_background: np.ndarray,
    gain: float = 1.0,
) -> np.ndarray:
    """Compute flux errors for aperture photometry using DAOPHOT-style computation."""
    bg_variance_terms = (aperture_area * uncertainty_background**2.0) * (
        1.0 + aperture_area / annulus_area
    )
    variance = flux_variance / gain + bg_variance_terms
    flux_error = variance**0.5
    return flux_error


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


def extraction_aperture(
    image: AnalysisImage,
    radius_aperture: float,
    inner_annulus_radius: float,
    outer_annulus_radius: float,
    radii_unit: str = "pixel",
    background_estimate_simple: bool = False,
    plot_aperture_positions: bool = False,
    terminal_logger: terminal_output.TerminalLog | None = None,
    file_type_plots: str = "pdf",
    indent: int = 2,
) -> None:
    """Perform aperture photometry using the photutils aperture package."""
    ccd = image.read_image()
    data = ccd.data
    uncertainty = ccd.uncertainty.array
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

    photometry_tbl = aperture_photometry(
        data,
        aperture,
        mask=ccd.mask,
        error=uncertainty,
    )

    aperture_area = aperture.area_overlap(data, mask=ccd.mask)
    annulus_aperture_area = annulus_aperture.area_overlap(data, mask=ccd.mask)

    if background_estimate_simple:
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        bkg_stats = ApertureStats(data, annulus_aperture, sigma_clip=sigma_clip)
        bkg_median = bkg_stats.median
        bkg_err = bkg_stats.std
        photometry_tbl["annulus_median"] = bkg_median
        photometry_tbl["aper_bkg"] = bkg_median * aperture_area
    else:
        bkg_phot = aperture_photometry(
            data,
            annulus_aperture,
            mask=ccd.mask,
            error=uncertainty,
        )
        photometry_tbl["aper_bkg"] = (
            bkg_phot["aperture_sum"] * aperture_area / annulus_aperture_area
        )
        bkg_err = photometry_tbl["aper_bkg_err"] = (
            bkg_phot["aperture_sum_err"] * aperture_area / annulus_aperture_area
        )

    photometry_tbl["flux_fit"] = (
        photometry_tbl["aperture_sum"] - photometry_tbl["aper_bkg"]
    )
    
    if uncertainty is not None:
        err_column = photometry_tbl["aperture_sum_err"]
    else:
        err_column = photometry_tbl["flux_fit"] ** 0.5

    photometry_tbl["flux_err"] = compute_aperture_photometry_uncertainties(
        err_column,
        aperture_area,
        annulus_aperture_area,
        bkg_err,
    )

    photometry_tbl.rename_column("x_center", "x_fit")
    photometry_tbl.rename_column("y_center", "y_fit")

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

    flux = np.array(photometry_tbl["flux_fit"])
    mask = np.argwhere(flux > 0.0).ravel()
    photometry_tbl = photometry_tbl[mask]

    image.photometry = photometry_tbl

    if plot_aperture_positions:
        plots.plot_apertures(
            image.out_path.name,
            data,
            aperture,
            annulus_aperture,
            f"{filter_}_{image.image_id}",
            file_type=file_type_plots,
            pixel_scale=image.pixel_scale,
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
    n_cores_multiprocessing: int,
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
    strict_cleaning_epsf_results: bool = True,
    minimum_n_eps_stars: int = 15,
    photometry_extraction_method: str = "PSF",
    psf_find_in_residuals: bool = False,
    radius_aperture: float = 5.0,
    inner_annulus_radius: float = 7.0,
    outer_annulus_radius: float = 10.0,
    radii_unit: str = "arcsec",
    strict_epsf_checks: bool = True,
    plots_for_all_images: bool = False,
    use_wcs_projection_for_star_maps: bool = True,
    file_type_plots: str = "pdf",
    fwhm_estimate_min: float = 2.0,
    fwhm_estimate_max: float = 15.0,
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
                "strict_cleaning_epsf_results": strict_cleaning_epsf_results,
                "minimum_n_eps_stars": minimum_n_eps_stars,
                "strict_epsf_checks": strict_epsf_checks,
                "id_reference_image": image_series.reference_image_index,
                "photometry_extraction_method": photometry_extraction_method,
                "psf_find_in_residuals": psf_find_in_residuals,
                "radius_aperture": radius_aperture,
                "inner_annulus_radius": inner_annulus_radius,
                "outer_annulus_radius": outer_annulus_radius,
                "radii_unit": radii_unit,
                "plots_for_all_images": plots_for_all_images,
                "file_type_plots": file_type_plots,
                "use_wcs_projection_for_star_maps": use_wcs_projection_for_star_maps,
                "fwhm_estimate_min": fwhm_estimate_min,
                "fwhm_estimate_max": fwhm_estimate_max,
            },
        )

    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Extraction using multiprocessing failed "
            f"for {filter_} :({style.Bcolors.ENDC}"
        )

    executor.wait()
    res = executor.res

    tmp_list = []
    for img in image_series.image_list:
        for img_id, tbl in res:
            if img_id == img.image_id:
                img.photometry = tbl
                tmp_list.append(img)

    image_series.image_list = tmp_list


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
    strict_cleaning_epsf_results: bool = True,
    minimum_n_eps_stars: int = 15,
    id_reference_image: int = 0,
    photometry_extraction_method: str = "PSF",
    psf_find_in_residuals: bool = False,
    radius_aperture: float = 4.0,
    inner_annulus_radius: float = 7.0,
    outer_annulus_radius: float = 10.0,
    radii_unit: str = "arcsec",
    strict_epsf_checks: bool = True,
    cosmic_ray_removal: bool | str = "auto",
    limiting_contrast_rm_cosmics: float = 5.0,
    read_noise: float = 8.0,
    sigma_clipping_value: float = 4.5,
    saturation_level: float = 65535.0,
    plots_for_all_images: bool = False,
    file_type_plots: str = "pdf",
    use_wcs_projection_for_star_maps: bool = True,
    fwhm_estimate_min: float = 2.0,
    fwhm_estimate_max: float = 15.0,
) -> None | tuple[int, Table]:
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
        )

        if plots_for_all_images or image.image_id == id_reference_image:
            plots.starmap(
                image.out_path.name,
                image.get_data(),
                image.filter_,
                image.positions,
                tbl_2=epsf_stars,
                label="identified stars",
                label_2="stars used to determine the ePSF",
                rts=(
                    f"Initial object identification [Image: {image.image_id}"
                    f" ({image.filename})]"
                ),
                filename_suffix=(
                    f"Initial object identification [Image: {image.image_id}]"
                ),
                wcs_image=image.wcs,
                use_wcs_projection=use_wcs_projection_for_star_maps,
                terminal_logger=terminal_logger,
                file_type=file_type_plots,
            )

        determine_epsf(
            image,
            epsf_stars,
            size_epsf_region=size_epsf_region,
            oversampling_factor=oversampling_factor_epsf,
            max_n_iterations=max_n_iterations_epsf_determination,
            minimum_n_stars=minimum_n_eps_stars,
            multiprocess_plots=False,
            terminal_logger=terminal_logger,
            file_type_plots=file_type_plots,
        )

        plots.plot_epsf(
            image.out_path.name,
            {f"img-{image.image_id}-{image.filter_}": [image.epsf]},
            terminal_logger=terminal_logger,
            file_type=file_type_plots,
            id_image=f"_{image.image_id}_{image.filter_}",
            indent=2,
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
            strict_cleaning_results=strict_cleaning_epsf_results,
            terminal_logger=terminal_logger,
            finder_sharpness_range=finder_sharpness_range,
            finder_roundness_range=finder_roundness_range,
            finder_min_separation_fwhm=finder_min_separation_fwhm,
            psf_find_in_residuals=psf_find_in_residuals,
        )

        plots.plot_residual(
            {f"{image.filter_}, Image ID: {image.image_id}": image.get_data()},
            {f"{image.filter_}, Image ID: {image.image_id}": image.residual_image},
            image.out_path.name,
            terminal_logger=terminal_logger,
            file_type=file_type_plots,
            indent=2,
        )

    elif photometry_extraction_method == "APER":
        if image.image_id == id_reference_image:
            plot_aperture_positions = True
        else:
            plot_aperture_positions = False

        extraction_aperture(
            image,
            radius_aperture,
            inner_annulus_radius,
            outer_annulus_radius,
            radii_unit=radii_unit,
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
    magnitudes, magnitudes_error = utilities.flux_to_magnitudes(
        image.photometry["flux_fit"],
        image.photometry["flux_err"],
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
    }.get(str(photometry_extraction_method).upper(), str(photometry_extraction_method))

    if plots_for_all_images or image.image_id == id_reference_image:
        utilities.prepare_and_plot_starmap(
            image,
            terminal_logger=terminal_logger,
            file_type_plots=file_type_plots,
            label=f"Stars with photometric extractions ({method_label})",
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )

    if multiprocessing:
        terminal_logger.print_to_terminal("")
    else:
        terminal_output.print_to_terminal("")

    if multiprocessing:
        return image.image_id, image.photometry


__all__ = [
    "main_extract",
    "extract_multiprocessing",
]
