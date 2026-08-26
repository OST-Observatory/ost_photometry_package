"""Reduction workflow: science module."""

from pathlib import Path

import astropy.units as u
import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData

from ... import checks, style, terminal_output
from ... import utilities as base_utilities
from ...core.parallel import Executor
from .. import utilities, validation
from .constants import (
    REDUCE_STATUS_REDUCED,
    REDUCE_STATUS_SKIP_NO_FILTER,
    REDUCE_STATUS_SKIP_NO_MASTER_FLAT,
)


def reduce_light(
    image_path: str | Path,
    output_dir: str | Path,
    image_type: dict[str, list[str]],
    rm_cosmic_rays: bool = True,
    mask_cosmics: bool = False,
    gain: float | None = None,
    read_noise: float = 8.0,
    saturation_level: float | None = 65535.0,
    limiting_contrast_rm_cosmic_rays: float = 5.0,
    sigma_clipping_value_rm_cosmic_rays: float = 4.5,
    scale_image_with_exposure_time: bool = True,
    rm_bias: bool = False,
    verbose: bool = False,
    add_hot_bad_pixel_mask: bool = True,
    exposure_time_tolerance: float = 0.5,
    target_name: str | None = None,
    n_cores_multiprocessing: int | None = None,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
    fail_on_missing_flat: bool = True,
) -> None:
    """
    Reduce the science images

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be stored

    image_type
        Image types of the images. Possibilities: bias, dark, flat,
        light

    rm_cosmic_rays
        If True cosmic rays will be removed.
        Default is ``True``.

    mask_cosmics
        If True cosmics will ''only'' be masked. If False the
        cosmics will be removed from the input image and the mask will
        be added.
        Default is ``False``.

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    saturation_level
        Saturation limit of the camera chip.
        Default is ``65535``.

    limiting_contrast_rm_cosmic_rays
        Parameter for the cosmic ray removal: Minimum contrast between
        Laplacian image and the fine structure image.
        Default is ``5``.

    sigma_clipping_value_rm_cosmic_rays
        Parameter for the cosmic ray removal: Fractional detection limit
        for neighboring pixels.
        Default is ``4.5``.

    scale_image_with_exposure_time
        If True the image will be scaled with the exposure time.
        Default is ``True``.

    rm_bias
        If True the master bias image will be subtracted from the flats
        Default is ``False``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    add_hot_bad_pixel_mask
        If True add hot and bad pixel mask to the reduced science
        images.
        Default is ``True``.

    exposure_time_tolerance
        Tolerance between science and dark exposure times in s.
        Default is ``0.5``s.

    target_name
        Name of the target. Used for file selection.
        Default is ``None``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    trim_x_start
        Number of pixels to trim from the start of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_x_end
        Number of pixels to trim from the end of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_start
        Number of pixels to trim from the start of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_end
        Number of pixels to trim from the end of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.
    """
    terminal_output.print_to_terminal("Reduce light images...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Get image collection for the science images
    image_file_collection = utilities.image_file_collection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \tNo object image detected.\n\t"
            f"-> EXIT{style.Bcolors.ENDC}"
        )

    #   Limit images to those of the target. If a target is given.
    if target_name is not None:
        image_file_collection = image_file_collection.filter(object=target_name)

    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \tERROR: No image left after filtering by "
            f"object name.\n\t-> EXIT{style.Bcolors.ENDC}"
        )

    #   Find science images
    light_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class="light",
    )
    if not light_image_type:
        return
    light_files = list(
        image_file_collection.files_filtered(imagetyp=light_image_type)
    )
    if not light_files:
        return

    #   Get image collection for the reduced files
    image_file_collection_reduced = utilities.image_file_collection(out_path)

    #   Load combined darks and flats in dictionary for easy access
    dark_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class="dark",
    )
    combined_darks: dict[float, CCDData] = {
        ccd.header["exptime"]: ccd
        for ccd in image_file_collection_reduced.ccds(
            imagetyp=dark_image_type,
            combined=True,
        )
    }
    flat_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class="flat",
    )
    combined_flats: dict[str, CCDData] = {
        ccd.header["filter"]: ccd
        for ccd in image_file_collection_reduced.ccds(
            imagetyp=flat_image_type,
            combined=True,
        )
    }

    #   Get master bias
    combined_bias: CCDData | None = None
    if rm_bias:
        bias_image_type = utilities.get_image_type(
            image_file_collection_reduced,
            image_type,
            image_class="bias",
        )

        combined_bias = CCDData.read(
            image_file_collection_reduced.files_filtered(
                imagetyp=bias_image_type,
                combined=True,
                include_path=True,
            )[0]
        )

    #   Set science image path
    light_path = Path(out_path / "light")

    dir_empty = checks.check_if_directory_is_empty(light_path)

    if not dir_empty:
        user_input, timed_out = base_utilities.get_input(
            f"{style.Bcolors.OKBLUE}   Reduced images from a previous run "
            f"found. Should these be used? [yes/no] {style.Bcolors.ENDC}"
        )
        if user_input in ["y", "yes"]:
            return

    checks.clear_directory(light_path)

    #   Get possible image types
    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(image_file_collection.files_filtered(imagetyp=light_image_type)),
        add_progress_bar=True,
    )

    #   Reduce science images and save to an extra directory
    for file_name in image_file_collection.files_filtered(
        include_path=True,
        imagetyp=light_image_type,
        # ccd_kwargs=dict(unit='adu'),
    ):
        executor.schedule(
            reduce_light_image,
            args=(
                file_name,
                combined_bias,
                combined_darks,
                combined_flats,
                out_path,
                light_path,
            ),
            kwargs={
                "gain": gain,
                "read_noise": read_noise,
                "rm_bias": rm_bias,
                "exposure_time_tolerance": exposure_time_tolerance,
                "add_hot_bad_pixel_mask": add_hot_bad_pixel_mask,
                "rm_cosmic_rays": rm_cosmic_rays,
                "limiting_contrast_rm_cosmic_rays": limiting_contrast_rm_cosmic_rays,
                "sigma_clipping_value_rm_cosmic_rays": sigma_clipping_value_rm_cosmic_rays,
                "saturation_level": saturation_level,
                "mask_cosmics": mask_cosmics,
                "scale_image_with_exposure_time": scale_image_with_exposure_time,
                "verbose": verbose,
                "trim_x_start": trim_x_start,
                "trim_x_end": trim_x_end,
                "trim_y_start": trim_y_start,
                "trim_y_end": trim_y_end,
            },
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Light image reduction using multiprocessing"
            f" failed :({style.Bcolors.ENDC}"
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    validation.summarize_light_reduction_results(
        executor.res,
        fail_on_missing_flat=fail_on_missing_flat,
    )


def reduce_light_image(
    light_file_name: str,
    combined_bias: CCDData | None,
    combined_darks: dict[float, CCDData],
    combined_flats: dict[str, CCDData],
    out_path: Path,
    light_path: Path,
    gain: float | None = None,
    read_noise: float = 8.0,
    rm_bias: bool = False,
    exposure_time_tolerance: float = 0.5,
    add_hot_bad_pixel_mask: bool = True,
    rm_cosmic_rays: bool = True,
    limiting_contrast_rm_cosmic_rays: float = 5.0,
    sigma_clipping_value_rm_cosmic_rays: float = 4.5,
    saturation_level: float | None = 65535.0,
    mask_cosmics: bool = False,
    scale_image_with_exposure_time: bool = True,
    verbose: bool = False,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
) -> str:
    """
    Reduce an individual image

    Parameters
    ----------
    light_file_name
        The CCDData object that should be reduced.

    combined_bias
        Reduced and stacked Bias CCDData object

    combined_darks
        Combined darks in a dictionary with exposure times as keys and
        CCDData object as values.

    combined_flats
        Combined flats in a dictionary with exposure times as keys and
        CCDData object as values.

    out_path
        Path to the general output directory

    light_path
        Path where the reduced images should be saved

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    rm_bias
        If True the master bias image will be subtracted from the flats
        Default is ``False``.

    exposure_time_tolerance
        Tolerance between science and dark exposure times in s.
        Default is ``0.5``s.

    add_hot_bad_pixel_mask
        If True add hot and bad pixel mask to the reduced science
        images.
        Default is ``True``.

    rm_cosmic_rays
        If True cosmic rays will be removed.
        Default is ``True``.

    limiting_contrast_rm_cosmic_rays
        Parameter for the cosmic ray removal: Minimum contrast between
        Laplacian image and the fine structure image.
        Default is ``5``.

    sigma_clipping_value_rm_cosmic_rays
        Parameter for the cosmic ray removal: Fractional detection limit
        for neighboring pixels.
        Default is ``4.5``.

    saturation_level
        Saturation limit of the camera chip.
        Default is ``65535``.

    mask_cosmics
        If True cosmics will ''only'' be masked. If False the
        cosmics will be removed from the input image and the mask will
        be added.
        Default is ``False``.

    scale_image_with_exposure_time
        If True the image will be scaled with the exposure time.
        Default is ``True``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    trim_x_start
        Number of pixels to trim from the start of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_x_end
        Number of pixels to trim from the end of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_start
        Number of pixels to trim from the start of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_end
        Number of pixels to trim from the end of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.
    """
    #   Read light image
    light = CCDData.read(light_file_name, unit="adu")

    #   Trimming the image, for example to remove an overscan region
    image_shape = light.data.shape
    light = light[
        trim_y_start : image_shape[0] - trim_y_end,
        trim_x_start : image_shape[1] - trim_x_end,
    ]

    #   Get base file name
    file_name = light_file_name.split("/")[-1]

    #   Set gain -> get it from Header if not provided
    if gain is None:
        try:
            gain = light.header["EGAIN"]
        except KeyError:
            gain = 1.0
            terminal_output.print_to_terminal(
                "WARNING: Gain could not de derived from the "
                "image header. Use 1.0 instead",
                style_name="WARNING",
                indent=2,
            )

    #   Calculated uncertainty
    light = ccdp.create_deviation(
        light,
        gain=gain * u.electron / u.adu,
        readnoise=read_noise * u.electron,
        disregard_nan=True,
    )

    #   Subtract bias
    if rm_bias:
        light = ccdp.subtract_bias(light, combined_bias)

    #   Find the correct dark exposure
    valid_dark_available, closest_dark_exposure_time = (
        utilities.find_nearest_exposure_time_to_reference_image(
            light,
            list(combined_darks.keys()),
            time_tolerance=exposure_time_tolerance,
        )
    )

    #   Exit if no dark with a similar exposure time have been found
    if not valid_dark_available and not rm_bias:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Closest dark exposure time is "
            f"{closest_dark_exposure_time} for science image of exposure "
            f"time {light.header['exptime']}. {style.Bcolors.ENDC}"
        )

    #   Subtract dark
    reduced: CCDData = ccdp.subtract_dark(
        light,
        combined_darks[closest_dark_exposure_time],
        exposure_time="exptime",
        exposure_unit=u.second,
        scale=rm_bias,
    )

    #   Mask negative pixel
    mask = reduced.data < 0.0
    reduced.mask = reduced.mask | mask

    #   Check if the "FILTER" keyword is set in Header
    if "filter" not in reduced.header:
        terminal_output.print_to_terminal(
            f"WARNING: FILTER keyword not found in HEADER. \n Skip file: {file_name}.",
            style_name="WARNING",
            indent=2,
        )
        return REDUCE_STATUS_SKIP_NO_FILTER

    filt = str(reduced.header["filter"])
    if filt not in combined_flats:
        terminal_output.print_to_terminal(
            f"WARNING: No master flat for filter '{filt}'. Skip file: {file_name}.",
            style_name="WARNING",
            indent=2,
        )
        return REDUCE_STATUS_SKIP_NO_MASTER_FLAT

    #   Get master flat field
    flat_master = combined_flats[filt]

    #   Divided science by the master flat
    reduced: CCDData = ccdp.flat_correct(reduced, flat_master)

    if add_hot_bad_pixel_mask:
        #   Get mask of bad and hot pixel
        mask_available, bad_hot_pixel_mask = utilities.get_pixel_mask(
            out_path,
            reduced.shape,
        )

        #   Add bad pixel mask: If there was already a mask, keep it
        if mask_available:
            if reduced.mask is not None:
                reduced.mask = reduced.mask | bad_hot_pixel_mask
            else:
                reduced.mask = bad_hot_pixel_mask

    #   Gain correct data
    reduced = ccdp.gain_correct(reduced, gain * u.electron / u.adu)

    #   Remove cosmic rays
    if rm_cosmic_rays:
        if verbose:
            terminal_output.print_to_terminal(
                f"Remove cosmic rays from image {file_name}"
            )

        #   Sanitize saturation level
        if saturation_level is None:
            terminal_output.print_to_terminal(
                "Saturation level not specified. Assume 16bit == 65535",
                indent=1,
                style_name="WARNING",
            )
            saturation_level = 65535

        reduced_without_cosmics = ccdp.cosmicray_lacosmic(
            reduced,
            objlim=limiting_contrast_rm_cosmic_rays,
            readnoise=read_noise,
            sigclip=sigma_clipping_value_rm_cosmic_rays,
            satlevel=saturation_level,
            verbose=verbose,
        )

        if mask_cosmics:
            if add_hot_bad_pixel_mask:
                reduced.mask = reduced.mask | reduced_without_cosmics.mask

                #   Add a header keyword to indicate that the cosmics have been
                #   masked
                reduced.meta["cosmic_mas"] = True
            else:
                terminal_output.print_to_terminal(
                    "WARNING: mask_cosmics=True requires add_hot_bad_pixel_mask=True; "
                    "cosmic-ray mask not applied.",
                    style_name="WARNING",
                    indent=2,
                )
        else:
            reduced = reduced_without_cosmics
            if not add_hot_bad_pixel_mask:
                reduced.mask = np.zeros(reduced.shape, dtype=bool)

            #   Add header keyword to indicate that cosmics have been removed
            reduced.meta["cosmics_rm"] = True

        if verbose:
            terminal_output.print_to_terminal("")

    #   Scale image with exposure time
    if scale_image_with_exposure_time:
        #   Get exposure time and all meta data
        exposure_time = reduced.header["exptime"]
        reduced_meta = reduced.meta

        #   Scale image
        reduced = reduced.divide(exposure_time * u.second)

        #   Put metadata back on the image, because it is lost while
        #   dividing
        reduced.meta = reduced_meta
        reduced.meta["HIERARCH"] = "Image scaled by exposure time:"
        reduced.meta["HIERARCH"] = "Unit: e-/s/pixel"

        #   Set data units to electron / s
        reduced.unit = u.electron / u.s

    #   Write reduced science image to disk
    reduced.write(light_path / file_name, overwrite=True)
    return REDUCE_STATUS_REDUCED


