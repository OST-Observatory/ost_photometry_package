"""Reduction workflow: flat module."""

import shutil
from pathlib import Path

import astropy.units as u
import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import mad_std

from ... import checks, style, terminal_output
from ...core.parallel import Executor
from .. import plots, utilities


def reduce_flat(
    image_path: str | Path,
    output_dir: str | Path,
    image_type: dict[str, list[str]],
    gain: float | None = None,
    read_noise: float = 8.0,
    rm_bias: bool = False,
    exposure_time_tolerance: float = 0.5,
    n_cores_multiprocessing: int | None = None,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
    **kwargs,
) -> None:
    """
    Reduce flat images: This function reduces the raw flat frames,
                        subtracts master dark and if necessary also
                        master bias

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type
        Image types of the images. Possibilities: bias, dark, flat,
        light

    gain
        The gain (e-/adu) of the camera. If set to `None` the gain will
        be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera.
        Default is 8 e-.

    rm_bias
        If True the master bias image will be subtracted from the flats
        Default is ``False``.

    exposure_time_tolerance
        Maximum difference, in seconds, between the image and the
        closest entry from the exposure time list. Set to ``None`` to
        skip the tolerance test.
        Default is ``0.5``.

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
    terminal_output.print_to_terminal("Reduce flats...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create image collection for the flats
    image_file_collection = utilities.image_file_collection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        return

    flat_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class="flat",
    )
    if not flat_image_type:
        return
    flat_files = list(
        image_file_collection.files_filtered(imagetyp=flat_image_type)
    )
    if not flat_files:
        return

    #   Get image collection for the reduced files
    image_file_collection_reduced = utilities.image_file_collection(out_path)

    #   Get master dark
    dark_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class="dark",
    )
    combined_darks = {
        ccd.header["exptime"]: ccd
        for ccd in image_file_collection_reduced.ccds(
            imagetyp=dark_image_type,
            combined=True,
        )
    }

    #   Get master bias
    combined_bias = None
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

    #   Set new flat path
    flat_path = Path(out_path / "flat")
    checks.clear_directory(flat_path)

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(image_file_collection.files_filtered(imagetyp=flat_image_type)),
        add_progress_bar=True,
    )

    #   Reduce science images and save to an extra directory
    for file_name in image_file_collection.files_filtered(
        include_path=True,
        imagetyp=flat_image_type,
    ):
        executor.schedule(
            reduce_flat_image,
            args=(file_name, combined_bias, combined_darks, flat_path),
            kwargs={
                "gain": gain,
                "read_noise": read_noise,
                "rm_bias": rm_bias,
                "exposure_time_tolerance": exposure_time_tolerance,
                "trim_x_start": trim_x_start,
                "trim_x_end": trim_x_end,
                "trim_y_start": trim_y_start,
                "trim_y_end": trim_y_end,
            },
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Flat image reduction using multiprocessing"
            f" failed :({style.Bcolors.ENDC}"
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()


def reduce_flat_image(
    flat_file_name: str,
    combined_bias: CCDData | None,
    combined_darks: dict[float, CCDData],
    flat_path: Path,
    gain: float | None = None,
    read_noise: float = 8.0,
    rm_bias: bool = False,
    exposure_time_tolerance: float = 0.5,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
) -> None:
    """
    Reduce an individual image

    Parameters
    ----------
    flat_file_name
        The CCDData object of the flat that should be reduced.

    combined_bias
        Reduced and stacked Bias CCDData object

    combined_darks
        Combined darks in a dictionary with exposure times as keys and
        CCDData object as values.

    flat_path
        Path where the reduced flats should be saved

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
    #   Read fla image
    flat = CCDData.read(flat_file_name, unit="adu")

    #   Trimming the image, for example to remove an overscan region
    image_shape = flat.data.shape
    flat = flat[
        trim_y_start : image_shape[0] - trim_y_end,
        trim_x_start : image_shape[1] - trim_x_end,
    ]

    #   Set gain _> get it from Header if not provided
    if gain is None:
        gain = flat.header["EGAIN"]

    #   Calculated uncertainty
    flat = ccdp.create_deviation(
        flat,
        gain=gain * u.electron / u.adu,
        readnoise=read_noise * u.electron,
        disregard_nan=True,
    )

    # Subtract bias
    if rm_bias:
        flat = ccdp.subtract_bias(flat, combined_bias)

    #   Find the correct dark exposure
    valid_dark_available, closest_dark_exposure_time = (
        utilities.find_nearest_exposure_time_to_reference_image(
            flat,
            list(combined_darks.keys()),
            time_tolerance=exposure_time_tolerance,
        )
    )

    #   Exit if no dark with a similar exposure time have been found
    if not valid_dark_available and not rm_bias:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Closest dark exposure time is "
            f"{closest_dark_exposure_time} for flat of exposure time "
            f"{flat.header['exptime']}. {style.Bcolors.ENDC}"
        )

    #   Subtract the dark current
    flat = ccdp.subtract_dark(
        flat,
        combined_darks[closest_dark_exposure_time],
        exposure_time="exptime",
        exposure_unit=u.second,
        scale=rm_bias,
    )

    #   Save the result
    file_name = flat_file_name.split("/")[-1]
    flat.write(flat_path / file_name, overwrite=True)


def master_flat(
    image_path: str | Path,
    output_dir: str | Path,
    image_type: dict[str, list[str]],
    mk_bad_pixel_mask: bool = True,
    plot_plots: bool = False,
    debug: bool = False,
    n_cores_multiprocessing: int | None = None,
    dtype: str | np.dtype | None = None,
    **kwargs,
) -> None:
    """
    This function calculates master flats from individual flat field
    images located in one directory. The flat field images are group
    according to their exposure time.

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type
        Image types of the images. Possibilities: bias, dark, flat,
        light

    mk_bad_pixel_mask
        If True a bad pixel mask is created.
        Default is ``True``.

    plot_plots
        If True some plots showing some statistic on the flat fields are
        created.
        Default is ``False``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    dtype
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    terminal_output.print_to_terminal("Stack flats...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create new image collection for the reduced flat images
    image_file_collection = utilities.image_file_collection(file_path)

    #   Determine filter
    flat_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class="flat",
    )
    filters = set(
        h["filter"] for h in image_file_collection.headers(imagetyp=flat_image_type)
    )

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(filters),
        add_progress_bar=True,
    )

    #   Reduce science images and save to an extra directory
    for filter_ in filters:
        executor.schedule(
            stack_flat_images,
            args=(
                image_file_collection,
                flat_image_type,
                filter_,
                out_path,
            ),
            kwargs={
                "plot_plots": plot_plots,
                "dtype": dtype,
            },
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Stacking of flat images using multiprocessing"
            f" failed :({style.Bcolors.ENDC}"
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    #   Collect multiprocessing results
    #
    #   Get bad pixel masks
    bad_pixel_mask_list: list[np.ndarray] = executor.res

    if mk_bad_pixel_mask:
        utilities.make_bad_pixel_mask(
            bad_pixel_mask_list,
            out_path,
            verbose=debug,
        )

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)


def stack_flat_images(
    image_file_collection: ccdp.ImageFileCollection,
    flat_image_type: str | list[str] | None,
    filter_: str,
    out_path: Path,
    plot_plots: bool = False,
    dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """
    Stack flats for the individual filters

    Parameters
    ----------
    image_file_collection
        Image file collection for referencing all dark files

    flat_image_type
        Image type designation used for dark files

    filter_
        Current filter

    out_path
        Path to the directory where the master files should be saved to

    plot_plots
        If True some plots showing some statistic on the flat fields are
        created.
        Default is ``False``.

    dtype
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64

    Returns
    -------
    bad_pixel_mask_list
    """
    #   Select flats to combine
    flats_to_combine = image_file_collection.files_filtered(
        imagetyp=flat_image_type,
        filter=filter_,
        include_path=True,
    )

    #   Combine darks: Average images + sigma clipping to remove
    #                  outliers, set memory limit to 15GB, scale the
    #                  frames so that they have the same median value
    #                  ('inv_median')
    combined_flat = ccdp.combine(
        flats_to_combine,
        method="average",
        scale=utilities.inverse_median,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=15e9,
        dtype=dtype,
    )

    #   Add Header keyword to mark the file as a Master
    combined_flat.meta["combined"] = True

    #   Define name and write file to disk
    flat_file_name = "combined_flat_filter_{}.fit".format(filter_.replace("''", "p"))
    combined_flat.write(out_path / flat_file_name, overwrite=True)

    #   Plot flat medians and means
    if plot_plots:
        plots.plot_median_of_flat_fields(
            image_file_collection,
            flat_image_type,
            out_path,
            filter_,
        )

    return ccdp.ccdmask(combined_flat.data)


