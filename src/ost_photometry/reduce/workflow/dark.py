"""Reduction workflow: dark module."""

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


def reduce_dark(
    image_path: str | Path,
    output_dir: str | Path,
    image_type: dict[str, list[str]],
    gain: float | None = None,
    read_noise: float = 8.0,
    n_cores_multiprocessing: int | None = None,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
) -> None:
    """
    Reduce dark images: This function reduces the raw dark frames

    Parameters
    ----------
    image_path          : `string` or `pathlib.Path`
        Path to the images

    output_dir          : `string` or `pathlib.Path`
        Path to the directory where the master files should be saved to

    image_type          : `dictionary`
        Image types of the images. Possibilities: bias, dark, flat,
        light

    gain                : `float` or `None`, optional
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise          : `float`, optional
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

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
    terminal_output.print_to_terminal("Reduce darks...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create image collection for the raw data
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Create image collection for the reduced data
    image_file_collection_reduced = ccdp.ImageFileCollection(out_path)

    #   Get master bias
    bias_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class="bias",
    )
    stacked_bias = CCDData.read(
        image_file_collection_reduced.files_filtered(
            imagetyp=bias_image_type,
            combined=True,
            include_path=True,
        )[0]
    )

    #   Set new dark path
    dark_path = Path(out_path / "dark")
    checks.clear_directory(dark_path)

    #   Determine possible image types
    dark_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class="dark",
    )

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(image_file_collection.files_filtered(imagetyp=dark_image_type)),
        add_progress_bar=True,
    )

    #   Loop over darks and reduce darks
    for file_name in image_file_collection.files_filtered(
        include_path=True,
        imagetyp=dark_image_type,
    ):
        executor.schedule(
            reduce_dark_image,
            args=(
                file_name,
                stacked_bias,
                dark_path,
            ),
            kwargs={
                "gain": gain,
                "read_noise": read_noise,
                "trim_x_start": trim_x_start,
                "trim_x_end": trim_x_end,
                "trim_y_start": trim_y_start,
                "trim_y_end": trim_y_end,
            },
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Dark image reduction using multiprocessing"
            f" failed :({style.Bcolors.ENDC}"
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()


def reduce_dark_image(
    dark_file_name: str,
    stacked_bias: CCDData,
    dark_path: Path,
    gain: float | None = None,
    read_noise: float = 8.0,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
) -> None:
    """
    This function reduces the individual raw dark frame images

    Parameters
    ----------
    dark_file_name
        The file name of the dark image that will be reduced

    stacked_bias
        Reduced and stacked Bias CCDData object

    dark_path
        Path where the reduced images should be saved

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

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
    #   Read image file
    dark = CCDData.read(dark_file_name, unit="adu")

    #   Set gain _> get it from Header if not provided
    if gain is None:
        gain = dark.header["EGAIN"]

    #   Trimming the image, for example to remove an overscan region
    image_shape = dark.data.shape
    dark = dark[
        trim_y_start : image_shape[0] - trim_y_end,
        trim_x_start : image_shape[1] - trim_x_end,
    ]

    #   Calculated uncertainty
    dark = ccdp.create_deviation(
        dark,
        gain=gain * u.electron / u.adu,
        readnoise=read_noise * u.electron,
        disregard_nan=True,
    )

    # Subtract bias
    dark = ccdp.subtract_bias(dark, stacked_bias)

    #   Save the result
    file_name = dark_file_name.split("/")[-1]
    dark.write(dark_path / file_name, overwrite=True)


def master_dark(
    image_path: str | Path,
    output_dir: str | Path,
    image_type: dict[str, list[str]],
    gain: float | None = None,
    read_noise: float = 8.0,
    dark_rate: float | None = None,
    mk_hot_pixel_mask: bool = True,
    plot_plots: bool = False,
    debug: bool = False,
    n_cores_multiprocessing: int | None = None,
    rm_bias: bool = False,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
    dtype: str | np.dtype | None = None,
    **kwargs,
) -> None:
    """
    This function calculates master darks from individual dark images
    located in one directory. The dark images are group according to
    their exposure time.

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
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    dark_rate
        Temperature dependent dark rate in e-/pix/s:
        Default is ``None``.

    mk_hot_pixel_mask
        If True a hot pixel mask is created.
        Default is ``True``.

    plot_plots
        If True some plots showing some statistic on the dark frames are
        created.
        Default is ``False``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    rm_bias
        If True the master bias image will be subtracted from the flats
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

    dtype
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    terminal_output.print_to_terminal("Stack darks...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Sanitize dark rate
    if dark_rate is None:
        terminal_output.print_to_terminal(
            "Dark current not specified. Assume 0.1 e-/pix/s.",
            indent=1,
            style_name="WARNING",
        )
        # dark_rate = {0: 0.1}
        dark_rate = 0.1

    #   Create image collection
    try:
        image_file_collection = ccdp.ImageFileCollection(out_path / "dark")
    except FileNotFoundError:
        image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        return

    #   Find darks
    dark_mask = [
        True if file in image_type["dark"] else False
        for file in image_file_collection.summary["imagetyp"]
    ]

    #   Return if no darks are found in this directory
    if not dark_mask:
        return

    #   Get all available shapes with exposure times
    all_available_image_shapes_and_exposure_times: set[tuple[int, int, float]] = set(
        tuple(
            zip(
                image_file_collection.summary["naxis1"][dark_mask],
                image_file_collection.summary["naxis2"][dark_mask],
                image_file_collection.summary["exptime"][dark_mask],
                strict=True,
            )
        )
    )

    #   Get only the shapes
    all_available_image_shapes: set[tuple[int, int]] = set(
        tuple(
            zip(
                image_file_collection.summary["naxis1"][dark_mask],
                image_file_collection.summary["naxis2"][dark_mask],
                strict=True,
            )
        )
    )

    #   Get the maximum exposure time for each shape
    max_exposure_time_per_shape: list = []
    for shape in all_available_image_shapes:
        exposure_times: list = []
        for shape_expo_time in all_available_image_shapes_and_exposure_times:
            if shape[0] == shape_expo_time[0] and shape[1] == shape_expo_time[1]:
                exposure_times.append(shape_expo_time[2])
        max_exposure_time_per_shape.append((*shape, np.max(exposure_times)))

    #   Get exposure times (set allows to return only unique values)
    dark_exposure_times = set(image_file_collection.summary["exptime"][dark_mask])

    #   Get dark image type
    dark_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class="dark",
    )

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(sorted(dark_exposure_times)),
        add_progress_bar=True,
    )
    # executor = Executor(n_cores_multiprocessing)

    #   Reduce science images and save to an extra directory
    for exposure_time in sorted(dark_exposure_times):
        executor.schedule(
            master_dark_stacking,
            args=(
                image_file_collection,
                exposure_time,
                dark_image_type,
                max_exposure_time_per_shape,
                out_path,
                dark_rate,
            ),
            kwargs={
                "gain": gain,
                "read_noise": read_noise,
                "mk_hot_pixel_mask": mk_hot_pixel_mask,
                "plot_plots": plot_plots,
                "rm_bias": rm_bias,
                "trim_x_start": trim_x_start,
                "trim_x_end": trim_x_end,
                "trim_y_start": trim_y_start,
                "trim_y_end": trim_y_end,
                "dtype": dtype,
            },
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f"\n{style.Bcolors.FAIL}Dark image stacking using multiprocessing"
            f" failed :({style.Bcolors.ENDC}"
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(out_path / "dark", ignore_errors=True)


def master_dark_stacking(
    image_file_collection: ccdp.ImageFileCollection,
    exposure_time: float,
    dark_image_type: str | list[str] | None,
    max_exposure_time_per_shape: list[tuple[int, int, float]],
    out_path: Path,
    dark_rate: float,
    gain: int | None = None,
    read_noise: float = 8.0,
    mk_hot_pixel_mask: bool = True,
    plot_plots: bool = False,
    debug: bool = False,
    rm_bias: bool = False,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
    dtype: str | np.dtype | None = None,
) -> None:
    """
    This function stacks all dark images with the same exposure time.

    Parameters
    ----------
    image_file_collection
        Image file collection for referencing all dark files

    exposure_time
        Exposure time of the current set of dark images

    dark_image_type
        Image type designation used for dark files

    out_path
        Path to the directory where the master files should be saved to

    max_exposure_time_per_shape
        Maximum exposure time for each available image shape

    dark_rate
        Temperature dependent dark rate in e-/pix/s:

    gain
        The gain (e-/adu) of the camera. If set to `None` the gain will
        be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera.
        Default is 8 e-.

    mk_hot_pixel_mask
        If True a hot pixel mask is created.
        Default is ``True``.

    plot_plots
        If True some plots showing some statistic on the dark frames are
        created.
        Default is ``False``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    rm_bias
        If True the master bias image will be subtracted from the flats
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

    dtype
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    #   Get only the darks with the correct exposure time
    calibrated_darks = image_file_collection.files_filtered(
        imagetyp=dark_image_type,
        exptime=exposure_time,
        include_path=True,
    )

    #   Combine darks: Average images + sigma clipping to remove
    #                  outliers, set memory limit to 15GB, set unit to
    #                  'adu' since this is not set in our images
    #                  -> find better solution
    combined_dark = ccdp.combine(
        calibrated_darks,
        method="average",
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=15e9,
        unit="adu",
        dtype=dtype,
    )

    #   Trimming the image, for example to remove an overscan region
    if not rm_bias:
        image_shape = combined_dark.data.shape
        combined_dark = combined_dark[
            trim_y_start : image_shape[0] - trim_y_end,
            trim_x_start : image_shape[1] - trim_x_end,
        ]

    #   Add Header keyword to mark the file as a Master
    combined_dark.meta["combined"] = True

    #   Write file to disk
    dark_file_name = f"combined_dark_{exposure_time:4.2f}.fit"
    combined_dark.write(out_path / dark_file_name, overwrite=True)

    #   Set gain _> get it from Header if not provided
    if gain is None:
        gain = int(combined_dark.header["EGAIN"])

    #   Plot histogram
    if plot_plots:
        plots.plot_histogram(
            combined_dark.data,
            out_path,
            gain,
            exposure_time,
        )
        plots.plot_dark_with_distributions(
            combined_dark.data,
            read_noise,
            dark_rate,
            out_path,
            exposure_time=exposure_time,
            gain=gain,
        )

    #   Create mask with hot pixels
    current_shape_x = combined_dark.meta["naxis1"]
    current_shape_y = combined_dark.meta["naxis2"]
    if (
        current_shape_x,
        current_shape_y,
        exposure_time,
    ) in max_exposure_time_per_shape and mk_hot_pixel_mask:
        utilities.make_hot_pixel_mask(
            combined_dark,
            gain,
            out_path,
            verbose=debug,
        )


