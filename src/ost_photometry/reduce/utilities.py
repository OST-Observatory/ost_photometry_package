############################################################################
#                               Libraries                                  #
############################################################################

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars
from scipy.ndimage import median_filter

from .. import calibration_parameters, checks, style, terminal_output
from .. import utilities as base_utilities
from ..fits_headers import ensure_mjd_obs_in_header
from ..fwhm import (
    estimate_fwhm_from_positions,
    filter_table_finite_cutouts,
    source_positions_from_table,
)
from . import plots
from .exposure import (
    check_exposure_times,
    find_nearest_exposure_time,
    find_nearest_exposure_time_to_reference_image,
    get_exposure_times,
)
from .image_collection import image_file_collection
from .image_collection import image_file_collection as make_image_file_collection
from .image_types import get_image_type
from .instrument import get_instrument_info
from .masks import get_pixel_mask, make_bad_pixel_mask, make_hot_pixel_mask
from .wcs_reduce import determine_wcs_all_images

__all__ = [
    "adjust_edian_compatibility",
    "adjust_endian_compatibility",
    "bin_image",
    "check_exposure_times",
    "check_filter_keywords",
    "check_master_files_on_disk",
    "detect_outlier",
    "determine_wcs_all_images",
    "estimate_fwhm",
    "find_nearest_exposure_time",
    "find_nearest_exposure_time_to_reference_image",
    "flip_image",
    "get_exposure_times",
    "get_image_type",
    "get_instrument_info",
    "get_pixel_mask",
    "image_file_collection",
    "inverse_median",
    "make_bad_pixel_mask",
    "make_hot_pixel_mask",
    "make_symbolic_links",
    "prepare_reduction",
    "sanitize_image_types",
    "update_header_information",
]

############################################################################
#                           Routines & definitions                         #
############################################################################


def make_symbolic_links(path_list: list[str], temp_dir: TemporaryDirectory) -> None:
    """
    Make symbolic links

    Parameters
    ----------
    path_list
        List with paths to files

    temp_dir
        Temporary directory to store the symbolic links
    """
    #   Set current working directory
    working_dir = os.getcwd()

    #   Loop over directories
    i: int = 0
    for path in path_list:
        #   Get file list
        files = os.listdir(path)
        files.sort()
        #   Loop over files
        for file_ in files:
            if os.path.isfile(os.path.join(path, file_)):
                #   Add ID to file name
                new_filename = f"{i}_{file_}"

                #   Fill temp directory with file links
                os.symlink(
                    os.path.join(working_dir, path, file_),
                    os.path.join(temp_dir.name, new_filename),
                )

                i += 1


def inverse_median(data: np.ndarray) -> np.floating:
    """
    Inverse median

    Parameters
    ----------
    data
        Data

    Returns
    -------
    float
        Inverse median
    """
    return 1 / np.median(data)


def check_filter_keywords(
    path: str, temp_dir: TemporaryDirectory, image_type: str
) -> Path | str | None:
    """
    Consistency check - Check if the image type of the images in 'path'
                        fit to the one supplied with 'image_type'.
    Parameters
    ----------
    path
        File path to check

    temp_dir
        Temporary directory to store the symbolic links to the images

    image_type
        Internal image type of the images in 'path' should have

    Returns
    -------
    return_path

    """
    #   Sanitize the provided path
    file_path = Path(path)

    #   Check weather path exists
    if not file_path.exists():
        raise RuntimeError(
            f"{style.Bcolors.FAIL}The provided path ({path}) does not "
            f"exists {style.Bcolors.ENDC}"
        )

    #   Create image collection
    image_file_collection = make_image_file_collection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        return file_path

    #   Get image types
    image_type_dict = calibration_parameters.get_image_types()
    image_type = image_type_dict[image_type]

    #   Find all images that have the correct image type
    image_with_correct_image_type = []
    for type_img in image_type:
        image_with_correct_image_type += list(
            image_file_collection.files_filtered(imagetyp=type_img)
        )

    #   Find those images with a wrong image type
    #   -> Compare image file collection with 'image_with_correct_image_type'
    list_1 = list(image_file_collection.files)
    list_2 = image_with_correct_image_type
    result = [x for x in list_1 if x not in list_2]

    if result:
        sanitize_image_types(file_path, temp_dir, image_type)
        return None

    return str(file_path)


def sanitize_image_types(
    file_path: Path, temp_dir: TemporaryDirectory, image_type: str | list[str]
) -> None:
    """
    Sanitize image types according to prerequisites

    Parameters
    ----------
    file_path

    temp_dir
        Temporary directory to store the symbolic links to the images

    image_type
        Expected image type
    """
    #   Sanitize
    image_file_collection = make_image_file_collection(file_path)

    for image_ccd, file_name in image_file_collection.ccds(
        ccd_kwargs={"unit": "adu"}, return_fname=True
    ):
        if isinstance(image_type, list):
            image_ccd.meta["imagetyp"] = image_type[0]
        else:
            image_ccd.meta["imagetyp"] = image_type

        image_ccd.write(temp_dir.name + "/" + file_name)


def prepare_reduction(
    output_dir: str,
    bias_path: str,
    darks_path: str,
    flats_path: str,
    images_path: str,
    raw_files_path: str,
    temp_dir: TemporaryDirectory,
    image_type: dict[str, str] | None = None,
) -> str:
    """
    Prepare directories and files for the reduction procedure

    Parameters
    ----------
    output_dir
        Path to the directory where the master files should be saved to

    bias_path
        Path to the bias or '?'

    darks_path
        Path to the darks or '?'

    flats_path
        Path to the flats or '?'

    images_path
        Path to the science images or '?'

    raw_files_path
        Path to all raw images or '?', if bias, darks, flats, and images
        are provided.

    temp_dir
        Temporary directory to store the symbolic links to the images

    image_type
        Image type to select. Possibilities: bias, dark, flat, light
        Default is ``None``.

    Returns
    -------
    raw_files_path
        Points to the path with the raw files. Either the temporary
        directory or the already provided 'raw_files_path' directory.
    """
    #   Check directories
    terminal_output.print_to_terminal("Check if directories exists...")

    checks.check_output_directories(output_dir)
    if raw_files_path == "?":
        checks.check_path(darks_path)
        checks.check_path(flats_path)
        checks.check_path(images_path)
        if bias_path != "?":
            checks.check_path(bias_path)

        #   Find sub directories
        darks_path_list = checks.list_subdirectories(darks_path)
        flats_path_list = checks.list_subdirectories(flats_path)
        images_path_list = checks.list_subdirectories(images_path)
        if bias_path != "?":
            bias_path_list = checks.list_subdirectories(bias_path)

        #   Check consistency between images and fits header keywords
        terminal_output.print_to_terminal(
            "Check header keywords for consistency...",
        )
        raw_files_path_list = []
        if bias_path != "?":
            for path in bias_path_list:
                if image_type is not None:
                    image_type_keyword = image_type["bias"]
                else:
                    image_type_keyword = "bias"
                new_bias_path = check_filter_keywords(
                    path,
                    temp_dir,
                    image_type_keyword,
                )
                if isinstance(new_bias_path, str):
                    raw_files_path_list.append(new_bias_path)

        for path in darks_path_list:
            if image_type is not None:
                image_type_keyword = image_type["dark"]
            else:
                image_type_keyword = "dark"
            new_darks_path = check_filter_keywords(
                path,
                temp_dir,
                image_type_keyword,
            )
            if isinstance(new_darks_path, str):
                raw_files_path_list.append(new_darks_path)

        for path in flats_path_list:
            if image_type is not None:
                image_type_keyword = image_type["flat"]
            else:
                image_type_keyword = "flat"
            new_flats_path = check_filter_keywords(
                path,
                temp_dir,
                image_type_keyword,
            )
            if isinstance(new_flats_path, str):
                raw_files_path_list.append(new_flats_path)

        for path in images_path_list:
            if image_type is not None:
                image_type_keyword = image_type["light"]
            else:
                image_type_keyword = "light"
            new_images_path = check_filter_keywords(
                path,
                temp_dir,
                image_type_keyword,
            )
            if isinstance(new_images_path, str):
                raw_files_path_list.append(new_images_path)

        #   Link all files to the temporary directory
        make_symbolic_links(raw_files_path_list, temp_dir)

        raw_files_path_new = temp_dir.name
    else:
        #   Check directories
        checks.check_path(raw_files_path)
        raw_files_path_list = checks.list_subdirectories(raw_files_path)

        if len(raw_files_path) >= 1:
            #   Link all files to the temporary directory
            make_symbolic_links(raw_files_path_list, temp_dir)

            raw_files_path_new = temp_dir.name
        else:
            #   This should not happen...
            raise RuntimeError(
                f"{style.Bcolors.FAIL}Raw file path could not be "
                f"decoded...\n {style.Bcolors.ENDC}"
            )

    return raw_files_path_new


def estimate_fwhm(
    image_path: Path,
    output_dir: Path,
    image_type: list[str],
    plot_subplots: bool = False,
    indent: int = 2,
) -> None:
    """
    Estimates the FWHM of the objects

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type
        Header keyword characterizing the image type for which the
        shifts shall be determined

    plot_subplots
        Plot subplots around the stars used to estimate the FWHM
        Default is ``False``.

    indent
        Indentation for the console output lines.
        Default is ``2``.
    """
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   New image collection for the images
    image_file_collection = make_image_file_collection(file_path)

    #   Determine filter
    filter_set = set(
        h["filter"] for h in image_file_collection.headers(imagetyp=image_type)
    )

    #   Combine images for the individual filters
    for filter_ in filter_set:
        #   Select images to combine
        ifc_filtered = image_file_collection.filter(imagetyp=image_type, filter=filter_)

        #   List for the median FWHM for individual images
        img_fwhm = []

        #   Loop over images
        for img_ccd, file_name in ifc_filtered.ccds(return_fname=True):
            #   Get background
            mean, median, std = sigma_clipped_stats(img_ccd.data, sigma=3.0)

            #   Find stars
            dao_finder = DAOStarFinder(fwhm=3.0, threshold=10.0 * std)
            object_tbl = dao_finder(img_ccd.data - median, mask=img_ccd.mask)

            #   Exclude objects close the image edges
            extraction_box = 25
            half_box = (extraction_box - 1) / 2

            x = object_tbl["x_centroid"]
            y = object_tbl["y_centroid"]
            flux = object_tbl["flux"]

            mask = (
                (x > half_box)
                & (x < (img_ccd.data.shape[1] - 1 - half_box))
                & (y > half_box)
                & (y < (img_ccd.data.shape[0] - 1 - half_box))
            )

            objects_tbl_filtered = Table()
            objects_tbl_filtered["x"] = x[mask]
            objects_tbl_filtered["y"] = y[mask]
            objects_tbl_filtered["y"] = y[mask]
            objects_tbl_filtered["flux"] = flux[mask]

            #   Exclude the brightest stars that are often saturated
            #   (rm the brightest 1% of all stars)

            #   Sort list with star positions according to flux
            tbl_sort = objects_tbl_filtered.group_by("flux")

            # Determine the 99 percentile
            percentile_99 = np.percentile(tbl_sort["flux"], 99)

            #   Determine the position of the 99 percentile in the position
            #   list
            id_percentile_99 = np.argmin(np.absolute(tbl_sort["flux"] - percentile_99))

            #   Use 25 stars to estimate the FWHM
            n_fwhm_stars = 25

            #   Check if enough stars were detected
            if id_percentile_99 - n_fwhm_stars < 1:
                n_fwhm_stars = 1

            #   Resize table -> limit it to the suitable stars
            objects_tbl_filtered = tbl_sort[:][
                id_percentile_99 - n_fwhm_stars : id_percentile_99
            ]

            objects_tbl_filtered, n_nonfinite = filter_table_finite_cutouts(
                objects_tbl_filtered,
                img_ccd.data,
                25,
                extra_mask=img_ccd.mask,
            )
            if n_nonfinite:
                terminal_output.print_to_terminal(
                    f"Skipped {n_nonfinite} FWHM-plot star(s) with non-finite "
                    "cutouts.",
                    style_name="WARNING",
                    indent=indent,
                )
            if plot_subplots and len(objects_tbl_filtered) > 0:
                object_cutouts = extract_stars(
                    img_ccd,
                    objects_tbl_filtered,
                    size=25,
                )
                plots.cutouts_fwhm_stars(
                    out_path,
                    len(objects_tbl_filtered),
                    object_cutouts,
                    filter_,
                    base_utilities.get_basename(file_name),
                )

            xy_pos = source_positions_from_table(objects_tbl_filtered)
            mean_fwhm, fwhm_error = estimate_fwhm_from_positions(
                img_ccd.data,
                xy_pos,
                mask=img_ccd.mask,
                error=(
                    img_ccd.uncertainty.array
                    if img_ccd.uncertainty is not None
                    else None
                ),
                default_fwhm=3.0,
            )
            if fwhm_error is None:
                img_fwhm.append(mean_fwhm)

        if img_fwhm:
            terminal_output.print_to_terminal(
                f"FWHM (median) of the stars in Filter {filter_}: {np.median(img_fwhm)}",
                indent=indent,
            )


def check_master_files_on_disk(
    image_path: str | Path,
    image_type_dict: dict[str, list[str]],
    required_dark_exposure_times: list[float],
    science_filters: list[str] | set[str],
    check_bias: bool,
    *,
    exposure_time_tolerance: float = 0.5,
) -> bool:
    """
    Check if master files are already prepared on disk.

    Parameters
    ----------
    image_path
        Path to the reduced output directory with combined masters.

    image_type_dict
        Image types of the images (bias, dark, flat, light).

    required_dark_exposure_times
        Exposure times needed for science and flat frames.

    science_filters
        Filters used by science images that require master flats.

    check_bias
        If True, verify that a combined master bias exists.

    exposure_time_tolerance
        Tolerance when matching science/flat exptimes to master darks.
    """
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)

    #   Get image collection for the reduced files
    image_file_collection = make_image_file_collection(file_path)

    if not image_file_collection.files:
        return False

    ###
    #   Get master dark
    #
    dark_image_type = get_image_type(
        image_file_collection,
        image_type_dict,
        image_class="dark",
    )

    #   Return if no flats found
    if not dark_image_type:
        return False

    #   Prepare dict with master darks
    combined_darks_dict = {
        ccd.header["exptime"]: ccd
        for ccd in image_file_collection.ccds(
            imagetyp=dark_image_type,
            combined=True,
        )
    }

    master_available = True
    #   Check if master darks exist for all required exposure times
    master_dark_exptimes = list(combined_darks_dict.keys())
    for req_time in required_dark_exposure_times:
        valid, _ = find_nearest_exposure_time(
            req_time,
            master_dark_exptimes,
            time_tolerance=exposure_time_tolerance,
        )
        if not valid:
            master_available = False
            break

    ###
    #   Get master flats
    #
    flat_image_type = get_image_type(
        image_file_collection,
        image_type_dict,
        image_class="flat",
    )

    #   Return if no flats found
    if not flat_image_type:
        return False

    #   Prepare dict with master flats
    combined_flats_dict = {
        ccd.header["filter"]: ccd
        for ccd in image_file_collection.ccds(
            imagetyp=flat_image_type,
            combined=True,
        )
    }

    #   Check if master flats exist for all science filters
    science_filter_set = set(science_filters)
    for filt in science_filter_set:
        if filt not in combined_flats_dict:
            master_available = False
            break

    if check_bias:
        ###
        #   Get master bias
        #
        bias_image_type = get_image_type(
            image_file_collection,
            image_type_dict,
            image_class="bias",
        )

        #   Return if no flats found
        if not bias_image_type:
            return False

        #   Prepare list with master biases
        combined_bias = image_file_collection.files_filtered(
            imagetyp=bias_image_type,
            combined=True,
            include_path=True,
        )

        if not combined_bias:
            master_available = False

    return master_available


def flip_image(
    image_file_collection: ccdp.ImageFileCollection, output_path: Path
) -> ccdp.ImageFileCollection:
    """
    Flip images in X and Y direction

    Parameters
    ----------
    image_file_collection
        Image file collection

    output_path
        Path to save the individual images

    Returns
    -------
    flipped_images_ifc
        Image file collection pointing to the flipped images
    """
    terminal_output.print_to_terminal("Flip images", indent=2)

    #   Check directory
    checks.check_output_directories(output_path)
    output_path_flipped = output_path / "flipped"
    checks.check_output_directories(output_path_flipped)

    for image, file_name in image_file_collection.ccds(
        ccd_kwargs={"unit": "adu"},
        return_fname=True,
    ):
        #   Flip image
        image_flipped = ccdp.transform_image(image, np.flip, axis=(0, 1))

        #   Save the result
        image_flipped.write(output_path_flipped / file_name, overwrite=True)

    #   Replace new image file collection
    return make_image_file_collection(output_path_flipped)


def bin_image(
    image_file_collection: ccdp.ImageFileCollection,
    output_path: Path,
    binning_value: int,
) -> ccdp.ImageFileCollection:
    """
    Bin images in X and Y direction

    Parameters
    ----------
    image_file_collection
        Image file collection

    output_path
        Path to save the individual images

    binning_value
        Number of pixel that the image should be binned in X and Y
        direction.

    Returns
    -------
    binned_ifc
        Image file collection pointing to the binned images
    """
    terminal_output.print_to_terminal("Bin images", indent=2)

    #   Check directory
    checks.check_output_directories(output_path)
    output_path_binned = output_path / "binned"
    checks.check_output_directories(output_path_binned)

    for image, file_name in image_file_collection.ccds(
        ccd_kwargs={"unit": "adu"},
        return_fname=True,
    ):
        #   Bin image
        binned_image = ccdp.block_average(image, binning_value)

        #   Correct Header
        binned_image.meta["XBINNING"] = binning_value
        binned_image.meta["YBINNING"] = binning_value
        binned_image.meta["INFO_0"] = "Software binned using numpy mean function"
        binned_image.meta["INFO_1"] = "    Exposure time scaled accordingly"

        #   Save the result
        binned_image.write(output_path_binned / file_name, overwrite=True)

    #   Replace new image file collection
    return make_image_file_collection(output_path_binned)


def update_header_information(
    image: CCDData, n_image_stacked: int = 1, new_target_name: str | None = None
) -> None:
    """
    Updates Header information. Adds among other Header keywords required
    for the GRANDMA project.

    Parameters
    ----------
    image
        The image class with all image specific properties

    n_image_stacked
        Number of stacked images
        Default is ``1``.

    new_target_name
        Name of the target. If not None, this target name will be written
        to the FITS header.
        Default is ``None``.
    """
    #   Add Header keyword to mark the file as stacked
    if n_image_stacked > 1:
        image.meta["COMBINED"] = True
        image.meta["N-IMAGES"] = n_image_stacked
        image.meta["EXPTIME"] = n_image_stacked * image.meta["EXPTIME"]

        #  GRANDMA
        image.meta["STACK"] = 1

    #  GRANDMA
    image.meta["EXPOSURE"] = image.meta["EXPTIME"]

    #   Add MJD of start and center of the observation
    try:
        jd = image.meta["JD"]
        mjd = jd - 2400000.5
        image.meta["MJD_STA"] = mjd

        mjd_mid = mjd + image.meta["EXPTIME"] / 172800
        image.meta["MJD_MID"] = mjd_mid

        image.meta["DATE-MID"] = Time(mjd_mid, format="mjd").fits

    except Exception as e:
        terminal_output.print_to_terminal(
            f"MJD could not be added to the header:\n {e}",
            style_name="WARNING",
        )
    ensure_mjd_obs_in_header(image.meta)

    #   Add observation date using a second keyword (GRANDMA)
    try:
        obs_date = image.meta["DATE-OBS"]
        image.meta["OBSDATE"] = obs_date

    except Exception as e:
        terminal_output.print_to_terminal(
            f"OBSDATE could not be added to the header:\n {e}",
            style_name="WARNING",
        )

    #   Add gain using a second keyword (GRANDMA)
    gain = image.meta["EGAIN"]
    image.meta["GAIN"] = gain

    #   Add target name using a second keyword
    if new_target_name is not None:
        image.meta["OBJECT"] = new_target_name
        #   GRANDMA
        image.meta["TARGET"] = new_target_name
    else:
        #   GRANDMA
        target = image.meta["OBJECT"]
        image.meta["TARGET"] = target

    #   Username and instrument string (GRANDMA)
    image.meta["USERNAME"] = "OST"
    image.meta["INSTRU"] = "CDK"

    #   Add filter system to the Header
    filter_ = image.meta["FILTER"]
    try:
        filter_system = calibration_parameters.filter_systems[filter_]
        image.meta["FILTER-S"] = filter_system
    except Exception as e:
        terminal_output.print_to_terminal(
            f"Filter system could not be determined:\n {e}",
            style_name="WARNING",
        )


def detect_outlier(
    data: np.ndarray, filter_window: int = 8, threshold: float | int = 10.0
) -> np.ndarray:
    """
    Find outliers in a data array

    Parameters
    ----------
    data
        The data

    filter_window
        Width of the median filter window
        Default is ``8``.

    threshold
        Difference above the running median above an element is
        considered to be an outlier.
        Default is ``10.``.

    Returns
    -------

        Index of the elements along axis 0 that are below the threshold
    """
    #   Calculate running median
    run_median = median_filter(data, size=(1, filter_window))

    #   Difference compared to median and sum along axis 0
    score = np.sum(np.abs(data - run_median), axis=0)

    #   Return outliers
    return np.argwhere(score > threshold)


def adjust_endian_compatibility(ccd_data: CCDData) -> CCDData:
    """
    This function adapts the endianness of the supplied image files to those
    of the system.

    Parameters
    ----------
    ccd_data
        Image file
    """
    #   Map with endianness symbols
    endian_map = {
        ">": "big",
        "<": "little",
        "=": sys.byteorder,
        "|": "not applicable",
    }
    if endian_map[ccd_data.data.dtype.byteorder] != sys.byteorder:
        ccd_data.data = ccd_data.data.byteswap()
        ccd_data.data = ccd_data.data.view(ccd_data.data.dtype.newbyteorder())

        u_img = ccd_data.uncertainty.array.byteswap()
        u_img = u_img.view(u_img.dtype.newbyteorder())

        ccd_data.uncertainty = StdDevUncertainty(u_img)

    return ccd_data


adjust_edian_compatibility = adjust_endian_compatibility
