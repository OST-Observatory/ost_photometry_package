"""WCS determination utilities for data reduction."""

from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from .. import checks, style, terminal_output
from .. import utilities as base_utilities
from .image_collection import image_file_collection as make_image_file_collection
from .image_types import get_image_type


def determine_wcs(
    input_dir: str | Path,
    output_dir: str | Path,
    reference_image_index: int = 0,
    force_wcs_determination: bool = False,
    wcs_method: str = "astrometry",
    x_pixel_coordinates: np.ndarray | None = None,
    y_pixel_coordinates: np.ndarray | None = None,
    indent: int = 2,
) -> None:
    """
    Determine the WCS of the reference image and add the WCS to all
    images in the input directory. The latter is to save computing time.
    It is assumed that the images are already aligned and trimmed to
    the same filed of view. However, the observation time of these
    images will be overwritten by this procedure.

    Parameters
    ----------
    input_dir
        Path to the input directory.

    output_dir
        Path to the output directory.

    reference_image_index
        ID of the reference image.
        Default is ``0``.

    force_wcs_determination
        If ``True`` a new WCS determination will be calculated even if
        a WCS is already present in the FITS Header.
        Default is ``False``.

    wcs_method
        Method to use for the WCS determination
        Options: 'astrometry', 'astap', or 'twirl'
        Default is ``astrometry``.

    x_pixel_coordinates
        Pixel coordinates of the objects
        Default is ``None``.

    y_pixel_coordinates
        Pixel coordinates of the objects
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    ###
    #   Prepare variables
    #
    #   Check directories
    file_path = checks.check_pathlib_path(input_dir)
    checks.check_output_directories(output_dir)

    #   Set up image collection for the images
    image_file_collection = make_image_file_collection(file_path)

    #   Filter priority list:
    #   Give the highest priority to the filter with the highest
    #   probability of detecting a large number of stars
    filter_list = ["I", "R", "V", "B", "U"]

    #   Filter image_file_collection according to filter list
    for filter_ in filter_list:
        ifc_filtered = image_file_collection.filter(filter=filter_)

        #   Exit loop when images are found for the current filter
        if ifc_filtered.files:
            reference_filter = filter_
            break

    #   Check again if image_file_collection is empty. If True use first
    #   filter from the image_file_collection filter list.
    if not ifc_filtered.files:
        #   Determine image_file_collection filter
        filters = set(h["filter"] for h in image_file_collection.headers())
        reference_filter = list(filters)[0]

        ifc_filtered = image_file_collection.filter(filter=reference_filter)

    #   Get reference image
    reference_image_path = ifc_filtered.files[reference_image_index]

    reference_image = base_utilities.Image(
        reference_image_index,
        reference_filter,
        reference_image_path,
        output_dir,
    )

    #   Test if the image contains already a WCS
    wcs_available = base_utilities.check_wcs_exists(reference_image)

    #   Determine WCS
    if not wcs_available or force_wcs_determination:
        wcs = determine_wcs_core(
            reference_image,
            wcs_method=wcs_method,
            x_pixel_coordinates=x_pixel_coordinates,
            y_pixel_coordinates=y_pixel_coordinates,
            indent=indent,
        )

        #   Add WCS to images
        if wcs is not None:
            for image, file_name in image_file_collection.ccds(return_fname=True):
                image.wcs = wcs

                #   Save the image
                image.write(output_dir / file_name, overwrite=True)


def determine_wcs_all_images(
    input_dir: str | Path,
    output_dir: Path,
    force_wcs_determination: bool = False,
    wcs_method: str = "astrometry",
    x_pixel_coordinates: np.ndarray | None = None,
    y_pixel_coordinates: np.ndarray | None = None,
    only_combined_images: bool = False,
    image_type: list[str] | None = None,
    indent: int = 2,
) -> None:
    """
    Determine the WCS of each image individually. Images can be filtered
    based on image type and the 'combined' keyword.

    Parameters
    ----------
    input_dir
        Path to the input directory.

    output_dir
        Path to the output directory.

    force_wcs_determination
        If ``True`` a new WCS determination will be calculated even if
        a WCS is already present in the FITS Header.
        Default is ``False``.

    wcs_method
        Method to use for the WCS determination
        Options: 'astrometry', 'astap', or 'twirl'
        Default is ``astrometry``.

    x_pixel_coordinates
        Pixel coordinates of the objects
        Default is ``None``.

    y_pixel_coordinates
        Pixel coordinates of the objects
        Default is ``None``.

    only_combined_images
        Filter for images that have a 'combined' fits header keyword.
        Default is ``False``.

    image_type
        Image type to select. Possibilities: bias, dark, flat, light
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    ###
    #   Prepare variables
    #
    #   Check directories
    file_path = checks.check_pathlib_path(input_dir)
    checks.check_output_directories(output_dir)

    #   Set up image collection for the images
    #   and filter according to requirements
    image_file_collection = make_image_file_collection(file_path)

    if image_type is not None:
        true_img_type = get_image_type(
            image_file_collection,
            image_type,
        )
        image_file_collection = image_file_collection.filter(imagetyp=true_img_type)

    if only_combined_images:
        image_file_collection = image_file_collection.filter(
            combined=only_combined_images
        )

    ###
    #   Derive WCS
    #
    for i, (current_ccd_image, file_name) in enumerate(
        image_file_collection.ccds(return_fname=True)
    ):
        #   Prepare image object
        image_object = base_utilities.Image(
            i,
            "filter",
            file_path / file_name,
            output_dir,
        )

        #   Test if the image contains already a WCS
        wcs_available = base_utilities.check_wcs_exists(image_object)

        if not wcs_available or force_wcs_determination:
            wcs = determine_wcs_core(
                image_object,
                wcs_method=wcs_method,
                x_pixel_coordinates=x_pixel_coordinates,
                y_pixel_coordinates=y_pixel_coordinates,
                indent=indent,
            )

            #   Add WCS to image (not necessary for ASTAP method)
            if wcs_method in ["astrometry", "twirl"]:
                current_ccd_image.wcs = wcs

                #   Save the image
                current_ccd_image.write(output_dir / file_name, overwrite=True)


def determine_wcs_core(
    image: base_utilities.Image,
    wcs_method: str = "astrometry",
    x_pixel_coordinates: np.ndarray | None = None,
    y_pixel_coordinates: np.ndarray | None = None,
    indent: int = 2,
) -> WCS | None:
    """
    Branch between different WCS methods

    Parameters
    ----------
    image
        The image class with all image specific properties

    wcs_method
        Method to use for the WCS determination
        Options: 'astrometry', 'astap', or 'twirl'
        Default is ``astrometry``.

    x_pixel_coordinates
        Pixel coordinates of the objects
        Default is ``None``.

    y_pixel_coordinates
        Pixel coordinates of the objects
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``2``.

    Returns
    -------
    wcs
        The WCS information
    """
    #   astrometry.net:
    if wcs_method == "astrometry":
        try:
            wcs = base_utilities.find_wcs_astrometry(
                image,
                wcs_working_dir="/tmp/",
                indent=indent,
            )
        except RuntimeError:
            terminal_output.print_to_terminal(
                "No WCS solution found :(\n",
                indent=indent,
                style_name="WARNING",
            )
            wcs = None

    #   ASTAP program
    elif wcs_method == "astap":
        try:
            wcs = base_utilities.find_wcs_astap(
                image,
                indent=indent,
            )
            terminal_output.print_to_terminal("")
        except RuntimeError:
            terminal_output.print_to_terminal(
                "No WCS solution found :(\n",
                indent=indent,
                style_name="WARNING",
            )
            wcs = None

    #   twirl library
    elif wcs_method == "twirl":
        try:
            if x_pixel_coordinates is None or y_pixel_coordinates is None:
                raise RuntimeError(
                    f"{style.Bcolors.FAIL} \nException in find_wcs(): \n"
                    f"'x' or 'y' is None -> Exit {style.Bcolors.ENDC}"
                )
            wcs = base_utilities.find_wcs_twirl(
                image,
                x_pixel_coordinates,
                y_pixel_coordinates,
                indent=indent,
            )
        except RuntimeError:
            terminal_output.print_to_terminal(
                "No WCS solution found :(\n",
                indent=indent,
                style_name="WARNING",
            )
            wcs = None

    #   Raise exception
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nException in find_wcs(): '"
            f"\nWCS method not known -> Supplied method was {wcs_method}"
            f"{style.Bcolors.ENDC}"
        )

    return wcs
