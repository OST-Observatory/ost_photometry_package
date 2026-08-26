"""CCD trim helpers for alignment and fixed-margin cuts."""

from __future__ import annotations

from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData
from scipy.ndimage import shift as shift_scipy

from ... import checks, style, terminal_output
from ..image_collection import image_file_collection as make_image_file_collection
from ..trim_slices import aa_common_trim_margins, ccd_trim_slices


def calculate_min_max_image_shifts(
        shifts: np.ndarray, python_format: bool = False
        ) -> tuple[float, float, float, float]:
    """
    Calculate shifts

    Parameters
    ----------
    shifts
        2D numpy array with the image shifts in X and Y direction

    python_format
        If True the python style of image ordering is used. If False the
        natural/fortran style of image ordering is use.
        Default is ``False``.

    Returns
    -------
    minimum_shift_x
        Minimum shift in X direction

    maximum_shift_x
        Maximum shift in X direction

    minimum_shift_y
        Minimum shift in Y direction

    maximum_shift_y
        Maximum shift in Y direction
    """
    #   Distinguish between python format and natural format
    if python_format:
        id_x = 1
        id_y = 0
    else:
        id_x = 0
        id_y = 1

    #   Maximum and minimum shifts
    minimum_shift_x = np.nanmin(shifts[id_x, :])
    maximum_shift_x = np.nanmax(shifts[id_x, :])

    minimum_shift_y = np.nanmin(shifts[id_y, :])
    maximum_shift_y = np.nanmax(shifts[id_y, :])

    return minimum_shift_x, maximum_shift_x, minimum_shift_y, maximum_shift_y


def calculate_index_from_shifts(
        shifts: np.ndarray, id_current_image: int
        ) -> tuple[float, float, float, float]:
    """
    Calculate image index positions from image shifts

    Parameters
    ----------
    shifts
        The shifts of all images in X and Y direction

    id_current_image
        ID of the current image

    Returns
    -------
    x_start, x_end, y_start, y_end
        Start/End pixel index in X and Y direction.
    """
    #   Calculate maximum and minimum shifts
    min_shift_x, max_shift_x, min_shift_y, max_shift_y = (
        calculate_min_max_image_shifts(shifts, python_format=True)
    )

    #   Calculate indexes from image shifts
    if min_shift_x >= 0 and max_shift_x >= 0:
        x_start = max_shift_x - shifts[1, id_current_image]
        x_end = shifts[1, id_current_image] * -1
    elif min_shift_x < 0 and max_shift_x < 0:
        x_start = shifts[1, id_current_image] * -1
        x_end = max_shift_x - shifts[1, id_current_image]
    else:
        x_start = max_shift_x - shifts[1, id_current_image]
        x_end = min_shift_x - shifts[1, id_current_image]

    if min_shift_y >= 0 and max_shift_y >= 0:
        y_start = max_shift_y - shifts[0, id_current_image]
        y_end = shifts[0, id_current_image] * -1
    elif min_shift_y < 0 and max_shift_y < 0:
        y_start = shifts[0, id_current_image] * -1
        y_end = max_shift_y - shifts[0, id_current_image]
    else:
        y_start = max_shift_y - shifts[0, id_current_image]
        y_end = min_shift_y - shifts[0, id_current_image]

    return (int(np.around(x_start, decimals=0)),
            int(np.around(x_end, decimals=0)),
            int(np.around(y_start, decimals=0)),
            int(np.around(y_end, decimals=0)))


def trim_ccd(
        image: CCDData,
        *,
        x_start: int = 0,
        x_end: int = 0,
        y_start: int = 0,
        y_end: int = 0,
        end_as_positive_margin: bool = True,
        ) -> CCDData:
    """
    Trim a single ``CCDData`` image by pixel margins.

    Shared by :func:`trim_image` (alignment) and :func:`trim_image_simple`
    (fixed margins on a file collection).
    """
    y_slice, x_slice = ccd_trim_slices(
        image.shape,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        end_as_positive_margin=end_as_positive_margin,
    )
    return ccdp.trim_image(image[y_slice, x_slice])


def trim_image(
        image: CCDData, image_id: int, image_shift: np.ndarray,
        correlation_method: str = 'skimage', verbose: bool = False,
        aa_trim_margins: tuple[int, int, int, int] | None = None,
        ) -> CCDData:
    """
    Trim image based on a shift compared to a reference image

    Parameters
    ----------
    image
        The image

    image_id
        Number of the image in the sequence

    image_shift
        Shift of this specific image in X and Y direction

    correlation_method
        Method to use for image alignment.
        Possibilities: 'aa'      = astroalign module only accounting for
                                   xy shifts
                       'aa_true' = astroalign module with corresponding
                                   transformation
                       'own'     = own correlation routine based on
                                   phase correlation, applying fft to
                                   the images
                       'skimage' = phase correlation with skimage
        Default is ``skimage``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    aa_trim_margins
        Precomputed ``(x_start, x_end, y_start, y_end)`` for ``correlation_method='aa'``.
        If omitted, margins are computed from ``image_shift`` (same for every image).

    Returns
    -------
    trimmed_image
        The trimmed image
    """
    if verbose:
        #   Write status to console
        terminal_output.print_to_terminal(
            f"\r\tApply shift to image {image_id}",
        )

    if correlation_method in ['own', 'skimage']:
        #   Calculate indexes from image shifts
        x_start, x_end, y_start, y_end = calculate_index_from_shifts(
            image_shift,
            image_id,
        )
    elif correlation_method == 'aa':
        #   Shift image on sub pixel basis
        image = ccdp.transform_image(
            image,
            shift_scipy,
            shift=image_shift[:, image_id],
            order=1,
        )
        if aa_trim_margins is None:
            aa_trim_margins = aa_common_trim_margins(image_shift)
        x_start, x_end, y_start, y_end = aa_trim_margins

    else:
        raise ValueError(
            f'{style.Bcolors.FAIL}Shift method not known. Expected: '
            f'"pixel" or "sub_pixel", but got '
            f'"{correlation_method}" {style.Bcolors.ENDC}'
        )

    return trim_ccd(
        image,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        end_as_positive_margin=False,
    )


def trim_image_simple(
        image_file_collection: ccdp.ImageFileCollection, output_path: Path,
        redundant_pixel_x_start: int = 100, redundant_pixel_x_end: int = 100,
        redundant_pixel_y_start: int = 100, redundant_pixel_y_end: int = 100
        ) -> ccdp.ImageFileCollection:
    """
    Trim all images in a collection by fixed pixel margins.

    Used e.g. by the N1 BACHES master-image script. Single-image trimming
    goes through :func:`trim_ccd`.

    Parameters
    ----------
    image_file_collection
        Image file collection

    output_path
        Path to save the individual images

    redundant_pixel_x_start
        Number of Pixel to be removed from the start of the image in
        X direction.

    redundant_pixel_x_end
        Number of Pixel to be removed from the end of the image in
        X direction.

    redundant_pixel_y_start
        Number of Pixel to be removed from the start of the image in
        Y direction.

    redundant_pixel_y_end
        Number of Pixel to be removed from the end of the image in
        Y direction.

    Returns
    -------
    trimmed_images_ifc
        Image file collection pointing to the trimmed images
    """
    terminal_output.print_to_terminal("Trim images", indent=2)

    #   Check directory
    checks.check_output_directories(output_path)
    output_path_trimmed = output_path / 'trimmed'
    checks.check_output_directories(output_path_trimmed)

    for image, file_name in image_file_collection.ccds(
            ccd_kwargs={'unit': 'adu'},
            return_fname=True,
    ):
        trimmed_image = trim_ccd(
            image,
            x_start=redundant_pixel_x_start,
            x_end=redundant_pixel_x_end,
            y_start=redundant_pixel_y_start,
            y_end=redundant_pixel_y_end,
            end_as_positive_margin=True,
        )

        #   Save the result
        trimmed_image.write(output_path_trimmed / file_name, overwrite=True)

    #   Return new image file collection
    return make_image_file_collection(output_path_trimmed)
