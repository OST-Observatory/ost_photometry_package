"""Exposure time utilities for data reduction."""

import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData

from .. import style


def get_exposure_times(
    image_file_collection: ccdp.ImageFileCollection, image_type: list[str]
) -> list[float]:
    """
    Extract the exposure time of a specific image type from an image
    collections.

    Parameters
    ----------
    image_file_collection
        Image file collection with all images

    image_type
        Image type to select. Possibilities: bias, dark, flat, light

    Returns
    -------
    exposure_times
        List of exposure times
    """
    #   Calculate mask to restrict images to the provided image type
    mask = [
        True if file in image_type else False
        for file in image_file_collection.summary["imagetyp"]
    ]

    #   Except if no files are found in this directory
    if not np.any(mask):
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No images with image type {image_type} "
            f"found -> EXIT\n\t=> Check paths to the images!"
            f"{style.Bcolors.ENDC}"
        )

    #   Exposure exposure_times
    exposure_times = list(set(image_file_collection.summary["exptime"][mask]))

    return exposure_times


def find_nearest_exposure_time(
    reference_exposure_time: float,
    exposure_times: list[float],
    time_tolerance: float | None = 0.5,
) -> tuple[bool, np.ndarray]:
    """
    Find the nearest match between a test exposure time and a list of
    exposure times, raising an error if the difference in exposure time
    is more than the tolerance.

    Parameters
    ----------
    reference_exposure_time
        Exposure time for which a match from a list of exposure times
        should be found.

    exposure_times
        Exposure times for which there are images

    time_tolerance
        Maximum difference, in seconds, between the image and the
        closest entry from the exposure time list. Set to ``None`` to
        skip the tolerance test.
        Default is ``0.5``.

    Returns
    -------
    _
        `True` if an exposure was detected within the tolerance time

    nearest_exposure_time
        Nearest exposure time
    """
    #   Find closest exposure time
    exposure_times_array = np.array(list(exposure_times))
    id_nearest = np.argmin(np.abs(exposure_times_array - reference_exposure_time))
    nearest_exposure_time = exposure_times_array[id_nearest]

    #   Check if closest exposure time is within the tolerance
    time_deltas = reference_exposure_time - nearest_exposure_time
    if time_tolerance is not None and np.abs(time_deltas) > time_tolerance:
        return False, nearest_exposure_time

    return True, nearest_exposure_time


def find_nearest_exposure_time_to_reference_image(
    image: CCDData,
    exposure_times_other_images: list[float],
    time_tolerance: float | None = 0.5,
) -> tuple[bool, float]:
    """
    Find the nearest exposure time of a list of exposure times to that
    of an image, raising an error if the difference in exposure time is
    more than the tolerance.

    Parameters
    ----------
    image
        The image for which a matching exposure time is needed

    exposure_times_other_images
        Exposure times for which there are images

    time_tolerance
        Maximum difference, in seconds, between the image and the
        closest entry from the exposure time list. Set to ``None`` to
        skip the tolerance test.
        Default is ``0.5``.

    Returns
    -------
    _
        `True` if an exposure was detected within the tolerance time

    _
        Nearest exposure time
    """
    #   Get exposure time from the image
    exposure_time_reference_image = image.header["exptime"]

    return find_nearest_exposure_time(
        exposure_time_reference_image,
        exposure_times_other_images,
        time_tolerance=time_tolerance,
    )


def check_dark_scaling_possible(
    image_file_collection: ccdp.ImageFileCollection,
    image_id: int,
    image_type: list[str],
    exposure_time: float,
    maximum_dark_time: float,
    bias_available: bool,
) -> bool:
    """
    Check if scaling of dark frames to the given exposure time 'time' is
    possible and handles exceptions

    Parameters
    ----------
    image_file_collection
        File collection with all images

    image_id
        ID of the image

    image_type
        String that characterizes the image type, such as 'science' or
        'flat'. This is used in the exception messages.

    exposure_time
        Exposure time that should be checked

    maximum_dark_time
        Longest dark time that is available

    bias_available
        True if bias frames are available

    Returns
    -------
    bool
        True if dark scaling is possible
    """
    filename = image_file_collection.summary["file"][image_id]

    #   Raise exception if no bias frames are available
    if not bias_available:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No darks with matching exposure time "
            f"found for image: {filename} (exposure time = "
            f"{exposure_time}s). {style.Bcolors.ENDC}"
        )

    #   Check if scaling is possible -> dark frames can only be scaled
    #   to a smaller exposure time and not to a larger one because this
    #   most likely will amplify read noise
    if exposure_time < maximum_dark_time:
        return True
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Scaling the dark frames to the exposure time"
            f" of the image {filename} ({image_type}, exposure time = "
            f"{exposure_time}s) is not possible because the longest dark "
            f"exposure is only {maximum_dark_time}s and dark frames should not"
            f' be scaled "up". {style.Bcolors.ENDC}'
        )


def check_exposure_times(
    image_file_collection: ccdp.ImageFileCollection,
    image_type: list[str],
    exposure_times: list[float],
    dark_times: list[float],
    bias_available: bool,
    exposure_time_tolerance: float = 0.5,
) -> bool:
    """
    Check if relevant dark exposures are available for the exposure
    times in the supplied list

    Parameters
    ----------
    image_file_collection
        File collection with all images

    image_type
        String that characterizes the image type, such as 'science' or
        'flat'. This is used in the exception messages.

    exposure_times
        Exposure times that should be checked

    dark_times
        Dark exposure times that are available

    bias_available
        True if bias frames are available

    exposure_time_tolerance
        Tolerance between science and dark exposure times in s.
        Default is ``0.5``s.

    Returns
    -------
    scale_necessary
        True if dark scaling is possible
    """
    #   Loop over exposure times
    for image_id, time in enumerate(exposure_times):
        #   Find nearest dark frame
        valid, closest_dark = find_nearest_exposure_time(
            time,
            dark_times,
            time_tolerance=exposure_time_tolerance,
        )
        #   In case there is no valid dark, check if scaling is possible
        if not valid:
            scale_necessary = check_dark_scaling_possible(
                image_file_collection,
                image_id,
                image_type,
                time,
                np.max(dark_times),
                bias_available,
            )
            return scale_necessary
        return False
