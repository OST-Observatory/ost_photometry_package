############################################################################
#                               Libraries                                  #
############################################################################

import os

import sys

from pathlib import Path

from tempfile import TemporaryDirectory

import numpy as np

from scipy.ndimage import median_filter

from astropy.nddata import CCDData, StdDevUncertainty
import astropy.units as u
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars

from scipy.interpolate import UnivariateSpline

import ccdproc as ccdp

from .. import style, checks, calibration_parameters, terminal_output
from .. import utilities as base_utilities

from . import plots
from ..terminal_output import print_to_terminal


############################################################################
#                           Routines & definitions                         #
############################################################################

def make_symbolic_links(
        path_list: list[str], temp_dir: TemporaryDirectory
    ) -> None:
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
                new_filename = f'{i}_{file_}'

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


def get_instruments(
        image_file_collection: ccdp.ImageFileCollection
    ) -> set[str] | None:
    """
    Extract instrument information.

    Parameters
    ----------
    image_file_collection
        Image file collection with all images

    Returns
    -------
    instruments
        List of instruments
    """
    #   Except if no files are found
    if not image_file_collection.files:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.Bcolors.ENDC}'
        )

    #   Get instruments
    if image_file_collection.summary is not None:
        instruments: set[str] = set(image_file_collection.summary['instrume'])
    else:
        terminal_output.print_to_terminal(
            "WARNING: Instruments could not be determined because the image "
            "file collection does not contain a summery -> Returning None",
            style_name='WARNING',
        )
        return None

    return instruments


def get_instrument_info(
        image_file_collection: ccdp.ImageFileCollection,
        temperature_tolerance: float,
        ignore_readout_mode_mismatch: bool = False
        ) -> tuple[str, str, int | None, int, float]:
    """
    Extract information regarding the instruments and readout mode.
    Currently the instrument and readout mode need to be unique. An
    exception will be raised in case multiple readout modes or
    instruments are detected.
    -> TODO: make vector with instruments and readout modes

    Parameters
    ----------
    image_file_collection
        Image file collection with all images

    temperature_tolerance
        The images are required to have the temperature. This value
        specifies the temperature difference that is acceptable.

    ignore_readout_mode_mismatch
        If set to `True` a mismatch of the detected readout modes will
        be ignored.
        Default is ``False``.

    Returns
    -------
    instrument
        List of instruments

    readout_mode
        Mode used to readout the data from the camera chip.

    gain_setting
        Gain used in the camera setting for cameras such as the QHYs.
        This is not the system gain, but it can be calculated from this
        value. See below.

    pixel_bit_value
        Bit value of each pixel

    temperature
        Temperature of the images
    """
    #   Except if no files are found
    if not image_file_collection.files:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.Bcolors.ENDC}'
        )

    #   Get instruments
    instrument_mask = image_file_collection.summary['instrume'].mask
    files_without_instrument = np.array(
        image_file_collection.files
    )[instrument_mask]
    for file_name in files_without_instrument:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without instrument information: \n "
            f"{file_name} \n Skip file.",
            style_name='WARNING',
            indent=2,
        )

    instruments = set(
        image_file_collection.summary['instrume'][np.invert(instrument_mask)]
    )

    if len(instruments) > 1:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Multiple instruments detected.\n'
            f'This is currently not supported -> EXIT \n{style.Bcolors.ENDC}'
        )
    instrument = list(instruments)[0]

    #   Sanitize camera strings from Kstars
    #   TODO: Replace this with an alias list for the cameras
    if 'QHY268M' in instrument:
        instrument = 'QHY268M'
    if 'QHY600M' in instrument:
        instrument = 'QHY600M'

    #   Get the instrument in case of QHY cameras
    if instrument in ['QHYCCD-Cameras-Capture', 'QHYCCD-Cameras2-Capture']:
        #   Get image dimensions and binning
        x_dimensions = set(image_file_collection.summary['naxis1'])
        if len(x_dimensions) > 1:
            raise RuntimeError(
                f'{style.Bcolors.FAIL}Multiple image dimensions detected.\n'
                f'This is not supported -> EXIT \n{style.Bcolors.ENDC}'
            )
        x_dimension = list(x_dimensions)[0]

        y_dimensions = set(image_file_collection.summary['naxis2'])
        if len(y_dimensions) > 1:
            raise RuntimeError(
                f'{style.Bcolors.FAIL}Multiple image dimensions detected.\n'
                f'This is not supported -> EXIT \n{style.Bcolors.ENDC}'
            )
        y_dimension = list(y_dimensions)[0]

        x_bins = set(image_file_collection.summary['xbinning'])
        if len(x_bins) > 1:
            raise RuntimeError(
                f'{style.Bcolors.FAIL}Multiple binning values detected.\n'
                f'This is not supported -> EXIT \n{style.Bcolors.ENDC}'
            )
        x_bin = list(x_bins)[0]

        y_bins = set(image_file_collection.summary['ybinning'])
        if len(y_bins) > 1:
            raise RuntimeError(
                f'{style.Bcolors.FAIL}Multiple binning values detected.\n'
                f'This is not supported -> EXIT \n{style.Bcolors.ENDC}'
            )
        y_bin = list(y_bins)[0]

        #   Physical chip dimensions in pixel
        x_dimension_physical = x_dimension * x_bin
        y_dimension_physical = y_dimension * y_bin

        #   Set instrument
        if x_dimension_physical == 9576 and y_dimension_physical in [6388, 6387]:
            instrument = 'QHY600M'
        elif x_dimension_physical in [6280, 6279] and y_dimension_physical in [4210, 4209]:
            instrument = 'QHY268M'
        elif x_dimension_physical == 3864 and y_dimension_physical in [2180, 2178]:
            instrument = 'QHY485C'
        else:
            instrument = ''

    #   Set default readout mode
    readout_mode = 'default'

    #   Determine readout mode keyword
    if isinstance(image_file_collection.summary, Table):
        if 'readoutm' in image_file_collection.summary.colnames:
            readout_mode_keyword = 'readoutm'
        elif 'readmode' in image_file_collection.summary.colnames:
            readout_mode_keyword = 'readmode'
        else:
            raise KeyError(
                f"{style.Bcolors.FAIL} \nReadout mode keyword for FITS Header could not"
                f" be determined -> ABORT {style.Bcolors.ENDC}"
            )
    else:
        raise ValueError(
            f"{style.Bcolors.FAIL} \nReadout mode keyword for FITS Header "
            "could notbe determined. Summary table of image file collectiont "
            f"is not available. -> ABORT {style.Bcolors.ENDC}"
        )


    #   Readout mode: Restricting files to once with a set read mode
    readout_mode_mask = image_file_collection.summary[readout_mode_keyword].mask
    files_without_readout_mode = np.array(
        image_file_collection.files
    )[readout_mode_mask]
    for file_name in files_without_readout_mode:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without readout mode information: \n "
            f"{file_name} \n Skip file.",
            style_name='WARNING',
            indent=2,
        )

    #   Determine readout modes in the data
    readout_modes = list(set(
        image_file_collection.summary[readout_mode_keyword][np.invert(readout_mode_mask)]
    ))

    if len(readout_modes) > 1:
        if ignore_readout_mode_mismatch:
            readout_mode = readout_modes[0]
            terminal_output.print_to_terminal(
                f"Multiple readout modes detected. Use first one "
                f"detected: {readout_mode}",
                style_name='WARNING',
            )
        else:
            raise RuntimeError(
                f'{style.Bcolors.FAIL}Multiple readout modes detected.\n'
                f'This is currently not supported -> EXIT \n{style.Bcolors.ENDC}'
            )

    #   Readout mode: Fix for QHY models:
    if instrument in ['QHY600M', 'QHY268M']:
        if not readout_modes:
            #   Guess that the readout mode is 'Extend Fullwell 2CMS' if none
            #   was specified in the Header
            readout_mode = 'Extend Fullwell 2CMS'
        elif len(readout_modes) == 1:
            #   Use the first detected readout mode if multiple are specified
            readout_mode = list(readout_modes)[0]

            #   This is a dirty fix for the inadequacy of Maxim-DL to write
            #   the correct readout mode in the Header.
            if readout_mode in ['Fast', 'Slow', 'Normal']:
                readout_mode = 'Extend Fullwell 2CMS'

            #   Kstars treats the readout mode by numbers.
            if readout_mode == 0:
                readout_mode = 'PhotoGraphic DSO'
            elif readout_mode == 1:
                readout_mode = 'High Gain Mode'
            elif readout_mode == 2:
                readout_mode = 'Extend Fullwell'
            elif readout_mode == 3:
                readout_mode = 'Extend Fullwell 2CMS'

        elif ignore_readout_mode_mismatch:
            terminal_output.print_to_terminal(
                "WARNING: Multiple readout modes detected. "
                "Assume Extend Fullwell 2CMS",
                style_name='WARNING',
                indent=2,
            )
            readout_mode = 'Extend Fullwell 2CMS'

    #   Get gain setting
    gain_mask = image_file_collection.summary['gain'].mask
    files_without_gain = np.array(
        image_file_collection.files
    )[gain_mask]
    for file_name in files_without_gain:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without gain information: \n "
            f"{file_name} \n Skip file.",
            style_name='WARNING',
            indent=2,
        )

    gain_settings = set(
        image_file_collection.summary['gain'][np.invert(gain_mask)]
    )

    if len(gain_settings) > 1:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Multiple gain values detected.\n'
            f'This is not supported -> EXIT \n{style.Bcolors.ENDC}'
        )
    gain_setting = list(gain_settings)[0]

    #   Offset settings
    offset_mask = image_file_collection.summary['offset'].mask
    files_without_offset = np.array(
        image_file_collection.files
    )[offset_mask]
    for file_name in files_without_offset:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without offset information: \n "
            f"{file_name} \n Skip file.",
            style_name='WARNING',
            indent=2,
        )

    offset_settings = set(
        image_file_collection.summary['offset'][np.invert(offset_mask)]
    )

    if len(offset_settings) > 1:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Multiple offset values detected.\n'
            f'This is not supported -> EXIT \n{style.Bcolors.ENDC}'
        )

    #   Get the bit setting
    pixel_bit_mask = image_file_collection.summary['bitpix'].mask
    files_without_pixel_bit = np.array(
        image_file_collection.files
    )[pixel_bit_mask]
    for file_name in files_without_pixel_bit:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without pixel bit information: \n "
            f"{file_name} \n Skip file.",
            style_name='WARNING',
            indent=2,
        )

    pixel_bit_set = set(
        image_file_collection.summary['bitpix'][np.invert(pixel_bit_mask)]
    )

    if len(pixel_bit_set) > 1:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Multiple bit values detected.\n'
            f'This is not supported -> EXIT \n{style.Bcolors.ENDC}'
        )
    pixel_bit_value = list(pixel_bit_set)[0]

    #   Get image temperature and avoid images without temperature in HEADER.
    mask = image_file_collection.summary['ccd-temp'].mask
    files_without_ccd_temperature = np.array(image_file_collection.files)[mask]
    for file_name in files_without_ccd_temperature:
        terminal_output.print_to_terminal(
            f"WARNING: Found file without temperature information: "
            f"{file_name} -> Skip file.",
            style_name='WARNING',
            indent=2,
        )

    files_with_ccd_temperature = np.array(image_file_collection.files)[np.invert(mask)]
    temperatures = image_file_collection.summary['ccd-temp'][np.invert(mask)]

    #   Fix for weird crash due to dtype error in 'sigma_clip'
    if temperatures.fill_value == '?':
        temperatures.fill_value = 999.
    if temperatures.dtype == 'object':
        temperatures = temperatures.astype(float)

    median_temperature = np.median(temperatures)
    std_temperature = np.std(temperatures)

    if std_temperature > temperature_tolerance:
        clipped_temperatures_mask = sigma_clip(temperatures).mask
        clipped_temperatures = temperatures[clipped_temperatures_mask]
        clipped_images = files_with_ccd_temperature[clipped_temperatures_mask]

        raise RuntimeError(
            f'{style.Bcolors.FAIL}Significant temperature difference '
            f'detected. The median temperature is {median_temperature}°C.'
            f'The following images have temperatures (°C) of: \n'
            f'{clipped_temperatures.value} \n {clipped_images} \n{style.Bcolors.ENDC}'
        )

    return instrument, readout_mode, gain_setting, pixel_bit_value, median_temperature


#   TODO: Check if the following function can be removed
def get_imaging_software(
        image_file_collection: ccdp.ImageFileCollection) -> set[str]:
    """
    Extract imaging software version.

    Parameters
    ----------
    image_file_collection
        Image file collection with all images

    Returns
    -------
    imaging_software
        List of used imaging software
    """
    #   Except if no files are found
    if not image_file_collection.files:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.Bcolors.ENDC}'
        )

    #   Imaging software (set() allows to return only unique values)
    imaging_software = set(image_file_collection.summary['swcreate'])

    return imaging_software


def get_exposure_times(
        image_file_collection: ccdp.ImageFileCollection,
        image_type: list[str]) -> list[float]:
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
    mask = [True if file in image_type else False
            for file in image_file_collection.summary['imagetyp']]

    #   Except if no files are found in this directory
    if not np.any(mask):
        raise RuntimeError(
            f'{style.Bcolors.FAIL}No images with image type {image_type} '
            f'found -> EXIT\n\t=> Check paths to the images!'
            f'{style.Bcolors.ENDC}'
        )

    #   Exposure exposure_times
    exposure_times = list(set(image_file_collection.summary['exptime'][mask]))

    return exposure_times


def find_nearest_exposure_time(
        reference_exposure_time: float, exposure_times: list[float],
        time_tolerance: float | None = 0.5) -> tuple[bool, np.ndarray]:
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
    id_nearest = np.argmin(
        np.abs(exposure_times_array - reference_exposure_time)
    )
    nearest_exposure_time = exposure_times_array[id_nearest]

    #   Check if closest exposure time is within the tolerance
    time_deltas = reference_exposure_time - nearest_exposure_time
    if time_tolerance is not None and np.abs(time_deltas) > time_tolerance:
        return False, nearest_exposure_time

    return True, nearest_exposure_time


def find_nearest_exposure_time_to_reference_image(
        image: CCDData, exposure_times_other_images: list[float],
        time_tolerance: float | None = 0.5) -> tuple[bool, float]:
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
    exposure_time_reference_image = image.header['exptime']

    return find_nearest_exposure_time(
        exposure_time_reference_image,
        exposure_times_other_images,
        time_tolerance=time_tolerance,
    )


def get_image_type(
        image_file_collection: ccdp.ImageFileCollection,
        image_type_dict: dict[str, list[str]] | list[str],
        image_class: str | None = None) -> str | list[str] | None:
    """
    From an image file collection get the existing image type from a
    list of possible images

    Parameters
    ----------
    image_file_collection
        Image file collection

    image_type_dict
        Image types of the images.
        Possibilities: bias, dark, flat, light

    image_class
        Image file type class to look for.
        Default is ``None``.

    Returns
    -------
    image_types
        Image types or list of image types
    """
    #   Create mask
    if not image_class:
        mask = [True if image_type in image_file_collection.summary['imagetyp']
                else False for image_type in image_type_dict]
    else:
        mask = [True if image_type in image_file_collection.summary['imagetyp']
                else False for image_type in image_type_dict[image_class]]

    #   Get image type ID
    id_image_type = np.argwhere(mask).ravel()
    if not id_image_type.size:
        return None

    #   Get image type
    #   Restricted to only one result -> this is currently necessary
    id_image_type = id_image_type[0]

    #   Return the image type
    if not image_class:
        return image_type_dict[id_image_type]
    else:
        return image_type_dict[image_class][id_image_type]


def check_dark_scaling_possible(
        image_file_collection: ccdp.ImageFileCollection, image_id: int,
        image_type: list[str], exposure_time: float, maximum_dark_time: float,
        bias_available: bool) -> bool:
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
    #   Calculate mask to restrict images to the provided image type
    mask = [True if type_ in image_type else False
            for type_ in image_file_collection.summary['imagetyp']]

    #   Get filename
    filename = image_file_collection.summary['file'][mask][image_id]

    #   Raise exception if no bias frames are available
    if not bias_available:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}No darks with matching exposure time '
            f'found for image: {filename} (exposure time = '
            f'{exposure_time}s). {style.Bcolors.ENDC}'
        )

    #   Check if scaling is possible -> dark frames can only be scaled
    #   to a smaller exposure time and not to a larger one because this
    #   most likely will amplify read noise
    if exposure_time < maximum_dark_time:
        return True
    else:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Scaling the dark frames to the exposure time'
            f' of the image {filename} ({image_type}, exposure time = '
            f'{exposure_time}s) is not possible because the longest dark '
            f'exposure is only {maximum_dark_time}s and dark frames should not'
            f' be scaled "up". {style.Bcolors.ENDC}'
        )


def check_exposure_times(
        image_file_collection: ccdp.ImageFileCollection, image_type: list[str],
        exposure_times: list[float], dark_times: list[float],
        bias_available: bool, exposure_time_tolerance: float = 0.5) -> bool:
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
            f'{style.Bcolors.FAIL}The provided path ({path}) does not '
            f'exists {style.Bcolors.ENDC}'
        )

    #   Create image collection
    image_file_collection = ccdp.ImageFileCollection(file_path)

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
        file_path: Path, temp_dir: TemporaryDirectory,
        image_type: str | list[str]) -> None:
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
    image_file_collection = ccdp.ImageFileCollection(file_path)

    for image_ccd, file_name in image_file_collection.ccds(ccd_kwargs={'unit': 'adu'}, return_fname=True):
        if isinstance(image_type, list):
            image_ccd.meta['imagetyp'] = image_type[0]
        else:
            image_ccd.meta['imagetyp'] = image_type

        image_ccd.write(temp_dir.name + '/' + file_name)


def get_pixel_mask(
        out_path: Path, shape: np.ndarray) -> tuple[bool, CCDData]:
    """
    Calculates or loads a pixel mask highlighting bad and hot pixel.

    Tries to load a precalculated bad pixel mask. If that fails tries to
    load pixel masks calculated by the 'master_dark' and 'master_flat'
    routine and combine those. Assumes default names for the individual
    masks.

    Parameters
    ----------
    out_path
        Path pointing to the main storage location

    shape
        2D array with image dimensions. Is used to check if a
        precalculate mask fits to the image.

    Returns
    -------
    success
        True if either a precalculate bad pixel mask has been found or
        if masks calculated by the 'master_dark' and 'master_flat' have
        been found.

    mask
        Precalculated or combined pixel mask
    """
    #   Load pixel mask
    try:
        mask = CCDData.read(out_path / 'bad_pixel_mask.fit')
        if mask.shape == shape:
            #   If shape is the same, set success to True.
            success = True
        else:
            terminal_output.print_to_terminal(
                "No default bad pixel mask available. Try to use "
                "the mask calculated in the data reduction...",
                indent=1,
                style_name='WARNING',
            )
            #   Raise RuntimeError to trigger except.
            raise RuntimeError('')
    except (FileNotFoundError, RuntimeError):
        #   If no precalculated mask are available, try to load masks
        #   calculated by 'master_dark' and 'master_flat'

        try:
            #   Set default masks
            mask_hot_pixel = np.zeros(shape, dtype=bool)
            mask_bad_pixel = np.zeros(shape, dtype=bool)

            #   New image collection
            image_file_collection = ccdp.ImageFileCollection(out_path)

            #   Get hot pixel masks
            ifc_hot_pixel = image_file_collection.filter(imagetyp='dark mask')

            #   Get correct mask in terms of binning
            for mask_data, file_name in ifc_hot_pixel.data(return_fname=True):
                if mask_data.shape == shape:
                    mask_hot_pixel = mask_data.astype('bool')

            #   Get bad pixel masks
            ifc_bad_pixel = image_file_collection.filter(imagetyp='flat mask')

            #   Get correct mask in terms of binning
            for mask_data, file_name in ifc_bad_pixel.data(return_fname=True):
                if mask_data.shape == shape:
                    mask_bad_pixel = mask_data.astype('bool')

            #   Combine mask
            mask = np.logical_or(mask_hot_pixel, mask_bad_pixel)
            success = True
        except ValueError:
            terminal_output.print_to_terminal(
                "No bad pixel mask available. Skip adding bad pixel mask.",
                indent=1,
                style_name='WARNING',
            )
            mask = np.zeros(shape, dtype=bool)
            success = False

    return success, mask


def make_hot_pixel_mask(
        dark_image: CCDData, gain: float | None, output_dir: str | Path,
        verbose: bool = False) -> None:
    """
    Make a hot pixel mask from a dark frame

    Parameters
    ----------
    dark_image
        Dark image

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.

    output_dir
        Path to the directory where the master files should be saved to

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.
    """
    #   Sanitize the provided paths
    out_path = checks.check_pathlib_path(output_dir)

    #   Get exposure time
    exposure_time = dark_image.header['EXPTIME']

    #   Get image shape
    image_dimension_x = dark_image.meta['naxis1']
    image_dimension_y = dark_image.meta['naxis2']

    #   Scale image with exposure time and gain
    dark_image = dark_image.multiply(gain * u.electron / u.adu)
    dark_image = dark_image.divide(exposure_time * u.second)

    #   Number of pixel
    n_pixel = dark_image.shape[1] * dark_image.shape[0]

    #   Calculate the hot pixel mask. Increase the threshold if the number of
    #   hot pixels is unrealistically high
    threshold_hot_pixel = 2
    hot_pixel_sum = 0
    hot_pixels = np.zeros(dark_image.shape)
    for i in range(0, 100):
        hot_pixels = (dark_image.data > threshold_hot_pixel)
        hot_pixel_sum = hot_pixels.sum()
        #   Check if number of hot pixel is realistic
        if hot_pixel_sum / n_pixel <= 0.03:
            break
        threshold_hot_pixel += 1

    if verbose:
        sys.stdout.write(
            '\r\tNumber of hot pixels: {}\n'.format(hot_pixel_sum)
        )
        sys.stdout.write(
            '\r\tLimit (e-/s/pix) used: {}\n'.format(threshold_hot_pixel)
        )
        sys.stdout.flush()

    #   Save mask with hot pixels
    mask_as_ccd_data_object = CCDData(
        data=hot_pixels.astype('uint8'),
        unit=u.dimensionless_unscaled,
    )
    mask_as_ccd_data_object.header['imagetyp'] = 'dark mask'
    file_name = f'mask_from_dark_{image_dimension_x}x{image_dimension_y}.fit'
    mask_as_ccd_data_object.write(out_path / file_name, overwrite=True)


def make_bad_pixel_mask(
        bad_pixel_mask_list: list[np.ndarray], output_dir: str | Path,
        verbose: bool = False) -> None:
    """
    Calculate a bad pixel mask from a list of bad pixel masks

    Parameters
    ----------
    bad_pixel_mask_list
        List with bad pixel masks

    output_dir
        Path to the directory where the master files should be saved to

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.
    """
    #   Sanitize the provided paths
    out_path = checks.check_pathlib_path(output_dir)

    #   Get information on the image dimensions/binning
    mask_shape_list = []
    for bad_pixel_mask in bad_pixel_mask_list:
        mask_shape_list.append(bad_pixel_mask.shape)
    mask_shape_set = set(mask_shape_list)

    #   Loop over all image shapes (binning options)
    for shape in mask_shape_set:
        #   Calculate overall bad pixel mask
        combined_mask = np.zeros(shape)
        for bad_pixel_mask in bad_pixel_mask_list:
            if bad_pixel_mask.shape == shape:
                combined_mask = np.logical_or(combined_mask, bad_pixel_mask)

        if verbose:
            terminal_output.print_to_terminal(
                f"Number of bad pixels ({shape}): {combined_mask.sum()}",
                indent=1,
            )

        #   Save mask
        mask_as_ccd_data_object = CCDData(
            data=combined_mask.astype('uint8'),
            unit=u.dimensionless_unscaled,
        )
        mask_as_ccd_data_object.header['imagetyp'] = 'flat mask'
        file_name = f'mask_from_ccdmask_{shape[1]}x{shape[0]}.fit'
        mask_as_ccd_data_object.write(out_path / file_name, overwrite=True)


def prepare_reduction(
        output_dir: str, bias_path: str, darks_path: str, flats_path: str,
        images_path: str, raw_files_path: str, temp_dir: TemporaryDirectory,
        image_type: dict[str, str] | None = None) -> str:
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
    if raw_files_path == '?':
        checks.check_path(darks_path)
        checks.check_path(flats_path)
        checks.check_path(images_path)
        if bias_path != '?':
            checks.check_path(bias_path)

        #   Find sub directories
        darks_path_list = checks.list_subdirectories(darks_path)
        flats_path_list = checks.list_subdirectories(flats_path)
        images_path_list = checks.list_subdirectories(images_path)
        if bias_path != '?':
            bias_path_list = checks.list_subdirectories(bias_path)

        #   Check consistency between images and fits header keywords
        terminal_output.print_to_terminal(
            "Check header keywords for consistency...",
        )
        raw_files_path_list = []
        if bias_path != '?':
            for path in bias_path_list:
                if image_type is not None:
                    image_type_keyword = image_type['bias']
                else:
                    image_type_keyword = 'bias'
                new_bias_path = check_filter_keywords(
                    path,
                    temp_dir,
                    image_type_keyword,
                )
                if isinstance(new_bias_path, str):
                    raw_files_path_list.append(new_bias_path)

        for path in darks_path_list:
            if image_type is not None:
                image_type_keyword =  image_type['dark']
            else:
                image_type_keyword = 'dark'
            new_darks_path = check_filter_keywords(
                path,
                temp_dir,
                image_type_keyword,
            )
            if isinstance(new_darks_path, str):
                raw_files_path_list.append(new_darks_path)

        for path in flats_path_list:
            if image_type is not None:
                image_type_keyword = image_type['flat']
            else:
                image_type_keyword = 'flat'
            new_flats_path = check_filter_keywords(
                path,
                temp_dir,
                image_type_keyword,
            )
            if isinstance(new_flats_path, str):
                raw_files_path_list.append(new_flats_path)

        for path in images_path_list:
            if image_type is not None:
                image_type_keyword = image_type['light']
            else:
                image_type_keyword = 'light'
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
                f'{style.Bcolors.FAIL}Raw file path could not be '
                f'decoded...\n {style.Bcolors.ENDC}'
            )

    return raw_files_path_new


def get_star_profiles(
        cutout_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get star profiles

    Parameters
    ----------
    cutout_data
        Image (square) extracted around the star

    Returns
    -------
    profile_x_direction
        Profile in X direction

    profile_y_direction
        Profile in Y direction
    """
    #   Get image shape
    shape = cutout_data.shape

    #   Get central row and column
    if shape[0] % 2 == 0:
        central_column = shape[0] / 2
    else:
        central_column = (shape[0] - 1) / 2 + 1

    if shape[1] % 2 == 0:
        central_row = shape[1] / 2
    else:
        central_row = (shape[1] - 1) / 2 + 1

    #   Get profiles
    profile_x_direction = np.take(cutout_data, central_row, axis=1)
    profile_y_direction = np.take(cutout_data, central_column, axis=0)

    return profile_x_direction, profile_y_direction


def interpolate_fwhm(profile: np.ndarray) -> float:
    """
    Find FWHM by means of interpolation on a stellar profile

    Idea: https://stackoverflow.com/questions/52320873/computing-the-fwhm-of-a-star-profile

    Parameters
    ----------
    profile
        Stellar profile along a specific axis

    Returns
    -------
    full_width_half_maximum
        FWHM of the profile
    """
    #   Prepare interpolation
    half_maximum = 0.5 * np.max(profile)
    x_data = np.linspace(0, len(profile), len(profile))

    #   Do the interpolation
    spline = UnivariateSpline(x_data, profile - half_maximum, s=0)
    r1, r2 = spline.roots()

    return r2 - r1


def estimate_fwhm(
        image_path: Path, output_dir: Path, image_type: list[str],
        plot_subplots: bool = False, indent: int = 2) -> None:
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
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Determine filter
    filter_set = set(h['filter'] for h in image_file_collection.headers(imagetyp=image_type))

    #   Combine images for the individual filters
    for filter_ in filter_set:
        #   Select images to combine
        ifc_filtered = image_file_collection.filter(
            imagetyp=image_type,
            filter=filter_
        )

        #   List for the median FWHM for individual images
        img_fwhm = []

        #   Loop over images
        for img_ccd, file_name in ifc_filtered.ccds(return_fname=True):
            #   Get background
            mean, median, std = sigma_clipped_stats(img_ccd.data, sigma=3.0)

            #   Find stars
            dao_finder = DAOStarFinder(fwhm=3.0, threshold=10. * std)
            object_tbl = dao_finder(img_ccd.data - median, mask=img_ccd.mask)

            #   Exclude objects close the image edges
            extraction_box = 25
            half_box = (extraction_box - 1) / 2

            x = object_tbl['xcentroid']
            y = object_tbl['ycentroid']
            flux = object_tbl['flux']

            mask = (
                (x > half_box) & (x < (img_ccd.data.shape[1] - 1 - half_box)) &
                (y > half_box) & (y < (img_ccd.data.shape[0] - 1 - half_box))
            )

            objects_tbl_filtered = Table()
            objects_tbl_filtered['x'] = x[mask]
            objects_tbl_filtered['y'] = y[mask]
            objects_tbl_filtered['y'] = y[mask]
            objects_tbl_filtered['flux'] = flux[mask]

            #   Exclude the brightest stars that are often saturated
            #   (rm the brightest 1% of all stars)

            #   Sort list with star positions according to flux
            tbl_sort = objects_tbl_filtered.group_by('flux')

            # Determine the 99 percentile
            percentile_99 = np.percentile(tbl_sort['flux'], 99)

            #   Determine the position of the 99 percentile in the position
            #   list
            id_percentile_99 = np.argmin(
                np.absolute(tbl_sort['flux'] - percentile_99)
            )

            #   Use 25 stars to estimate the FWHM
            n_fwhm_stars = 25

            #   Check if enough stars were detected
            if id_percentile_99 - n_fwhm_stars < 1:
                n_fwhm_stars = 1

            #   Resize table -> limit it to the suitable stars
            objects_tbl_filtered = tbl_sort[:][id_percentile_99 - n_fwhm_stars:id_percentile_99]

            #   Extract cutouts
            object_cutouts = extract_stars(img_ccd, objects_tbl_filtered, size=25)

            #   Plot subplots
            if plot_subplots:
                plots.cutouts_fwhm_stars(
                    out_path,
                    len(objects_tbl_filtered),
                    object_cutouts,
                    filter_,
                    base_utilities.get_basename(file_name),
                )

            ###
            #   Loop over all stars and determine the FWHM
            #
            fwhm_x_list = []
            fwhm_y_list = []

            for i in range(len(objects_tbl_filtered)):  # can be optimized -> loop stars
                #   Get star profile
                horizontal, vertical = get_star_profiles(object_cutouts[i])
                #   Try to find FWHM, skip if this is not successful
                try:
                    fwhm_x = interpolate_fwhm(horizontal)
                    fwhm_y = interpolate_fwhm(vertical)

                    fwhm_x_list.append(fwhm_x)
                    fwhm_y_list.append(fwhm_y)
                except ValueError:
                    pass

            #   Get median of the FWHMs
            median_fwhm_x = np.median(fwhm_x_list)
            median_fwhm_y = np.median(fwhm_y_list)

            #   Average the FWHM from both directions
            mean_fwhm = np.mean([median_fwhm_x, median_fwhm_y])

            img_fwhm.append(mean_fwhm)

        terminal_output.print_to_terminal(
            f"FWHM (median) of the stars in Filter {filter_}: "
            f"{np.median(img_fwhm)}",
            indent=indent,
        )


def check_master_files_on_disk(
        image_path: str | Path, image_type_dict: dict[str, list[str]],
        dark_exposure_times: list[float], filter_list: list[str] | set[str],
        check_bias: bool) -> bool:
    """
    Check if master files are already prepared

    Parameters
    ----------
    image_path
        Path to the images

    image_type_dict
        Image types of the images.
        Possibilities: bias, dark, flat, light

    dark_exposure_times
        Exposure times of the raw dark images

    filter_list
        Filter that have been used

    check_bias
        If True bias will be checked as well.

    Returns
    -------
    master_available
        Is True, if all required master files were detected.
    """
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)

    #   Get image collection for the reduced files
    image_file_collection = ccdp.ImageFileCollection(file_path)

    if not image_file_collection.files:
        return False

    ###
    #   Get master dark
    #
    dark_image_type = get_image_type(
        image_file_collection,
        image_type_dict,
        image_class='dark',
    )

    #   Return if no flats found
    if not dark_image_type:
        return False

    #   Prepare dict with master darks
    combined_darks_dict = {
        ccd.header['exptime']: ccd for ccd in image_file_collection.ccds(
            imagetyp=dark_image_type,
            combined=True,
        )
    }

    #   Check if master darks exists for all exposure times
    master_available = True
    for key in combined_darks_dict.keys():
        if key not in dark_exposure_times:
            master_available = False

    ###
    #   Get master flats
    #
    flat_image_type = get_image_type(
        image_file_collection,
        image_type_dict,
        image_class='flat',
    )

    #   Return if no flats found
    if not flat_image_type:
        return False

    #   Prepare dict with master flats
    combined_flats_dict = {
        ccd.header['filter']: ccd for ccd in image_file_collection.ccds(
            imagetyp=flat_image_type,
            combined=True,
        )
    }

    #   Check if master flats exists for all filters
    for key in combined_flats_dict.keys():
        if key not in filter_list:
            master_available = False

    if check_bias:
        ###
        #   Get master bias
        #
        bias_image_type = get_image_type(
            image_file_collection,
            image_type_dict,
            image_class='bias',
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
    output_path_flipped = output_path / 'flipped'
    checks.check_output_directories(output_path_flipped)

    for image, file_name in image_file_collection.ccds(
            ccd_kwargs={'unit': 'adu'},
            return_fname=True,
            ):
        #   Flip image
        image_flipped = ccdp.transform_image(image, np.flip, axis=(0, 1))

        #   Save the result
        image_flipped.write(output_path_flipped / file_name, overwrite=True)

    #   Replace new image file collection
    return ccdp.ImageFileCollection(output_path_flipped)


def bin_image(
        image_file_collection: ccdp.ImageFileCollection, output_path: Path,
        binning_value: int ) -> ccdp.ImageFileCollection:
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
    output_path_binned = output_path / 'binned'
    checks.check_output_directories(output_path_binned)

    for image, file_name in image_file_collection.ccds(
            ccd_kwargs={'unit': 'adu'},
            return_fname=True,
            ):
        #   Bin image
        binned_image = ccdp.block_average(image, binning_value)

        #   Correct Header
        binned_image.meta['XBINNING'] = binning_value
        binned_image.meta['YBINNING'] = binning_value
        binned_image.meta['INFO_0'] = ('Software binned using numpy mean '
                                       'function')
        binned_image.meta['INFO_1'] = '    Exposure time scaled accordingly'

        #   Save the result
        binned_image.write(output_path_binned / file_name, overwrite=True)

    #   Replace new image file collection
    return ccdp.ImageFileCollection(output_path_binned)


def determine_wcs(
        input_dir: str | Path, output_dir: str | Path,
        reference_image_id: int = 0, force_wcs_determination:bool = False,
        wcs_method: str = 'astrometry',
        x_pixel_coordinates: np.ndarray | None = None,
        y_pixel_coordinates: np.ndarray | None = None, indent: int = 2
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

    reference_image_id
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
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Filter priority list:
    #   Give the highest priority to the filter with the highest
    #   probability of detecting a large number of stars
    filter_list = ['I', 'R', 'V', 'B', 'U']

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
        filters = set(h['filter'] for h in image_file_collection.headers())
        reference_filter = list(filters)[0]

        ifc_filtered = image_file_collection.filter(filter=reference_filter)

    #   Get reference image
    reference_image_path = ifc_filtered.files[reference_image_id]

    reference_image = base_utilities.Image(
        reference_image_id,
        reference_filter,
        reference_image_path,
        output_dir,
    )

    # base_utilities.calculate_field_of_view(reference_image)

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
        input_dir: str | Path, output_dir: Path,
        force_wcs_determination: bool = False, wcs_method: str = 'astrometry',
        x_pixel_coordinates: np.ndarray | None = None,
        y_pixel_coordinates: np.ndarray | None =None,
        only_combined_images: bool = False,
        image_type: list[str] | None = None, indent: int = 2) -> None:
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
    image_file_collection = ccdp.ImageFileCollection(file_path)

    if image_type is not None:
        true_img_type = get_image_type(
            image_file_collection,
            image_type,
        )
        image_file_collection = image_file_collection.filter(
            imagetyp=true_img_type
        )

    if only_combined_images:
        image_file_collection = image_file_collection.filter(
            combined=only_combined_images
        )

    ###
    #   Derive WCS
    #
    for i, (current_ccd_image, file_name) in enumerate(image_file_collection.ccds(return_fname=True)):
        #   Prepare image object
        image_object = base_utilities.Image(
            i,
            'filter',
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
            if wcs_method in ['astrometry', 'twirl']:
                current_ccd_image.wcs = wcs

                #   Save the image
                current_ccd_image.write(output_dir / file_name, overwrite=True)


def determine_wcs_core(
        image: base_utilities.Image, wcs_method: str = 'astrometry',
        x_pixel_coordinates: np.ndarray | None = None,
        y_pixel_coordinates: np.ndarray | None = None, indent: int = 2
        ) -> WCS:
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
    if wcs_method == 'astrometry':
        try:
            wcs = base_utilities.find_wcs_astrometry(
                image,
                wcs_working_dir='/tmp/',
                indent=indent,
            )
        except RuntimeError:
            terminal_output.print_to_terminal(
                "No WCS solution found :(\n",
                indent=indent,
                style_name='WARNING',
            )
            wcs = None

    #   ASTAP program
    elif wcs_method == 'astap':
        try:
            wcs = base_utilities.find_wcs_astap(
                image,
                indent=indent,
            )
            terminal_output.print_to_terminal('')
        except RuntimeError:
            terminal_output.print_to_terminal(
                "No WCS solution found :(\n",
                indent=indent,
                style_name='WARNING',
            )
            wcs = None

    #   twirl library
    elif wcs_method == 'twirl':
        try:
            if x_pixel_coordinates is None or y_pixel_coordinates is None:
                raise RuntimeError(
                    f'{style.Bcolors.FAIL} \nException in find_wcs(): \n'
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
                style_name='WARNING',
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


def update_header_information(
        image: CCDData, n_image_stacked: int = 1,
        new_target_name: str | None = None) -> None:
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
        image.meta['COMBINED'] = True
        image.meta['N-IMAGES'] = n_image_stacked
        image.meta['EXPTIME'] = n_image_stacked * image.meta['EXPTIME']

        #  GRANDMA
        image.meta['STACK'] = 1

    #  GRANDMA
    image.meta['EXPOSURE'] = image.meta['EXPTIME']

    #   Add MJD of start and center of the observation
    try:
        jd = image.meta['JD']
        mjd = jd - 2400000.5
        image.meta['MJD_STA'] = mjd

        mjd_mid = mjd + image.meta['EXPTIME'] / 172800
        image.meta['MJD_MID'] = mjd_mid

        image.meta['DATE-MID'] = Time(mjd_mid, format='mjd').fits

    except Exception as e:
        terminal_output.print_to_terminal(
            f"MJD could not be added to the header:\n {e}",
            style_name='WARNING',
        )

    #   Add observation date using a second keyword (GRANDMA)
    try:
        obs_date = image.meta['DATE-OBS']
        image.meta['OBSDATE'] = obs_date

    except Exception as e:
        terminal_output.print_to_terminal(
            f"OBSDATE could not be added to the header:\n {e}",
            style_name='WARNING',
        )

    #   Add gain using a second keyword (GRANDMA)
    gain = image.meta['EGAIN']
    image.meta['GAIN'] = gain

    #   Add target name using a second keyword
    if new_target_name is not None:
        image.meta['OBJECT'] = new_target_name
        #   GRANDMA
        image.meta['TARGET'] = new_target_name
    else:
        #   GRANDMA
        target = image.meta['OBJECT']
        image.meta['TARGET'] = target

    #   Username and instrument string (GRANDMA)
    image.meta['USERNAME'] = 'OST'
    image.meta['INSTRU'] = 'CDK'

    #   Add filter system to the Header
    filter_ = image.meta['FILTER']
    try:
        filter_system = calibration_parameters.filter_systems[filter_]
        image.meta['FILTER-S'] = filter_system
    except Exception as e:
        terminal_output.print_to_terminal(
            f"Filter system could not be determined:\n {e}",
            style_name='WARNING',
        )


def detect_outlier(
        data: np.ndarray, filter_window: int = 8, threshold: float | int = 10.
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


def adjust_edian_compatibility(ccd_data: CCDData) -> CCDData:
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
        '>': 'big',
        '<': 'little',
        '=': sys.byteorder,
        '|': 'not applicable',
    }
    if endian_map[ccd_data.data.dtype.byteorder] != sys.byteorder:
        ccd_data.data = ccd_data.data.byteswap()
        ccd_data.data = ccd_data.data.view(
            ccd_data.data.dtype.newbyteorder()
        )

        u_img = ccd_data.uncertainty.array.byteswap()
        u_img = u_img.view(u_img.dtype.newbyteorder())

        ccd_data.uncertainty = StdDevUncertainty(u_img)

    return ccd_data
