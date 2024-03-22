############################################################################
#                               Libraries                                  #
############################################################################

import os

import sys

from pathlib import Path

import numpy as np

import math

from scipy.ndimage import shift as shift_scipy

from astropy.nddata import CCDData, StdDevUncertainty
import astropy.units as u
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time

from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars

from scipy.interpolate import UnivariateSpline

import ccdproc as ccdp

from skimage.registration import (
    phase_cross_correlation,
    optical_flow_tvl1,
    # optical_flow_ilk,
)
from skimage.transform import warp

import astroalign as aa

from .. import style, checks, calibration_data, terminal_output
from .. import utilities as base_aux

from . import plot


############################################################################
#                           Routines & definitions                         #
############################################################################

def make_symbolic_links(path_list, temp_dir):
    """
        Make symbolic links

        Parameters
        ----------
        path_list           : `list` of `string`s
            List with paths to files

        temp_dir            : `tempfile.TemporaryDirectory`
            Temporary directory to store the symbolic links
    """
    #   Set current working directory
    working_dir = os.getcwd()

    #   Loop over directories
    for path in path_list:
        #   Get file list
        files = os.listdir(path)
        #   Loop over files
        for file_ in files:
            if os.path.isfile(os.path.join(path, file_)):
                #   Check if a file of the same name already exist in the
                #   temp directory
                if os.path.isfile(os.path.join(temp_dir.name, file_)):
                    random_string = base_aux.random_string_generator(7)
                    new_filename = f'{random_string}_{file_}'
                else:
                    new_filename = file_

                #   Fill temp directory with file links
                os.symlink(
                    os.path.join(working_dir, path, file_),
                    os.path.join(temp_dir.name, new_filename),
                )


def inverse_median(data):
    """
        Inverse median

        Parameters
        ----------
        data            : `numpy.ndarray`
            Data

        Returns
        -------
                        : `float`
            Inverse median
    """
    return 1 / np.median(data)


def get_instruments(image_file_collection):
    """
        Extract instrument information.

        Parameters
        ----------
        image_file_collection : `ccdproc.ImageFileCollection`
            Image file collection with all images

        Returns
        -------
        instruments           : `set`
            List of instruments
    """
    #   Except if no files are found
    if not image_file_collection.files:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.Bcolors.ENDC}'
        )

    #   Get instruments
    instruments = set(image_file_collection.summary['instrume'])

    return instruments


def get_instrument_info(image_file_collection, temperature_tolerance):
    """
        Extract information regarding the instruments and readout mode.
        Currently the instrument and readout mode need to be unique. An
        exception will be raised in case multiple readout modes or
        instruments are detected.
        -> TODO: make vector with instruments and readout modes

        Parameters
        ----------
        image_file_collection      : `ccdproc.ImageFileCollection`
            Image file collection with all images

        temperature_tolerance      : `float`, optional
            The images are required to have the temperature. This value
            specifies the temperature difference that is acceptable.

        Returns
        -------
        instrument                  : `string`
            List of instruments

        readout_mode                : `string`
            Mode used to readout the data from the camera chip.

        gain_setting                : `integer` or `None`
            Gain used in the camera setting for cameras such as the QHYs.
            This is not the system gain, but it can be calculated from this
            value. See below.

        pixel_bit_value             : `integer`
            Bit value of each pixel

        temperature                 : `float`
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

    #   Get readout mode
    readout_mode_mask = image_file_collection.summary['readoutm'].mask
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

    readout_modes = set(
        image_file_collection.summary['readoutm'][np.invert(readout_mode_mask)]
    )

    if not readout_modes:
        readout_mode = 'Extend Fullwell 2CMS'
    elif len(readout_modes) == 1:
        readout_mode = list(readout_modes)[0]

        #   This is a dirty fix for the inadequacy of Maxim-DL to write
        #   the correct readout mode in the Header.
        if readout_mode in ['Fast', 'Slow']:
            readout_mode = 'Extend Fullwell 2CMS'
    else:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Multiple readout modes detected.\n'
            f'This is currently not supported -> EXIT \n{style.Bcolors.ENDC}'
        )

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
    clipped_temperatures_mask = sigma_clip(temperatures).mask

    if np.any(clipped_temperatures_mask):
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
def get_imaging_software(image_file_collection):
    """
        Extract imaging software version.

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            Image file collection with all images

        Returns
        -------
        imaging_software        : `set`
            List of instruments
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


def get_exposure_times(image_file_collection, image_type):
    """
        Extract the exposure time of a specific image type from an image
        collections.

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            Image file collection with all images

        image_type              : `list` of `string`
            Image type to select. Possibilities: bias, dark, flat, light

        Returns
        -------
        exposure_times          : `list`
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
    exposure_times = set(image_file_collection.summary['exptime'][mask])

    return exposure_times


def find_nearest_exposure_time(reference_exposure_time, exposure_times,
                               time_tolerance=0.5):
    """
        Find the nearest match between a test exposure time and a list of
        exposure times, raising an error if the difference in exposure time
        is more than the tolerance.

        Parameters
        ----------
        reference_exposure_time : `float`
            Exposure time for which a match from a list of exposure times
            should be found.

        exposure_times          : `list of floats`
            Exposure times for which there are images

        time_tolerance          : `float` or `None`, optional
            Maximum difference, in seconds, between the image and the
            closest entry from the exposure time list. Set to ``None`` to
            skip the tolerance test.
            Default is ``0.5``.

        Returns
        -------
        _                       : `boolean`
            `True` if an exposure was detected within the tolerance time

        nearest_exposure_time   : `float`
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


def find_nearest_exposure_time_to_reference_image(image,
                                                  exposure_times_other_images,
                                                  time_tolerance=0.5):
    """
        Find the nearest exposure time of a list of exposure times to that
        of an image, raising an error if the difference in exposure time is
        more than the tolerance.

        Parameters
        ----------
        image                       : `astropy.nddata.CCDData`
            Image for which a matching exposure time is needed

        exposure_times_other_images : `list`
            Exposure times for which there are images

        time_tolerance              : `float` or `None`, optional
            Maximum difference, in seconds, between the image and the
            closest entry from the exposure time list. Set to ``None`` to
            skip the tolerance test.
            Default is ``0.5``.

        Returns
        -------
        _                           : `boolean`
            `True` if an exposure was detected within the tolerance time

        _                           : `float`
            Nearest exposure time
    """
    #   Get exposure time from the image
    exposure_time_reference_image = image.header['exptime']

    return find_nearest_exposure_time(
        exposure_time_reference_image,
        exposure_times_other_images,
        time_tolerance=time_tolerance,
    )


def get_image_type(image_file_collection, image_type_dict, image_class=None):
    """
        From an image file collection get the existing image type from a
        list of possible images

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            Image file collection

        image_type_dict         : `dictionary` or `list`
            Image types of the images.
            Possibilities: bias, dark, flat, light

        image_class             : `string`, optional
            Image file type class to look for.
            Default is ``None``.

        Returns
        -------
                                : `string` or `list`
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


def check_dark_scaling_possible(image_file_collection, image_id, image_type,
                                exposure_time, maximum_dark_time,
                                bias_available):
    """
        Check if scaling of dark frames to the given exposure time 'time' is
        possible and handles exceptions

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            File collection with all images

        image_id                : `integer`
            ID of the image

        image_type              : `list` of `string`
            String that characterizes the image type, such as 'science' or
            'flat'. This is used in the exception messages.

        exposure_time           : `float`
            Exposure time that should be checked

        maximum_dark_time       : `float`
            Longest dark time that is available

        bias_available          : `boolean`
            True if bias frames are available

        Returns
        -------
                                : `boolean`
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


def check_exposure_times(image_file_collection, image_type, exposure_times,
                         dark_times, bias_available,
                         exposure_time_tolerance=0.5):
    """
        Check if relevant dark exposures are available for the exposure
        times in the supplied list

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            File collection with all images

        image_type              : `list` of `string`
            String that characterizes the image type, such as 'science' or
            'flat'. This is used in the exception messages.

        exposure_times          : `list` of `float`
            Exposure times that should be checked

        dark_times              : `list` of `float`
            Dark exposure times that are available

        bias_available          : `boolean`
            True if bias frames are available

        exposure_time_tolerance          : 'float', optional
            Tolerance between science and dark exposure times in s.
            Default is ``0.5``s.

        Returns
        -------
        scale_necessary         : `boolean`
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


def check_filter_keywords(path, image_type):
    """
        Consistency check - Check if the image type of the images in 'path'
                            fit to one supplied with 'image_type'.
        Parameters
        ----------
        path            : `string`
            File path to check

        image_type      : `string`
            Internal image type of the images in 'path' should have
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

    # #   Get and check imaging software
    # imaging_soft = get_imaging_soft(image_file_collection)
    # if len(imaging_soft) > 1:
    # terminal_output.print_terminal(
    # imaging_soft,
    # string="Images are taken with multiple software versions: {}. "\
    # "The pipeline cannot account for that, but will try anyway...",
    # indent=2,
    # style_name='WARNING',
    # )

    #   Get image types
    image_type_dict = calibration_data.get_image_types()
    image_type = image_type_dict[image_type]

    #   Find all images that have the correct image type
    image_with_correct_image_type = []
    for type_img in image_type:
        image_with_correct_image_type.append(
            list(image_file_collection.files_filtered(imagetyp=type_img))
        )

    #   Find those images with a wrong image type
    #   -> Compare image file collection with 'image_with_correct_image_type'
    list_1 = list(image_file_collection.files)
    list_2 = image_with_correct_image_type
    # TODO: Check if there is a bug here when image_with_correct_image_type is a list of lists
    result = [x for x in list_1 + list_2 if x not in list_1 or x not in list_2]

    #   TODO: Replace '_str' methods
    if result:
        return sanitize_image_types(file_path, image_type)._str
        # raise RuntimeError(
        # 'The following images do not have the correct '
        # 'image type: \n {} \n Expected: {} \n Got: \n {} \n Path: {} '
        # ''.format(result, image_type, image_file_collection.summary['file', 'imagetyp'], path)
        # )

    return file_path._str


def sanitize_image_types(file_path, image_type):
    """
        Sanitize image types according to prerequisites

        Parameters
        ----------
        file_path           : `pathlib.Path`

        image_type          : `string` or `list`
            Expected image type
    """
    #   Set path
    path = Path('/tmp')
    path = path / base_aux.random_string_generator(7)
    checks.check_output_directories(path)

    #   Sanitize
    image_file_collection = ccdp.ImageFileCollection(file_path)

    for image_ccd, file_name in image_file_collection.ccds(ccd_kwargs={'unit': 'adu'}, return_fname=True):
        if isinstance(image_type, list):
            image_ccd.meta['imagetyp'] = image_type[0]
        else:
            image_ccd.meta['imagetyp'] = image_type

        image_ccd.write(path / file_name)

    return path


def get_pixel_mask(out_path, shape):
    """
        Calculates or loads a pixel mask highlighting bad and hot pixel.

        Tries to load a precalculated bad pixel mask. If that fails tries to
        load pixel masks calculated by the 'master_dark' and 'master_flat'
        routine and combine those. Assumes default names for the individual
        masks.

        Parameters
        ----------
        out_path        : `pathlib.Path`
            Path pointing to the main storage location

        shape           : `numpy.ndarray`
            2D array with image dimensions. Is used to check if a
            precalculate mask fits to the image.

        Returns
        -------
        success         : `boolean`
            True if either a precalculate bad pixel mask has been found or
            if masks calculated by the 'master_dark' and 'master_flat' have
            been found.

        mask            : `astropy.nddata.CCDData`
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
    except:
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
            mask = mask_hot_pixel | mask_bad_pixel
            success = True
        except:
            terminal_output.print_to_terminal(
                "No bad pixel mask available. Skip adding bad pixel mask.",
                indent=1,
                style_name='WARNING',
            )
            mask = np.zeros(shape, dtype=bool)
            success = False

    return success, mask


def make_hot_pixel_mask(dark_image, gain, output_dir, verbose=False):
    """
        Make a hot pixel mask from a dark frame

        Parameters
        ----------
        dark_image          : `astropy.nddata.CCDData`
            Dark image

        output_dir          : `string`
            Path to the directory where the master files should be saved to

        gain                : `float` or `None`, optional
            The gain (e-/adu) of the camera chip. If set to `None` the gain
            will be extracted from the FITS header.
            Default is ``None``.

        verbose             : `boolean`, optional
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
    for i in range(0, 100):
        hot_pixels = (dark_image.data > threshold_hot_pixel)
        #   Check if number of hot pixel is realistic
        if hot_pixels.sum() / n_pixel <= 0.03:
            break
        threshold_hot_pixel += 1

    if verbose:
        sys.stdout.write(
            '\r\tNumber of hot pixels: {}\n'.format(hot_pixels.sum())
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


def make_bad_pixel_mask(bad_pixel_mask_list, output_dir, verbose=False):
    """
        Calculate a bad pixel mask from a list of bad pixel masks

        Parameters
        ----------
        bad_pixel_mask_list : `list` of `astropy.nddata.CCDData` objects
            List with bad pixel masks

        output_dir          : `string`
            Path to the directory where the master files should be saved to

        verbose             : `boolean`, optional
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

    #   Loop over all image dimensions/binning
    for shape in mask_shape_set:
        #   Calculate overall bad pixel mask
        for i, bad_pixel_mask in enumerate(bad_pixel_mask_list):
            if bad_pixel_mask.shape == shape:
                if i == 0:
                    combined_mask = bad_pixel_mask
                else:
                    #   TODO: Check which makes more sense: & or |
                    combined_mask = combined_mask & bad_pixel_mask

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


def cross_correlate_images(image_1, image_2, maximum_shift_x, maximum_shift_y,
                           debug):
    """
        Cross correlation:

        Adapted from add_images written by Nadine Giese for use within the
        astrophysics lab course at Potsdam University.
        The source code may be modified, reused, and distributed as long as
        it retains a reference to the original author(s).

        Idea and further information:
        http://en.wikipedia.org/wiki/Phase_correlation

        Parameters
        ----------
        image_1                 : `numpy.ndarray`
            Data of first image

        image_2                 : `numpy.ndarray`
            Data of second image

        maximum_shift_x         : `integer`
            Maximal allowed shift between the images in Pixel - X axis

        maximum_shift_y         : `integer`
            Maximal allowed shift between the images in Pixel - Y axis

        debug                   : `boolean`
            If True additional plots will be created
    """

    image_dimension_x = image_1.shape[1]
    image_dimension_y = image_1.shape[0]

    #   Fast fourier transformation
    image_1_fft = np.fft.fft2(image_1)
    image_2_fft = np.fft.fft2(image_2)
    image_2_fft_cc = np.conj(image_2_fft)
    fft_cc = image_1_fft * image_2_fft_cc
    fft_cc = fft_cc / np.absolute(fft_cc)
    # cc = np.fft.ifft2(fft_cc)
    cc_matrix = np.fft.fft2(fft_cc)
    cc_matrix[0, 0] = 0.

    #   Limit to allowed shift range
    for i in range(maximum_shift_x, image_dimension_x - maximum_shift_x):
        for j in range(0, image_dimension_y):
            cc_matrix[j, i] = 0
    for i in range(0, image_dimension_x):
        for j in range(maximum_shift_y, image_dimension_y - maximum_shift_y):
            cc_matrix[j, i] = 0

    #   Debug plot showing the cc matrix
    if debug:
        plot.cross_correlation_matrix(image_2, cc_matrix)

    #   Find the maximum in cc to identify the shift
    index_1, index_2 = np.unravel_index(cc_matrix.argmax(), cc_matrix.shape)

    # if index_2 > image_dimension_x/2.:
    # index_2 = (index_2-1)-image_dimension_x+1
    # else:
    # index_2 = index_2 - 1
    # if index_1 > image_dimension_y/2.:
    # index_1 = (index_1-1)-image_dimension_y+1
    # else:
    # index_1 = index_1 - 1
    if index_2 > image_dimension_x / 2.:
        index_2 = index_2 - image_dimension_x - 2
    else:
        index_2 = index_2 + 2
    if index_1 > image_dimension_y / 2.:
        index_1 = index_1 - image_dimension_y - 2
    else:
        index_1 = index_1 + 2

    return -index_1, -index_2


def calculate_min_max_image_shifts(shifts, python_format=False):
    """
        Calculate shifts

        Parameters
        ----------
        shifts              : `numpy.ndarray`
            2D numpy array with the image shifts in X and Y direction

        python_format       : `boolean`
            If True the python style of image ordering is used. If False the
            natural/fortran style of image ordering is use.
            Default is ``False``.

        Returns
        -------
        minimum_shift_x     : `float'
            Minimum shift in X direction

        maximum_shift_x     : `float'
            Maximum shift in X direction

        minimum_shift_y     : `float'
            Minimum shift in Y direction

        maximum_shift_y     : `float'
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
    minimum_shift_x = np.min(shifts[id_x, :])
    maximum_shift_x = np.max(shifts[id_x, :])

    minimum_shift_y = np.min(shifts[id_y, :])
    maximum_shift_y = np.max(shifts[id_y, :])

    return minimum_shift_x, maximum_shift_x, minimum_shift_y, maximum_shift_y


def calculate_image_shifts_core(image_ccd_object, reference_ccd_object,
                                image_id, file_name_image,
                                correlation_method='skimage'):
    """
        Calculate image shifts using different methods

        Parameters
        ----------
        image_ccd_object        : `astropy.nddata.CCDData` object
            Image data

        reference_ccd_object    : `astropy.nddata.CCDData` object
            Image data of the reference image

        image_id                : `integer`
            ID of the image

        file_name_image         : `string`
            Name of the image

        correlation_method      : `string`, optional
            Method to use for image alignment.
            Possibilities: 'own'     = own correlation routine based on
                                       phase correlation, applying fft to
                                       the images
                           'skimage' = phase correlation with skimage'
                           'aa'      = astroalign module
            Default is 'skimage'.

        Returns
        -------
        image_id                : `integer`
            ID of the image

        image_shift             : `tuple`
            Shifts of the image in X and Y direction

        flip_necessary          : `boolean`
            If `True` the image needs to be flipped
    """
    #   Get reference image, reference mask, and corresponding file name
    reference_data = reference_ccd_object.data
    reference_mask = np.invert(reference_ccd_object.mask)
    reference_pier = reference_ccd_object.meta.get('PIERSIDE', 'EAST')

    #   Image and mask to compare with
    current_ccd = image_ccd_object
    current_data = image_ccd_object.data
    current_mask = np.invert(image_ccd_object.mask)

    #   Image pier side
    current_pier = image_ccd_object.meta.get('PIERSIDE', 'EAST')

    #   Flip if pier side changed
    if current_pier != reference_pier:
        current_ccd = ccdp.transform_image(
            image_ccd_object,
            np.flip,
            axis=(0, 1),
        )
        current_data = np.flip(current_data, axis=(0, 1))
        current_mask = np.flip(current_mask, axis=(0, 1))
        flip_necessary = True
    else:
        flip_necessary = False

    #   Calculate shifts
    if correlation_method == 'skimage':
        image_shift = phase_cross_correlation(
            reference_data,
            current_data,
            reference_mask=reference_mask,
            moving_mask=current_mask,
        )
    elif correlation_method == 'own':
        image_shift = cross_correlate_images(
            reference_data,
            current_data,
            1000,
            1000,
            False,
        )
    elif correlation_method == 'aa':
        #   Map with endianness symbols
        endian_map = {
            '>': 'big',
            '<': 'little',
            '=': sys.byteorder,
            '|': 'not applicable',
        }
        if endian_map[image_ccd_object.data.dtype.byteorder] != sys.byteorder:
            image_ccd_object.data = image_ccd_object.data.byteswap().newbyteorder()
            reference_ccd_object.data = reference_ccd_object.data.byteswap().newbyteorder()
            u_img = image_ccd_object.uncertainty.array.byteswap().newbyteorder()
            image_ccd_object.uncertainty = StdDevUncertainty(u_img)
            u_re = reference_ccd_object.uncertainty.array.byteswap().newbyteorder()
            reference_ccd_object.uncertainty = StdDevUncertainty(u_re)

        #   Determine transformation between the images
        try:
            transformation_coefficiants, (_, _) = aa.find_transform(
                current_ccd,
                reference_ccd_object,
                detection_sigma=3.0,
            )

            image_shift = (
                transformation_coefficiants.translation[1],
                transformation_coefficiants.translation[0]
            )
        except:
            image_shift = (0., 0.)
            terminal_output.print_to_terminal(
                f"WARNING: Offset determination for image {image_id}"
                " failed. Assume offset is 0.",
                style_name='WARNING',
            )
    else:
        #   This should not happen...
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Image correlation method '
            f'{correlation_method} not known\n {style.Bcolors.ENDC}'
        )
    terminal_output.print_to_terminal(
        f'\t{image_id}\t{image_shift[1]:+.1f}\t{image_shift[0]:+.1f}'
        f'\t{file_name_image}',
        indent=0,
    )

    return image_id, image_shift, flip_necessary


def calculate_image_shifts(image_file_collection, id_reference_image, comment,
                           correlation_method='skimage'):
    """
        Calculate image shifts

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            Image file collection

        id_reference_image      : `integer`
            Number of the reference image

        comment                 : `string`
            Information regarding for which images the shifts will be
            calculated

        correlation_method      : `string`, optional
            Method to use for image alignment.
            Possibilities: 'own'     = own correlation routine based on
                                       phase correlation, applying fft to
                                       the images
                           'skimage' = phase correlation with skimage'
                           'aa'      = astroalign module
            Default is 'skimage'.


        Returns
        -------
        image_shift             : `numpy.ndarray`
            Shifts of the images in X and Y direction

        flip_necessary          : `numpy.ndarray`
            Flip necessary to account for pier flips
    """
    #   Number of images
    n_files = len(image_file_collection.files)

    #   Get reference image, reference mask, and corresponding file name
    reference_file_name = image_file_collection.files[id_reference_image]
    reference_ccd_object = CCDData.read(reference_file_name)

    #   Prepare an array for the shifts
    image_shift = np.zeros((2, n_files))
    flip_necessary = np.zeros(n_files, dtype=bool)

    terminal_output.print_to_terminal(comment, indent=0)
    terminal_output.print_to_terminal('\tImage\tx\ty\tFilename', indent=0)
    terminal_output.print_to_terminal(
        '\t----------------------------------------',
        indent=0,
    )
    terminal_output.print_to_terminal(
        f'\t{id_reference_image}\t{0:+.1f}\t{0:+.1f}\t'
        f'{reference_file_name.split("/")[-1]}',
        indent=0,
    )

    # from ..analyze import aux as aux_ana
    # #   Initialize multiprocessing object
    # n_cores_multiprocessing=6
    # n_cores_multiprocessing=3
    # n_cores_multiprocessing=2
    # executor = aux_ana.Executor(n_cores_multiprocessing)

    #   Calculate image shifts
    for i, (current_ccd_object, file_name) in enumerate(image_file_collection.ccds(return_fname=True)):
        if i != id_reference_image:
            _, image_shift[:, i], flip_necessary[i] = calculate_image_shifts_core(
                # _, _, flip_necessary[i] = calculate_image_shifts_core(
                current_ccd_object,
                reference_ccd_object,
                i,
                file_name,
                correlation_method=correlation_method,
            )
            # calculate_image_shifts_core(
            # current_ccd_object,
            # reference_ccd_object,
            # img_id,
            # method='skimage',
            # )
            # executor.schedule(
            # calculate_image_shifts_core,
            # args=(
            # current_ccd_object,
            # reference_ccd_object,
            # i,
            # file_name,
            # ),
            # kwargs={
            # 'method':method,
            # }
            # )

    # #   Close multiprocessing pool and wait until it finishes
    # executor.wait()

    # #   Exit if exceptions occurred
    # if executor.err is not None:
    # raise RuntimeError(
    # f'\n{style.Bcolors.FAIL}Image offset determination failed '
    # f':({style.Bcolors.ENDC}'
    # )

    ####
    # #   Sort multiprocessing results
    # #
    # #   Extract results
    # res = executor.res

    # #   Sort observation times and images & build dictionary for the
    # #   tables with the extraction results
    # #for j in range(0, n_files):
    # for ref_id, shift_i, flip_i in res:
    # print(ref_id, shift_i, flip_i)
    # image_shift[:,ref_id] = shift_i
    # flip_necessary[ref_id] = flip_i

    terminal_output.print_to_terminal('')

    return image_shift, flip_necessary


def image_shift_astroalign_method(reference_ccd_object, current_ccd_object):
    """
        Calculate image shifts using the astroalign method

        Parameters
        ----------
        reference_ccd_object        : `astropy.nddata.CCDData` object
            Reference image

        current_ccd_object          : `astropy.nddata.CCDData` object
            Current image

        Returns
        -------
                                    : `astropy.nddata.CCDData` object
            Aligned image
    """
    #   Map with endianness symbols
    endian_map = {
        '>': 'big',
        '<': 'little',
        '=': sys.byteorder,
        '|': 'not applicable',
    }
    if endian_map[current_ccd_object.data.dtype.byteorder] != sys.byteorder:
        current_ccd_object.data = current_ccd_object.data.byteswap().newbyteorder()
        reference_ccd_object.data = reference_ccd_object.data.byteswap().newbyteorder()
        current_ccd_object.uncertainty = StdDevUncertainty(
            current_ccd_object.uncertainty.array.byteswap().newbyteorder()
        )
        reference_ccd_object.uncertainty = StdDevUncertainty(
            reference_ccd_object.uncertainty.array.byteswap().newbyteorder()
        )

    #   Determine transformation between the images
    transformation_coefficiants, (_, _) = aa.find_transform(
        current_ccd_object,
        reference_ccd_object,
        detection_sigma=3.0,
    )

    #   Transform image data
    image_data, footprint_mask = aa.apply_transform(
        transformation_coefficiants,
        current_ccd_object,
        reference_ccd_object,
        propagate_mask=True,
    )

    #   Transform uncertainty array
    image_uncertainty, _ = aa.apply_transform(
        transformation_coefficiants,
        current_ccd_object.uncertainty.array,
        reference_ccd_object.uncertainty.array,
    )

    #   Build new CCDData object
    return CCDData(
        image_data,
        mask=footprint_mask,
        meta=current_ccd_object.meta,
        unit=current_ccd_object.unit,
        uncertainty=StdDevUncertainty(image_uncertainty),
    )


def image_shift_optical_flow_method(reference_ccd_object, current_ccd_object):
    """
        Calculate image shifts using the optical flow method

        Parameters
        ----------
        reference_ccd_object        : `astropy.nddata.CCDData` object
            Reference image

        current_ccd_object          : `astropy.nddata.CCDData` object
            Current image

        Returns
        -------
                                    : `astropy.nddata.CCDData` object
            Aligned image
    """
    #   Prepare data, mask, and uncertainty arrays
    current_data = current_ccd_object.data
    current_mask = current_ccd_object.mask
    current_uncertainty = current_ccd_object.uncertainty.array

    #   Compute optical flow
    flow_v, flow_u = optical_flow_tvl1(reference_ccd_object.data, current_data)

    #   Prepare grid for flow map
    image_dimension_x, image_dimension_y = reference_ccd_object.data.shape
    row_coordinates, column_coordinates = np.meshgrid(
        np.arange(image_dimension_x),
        np.arange(image_dimension_y),
        indexing='ij',
    )

    #   Registrate image data, mask, and uncertainty
    image_out_data = warp(
        current_data,
        np.array([row_coordinates + flow_v, column_coordinates + flow_u]),
        mode='edge',
    )
    image_out_mask = warp(
        current_mask,
        np.array([row_coordinates + flow_v, column_coordinates + flow_u]),
        mode='edge',
    )
    image_out_uncertainty = warp(
        current_uncertainty,
        np.array([row_coordinates + flow_v, column_coordinates + flow_u]),
        mode='edge',
    )

    #   Build new CCDData object
    return CCDData(
        image_out_data,
        mask=image_out_mask,
        meta=current_ccd_object.meta,
        unit=current_ccd_object.unit,
        uncertainty=StdDevUncertainty(image_out_uncertainty),
    )


def make_index_from_shifts(shifts, id_current_image):
    """
        Calculate image index positions from image shifts

        Parameters
        ----------
        shifts                          : `numpy.ndarray`
            Shifts of all images in X and Y direction

        id_current_image                : `integer`
            ID of the current image

        Returns
        -------
        x_start, x_end, y_start, y_end  : `float`
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

    return x_start, x_end, y_start, y_end


def trim_image(image, image_id, n_files, image_shift,
               correlation_method='skimage', verbose=False):
    """
        Trim image based on a shift compared to a reference image

        Parameters
        ----------
        image                   : `astropy.nddata.CCDData`
            Image

        image_id                : `integer`
            Number of the image in the sequence

        n_files                 : `integer`
            Number of all images

        image_shift             : `numpy.ndarray`
            Shift of this specific image in X and Y direction

        correlation_method      : `string`, optional
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

        verbose                 : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    """
    if verbose:
        #   Write status to console
        print(f"\r\tApply shift to image {image_id + 1}/{n_files}\n")

    if correlation_method in ['own', 'skimage']:
        #   Ensure full pixel shifts
        if not issubclass(type(image_shift[0, 0]), np.integer):
            image_shift = image_shift.astype('int')

        #   Calculate indexes from image shifts
        x_start, x_end, y_start, y_end = make_index_from_shifts(
            image_shift,
            image_id,
        )
    elif correlation_method == 'aa':
        #   Calculate maximum and minimum shifts
        min_shift_x, max_shift_x, min_shift_y, max_shift_y = calculate_min_max_image_shifts(
            image_shift,
            python_format=True,
        )

        #   Shift image on sub pixel basis
        image = ccdp.transform_image(
            image,
            shift_scipy,
            shift=image_shift[:, image_id],
            order=1,
        )

        #   Set trim margins
        if min_shift_x > 0:
            x_start = int(math.ceil(max_shift_x))
            x_end = 0
        elif min_shift_x < 0 and max_shift_x < 0:
            x_start = 0
            x_end = int(math.ceil(np.abs(min_shift_x))) * -1
        else:
            x_start = int(math.ceil(max_shift_x))
            x_end = int(math.ceil(np.abs(min_shift_x))) * -1

        if min_shift_y > 0:
            y_start = int(math.ceil(max_shift_y))
            y_end = 0
        elif min_shift_y < 0 and max_shift_y < 0:
            y_start = 0
            y_end = int(math.ceil(np.abs(min_shift_y))) * -1
        else:
            y_start = int(math.ceil(max_shift_y))
            y_end = int(math.ceil(np.abs(min_shift_y))) * -1
    else:
        raise ValueError(
            f'{style.Bcolors.FAIL}Shift method not known. Expected: '
            f'"pixel" or "sub_pixel", but got '
            f'"{correlation_method}" {style.Bcolors.ENDC}'
        )

    #   Trim the image
    return ccdp.trim_image(
        image[y_start:image.shape[0] + y_end, x_start:image.shape[1] + x_end]
    )


def prepare_reduction(output_dir, bias_path, darks_path, flats_path,
                      images_path, raw_files_path, temp_dir, image_type=None):
    """
        Prepare directories and files for the reduction procedure

        Parameters
        ----------
        output_dir          : `list` of `string`s
            Path to the directory where the master files should be saved to

        bias_path           : `list` of `string`s
            Path to the bias or '?'

        darks_path          : `list` of `string`s
            Path to the darks or '?'

        flats_path          : `list` of `string`s
            Path to the flats or '?'

        images_path         : `list` of `string`s
            Path to the science images or '?'

        raw_files_path      : 'string`
            Path to all raw images or '?', if bias, darks, flats, and images
            are provided.

        temp_dir            : `tempfile.TemporaryDirectory`
            Temporary directory to store the symbolic links to the images

        image_type          : `dict` of `string`, optional
            Image type to select. Possibilities: bias, dark, flat, light
            Default is ``None``.

        Returns
        -------
        raw_files_path      : 'string`
            Points to the path with the raw files. Either the temporary
            directory or the already provided 'raw_files_path' directory.
    """
    ###
    #   Check directories
    #
    terminal_output.print_to_terminal("Check if directories exists...")

    checks.check_output_directories(output_dir)
    if raw_files_path == '?':
        checks.check_path(darks_path)
        checks.check_path(flats_path)
        checks.check_path(images_path)
        if bias_path != '?':
            checks.check_path(bias_path)

        #   Find sub directories
        darks_path = checks.list_subdirectories(darks_path)
        flats_path = checks.list_subdirectories(flats_path)
        images_path = checks.list_subdirectories(images_path)
        if bias_path != '?':
            bias_path = checks.list_subdirectories(bias_path)

    else:
        checks.check_path(raw_files_path)

    ###
    #   Check consistency between images and fits header keywords
    #
    if raw_files_path == '?':
        terminal_output.print_to_terminal(
            "Check header keywords for consistency...",
        )
        if bias_path != '?':
            bias_path_new = []
            for path in bias_path:
                if image_type is not None:
                    bias_path_new.append(
                        check_filter_keywords(path, image_type['bias'])
                    )
                else:
                    bias_path_new.append(check_filter_keywords(path, 'bias'))
            bias_path = bias_path_new

        darks_path_new = []
        for path in darks_path:
            if image_type is not None:
                darks_path_new.append(
                    check_filter_keywords(path, image_type['dark'])
                )
            else:
                darks_path_new.append(check_filter_keywords(path, 'dark'))
        darks_path = darks_path_new

        flats_path_new = []
        for path in flats_path:
            if image_type is not None:
                flats_path_new.append(
                    check_filter_keywords(path, image_type['flat'])
                )
            else:
                flats_path_new.append(check_filter_keywords(path, 'flat'))
        flats_path = flats_path_new

        images_path_new = []
        for path in images_path:
            if image_type is not None:
                images_path_new.append(
                    check_filter_keywords(path, image_type['light'])
                )
            else:
                images_path_new.append(check_filter_keywords(path, 'light'))
        images_path = images_path_new

    ###
    #   Prepare temporary directory, if individual
    #   directories were defined above
    #
    if raw_files_path == '?':
        #   Combine directories
        raw_files_path = darks_path + flats_path + images_path
        if bias_path != '?':
            raw_files_path = raw_files_path + bias_path

        #   Link all files to the temporary directory
        make_symbolic_links(raw_files_path, temp_dir)

        raw_files_path = temp_dir.name
    else:
        raw_files_path = checks.list_subdirectories(raw_files_path)

        if len(raw_files_path) == 1:
            raw_files_path = raw_files_path[0]
        elif len(raw_files_path) > 1:
            #   Link all files to the temporary directory
            make_symbolic_links(raw_files_path, temp_dir)

            raw_files_path = temp_dir.name
        else:
            #   This should not happen...
            raise RuntimeError(
                f'{style.Bcolors.FAIL}Raw file path could not be '
                f'decoded...\n {style.Bcolors.ENDC}'
            )

    return raw_files_path


def get_star_profiles(cutout_data):
    """
        Get star profiles

        Parameters
        ----------
        cutout_data                   : `numpy.ndarray`
            Image (square) extracted around the star

        Returns
        -------
        profile_x_direction           : `numpy.ndarray`
            Profile in X direction

        profile_y_direction           : `numpy.ndarray`
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


def interpolate_fwhm(profile):
    """
        Find FWHM by means of interpolation on a stellar profile

        Idea: https://stackoverflow.com/questions/52320873/computing-the-fwhm-of-a-star-profile

        Parameters
        ----------
        profile        : `numpy.ndarray`
            Stellar profile along a specific axis

        Returns
        -------
                        : `float`
            FWHM of the profile
    """
    #   Prepare interpolation
    half_maximum = 0.5 * np.max(profile)
    x_data = np.linspace(0, len(profile), len(profile))

    #   Do the interpolation
    spline = UnivariateSpline(x_data, profile - half_maximum, s=0)
    r1, r2 = spline.roots()

    return r2 - r1


def estimate_fwhm(image_path, output_dir, image_type, plot_subplots=False,
                  indent=2):
    """
        Combine images

        Parameters
        ----------
        image_path              : `pathlib.Path`
            Path to the images

        output_dir              : `pathlib.Path`
            Path to the directory where the master files should be saved to

        image_type              : `list`
            Header keyword characterizing the image type for which the
            shifts shall be determined

        plot_subplots           : `boolean`, optional
            Plot subplots around the stars used to estimate the FWHM
            Default is ``False``.

        indent                  : `integer`
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
            object_tbl = dao_finder(img_ccd.data - median)

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
                plot.cutouts_fwhm_stars(
                    out_path,
                    len(objects_tbl_filtered),
                    object_cutouts,
                    filter_,
                    base_aux.get_basename(file_name),
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
                except:
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


def check_master_files_on_disk(image_path, image_type_dict,
                               dark_exposure_times, filter_list, check_bias):
    """
        Check if master files are already prepared

        Parameters
        ----------
        image_path              : `string` or `pathlib.Path`
            Path to the images

        image_type_dict         : `dictionary`
            Image types of the images.
            Possibilities: bias, dark, flat, light

        dark_exposure_times     : `list`
            Exposure times of the raw dark images

        filter_list             : `list` or `set`
            Filter that have been used

        check_bias              : `boolean`
            If True bias will be checked as well.

        Returns
        -------
        master_available        : `boolean`
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


def flip_image(image_file_collection, output_path):
    """
        Flip images in X and Y direction

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            Image file collection

        output_path             : `pathlib.Path`
            Path to save the individual images

        Returns
        -------
                                : `ccdproc.ImageFileCollection`
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


def bin_image(image_file_collection, output_path, binning_value):
    """
        Bin images in X and Y direction

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            Image file collection

        output_path             : `pathlib.Path`
            Path to save the individual images

        binning_value           : `integer`
            Number of pixel that the image should be binned in X and Y
            direction.

        Returns
        -------
                                : `ccdproc.ImageFileCollection`
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


#   TODO: Check if this function can be merged with `trim_image` -> Used by N1 script
def trim_image_simple(image_file_collection, output_path,
                      redundant_pixel_x_start=100, redundant_pixel_x_end=100,
                      redundant_pixel_y_start=100, redundant_pixel_y_end=100):
    """
        Trim images in X and Y direction

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            Image file collection

        output_path             : `pathlib.Path`
            Path to save the individual images

        redundant_pixel_x_start : `integer`
            Number of Pixel to be removed from the start of the image in
            X direction.

        redundant_pixel_x_end   : `integer`
            Number of Pixel to be removed from the end of the image in
            X direction.

        redundant_pixel_y_start : `integer`
            Number of Pixel to be removed from the start of the image in
            Y direction.

        redundant_pixel_y_end   : `integer`
            Number of Pixel to be removed from the end of the image in
            Y direction.

        Returns
        -------
                            : `ccdproc.ImageFileCollection`
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
        #   Trim image
        trimmed_image = ccdp.trim_image(image[
            redundant_pixel_y_start:-redundant_pixel_y_end,
            redundant_pixel_x_start:-redundant_pixel_x_end
            ])

        #   Save the result
        trimmed_image.write(output_path_trimmed / file_name, overwrite=True)

    #   Return new image file collection
    return ccdp.ImageFileCollection(output_path_trimmed)


def find_wcs(input_dir, output_dir, reference_image_id=0,
             force_wcs_determination=False, wcs_method='astrometry',
             x_pixel_coordinates=None, y_pixel_coordinates=None, indent=2):
    """
        Determine the WCS of the reference image and add the WCS to all
        images in the input directory. The latter is to save computing time.
        It is assumed that the images are already aligned and trimmed to
        the same filed of view. However, the observation time of these
        images will be overwritten by this procedure.

        Parameters
        ----------
        input_dir               : `pathlib.Path` or string
            Path to the input directory.

        output_dir              : `pathlib.Path` or string
            Path to the output directory.

        reference_image_id      : `integer', optional
            ID of the reference image.
            Default is ``0``.

        force_wcs_determination : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        wcs_method              : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        x_pixel_coordinates     : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        y_pixel_coordinates     : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        indent                  : `integer`, optional
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
    #    probability of detecting a large number of stars
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

    ###
    #   Get reference image
    #
    reference_image_path = ifc_filtered.files[reference_image_id]

    reference_image = base_aux.Image(
        reference_image_id,
        reference_filter,
        'target',
        reference_image_path,
        output_dir,
    )

    base_aux.calculate_field_of_view(reference_image)

    #   Test if the image contains already a WCS
    wcs_available = base_aux.check_wcs_exists(reference_image)

    ###
    #   Determine WCS
    #
    if not wcs_available or force_wcs_determination:
        wcs = find_wcs_core(
            reference_image,
            wcs_method=wcs_method,
            x_pixel_coordinates=x_pixel_coordinates,
            y_pixel_coordinates=y_pixel_coordinates,
            indent=indent,
        )

        ###
        #   Add WCS to images
        #
        if wcs is not None:
            for image, file_name in image_file_collection.ccds(return_fname=True):
                image.wcs = wcs

                #   Save the image
                image.write(output_dir / file_name, overwrite=True)


def find_wcs_all_images(input_dir, output_dir, force_wcs_determination=False,
                        wcs_method='astrometry', x_pixel_coordinates=None,
                        y_pixel_coordinates=None, only_combined_images=False,
                        image_type=None, indent=2):
    """
        Determine the WCS of each image individually. Images can be filtered
        based on image type and the 'combined' keyword.

        Parameters
        ----------
        input_dir               : `pathlib.Path` or string
            Path to the input directory.

        output_dir              : `pathlib.Path` or string
            Path to the output directory.

        force_wcs_determination        : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        wcs_method              : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        x_pixel_coordinates     : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        y_pixel_coordinates     : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        only_combined_images    : `boolean`, optional
            Filter for images that have a 'combined' fits header keyword.
            Default is ``False``.

        image_type              : `list` of `string` or `None`, optional
            Image type to select. Possibilities: bias, dark, flat, light
            Default is ``None``.

        indent                  : `integer`, optional
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
        image_object = base_aux.Image(
            i,
            'filter',
            'target',
            file_path / file_name,
            output_dir,
        )
        base_aux.calculate_field_of_view(image_object, verbose=False)

        #   Test if the image contains already a WCS
        wcs_available = base_aux.check_wcs_exists(image_object)

        if not wcs_available or force_wcs_determination:
            wcs = find_wcs_core(
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


def find_wcs_core(image, wcs_method='astrometry', x_pixel_coordinates=None,
                  y_pixel_coordinates=None, indent=2):
    """
        Branch between different WCS methods

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        wcs_method              : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        x_pixel_coordinates     : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        y_pixel_coordinates     : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        indent                  : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        wcs                   : `astropy.wcs.WCS`
            WCS information
    """
    #   astrometry.net:
    if wcs_method == 'astrometry':
        try:
            wcs = base_aux.find_wcs_astrometry(
                image,
                wcs_working_dir='/tmp/',
                indent=indent,
            )
        except:
            terminal_output.print_to_terminal(
                "No WCS solution found :(\n",
                indent=indent,
                style_name='WARNING',
            )
            wcs = None

    #   ASTAP program
    elif wcs_method == 'astap':
        try:
            wcs = base_aux.find_wcs_astap(
                image,
                indent=indent,
            )
            terminal_output.print_to_terminal('')
        except:
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
            wcs = base_aux.find_wcs_twirl(
                image,
                x_pixel_coordinates,
                y_pixel_coordinates,
                indent=indent,
            )
        except:
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


def update_header_information(image, n_image_stacked=1, new_target_name=None):
    """
        Updates Header information. Adds among other Header keywords required
        for the GRANDMA project.

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        n_image_stacked     : `integer`, optional
            Number of stacked images
            Default is ``1``.

        new_target_name     : str or None, optional
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
        filter_system = calibration_data.filter_systems[filter_]
        image.meta['FILTER-S'] = filter_system
    except Exception as e:
        terminal_output.print_to_terminal(
            f"Filter system could not be determined:\n {e}",
            style_name='WARNING',
        )
