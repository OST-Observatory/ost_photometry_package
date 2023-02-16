############################################################################
####                            Libraries                               ####
############################################################################

import os

import sys

from pathlib import Path

import numpy as np

import math

from scipy.ndimage import shift as shift_scipy

from astropy.nddata import CCDData, StdDevUncertainty
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars

from scipy.interpolate import UnivariateSpline

import ccdproc as ccdp

from skimage.registration import (
    phase_cross_correlation,
    optical_flow_tvl1,
    #optical_flow_ilk,
    )
from skimage.transform import warp

import astroalign as aa

from .. import style, checks, calibration_data, terminal_output
from .. import aux as base_aux

from . import plot

############################################################################
####                        Routines & definitions                      ####
############################################################################

def make_symbolic_links(path_list, temp_dir):
    '''
        Make symbolic links

        Parameters
        ----------
        path_list           : `list` of `string`s
            List with paths to files

        temp_dir            : `tempfile.TemporaryDirectory`
            Temporary directory to store the symbolic links
    '''
    #   Set current working directory
    pwd = os.getcwd()

    #   Loop over directories
    for path in path_list:
        #   Get file list
        files = os.listdir(path)
        #   Loop over files
        for fil in files:
            if os.path.isfile(os.path.join(path, fil)):
                #   Check if a file of the same name already exist in the
                #   temp directory
                if os.path.isfile(os.path.join(temp_dir.name, fil)):
                    fil_new = base_aux.random_string_generator(7)+'_'+fil
                else:
                    fil_new = fil

                #   Fill temp directory with file links
                os.symlink(
                    os.path.join(pwd, path, fil),
                    os.path.join(temp_dir.name, fil_new),
                    )



def inv_median(a):
    '''
        Inverse median

        Parameters
        ----------
        a               : `numpy.ndarray`
            Data

        Returns
        -------
                        : `float`
            Inverse median
    '''
    return 1 / np.median(a)


def get_instruments(ifc):
    '''
        Extract instrument informations.

        Parameters
        ----------
        ifc             : `ccdproc.ImageFileCollection`
            Image file collection with all images

        Returns
        -------
        instruments           : `set`
            List of instruments
    '''
    #   Except if no files are found
    if not ifc.files:
        raise RuntimeError(
            f'{style.bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.bcolors.ENDC}'
            )

    #   Get instruments (set() allows to return only unique values)
    instruments = set(ifc.summary['instrume'])

    return instruments


def get_imaging_soft(ifc):
    '''
        Extract imaging software version.

        Parameters
        ----------
        ifc             : `ccdproc.ImageFileCollection`
            Image file collection with all images

        Returns
        -------
        imaging_soft        : `set`
            List of instruments
    '''
    #   Except if no files are found
    if not ifc.files:
        raise RuntimeError(
            f'{style.bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.bcolors.ENDC}'
            )

    #   Imaging software (set() allows to return only unique values)
    imaging_soft = set(ifc.summary['swcreate'])

    return imaging_soft


def get_exposure_times(ifc, img_type):
    '''
        Extract the exposure time of a specific image type from an image
        collections.

        Parameters
        ----------
        ifc             : `ccdproc.ImageFileCollection`
            Image file collection with all images

        img_type        : `string`
            Image type to select. Possibilities: bias, dark, flat, light

        Returns
        -------
        times           : `list`
            List of exposure times
    '''
    #   Calculate mask to restrict images to the provided image type
    mask = [True if file in img_type else False \
            for file in ifc.summary['imagetyp']]

    #   Except if no files are found in this directory
    if not np.any(mask):
        raise RuntimeError(
            f'{style.bcolors.FAIL}No images with image type {img_type} '
            f'found -> EXIT\n\t=> Check paths to the images!'
            f'{style.bcolors.ENDC}'
            )

    #   Exposure times (set() allows to return only unique values)
    times = set(ifc.summary['exptime'][mask])

    return times


def find_nearest_core(test_exposure_time, exposure_times, tolerance=0.5):
    '''
        Find the nearest match between a test exposure time and a list of
        exposure times, raising an error if the difference in exposure time
        is more than the tolerance.

        Parameters
        ----------
        test_exposure_time  : `float`
            Exposure time for which a match from a list of exposure times
            should be found.

        exposure_times      : `list of floats`
            Exposure times for which there are images

        tolerance           : `float` or `None`, optional
            Maximum difference, in seconds, between the image and the
            closest entry from the exposure time list. Set to ``None`` to
            skip the tolerance test.
            Default is ``0.5``.

        Returns
        -------
                            : `boolean`
            Image within the tolerance of an entry from the exposure time
            list

        closest_exposure    : `float`
            Closest exposure time to the image.
    '''
    #   Find closest exposure time
    exposures        = np.array(list(exposure_times))
    idx              = np.argmin(np.abs(exposures - test_exposure_time))
    closest_exposure = exposures[idx]

    #   Check if closest exposure time is within the tolerance
    if (tolerance is not None and
        np.abs(test_exposure_time - closest_exposure) > tolerance):

        return False, closest_exposure

    return True, closest_exposure


def find_nearest_exposure(image, exposure_times, tolerance=0.5):
    '''
        Find the nearest exposure time of a list of exposure times to that
        of an image, raising an error if the difference in exposure time is
        more than the tolerance.

        Parameters
        ----------
        image               : `astropy.nddata.CCDData`
            Image for which a matching exposure time is needed

        exposure_times      : `list`
            Exposure times for which there are images

        tolerance           : `float` or `None`, optional
            Maximum difference, in seconds, between the image and the
            closest entry from the exposure time list. Set to ``None`` to
            skip the tolerance test.
            Default is ``0.5``.

        Returns
        -------
                            : `boolean`
            Image within the tolerance of an entry from the exposure time
            list

        closest_exposure    : `float`
            Closest exposure time to the image.
    '''
    #   Get exposure time from the image
    img_expo_time = image.header['exptime']

    return find_nearest_core(img_expo_time,exposure_times, tolerance=tolerance)


def get_image_type(ifc, image_type, image_class=None):
    '''
        From an image file collection get the existing image type from a
        list of possible images

        Parameters
        ----------
        idc             : `ccdproc.ImageFileCollection`
            Image file collection

        image_type      : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light

        image_class     : `string`, optional
            Image file type class to look for.
            Default is ``None``.
    '''
    #   Create mask
    if not image_class:
        mask = [True if img_type in ifc.summary['imagetyp'] else \
                False for img_type in image_type]
    else:
        mask = [True if img_type in ifc.summary['imagetyp'] else \
                False for img_type in image_type[image_class]]

    #   Get image type
    id_type = np.argwhere(mask).ravel()
    if not id_type.size:
        return None

    #   Get ID
    #   Restricted to only one result -> this is currently necessary
    id_type = id_type[0]

    #   Return the image type
    if not image_class:
        return image_type[id_type]
    else:
        return image_type[image_class][id_type]


def check_dark_scaling(img_string, time, max_dark_time, bias_true):
    '''
        Check if scaling of dark frames to the given exposure time 'time' is
        possible and handles exceptions

        Parameters
        ----------
        img_string          : `string`
            String that characterizes the image type, such as 'science' or
            'flat'. This is used in the exception messages.

        time                : `float`
            Exposure time that should be checked

        max_dark_time       : `float`
            Longest dark time that is available

        bias_true           : `boolean`
            True if bias frames are available

        Returns
        -------
                            : `boolean`
            True if dark scaling is possible
    '''
    #   Raise exception if no bias frames are available
    if not bias_true:
        raise RuntimeError(
            f'{style.bcolors.FAIL}No darks with a matching exposure time '
            f'found for the {img_string} exposures with the following '
            f'exposure time {time}s -> EXIT {style.bcolors.ENDC}'
            )
    #   Check if scaling is possible -> dark frames can only be scaled
    #   to a smaller exposure time and not to a larger one because this
    #   most likely will amplify read noise
    if time < max_dark_time:
        return True
    else:
        raise RuntimeError(
            f'{style.bcolors.FAIL}Scaling of the dark frames to the '
            f'exposure time of the {img_string} ({time}s) is not possible, '
            f'since the longest dark exposure is only {max_dark_time}s and '
            f'dark frames should not be scaled "up".'
            f'-> EXIT{style.bcolors.ENDC}'
            )


def check_exposure_times(img_string, test_times, dark_times, bias_true,
                         tolerance=0.5):
    '''
        Check if relevant dark exposures are available for the exposure
        times in the supplied list

        Parameters
        ----------
        img_string          : `string`
            String that characterizes the image type, such as 'science' or
            'flat'. This is used in the exception messages.

        test_times          : `list` of `float`
            Exposure times that should be checked

        dark_times          : `list` of `float`
            Dark exposure times that are available

        bias_true           : `boolean`
            True if bias frames are available

        tolerance           : 'float', optional
            Tolerance between science and dark exposure times in s.
            Default is ``0.5``s.

        Returns
        -------
        scale_necessary      : `boolean`
            True if dark scaling is possible
    '''
    #   Loop over exposure times
    for time in test_times:
        #   Find nearest dark frame
        valid, closest_dark = find_nearest_core(
            time,
            dark_times,
            tolerance=tolerance,
        )
        #   In case there is no valid dark, check if scaling is possible
        if not valid:
            scale_necessary = check_dark_scaling(
                img_string,
                time,
                np.max(dark_times),
                bias_true,
            )
            return scale_necessary
        return False


def check_filter_keywords(path, image_type):
    '''
        Consistency check - Check if the image type of the images in 'path'
        fit
                            to one supplied with 'image_type'.
        Parameters
        ----------
        path            : `string`
            File path to check

        image_type      : `string`
            Internal image type of the images in 'path' should have
    '''
    #   Sanitize the provided path
    file_path = Path(path)

    #   Check weather path exists
    if not file_path.exists():
        raise RuntimeError(
            f'{style.bcolors.FAIL}The provided path ({path}) does not '
            f'exists {style.bcolors.ENDC}'
            )

    #   Create image collection
    ifc = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not ifc.files:
        return file_path

    #   Get and check imaging software
    #   (set() allows to return only unique values)
    imaging_soft = get_imaging_soft(ifc)
    if len(imaging_soft) > 1:
        terminal_output.print_terminal(
            imaging_soft,
            string="Images are taken with multiple software versions: {}. "\
                "The pipeline cannot account for that, but will try anyway...",
            indent=2,
            style_name='WARNING',
            )

    #   Get image types corresponding to the imaging software
    img_type = calibration_data.get_img_types(list(imaging_soft)[0])
    image_type = img_type[image_type]

    #   Find all images that have the correct image type
    good = ifc.files_filtered(imagetyp=image_type)

    #   Compare the results (image collection vs 'good') to find
    #   those images with a wrong image type
    list1 = list(ifc.files)
    list2 = list(good)
    res = [x for x in list1 + list2 if x not in list1 or x not in list2]

    if res:
        return sanitize_image_types(file_path, image_type)._str
        #raise RuntimeError(
            #'The following images do not have the correct '
            #'image type: \n {} \n Expected: {} \n Got: \n {} \n Path: {} '
            #''.format(res, image_type, ifc.summary['file', 'imagetyp'], path)
            #)

    return file_path._str


def sanitize_image_types(file_path, image_type):
    '''
        Sanitize image types according to prerequisites

        Parameters
        ----------
        file_path           : `pathlib.Path`

        image_type          : `string`
            Expected image type
    '''
    #   Set path
    path = Path('/tmp')
    path = path / base_aux.random_string_generator(7)
    checks.check_out(path)

    #   Sanitize
    ifc = ccdp.ImageFileCollection(file_path)

    for img, fname in ifc.ccds(ccd_kwargs={'unit':'adu'}, return_fname=True):
        img.meta['imagetyp'] = image_type

        img.write(path / fname)

    return path


def get_pixel_mask(out_path, shape):
    '''
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
    '''
    #   Load pixel mask
    try:
        mask    = CCDData.read(out_path / 'bad_pixel_mask.fit')
        if mask.shape == shape:
            #   If shape is the same, set success to True.
            success = True
        else:
            terminal_output.print_terminal(
                string="No default bad pixel mask available. Try to use "\
                       "the mask calculated in the data reduction...",
                indent=1,
                style_name='WARNING',
                )
            #   Raise RuntimeError to trigger except.
            raise RuntimeError('')
    except:
        #   If no precalculated mask are available, try to load masks
        #   calculated by  'master_dark' and 'master_flat'

        try:
            #   New image collection
            ifc = ccdp.ImageFileCollection(out_path)

            #   Get hot pixel masks
            ifc_hot_pixel = ifc.filter(imagetyp='dark mask')

            #   Get correct mask in terms of binning
            for mask_data, fname in ifc_hot_pixel.data(return_fname=True):
                if mask_data.shape == shape:
                    mask_hot = mask_data.astype('bool')

            #   Get bad pixel masks
            ifc_bad_pixel = ifc.filter(imagetyp='flat mask')

            #   Get correct mask in terms of binning
            for mask_data, fname in ifc_bad_pixel.data(return_fname=True):
                if mask_data.shape == shape:
                    mask_bad = mask_data.astype('bool')

            #   Combine mask
            mask    = mask_hot | mask_bad
            success = True
        except:
            terminal_output.print_terminal(
                string="No bad pixel mask available. Skip adding bad pixel"\
                       " mask.",
                indent=1,
                style_name='WARNING',
                )
            mask    = ''            #   This should be solved differently
            success = False

    return success, mask


def make_hot_pixel_mask(dark, gain, outdir, verbose=False):
    '''
        Make a hot pixel mask from a dark frame

        Parameters
        ----------
        dark            : `astropy.nddata.CCDData`
            Dark image

        outdir          : `string`
            Path to the directory where the master files should be saved to

        gain            : `float` or `None`, optional
            The gain (e-/adu) of the camera chip. If set to `None` the gain
            will be extracted from the FITS header.
            Default is ``None``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    out_path  = checks.check_pathlib_Path(outdir)

    #   Get exposure time
    exposure   = dark.header['EXPTIME']

    #   Get image shape
    shape1 = dark.meta['naxis1']
    shape2 = dark.meta['naxis2']

    #   Scale image with exposure time and gain
    dark       = dark.multiply(gain * u.electron / u.adu)
    dark       = dark.divide(exposure * u.second)

    #   Number of pixel
    npix = dark.shape[1] * dark.shape[0]

    #   Calculate the hot pixel mask. Increase the limit if number of
    #   hot pixels is unrealistically high
    nlimit = 2
    for i in range(0, 100):
        hot_pixels = (dark.data > nlimit)
        #   Check if number of hot pixel is realistic
        if hot_pixels.sum()/npix <= 0.03:
            break
        nlimit += 1

    if verbose:
        sys.stdout.write(
            '\r\tNumber of hot pixels: {}\n'.format(hot_pixels.sum())
            )
        sys.stdout.write(
            '\r\tLimit (e-/s/pix) used: {}\n'.format(nlimit)
            )
        sys.stdout.flush()

    #   Save mask with hot pixels
    mask_as_ccd = CCDData(
        data=hot_pixels.astype('uint8'), unit=u.dimensionless_unscaled,
        )
    mask_as_ccd.header['imagetyp'] = 'dark mask'
    file_name = 'mask_from_dark_'+str(shape1)+'x'+str(shape2)+'.fit'
    mask_as_ccd.write(out_path / file_name, overwrite=True)


def make_bad_pixel_mask(mask_list, outdir, verbose=False):
    '''
        Calculate a bad pixel mask from a list of bad pixel masks

        Parameters
        ----------
        mask_list       : `list` of `astropy.nddata.CCDData` objects
            List with bad pixel masks

        outdir          : `string`
            Path to the directory where the master files should be saved to

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    out_path  = checks.check_pathlib_Path(outdir)

    #   Get information on the image dimensions/binning
    shape_list = []
    for bpm in mask_list:
        shape_list.append(bpm.shape)
    shape_list = set(shape_list)

    #   Loop over all image dimensions/binning
    for binn in shape_list:
        #   Calculate overall bad pixel mask
        for i, bpm in enumerate(mask_list):
            if bpm.shape == binn:
                if i == 0:
                    combined_mask = bpm
                else:
                    combined_mask = combined_mask & bpm

        if verbose:
            terminal_output.print_terminal(
                binn,
                combined_mask.sum(),
                string="Number of bad pixels ({}): {}",
                indent=1,
                )

        #   Save mask
        mask_as_ccd = CCDData(
            data=combined_mask.astype('uint8'),
            unit=u.dimensionless_unscaled,
            )
        mask_as_ccd.header['imagetyp'] = 'flat mask'
        file_name = 'mask_from_ccdmask_'+str(binn[1])+'x'+str(binn[0])+'.fit'
        mask_as_ccd.write(out_path / file_name, overwrite=True)


def correl_images(img_a, img_b, xMax, yMax, debug):
    '''
        Cross correlation:

        Adapted from add_images written by Nadine Giese for use within the
        astrophysics lab course at Potsdam University.
        The source code may be modified, reused, and distributed as long as
        it retains a reference to the original author(s).

        Idea and further information:
        http://en.wikipedia.org/wiki/Phase_correlation

        Parameters
        ----------
        img_a       : `numpy.ndarray`
            Data of first image

        img_b       : `numpy.ndarray`
            Data of second image

        xMax        : `integer`
            Maximal allowed shift between the images in Pixel - X axis

        yMax        : `integer`
            Maximal allowed shift between the images in Pixel - Y axis

        debug       : `boolean`
            If True additional plots will be created
    '''

    lx = img_a.shape[1]
    ly = img_a.shape[0]


    imafft = np.fft.fft2(img_a)
    imbfft = np.fft.fft2(img_b)
    imbfftcc = np.conj(imbfft)
    fftcc = imafft*imbfftcc
    fftcc = fftcc/np.absolute(fftcc)
    #cc = np.fft.ifft2(fftcc)
    cc = np.fft.fft2(fftcc)
    cc[0,0] = 0.

    for i in range(xMax,lx-xMax):
        for j in range(0,ly):
            cc[j,i] = 0
    for i in range(0,lx):
        for j in range(yMax,ly-yMax):
            cc[j,i] = 0

    #   Debug plot showing the cc matrix
    if debug:
        plot.debug_plot_cc_matrix(img_b, cc)
        #from matplotlib import pyplot as plt
        #from astropy.visualization import simple_norm
        #norm = simple_norm(img_b, 'log', percent=99.)
        #plt.subplot(121),plt.imshow(img_b, norm=norm, cmap = 'gray')
        #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        #norm = simple_norm(np.absolute(cc), 'log', percent=99.)
        #plt.subplot(122),plt.imshow(np.absolute(cc), norm=norm,
                                    #cmap = 'gray')
        #plt.title('cc'), plt.xticks([]), plt.yticks([])
        #plt.show()

    #   Find the maximum in cc to identify the shift
    ind1, ind2 = np.unravel_index(cc.argmax(), cc.shape)

    #if ind2 > lx/2.:
        #ind2 = (ind2-1)-lx+1
    #else:
        #ind2 = ind2 - 1
    #if ind1 > ly/2.:
        #ind1 = (ind1-1)-ly+1
    #else:
        #ind1 = ind1 - 1
    if ind2 > lx/2.:
        ind2 = ind2-lx - 2
    else:
        ind2 = ind2 + 2
    if ind1 > ly/2.:
        ind1 = ind1-ly - 2
    else:
        ind1 = ind1 + 2

    return -ind1, -ind2


def calc_min_max_shifts(shifts, pythonFORMAT=False):
    '''
        Calculate shifts

        Parameters
        ----------
        shifts              : `numpy.ndarray`
            2D numpy array with the image shifts in X and Y direction

        pythonFORMAT        : `boolean`
            If True the python style of image ordering is used. If False the
            natural/fortran style of image ordering is use.
            Default is ``False``.

        Returns
        -------
        minx            : `float'
            Minimum shift in X direction

        maxx            : `float'
            Maximum shift in X direction

        miny            : `float'
            Minimum shift in Y direction

        maxy            : `float'
            Maximum shift in Y direction


    '''
    #   Distinguish between python format and natural format
    if pythonFORMAT:
        id_x = 1
        id_y = 0
    else:
        id_x = 0
        id_y = 1

    #   Maximum and minimum shifts
    minx = np.min(shifts[id_x,:])
    maxx = np.max(shifts[id_x,:])

    miny = np.min(shifts[id_y,:])
    maxy = np.max(shifts[id_y,:])

    return minx, maxx, miny, maxy


def calculate_image_shifts_core(img_ccd, reff_ccd, img_id, fname,
                                method='skimage'):
    '''
        Calculate image shifts using different methods

        Parameters
        ----------
        img_ccd             : `astropy.nddata.CCDData` object
            Image data

        reff_ccd            : `astropy.nddata.CCDData` object
            Image data of the reference image

        img_id              : `integer`
            Id of the image

        fname               : `string`
            Name of the image

        method              : `string`, optional
            Method to use for image alignment.
            Possibilities: 'own'     = own correlation routine based on
                                       phase correlation, applying fft to
                                       the images
                           'skimage' = phase correlation with skimage'
                           'aa'      = astroalign module
            Default is 'skimage'.

        Returns
        -------
        img_id          : `integer`
            Id of the image

        shift           : `tupel`
            Shifts of the image in X and Y direction

        flip            : `boolean`
            If `True` the image needs to be flipped
    '''
    #   Get reference image, reference mask, and corresponding file name
    reff_data = reff_ccd.data
    reff_mask = np.invert(reff_ccd.mask)
    reff_pier = reff_ccd.meta.get('PIERSIDE', 'EAST')

    #   Image and mask to compare with
    test_ccd = img_ccd
    test_data = img_ccd.data
    test_mask = np.invert(img_ccd.mask)

    #   Image pier side
    test_pier = img_ccd.meta.get('PIERSIDE', 'EAST')

    #   Flip if pier side changed
    if test_pier != reff_pier:
        test_ccd = ccdp.transform_image(img_ccd, np.flip, axis=(0,1))
        test_data = np.flip(test_data, axis=(0,1))
        test_mask = np.flip(test_mask, axis=(0,1))
        flip = True
    else:
        flip = False

    #   Calculate shifts
    if method == 'skimage':
        shift = phase_cross_correlation(
            reff_data,
            test_data,
            reference_mask=reff_mask,
            moving_mask=test_mask,
            )
    elif method == 'own':
        shift = correl_images(
            reff_data,
            test_data,
            1000,
            1000,
            False,
            )
    elif method == 'aa':
        #   Map with endianness symbols
        endian_map = {
            '>': 'big',
            '<': 'little',
            '=': sys.byteorder,
            '|': 'not applicable',
        }
        if endian_map[img_ccd.data.dtype.byteorder] != sys.byteorder:
            img_ccd.data = img_ccd.data.byteswap().newbyteorder()
            reff_ccd.data = reff_ccd.data.byteswap().newbyteorder()
            u_img = img_ccd.uncertainty.array.byteswap().newbyteorder()
            img_ccd.uncertainty = StdDevUncertainty(u_img)
            u_re = reff_ccd.uncertainty.array.byteswap().newbyteorder()
            reff_ccd.uncertainty = StdDevUncertainty(u_re)

        #   Determine transformation between the images
        try:
            p, (pos_img, pos_img_reff) = aa.find_transform(
                test_ccd,
                reff_ccd,
                detection_sigma=3.0,
                )

            shift = (p.translation[1], p.translation[0])
        except:
            shift = (0.,0.)
            terminal_output.print_terminal(
                img_id,
                string="WARNING: Offset determination for image {}"
                    " failed. Assume offset is 0.",
                style_name='WARNING',
                )
    else:
        #   This should not happen...
        raise RuntimeError(
            f'{style.bcolors.FAIL}Image correlation method {method} not '
            f'known\n {style.bcolors.ENDC}'
            )
    terminal_output.print_terminal(
        img_id,
        shift[1],
        shift[0],
        fname,
        string='\t{}\t{:+.1f}\t{:+.1f}\t{}',
        indent=0,
        )

    return img_id, shift, flip


def calculate_image_shifts(ifc, ref_img, comment, method='skimage'):
    '''
        Calculate image shifts

        Parameters
        ----------
        ifc        : `ccdproc.ImageFileCollection`
            Image file collection

        ref_img         : `integer`
            Number of the reference image

        comment         : `string`
            Information regarding for which images the shifts will be
            calculated

        method          : `string`, optional
            Method to use for image alignment.
            Possibilities: 'own'     = own correlation routine based on
                                       phase correlation, applying fft to
                                       the images
                           'skimage' = phase correlation with skimage'
                           'aa'      = astroalign module
            Default is 'skimage'.


        Returns
        -------
        shift           : `numpy.ndarray`
            Shifts of the images in X and Y direction

        flip            : `numpy.ndarray`
            Flip necessary to account for pier flips
    '''
    #   Number of images
    nfiles = len(ifc.files)

    #   Get reference image, reference mask, and corresponding file name
    reff_name = ifc.files[ref_img]
    reff_ccd  = CCDData.read(reff_name)

    #   Prepare an array for the shifts
    shift = np.zeros((2,nfiles))
    flip  = np.zeros(nfiles, dtype=bool)

    terminal_output.print_terminal(string=comment, indent=0)
    terminal_output.print_terminal(
        string='\tImage\tx\ty\tFilename',
        indent=0,
        )
    terminal_output.print_terminal(
        string='\t----------------------------------------',
        indent=0,
        )
    terminal_output.print_terminal(
        ref_img,
        0,
        0,
        reff_name.split('/')[-1],
        string='\t{}\t{:+.1f}\t{:+.1f}\t{}',
        indent=0,
        )

    #from ..analyze import aux as aux_ana
    ##   Initialize multiprocessing object
    #ncores=6
    #ncores=3
    #ncores=2
    #executor = aux_ana.Executor(ncores)

    #   Calculate image shifts
    for i, (img_ccd, fname) in enumerate(ifc.ccds(return_fname=True)):
        if i != ref_img:
            _, shift[:,i], flip[i] = calculate_image_shifts_core(
            #_, _, flip[i] = calculate_image_shifts_core(
                img_ccd,
                reff_ccd,
                i,
                fname,
                method=method,
                )
            #calculate_image_shifts_core(
                #img_ccd,
                #reff_ccd,
                #img_id,
                #method='skimage',
                #)
            #executor.schedule(
                #calculate_image_shifts_core,
                #args=(
                    #img_ccd,
                    #reff_ccd,
                    #i,
                    #fname,
                    #),
                #kwargs={
                    #'method':method,
                    #}
                #)

    ##   Close multiprocessing pool and wait until it finishes
    #executor.wait()

    ##   Exit if exceptions occurred
    #if executor.err is not None:
        #raise RuntimeError(
            #f'\n{style.bcolors.FAIL}Image offset determination failed '
            #f':({style.bcolors.ENDC}'
            #)

    ####
    ##   Sort multiprocessing results
    ##
    ##   Extract results
    #res = executor.res

    ##   Sort observation times and images & build dictionary for the
    ##   tables with the extraction results
    ##for j in range(0, nfiles):
    #for ref_id, shift_i, flip_i in res:
        #print(ref_id, shift_i, flip_i)
        #shift[:,ref_id] = shift_i
        #flip[ref_id] = flip_i

    terminal_output.print_terminal()

    return shift, flip


def shift_astroalign_method(reff_ccd, img_ccd):
    '''
        Calculate image shifts using the astroalign method

        Parameters
        ----------
        reff_ccd        : `astropy.nddata.CCDData` object
            Reference image

        img_ccd         : `astropy.nddata.CCDData` object
            Current image

        Returns
        -------
                        : `astropy.nddata.CCDData` object
            Aligned image
    '''
    #   Map with endianness symbols
    endian_map = {
        '>': 'big',
        '<': 'little',
        '=': sys.byteorder,
        '|': 'not applicable',
    }
    if endian_map[img_ccd.data.dtype.byteorder] != sys.byteorder:
        img_ccd.data   = img_ccd.data.byteswap().newbyteorder()
        reff_ccd.data  = reff_ccd.data.byteswap().newbyteorder()
        img_ccd.uncertainty = StdDevUncertainty(
            img_ccd.uncertainty.array.byteswap().newbyteorder()
            )
        reff_ccd.uncertainty = StdDevUncertainty(
            reff_ccd.uncertainty.array.byteswap().newbyteorder()
            )

    #   Determine transformation between the images
    p, (pos_img, pos_img_reff) = aa.find_transform(
        img_ccd,
        reff_ccd,
        detection_sigma=3.0,
        )

    #   Transform image data
    img_data, footprint = aa.apply_transform(
        p,
        img_ccd,
        reff_ccd,
        propagate_mask=True,
        )

    #   Transform uncertainty array
    img_uncert, footprint_uncert = aa.apply_transform(
        p,
        img_ccd.uncertainty.array,
        reff_ccd.uncertainty.array,
        )

    #   Build new CCDData object
    return CCDData(
        img_data,
        mask=footprint,
        meta=img_ccd.meta,
        unit=img_ccd.unit,
        uncertainty=StdDevUncertainty(img_uncert),
        )


def shift_optical_flow_method(reff_ccd, img_ccd):
    '''
        Calculate image shifts using the optical flow method

        Parameters
        ----------
        reff_ccd        : `astropy.nddata.CCDData` object
            Reference image

        img_ccd         : `astropy.nddata.CCDData` object
            Current image

        Returns
        -------
                        : `astropy.nddata.CCDData` object
            Aligned image
    '''
    #   Prepare data, mask, and uncertainty arrays
    test_data = img_ccd.data
    test_mask = img_ccd.mask
    test_unc = img_ccd.uncertainty.array

    #   Compute optical flow
    v, u = optical_flow_tvl1(reff_ccd.data, test_data)

    #   Prepare grid for flow map
    nr, nc = reff_ccd.data.shape
    row_coords, col_coords = np.meshgrid(
        np.arange(nr),
        np.arange(nc),
        indexing='ij',
        )

    #   Registrate image data, mask, and uncertainty
    img_out_data = warp(
        test_data,
        np.array([row_coords + v, col_coords + u]),
        mode='edge',
        )
    img_out_mask = warp(
        test_mask,
        np.array([row_coords + v, col_coords + u]),
        mode='edge',
        )
    img_out_unc = warp(
        test_unc,
        np.array([row_coords + v, col_coords + u]),
        mode='edge',
        )

    #   Build new CCDData object
    return CCDData(
        img_out_data,
        mask=test_mask,
        meta=img_ccd.meta,
        unit=img_ccd.unit,
        uncertainty=StdDevUncertainty(img_out_unc),
        )


def make_index_from_shifts(shift, i):
    '''
        Calculate image index positions from image shifts

        Parameters
        ----------
        shift           : `numpy.ndarray`
            Shifts of all images in X and Y direction

        i               : `integer`
            ID of the current image

        Returns
        -------
        xs, xe, ys, ye  : `float`
            Start/End pixel index in X direction `xs`/`xe` and start/end
            pixel index in Y direction.
    '''
    #   Calculate maximum and minimum shifts
    minx, maxx, miny, maxy = calc_min_max_shifts(shift, pythonFORMAT=True)

    #   Calculate indexes from image shifts
    if minx >= 0 and maxx >= 0:
        xs = maxx - shift[1, i]
        xe = shift[1, i] * -1
    elif minx < 0 and maxx < 0:
        xs = shift[1, i] * -1
        xe = maxx - shift[1, i]
    else:
        xs = maxx - shift[1, i]
        xe = minx - shift[1, i]

    if miny >= 0 and maxy >= 0:
        ys = maxy - shift[0, i]
        ye = shift[0, i] * -1
    elif miny < 0 and maxy < 0:
        ys = shift[0, i] * -1
        ye = maxy - shift[0, i]
    else:
        ys = maxy - shift[0, i]
        ye = miny - shift[0, i]

    return xs, xe, ys, ye


def trim_core(img, i, nfiles, shift, method='skimage', verbose=False):
    '''
        Trim image 'i' based on a shift compared to a reference image

        Parameters
        ----------
        img             : `astropy.nddata.CCDData`
            Image

        i               : `integer`
            Number of the image in the sequence

        nfiles          : `integer`
            Number of all images

        shift           : `numpy.ndarray`
            Shift of this specific image in X and Y direction

        method          : `string`, optional
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

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    '''
    if verbose:
        #   Write status to console
        sys.stdout.write("\r\tApply shift to image %i/%i\n" % (i+1, nfiles))
        sys.stdout.flush()


    if method in ['own', 'skimage']:
        #   Ensure full pixel shifts
        if not issubclass(type(shift[0,0]), np.integer):
            shift = shift.astype('int')

        #   Calculate indexes from image shifts
        xs, xe, ys, ye = make_index_from_shifts(shift, i)
    elif method == 'aa':
        #   Calculate maximum and minimum shifts
        minx, maxx, miny, maxy = calc_min_max_shifts(
            shift,
            pythonFORMAT=True,
            )

        #   Shift image on sub pixel basis
        img = ccdp.transform_image(
            img,
            shift_scipy,
            shift=shift[:,i],
            order=1,
            )

        #   Set trim margins
        if minx > 0:
            xs = int(math.ceil(maxx))
            xe = 0
        elif minx < 0 and maxx < 0:
            xs = 0
            xe = int(math.ceil(np.abs(minx))) * -1
        else:
            xs = int(math.ceil(maxx))
            xe = int(math.ceil(np.abs(minx))) * -1

        if miny > 0:
            ys = int(math.ceil(maxy))
            ye = 0
        elif miny < 0 and maxy < 0:
            ys = 0
            ye = int(math.ceil(np.abs(miny))) * -1
        else:
            ys = int(math.ceil(maxy))
            ye = int(math.ceil(np.abs(miny))) * -1
    else:
        raise ValueError(
            f'{style.bcolors.FAIL}Shift method not known. Expected: '
            f'"pixel" or "sub_pixel", but got '
            f'"{method}" {style.bcolors.ENDC}'
            )


    #   Trim the image
    return ccdp.trim_image(img[ys:img.shape[0]+ye, xs:img.shape[1]+xe])


def prepare_reduction(outdir, bias, darks, flats, imgs, rawfiles, temp_dir,
                      img_type=None):
    '''
        Prepare directories and files for the reduction procedure

        Parameters
        ----------
        outdir          : `list` of `string`s
            Path to the directory where the master files should be saved to

        bias            : `list` of `string`s
            Path to the bias or '?'

        darks           : `list` of `string`s
            Path to the darks or '?'

        flats           : `list` of `string`s
            Path to the flats or '?'

        imgs            : `list` of `string`s
            Path to the science images or '?'

        rawfiles        : 'string`
            Path to all raw images or '?', if bias, darks, flats, and imgs
            are provided.

        temp_dir        : ``tempfile.TemporaryDirectory`
            Temporary directory to store the symbolic links to the images

        img_type        : `dict` of `string`, optional
            Image type to select. Possibilities: bias, dark, flat, light
            Default is ``None``.

        Returns
        -------
        rawfiles_path   : 'string`
            Points to the path with the raw files. Either the temporary
            directory or the already provided 'rawfiles' directory.
    '''
    ###
    #   Check directories
    #
    terminal_output.print_terminal(
        string="Check if directories exists...",
        )
    checks.check_out(outdir)
    if rawfiles == '?':
        checks.check_path(darks)
        checks.check_path(flats)
        checks.check_path(imgs)
        if bias != '?':
            checks.check_path(bias)

        #   Find sub directories
        darks = checks.list_subdir(darks)
        flats = checks.list_subdir(flats)
        imgs  = checks.list_subdir(imgs)
        if bias != '?':
            bias  = checks.list_subdir(bias)

    else:
        checks.check_path(rawfiles)


    ###
    #   Check consistency between images and fits header keywords
    #
    if rawfiles == '?':
        terminal_output.print_terminal(
            string="Check header keywords for consistency...",
            )
        if bias != '?':
            bias_new = []
            for path in bias:
                if img_type is not None:
                    bias_new.append(
                        check_filter_keywords(path, img_type['bias'])
                        )
                else:
                    bias_new.append(check_filter_keywords(path, 'bias'))
            bias = bias_new

        darks_new = []
        for path in darks:
            if img_type is not None:
                darks_new.append(
                    check_filter_keywords(path, img_type['dark'])
                    )
            else:
                darks_new.append(check_filter_keywords(path, 'dark'))
        darks = darks_new

        flats_new = []
        for path in flats:
            if img_type is not None:
                flats_new.append(
                    check_filter_keywords(path, img_type['flat'])
                    )
            else:
                flats_new.append(check_filter_keywords(path, 'flat'))
        flats =  flats_new

        imgs_new = []
        for path in imgs:
            if img_type is not None:
                imgs_new.append(
                    check_filter_keywords(path, img_type['light'])
                    )
            else:
                imgs_new.append(check_filter_keywords(path, 'light'))
        imgs = imgs_new


    ###
    #   Prepare temporary directory, if individual
    #   directories were defined above
    #
    if rawfiles == '?':
        #   Combine directories
        rawfiles = darks+flats+imgs
        if bias != '?':
            rawfiles = rawfiles+bias

        #   Link all files to the temporary directory
        make_symbolic_links(rawfiles, temp_dir)

        rawfiles_path = temp_dir.name
    else:
        rawfiles_path = checks.list_subdir(rawfiles)

        if len(rawfiles_path) == 1:
            rawfiles_path = rawfiles
        elif len(rawfiles_path) > 1:
            #   Link all files to the temporary directory
            make_symbolic_links(rawfiles_path, temp_dir)

            rawfiles_path = temp_dir.name
        else:
            #   This should not happen...
            raise RuntimeError(
                f'{style.bcolors.FAIL}Raw file path could not be '
                f'decoded...\n {style.bcolors.ENDC}'
                )

    return rawfiles_path


def profiles(data):
    '''
        Get star profiles

        Parameters
        ----------
        data        : `numpy.ndarray`
            Image (square) extracted around the star

        Returns
        -------
        x           : `numpy.ndarray`
            Profile in X direction

        y           : `numpy.ndarray`
            Profile in Y direction
    '''
    #   Get image shape
    shape = data.shape

    #   Get central rows
    if shape[0] % 2 == 0:
        ypix = shape[0] / 2
    else:
        ypix = (shape[0] - 1) / 2 + 1

    if shape[1] % 2 == 0:
        xpix = shape[1] / 2
    else:
        xpix = (shape[1] - 1) / 2 + 1

    #   Get profiles
    x = np.take(data, xpix, axis=1)
    y = np.take(data, ypix, axis=0)

    return x, y


def interpolate_width(axis):
    '''
        Find FWHM by means of interpolation on a stellar profile

        Idea: https://stackoverflow.com/questions/52320873/computing-the-fwhm-of-a-star-profile

        Parameters
        ----------
        axis        : `numpy.ndarray`
            Stellar profile along a specific axis

        Returns     : `float`
            FWHM of the profile
    '''
    #   Prepare interpolation
    half_max = 0.5 * np.max(axis)
    x = np.linspace(0, len(axis), len(axis))

    #   Do the interpolation
    spline = UnivariateSpline(x, axis-half_max, s=0)
    r1, r2 = spline.roots()

    return r2-r1


def estimate_fwhm(path, outdir, image_type, plot_subplots=False,
                  indent='      '):
    '''
        Combine images

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        image_type      : `string`
            Header keyword characterizing the image type for which the
            shifts shall be determined

        plot_subplots   : `boolean`, optional
            Plot subplots around the stars used to estimate the FWHM
            Default is ``False``.

        indent          : `string`
            Indentation for the console output lines.
            Default is ``      ``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   New image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)

    #   Determine filter
    filters = set(h['filter'] for h in ifc.headers(imagetyp=image_type))

    #   Combine images for the individual filters
    for filt in filters:
        #   Select images to combine
        ifc_mod = ifc.filter(
            imagetyp=image_type,
            filter=filt
            )

        #   List for the median FWHM for individual images
        img_fwhm = []

        #   Loop over images
        for img_ccd, fname in ifc_mod.ccds(return_fname=True):
            #   Get background
            mean, median, std = sigma_clipped_stats(img_ccd.data, sigma=3.0)

            #   Find stars
            daofind = DAOStarFinder(fwhm=3.0, threshold=10.*std)
            sources = daofind(img_ccd.data - median)

            #   Exclude objects close the image edges -> create new table
            size = 25
            hsize = (size - 1) / 2

            x = sources['xcentroid']
            y = sources['ycentroid']
            flux = sources['flux']

            mask = ((x > hsize) & (x < (img_ccd.data.shape[1] -1 - hsize)) &
                    (y > hsize) & (y < (img_ccd.data.shape[0] -1 - hsize)))

            stars_tbl = Table()
            stars_tbl['x'] = x[mask]
            stars_tbl['y'] = y[mask]
            stars_tbl['y'] = y[mask]
            stars_tbl['flux'] = flux[mask]


            #   Exclude the brightest stars that are often saturated
            #   (rm the brightest 1% of all stars)

            #   Sort list with star positions according to flux
            tbl_sort = stars_tbl.group_by('flux')

            # Determine the 99 percentile
            p99 = np.percentile(tbl_sort['flux'], 99)

            #   Determine the position of the 99 percentile in the position
            #   list
            id_p99 = np.argmin(np.absolute(tbl_sort['flux']-p99))

            #   Use 25 stars to estimate the FWHM
            min_stars = 25

            #   Check if enough stars were detected
            if id_p99-min_stars<1:
                min_stars = 1

            #   Resize table -> limit it to the suitable stars
            stars_tbl = tbl_sort[:][id_p99-min_stars:id_p99]

            #   Extract cutouts
            stars = extract_stars(img_ccd, stars_tbl, size=25)

            #   Plot subplots
            #plot_subplots = True
            if plot_subplots:
                plot.subplots_stars_fwhm_estimate(
                    outdir,
                    len(stars_tbl),
                    stars,
                    filt,
                    base_aux.get_basename(fname),
                    )

            ###
            #   Loop over all stars and determine the FWHM
            #
            fwhm_x_list = []
            fwhm_y_list = []

            for i in range(len(stars_tbl)):   # can be optimized -> loop stars
                #   Get star profile
                horizontal, vertical = profiles(stars[i])
                #   Try to find FWHM, skip if this is not successful
                try:
                    fwhm_x = interpolate_width(horizontal)
                    fwhm_y = interpolate_width(vertical)

                    fwhm_x_list.append(fwhm_x)
                    fwhm_y_list.append(fwhm_y)
                except:
                    pass

            #   Get median of the FWHMs
            median_fwhm_x = np.median(fwhm_x_list)
            median_fwhm_y = np.median(fwhm_y_list)

            #   Average the FWHM from both directions
            mean_fwhm = np.mean([median_fwhm_x, median_fwhm_y])

            print(fname, mean_fwhm)
            img_fwhm.append(mean_fwhm)

        terminal_output.print_terminal(
            filt,
            np.median(img_fwhm),
            string="FWHM (median) of the stars in Filter {}: {}",
            indent=indent,
            )


def check_master_on_disk(path, image_type, dark_times, filters,
                         bias_bool):
    '''
        Check if master files are already prepared

        Parameters
        ----------
        path            : `string`
            Path to the images

        image_type      : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light

        dark_times      : `list`
            Exposure times of the raw dark images

        filters         : `list`
            Filter that have been used

        bias_bool       : `boolean`
            If True bias will be checked as well.

        Returns
        -------
        check_master    : 'boolean`
            Is True if all required master files are already prepared.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)

    #   Get image collection for the reduced files
    ifc_reduced = ccdp.ImageFileCollection(file_path)

    if not ifc_reduced.files:
        return False

    #   Load combined bias, darks, and flats in dictionary for easy
    #   access
    if bias_bool:
        ###
        #   Get master bias
        #
        type_bias = get_image_type(
            ifc_reduced,
            image_type,
            image_class='bias',
            )

        #   Return if no flats found
        if not type_bias:
            return False

        #   Prepare dict with master biases
        combined_bias = ifc_reduced.files_filtered(
            imagetyp=type_bias,
            combined=True,
            include_path=True,
        )


    ###
    #   Get master dark
    #
    type_dark = get_image_type(ifc_reduced, image_type, image_class='dark')

    #   Return if no flats found
    if not type_dark:
        return False

    #   Prepare dict with master darks
    combined_darks = {
        ccd.header['exptime']: ccd for ccd in ifc_reduced.ccds(
                                                imagetyp=type_dark,
                                                combined=True,
                                                )
        }

    #   Check if master darks exists for all all exposure times
    check_master = True
    for key in combined_darks.keys():
        if key not in dark_times:
            check_master = False

    ###
    #   Get master flats
    #
    type_flat = get_image_type(ifc_reduced, image_type, image_class='flat')

    #   Return if no flats found
    if not type_flat:
        return False

    #   Prepare dict with master flats
    combined_flats = {
        ccd.header['filter']: ccd for ccd in ifc_reduced.ccds(
                                                imagetyp=type_flat,
                                                combined=True,
                                                )
        }

    #   Check if master flats exists for all all filters
    for key in combined_flats.keys():
        if key not in filters:
            check_master = False

    return check_master


def flip_img(ifc, out_path):
    '''
        Flip images in X and Y direction

        Parameters
        ----------
        ifc                 : `ccdproc.ImageFileCollection`
            Image file collection

        out_path            : `pathlib.Path`
            Path to save the individual images

        Returns
        -------
                            : `ccdproc.ImageFileCollection`
            Image file collection pointing to the flipped images
    '''
    terminal_output.print_terminal(
        indent=2,
        string="Flip images",
        )

    #   Check directory
    checks.check_out(out_path)

    for img, file_name in ifc.ccds(
        ccd_kwargs={'unit': 'adu'},
        return_fname=True,
        ):

        #   Flip image
        img_fliped = ccdp.transform_image(img, np.flip, axis = (0,1))

        #   Save the result
        out_flipped = out_path / 'flipped'
        checks.check_out(out_flipped)
        img_fliped.write(out_flipped / file_name, overwrite=True)

    #   Replace new image file collection
    return ccdp.ImageFileCollection(out_flipped)


def bin_img(ifc, out_path, bin_value):
    '''
        Bin images in X and Y direction

        Parameters
        ----------
        ifc                 : `ccdproc.ImageFileCollection`
            Image file collection

        out_path            : `pathlib.Path`
            Path to save the individual images

        bin_value           : `integer`
            Number of pixel that the image should be binned in X and Y
            direction.

        Returns
        -------
                            : `ccdproc.ImageFileCollection`
            Image file collection pointing to the binned images
    '''
    terminal_output.print_terminal(
        indent=2,
        string="Bin images",
        )

    #   Check directory
    checks.check_out(out_path)

    for img, file_name in ifc.ccds(
        ccd_kwargs={'unit': 'adu'},
        return_fname=True,
        ):

        #   Bin image
        img_binned = ccdp.block_average(img, bin_value)

        #   Correct Header
        img_binned.meta['XBINNING'] = bin_value
        img_binned.meta['YBINNING'] = bin_value
        #img_binned.meta['EXPTIME'] = img.meta['EXPTIME']
        #img_binned.meta['EXPOSURE'] = img.meta['EXPOSURE']
        img_binned.meta['INFO_0'] = 'Software binned using numpy mean function'
        img_binned.meta['INFO_1'] = '    Exposure time scaled accordingly'

        #   Save the result
        out_binned = out_path / 'binned'
        checks.check_out(out_binned)
        img_binned.write(out_binned / file_name, overwrite=True)

    #   Replace new image file collection
    return ccdp.ImageFileCollection(out_binned)


def trim_img(ifc, out_path, xs=100, xe=100, ys=100, ye=100):
    '''
        Trim images in X and Y direction

        Parameters
        ----------
        ifc                 : `ccdproc.ImageFileCollection`
            Image file collection

        out_path            : `pathlib.Path`
            Path to save the individual images

        xs                  : `integer`
            Number of Pixel to be removed from the start of the image in
            X direction.

        xe                  : `integer`
            Number of Pixel to be removed from the end of the image in
            X direction.

        ys                  : `integer`
            Number of Pixel to be removed from the start of the image in
            Y direction.

        ye                  : `integer`
            Number of Pixel to be removed from the end of the image in
            Y direction.

        Returns
        -------
                            : `ccdproc.ImageFileCollection`
            Image file collection pointing to the trimmed images
    '''
    terminal_output.print_terminal(
        indent=2,
        string="Trim images",
        )

    #   Check directory
    checks.check_out(out_path)

    for img, file_name in ifc.ccds(
        ccd_kwargs={'unit': 'adu'},
        return_fname=True,
        ):

        #   Trim image
        img_trimmed = ccdp.trim_image(img[ys:-ye, xs:-xe])

        #   Save the result
        out_trimmed = out_path / 'trimmed'
        checks.check_out(out_trimmed)
        img_trimmed.write(out_trimmed / file_name, overwrite=True)

    #   Return new image file collection
    return ccdp.ImageFileCollection(out_trimmed)


def find_wcs(input_dir, output_dir, ref_id=0, force_wcs_determ=False,
             method='astrometry', x=None, y=None, indent=2):
    '''
        Determine the WCS of the reference image and add the WCS to all
        images in the input directory. The latter is to save computing time.
        It is assumed that the images are already aligned and trimmed to
        the same filed of view.

        Parameters
        ----------
        input_dir           : `pathlib.Path` or string
            Path to the input directory.

        output_dir          : `pathlib.Path` or string
            Path to the output directory.

        ref_id              : `integer', optional
            ID of the reference image.
            Default is ``0``.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        method              : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        x, y                : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        w                   : `astropy.wcs.WCS`
            WCS information
    '''
    ###
    #   Prepare variables
    #
    #   Check directories
    file_path = checks.check_pathlib_Path(input_dir)
    checks.check_out(output_dir)

    #   Set up image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)

    #   Filter priority list:
    #   Give highest priority to the filter with the highest probability of
    #   detecting a large number of stars
    filt_list = ['I', 'R', 'V', 'B', 'U']

    #   Filter ifc according to filter list
    for filt in filt_list:
        ifc_filtered = ifc.filter(filter=filt)

        #   Exit loop when images are found for the current filter
        if ifc_filtered.files:
            break

    #   Check again if ifc is empty. If True use first filter from
    #   the ifc filter list.
    if not ifc_filtered.files:
        #   Determine ifc filter
        filters = set(h['filter'] for h in ifc.headers())
        filt = list(filters)[0]

        ifc_filtered = ifc.filter(filter=filt)


    ###
    #   Get reference image
    #
    reff_name = ifc_filtered.files[ref_id]
    reff_ccd = CCDData.read(reff_name)

    reff_img = base_aux.image(ref_id, filt, 'target', reff_name, output_dir)

    base_aux.cal_fov(reff_img)

    #   Test if the image contains already a WCS
    cal_wcs = base_aux.check_wcs_exists(reff_img)


    ###
    #   Determine WCS
    #
    if not cal_wcs or force_wcs_determ:
        w = find_wcs_distinguish(reff_img, method=method, indent=indent)

        ###
        #   Add WCS to images
        #
        if w is not None:
            for img, file_name in ifc.ccds(return_fname=True):

                img.wcs = w

                #   Save the image
                img.write(output_dir / file_name, overwrite=True)


def find_wcs_all_imgs(input_dir, output_dir, force_wcs_determ=False,
                      method='astrometry', x=None, y=None, indent=2):
    '''
        Determine the WCS of each image individually.

        Parameters
        ----------
        input_dir           : `pathlib.Path` or string
            Path to the input directory.

        output_dir          : `pathlib.Path` or string
            Path to the output directory.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        method              : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        x, y                : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    '''
    ###
    #   Prepare variables
    #
    #   Check directories
    file_path = checks.check_pathlib_Path(input_dir)
    checks.check_out(output_dir)

    #   Set up image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)


    ###
    #   Derive WCS
    #
    for i, (img_ccd, file_name) in enumerate(ifc.ccds(return_fname=True)):
        #   Prepare image object
        img = base_aux.image(
            i,
            'filt',
            'target',
            file_path / file_name,
            output_dir,
            )
        base_aux.cal_fov(img, verbose=False)

        #   Test if the image contains already a WCS
        cal_wcs = base_aux.check_wcs_exists(img)

        if not cal_wcs or force_wcs_determ:
            w = find_wcs_distinguish(img, method=method, indent=indent)

            #   Add WCS to image (not necessary for ASTAP method)
            if method in ['astrometry', 'twirl']:
                img_ccd.wcs = w

                #   Save the image
                img_ccd.write(output_dir / file_name, overwrite=True)


def find_wcs_distinguish(img, method='astrometry', x=None, y=None,
                         indent=2):
    '''
        Branch between different WCS methods

        Parameters
        ----------
        img                 : `image.class`
            Image class with all image specific properties

        method              : `string`, optional
            Method to use for the WCS determination
            Options: 'astrometry', 'astap', or 'twirl'
            Default is ``astrometry``.

        x, y            : `numpy.ndarray`, optional
            Pixel coordinates of the objects
            Default is ``None``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        w                   : `astropy.wcs.WCS`
            WCS information
    '''
    #   astrometry.net:
    if method == 'astrometry':
        try:
            w = base_aux.find_wcs_astrometry(
                img,
                wcs_dir='/tmp/',
                indent=indent,
                )
        except:
            terminal_output.print_terminal(
                indent=indent,
                string="No WCS solution found :(\n",
                style_name='WARNING',
                )
            w = None

    #   ASTAP program
    elif method == 'astap':
        try:
            w = base_aux.find_wcs_astap(
                img,
                indent=indent,
                )
            terminal_output.print_terminal()
        except:
            terminal_output.print_terminal(
                indent=indent,
                string="No WCS solution found :(\n",
                style_name='WARNING',
                )
            w = None

    #   twirl libary
    elif method == 'twirl':
        try:
            if x is None or y is None:
                raise RuntimeError(
                    f'{style.bcolors.FAIL} \nException in find_wcs(): \n'
                    f"'x' or 'y' is None -> Exit {style.bcolors.ENDC}"
                    )
            w = base_aux.find_wcs_twirl(img, x, y, indent=indent)
        except:
            terminal_output.print_terminal(
                indent=indent,
                string="No WCS solution found :(\n",
                style_name='WARNING',
                )
            w = None

    #   Raise exception
    else:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nException in find_wcs(): '"
            f"\nWCS method not known -> Supplied method was {method}"
            f"{style.bcolors.ENDC}"
            )

    return w
