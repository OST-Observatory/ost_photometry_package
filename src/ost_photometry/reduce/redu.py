############################################################################
####                            Libraries                               ####
############################################################################

import sys

import shutil

from pytimedinput import timedInput

from pathlib import Path

import numpy as np

from scipy.ndimage import median_filter

import ccdproc as ccdp

from astropy.stats import mad_std
from astropy.nddata import CCDData, StdDevUncertainty
import astropy.units as u

import astroalign as aa

from . import aux, plot

from .. import checks, style, terminal_output, calibration_data

from .. import aux as aux_general

############################################################################
####                        Routines & definitions                      ####
############################################################################

def reduce_main(path, outdir, img_type=None, gain=None, readnoise=None,
                dr=None, cosmics=True, mask_cosmics=False, satlevel=None,
                objlim=5., sigclip=4.0, scale_expo=True, ref_img=0,
                bias_enforce=False, verbose=False, addmask=True,
                shift_method='skimage', stack=True, estimate_fwhm=False,
                shift_all=False, tolerance=0.5, stack_method='average',
                target=None, find_wcs=True, wcs_method='astrometry',
                wcs_all=False, rm_outliers_shift=True, filter_window_shift=8,
                threshold_shift=10., debug=False):
    '''
        Main reduction routine: Creates master images for bias, darks,
                                flats, reduces the science images and trims
                                them to the same filed of view.


        Reduce the science images

        Parameters
        ----------
        path                : `string`
            Path to the images

        outdir              : `string`
            Path to the directory where the master files should be stored

        img_type            : `dictionary` of `string`, optional
            Image types of the images. Possibilities: bias, dark, flat,
            light
            Default is ``None``.

        gain                : `float` or `None`, optional
            The gain (e-/adu) of the camera chip. If set to `None` the gain
            will be extracted from the FITS header.
            Default is ``None``.

        readnoise           : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``None``.

        dr                  : `dictionary(float:float)`, optional
            Temperature dependent dark rate in e-/pix/s:
            key = temperature, value = dark rate
            Default is ``None``.

        cosmics             : `boolean`, optional
            If True cosmics rays will be removed.
            Default is ``True``.

        mask_cosmics        : `boolean`, optional
            If True cosmics will ''only'' be masked. If False the
            cosmics will be removed from the input image and the mask will
            be added.
            Default is ``False``.

        satlevel            : `float`, optional
            Saturation limit of the camera chip.
            Default is ``None``.

        objlim              : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        sigclip             : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        scale_expo          : `boolean`, optional
            If True the image will be scaled with the exposure time.
            Default is ``True``.

        ref_img             : `integer`, optional
            ID of the image that should be used as a reference
            Default is ``0``.

        bias_enforce        : 'boolean', optional
            If True the usage of bias frames during the reduction is
            enforced if possible.
            Default is ``False``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        addmask             : `boolean`, optional
            If True add hot and bad pixel mask to the reduced science
            images.
            Default is ``True``.

        shift_method        :`string`, optional
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

        stack               : `boolean`, optional
            If True the individual images of each filter will be stacked and
            those images will be aligned to each other.
            Default is ``True``.

        estimate_fwhm       : `boolean`, optional
            If True the FWHM of each image will be estimated.
            Default is ``False``.

        shift_all           : `boolean`, optional
            If False shifts between images are only calculated for images of
            the same filter. If True shifts between all images are
            estimated.
            Default is ``False``.

        tolerance           : `float`, optional
            Tolerance between science and dark exposure times in s.
            Default is ``0.5``s.

        stack_method        : `string`, optional
            Method used for combining the images.
            Possibilities: ``median`` or ``average`` or ``sum``
            Default is ``average`.

        target              : `string` or ``None``, optional
            Name of the target. Used for file selection.
            Default is ``None``.

        find_wcs            : `boolean`, optional
            If `True` the WCS will be determined for the images.
            Default is ``True``.

        wcs_method          :   `string`, optional
            Method to use for WCS determination.
            Possibilities are 'astrometry', 'astap', and 'twirl'
            Default is ``astrometry``.

        wcs_all             : `boolean`, optional
            If `True` the WCS will be calculated for each image
            individually.
            Default is ``False``.

        rm_outliers_shift   : `boolean`, optional
            If True outliers in the image shifts will be detected and removed.
            Default is ``True``.

        filter_window_shift : `integer`, optional
            Width of the median filter window
            Default is ``8``.

        threshold_shift     : `float` or `integer`, optional
            Difference above the running median above an element is
            considered to be an outlier.
            Default is ``10.``.

        debug               : `boolean`, optional
            If `True` the intermediate files of the data reduction will not
            be removed.
            Default is ``False``.
    '''

    ###
    #   Prepare reduction
    #
    #   Sanitize the provided paths
    file_path = Path(path)
    out_path  = Path(outdir)

    #   Get image file collection
    ifc = ccdp.ImageFileCollection(file_path)

    #   Get image types and check imaging software
    imaging_soft = aux.get_imaging_soft(ifc)
    if len(imaging_soft) > 1:
        terminal_output.print_terminal(
            imaging_soft,
            string="Images are taken with multiple software versions: {}. "\
                "The pipeline cannot account for that, but will try anyway...",
            indent=2,
            style_name='WARNING',
            )
    if img_type is None:
        img_type = calibration_data.get_img_types(list(imaging_soft)[0])

    #   Except if image collection is empty
    #if len(ifc.files) == 0:
    if not ifc.files:
        raise RuntimeError(
            f'{style.bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.bcolors.ENDC}'
            )

    #   Get image types (set allows to return only unique values)
    type_imgs = set(ifc.summary['imagetyp'])

    #   Check exposure times:   Successful if dark frames with ~ the same
    #                           exposure time are available all flat and
    #                           science
    #   Dark times
    dark_times = aux.get_exposure_times(ifc, img_type['dark'])

    #   Flat times
    flat_times = aux.get_exposure_times(ifc, img_type['flat'])

    #   Science times
    sciece_times = aux.get_exposure_times(ifc, img_type['light'])

    #   Check if bias frames are available
    bias_true = np.any(
        [True if t in type_imgs else False for t in img_type['bias']]
        )

    #   Check flats
    scale_necessary = aux.check_exposure_times(
        'flats',
        flat_times,
        dark_times,
        bias_true,
        tolerance=tolerance,
        )

    #   Check science exposures
    scale_necessary = scale_necessary | aux.check_exposure_times(
        'science',
        sciece_times,
        dark_times,
        bias_true,
        tolerance=tolerance,
        )


    ###
    #   Check instrument
    #
    instruments = aux.get_instruments(ifc)
    if len(instruments) > 1:
        terminal_output.print_terminal(
            instruments,
            string="Images are taken with several instruments: {}. "\
                "The pipeline cannot account for that, but will try anyway...",
            indent=2,
            style_name='WARNING',
            )


    ###
    #   Get camera specific parameters
    #
    if (readnoise is None or gain is None or dr is None or satlevel is None):
        camera_info = calibration_data.camera_info(list(instruments)[0])
        if readnoise == None:
            readnoise = camera_info[0]
        if gain == None:
            gain = camera_info[1]
        if dr == None:
            dr = camera_info[2]
        if satlevel == None:
            satlevel = camera_info[3]


    ###
    #   Check master files on disk
    #
    #   Get all filter
    filters = set(
        ifc.summary['filter'][np.invert(ifc.summary['filter'].mask)]
        )

    #   Check is master files already exist
    check_master = aux.check_master_on_disk(
        out_path,
        img_type,
        dark_times,
        filters,
        scale_necessary,
    )

    mk_new_master = True
    if check_master:
        user_input, timedOut = timedInput(
            f"{style.bcolors.OKBLUE}   Master files are already calculated."
            f" Should these files be used? {style.bcolors.ENDC}",
            timeout=30,
            )
        if timedOut:
            user_input = 'n'

        if user_input in ['y', 'yes']:
            mk_new_master = False

    #   Set master boolean for bias subtraction
    rmbias = True if scale_necessary or bias_enforce else False

    if mk_new_master:
        ###
        #   Reduce bias
        #
        if rmbias:
            terminal_output.print_terminal(
                string="Create master bias...",
                indent=1,
                )
            master_bias(file_path, out_path, img_type)


        ###
        #   Master dark and master flat darks
        #
        terminal_output.print_terminal(
            string="Create master darks...",
            indent=1,
            )

        if rmbias:
            #   Reduce dark frames and apply bias subtraction
            reduce_dark(
                file_path,
                out_path,
                img_type,
                gain=gain,
                readnoise=readnoise,
                )

            #   Set dark path
            dark_path = Path(out_path / 'dark')
        else:
            dark_path = file_path

        #   Create master dark
        master_dark(
            file_path,
            out_path,
            img_type,
            gain=gain,
            readnoise=readnoise,
            dr=dr,
            verbose=verbose,
            debug=debug,
            )


        ###
        #   Master flat
        #
        terminal_output.print_terminal(
            string="Create master flat...",
            indent=1,
            )

        #   Reduce flats
        reduce_flat(
            file_path,
            out_path,
            img_type,
            gain=gain,
            readnoise=readnoise,
            rmbias=rmbias,
            tolerance=tolerance,
            debug=debug,
            )

        #   Create master flat
        master_flat(
            Path(out_path / 'flat'),
            out_path,
            img_type,
            verbose=verbose,
            )


    ###
    #   Image reduction & stacking (calculation of image shifts, etc. )
    #
    terminal_output.print_terminal(
        string="Reduce science images...",
        indent=1,
        )

    reduce_light(
        file_path,
        out_path,
        img_type,
        cosmics=cosmics,
        mask_cosmics=mask_cosmics,
        gain=gain,
        readnoise=readnoise,
        objlim=objlim,
        sigclip=sigclip,
        satlevel=satlevel,
        rmbias=rmbias,
        verbose=verbose,
        addmask=addmask,
        tolerance=tolerance,
        target=target,
        )


    ###
    #   Calculate and apply image shifts for individual filters or all
    #   images
    #
    terminal_output.print_terminal(
        string="Trim images to the same FOV...",
        indent=1,
        )
    if shift_all:
        shift_img_all(
            out_path / 'light',
            out_path,
            img_type['light'],
            ref_img=ref_img,
            shift_method=shift_method,
            rm_outliers=rm_outliers_shift,
            filter_window=filter_window_shift,
            threshold=threshold_shift,
            verbose=verbose,
            debug=debug,
            )
    else:
        shift_img(
            out_path / 'light',
            out_path,
            img_type['light'],
            ref_img=ref_img,
            shift_method=shift_method,
            rm_outliers=rm_outliers_shift,
            filter_window=filter_window_shift,
            threshold=threshold_shift,
            verbose=verbose,
            debug=debug,
            )

    if find_wcs:
        ###
        #   Determine WCS and add it to all images
        #
        terminal_output.print_terminal(string="Determine WCS ...", indent=1)
        if wcs_all:
            aux.find_wcs_all_imgs(
                out_path / 'cut',
                out_path / 'cut',
                method=wcs_method,
                )
        else:
            aux.find_wcs(
                out_path / 'cut',
                out_path / 'cut',
                ref_id=ref_img,
                method=wcs_method,
                )

    if estimate_fwhm:
        ###
        #   Estimate FWHM
        #
        terminal_output.print_terminal(string="Estimate FWHM ...", indent=1)
        aux.estimate_fwhm(
            out_path / 'cut',
            out_path,
            img_type['light'],
            )

    if stack:
        ###
        #   Stack images of the individual filters
        #
        terminal_output.print_terminal(
            string="Combine the images of the individual filter...",
            indent=1,
            )
        stack_img(
            out_path / 'cut',
            out_path,
            img_type['light'],
            method=stack_method,
            debug=debug,
        )

        if not shift_all:
            if shift_method == 'aa_true':
                ###
                #   Trim stacked images using astroalign
                #
                shift_stack_aa(out_path, out_path, img_type['light'])

            elif shift_method in ['own', 'skimage', 'aa']:
                ###
                #   Make large images with the same dimensions to allow
                #   cross correlation
                #
                make_big(out_path, out_path, img_type['light'])


                ###
                #   Calculate and apply image shifts between filters
                #
                terminal_output.print_terminal(
                    string="Trim stacked images of the filters to the same"\
                           " FOV...",
                    indent=1,
                    )

                trim_img(
                    out_path,
                    out_path,
                    img_type['light'],
                    shift_method=shift_method,
                    rm_outliers=rm_outliers_shift,
                    filter_window=filter_window_shift,
                    threshold=threshold_shift,
                    verbose=verbose,
                    )

            else:
                raise RuntimeError(
                    f"{style.bcolors.FAIL}Method for determining image "
                    f"shifts {shift_method} not known {style.bcolors.ENDC}"
                    )

    else:
        #   Sort files according to filter into subdirectories
        light_type = aux.get_image_type(
            ifc,
            img_type,
            image_class='light',
            )
        ifc_filter = ifc.filter(imagetyp=light_type)
        filters = set(
            ifc_filter.summary['filter'][
                np.invert(ifc_filter.summary['filter'].mask)
                ]
            )
        for filt in filters:
            #   Remove old files in the output directory
            checks.clear_directory(out_path / filt)

            #   Set path to files
            file_path = checks.check_pathlib_Path(out_path / 'cut')

            #   New image collection for the images
            ifc = ccdp.ImageFileCollection(file_path)

            #   Restrict to current filter
            filt_files = ifc.files_filtered(filter=filt, include_path=True)

            #   Link files to corresponding directory
            aux_general.link_files(out_path / filt, filt_files)


def master_bias(path, outdir, img_type):
    '''
        This function calculates master biases from individual bias images
        located in one directory.

        Parameters
        ----------
        path            : `string` or `pathlib.Path`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        img_type        : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   Create image collection
    ifc = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not ifc.files:
        return

    #   Get bias frames
    bias_frames = ifc.files_filtered(
        imagetyp=img_type['bias'],
        include_path=True,
        )

    #   Combine biases: Average images + sigma clipping to remove outliers,
    #                   set memory limit to 15GB, set unit to 'adu' since
    #                   this is not set in our images -> find better
    #                   solution
    combined_bias = ccdp.combine(
        bias_frames,
        method='average',
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        signma_clip_dev_func=mad_std,
        mem_limit=15e9,
        unit='adu',
    )

    #   Add Header keyword to mark the file as an Master
    combined_bias.meta['combined'] = True

    #   Write file to disk
    combined_bias.write(out_path / 'combined_bias.fit',  overwrite=True)


def master_image_list(*args, **kwargs):
    '''
        Wrapper function to create a master calibration image for the files
        in the directories given in the path list 'paths'
    '''
    if kwargs['calib_type'] == 'dark':
        master_dark(*args, **kwargs)
    elif kwargs['calib_type'] == 'flat':
        master_flat(*args, **kwargs)


def reduce_dark(path, outdir, image_type, gain=None, readnoise=8.):
    '''
        Reduce dark images: This function reduces the raw dark frames

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        image_type      : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light

        gain            : `float` or `None`, optional
            The gain (e-/adu) of the camera chip. If set to `None` the gain
            will be extracted from the FITS header.
            Default is ``None``.

        readnoise       : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   Create image collection for the flats
    ifc = ccdp.ImageFileCollection(file_path)

    #   Create image collection for the reduced data
    ifc_reduced = ccdp.ImageFileCollection(out_path)

    #   Get master bias
    bias_type = aux.get_image_type(
        ifc_reduced,
        image_type,
        image_class='bias',
        )
    combined_bias = CCDData.read(ifc_reduced.files_filtered(
                                        imagetyp=bias_type,
                                        combined=True,
                                        include_path=True,
                                        )[0]
                                    )

    #   Set new dark path
    dark_path = Path(out_path / 'dark')
    checks.clear_directory(dark_path)

    #   Loop over darks and reduce darks
    dark_type = aux.get_image_type(ifc, image_type, image_class='dark')
    for dark, file_name in ifc.ccds(
        imagetyp=dark_type,
        ccd_kwargs={'unit': 'adu'},
        return_fname=True,
        ):

        #   Set gain _> get it from Header if not provided
        if gain is None:
            gain = dark.header['EGAIN']

        #   Calculated uncertainty
        dark = ccdp.create_deviation(
            dark,
            gain = gain * u.electron/u.adu,
            readnoise = readnoise * u.electron,
            disregard_nan=True,
            )

        # Subtract bias
        dark = ccdp.subtract_bias(dark, combined_bias)

        #   Save the result
        dark.write(dark_path / file_name, overwrite=True)



def master_dark(path, outdir, image_type, gain=None, readnoise=8., dr={0:0.1},
                mask=True, plots=False, verbose=False, debug=False, **kwargs):
    '''
        This function calculates master darks from individual dark images
        located in one directory. The dark images are group according to
        their exposure time.

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        image_type      : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light

        gain            : `float` or `None`, optional
            The gain (e-/adu) of the camera chip. If set to `None` the gain
            will be extracted from the FITS header.
            Default is ``None``.

        readnoise       : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        dr              : `dictionary(float:float)`, optional
            Temperature dependent dark rate in e-/pix/s:
            key = temperature, value = dark rate
            Default is ``{0:0.1}``.

        mask            : `boolean`, optional
            If True a hot pixel mask is created.
            Default is ``True``.

        plots           : `boolean`, optional
            If True some plots showing some statistic on the dark frames are
            created.
            Default is ``False``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        debug           : `boolean`, optional
            If `True` the intermediate files of the data reduction will not
            be removed.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   Create image collection
    try:
        ifc = ccdp.ImageFileCollection(out_path / 'dark')
    except:
        ifc = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not ifc.files:
        return

    #   Find darks
    dark_mask = [True if file in image_type['dark'] else False \
                 for file in ifc.summary['imagetyp']]

    #   Get all available shapes with exposure times
    shape_expos = set(tuple(zip(
        ifc.summary['naxis1'][dark_mask],
        ifc.summary['naxis2'][dark_mask],
        ifc.summary['exptime'][dark_mask]
    )))

    #   Get only the shapes
    shapes = set(tuple(zip(
        ifc.summary['naxis1'][dark_mask],
        ifc.summary['naxis2'][dark_mask]
    )))

    #   Get the maximum exposure time for each shape
    shape_maxexpo = []
    for shape in shapes:
        expo_times = []
        for shape_expo in shape_expos:
            if shape[0] == shape_expo[0] and shape[1] == shape_expo[1]:
                expo_times.append(shape_expo[2])
        shape_maxexpo.append((*shape, np.max(expo_times)))

    #   Return if no darks are found in this directory
    if not dark_mask:
        return

    #   Get exposure times (set allows to return only unique values)
    dark_times = set(ifc.summary['exptime'][dark_mask])

    #   Larges exposure time
    max_expo = np.max(dark_times)

    #   Loop over exposure times
    dark_type = aux.get_image_type(ifc, image_type, image_class='dark')
    for exp_time in sorted(dark_times):
        #   Get only the darks with the correct exposure time
        calibrated_darks = ifc.files_filtered(
            imagetyp=dark_type,
            exptime=exp_time,
            include_path=True,
            )

        #   Combine darks: Average images + sigma clipping to remove
        #                  outliers, set memory limit to 15GB, set unit to
        #                  'adu' since this is not set in our images
        #                  -> find better solution
        combined_dark = ccdp.combine(
            calibrated_darks,
            method='average',
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=15e9,
            unit='adu',
            )

        #   Add Header keyword to mark the file as an Master
        combined_dark.meta['combined'] = True

        #   Write file to disk
        dark_file_name = 'combined_dark_{:4.2f}.fit'.format(exp_time)
        combined_dark.write(out_path / dark_file_name, overwrite=True)

        #   Set gain _> get it from Header if not provided
        if gain is None:
            gain = combined_dark.header['EGAIN']

        #   Plot histogram
        if plots:
            plot.plot_hist(combined_dark.data, out_path, gain, exp_time)
            plot.plot_dark_with_distributions(
                combined_dark.data,
                rn,
                dr,
                out_path,
                exposure=exp_time,
                gain=gain,
                )

        #   Create mask with hot pixels
        shape1 = combined_dark.meta['naxis1']
        shape2 = combined_dark.meta['naxis2']
        if (shape1, shape2, exp_time) in shape_maxexpo and mask:
            aux.make_hot_pixel_mask(
                combined_dark,
                gain,
                out_path,
                verbose=verbose,
            )

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(out_path / 'dark', ignore_errors=True)


def reduce_flat(path, outdir, image_type, gain=None, readnoise=8.,
                rmbias=False, tolerance=0.5, **kwargs):
    '''
        Reduce flat images: This function reduces the raw flat frames,
                            subtracts master dark and if necessary also
                            master bias

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        image_type      : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light

        gain            : `float` or `None`, optional
            The gain (e-/adu) of the camera. If set to `None` the gain will
            be extracted from the FITS header.
            Default is ``None``.

        readnoise       : `float`, optional
            The read noise (e-) of the camera.
            Default is 8 e-.

        rmbias          : boolean`, optional
            If True the master bias image will be subtracted from the flats
            Default is ``False``.

        tolerance           : `float` or `None`, optional
            Maximum difference, in seconds, between the image and the
            closest entry from the exposure time list. Set to ``None`` to
            skip the tolerance test.
            Default is ``0.5``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   Create image collection for the flats
    ifc_flats = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not ifc_flats.files:
        return

    #   Find flats
    flats = [
        True if file in image_type['flat'] else False for file in \
        ifc_flats.summary['imagetyp']
        ]

    #   Return if no flats are found in this directory
    if not flats:
        return

    #   Get image collection for the reduced files
    ifc_reduced = ccdp.ImageFileCollection(out_path)

    #   Get master dark
    dark_type = aux.get_image_type(
        ifc_reduced,
        image_type,
        image_class='dark',
        )
    combined_darks = {
        ccd.header['exptime']: ccd for ccd in ifc_reduced.ccds(
                                                imagetyp=dark_type,
                                                combined=True,
                                                )
        }

    #   Get master bias
    bias_type = aux.get_image_type(
        ifc_reduced,
        image_type,
        image_class='bias',
        )
    if rmbias:
        combined_bias = CCDData.read(ifc_reduced.files_filtered(
                                        imagetyp=bias_type,
                                        combined=True,
                                        include_path=True,
                                        )[0]
                                    )

    #   Set new flat path
    flat_path = Path(out_path / 'flat')
    checks.clear_directory(flat_path)

    #   Loop over flats and reduce flats
    flat_type = aux.get_image_type(
        ifc_flats,
        image_type,
        image_class='flat',
        )
    for flat, file_name in ifc_flats.ccds(
        imagetyp=flat_type,
        ccd_kwargs={'unit': 'adu'},
        return_fname=True,
        ):

        #   Set gain _> get it from Header if not provided
        if gain is None:
            gain = flat.header['EGAIN']

        #   Calculated uncertainty
        flat = ccdp.create_deviation(
            flat,
            gain = gain * u.electron/u.adu,
            readnoise = readnoise * u.electron,
            disregard_nan=True,
            )

        # Subtract bias
        if rmbias:
            flat = ccdp.subtract_bias(flat, combined_bias)

        #   Find the correct dark exposure
        valid, closest_dark = aux.find_nearest_exposure(
            flat,
            combined_darks.keys(),
            tolerance=tolerance,
            )

        #   Exit if no dark with a similar exposure time have been found
        if not valid and not rmbias:
            raise RuntimeError(
                f"{style.bcolors.FAIL}Closest dark exposure time is "
                f"{closest_dark} for flat of exposure time "
                f"{flat.header['exptime']}. {style.bcolors.ENDC}"
                )

        #   Subtract the dark current
        flat = ccdp.subtract_dark(
            flat,
            combined_darks[closest_dark],
            exposure_time='exptime',
            exposure_unit=u.second,
            scale=rmbias,
            )

        #   Save the result
        flat.write(flat_path / file_name, overwrite=True)


def master_flat(path, outdir, image_type, mask=True, plots=False,
                verbose=False, debug=False, **kwargs):
    '''
        This function calculates master flats from individual flat field
        images located in one directory. The flat field images are group
        according to their exposure time.

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        image_type      : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light

        mask            : `boolean`, optional
            If True a bad pixel mask is created.
            Default is ``True``.

        plots           : `boolean`, optional
            If True some plots showing some statistic on the flat fields are
            created.
            Default is ``False``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        debug           : `boolean`, optional
            If `True` the intermediate files of the data reduction will not
            be removed.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   Create new image collection for the reduced flat images
    ifc_flats = ccdp.ImageFileCollection(file_path)

    #   Determine filter
    flat_type = aux.get_image_type(
        ifc_flats,
        image_type,
        image_class='flat',
        )
    filters = set(
        h['filter'] for h in ifc_flats.headers(imagetyp=flat_type)
        )

    #   List for the bad pixels masks
    bpmask_list = []

    #   Combine flats for the individual filters
    for filt in filters:
        #   Select flats to combine
        to_combine = ifc_flats.files_filtered(
            imagetyp=flat_type,
            filter=filt,
            include_path=True,
            )

        #   Combine darks: Average images + sigma clipping to remove
        #                  outliers, set memory limit to 15GB, scale the
        #                  frames so that they have the same median value
        #                  ('inv_median')
        combined_flat = ccdp.combine(
            to_combine,
            method='average',
            scale=aux.inv_median,
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            signma_clip_dev_func=mad_std,
            mem_limit=15e9,
            )

        #   Add Header keyword to mark the file as an Master
        combined_flat.meta['combined'] = True

        #   Define name and write file to disk
        flat_file_name = 'combined_flat_filter_{}.fit'.format(
            filt.replace("''", "p")
            )
        combined_flat.write(out_path / flat_file_name, overwrite=True)

        #   Plot flat medians and means
        if plots:
            plot.plot_flat_median(
                ifc_flats,
                flat_type,
                out_path,
                filt,
                )

        #   Calculate bad pixel mask
        if mask:
            bpmask_list.append(ccdp.ccdmask(combined_flat.data))

    if mask:
        aux.make_bad_pixel_mask(
            bpmask_list,
            out_path,
            verbose=verbose,
            )

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)


def reduce_master(paths, *args, **kwargs):
    '''
        Wrapper function for reduction of the science images

        Parameters
        ----------
        paths           : `list of strings`
            List with paths to the images
    '''
    if isinstance(paths, list):
        for path in paths:
            reduce_light(path, *args, **kwargs)
    elif isinstance(paths, str) or isinstance(paths, Path):
        reduce_light(paths, *args, **kwargs)
    else:
        raise RuntimeError(
            f'{style.bcolors.FAIL}Supplied path is neither str nor list'
            f'{style.bcolors.ENDC}'
            )


def reduce_light(path, outdir, image_type, cosmics=True, mask_cosmics=False,
                 gain=None, readnoise=8., satlevel=65535., objlim=5.,
                 sigclip=4.5, scale_expo=True, rmbias=False, verbose=False,
                 addmask=True, tolerance=0.5, target=None):
    '''
        Reduce the science images

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be stored

        image_type      : `dictionary`
            Image types of the images. Possibilities: bias, dark, flat,
            light

        cosmics         : `boolean`, optional
            If True cosmic rays will be removed.
            Default is ``True``.

        mask_cosmics    : `boolean`, optional
            If True cosmics will ''only'' be masked. If False the
            cosmics will be removed from the input image and the mask will
            be added.
            Default is ``False``.

        gain            : `float` or `None`, optional
            The gain (e-/adu) of the camera chip. If set to `None` the gain
            will be extracted from the FITS header.
            Default is ``None``.

        readnoise       : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        satlevel        : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        objlim          : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        sigclip         : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        scale_expo      : `boolean`, optional
            If True the image will be scaled with the exposure time.
            Default is ``True``.

        rmbias          : boolean`, optional
            If True the master bias image will be subtracted from the flats
            Default is ``False``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        addmask         : `boolean`, optional
            If True add hot and bad pixel mask to the reduced science
            images.
            Default is ``True``.

        tolerance       : `float`, optional
            Tolerance between science and dark exposure times in s.
            Default is ``0.5``s.

        target          : `string` or ``None``, optional
            Name of the target. Used for file selection.
            Default is ``None``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   Get image collection for the science images
    ifc_lights = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not ifc_lights.files:
        return

    #   Limit images to those of the target. If a target is given.
    if target is not None:
        ifc_lights = ifc_lights.filter(object=target)

    #   Find science images
    lights = [True if file in image_type['light'] else False for file in \
              ifc_lights.summary['imagetyp']]

    #   Return if no science images are found in this directory
    if not lights:
        return

    #   Get image collection for the reduced files
    ifc_reduced = ccdp.ImageFileCollection(out_path)

    #   Load combined darks and flats in dictionary for easy access
    dark_type = aux.get_image_type(
        ifc_reduced,
        image_type,
        image_class='dark',
        )
    combined_darks = {
        ccd.header['exptime']: ccd for ccd in ifc_reduced.ccds(
                                                imagetyp=dark_type,
                                                combined=True,
                                                )
        }
    flat_type = aux.get_image_type(
        ifc_reduced,
        image_type,
        image_class='flat',
        )
    combined_flats = {
        ccd.header['filter']: ccd for ccd in ifc_reduced.ccds(
                                                imagetyp=flat_type,
                                                combined=True,
                                                )
        }

    #   Get master bias
    bias_type = aux.get_image_type(
        ifc_reduced,
        image_type,
        image_class='bias',
        )
    if rmbias:
        combined_bias = CCDData.read(ifc_reduced.files_filtered(
                                        imagetyp=bias_type,
                                        combined=True,
                                        include_path=True,
                                        )[0]
                                    )

    #   Set science image path
    light_path = Path(out_path / 'light')

    dir_empty = checks.check_dir_empty(light_path)

    if not dir_empty:
        user_input, timedOut = timedInput(
            f"{style.bcolors.OKBLUE}   Reduced images from a previous run "
            f"found. Should these be used? {style.bcolors.ENDC}",
            timeout=30,
            )
        if timedOut:
            user_input = 'n'

        if user_input in ['y', 'yes']:
            return

    checks.clear_directory(light_path)

    #   Reduce science images and save to an extra directory
    light_type = aux.get_image_type(
        ifc_lights,
        image_type,
        image_class='light',
        )
    for light, file_name in ifc_lights.ccds(
        imagetyp=light_type,
        return_fname=True,
        ccd_kwargs=dict(unit='adu'),
        ):

        #   Set gain _> get it from Header if not provided
        if gain is None:
            try:
                gain = light.header['EGAIN']
            except:
                gain = 1.
                terminal_output.print_terminal(
                    string="WARNING: Gain could not de derived from the "\
                           "image header. Use 1.0 instead",
                    style_name='WARNING',
                    indent=2,
                    )

        #   Calculated uncertainty dd
        light = ccdp.create_deviation(
            light,
            gain = gain * u.electron/u.adu,
            readnoise = readnoise * u.electron,
            disregard_nan=True,
            )

        #   Subtract bias
        if rmbias:
            light = ccdp.subtract_bias(light, combined_bias)

        #   Find the correct dark exposure
        valid, closest_dark = aux.find_nearest_exposure(
            light,
            combined_darks.keys(),
            tolerance=tolerance,
            )

        #   Exit if no dark with a similar exposure time have been found
        if not valid and not rmbias:
            raise RuntimeError(
                f"{style.bcolors.FAIL}Closest dark exposure time is "
                f"{closest_dark} for science image of exposure "
                f"time {light.header['exptime']}. {style.bcolors.ENDC}"
                )

        #   Subtract dark
        reduced = ccdp.subtract_dark(
            light,
            combined_darks[closest_dark],
            exposure_time='exptime',
            exposure_unit=u.second,
            scale=rmbias,
            )

        #   Mask negative pixel
        mask = reduced.data < 0.
        reduced.mask = reduced.mask | mask

        #   Get master flat field
        master_flat = combined_flats[reduced.header['filter']]

        #   Divided science by the master flat
        reduced = ccdp.flat_correct(reduced, master_flat)

        if addmask:
            #   Get mask of bad and hot pixel
            badpix, badmask = aux.get_pixel_mask(
                out_path,
                reduced.shape,
                )

            #   Add bad pixel mask: If there was already a mask, keep it
            if badpix:
                if reduced.mask is not None:
                    reduced.mask = reduced.mask | badmask
                else:
                    reduced.mask = badmask

        #   Gain correct data
        reduced = ccdp.gain_correct(reduced, gain * u.electron / u.adu)

        #   Remove cosmic rays
        if cosmics:
            if verbose:
                sys.stdout.write(
                    f'\r\tRemove cosmic rays from image {file_name}\n'
                    )
                sys.stdout.flush()
            no_cosm = ccdp.cosmicray_lacosmic(
                reduced,
                objlim=objlim,
                readnoise=readnoise,
                sigclip=sigclip,
                satlevel=satlevel,
                verbose=verbose,
                )

            if mask_cosmics:
                if addmask:
                    reduced.mask = reduced.mask | no_cosm.mask

                    #   Add Header keyword to mark the file as combined
                    reduced.meta['cosmic_mas'] = True
            else:
                reduced = no_cosm
                if not addmask:
                    reduced.mask = np.zeros(reduced.shape, dtype=bool)

                #   Add Header keyword to mark the file as combined
                reduced.meta['cosmics_rm'] = True

            if verbose:
                terminal_output.print_terminal()

        #   Scale image with exposure time
        if scale_expo:
            #   Get exposure time and all meta data
            exposure = reduced.header['exptime']
            reduced_meta = reduced.meta

            #   Scale image
            reduced  = reduced.divide(exposure * u.second)

            #   Put meta data back on the image, because it is lost while
            #   dividing
            reduced.meta = reduced_meta
            reduced.meta['HIERARCH'] = 'Image scaled by exposure time:'

        #   Write reduced science image to disk
        reduced.write(light_path / file_name, overwrite=True)


def shift_img_apply(img_ccd, reff_ccd, nfiles, shift, flip, img_id, out_path,
                    fname, shift_method='skimage', modify_file_name=False,
                    enlarged=False, verbose=False):
    '''
        Apply shift to an individual image

        Parameters
        ----------
        img_ccd             : `astropy.nddata.CCDData` object
            Image data

        reff_ccd            : `astropy.nddata.CCDData` object
            Data of the reference image

        nfiles              : `integer`
            Number of images

        shift               : `numpy.ndarray`
            Shifts of the images in X and Y direction

        flip                : `numpy.ndarray`
            Flip necessary to account for pier flips

        img_id              : `integer`
            ID of the image

        out_path            : `pathlib.Path` object
            Path to the output directory

        fname               : `string`
            Name of the image

        shift_method        : `string`, optional
            Method to use for image alignment.
            Possibilities: 'aa'      = astroalign module only accounting for
                                       xy shifts
                           'aa_true' = astroalign module with corresponding
                                       transformation
                           'own'     = own correlation routine based on
                                       phase correlation, applying fft to
                                       the images
                           'skimage' = phase correlation implemented by
                                       skimage
                           'flow'    = image registration using optical flow
                                       implementation by skimage
            Default is ``skimage``.

        modify_file_name    : `boolean`, optional
            It true the trimmed image will be saved, using a modified file
            name.
            Default is ``False``.

        enlarged            : `boolean`, optional
            It true the header keyword 'enlarged' will be removed.
            Default is ``False``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the console
            Default is ``False``.
    '''
    #   Trim images
    if shift_method in ['own', 'skimage', 'aa']:
        #   Flip image if pier side changed
        if flip[img_id]:
            img_ccd = ccdp.transform_image(img_ccd, np.flip, axis=(0,1))

        img_out = aux.trim_core(
            img_ccd,
            img_id,
            nfiles,
            shift,
            method=shift_method,
            verbose=verbose,
            )
    elif shift_method == 'flow':
        img_out = aux.shift_optical_flow_method(reff_ccd, img_ccd)

    #   Using astroalign to align the images
    elif shift_method == 'aa_true':
        img_out = aux.shift_astroalign_method(reff_ccd, img_ccd)

    #   Add Header keyword to mark the file as trimmed
    img_out.meta['trimmed'] = True
    if enlarged:
        img_out.meta.remove('enlarged')

    if modify_file_name:
        #   Get filter
        filt = img_out.meta['filter']

        #   Define name and write trimmed image to disk
        fname = 'combined_trimmed_filter_{}.fit'.format(
            filt.replace("''", "p")
            )

    #   Write trimmed image to disk
    img_out.write(out_path / fname, overwrite=True)


def detect_outlier(data, filter_window=8, threshold=10.):
    '''
        Find outliers in a data array

        Parameters
        ----------
        data            : `numpy.ndarray`
            Data

        filter_window   : `integer`, optional
            Width of the median filter window
            Default is ``8``.

        threshold       : `float` or `integer`, optional
            Difference above the running median above an element is
            considered to be an outlier.
            Default is ``10.``.

        Returns
        -------
                        : `numpy.ndarray`
            Index of the elements along axis 0 that are below the threshold
    '''
    #   Calculate running median
    run_median = median_filter(data, size=(1,filter_window))

    #   Difference compared to median and sum along axis 0
    score = np.sum(np.abs(data-run_median), axis=0)

    #   Return outliers
    return np.argwhere(score > threshold)


def shift_img_core(ifc, out_path, shift_method='skimage', ref_img=0,
                   shift_text='\tImage displacement:', enlarged=False,
                   modify_file_name=False, rm_outliers=True, filter_window=8,
                   threshold=10., verbose=False):
    '''
        Core steps of the image shift calculations and trimming to a
        common filed of view

        Parameters
        ----------
        ifc                 : `ccdproc.ImageFileCollection`
            Image file collection with all images

        out_path            : `pathlib.Path` object
            Path to the output directory

        shift_method        : `string`, optional
            Method to use for image alignment.
            Possibilities: 'aa'      = astroalign module only accounting for
                                       xy shifts
                           'aa_true' = astroalign module with corresponding
                                       transformation
                           'own'     = own correlation routine based on
                                       phase correlation, applying fft to
                                       the images
                           'skimage' = phase correlation implemented by
                                       skimage
                           'flow'    = image registration using optical flow
                                       implementation by skimage
            Default is ``skimage``.

        ref_img             : `integer`, optional
            ID of the image that should be used as a reference
            Default is ``0``.

        shift_text          : `string`, optional
            Text string that is used to label the output.
            Default is ``Image displacement:``.

        enlarged            : `boolean`, optional
            It True the header keyword 'enlarged' will be removed.
            Default is ``False``.

        modify_file_name    : `boolean`, optional
            It True the trimmed image will be saved, using a modified file
            name.
            Default is ``False``.

        rm_outliers         : `boolean`, optional
            If True outliers in the image shifts will be detected and removed.
            Default is ``True``.

        filter_window       : `integer`, optional
            Width of the median filter window
            Default is ``8``.

        threshold           : `float` or `integer`, optional
            Difference above the running median above an element is
            considered to be an outlier.
            Default is ``10.``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the console
            Default is ``False``.
    '''
    #   Calculate image shifts
    if shift_method in ['own', 'skimage', 'aa']:
        shift, flip = aux.calculate_image_shifts(
            ifc,
            ref_img,
            shift_text,
            method=shift_method,
            )
        reff_ccd = None
    elif shift_method in ['aa_true', 'flow']:
        reff_name = ifc.files[ref_img]
        reff_ccd = CCDData.read(reff_name)
        shift = None
        flip = None
    else:
        raise RuntimeError(
            f'{style.bcolors.FAIL}Method {shift_method} not known '
            f'-> EXIT {style.bcolors.ENDC}'
            )

    #   Number of images
    nimg = len(ifc.files)

    #   Find IDs of potential outlier
    if rm_outliers and shift is not None:
        outliers = detect_outlier(
            shift,
            filter_window=filter_window,
            threshold=threshold,
            )
        if outliers.size:
            terminal_output.print_terminal(
                outliers.ravel(),
                string="The images with the following IDs will be removed "\
                       "because of not reliable shifts:\n {}.",
                indent=2,
                style_name='WARNING',
                )
    else:
        outliers = []

    #   Loop over and trim all images
    for i, (img_ccd, fname) in enumerate(ifc.ccds(return_fname=True)):
        if i not in outliers:
            shift_img_apply(
                img_ccd,
                reff_ccd,
                nimg,
                shift,
                flip,
                i,
                out_path,
                fname,
                shift_method=shift_method,
                modify_file_name=modify_file_name,
                enlarged=enlarged,
                verbose=verbose,
                )


def shift_img(path, outdir, image_type, ref_img=0, shift_method='skimage',
              rm_outliers=True, filter_window=8, threshold=10., verbose=False,
              debug=False):
    '''
        Calculate shift between images taken in the same filter
        and trim those to the save field of view

        Parameters
        ----------
        path                : `string`
            Path to the images

        outdir              : `string`
            Path to the directory where the master files should be saved to

        image_type          : `string`
            Header keyword characterizing the image type for which the
            shifts shall be determined

        ref_img             : `integer`, optional
            ID of the image that should be used as a reference
            Default is ``0``.

        shift_method        : `string`, optional
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

        rm_outliers         : `boolean`, optional
            If True outliers in the image shifts will be detected and removed.
            Default is ``True``.

        filter_window       : `integer`, optional
            Width of the median filter window
            Default is ``8``.

        threshold           : `float` or `integer`, optional
            Difference above the running median above an element is
            considered to be an outlier.
            Default is ``10.``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the console
            Default is ``False``.

        debug               : `boolean`, optional
            If `True` the intermediate files of the data reduction will not
            be removed.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   New image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)

    #   Check if ifc is not empty
    if not ifc.files:
        raise RuntimeError(
            f"{style.bcolors.FAIL}No FITS files found in {file_path}. "
            f"=> EXIT {style.bcolors.ENDC}"
            )

    #   Determine filter
    img_type = aux.get_image_type(ifc, image_type)
    filters = set(h['filter'] for h in ifc.headers(imagetyp=img_type))

    #   Set science image path
    trim_path = Path(out_path / 'cut')
    checks.clear_directory(trim_path)

    #   Calculate shifts for the images in the individual filters
    for filt in filters:
        #   Restrict image collection to those images with the correct
        #   filter
        ifc_filter = ifc.filter(filter=filt)

        #   Calculate image shifts and trim images accordingly
        shift_img_core(
            ifc_filter,
            trim_path,
            shift_method=shift_method,
            ref_img=ref_img,
            shift_text=f'\tDisplacement for images in filter: {filt}',
            rm_outliers=rm_outliers,
            filter_window=filter_window,
            threshold=threshold,
            verbose=verbose,
            )

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)


def shift_img_all(path, outdir, image_type, ref_img=0,
                  shift_method='skimage', rm_outliers=True, filter_window=8,
                  threshold=10., verbose=False, debug=False):
    '''
        Calculate shift between images and trim those to the save field of
        view

        Parameters
        ----------
        path                : `string`
            Path to the images

        outdir              : `string`
            Path to the directory where the master files should be saved to

        image_type          : `string`
            Header keyword characterizing the image type for which the
            shifts shall be determined

        ref_img             : `integer`, optional
            ID of the image that should be used as a reference
            Default is ``0``.

        shift_method        : `string`, optional
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

        rm_outliers         : `boolean`, optional
            If True outliers in the image shifts will be detected and removed.
            Default is ``True``.

        filter_window       : `integer`, optional
            Width of the median filter window
            Default is ``8``.

        threshold           : `float` or `integer`, optional
            Difference above the running median above an element is
            considered to be an outlier.
            Default is ``10.``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the console
            Default is ``False``.

        debug               : `boolean`, optional
            If `True` the intermediate files of the data reduction will not
            be removed.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   New image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)

    #   Check if ifc is not empty
    if not ifc.files:
        raise RuntimeError(
            f"{style.bcolors.FAIL}No FITS files found in {file_path}. "
            f"=> EXIT {style.bcolors.ENDC}"
            )

    #   Apply ifc filter to the image collection
    #   -> This is necessary so that the path to the image directory is
    #      added to the file names. This is required for
    #      `shift_img_core`.
    img_type = aux.get_image_type(ifc, image_type)
    ifc_mod = ifc.filter(imagetyp=img_type)

    #   Set output path
    trim_path = Path(out_path / 'cut')
    checks.clear_directory(trim_path)

    #   Calculate image shifts and trim images accordingly
    shift_img_core(
        ifc_mod,
        trim_path,
        shift_method=shift_method,
        ref_img=ref_img,
        rm_outliers=rm_outliers,
        filter_window=filter_window,
        threshold=threshold,
        verbose=verbose,
        )

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)



def shift_stack_aa(path, outdir, image_type):
    '''
        Calculate shift between stacked images and trim those
        to the save field of view

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        image_type      : `string`
            Header keyword characterizing the image type for which the
            shifts shall be determined
    '''
    #   New image collection for the images
    ifc = ccdp.ImageFileCollection(path)
    img_type = aux.get_image_type(ifc, image_type)
    ifc_mod = ifc.filter(combined=True, imagetyp=img_type)

    for i, (img_ccd, fname) in enumerate(ifc_mod.ccds(return_fname=True)):
        if i == 0:
            reff_ccd = img_ccd
            img_out  = reff_ccd
        else:
            #   Byte order of the system
            sbo = sys.byteorder

            #   Map with endianness symbols
            endian_map = {
                '>': 'big',
                '<': 'little',
                '=': sbo,
                '|': 'not applicable',
            }
            if endian_map[img_ccd.data.dtype.byteorder] != sbo:
                img_ccd.data = img_ccd.data.byteswap().newbyteorder()
                reff_ccd.data = reff_ccd.data.byteswap().newbyteorder()
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
            img_out = CCDData(
                img_data,
                mask=footprint,
                meta=img_ccd.meta,
                unit=img_ccd.unit,
                uncertainty=StdDevUncertainty(img_uncert),
                )

        #   Get filter
        filt = img_out.meta['filter']

        #   Define name and write trimmed image to disk
        file_name = 'combined_trimmed_filter_{}.fit'.format(
            filt.replace("''", "p")
            )
        img_out.write(outdir / file_name, overwrite=True)


def stack_img(path, outdir, image_type, method='average', debug=False):
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

        method          : `string`, optional
            Method used for combining the images.
            Possibilities: ``median`` or ``average`` or ``sum``
            Default is ``average`.

        debug           : `boolean`, optional
            If `True` the intermediate files of the data reduction will not
            be removed.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   New image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)

    #   Check if ifc is not empty
    if not ifc.files:
        raise RuntimeError(
            f"{style.bcolors.FAIL}No FITS files found in {file_path}. "
            f"=> EXIT {style.bcolors.ENDC}"
            )

    #   Determine filter
    img_type = aux.get_image_type(ifc, image_type)
    filters = set(h['filter'] for h in ifc.headers(imagetyp=img_type))

    #   Combine images for the individual filters
    for filt in filters:
        #   Select images to combine
        to_combine = ifc.files_filtered(
            imagetyp=img_type,
            filter=filt,
            include_path=True,
            )

        #   Combine darks: Average images + sigma clipping to remove
        #                  outliers, set memory limit to 15GB
        combined_img = ccdp.combine(
            to_combine,
            method=method,
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            signma_clip_dev_func=mad_std,
            mem_limit=15e9,
            )

        #   Add Header keyword to mark the file as combined
        combined_img.meta['COMBINED'] = True
        nimg                          = len(to_combine)
        combined_img.meta['N-IMAGES'] = nimg
        combined_img.meta['EXPTIME']  = nimg * combined_img.meta['EXPTIME']
        combined_img.meta['EXPOSURE'] = combined_img.meta['EXPTIME']

        #   Define name and write file to disk
        file_name = 'combined_filter_{}.fit'.format(
            filt.replace("''", "p")
            )
        combined_img.write(out_path / file_name, overwrite=True)

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)


def make_big(path, outdir, image_type, combined=True):
    '''
        Image size unification:
            Find the largest image and use this for all other images

        Parameters
        ----------
        path            : `string`
            Path to the images

        outdir          : `string`
            Path to the directory where the master files should be saved to

        image_type      : `string`
            Header keyword characterizing the image type for which the
            shifts shall be determined

        combined        : `boolean`, optional
            It true the file selection will be restricted to images with a
            header keyword 'combined' that is set to True.
            Default is ``True``.

    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   New image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)

    #   Image list
    img_type = aux.get_image_type(ifc, image_type)
    img_dict = {file_name: ccd for ccd, file_name in ifc.ccds(
                                        imagetyp=img_type,
                                        return_fname=True,
                                        combined=combined,
                                        )
        }

    #   Image list
    imgs = list(img_dict.values())

    #   File name list
    file_names = list(img_dict.keys())

    #   Number of images
    nimg = len(file_names)

    #   Get image dimensions
    nx = np.zeros((nimg), dtype='int')
    ny = np.zeros((nimg), dtype='int')
    for i, img in enumerate(imgs):
        #   Original image dimension
        nx[i] = img.shape[1]
        ny[i] = img.shape[0]

    #   Maximum size
    nxmax = np.max(nx)
    nymax = np.max(ny)

    for i, img in enumerate(imgs):
        #   Make big image ans mask
        BIGimg    = np.zeros((nymax,nxmax))
        BIGmask   = np.ones((nymax,nxmax), dtype=bool)
        BIGuncert = np.zeros((nymax,nxmax))

        #   Fill image and mask
        BIGimg[0:ny[i],0:nx[i]]    = img.data
        BIGmask[0:ny[i],0:nx[i]]   = img.mask
        BIGuncert[0:ny[i],0:nx[i]] = img.uncertainty.array

        #   Replace
        img.data        = BIGimg
        img.mask        = BIGmask
        img.uncertainty.array = BIGuncert

        #   Add Header keyword to mark the file as an Master
        img.meta['enlarged'] = True
        img.meta.remove('combined')

        #   Get filter
        filt = img.meta['filter']

        #   Define name and write trimmed image to disk
        file_name = 'combined_enlarged_filter_{}.fit'.format(
            filt.replace("''", "p")
            )
        img.write(out_path / file_name, overwrite=True)


def trim_img(path, outdir, image_type, ref_img=0, enlarged=True,
             shift_method='skimage', rm_outliers=True, filter_window=8,
             threshold=10., verbose=False):
    '''
        Trim images to the same field of view

        Parameters
        ----------
        path                : `string`
            Path to the images

        outdir              : `string`
            Path to the directory where the master files should be saved to

        image_type          : `string`
            Header keyword characterizing the image type for which the
            shifts shall be determined

        ref_img             : `integer`, optional
            ID of the image that should be used as a reference
            Default is ``0``.

        enlarged            : `boolean`, optional
            It true the file selection will be restricted to images with a
            header keyword 'enlarged' that is set to True.
            Default is ``True``.

        shift_method        :`string`, optional
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

        rm_outliers         : `boolean`, optional
            If True outliers in the image shifts will be detected and removed.
            Default is ``True``.

        filter_window   : `integer`, optional
            Width of the median filter window
            Default is ``8``.

        threshold       : `float` or `integer`, optional
            Difference above the running median above an element is
            considered to be an outlier.
            Default is ``10.``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    '''
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_Path(path)
    out_path  = checks.check_pathlib_Path(outdir)

    #   New image collection for the images
    ifc = ccdp.ImageFileCollection(file_path)

    #   Restrict image collection to those images correct image type and
    #   the 'enlarged' Header keyword
    img_type = aux.get_image_type(ifc, image_type)
    ifc_filtered = ifc.filter(imagetyp=img_type, enlarged=enlarged)

    #   Calculate image shifts and trim images accordingly
    shift_img_core(
        ifc_filtered,
        out_path,
        shift_method=shift_method,
        ref_img=ref_img,
        shift_text='\tDisplacement between the images of the different filters',
        enlarged=enlarged,
        modify_file_name=True,
        rm_outliers=rm_outliers,
        filter_window=filter_window,
        threshold=threshold,
        verbose=verbose,
        )
