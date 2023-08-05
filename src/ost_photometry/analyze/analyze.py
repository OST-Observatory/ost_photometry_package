############################################################################
#                               Libraries                                  #
############################################################################

import os

import numpy as np

from uncertainties import unumpy

from collections import Counter

from pathlib import Path

import warnings

from photutils import (
    DAOStarFinder,
    EPSFBuilder,
)
from photutils.psf import (
    extract_stars,
    DAOGroup,
    IterativelySubtractedPSFPhotometry,
)
from photutils.detection import IRAFStarFinder
from photutils.background import (
    MMMBackground,
    MADStdBackgroundRMS,
)

import ccdproc as ccdp

from astropy.stats import SigmaClip

from astropy.table import Table
from astropy.time import Time
from astropy.nddata import NDData
from astropy.stats import (gaussian_sigma_to_fwhm, sigma_clipped_stats)
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

#   hips2fits module is not in the Ubuntu 22.04 package version
#   of astroquery (0.4.1)
# from astroquery.hips2fits import hips2fits
from astroquery.hips2fits import hips2fitsClass

from astropy.nddata import CCDData

from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus,
)
from photutils.background import Background2D, MedianBackground
# from photutils.utils import calc_total_error

import multiprocessing as mp

from . import utilities, calib, trans, plot, correlate
# from . import subtraction

from .. import style, checks, terminal_output, calibration_data

from .. import utilities as base_utilities

warnings.filterwarnings('ignore', category=UserWarning, append=True)


############################################################################
#                           Routines & definitions                         #
############################################################################


class ImageContainer:
    """
        Container class for image class objects
    """

    def __init__(self, **kwargs):
        #   Prepare dictionary
        self.ensembles = {}

        #   Add additional keywords
        self.__dict__.update(kwargs)

        #   Check for right ascension and declination
        ra = kwargs.get('ra', None)
        dec = kwargs.get('dec', None)
        if ra is not None:
            self.ra = Angle(ra, unit='hour').degree
        else:
            self.ra = None
        if dec is not None:
            self.dec = Angle(dec, unit='degree').degree
        else:
            self.dec = None

        #   Check for an object name
        self.name = kwargs.get('name', None)

        #   Create SkyCoord object
        if self.name is not None and self.ra is None and self.dec is None:
            self.coord = SkyCoord.from_name(self.name)
        elif self.ra is not None and self.dec is not None:
            self.coord = SkyCoord(
                ra=self.ra,
                dec=self.dec,
                unit=(u.degree, u.degree),
                frame="icrs"
            )
        else:
            self.coord = None

        #   Check if uncertainty should be calculated by means of the
        #   "uncertainties" package. Default is ``True``.
        self.unc = kwargs.get('unc', True)

    #   Get ePSF objects of all images
    def get_epsf(self):
        epsf_dict = {}
        for key, ensemble in self.ensembles.items():
            epsf_list = []
            for img in ensemble.image_list:
                epsf_list.append(img.epsf)
            epsf_dict[key] = epsf_list

        return epsf_dict

    #   Get ePSF object of the reference image
    def get_ref_epsf(self):
        epsf_dict = {}
        for key, ensemble in self.ensembles.items():
            reference_image_id = ensemble.reference_image_id

            img = ensemble.image_list[reference_image_id]

            epsf_dict[key] = img.epsf

        return epsf_dict

    #   Get reference image
    def get_ref_img(self):
        img_dict = {}
        for key, ensemble in self.ensembles.items():
            reference_image_id = ensemble.reference_image_id

            img = ensemble.image_list[reference_image_id]

            img_dict[key] = img.get_data()

        return img_dict

    #   Get residual image belonging to the reference image
    def get_ref_residual_img(self):
        img_dict = {}
        for key, ensemble in self.ensembles.items():
            reference_image_id = ensemble.reference_image_id

            img = ensemble.image_list[reference_image_id]

            img_dict[key] = img.residual_image

        return img_dict

    #   Get image ensembles for a specific set of filter
    def get_ensembles(self, filter_list):
        ensembles = {}
        for filt in filter_list:
            ensembles[filt] = self.ensembles[filt]

        return ensembles

    #   Get calibrated magnitudes as numpy.ndarray
    def get_calibrated_magnitudes(self):
        #   Get type of the magnitude arrays
        #   Possibilities: unumpy.uarray & numpy structured ndarray
        unc = getattr(self, 'unc', True)

        #   Get calibrated magnitudes
        cali_mags = getattr(self, 'cali', None)
        if unc:
            if (cali_mags is None or
                    np.all(unumpy.nominal_values(cali_mags) == 0.)):
                #   If array with magnitude transformation is not available
                #   or if it is empty get the array without magnitude
                #   transformation
                cali_mags = getattr(self, 'noT', None)
                if cali_mags is not None:
                    #   Get only the magnitude values
                    cali_mags = unumpy.nominal_values(cali_mags)
            else:
                #   Get only the magnitude values
                cali_mags = unumpy.nominal_values(cali_mags)

        #   numpy structured ndarray type:
        else:
            if cali_mags is None or np.all(cali_mags['mag'] == 0.):
                #   If array with magnitude transformation is not available
                #   or if it is empty get the array without magnitude
                #   transformation
                cali_mags = getattr(self, 'noT', None)
                if cali_mags is not None:
                    cali_mags = cali_mags['mag']
            else:
                cali_mags = cali_mags['mag']

        return cali_mags


class ImageEnsemble:
    """
        Image ensemble class: Used to handle multiple images, e.g.,
        an image series taken in a specific filter
    """

    def __init__(self, filt, obj_name, path, outdir, reference_image_id=0):
        ###
        #   Get file list, if path is a directory, if path is a file put
        #   base name of this file in a list
        #
        if os.path.isdir(path):
            formats = [".FIT", ".fit", ".FITS", ".fits"]
            file_list = os.listdir(path)

            #   Remove not FITS entries
            temp_list = []
            for file_i in file_list:
                for j, form in enumerate(formats):
                    if file_i.find(form) != -1:
                        temp_list.append(file_i)
            file_list = temp_list
        elif os.path.isfile(path):
            file_list = [str(path).split('/')[-1]]
            path = os.path.dirname(path)
        else:
            raise RuntimeError(
                f'{style.Bcolors.FAIL}ERROR: Provided path is neither a file'
                f' nor a directory -> EXIT {style.Bcolors.ENDC}'
            )

        ###
        #   Check if the id of the reference image is valid
        #
        if reference_image_id > len(file_list):
            raise ValueError(
                f'{style.Bcolors.FAIL} ERROR: Reference image ID '
                '[reference_image_id] is larger than the total number of '
                f'images! -> EXIT {style.Bcolors.ENDC}'
            )

        #   Set filter
        self.filt = filt

        #   Set number of images
        self.nfiles = len(file_list)

        #   Set ID of the reference image
        self.reference_image_id = reference_image_id

        #   Prepare image list
        self.image_list = []

        #   Set path to output directory
        self.outpath = Path(outdir)

        #   Set object name
        self.objname = obj_name

        #   Fill image list
        terminal_output.print_terminal(
            string="Read images and calculate FOV, PIXEL scale, etc. ... ",
            indent=2,
        )
        for image_id, file_name in enumerate(file_list):
            self.image_list.append(
                #   Prepare image class instance
                self.Image(image_id, filt, obj_name, path, file_name, outdir)
            )

            #   Calculate field of view and additional quantities and add
            #   them to the image class instance
            base_utilities.cal_fov(self.image_list[image_id], verbose=False)

        #   Set reference image
        self.ref_img = self.image_list[reference_image_id]

        #   Set field of view
        # self.fov = self.ref_img.fov
        self.fov = getattr(self.ref_img, 'fov', None)

        #   Set PixelRegion for the field of view
        # self.region_pix = self.ref_img.region_pix
        self.region_pix = getattr(self.ref_img, 'region_pix', None)

        #   Set pixel scale
        # self.pixscale = self.ref_img.pixscale
        self.pixscale = getattr(self.ref_img, 'pixscale', None)

        #   Set coordinates of image center
        # self.coord = self.ref_img.coord
        self.coord = getattr(self.ref_img, 'coord', None)

        #   Set instrument
        # self.instrument = self.ref_img.instrument
        self.instrument = getattr(self.ref_img, 'instrument', None)

        #   Get image shape
        self.img_shape = self.ref_img.get_data().shape

        #   Set wcs default
        self.wcs = None

    #   Image class
    class Image:
        def __init__(self, pd, filt, obj_name, path, file_name, outdir):
            #   Set image ID
            self.pd = pd
            #   Set filter
            self.filt = filt
            #   Set object name
            self.objname = obj_name
            #   Set file name
            self.filename = file_name
            #   Set complete path
            self.path = Path(Path(path) / file_name)
            #   Set path to output directory
            self.outpath = Path(outdir)

        #   Read image
        def read_image(self):
            return CCDData.read(self.path)

        #   Get header
        def get_header(self):
            return CCDData.read(self.path).meta

        #   Get data
        def get_data(self):
            return CCDData.read(self.path).data

        #   Get shape
        def get_shape(self):
            return CCDData.read(self.path).data.shape

    #   Set wcs
    def set_wcs(self, w):
        self.wcs = w
        for img in self.image_list:
            img.wcs = w

    #   Get extracted photometry of all images
    def get_photometry(self):
        photo_dict = {}
        for img in self.image_list:
            # photo_dict[str(img.pd)] = img.photometry
            photo_dict[str(img.pd)] = getattr(img, 'photometry', None)

        return photo_dict

    #   Get image IDs of all images
    def get_image_ids(self):
        img_ids = []
        for img in self.image_list:
            img_ids.append(img.pd)

        return img_ids

    #   Get sigma clipped mean of the air mass
    def mean_sigma_clip_air_mass(self):
        am_list = []
        for img in self.image_list:
            # am_list.append(img.air_mass)
            am_list.append(getattr(img, 'air_mass', 0.))

        return sigma_clipped_stats(am_list, sigma=1.5)[0]

    #   Get median of the air mass
    def median_air_mass(self):
        am_list = []
        for img in self.image_list:
            # am_list.append(img.air_mass)
            am_list.append(getattr(img, 'air_mass', 0.))

        return np.median(am_list)

    #   Get air mass
    def get_air_mass(self):
        am_list = []
        for img in self.image_list:
            # am_list.append(img.air_mass)
            am_list.append(getattr(img, 'air_mass', 0.))

        return am_list

    #   Get observation times
    def get_obs_time(self):
        obs_time_list = []
        for img in self.image_list:
            # obs_time_list.append(img.jd)
            obs_time_list.append(getattr(img, 'jd', 0.))

        return np.array(obs_time_list)

    #   Get median of the observation time
    def median_obs_time(self):
        obs_time_list = []
        for img in self.image_list:
            # obs_time_list.append(img.jd)
            obs_time_list.append(getattr(img, 'jd', 0.))

        return np.median(obs_time_list)

    #   Get list with dictionary and image class objects
    def get_list_dict(self):
        dict_list = []
        for img in self.image_list:
            dict_list.append({img.filt: img})

        return dict_list

    #   Get object positions in pixel coordinates
    #   TODO: improve?
    def get_object_positions_pixel(self):
        tbls = self.get_photometry()
        nmax_list = []
        x = []
        y = []
        for i, tbl in enumerate(tbls.values()):
            x.append(tbl['x_fit'])
            y.append(tbl['y_fit'])
            nmax_list.append(len(x[i]))

        return x, y, np.max(nmax_list)

    def get_flux_uarray(self):
        #   Get data
        tbls = list(self.get_photometry().values())

        #   Expects the number of objects in each table to be the same.
        n_images = len(tbls)
        n_objects = len(tbls[0])

        flux = np.zeros((n_images, n_objects))
        flux_unc = np.zeros((n_images, n_objects))

        for i, tbl in enumerate(tbls):
            flux[i] = tbl['flux_fit']
            flux_unc[i] = tbl['flux_unc']

        return unumpy.uarray(flux, flux_unc)


def rm_cosmic(image, objlim=5., readnoise=8., sigclip=4.5, satlevel=65535.,
              verbose=False, addmask=True, terminal_logger=None):
    """
        Remove cosmic rays

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        objlim          : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        readnoise       : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigclip         : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        satlevel        : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        addmask         : `boolean`, optional
            If True add hot and bad pixel mask to the reduced science images.
            Default is ``True``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.
    """
    if terminal_logger is not None:
        terminal_logger.add_to_cache("Remove cosmic rays ...")
    else:
        terminal_output.print_to_terminal("Remove cosmic rays ...")

    #   Get image
    ccd = image.read_image()

    #   Get status cosmic ray removal status
    status = ccd.meta.get('cosmics_rm', False)

    #   Get exposure time
    exposure = ccd.meta.get('exptime', 1.)

    #   Get unit of the image to check if the image was scaled with the
    #   exposure time
    if ccd.unit == u.electron / u.s:
        scaled = True
        reduced = ccd.multiply(exposure * u.second)
    else:
        scaled = False
        reduced = ccd

    if not status:
        #   Remove cosmic rays
        reduced = ccdp.cosmicray_lacosmic(
            reduced,
            objlim=objlim,
            readnoise=readnoise,
            sigclip=sigclip,
            satlevel=satlevel,
            verbose=verbose,
        )
        if not addmask:
            reduced.mask = np.zeros(reduced.shape, dtype=bool)
        if verbose:
            if terminal_logger is not None:
                terminal_logger.add_to_cache("")
            else:
                terminal_output.print_to_terminal("")

        #   Add Header keyword to mark the file as combined
        reduced.meta['cosmics_rm'] = True

        #   Reapply scaling if image was scaled with the exposure time
        if scaled:
            reduced = reduced.divide(exposure * u.second)

        #   Set file name
        basename = base_utilities.get_basename(image.filename)
        file_name = basename + '_cosmic-rm.fit'

        #   Set new file name and path
        image.filename = file_name
        image.path = os.path.join(
            str(image.outpath),
            'cosmics_rm',
            file_name,
        )

        #   Check if the 'cosmics_rm' directory already exits.
        #   If not, create it.
        checks.check_out(os.path.join(str(image.outpath), 'cosmics_rm'))

        #   Save image
        reduced.write(image.path, overwrite=True)


def mk_bg(image, sigma_bkg=5., d2=True, apply_background=True,
          verbose=False):
    """
        Determine background, using photutils

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        sigma_bkg           : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        d2                  : `boolean`, optional
            If True a 2D background will be estimated and subtracted.
            Default is ``True``.

        apply_background    : `boolean`, optional
            If True path and file name will be set to the background
            subtracted images, so that those will automatically be used in
            further processing steps.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    """
    if verbose:
        terminal_output.print_terminal(
            image.filt,
            string="Determine background: {:s} band",
            indent=2,
        )

    #   Load image data
    img = image.read_image()

    #   Set up sigma clipping
    sigma_clip = SigmaClip(sigma=sigma_bkg)

    #   Calculate background RMS
    bkgrms = MADStdBackgroundRMS(sigma_clip=sigma_clip)
    image.std_rms = bkgrms(img.data)

    #   2D background?
    if d2:
        #   Estimate 2D background
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            img.data,
            (50, 50),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )

        #   Remove background
        img_no_bg = img.subtract(bkg.background * u.electron / u.s)

        #   Put metadata back on the image, because it is lost while
        #   subtracting the background
        img_no_bg.meta = img.meta
        img_no_bg.meta['HIERARCH'] = '2D background removed'

        #   Add Header keyword to mark the file as background subtracted
        img_no_bg.meta['NO_BG'] = True

        #   Get median of the background
        bkg_value = bkg.background_median
    else:
        #   Estimate 1D background
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
        bkg_value = mmm_bkg.calc_background(img.data)

        #   Remove background
        img_no_bg = img.subtract(bkg_value)

        #   Put metadata back on the image, because it is lost while
        #   subtracting the background
        img_no_bg.meta = image.meta
        img_no_bg.meta['HIERARCH'] = '1D background removed'

        #   Add Header keyword to mark the file as background subtracted
        img_no_bg.meta['NO_BG'] = True

    #   Define name and save image
    file_name = base_utilities.get_basename(image.filename) + '_nobg.fit'
    outpath = image.outpath / 'nobg'
    checks.check_out(outpath)
    img_no_bg.write(outpath / file_name, overwrite=True)

    #   Set new path and file
    #   -> Background subtracted image will be used in further processing steps
    if apply_background:
        image.path = outpath / file_name
        image.filename = file_name

    #   Add background value to image class
    image.bkg_value = bkg_value


def find_stars(image, sigma_psf, multi_start=5., method='IRAF',
               terminal_logger=None, indent=2):
    """
        Find the stars on the images, using photutils and search and select
        stars for the ePSF stars

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        sigma_psf       : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        multi_start     : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5``.

        method         : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    if terminal_logger is not None:
        terminal_logger.add_to_cache("Identify stars", indent=indent)
    else:
        terminal_output.print_to_terminal("Identify stars", indent=indent)

    #   Load image data
    img = image.read_image()

    #   Get background RMS
    sigma = image.std_rms

    #   Distinguish between different finder options
    if method == 'DAO':
        #   Set up DAO finder
        daofind = DAOStarFinder(
            fwhm=sigma_psf * gaussian_sigma_to_fwhm,
            threshold=multi_start * sigma
        )

        #   Find stars - make table
        tbl_posi_all = daofind(img.data)
    elif method == 'IRAF':
        #   Set up IRAF finder
        iraffind = IRAFStarFinder(
            threshold=multi_start * sigma,
            fwhm=sigma_psf * gaussian_sigma_to_fwhm,
            minsep_fwhm=0.01,
            roundhi=5.0,
            roundlo=-5.0,
            sharplo=0.0,
            sharphi=2.0,
        )

        #   Find stars - make table
        tbl_posi_all = iraffind(img.data)
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}\nExtraction method ({method}) not valid: "
            f"use either IRAF or DAO {style.Bcolors.ENDC}"
        )

    #   Add positions to image class
    image.positions = tbl_posi_all


def check_epsf_stars(image, size=25, min_stars=25, frac_epsf=0.2,
                     terminal_logger=None, strict=True, indent=2):
    """
        Select ePSF stars and check if there are enough

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        size            : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        min_stars       : `float`, optional
            Minimal number of stars required for the ePSF calculations
            Default is ``25``.

        frac_epsf       : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        strict          : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        outstring       : `string`, optional
            Information to be printed to the terminal
    """
    #   Get object positions
    tbl = image.positions

    #   Number of objects
    num_stars = len(tbl)

    #   Get image data
    data = image.get_data()

    #   Combine identification string
    istring = str(image.pd) + '. ' + image.filt

    #   Useful information
    out_string = f"{num_stars} sources identified in the {istring} band image"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            out_string,
            indent=indent + 1,
            style_name='OK',
        )
    else:
        terminal_output.print_to_terminal(
            out_string,
            indent=indent + 1,
            style_name='OK',
        )

    #  Determine sample of stars used for estimating the ePSF
    #   (rm the brightest 1% of all stars because those are often saturated)
    #   Sort list with star positions according to flux
    tbl_sort = tbl.group_by('flux')
    # Determine the 99 percentile
    p99 = np.percentile(tbl_sort['flux'], 99)
    #   Determine the position of the 99 percentile in the position list
    id_p99 = np.argmin(np.absolute(tbl_sort['flux'] - p99))

    #   Based on the input list, set the minimal number of stars
    frac = int(num_stars * frac_epsf)
    #   If the minimal number of stars ('frac') is lower than 'min_stars'
    #   set it to 'min_stars' (the default is 25 as required by the cutout
    #   plots, 25 also appears to be reasonable for a good ePSF)
    if frac < min_stars:
        frac = min_stars

    #   Check if enough stars have been identified
    if (id_p99 - frac < min_stars and strict) or (id_p99 - frac < 1 and not strict):
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({id_p99 - frac}) found "
            f"to determine the ePSF in the {istring} band{style.Bcolors.ENDC}"
        )

    #   Resize table -> limit it to the suitable stars
    tbl_posi = tbl_sort[:][id_p99 - frac:id_p99]

    #   Exclude stars that are too close to the image boarder
    #   Size of the extraction box around each star
    hsize = (size - 1) / 2

    #   New lists with x and y positions
    x = tbl_posi['xcentroid']
    y = tbl_posi['ycentroid']

    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] - 1 - hsize)))

    #   Updated positions table
    tbl_posi = tbl_posi[:][mask]
    num_clean = len(tbl_posi)

    #   Check if there are still enough stars
    if (num_clean < min_stars and strict) or (num_clean < 1 and not strict):
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({num_clean}) for the "
            f"ePSF determination in the {istring} band image. Too many "
            "potential ePSF stars have been removed, because they are too "
            "close to the image border. Check first that enough stars have "
            "been identified, using the starmap_?.pdf files.\n If that is "
            "the case, shrink extraction region or allow for higher fraction "
            "of ePSF stars (size_epsf) from all identified stars "
            f"(frac_epsf_stars). {style.Bcolors.ENDC}"
        )

    #   Find all potential ePSF stars with close neighbors
    dist_min = size

    #   Define and fill new arrays
    x1 = tbl_sort['xcentroid']
    y1 = tbl_sort['ycentroid']
    x2 = tbl_posi['xcentroid']
    y2 = tbl_posi['ycentroid']
    nmax = np.max((len(x1), len(x2)))
    xall = np.zeros((nmax, 2))
    yall = np.zeros((nmax, 2))
    xall[0:len(x1), 0] = x1
    xall[0:len(x2), 1] = x2
    yall[0:len(y1), 0] = y1
    yall[0:len(y2), 1] = y2

    id_p99 = correlate.newsrcor(
        xall,
        yall,
        dist_min,
        option=3,
        silent=True,
    )[1]

    #   Determine multiple entries -> stars that are contaminated
    id_p99_mult = [ite for ite, count in Counter(id_p99).items() if count > 1]

    #   Determine unique entries -> stars that are not contaminated
    id_p99_uniq = [ite for ite, count in Counter(id_p99).items() if count == 1]
    num_clean = len(id_p99_uniq)

    #   Remove ePSF stars with close neighbors from the corresponding table
    tbl_posi.remove_rows(id_p99_mult)

    #   Check if there are still enough stars
    if (num_clean < min_stars and strict) or (num_clean < 1 and not strict):
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({num_clean}) for the "
            f"ePSF determination in the {istring} band image. Too many "
            "potential ePSF stars have been removed, because other "
            "stars are in the extraction region. Check first that enough "
            "stars have been identified, using the starmap_?.pdf files.\n"
            "If that is the case, shrink extraction region or allow for "
            "higher fraction of ePSF stars (size_epsf) from all identified "
            f"stars (frac_epsf_stars). {style.Bcolors.ENDC}"
        )

    #   Add ePSF stars to image class
    image.positions_epsf = tbl_posi


def mk_epsf(image, size=25, oversampling=2, maxiters=7,
            min_stars=25, multi=True, terminal_logger=None, indent=2):
    """
        Main function to determine the ePSF, using photutils

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        size            : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        oversampling    : `integer`, optional
            ePSF oversampling factor
            Dewfault is ``2``.

        maxiters        : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        min_stars       : `float`, optional
            Minimal number of stars required for the ePSF calculations
            Default is ``25``.

        multi           : `boolean`, optional
            If True multiprocessing is used for plotting.
            Default is ``True``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    #   Get image data
    data = image.get_data()

    #   Get ePSF star positions
    tbl_posi = image.positions_epsf

    #   Number of ePSF stars
    num_fit = len(tbl_posi)

    #   TODO: Add check if minimal number of EPSF stars have been identified
    if num_fit < min_stars:
        terminal_logger.add_to_cache(
            f"The number of ePSF stars is less than required."
            f"{num_fit} ePSF stars available. {min_stars} were requested.",
            indent=indent,
            style_name='WARNING',
        )

    #   Get object name
    nameobj = image.objname

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            "Determine the point spread function",
            indent=indent
        )
        terminal_logger.add_to_cache(
            f"{num_fit} bright stars used",
            indent=indent + 1,
            style_name='OK',
        )
    else:
        terminal_output.print_to_terminal(
            "Determine the point spread function",
            indent=indent
        )
        terminal_output.print_to_terminal(
            f"{num_fit} bright stars used",
            indent=indent + 1,
            style_name='OK',
        )

    #   Create new table with the names required by "extract_stars"
    stars_tbl = Table()
    stars_tbl['x'] = tbl_posi['xcentroid']
    stars_tbl['y'] = tbl_posi['ycentroid']

    #   Put image into NDData container (required by "extract_stars")
    nddata = NDData(data=data)

    #   Extract cutouts of the selected stars
    stars = extract_stars(nddata, stars_tbl, size=size)

    #   Combine plot identification string
    string = f'img-{image.pd}-{image.filt}'

    #   Get output directory
    outdir = image.outpath.name

    #   Plot the brightest ePSF stars
    if multi:
        p = mp.Process(
            target=plot.plot_cutouts,
            args=(outdir, stars, string),
            kwargs={'nameobj': nameobj, }
        )
        p.start()
    else:
        plot.plot_cutouts(
            outdir,
            stars,
            string,
            nameobj=nameobj,
            terminal_logger=terminal_logger,
        )

    #   Build the ePSF (set oversampling and max. number of iterations)
    epsf_builder = EPSFBuilder(
        oversampling=oversampling,
        maxiters=maxiters,
        progress_bar=False,
    )
    epsf, fitted_stars = epsf_builder(stars)

    #   Add ePSF and fitted stars to image class
    image.epsf = epsf
    image.fitted_stars = fitted_stars


def epsf_extract(image, sigma_psf, sigma_bkg=5., use_init_guesses=True,
                 method_finder='IRAF', size_epsf=25., multi=5.0,
                 multi_grouper=2.0, strict_cleaning=True, terminal_logger=None,
                 rmbackground=False, indent=2):
    """
        Main function to perform the eEPSF photometry, using photutils

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        sigma_psf           : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_bkg           : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        use_init_guesses    : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method_finder       : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        size_epsf           : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        multi               : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper       : `float`, optional
            Multiplier for the DAO grouper
            Default is ``2.0``.

        strict_cleaning     : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        rmbackground        : `boolean`, optional
            If True the background will be estimated and considered.
            Default is ``False``. -> It is expected that the background
            was removed before.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2`.

    """
    #   Get output path
    out_path = image.outpath

    #   Check output directories
    checks.check_out(
        out_path,
        out_path / 'tables',
    )

    #   Get image data
    data = image.get_data()

    #   Get filter
    filt = image.filt

    #   Get already identified objects (position and flux)
    if use_init_guesses:
        try:
            #   Get position information
            positions_flux = image.positions
            init_guesses = Table(
                names=['x_0', 'y_0', 'flux_0'],
                data=[
                    positions_flux['xcentroid'],
                    positions_flux['ycentroid'],
                    positions_flux['flux'],
                ]
            )
        except RuntimeError:
            #   If positions and fluxes are not available,
            #   those will need to be determined. Set
            #   switch accordingly.
            use_init_guesses = False

    #   Set output and plot identification string
    istring = str(image.pd) + '-' + filt

    #   Get background RMS
    sigma = image.std_rms

    #   Get ePSF
    epsf = image.epsf

    outstr = f"Performing the actual PSF photometry ({istring} image)"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(outstr, indent=indent)
    else:
        terminal_output.print_to_terminal(outstr, indent=indent)

    #  Set up all necessary classes
    if method_finder == 'IRAF':
        #   IRAF finder
        finder = IRAFStarFinder(
            threshold=multi * sigma,
            fwhm=sigma_psf * gaussian_sigma_to_fwhm,
            minsep_fwhm=0.01,
            roundhi=5.0,
            roundlo=-5.0,
            sharplo=0.0,
            sharphi=2.0,
        )
    elif method_finder == 'DAO':
        #   DAO finder
        finder = DAOStarFinder(
            fwhm=sigma_psf * gaussian_sigma_to_fwhm,
            threshold=multi * sigma,
            exclude_border=True,
        )
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nExtraction method ({method_finder}) "
            f"not valid: use either IRAF or DAO {style.Bcolors.ENDC}"
        )
    #   Fitter used
    fitter = LevMarLSQFitter()

    #   Size of the extraction region
    if size_epsf % 2 == 0:
        sizepho = size_epsf + 1
    else:
        sizepho = size_epsf

    #   Number of iterations
    niter = 1

    #   Set up sigma clipping
    if rmbackground:
        sigma_clip = SigmaClip(sigma=sigma_bkg)
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
    else:
        mmm_bkg = None

    try:
        #   DAO grouper
        daogroup = DAOGroup(multi_grouper * sigma_psf * gaussian_sigma_to_fwhm)

        #  Set up the overall class to extract the data
        photometry = IterativelySubtractedPSFPhotometry(
            finder=finder,
            group_maker=daogroup,
            bkg_estimator=mmm_bkg,
            psf_model=epsf,
            fitter=fitter,
            niters=niter,
            fitshape=(sizepho, sizepho),
            aperture_radius=(sizepho - 1) / 2
        )

        #   Extract the photometry and make a table
        if use_init_guesses:
            result_tbl = photometry(image=data, init_guesses=init_guesses)
        else:
            result_tbl = photometry(image=data)
    except RuntimeError as e:
        if multi_grouper != 1.:
            terminal_output.print_terminal(
                indent=indent,
                string="IterativelySubtractedPSFPhotometry failed. "
                       "Will try again with 'multi_grouper' set to 1...",
                style_name='WARNING',
            )
            multi_grouper = 1.
            #   DAO grouper
            daogroup = DAOGroup(
                multi_grouper * sigma_psf * gaussian_sigma_to_fwhm
            )

            #  Set up the overall class to extract the data
            photometry = IterativelySubtractedPSFPhotometry(
                finder=finder,
                group_maker=daogroup,
                bkg_estimator=mmm_bkg,
                psf_model=epsf,
                fitter=fitter,
                niters=niter,
                fitshape=(sizepho, sizepho),
                aperture_radius=(sizepho - 1) / 2
            )

            #   Extract the photometry and make a table
            if use_init_guesses:
                result_tbl = photometry(image=data, init_guesses=init_guesses)
            else:
                result_tbl = photometry(image=data)
        else:
            terminal_output.print_to_terminal(
                "IterativelySubtractedPSFPhotometry failed. No recovery possible.",
                indent=0,
                style_name='ERROR'
            )
            raise e

    #   Check if result table contains a 'flux_unc' column
    #   For some reason, it's missing for some extractions....
    if 'flux_unc' not in result_tbl.colnames:
        #   Calculate a very, very rough approximation of the uncertainty
        #   by means of the actual extraction result 'flux_fit' and the
        #   early estimate 'flux_0'
        est_unc = np.absolute(
            result_tbl['flux_fit'] - result_tbl['flux_0']
        )
        result_tbl.add_column(est_unc, name='flux_unc')

    #   Clean output for objects with negative uncertainties
    try:
        spoiled_fits = np.where(result_tbl['flux_fit'].data < 0.)
        result_tbl.remove_rows(spoiled_fits)
        num_spoiled = np.size(spoiled_fits)
        if strict_cleaning:
            spoiled_fits = np.where(result_tbl['flux_unc'].data < 0.)
            num_spoiled += len(spoiled_fits)
            result_tbl.remove_rows(spoiled_fits)
    except:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of negative "
            f"uncertainties... {style.Bcolors.ENDC}"
        )

    #   Clean output for objects with negative pixel coordinates
    try:
        spoiled_fits = np.where(result_tbl['x_fit'].data < 0.)
        num_spoiled += np.size(spoiled_fits)
        result_tbl.remove_rows(spoiled_fits)
        spoiled_fits = np.where(result_tbl['y_fit'].data < 0.)
        num_spoiled += np.size(spoiled_fits)
        result_tbl.remove_rows(spoiled_fits)
    except:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of negative pixel "
            f"coordinates... {style.Bcolors.ENDC}"
        )

    if num_spoiled != 0:
        out_str = f"{num_spoiled} objects removed because of poor fit quality"
        if terminal_logger is not None:
            terminal_logger.add_to_cache(out_str, indent=indent + 1)
        else:
            terminal_output.print_to_terminal(out_str, indent=indent + 1)

    try:
        nstars = len(result_tbl['flux_fit'].data)
    except:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nTable produced by "
            "IterativelySubtractedPSFPhotometry is empty after cleaning up "
            "of objects with negative pixel coordinates and negative "
            f"uncertainties {style.Bcolors.ENDC}"
        )

    out_str = f"{nstars} good stars extracted from the image"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(out_str, indent=indent + 1, style_name='OK')
    else:
        terminal_output.print_to_terminal(
            out_str,
            indent=indent + 1,
            style_name='OK'
        )

    #   Remove objects that are too close to the image edges
    result_tbl = utilities.rm_edge_objects(
        result_tbl,
        data,
        int((sizepho - 1) / 2),
        terminal_logger=terminal_logger,
    )

    #   Write table
    filename = 'table_photometry_{}_PSF.dat'.format(istring)
    result_tbl.write(
        out_path / 'tables' / filename,
        format='ascii',
        overwrite=True,
    )

    #  Make residual image
    residual_image = photometry.get_residual_image()

    #   Add photometry and residual image to image class
    image.photometry = result_tbl
    image.residual_image = residual_image


def compute_phot_error(flux_variance, ap_area, nsky, stdev, gain=1.0):
    """
        This function is largely borrowed from the Space Telescope Science
        Institute's wfc3_photometry package:

        https://github.com/spacetelescope/wfc3_photometry

        It computes the flux errors using the DAOPHOT style computation:

        err = sqrt (Poisson_noise / gain
            + ap_area * stdev**2
            + ap_area**2 * stdev**2 / nsky)

        Parameters
        ----------
        flux_variance       : `numpy.ndarray`
            Extracted aperture flux data or the error^2 of the extraction
            if available -> proxy for the Poisson noise

        ap_area             : `float`
            Photometric aperture area

        nsky                : `fLoat`
            Sky annulus area

        stdev               : `numpy.ndarray`
            Uncertainty in the sky measurement

        gain               : `float`, optional
            Electrons per ADU
            Default is ``1.0``. Usually we already work with gain corrected
            data.
    """

    #   Calculate flux error as above
    bg_variance_terms = (ap_area * stdev ** 2.) * (1. + ap_area / nsky)
    variance = flux_variance / gain + bg_variance_terms
    flux_error = variance ** .5

    return flux_error


def define_apertures(image, r, r_in, r_out, r_unit):
    """
        Define stellar and background apertures

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        r                   : `float`
            Radius of the stellar aperture

        r_in                : `float`
            Inner radius of the background annulus

        r_out               : `float`
            Outer radius of the background annulus

        r_unit              : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        Returns
        -------
        aperture            : `photutils.aperture.CircularAperture`
            Stellar aperture

        annulus_aperture    : `photutils.aperture.CircularAnnulus`
            Background annulus
    """
    #   Get position information
    tbl = image.positions

    #   Extract positions and prepare a position list
    try:
        lst1 = tbl['x_fit']
        lst2 = tbl['y_fit']
    except:
        lst1 = tbl['xcentroid']
        lst2 = tbl['ycentroid']
    positions = list(zip(lst1, lst2))

    #   Check unit of radii
    if r_unit not in ['pixel', 'arcsec']:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nUnit of the aperture radii not valid: "
            f"set it either to pixel or arcsec {style.Bcolors.ENDC}"
        )

    #   Convert radii in arcsec to pixel
    #   (this part is prone to errors and needs to be rewritten)
    pixscale = image.pixscale
    if pixscale is not None and r_unit == 'arcsec':
        r = r / pixscale
        r_in = r_in / pixscale
        r_out = r_out / pixscale

    #   Make stellar aperture
    aperture = CircularAperture(positions, r=r)

    #   Make background annulus
    annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    return aperture, annulus_aperture


def background_simple(image, annulus_aperture):
    """
        Calculate background from annulus

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        annulus_aperture    : `photutils.aperture.CircularAnnulus`
            Background annulus

        Returns
        -------
        bkg_median          : `float`
            Median of the background

        bkg_stdev           : `float`
            Standard deviation of the background
    """
    bkg_median = []
    bkg_stdev = []

    #   Calculate mask from background annulus
    annulus_masks = annulus_aperture.to_mask(method='center')

    #   Loop over all masks
    for mask in annulus_masks:
        #   Extract annulus data
        annulus_data = mask.multiply(image.get_data())

        #   Convert annulus data to 1D
        annulus_data_1d = annulus_data[mask.data > 0]

        #   Sigma clipping
        _, median_sigclip, median_stdev = sigma_clipped_stats(annulus_data_1d)

        #   Add to list
        bkg_median.append(median_sigclip)
        bkg_stdev.append(median_stdev)

    #   Convert to numpy array
    bkg_median = np.array(bkg_median)
    bkg_stdev = np.array(bkg_stdev)

    return bkg_median, bkg_stdev


def aperture_extract(image, r, r_in, r_out, r_unit='pixel', bg_simple=False,
                     plotaper=False, terminal_logger=None, indent=2):
    """
        Perform aperture photometry using the photutils aperture package

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        r               : `float`
            Radius of the stellar aperture

        r_in            : `float`
            Inner radius of the background annulus

        r_out           : `float`
            Outer radius of the background annulus

        r_unit          : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        bg_simple       : `boolean`, optional
            If True the background will be extract by a simple algorithm that
            calculates the median within the background annulus. If False the
            background will be extracted using
            photutils.aperture.aperture_photometry.
            Default is ``False``.

        plotaper        : `boolean`, optional
            IF true a plot showing the apertures in relation to image is
            created.
            Default is ``False``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        outstring       : `string`, optional
            Information to be printed to the terminal

            img_err     - numpy.ndarray or None
                          Error array for 'image'
    """
    #   Load image data and uncertainty
    img = image.read_image()
    data = img.data
    err = img.uncertainty.array

    #   Get filter
    filt = image.filt

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            f"Performing aperture photometry ({filt} image)",
            indent=indent,
        )
    else:
        terminal_output.print_to_terminal(
            f"Performing aperture photometry ({filt} image)",
            indent=indent,
        )

    ###
    #   Define apertures
    #
    aperture, annulus_aperture = define_apertures(
        image,
        r,
        r_in,
        r_out,
        r_unit,
    )

    ###
    #   Extract photometry
    #
    #   Extract aperture
    phot = aperture_photometry(data, aperture, mask=img.mask, error=err)

    #   Extract background and calculate median - extract background aperture
    if bg_simple:
        bkg_median, bg_err = background_simple(image, annulus_aperture)

        #   Add median background to the output table
        phot['annulus_median'] = bkg_median

        #   Calculate background for the apertures add to the output table
        phot['aper_bkg'] = bkg_median * aperture.area
    else:
        bkg_phot = aperture_photometry(
            data,
            annulus_aperture,
            mask=img.mask,
            error=err,
        )

        #   Calculate aperture background and the corresponding error
        phot['aper_bkg'] = bkg_phot['aperture_sum'] * aperture.area \
            / annulus_aperture.area

        phot['aper_bkg_err'] = bkg_phot['aperture_sum_err'] * aperture.area \
            / annulus_aperture.area

        bg_err = phot['aper_bkg_err']

    #   Subtract background from aperture flux and add it to the
    #   output table
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']

    #   Define flux column
    #   (necessary to have the same column names for aperture and PSF
    #   photometry)
    phot['flux_fit'] = phot['aper_sum_bkgsub']

    # Error estimate
    if err is not None:
        err_column = phot['aperture_sum_err']
    else:
        err_column = phot['flux_fit'] ** 0.5

    phot['flux_unc'] = compute_phot_error(
        err_column,
        aperture.area,
        annulus_aperture.area,
        bg_err,
    )

    #   Rename position columns
    phot.rename_column('xcenter', 'x_fit')
    phot.rename_column('ycenter', 'y_fit')

    #   Convert distance/radius to the border to pixel.
    if r_unit == 'pixel':
        r_border = int(r_out)
    elif r_unit == 'arcsec':
        pixscale = image.pixscale
        r_border = int(round(r_out / pixscale))
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nException in aperture_extract(): '"
            f"\n'r_unit ({r_unit}) not known -> Exit {style.Bcolors.ENDC}"
        )

    #   Remove objects that are too close to the image edges
    phot = utilities.rm_edge_objects(
        phot,
        data,
        r_border,
        terminal_logger=terminal_logger,
    )

    #   Remove negative flux values as they are not physical
    flux = np.array(phot['flux_fit'])
    mask = np.where(flux > 0.).ravel()
    phot = phot[mask]

    #   Add photometry to image class
    image.photometry = phot

    ###
    #   Plot star map with aperture overlay
    #
    if plotaper:
        plot.plot_apertures(
            image.outpath.name,
            data,
            aperture,
            annulus_aperture,
            filt,
        )

    #   Number of stars
    nstars = len(flux)

    #   Useful info
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            f"{nstars} good stars extracted from the image",
            indent=indent + 1,
            style_name='OK',
        )
    else:
        terminal_output.print_to_terminal(
            f"{nstars} good stars extracted from the image",
            indent=indent + 1,
            style_name='OK',
        )


def correlate_ensemble_img(img_ensemble, dcr=3., option=1, maxid=1,
                           ref_obj=[], nmissed=1, bfrac=1.0, s_ref_obj=True,
                           correl_method='astropy', seplimit=2. * u.arcsec):
    """
        Correlate object positions from all stars in the image ensemble to
        identify those objects that are visible on all images

        Parameters
        ----------
        img_ensemble        : `image ensemble`
            Ensemble of images, e.g., taken in one filter

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        ref_obj             : `list` of `integer`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``[]``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_ref_obj           : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    """
    #   Number of images
    n_images = img_ensemble.nfiles

    #   Set proxy image position IDs
    arr_img_ids = np.arange(n_images)

    terminal_output.print_to_terminal(
        f"Correlate results from the images ({arr_img_ids})",
        indent=1,
    )

    #   Get WCS
    w = img_ensemble.wcs

    #   Extract pixel positions of the objects
    #   Returns list of lists for x and y
    x, y, n_objects = img_ensemble.get_object_positions_pixel()

    # #   Correlate the object positions from the images
    # #   -> find common objects
    ind_sr, ref_ori_new, reject, count = correlate.correlate_datasets(
        x,
        y,
        w,
        n_objects,
        n_images,
        ref_ori=img_ensemble.reference_image_id,
        ref_obj=ref_obj,
        nmissed=nmissed,
        s_ref_obj=s_ref_obj,
        seplimit=seplimit,
        dcr=dcr,
        bfrac=bfrac,
        option=option,
        maxid=maxid,
        correl_method=correl_method,
    )

    #   Remove "bad" images from image IDs
    arr_img_ids = np.delete(arr_img_ids, reject, 0)

    #   Remove images that are rejected (bad images) during the correlation process.
    img_ensemble.image_list = [img_ensemble.image_list[i] for i in arr_img_ids]
    # img_ensemble.image_list = np.delete(img_list, reject)
    img_ensemble.nfiles = len(arr_img_ids)
    img_ensemble.reference_image_id = ref_ori_new

    #   Limit the photometry tables to common objects.
    for j, image in enumerate(img_ensemble.image_list):
        image.photometry = image.photometry[ind_sr[j, :]]


def correlate_ensemble(img_container, filt_list, dcr=3., option=1, maxid=1,
                       ref_ori=0, ref_obj=[], nmissed=1, bfrac=1.0,
                       s_ref_obj=True, correl_method='astropy',
                       seplimit=2. * u.arcsec):
    """
        Correlate star lists from the stacked images of all filters to find
        those stars that are visible on all images -> write calibrated CMD

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        filt_list           : `list` of `string`
            List with filter identifiers.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        TODO: Remove ref_ori because it is already on the ensemble. Maybe here this is actually usefull...
        ref_ori             : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        ref_obj             : `list` of `integer`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``[]``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_ref_obj           : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    """
    terminal_output.print_to_terminal(
        "Correlate results from image ensembles",
        indent=1,
    )

    #   Get image ensembles
    ensemble_dict = img_container.get_ensembles(filt_list)
    ensemble_keys = list(ensemble_dict.keys())

    #   Define variables
    nobj_list = []
    x = []
    y = []
    w = []

    #   Number of objects in each table/image
    for ensemble in ensemble_dict.values():
        w.append(ensemble.wcs)

        _x = ensemble.image_list[0].photometry['x_fit']
        x.append(_x)
        y.append(ensemble.image_list[0].photometry['y_fit'])
        nobj_list.append(len(_x))

    #   Max. number of objects
    n_objects = np.max(nobj_list)

    #   Number of image ensembles
    n_ensembles = len(x)

    #   Correlate the object positions from the images
    #   -> find common objects
    ind_sr, ref_ori_new, reject, count = correlate.correlate_datasets(
        x,
        y,
        w[ref_ori],
        n_objects,
        n_ensembles,
        dataset_type='ensemble',
        ref_ori=ref_ori,
        ref_obj=ref_obj,
        nmissed=nmissed,
        s_ref_obj=s_ref_obj,
        seplimit=seplimit,
        cleanup_advanced=False,
        dcr=dcr,
        bfrac=bfrac,
        option=option,
        maxid=maxid,
        correl_method=correl_method,
    )

    #   Remove "bad"/rejected ensembles
    for ject in reject:
        ensemble_dict.pop(ensemble_keys[ject])

    #   Limit the photometry tables to common objects.
    for j, ensemble in enumerate(ensemble_dict.values()):
        for image in ensemble.image_list:
            image.photometry = image.photometry[ind_sr[j, :]]


#   TODO: Check if the following routine is still necessary? -> seems to be useful
# def correlate_preserve_calibs(img_ensemble, filter_list,
#                               calib_method='APASS', mag_range=(0., 18.5),
#                               vizier_dict=None, calib_file=None, dcr=3,
#                               option=1, verbose=False, maxid=1, reference_image_id=0,
#                               nmissed=1, bfrac=1.0, s_refOBJ=True,
#                               plot_test=True, correl_method='astropy',
#                               seplimit=2. * u.arcsec):
#     """
#         Correlate results from all images, while preserving the calibration
#         stars
#
#         Parameters
#         ----------
#         img_ensemble        : `image.ensemble` object
#             Ensemble class object with all image data taken in a specific
#             filter
#
#         filter_list         : `list` with `strings`
#             Filter list
#
#         calib_method       : `string`, optional
#             Calibration method
#             Default is ``APASS``.
#
#         mag_range           : `tupel` or `float`, optional
#             Magnitude range
#             Default is ``(0.,18.5)``.
#
#         vizier_dict         : `dictionary` or None, optional
#             Identifiers of catalogs, containing calibration data
#             Default is ``None``.
#
#         calib_file          : `string`, optional
#             Path to the calibration file
#             Default is ``None``.
#
#         dcr             : `float`, optional
#             Maximal distance between two objects in Pixel
#             Default is ``3``.
#
#         option          : `integer`, optional
#             Option for the srcor correlation function
#             Default is ``1``.
#
#         verbose         : `boolean`, optional
#             If True additional output will be printed to the command line.
#             Default is ``False``.
#
#         maxid               : `integer`, optional
#             Max. number of allowed identical cross identifications between
#             objects from a specific origin
#             Default is ``1``.
#
#         reference_image_id  : `integer`, optional
#             ID of the reference origin
#             Default is ``0``.
#
#         nmissed             : `integer`, optional
#             Maximum number an object is allowed to be not detected in an
#             origin. If this limit is reached the object will be removed.
#             Default is ``i`.
#
#         bfrac               : `float`, optional
#             Fraction of low quality source position origins, i.e., those
#             origins, for which it is expected to find a reduced number of
#             objects with valid source positions.
#             Default is ``1.0``.
#
#         s_refOBJ            : `boolean`, optional
#             If ``False`` also reference objects will be rejected, if they do
#             not fulfill all criteria.
#             Default is ``True``.
#
#         plot_test       : `boolean`, optional
#             If True only the masterplot for the reference image will
#             be created.
#             Default is ``True``.
#
#         correl_method       : `string`, optional
#             Correlation method to be used to find the common objects on
#             the images.
#             Possibilities: ``astropy``, ``own``
#             Default is ``astropy``.
#
#         seplimit            : `astropy.units`, optional
#             Allowed separation between objects.
#             Default is ``2.*u.arcsec``.
#     """
#     ###
#     #   Load calibration data
#     #
#     calib_tbl, col_names, ra_unit = calib.load_calib(
#         img_ensemble.image_list[reference_image_id],
#         filter_list,
#         calib_method=calib_method,
#         mag_range=mag_range,
#         vizier_dict=vizier_dict,
#         calib_file=calib_file,
#     )
#
#     #   Number of calibration stars
#     n_calib = len(calib_tbl)
#
#     if n_calib == 0:
#         raise Exception(
#             f"{style.Bcolors.FAIL} \nNo match between calibrations stars and "
#             f"the\n extracted stars detected. -> EXIT {style.Bcolors.ENDC}"
#         )
#
#     ###
#     #   Find IDs of calibration stars to ensure they are not deleted in
#     #   the correlation process
#     #
#     #   Lists for IDs, and xy coordinates
#     calib_IDs = []
#     calib_xs = []
#     calib_ys = []
#
#     #   Loop over all calibration stars
#     for k in range(0, n_calib):
#         #   Find the calibration star
#         inds_obj, ref_count, x_obj, y_obj = correlate.posi_obj_srcor_img(
#             img_ensemble.image_list[reference_image_id],
#             calib_tbl[col_names['ra']].data[k],
#             calib_tbl[col_names['dec']].data[k],
#             img_ensemble.wcs,
#             dcr=dcr,
#             option=option,
#             ra_unit=ra_unit,
#             verbose=verbose,
#         )
#         if verbose:
#             terminal_output.print_terminal()
#
#         #   Add ID and coordinates of the calibration star to the lists
#         if ref_count != 0:
#             calib_IDs.append(inds_obj[1][0])
#             calib_xs.append(x_obj)
#             calib_ys.append(y_obj)
#     terminal_output.print_terminal(
#         len(calib_IDs),
#         indent=3,
#         string="{:d} matches",
#         style_name='OKBLUE',
#     )
#     terminal_output.print_terminal()
#
#     ###
#     #   Correlate the results from all images
#     #
#     correlate_ensemble_img(
#         img_ensemble,
#         dcr=dcr,
#         option=option,
#         maxid=maxid,
#         ref_ori=reference_image_id,
#         ref_obj=calib_IDs,
#         nmissed=nmissed,
#         bfrac=bfrac,
#         s_ref_obj=s_refOBJ,
#         correl_method=correl_method,
#         seplimit=seplimit,
#     )
#
#     ###
#     #   Plot image with the final positions overlaid (final version)
#     #
#     utilities.prepare_and_plot_starmap_final_3(
#         img_ensemble,
#         calib_xs,
#         calib_ys,
#         plot_test=plot_test,
#     )


def correlate_preserve_variable(img_ensemble, ra_obj, dec_obj, dcr=3.,
                                option=1, maxid=1, reference_image_id=0,
                                nmissed=1, bfrac=1.0, protect_reference_obj=True,
                                correl_method='astropy',
                                seplimit=2. * u.arcsec, verbose=False,
                                plot_test=True):
    """
        Correlate results from all images, while preserving the variable
        star

        Parameters
        ----------
        img_ensemble    : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        ra_obj          : `float`
            Right ascension of the object

        dec_obj         : `float`
            Declination of the object

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        reference_image_id  : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        protect_reference_obj            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        plot_test       : `boolean`, optional
            If True only the masterplot for the reference image will
            be created.
            Default is ``True``.
    """
    ###
    #   Find position of the variable star I
    #
    terminal_output.print_terminal(
        indent=1,
        string="Identify the variable star",
    )

    variable_id, count, x_obj, y_obj = correlate.identify_star_in_dataset(
        img_ensemble.image_list[reference_image_id].photometry['x_fit'],
        img_ensemble.image_list[reference_image_id].photometry['y_fit'],
        ra_obj,
        dec_obj,
        img_ensemble.wcs,
        seplimit=seplimit,
        dcr=dcr,
        option=option,
        verbose=verbose,
    )

    ###
    #   Check if variable star was detected I
    #
    if count == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \tERROR: The variable object was not "
            f"detected in the reference image.\n\t-> EXIT{style.Bcolors.ENDC}"
        )

    ###
    #   Correlate the stellar positions from the different filter
    #
    correlate_ensemble_img(
        img_ensemble,
        dcr=dcr,
        option=option,
        maxid=maxid,
        ref_obj=[int(variable_id)],
        nmissed=nmissed,
        bfrac=bfrac,
        s_ref_obj=protect_reference_obj,
        correl_method=correl_method,
        seplimit=seplimit,
    )

    ###
    #   Find position of the variable star II
    #
    terminal_output.print_terminal(
        indent=1,
        string="Re-identify the variable star",
    )

    variable_id, count, x_obj, y_obj = correlate.identify_star_in_dataset(
        img_ensemble.image_list[reference_image_id].photometry['x_fit'],
        img_ensemble.image_list[reference_image_id].photometry['y_fit'],
        ra_obj,
        dec_obj,
        img_ensemble.wcs,
        seplimit=seplimit,
        dcr=dcr,
        option=option,
        verbose=verbose,
    )

    ###
    #   Check if variable star was detected II
    #
    if count == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \tERROR: The variable was not detected "
            f"in the reference image.\n\t-> EXIT{style.Bcolors.ENDC}"
        )

    ###
    #   Plot image with the final positions overlaid (final version)
    #
    utilities.prepare_and_plot_starmap_final_3(
        img_ensemble,
        [x_obj],
        [y_obj],
        plot_test=plot_test,
    )

    #   Add ID of the variable star to the image ensemble
    img_ensemble.variable_id = variable_id


def extract_multiprocessing(img_ensemble, ncores, sigma_psf, sigma_bkg=5.,
                            multi_start=5., size_epsf=25,
                            frac_epsf_stars=0.2,
                            oversampling=2, maxiters=7,
                            epsf_use_init_guesses=True, method='IRAF',
                            multi=5.0, multi_grouper=2.0,
                            strict_cleaning=True, min_eps_stars=25,
                            photometry='PSF', rstars=5., rbg_in=7.,
                            rbg_out=10., r_unit='arcsec', strict_eps=True,
                            search_image=True, plot_ifi=False, plot_test=True):
    """
        Extract flux and object positions using multiprocessing

        Parameters
        ----------
        img_ensemble    : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        ncores          : `integer`
            Number of cores to use during multiprocessing.

        sigma_psf       : `dictionary`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_bkg       : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multi_start     : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf       : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling    : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters        : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.
            Default is ``7``.

        epsf_use_init_guesses   : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method         : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi           : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper   : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars   : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        photometry      : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars          : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in          : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out         : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit          : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps      : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        search_image    : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        plot_ifi        : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test       : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    """
    #   Get filter
    filt = img_ensemble.filt

    ###
    #   Find the stars (via DAO or IRAF StarFinder)
    #
    if not search_image:
        mk_bg(img_ensemble.ref_img, sigma_bkg=sigma_bkg)

        find_stars(
            img_ensemble.ref_img,
            sigma_psf[filt],
            multi_start=multi_start,
            method=method,
        )

    ###
    #   Main loop: Extract stars and info from all images, using
    #              multiprocessing
    #
    #   Initialize multiprocessing object
    executor = utilities.Executor(ncores)

    #   Main loop
    for image in img_ensemble.image_list:
        #   Set positions of the reference image if required
        if not search_image:
            image.positions = img_ensemble.ref_img.positions

        #   Extract photometry
        executor.schedule(
            main_extract,
            args=(
                image,
                sigma_psf[filt],
            ),
            kwargs={
                'multiprocessing': True,
                'sigma_bkg': sigma_bkg,
                'multi_start': multi_start,
                'size_epsf': size_epsf,
                'frac_epsf_stars': frac_epsf_stars,
                'oversampling': oversampling,
                'maxiters': maxiters,
                'epsf_use_init_guesses': epsf_use_init_guesses,
                'method': method,
                'multi': multi,
                'multi_grouper': multi_grouper,
                'strict_cleaning': strict_cleaning,
                'min_eps_stars': min_eps_stars,
                'strict_eps': strict_eps,
                'refid': img_ensemble.reference_image_id,
                'photometry': photometry,
                'rstars': rstars,
                'rbg_in': rbg_in,
                'rbg_out': rbg_out,
                'r_unit': r_unit,
                'search_image': search_image,
                'plot_ifi': plot_ifi,
                'plot_test': plot_test,
            }
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Extraction using multiprocessing failed '
            f'for {filt} :({style.Bcolors.ENDC}'
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    ###
    #   Sort multiprocessing results
    #
    #   Extract results
    res = executor.res

    #   Sort observation times and images & build dictionary for the
    #   tables with the extraction results
    tmp_list = []
    for j in range(0, img_ensemble.nfiles):
        for img in res:
            pd = img.pd
            if pd == j:
                tmp_list.append(img)

    img_ensemble.image_list = tmp_list


def main_extract(image, sigma_psf, multiprocessing=False, sigma_bkg=5.,
                 multi_start=5., size_epsf=25, frac_epsf_stars=0.2,
                 oversampling=2, maxiters=7,
                 epsf_use_init_guesses=True, method='IRAF', multi=5.0,
                 multi_grouper=2.0, strict_cleaning=True,
                 min_eps_stars=25, refid=0, photometry='PSF',
                 rstars=5., rbg_in=7., rbg_out=10., r_unit='arcsec',
                 strict_eps=True, search_image=True, rmcos=False,
                 objlim=5., readnoise=8., sigclip=4.5, satlevel=65535.,
                 plot_ifi=False, plot_test=True):
    """
        Main function to extract the information from the individual images

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        sigma_psf               : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        multiprocessing         : `boolean`, optional
            If True, the routine is set up to meet the requirements of
            multiprocessing, such as returning results and delayed
            output to the terminal.

        sigma_bkg               : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5``.

        multi_start             : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf               : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars         : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling            : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters                : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        epsf_use_init_guesses   : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method                  : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi                   : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper           : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning         : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars           : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        refid                   : `integer`, optional
            ID of the reference image
            Default is ``0``.

        photometry              : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars                  : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in                  : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out                 : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit                  : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps              : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        search_image            : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        rmcos                   : `bool`
            If True cosmic rays will be removed from the image.
            Default is ``False``.

        objlim                  : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        readnoise               : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigclip                 : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        satlevel                : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        plot_ifi                : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test               : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    """
    ###
    #   Initialize output class in case of multiprocessing
    #
    if multiprocessing:
        log_terminal = terminal_output.TerminalLog()
        log_terminal.add_to_cache(f"Image: {image.pd}", style_name='UNDERLINE')
    else:
        terminal_output.print_to_terminal(
            f"Image: {image.pd}",
            indent=2,
            style_name='UNDERLINE',
        )
        log_terminal = None

    ###
    #   Remove cosmics (optional)
    #
    if rmcos:
        rm_cosmic(
            image,
            objlim=objlim,
            readnoise=readnoise,
            sigclip=sigclip,
            satlevel=satlevel,
        )

    ###
    #   Estimate and remove background
    #
    mk_bg(image, sigma_bkg=sigma_bkg)

    ###
    #   Find the stars (via DAO or IRAF StarFinder)
    #
    if search_image:
        find_stars(
            image,
            sigma_psf,
            multi_start=multi_start,
            method=method,
            terminal_logger=log_terminal,
        )

    if photometry == 'PSF':
        ###
        #   Check if enough stars have been detected to allow ePSF
        #   calculations
        #
        check_epsf_stars(
            image,
            size=size_epsf,
            min_stars=min_eps_stars,
            frac_epsf=frac_epsf_stars,
            terminal_logger=log_terminal,
            strict=strict_eps,
        )

        ###
        #   Plot images with the identified stars overlaid
        #
        if plot_ifi or (plot_test and image.pd == refid):
            plot.starmap(
                image.outpath.name,
                image.get_data(),
                image.filt,
                image.positions,
                tbl_2=image.positions_epsf,
                label='identified stars',
                label_2='stars used to determine the ePSF',
                rts='initial-img-' + str(image.pd),
                nameobj=image.objname,
                terminal_logger=log_terminal,
            )

        ###
        #   Calculate the ePSF
        #
        mk_epsf(
            image,
            size=size_epsf,
            oversampling=oversampling,
            maxiters=maxiters,
            min_stars=min_eps_stars,
            multi=False,
            terminal_logger=log_terminal,
        )

        ###
        #   Plot the ePSFs
        #
        plot.plot_epsf(
            image.outpath.name,
            {f'img-{image.pd}-{image.filt}': image.epsf},
            terminal_logger=log_terminal,
            nameobj=image.objname,
            indent=2,
        )

        ###
        #   Performing the PSF photometry
        #
        epsf_extract(
            image,
            sigma_psf,
            sigma_bkg=sigma_bkg,
            use_init_guesses=epsf_use_init_guesses,
            method_finder=method,
            size_epsf=size_epsf,
            multi=multi,
            multi_grouper=multi_grouper,
            strict_cleaning=strict_cleaning,
            terminal_logger=log_terminal,
        )

        ###
        #   Plot original and residual image
        #
        plot.plot_residual(
            image.objname,
            {f'{image.pd}-{image.filt}': image.get_data()},
            {f'{image.pd}-{image.filt}': image.residual_image},
            image.outpath.name,
            terminal_logger=log_terminal,
            nameobj=image.objname,
            indent=2,
        )

    elif photometry == 'APER':
        ###
        #   Perform aperture photometry
        #
        if image.pd == refid:
            plotaper = True
        else:
            plotaper = False

        aperture_extract(
            image,
            rstars,
            rbg_in,
            rbg_out,
            r_unit=r_unit,
            plotaper=plotaper,
            terminal_logger=log_terminal,
            indent=3,
        )

    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nExtraction method ({photometry}) not "
            f"valid: use either APER or PSF {style.Bcolors.ENDC}"
        )

    #   Conversion of flux to magnitudes
    uflux_img = unumpy.uarray(
        image.photometry['flux_fit'],
        image.photometry['flux_unc']
    )
    mags = utilities.mag_u_arr(uflux_img)

    image.photometry['mags_fit'] = unumpy.nominal_values(mags)
    image.photometry['mags_unc'] = unumpy.std_devs(mags)

    ###
    #   Plot images with extracted stars overlaid
    #
    if plot_ifi or (plot_test and image.pd == refid):
        utilities.prepare_and_plot_starmap(image, terminal_logger=log_terminal)

    if multiprocessing:
        log_terminal.print_to_terminal('')
    else:
        terminal_output.print_to_terminal('')

    if multiprocessing:
        return image


def extract_flux(img_container, filter_list, name, img_paths, outdir,
                 sigma_psf, wcs_method='astrometry', force_wcs_determ=False,
                 sigma_bkg=5., multi_start=5., size_epsf=25,
                 frac_epsf_stars=0.2, oversampling=2, maxiters=7,
                 epsf_use_init_guesses=True, method='IRAF', multi=5.0,
                 multi_grouper=2.0, strict_cleaning=True, min_eps_stars=25,
                 reference_image_id=0, strict_eps=True, photometry='PSF', rstars=5.,
                 rbg_in=7., rbg_out=10., r_unit='arcsec', rmcos=False,
                 objlim=5., readnoise=8., sigclip=4.5, satlevel=65535.,
                 plot_ifi=False, plot_test=True):
    """
        Extract flux and fill the image container

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        filter_list             : `list` of `string`
            Filter list

        name                    : `string`
            Name of the object

        img_paths               : `dictionary`
            Paths to images: key - filter name; value - path

        outdir                  : `string`
            Path, where the output should be stored.

        sigma_psf               : `dictionary`
            Sigma of the objects PSF, assuming it is a Gaussian

        wcs_method              : `string`, optional
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        force_wcs_determ        : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        sigma_bkg               : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multi_start             : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        size_epsf               : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars         : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling            : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters                : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        epsf_use_init_guesses   : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method                  : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi                   : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper           : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning         : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars           : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        reference_image_id      : `integer`, optional
            ID of the reference image
            Default is ``0``.

        photometry              : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars                  : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in                  : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out                 : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit                  : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``arcsec``.

        strict_eps              : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        rmcos                   : `bool`
            If True cosmic rays will be removed from the image.
            Default is ``False``.

        objlim                  : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        readnoise               : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigclip                 : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        satlevel                : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        plot_ifi                : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test               : `boolean`, optional
            If True a star map plots only for the reference image [reference_image_id] is
            created
            Default is ``True``.
    """
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'tables'),
    )

    ###
    #   Loop over all filter
    #
    for filt in filter_list:
        terminal_output.print_to_terminal(
            f"Analyzing {filt} images",
            style_name='HEADER',
        )

        #   Check input paths
        checks.check_file(img_paths[filt])

        #   Initialize image ensemble object
        img_container.ensembles[filt] = current_ensemble = ImageEnsemble(
            filt,
            name,
            img_paths[filt],
            outdir,
        )

        ###
        #   Find the WCS solution for the image
        #
        try:
            utilities.find_wcs(
                current_ensemble,
                reference_image_id=0,
                method=wcs_method,
                force_wcs_determ=force_wcs_determ,
                indent=3,
            )
        except Exception as e:
            #   Get the WCS from one of the other filters incase, if they have one
            for f in filter_list:
                wcs = getattr(img_container.ensembles[f], 'wcs', None)
                if wcs is not None:
                    current_ensemble.set_wcs(wcs)
                    terminal_output.print_to_terminal(
                        f"WCS could not be determined for filter {filt}"
                        f"The WCS of filter {f} will be used instead."
                        f"This could lead to problems...",
                        indent=1,
                        style_name='WARNING',
                    )
                    break
            else:
                raise RuntimeError('')

        ###
        #   Main extraction
        #
        main_extract(
            current_ensemble.image_list[reference_image_id],
            sigma_psf[filt],
            sigma_bkg=sigma_bkg,
            multi_start=multi_start,
            size_epsf=size_epsf,
            frac_epsf_stars=frac_epsf_stars,
            oversampling=oversampling,
            maxiters=maxiters,
            epsf_use_init_guesses=epsf_use_init_guesses,
            method=method,
            multi=multi,
            multi_grouper=multi_grouper,
            strict_cleaning=strict_cleaning,
            min_eps_stars=min_eps_stars,
            strict_eps=strict_eps,
            photometry=photometry,
            rstars=rstars,
            rbg_in=rbg_in,
            rbg_out=rbg_out,
            r_unit=r_unit,
            rmcos=rmcos,
            objlim=objlim,
            readnoise=readnoise,
            sigclip=sigclip,
            satlevel=satlevel,
            plot_ifi=plot_ifi,
            plot_test=plot_test,
        )

    if photometry == 'PSF':
        ###
        #   Plot the ePSFs
        #
        p = mp.Process(
            target=plot.plot_epsf,
            args=(outdir, img_container.get_ref_epsf(),),
        )
        p.start()

        ###
        #   Plot original and residual image
        #
        p = mp.Process(
            target=plot.plot_residual,
            args=(
                name,
                img_container.get_ref_img(),
                img_container.get_ref_residual_img(),
                outdir,
            ),
            kwargs={
                'nameobj': 'reference image'
            }
        )
        p.start()


def extract_flux_multi(img_container, filter_list, name, img_paths, outdir,
                       sigma_psf, ra_obj, dec_obj, ncores=6,
                       wcs_method='astrometry', force_wcs_determ=False,
                       sigma_bkg=5., multi_start=5., size_epsf=25,
                       frac_epsf_stars=0.2, oversampling=2, maxiters=7,
                       method='IRAF', multi=5.0, multi_grouper=2.0,
                       strict_cleaning=True, min_eps_stars=25, strict_eps=True,
                       photometry='PSF', rstars=5., rbg_in=7., rbg_out=10.,
                       r_unit='arcsec', dcr=3., option=1, maxid=1, reference_image_id=0,
                       nmissed=1, bfrac=1.0, protect_reference_obj=True,
                       correl_method='astropy', seplimit=2. * u.arcsec,
                       verbose=False, search_image=True, plot_ifi=False,
                       plot_test=True):
    """
        Extract flux from multiple images per filter and add results to
        the image container

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        filter_list         : `list` of `string`
            Filter list

        name                : `string`
            Name of the object

        img_paths           : `dictionary`
            Paths to images: key - filter name; value - path

        outdir              : `string`
            Path, where the output should be stored.

        sigma_psf           : `dictionary`
            Sigma of the objects PSF, assuming it is a Gaussian

        ra_obj              : `float`
            Right ascension of the object

        dec_obj             : `float`
            Declination of the object

        ncores              : `integer`, optional
            Number of cores to use for multicore processing
            Default is ``6``.

        wcs_method          : `string`, optional
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        sigma_bkg           : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multi_start         : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf           : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars     : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling        : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters            : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        method              : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi               : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper       : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning     : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars       : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        photometry          : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars              : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in              : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out         : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit              : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps          : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        reference_image_id  : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        protect_reference_obj            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        search_image        : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        plot_ifi            : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test           : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    """
    ###
    #   Check output directories
    #
    checks.check_out(outdir, os.path.join(outdir, 'tables'))

    ###
    #   Check image directories
    #
    checks.check_dir(img_paths)

    #   Outer loop over all filter
    for filt in filter_list:
        terminal_output.print_to_terminal(
            f"Analyzing {filt} images",
            style_name='HEADER',
        )

        #   Initialize image ensemble object
        img_container.ensembles[filt] = ImageEnsemble(
            filt,
            name,
            img_paths[filt],
            outdir,
            reference_image_id=reference_image_id,
        )

        ###
        #   Find the WCS solution for the image
        #
        utilities.find_wcs(
            img_container.ensembles[filt],
            reference_image_id=reference_image_id,
            method=wcs_method,
            force_wcs_determ=force_wcs_determ,
            indent=3,
        )

        ###
        #   Main extraction of object positions and object fluxes
        #   using multiprocessing
        #
        extract_multiprocessing(
            img_container.ensembles[filt],
            ncores,
            sigma_psf,
            sigma_bkg=sigma_bkg,
            multi_start=multi_start,
            size_epsf=size_epsf,
            frac_epsf_stars=frac_epsf_stars,
            oversampling=oversampling,
            maxiters=maxiters,
            method=method,
            multi=multi,
            multi_grouper=multi_grouper,
            strict_cleaning=strict_cleaning,
            min_eps_stars=min_eps_stars,
            strict_eps=strict_eps,
            photometry=photometry,
            rstars=rstars,
            rbg_in=rbg_in,
            rbg_out=rbg_out,
            r_unit=r_unit,
            search_image=search_image,
            plot_ifi=plot_ifi,
            plot_test=plot_test,
        )

        ###
        #   Correlate results from all images, while preserving
        #   the variable star
        #
        correlate_preserve_variable(
            img_container.ensembles[filt],
            ra_obj,
            dec_obj,
            dcr=dcr,
            option=option,
            maxid=maxid,
            reference_image_id=reference_image_id,
            nmissed=nmissed,
            bfrac=bfrac,

            protect_reference_obj=protect_reference_obj,
            verbose=verbose,
            plot_test=plot_test,
            correl_method=correl_method,
            seplimit=seplimit,
        )


def correlate_calibrate(img_container, filter_list, dcr=3, option=1,
                        ref_img=0, calib_method='APASS', vizier_dict=None,
                        calib_file=None, object_id=None, ra_unit=u.deg,
                        dec_unit=u.deg, mag_range=(0., 18.5), tcs=None,
                        derive_tcs=False, plot_sigma=False, photo_type='',
                        region=False, radius=600, data_cluster=False,
                        pm_median=False, max_distance_cluster=6.,
                        find_cluster_para_set=1, correl_method='astropy',
                        seplimit=2. * u.arcsec, r_limit=4., r_unit='arcsec',
                        convert_mags=False, target_filter_system='SDSS'):
    """
        Correlate photometric extraction results from 2 images and calibrate
        the magnitudes.

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        filter_list             : `list` of `strings`
            List with filter names

        dcr                     : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option                  : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        ref_img                 : `integer`, optional
            Reference image ID
            Default is ``0``.

        calib_method           : `string`, optional
            Calibration method
            Default is ``APASS``.

        vizier_dict             : `dictionary` or `None`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``None``.

        calib_file              : `string`, optional
            Path to the calibration file
            Default is ``None``.

        object_id                      : `integer` or `None`, optional
            ID of the object
            Default is ``None``.

        ra_unit                 : `astropy.unit`, optional
            Right ascension unit
            Default is ``u.deg``.

        dec_unit                : `astropy.unit`, optional
            Declination unit
            Default is ``u.deg``.

        mag_range               : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        tcs                     : `dictionary`, optional
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_tcs              : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``

        plot_sigma              : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        photo_type              : `string`, optional
            Applied extraction method. Posibilities: ePSF or APER`
            Default is ``''``.

        region                  : `boolean`, optional
            If True the extracted objects will be filtered such that only
            objects with ``radius`` will be returned.
            Default is ``False``.

        radius                  : `float`, optional
            Radius around the object in arcsec.
            Default is ``600``.

        data_cluster            : `boolean`, optional
            If True cluster in the Gaia distance and proper motion data
            will be identified.
            Default is ``False``.

        pm_median               : `boolean`, optional
            If True only the objects that are close to the median
            proper motion will be returned.
            Default is ``False``.

        max_distance_cluster    : `float`, optional
            Expected maximal distance of the cluster in kpc. Used to
            restrict the parameter space to facilitate an easy
            identification of the star cluster.
            Default is ``6``.

        find_cluster_para_set   : `integer`, optional
            Parameter set used to identify the star cluster in proper
            motion and distance data.

        correl_method           : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit                : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        r_limit                 : `float`, optional
            Radius of the aperture used to derive the limiting magnitude
            Default is ``4``.

        r_unit                  : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``arcsec``.

        convert_mags            : `boolean`, optional
            If True the magnitudes will be converted to another
            filter systems specified in `target_filter_system`.
            Default is ``False``.

        target_filter_system    : `string`, optional
            Photometric system the magnitudes should be converted to
            Default is ``SDSS``.
    """
    ###
    #   Correlate the stellar positions from the different filter
    #
    correlate_ensemble(
        img_container,
        filter_list,
        dcr=dcr,
        option=option,
        correl_method=correl_method,
        seplimit=seplimit,
    )

    ###
    #   Plot image with the final positions overlaid (final version)
    #
    if len(filter_list) > 1:
        utilities.prepare_and_plot_starmap_final(
            img_container,
            filter_list,
        )

    ###
    #   Calibrate the magnitudes
    #
    #   Load calibration information
    calib.deter_calib(
        img_container,
        filter_list,
        calib_method=calib_method,
        dcr=dcr,
        option=option,
        vizier_dict=vizier_dict,
        calib_file=calib_file,
        mag_range=mag_range,
        ra_unit=ra_unit,
        dec_unit=dec_unit,
    )

    #   Apply calibration and perform magnitude transformation
    trans.apply_calib(
        img_container,
        filter_list,
        tcs=tcs,
        derive_tcs=derive_tcs,
        plot_sigma=plot_sigma,
        photo_type=photo_type,
        refid=ref_img,
    )

    ###
    #   Restrict results to specific areas of the image and filter by means
    #   of proper motion and distance using Gaia
    #
    utilities.postprocess_results(
        img_container,
        filter_list,
        id_object=object_id,
        photo_type=photo_type,
        region=region,
        radius=radius,
        data_cluster=data_cluster,
        pm_median=pm_median,
        max_distance_cluster=max_distance_cluster,
        find_cluster_para_set=find_cluster_para_set,
        convert_mags=convert_mags,
        target_filter_system=target_filter_system,
    )

    ###
    #   Determine limiting magnitudes
    #
    utilities.derive_limiting_magnitude(
        img_container,
        filter_list,
        ref_img,
        r_limit=r_limit,
        r_unit=r_unit,
    )


def calibrate_data_mk_lc(img_container, filter_list, ra_obj, dec_obj, nameobj,
                         outdir, transit_time, period, valid_calibs=None,
                         binn=None, tcs=None, derive_tcs=False, reference_image_id=0,
                         calib_method='APASS', vizier_dict=None, calib_file=None,
                         mag_range=(0., 18.5), dcr=3., option=1, maxid=1,
                         nmissed=1, bfrac=1.0, protect_reference_obj=True, photo_type='',
                         correl_method='astropy', seplimit=2. * u.arcsec,
                         verbose=False, plot_test=True, plot_ifi=False,
                         plot_sigma=False):
    """
        Calculate magnitudes, calibrate, and plot light curves

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        filter_list         : `list` of `strings`
            List with filter names

        ra_obj              : `float`
            Right ascension of the object

        dec_obj             : `float`
            Declination of the object

        nameobj             : `string`
            Name of the object

        outdir              : `string`
            Path, where the output should be stored.

        transit_time        : `string`
            Date and time of the transit.
            Format: "yyyy:mm:ddThh:mm:ss" e.g., "2020-09-18T01:00:00"

        period              : `float`
            Period in [d]

        valid_calibs        : `list` of 'list` of `string` or None, optional
            Valid filter combinations to calculate magnitude transformation
            Default is ``None``.

        binn                : `float`, optional
            Binning factor for the light curve.
            Default is ``None```.

        tcs                 : `dictionary`, optional
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_tcs          : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``

        reference_image_id  : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        calib_method        : `string`, optional
            Calibration method
            Default is ``APASS``.

        vizier_dict         : `dictionary` or `None`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``None``.

        calib_file          : `string`, optional
            Path to the calibration file
            Default is ``None``.

        mag_range           : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        protect_reference_obj            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        photo_type          : `string`, optional
            Applied extraction method. Posibilities: ePSF or APER`
            Default is ``''``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        plot_test           : `boolean`, optional
            If True only the star map for the reference image will be
            plotted.
            Default is ``True``.

        plot_ifi            : `boolean`, optional
            If True star map plots for all stars will be created.
            Default is ``False``.

        plot_sigma          : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

    """
    if valid_calibs is None:
        valid_calibs = calibration_data.valid_calibs

    for filt in filter_list:
        terminal_output.print_to_terminal(
            f"Working on filter: {filt}",
            style_name='OKBLUE',
        )

        ###
        #   Try magnitude transformation
        #
        success = False
        #   Loop over valid filter combination for the transformation
        for calib_fil in valid_calibs:
            if filt in calib_fil:
                #   Check if filter combination is valid
                if calib_fil[0] in filter_list and calib_fil[1] in filter_list:

                    if filt == calib_fil[0]:
                        i = 0
                    else:
                        i = 1

                    #   Get object ID
                    obj_id = img_container.ensembles[filt].variable_id

                    ###
                    #   Correlate star positions from the different filter
                    #
                    correlate_ensemble(
                        img_container,
                        calib_fil,
                        dcr=dcr,
                        option=option,
                        maxid=maxid,
                        ref_obj=[obj_id],
                        nmissed=nmissed,
                        bfrac=bfrac,
                        s_ref_obj=protect_reference_obj,
                        correl_method=correl_method,
                        seplimit=seplimit,
                    )

                    ###
                    #   Re-identify position of the variable star
                    #
                    terminal_output.print_terminal(
                        string="Identify the variable star",
                    )

                    obj_id, count, _, _ = correlate.identify_star_in_dataset(
                        img_container.ensembles[filt].image_list[reference_image_id].photometry['x_fit'],
                        img_container.ensembles[filt].image_list[reference_image_id].photometry['y_fit'],
                        ra_obj,
                        dec_obj,
                        img_container.ensembles[filt].wcs,
                        seplimit=seplimit,
                        dcr=dcr,
                        option=option,
                        verbose=verbose,
                    )

                    #   Set new object ID
                    img_container.ensembles[filt].variable_id = obj_id

                    #   Check if variable star was detected
                    if count == 0:
                        raise RuntimeError(
                            f"{style.Bcolors.FAIL} \tERROR: The variable "
                            "star was not detected in the reference image.\n"
                            f"\t-> EXIT {style.Bcolors.ENDC}"
                        )

                    ###
                    #   Load calibration information
                    #
                    calib.deter_calib(
                        img_container,
                        calib_fil,
                        calib_method=calib_method,
                        dcr=dcr,
                        option=option,
                        vizier_dict=vizier_dict,
                        calib_file=calib_file,
                        mag_range=mag_range,
                        correl_method=correl_method,
                        seplimit=seplimit,
                    )
                    terminal_output.print_terminal()

                    #   Stop here if calibration data is not available
                    filter_calib = img_container.CalibParameters.column_names
                    if ('mag' + calib_fil[0] not in filter_calib or
                            'mag' + calib_fil[1] not in filter_calib):
                        err_filter = None
                        if 'mag' + calib_fil[0] not in filter_calib:
                            err_filter = calib_fil[0]
                        if 'mag' + calib_fil[1] not in filter_calib:
                            err_filter = calib_fil[1]
                        terminal_output.print_to_terminal(
                            "Magnitude transformation not "
                            "possible because no calibration data "
                            f"available for filter {err_filter}",
                            indent=2,
                            style_name='WARNING',
                        )
                        #   2023.08.04: Changed from 'break' to 'continue'
                        continue

                    ###
                    #   Calibrate magnitudes
                    #

                    #   Set boolean regarding magnitude plot
                    plot_mags = True if plot_test or plot_ifi else False

                    #   Apply calibration and perform magnitude
                    #   transformation
                    trans.apply_calib(
                        img_container,
                        calib_fil,
                        tcs=tcs,
                        derive_tcs=derive_tcs,
                        plot_mags=plot_mags,
                        photo_type=photo_type,
                        refid=reference_image_id,
                        plot_sigma=plot_sigma,
                    )
                    cali_mags = getattr(img_container, 'cali', None)
                    if not checks.check_unumpy_array(cali_mags):
                        cali = cali_mags['mag']
                    else:
                        cali = cali_mags
                    if np.all(cali == 0):
                        break

                    ###
                    #   Plot light curve
                    #
                    #   Create a Time object for the observation times
                    otime = Time(
                        img_container.ensembles[filt].get_obs_time(),
                        format='jd',
                    )

                    #   Create mask for time series to remove images
                    #   without entries
                    # mask_ts = np.isin(
                    # #cali_mags['med'][i][:,objID],
                    # cali_mags[i][:,objID],
                    # [0.],
                    # invert=True
                    # )

                    #   Create a time series object
                    ts = utilities.mk_ts(
                        otime,
                        cali_mags[i],
                        filt,
                        obj_id,
                    )

                    #   Write time series
                    ts.write(
                        outdir + '/tables/light_curce_' + filt + '.dat',
                        format='ascii',
                        overwrite=True,
                    )
                    ts.write(
                        outdir + '/tables/light_curce_' + filt + '.csv',
                        format='ascii.csv',
                        overwrite=True,
                    )

                    #   Plot light curve over JD
                    plot.light_curve_jd(
                        ts,
                        filt,
                        filt + '_err',
                        outdir,
                        nameobj=nameobj
                    )

                    #   Plot the light curve folded on the period
                    plot.light_curve_fold(
                        ts,
                        filt,
                        filt + '_err',
                        outdir,
                        transit_time,
                        period,
                        binn=binn,
                        nameobj=nameobj,
                    )

                    success = True
                    break

        if not success:
            #   Load calibration information
            calib.deter_calib(
                img_container,
                [filt],
                calib_method=calib_method,
                dcr=dcr,
                option=option,
                vizier_dict=vizier_dict,
                calib_file=calib_file,
                mag_range=mag_range,
            )

            #   Check if calibration data is available
            filter_calib = img_container.CalibParameters.column_names
            if 'mag' + filt not in filter_calib:
                terminal_output.print_terminal(
                    filt,
                    indent=2,
                    string="Magnitude calibration not "
                           "possible because no calibration data is "
                           "available for filter {}. Use normalized flux for light "
                           "curve.",
                    style_name='WARNING',
                )

                #   Get ensemble
                ensemble = img_container.ensembles[filt]

                #   Quasi calibration of the flux data
                trans.flux_calibrate_ensemble(ensemble)

                #   Normalize data if no calibration magnitudes are available
                trans.flux_normalize_ensemble(ensemble)

                plot_quantity = ensemble.uflux_norm
            else:
                #   Set boolean regarding magnitude plot
                plot_mags = True if plot_test or plot_ifi else False

                #   Apply calibration
                trans.apply_calib(
                    img_container,
                    [filt],
                    plot_mags=plot_mags,
                    photo_type=photo_type,
                )
                plot_quantity = getattr(img_container, 'noT', None)[0]

            ###
            #   Plot light curve
            #
            #   Create a Time object for the observation times
            otime = Time(
                img_container.ensembles[filt].get_obs_time(),
                format='jd',
            )

            #   Create a time series object
            ts = utilities.mk_ts(
                otime,
                plot_quantity,
                filt,
                img_container.ensembles[filt].variable_id,
            )

            #   Write time series
            ts.write(
                outdir + '/tables/light_curce_' + filt + '.dat',
                format='ascii',
                overwrite=True,
            )
            ts.write(
                outdir + '/tables/light_curce_' + filt + '.csv',
                format='ascii.csv',
                overwrite=True,
            )

            #   Plot light curve over JD
            plot.light_curve_jd(
                ts,
                filt,
                filt + '_err',
                outdir,
                nameobj=nameobj)

            #   Plot the light curve folded on the period
            plot.light_curve_fold(
                ts,
                filt,
                filt + '_err',
                outdir,
                transit_time,
                period,
                binn=binn,
                nameobj=nameobj,
            )


def subtract_archive_img_from_img(filt, name, path, outdir,
                                  wcs_method='astrometry', plot_comp=True,
                                  hips_source='CDS/P/DSS2/blue'):
    """
        Subtraction of a reference/archival image from the input image.
        The installation of Hotpants is required.

        Parameters
        ----------
        filt            : `string`
            Filter identifier

        name            : `string`
            Name of the object

        path            : `dictionary`
            Paths to images: key - filter name; value - path

        outdir          : `string`
            Path, where the output should be stored.

        wcs_method      : `string`, optional
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        plot_comp       : `boolean`, optional
            If `True` a plot with the original and reference image will
            be created.
            Default is ``True``.

        hips_source     : `string`
            ID string of the image catalog that will be queried using the
            hips service.
            Default is ``CDS/P/DSS2/blue``.
    """
    ###
    #   Check output directories
    #
    checks.check_out(
        outdir,
        os.path.join(outdir, 'subtract'),
    )
    outdir = os.path.join(outdir, 'subtract')

    ###
    #   Check input path
    #
    for p in path.keys():
        checks.check_file(p)

    ###
    #   Trim image as needed (currently images with < 4*10^6 are required)
    #
    #   Load image
    img = CCDData.read(path)

    #   Trim
    xtrim = 2501
    # xtrim = 2502
    ytrim = 1599
    img = ccdp.trim_image(img[0:ytrim, 0:xtrim])
    img.meta['NAXIS1'] = xtrim
    img.meta['NAXIS2'] = ytrim

    #   Save trimmed file
    basename = base_utilities.get_basename(path)
    file_name = basename + '_trimmed.fit'
    file_path = os.path.join(outdir, file_name)
    img.write(file_path, overwrite=True)

    ###
    #   Initialize image ensemble object
    #
    img_ensemble = ImageEnsemble(
        filt,
        name,
        path,
        outdir,
    )

    ###
    #   Find the WCS solution for the image
    #
    utilities.find_wcs(
        img_ensemble,
        reference_image_id=0,
        method=wcs_method,
        indent=3,
    )

    ###
    #   Get image via hips2fits
    #
    # from astropy.utils import data
    # data.Conf.remote_timeout=600
    hips_instance = hips2fitsClass()
    hips_instance.timeout = 120000
    # hipsInstance.timeout = 1200000000
    # hipsInstance.timeout = (200000000, 200000000)
    hips_instance.server = "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
    print(hips_instance.timeout)
    print(hips_instance.server)
    # hips_hdus = hips2fits.query_with_wcs(
    hips_hdus = hips_instance.query_with_wcs(
        hips=hips_source,
        wcs=img_ensemble.wcs,
        get_query_payload=False,
        format='fits',
        verbose=True,
    )
    #   Save downloaded file
    hips_hdus.writeto(os.path.join(outdir, 'hips.fits'), overwrite=True)

    ###
    #   Plot original and reference image
    #
    if plot_comp:
        plot.comp_img(
            outdir,
            img_ensemble.image_list[0].get_data(),
            hips_hdus[0].data,
        )

    ###
    #   Perform image subtraction
    #
    #   Get image and image data
    img = img_ensemble.image_list[0].read_image()
    hips_data = hips_hdus[0].data.astype('float64').byteswap().newbyteorder()

    #   Run hotpants
    subtraction.run_hotpants(
        img.data,
        hips_data,
        img.mask,
        np.zeros(hips_data.shape, dtype=bool),
        image_gain=1.,
        # template_gain=1,
        template_gain=None,
        err=img.uncertainty.array,
        # err=True,
        template_err=True,
        # verbose=True,
        _workdir=outdir,
        # _exe=exe_path,
    )
