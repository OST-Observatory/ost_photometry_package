############################################################################
#                               Libraries                                  #
############################################################################

import os

import copy

import numpy as np

from astropy import uncertainty as unc

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
from astropy.coordinates import SkyCoord, Angle, name_resolve
import astropy.units as u
from astropy.nddata import CCDData

#   hips2fits module is not in the Ubuntu 22.04 package version
#   of astroquery (0.4.1)
# from astroquery.hips2fits import hips2fits
from astroquery.hips2fits import hips2fitsClass

import regions

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

from .. import style, checks, terminal_output

from .. import utilities as base_utilities

warnings.filterwarnings('ignore', category=UserWarning, append=True)


############################################################################
#                           Routines & definitions                         #
############################################################################


class ObjectOfInterest:
    def __init__(self, ra, dec, ra_unit, dec_unit, name):
        #   Set right ascension
        self.ra = ra

        #   Set declination
        self.dec = dec

        #   Set right ascension unit
        self.ra_unit = ra_unit

        #   Set declination unit
        self.dec_unit = dec_unit

        #   Set object_name
        self.name = name

        #   Set sky coordinates object
        self.coordinates_object = SkyCoord(
            ra=ra,
            dec=dec,
            unit=(ra_unit, dec_unit),
            frame="icrs"
        )

        #   ID of object in the image series
        #   Syntax: {'filter': 'id'}
        self.id_in_image_series = {}

        #   Set defaults for period and transit time
        self.transit_time = None
        self.period = None


class Observation:
    """
        Container class for all data taken during an observation session
    """
    #   TODO: Make this the default class that needs to be instated every time and that gives birth to the
    #         tine series and all the rest - add all the analysis functions to this class

    def __init__(self, **kwargs):
        #   Prepare dictionary for image series
        self.image_series_dict = {}

        #   Add additional keywords
        self.__dict__.update(kwargs)

        #   Check for right ascension, declination, units, object names,
        #   periods, and transit times
        ra_objects: (list[str] | None) = kwargs.get('ra_objects', None)
        ra_unit: (str | None) = kwargs.get('ra_unit', None)
        dec_objects: (list[str] | None)  = kwargs.get('dec_objects', None)
        dec_unit: (str | None) = kwargs.get('dec_unit', None)
        object_names: (list[str] | None)  = kwargs.get('object_names', None)
        periods: (list[float] | None) = kwargs.get('periods', None)
        transit_times: (list[str] | None) = kwargs.get('transit_times', None)

        add_periods = False
        if all([periods, transit_times]):
            add_periods = True

        #   Setup object of interests
        self.objects_of_interest = []

        #   Case 1: All base parameters are provided
        if all([ra_objects, dec_objects, ra_unit, dec_unit, object_names]):
            len_names = len(object_names)
            if len_names == len(ra_objects) and len_names == len(ra_objects):
                for i, (name, ra, dec) in enumerate(zip(object_names, ra_objects, dec_objects)):
                    self.objects_of_interest.append(
                        ObjectOfInterest(
                            ra,
                            dec,
                            ra_unit,
                            dec_unit,
                            name,
                        )
                    )
                    if add_periods:
                        self.objects_of_interest[i].period = periods[i]
                        self.objects_of_interest[i].transit_time = transit_times[i]
        #   Case 2: Only the object name is provided
        elif object_names is not None:
            for i, name in enumerate(object_names):
                #   Case 2a: Object can be resolved
                try:
                    sky_coordinates = SkyCoord.from_name(name)
                    self.objects_of_interest.append(
                        ObjectOfInterest(
                            sky_coordinates.ra.degree,
                            sky_coordinates.dec.degree,
                            u.degree,
                            u.degree,
                            name,
                        )
                    )
                #   Case 2b: Object cannot be resolved
                except name_resolve.NameResolveError:
                    self.objects_of_interest.append(
                        ObjectOfInterest(
                            None,
                            None,
                            None,
                            None,
                            name,
                        )
                    )

                if add_periods:
                    self.objects_of_interest[i].period = periods[i]
                    self.objects_of_interest[i].transit_time = transit_times[i]

        #   Sky coordinates for all objects of interest
        if object_names:
            n_objects = len(object_names)
            coordinates_properties = [ra_objects, dec_objects]
            if all(len(x) == n_objects for x in coordinates_properties) and all(coordinates_properties):
                self.objects_of_interest_coordinates = SkyCoord(
                    ra_objects,
                    dec_objects,
                    unit=(ra_unit, dec_unit),
                    frame="icrs",
                )
            else:
                self.objects_of_interest_coordinates = None

        #   TODO: If the object objects are implemented, the following code
        #         will be removed
        #######################################################################
        if ra_objects is not None:
            if ra_unit is not None:
                self.ra_objects = Angle(ra_objects, unit=ra_unit).degree
            else:
                self.ra_objects = Angle(ra_objects, unit='hour').degree
        else:
            self.ra_objects = None
        if dec_objects is not None:
            if dec_unit is not None:
                self.dec_objects = Angle(dec_objects, unit=dec_unit).degree
            else:
                self.dec_objects = Angle(dec_objects, unit='degree').degree
        else:
            self.dec_objects = None

        #   Create SkyCoord object
        #   TODO: Fix that coordinates_objects is once a list and once a SkyCoord object
        self.object_names = object_names
        if self.object_names is not None and self.ra_objects is None and self.dec_objects is None:
            self.coordinates_objects = []
            for object_name in self.object_names:
                self.coordinates_objects.append(SkyCoord.from_name(object_name))
        elif self.ra_objects is not None and self.dec_objects is not None:
            self.coordinates_objects = SkyCoord(
                ra=self.ra_objects,
                dec=self.dec_objects,
                unit=(u.degree, u.degree),
                frame="icrs"
            )
        else:
            self.coordinates_objects = None
        #######################################################################

    #   Get ePSF objects of all images
    def get_epsf(self):
        epsf_dict = {}
        for key, image_series in self.image_series_dict.items():
            epsf_list = []
            for img in image_series.image_list:
                epsf_list.append(img.epsf)
            epsf_dict[key] = epsf_list

        return epsf_dict

    #   Get ePSF object of the reference image
    def get_ref_epsf(self):
        epsf_dict = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_id = image_series.reference_image_id

            img = image_series.image_list[reference_image_id]

            epsf_dict[key] = img.epsf

        return epsf_dict

    #   Get reference image
    def get_ref_img(self):
        img_dict = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_id = image_series.reference_image_id

            img = image_series.image_list[reference_image_id]

            img_dict[key] = img.get_data()

        return img_dict

    #   Get residual image belonging to the reference image
    def get_ref_residual_img(self):
        img_dict = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_id = image_series.reference_image_id

            img = image_series.image_list[reference_image_id]

            img_dict[key] = img.residual_image

        return img_dict

    #   Get image series for a specific set of filters
    def get_image_series(self, filter_list):
        image_series_dict = {}
        for filter_ in filter_list:
            image_series_dict[filter_] = self.image_series_dict[filter_]

        return image_series_dict

    #   Get the IDs of the objects of interest within the detected objects on
    #   the images
    def get_ids_object_of_interest(
            self, filter_=None, reference_image_series_id=None):
        if filter_ is None and reference_image_series_id is None:
            terminal_output.print_to_terminal(
                "Neither a filter nor an image series ID was provided to "
                "compile the IDs for the objects of interest.The image series ID "
                "is assumed to be 0.",
                style_name='WARNING',
            )
            reference_image_series_id = 0

        object_of_interest_ids = []
        for object_ in self.objects_of_interest:
            ids_object_of_interest = object_.id_in_image_series
            if ids_object_of_interest:
                if filter_ is not None:
                    object_of_interest_ids.append(
                        ids_object_of_interest[filter_]
                    )
                else:
                    #   TODO: This is dirty... :( Can you fix it?
                    object_of_interest_ids.append(
                        ids_object_of_interest[
                            list(
                                ids_object_of_interest.keys()
                            )[reference_image_series_id]
                        ]
                    )

        return object_of_interest_ids

    #   Get the names of the objects of interest.
    def get_object_of_interest_names(self):
        name_list = []

        for object_ in self.objects_of_interest:
            name_list.append(object_.name)

        return name_list

    #   Get object right ascensions
    def get_object_ras(self):
        ra_list = []

        for object_ in self.objects_of_interest:
            ra_list.append(object_.ra)

        return ra_list

    #   Get object declinations
    def get_object_decs(self):
        dec_list = []

        for object_ in self.objects_of_interest:
            dec_list.append(object_.dec)

        return dec_list

    #   Get ra unit
    def get_ra_unit(self):
        #   TODO: Check if None or u.degree is better
        ra_unit = None

        if self.objects_of_interest:
            ra_unit = self.objects_of_interest[0].ra_unit

        return ra_unit

    #   Get dec unit
    def get_dec_unit(self):
        dec_unit = None

        if self.objects_of_interest:
            dec_unit = self.objects_of_interest[0].ra_unit

        return dec_unit


#   Image class
class Image:
    """
        Image class: Provides relevant image information and some methods for
                     handling image data.
    """
    def __init__(self, pd, filter_, object_names, path, file_name, output_dir):
        #   Set image ID
        self.pd = pd

        #   Set filter
        self.filt = filter_

        #   TODO: rm parameter?
        #   Set object name
        self.object_name = object_names

        #   Set file name
        self.filename = file_name

        #   Set complete path
        self.path = Path(Path(path) / file_name)

        #   Set path to output directory
        self.outpath = Path(output_dir)

        #   Set wcs default
        self.wcs = None

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


class ImageSeries:
    """
        Image series class: Used to handle a series of images,
                            e.g. taken with a specific filter.
    """

    def __init__(self, filter_, object_names, path, output_dir,
                 reference_image_id=0):
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
        self.filt = filter_

        #   Set number of images
        #   TODO: Make sure that this attribute is always up to date or replace with method
        self.nfiles = len(file_list)

        #   Set ID of the reference image
        self.reference_image_id = reference_image_id

        #   Prepare image list
        self.image_list = []

        #   Set path to output directory
        self.outpath = Path(output_dir)

        #   TODO: rm parameter?
        #   Set object name
        self.object_name = object_names

        #   Fill image list
        terminal_output.print_to_terminal(
            "Read images and calculate field of view, PIXEL scale, etc. ... ",
            indent=2,
        )
        #   TODO: Convert image_list to dictionary
        for image_id, file_name in enumerate(file_list):
            self.image_list.append(
                #   Prepare image class instance
                Image(image_id, filter_, object_names, path, file_name, output_dir)
            )

            #   Calculate field of view and additional quantities and add
            #   them to the image class instance
            base_utilities.calculate_field_of_view(self.image_list[image_id], verbose=False)

        #   Set start time for image series
        self.start_jd = self.image_list[0].jd

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
        tbl_s = self.get_photometry()
        n_max_list = []
        x = []
        y = []
        for i, tbl in enumerate(tbl_s.values()):
            x.append(tbl['x_fit'])
            y.append(tbl['y_fit'])
            n_max_list.append(len(x[i]))

        return x, y, np.max(n_max_list)

    def get_flux_distribution(self, distribution_samples=1000):
        #   Get data
        tbl_s = list(self.get_photometry().values())

        #   Create list of distributions
        flux_list = []
        for tbl in tbl_s:
            flux_list.append(
                unc.normal(
                    tbl['flux_fit'] * u.mag,
                    std=tbl['flux_unc'] * u.mag,
                    n_samples=distribution_samples,
                )
            )

        return flux_list

    def get_flux_array(self):
        #   Get data
        tbl_s = list(self.get_photometry().values())

        #   Expects the number of objects in each table to be the same.
        n_images = len(tbl_s)
        n_objects = len(tbl_s[0])

        flux = np.zeros((n_images, n_objects))
        flux_unc = np.zeros((n_images, n_objects))

        for i, tbl in enumerate(tbl_s):
            flux[i] = tbl['flux_fit']
            flux_unc[i] = tbl['flux_unc']

        return flux, flux_unc


def rm_cosmic_rays(image, limiting_contrast=5., read_noise=8.,
                   sigma_clipping_value=4.5, saturation_level=65535.,
                   verbose=False, add_mask=True, terminal_logger=None):
    """
        Remove cosmic rays

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        limiting_contrast       : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        read_noise              : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigma_clipping_value    : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        saturation_level        : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        verbose                 : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        add_mask                : `boolean`, optional
            If True add hot and bad pixel mask to the reduced science images.
            Default is ``True``.

        terminal_logger         : `terminal_output.TerminalLog` or None, optional
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
    status_cosmics = ccd.meta.get('cosmics_rm', False)

    #   Get exposure time
    exposure_time = ccd.meta.get('exptime', 1.)

    #   Get unit of the image to check if the image was scaled with the
    #   exposure time
    if ccd.unit == u.electron / u.s:
        scaled = True
        reduced = ccd.multiply(exposure_time * u.second)
    else:
        scaled = False
        reduced = ccd

    if not status_cosmics:
        #   Remove cosmic rays
        reduced = ccdp.cosmicray_lacosmic(
            reduced,
            objlim=limiting_contrast,
            readnoise=read_noise,
            sigclip=sigma_clipping_value,
            satlevel=saturation_level,
            verbose=verbose,
        )
        if not add_mask:
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
            reduced = reduced.divide(exposure_time * u.second)

        #   Set file name
        basename = base_utilities.get_basename(image.filename)
        file_name = f'{basename}_cosmic-rm.fit'

        #   Set new file name and path
        image.filename = file_name
        image.path = os.path.join(
            str(image.outpath),
            'cosmics_rm',
            file_name,
        )

        #   Check if the 'cosmics_rm' directory already exits.
        #   If not, create it.
        checks.check_output_directories(os.path.join(str(image.outpath), 'cosmics_rm'))

        #   Save image
        reduced.write(image.path, overwrite=True)


def determine_background(image, sigma_background=5., two_d_background=True,
                         apply_background=True, verbose=False):
    """
        Determine background, using photutils

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        sigma_background    : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        two_d_background    : `boolean`, optional
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
        terminal_output.print_to_terminal(
            f"Determine background: {image.filt} filter",
            indent=2,
        )

    #   Load image data
    ccd = image.read_image()

    #   Set up sigma clipping
    sigma_clip = SigmaClip(sigma=sigma_background)

    #   Calculate background RMS
    background_rms = MADStdBackgroundRMS(sigma_clip=sigma_clip)
    image.std_rms = background_rms(ccd.data)

    #   2D background?
    if two_d_background:
        #   Estimate 2D background
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            ccd.data,
            (50, 50),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )

        #   Remove background
        image_no_bg = ccd.subtract(bkg.background * u.electron / u.s)

        #   Put metadata back on the image, because it is lost while
        #   subtracting the background
        image_no_bg.meta = ccd.meta
        image_no_bg.meta['HIERARCH'] = '2D background removed'

        #   Add Header keyword to mark the file as background subtracted
        image_no_bg.meta['NO_BG'] = True

        #   Get median of the background
        bkg_value = bkg.background_median
    else:
        #   Estimate 1D background
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
        bkg_value = mmm_bkg.calc_background(ccd.data)

        #   Remove background
        image_no_bg = ccd.subtract(bkg_value)

        #   Put metadata back on the image, because it is lost while
        #   subtracting the background
        image_no_bg.meta = ccd.meta
        image_no_bg.meta['HIERARCH'] = '1D background removed'

        #   Add Header keyword to mark the file as background subtracted
        image_no_bg.meta['NO_BG'] = True

    #   Define name and save image
    file_name = f'{base_utilities.get_basename(image.filename)}_no_bkg.fit'
    output_path = image.outpath / 'no_bkg'
    checks.check_output_directories(output_path)
    image_no_bg.write(output_path / file_name, overwrite=True)

    #   Set new path and file
    #   -> Background subtracted image will be used in further processing steps
    if apply_background:
        image.path = output_path / file_name
        image.filename = file_name

    #   Add background value to image class
    image.bkg_value = bkg_value


def find_stars(image, sigma_object_psf, multiplier_background_rms=5.,
               method='IRAF', terminal_logger=None, indent=2):
    """
        Find the stars on the images, using photutils and search and select
        stars for the ePSF stars

        Parameters
        ----------
        image                       : `image.class`
            Image class with all image specific properties

        sigma_object_psf            : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        multiplier_background_rms   : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5``.

        method                      : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        terminal_logger             : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent                      : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    if terminal_logger is not None:
        terminal_logger.add_to_cache("Identify stars", indent=indent)
    else:
        terminal_output.print_to_terminal("Identify stars", indent=indent)

    #   Load image data
    ccd = image.read_image()

    #   Get background RMS
    sigma = image.std_rms

    #   Distinguish between different finder options
    if method == 'DAO':
        #   Set up DAO finder
        dao_finder = DAOStarFinder(
            fwhm=sigma_object_psf * gaussian_sigma_to_fwhm,
            threshold=multiplier_background_rms * sigma
        )

        #   Find stars - make table
        tbl_objects = dao_finder(ccd.data)
    elif method == 'IRAF':
        #   Set up IRAF finder
        iraf_finder = IRAFStarFinder(
            threshold=multiplier_background_rms * sigma,
            fwhm=sigma_object_psf * gaussian_sigma_to_fwhm,
            minsep_fwhm=0.01,
            roundhi=5.0,
            roundlo=-5.0,
            sharplo=0.0,
            sharphi=2.0,
        )

        #   Find stars - make table
        tbl_objects = iraf_finder(ccd.data)
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}\nExtraction method ({method}) not valid: "
            f"use either IRAF or DAO {style.Bcolors.ENDC}"
        )

    #   Add positions to image class
    image.positions = tbl_objects


def check_epsf_stars(image, size_epsf_region=25, minimum_n_stars=25,
                     fraction_epsf_stars=0.2, terminal_logger=None,
                     strict_epsf_checks=True, indent=2):
    """
        Select ePSF stars and check if there are enough

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        size_epsf_region        : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        minimum_n_stars         : `float`, optional
            Minimal number of stars required for the ePSF calculations
            Default is ``25``.

        fraction_epsf_stars     : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        terminal_logger         : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        strict_epsf_checks      : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        indent                  : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    #   Get object positions
    tbl_positions = image.positions

    #   Number of objects
    n_stars = len(tbl_positions)

    #   Get image data
    image_data = image.get_data()

    #   Combine identification string
    identification_string = f'{image.pd}. {image.filt}'

    #   Useful information
    out_string = f"{n_stars} sources identified in the " \
                 f"{identification_string} band image"
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
    tbl_positions_sort = tbl_positions.group_by('flux')
    # Determine the 99 percentile
    percentile_99 = np.percentile(tbl_positions_sort['flux'], 99)
    #   Determine the position of the 99 percentile in the position list
    id_percentile_99 = np.argmin(
        np.absolute(tbl_positions_sort['flux'] - percentile_99)
    )

    #   Check that the minimum number of ePSF stars can be achieved
    available_epsf_stars = int(n_stars * fraction_epsf_stars)
    #   If the available number of stars is less than required (the default is
    #   25 as required by the cutout plots, 25 also seems reasonable for a
    #   good ePSF), use the required number anyway. The following check will
    #   catch any problems.
    if available_epsf_stars < minimum_n_stars:
        available_epsf_stars = minimum_n_stars

    #   Check if enough stars have been identified
    if ((id_percentile_99 - available_epsf_stars < minimum_n_stars and strict_epsf_checks)
            or (id_percentile_99 - available_epsf_stars < 1 and not strict_epsf_checks)):
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ("
            f"{id_percentile_99 - available_epsf_stars}) found to determine "
            f"the ePSF in the {identification_string} band{style.Bcolors.ENDC}"
        )

    #   Resize table -> limit it to the suitable stars
    tbl_epsf_stars = tbl_positions_sort[:][id_percentile_99 - available_epsf_stars:id_percentile_99]

    #   Exclude stars that are too close to the image boarder
    #   Size of the extraction box around each star
    half_size_epsf_region = (size_epsf_region - 1) / 2

    #   New lists with x and y positions
    x = tbl_epsf_stars['xcentroid']
    y = tbl_epsf_stars['ycentroid']

    mask = ((x > half_size_epsf_region) & (x < (image_data.shape[1] - 1 - half_size_epsf_region)) &
            (y > half_size_epsf_region) & (y < (image_data.shape[0] - 1 - half_size_epsf_region)))

    #   Updated positions table
    tbl_epsf_stars = tbl_epsf_stars[:][mask]
    n_useful_epsf_stars = len(tbl_epsf_stars)

    #   Check if there are still enough stars
    if ((n_useful_epsf_stars < minimum_n_stars and strict_epsf_checks) or
            (n_useful_epsf_stars < 1 and not strict_epsf_checks)):
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({n_useful_epsf_stars}) "
            f"for the ePSF determination in the {identification_string} band "
            "image. Too many potential ePSF stars have been removed, because "
            "they are too close to the image border. Check first that enough "
            "stars have been identified, using the starmap_?.pdf files.\n If "
            "that is the case, shrink extraction region or allow for higher "
            "fraction of ePSF stars (size_epsf) from all identified stars "
            f"(frac_epsf_stars). {style.Bcolors.ENDC}"
        )

    #   Find all potential ePSF stars with close neighbors
    x1 = tbl_positions_sort['xcentroid']
    y1 = tbl_positions_sort['ycentroid']
    x2 = tbl_epsf_stars['xcentroid']
    y2 = tbl_epsf_stars['ycentroid']
    max_objects = np.max((len(x1), len(x2)))
    x_all = np.zeros((max_objects, 2))
    y_all = np.zeros((max_objects, 2))
    x_all[0:len(x1), 0] = x1
    x_all[0:len(x2), 1] = x2
    y_all[0:len(y1), 0] = y1
    y_all[0:len(y2), 1] = y2

    #   TODO: Check if a bug hides here
    id_percentile_99 = correlate.correlation_own(
        x_all,
        y_all,
        max_pixel_between_objects=size_epsf_region,
        option=3,
        silent=True,
    )[1]

    #   Determine multiple entries -> stars that are contaminated
    index_percentile_99_mult = [ite for ite, count in Counter(id_percentile_99).items() if count > 1]

    #   Find unique entries -> stars that are not contaminated
    index_percentile_99_unique = [ite for ite, count in Counter(id_percentile_99).items() if count == 1]
    n_useful_epsf_stars = len(index_percentile_99_unique)

    #   Remove ePSF stars with close neighbors from the corresponding table
    tbl_epsf_stars.remove_rows(index_percentile_99_mult)

    #   Check if there are still enough stars
    if ((n_useful_epsf_stars < minimum_n_stars and strict_epsf_checks)
            or (n_useful_epsf_stars < 1 and not strict_epsf_checks)):
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNot enough stars ({n_useful_epsf_stars}) "
            f" for the ePSF determination in the {identification_string} band "
            "image. Too many potential ePSF stars have been removed, because "
            "other stars are in the extraction region. Check first that enough"
            " stars have been identified, using the starmap_?.pdf files.\n"
            "If that is the case, shrink extraction region or allow for "
            "higher fraction of ePSF stars (size_epsf) from all identified "
            f"stars (frac_epsf_stars). {style.Bcolors.ENDC}"
        )

    #   Add ePSF stars to image class
    image.positions_epsf = tbl_epsf_stars


def determine_epsf(image, size_epsf_region=25, oversampling_factor=2,
                   max_n_iterations=7, minimum_n_stars=25,
                   multiprocess_plots=True, terminal_logger=None, indent=2):
    """
        Main function to determine the ePSF, using photutils

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        size_epsf_region    : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        oversampling_factor : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        max_n_iterations    : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        minimum_n_stars     : `float`, optional
            Minimal number of stars required for the ePSF calculations
            Default is ``25``.

        multiprocess_plots  : `boolean`, optional
            If True multiprocessing is used for plotting.
            Default is ``True``.

        terminal_logger     : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    #   Get image data
    data = image.get_data()

    #   Get ePSF star positions
    tbl_positions = image.positions_epsf

    #   Number of ePSF stars
    n_epsf = len(tbl_positions)

    if n_epsf < minimum_n_stars:
        terminal_logger.add_to_cache(
            f"The number of ePSF stars is less than required."
            f"{n_epsf} ePSF stars available. {minimum_n_stars} were "
            "requested.",
            indent=indent,
            style_name='WARNING',
        )

    #   Get object name
    name_object = image.object_name

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            "Determine the point spread function",
            indent=indent
        )
        terminal_logger.add_to_cache(
            f"{n_epsf} bright stars used",
            indent=indent + 1,
            style_name='OK',
        )
    else:
        terminal_output.print_to_terminal(
            "Determine the point spread function",
            indent=indent
        )
        terminal_output.print_to_terminal(
            f"{n_epsf} bright stars used",
            indent=indent + 1,
            style_name='OK',
        )

    #   Create new table with the names required by "extract_stars"
    stars_tbl = Table()
    stars_tbl['x'] = tbl_positions['xcentroid']
    stars_tbl['y'] = tbl_positions['ycentroid']

    #   Put image into NDData container (required by "extract_stars")
    nd_data = NDData(data=data)

    #   Extract cutouts of the selected stars
    stars = extract_stars(nd_data, stars_tbl, size=size_epsf_region)

    #   Combine plot identification string
    string = f'img-{image.pd}-{image.filt}'

    #   Get output directory
    output_dir = image.outpath.name

    #   Plot the brightest ePSF stars
    if multiprocess_plots:
        p = mp.Process(
            target=plot.plot_cutouts,
            args=(output_dir, stars, string),
            kwargs={'name_object': name_object, }
        )
        p.start()
    else:
        plot.plot_cutouts(
            output_dir,
            stars,
            string,
            name_object=name_object,
            terminal_logger=terminal_logger,
        )

    #   Build the ePSF (set oversampling and max. number of iterations)
    epsf_builder = EPSFBuilder(
        oversampling=oversampling_factor,
        maxiters=max_n_iterations,
        progress_bar=False,
    )
    epsf, fitted_stars = epsf_builder(stars)

    #   Add ePSF and fitted stars to image class
    image.epsf = epsf
    image.fitted_stars = fitted_stars


def extraction_epsf(image, sigma_object_psf, sigma_background=5.,
                    use_initial_positions=True, finder_method='IRAF',
                    size_epsf_region=25., multiplier_background_rms=5.0,
                    multiplier_dao_grouper=2.0, strict_cleaning_results=True,
                    terminal_logger=None, rm_background=False, indent=2):
    """
        Main function to perform the eEPSF photometry, using photutils

        Parameters
        ----------
        image                       : `image.class`
            Image class with all image specific properties

        sigma_object_psf            : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_background            : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        use_initial_positions       : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        finder_method               : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        size_epsf_region            : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        multiplier_background_rms   : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multiplier_dao_grouper      : `float`, optional
            Multiplier for the DAO grouper
            Default is ``2.0``.

        strict_cleaning_results     : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        terminal_logger             : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        rm_background               : `boolean`, optional
            If True the background will be estimated and considered.
            Default is ``False``. -> It is expected that the background
            was removed before.

        indent                      : `integer`, optional
            Indentation for the console output lines
            Default is ``2`.

    """
    #   Get output path
    output_path = image.outpath

    #   Check output directories
    checks.check_output_directories(
        output_path,
        output_path / 'tables',
    )

    #   Get image data
    data = image.get_data()

    #   Get filter
    filter_ = image.filt

    #   Get already identified objects (position and flux)
    if use_initial_positions:
        try:
            #   Get position information
            positions_flux = image.positions
            initial_positions = Table(
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
            use_initial_positions = False

    #   Set output and plot identification string
    identification_str = f"{image.pd}-{filter_}"

    #   Get background RMS
    background_rms = image.std_rms

    #   Get ePSF
    epsf = image.epsf

    output_str = f"Performing the actual PSF photometry (" \
                 f"{identification_str} image)"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(output_str, indent=indent)
    else:
        terminal_output.print_to_terminal(output_str, indent=indent)

    #  Set up all necessary classes
    if finder_method == 'IRAF':
        #   IRAF finder
        finder = IRAFStarFinder(
            threshold=multiplier_background_rms * background_rms,
            fwhm=sigma_object_psf * gaussian_sigma_to_fwhm,
            minsep_fwhm=0.01,
            roundhi=5.0,
            roundlo=-5.0,
            sharplo=0.0,
            sharphi=2.0,
        )
    elif finder_method == 'DAO':
        #   DAO finder
        finder = DAOStarFinder(
            fwhm=sigma_object_psf * gaussian_sigma_to_fwhm,
            threshold=multiplier_background_rms * background_rms,
            exclude_border=True,
        )
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nExtraction method ({finder_method}) "
            f"not valid: use either IRAF or DAO {style.Bcolors.ENDC}"
        )
    #   Fitter used
    fitter = LevMarLSQFitter()

    #   Make sure the size of the extraction region is uneven
    if size_epsf_region % 2 == 0:
        size_extraction_region = size_epsf_region + 1
    else:
        size_extraction_region = size_epsf_region

    #   Number of iterations
    n_iterations = 1

    #   Set up sigma clipping
    if rm_background:
        sigma_clip = SigmaClip(sigma=sigma_background)
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
    else:
        mmm_bkg = None

    try:
        #   DAO grouper
        dao_group = DAOGroup(
            multiplier_dao_grouper * sigma_object_psf * gaussian_sigma_to_fwhm
        )

        #  Set up the overall class to extract the data
        photometry = IterativelySubtractedPSFPhotometry(
            finder=finder,
            group_maker=dao_group,
            bkg_estimator=mmm_bkg,
            psf_model=epsf,
            fitter=fitter,
            niters=n_iterations,
            fitshape=(size_extraction_region, size_extraction_region),
            aperture_radius=(size_extraction_region - 1) / 2
        )

        #   Extract the photometry and make a table
        if use_initial_positions:
            result_tbl = photometry(image=data, init_guesses=initial_positions)
        else:
            result_tbl = photometry(image=data)
    except RuntimeError as e:
        if multiplier_dao_grouper != 1.:
            terminal_output.print_to_terminal(
                "IterativelySubtractedPSFPhotometry failed. "
                "Will try again with 'multi_grouper' set to 1...",
                indent=indent,
                style_name='WARNING',
            )
            multiplier_dao_grouper = 1.
            #   DAO grouper
            dao_group = DAOGroup(
                multiplier_dao_grouper * sigma_object_psf * gaussian_sigma_to_fwhm
            )

            #  Set up the overall class to extract the data
            photometry = IterativelySubtractedPSFPhotometry(
                finder=finder,
                group_maker=dao_group,
                bkg_estimator=mmm_bkg,
                psf_model=epsf,
                fitter=fitter,
                niters=n_iterations,
                fitshape=(size_extraction_region, size_extraction_region),
                aperture_radius=(size_extraction_region - 1) / 2
            )

            #   Extract the photometry and make a table
            if use_initial_positions:
                result_tbl = photometry(
                    image=data,
                    init_guesses=initial_positions,
                )
            else:
                result_tbl = photometry(image=data)
        else:
            terminal_output.print_to_terminal(
                "IterativelySubtractedPSFPhotometry failed. "
                "No recovery possible.",
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
        estimated_uncertainty = np.absolute(
            result_tbl['flux_fit'] - result_tbl['flux_0']
        )
        result_tbl.add_column(estimated_uncertainty, name='flux_unc')

    #   Clean output for objects with NANs in uncertainties
    try:
        uncertainty_mask = np.invert(np.isnan(result_tbl['flux_unc'].value))
        result_tbl = result_tbl[uncertainty_mask]
    except:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of NANs in "
            f"uncertainties... {style.Bcolors.ENDC}"
        )

    #   Clean output for objects with negative uncertainties
    try:
        bad_results = np.where(result_tbl['flux_fit'].data < 0.)
        result_tbl.remove_rows(bad_results)
        n_bad_objects = np.size(bad_results)
        if strict_cleaning_results:
            bad_results = np.where(result_tbl['flux_unc'].data < 0.)
            n_bad_objects += len(bad_results)
            result_tbl.remove_rows(bad_results)
    except:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of negative "
            f"uncertainties... {style.Bcolors.ENDC}"
        )

    #   Clean output for objects with negative pixel coordinates
    try:
        bad_results = np.where(result_tbl['x_fit'].data < 0.)
        n_bad_objects += np.size(bad_results)
        result_tbl.remove_rows(bad_results)
        bad_results = np.where(result_tbl['y_fit'].data < 0.)
        n_bad_objects += np.size(bad_results)
        result_tbl.remove_rows(bad_results)
    except:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nProblem with cleanup of negative pixel "
            f"coordinates... {style.Bcolors.ENDC}"
        )

    if n_bad_objects != 0:
        out_str = f"{n_bad_objects} objects removed because of poor quality"
        if terminal_logger is not None:
            terminal_logger.add_to_cache(out_str, indent=indent + 1)
        else:
            terminal_output.print_to_terminal(out_str, indent=indent + 1)

    try:
        n_stars = len(result_tbl['flux_fit'].data)
    except:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nTable produced by "
            "IterativelySubtractedPSFPhotometry is empty after cleaning up "
            "of objects with negative pixel coordinates and negative "
            f"uncertainties {style.Bcolors.ENDC}"
        )

    out_str = f"{n_stars} good stars extracted from the image"
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            out_str,
            indent=indent + 1,
            style_name='OK',
        )
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
        int((size_extraction_region - 1) / 2),
        terminal_logger=terminal_logger,
    )

    #   Write table
    filename = 'table_photometry_{}_PSF.dat'.format(identification_str)
    result_tbl.write(
        output_path / 'tables' / filename,
        format='ascii',
        overwrite=True,
    )

    #  Make residual image
    residual_image = photometry.get_residual_image()

    #   Add photometry and residual image to image class
    image.photometry = result_tbl
    image.residual_image = residual_image


def compute_photometric_uncertainties(flux_variance, aperture_area,
                                      annulus_area, uncertainty_background,
                                      gain=1.0):
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
        flux_variance             : `numpy.ndarray`
            Extracted aperture flux data or the error^2 of the extraction
            if available -> proxy for the Poisson noise

        aperture_area             : `float`
            Photometric aperture area

        annulus_area              : `float`
            Sky annulus area

        uncertainty_background    : `numpy.ndarray`
            Uncertainty in the sky measurement

        gain                        : `float`, optional
            Electrons per ADU
            Default is ``1.0``. Usually we already work with gain corrected
            data.
    """

    #   Calculate flux error as above
    bg_variance_terms = ((aperture_area * uncertainty_background ** 2.) *
                         (1. + aperture_area / annulus_area))
    variance = flux_variance / gain + bg_variance_terms
    flux_error = variance ** .5

    return flux_error


def define_apertures(image, aperture_radius, inner_annulus_radius,
                     outer_annulus_radius, unit_radii):
    """
        Define stellar and background apertures

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        aperture_radius         : `float`
            Radius of the stellar aperture

        inner_annulus_radius    : `float`
            Inner radius of the background annulus

        outer_annulus_radius    : `float`
            Outer radius of the background annulus

        unit_radii              : `string`
            Unit of the radii above. Permitted values are ``pixel``
            and ``arcsec``.

        Returns
        -------
        aperture                : `photutils.aperture.CircularAperture`
            Stellar aperture

        annulus_aperture        : `photutils.aperture.CircularAnnulus`
            Background annulus
    """
    #   Get position information
    tbl = image.positions

    #   Extract positions and prepare a position list
    try:
        x_positions = tbl['x_fit']
        y_positions = tbl['y_fit']
    except:
        x_positions = tbl['xcentroid']
        y_positions = tbl['ycentroid']
    positions = list(zip(x_positions, y_positions))

    #   Check unit of radii
    if unit_radii not in ['pixel', 'arcsec']:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nUnit of the aperture radii not valid: "
            f"set it either to pixel or arcsec {style.Bcolors.ENDC}"
        )

    #   Convert radii in arcsec to pixel
    #   (this part is prone to errors and needs to be rewritten)
    pixel_scale = image.pixscale
    if pixel_scale is not None and unit_radii == 'arcsec':
        aperture_radius = aperture_radius / pixel_scale
        inner_annulus_radius = inner_annulus_radius / pixel_scale
        outer_annulus_radius = outer_annulus_radius / pixel_scale

    #   Make stellar aperture
    aperture = CircularAperture(positions, r=aperture_radius)

    #   Make background annulus
    annulus_aperture = CircularAnnulus(
        positions,
        r_in=inner_annulus_radius,
        r_out=outer_annulus_radius,
    )

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

        bkg_standard_deviation           : `float`
            Standard deviation of the background
    """
    bkg_median = []
    bkg_standard_deviation = []

    #   Calculate mask from background annulus
    annulus_masks = annulus_aperture.to_mask(method='center')

    #   Loop over all masks
    for mask in annulus_masks:
        #   Extract annulus data
        annulus_data = mask.multiply(image.get_data())

        #   Convert annulus data to 1D
        annulus_data_1d = annulus_data[mask.data > 0]

        #   Sigma clipping
        _, median, standard_deviation = sigma_clipped_stats(annulus_data_1d)

        #   Add to list
        bkg_median.append(median)
        bkg_standard_deviation.append(standard_deviation)

    #   Convert to numpy array
    bkg_median = np.array(bkg_median)
    bkg_standard_deviation = np.array(bkg_standard_deviation)

    return bkg_median, bkg_standard_deviation


def extraction_aperture(image, radius_aperture, inner_annulus_radius,
                        outer_annulus_radius, radii_unit='pixel',
                        background_estimate_simple=False,
                        plot_aperture_positions=False, terminal_logger=None,
                        indent=2):
    """
        Perform aperture photometry using the photutils aperture package

        Parameters
        ----------
        image                       : `image.class`
            Image class with all image specific properties

        radius_aperture             : `float`
            Radius of the stellar aperture

        inner_annulus_radius        : `float`
            Inner radius of the background annulus

        outer_annulus_radius        : `float`
            Outer radius of the background annulus

        radii_unit                  : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and
            ``arcsec``.
            Default is ``pixel``.

        background_estimate_simple  : `boolean`, optional
            If True the background will be extract by a simple algorithm that
            calculates the median within the background annulus. If False the
            background will be extracted using
            photutils.aperture.aperture_photometry.
            Default is ``False``.

        plot_aperture_positions     : `boolean`, optional
            IF true a plot showing the apertures in relation to image is
            created.
            Default is ``False``.

        terminal_logger             : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent                      : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.
    """
    #   Load image data and uncertainty
    ccd = image.read_image()
    data = ccd.data
    uncertainty = ccd.uncertainty.array

    #   Get filter
    filter_ = image.filt

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            f"Performing aperture photometry ({filter_} image)",
            indent=indent,
        )
    else:
        terminal_output.print_to_terminal(
            f"Performing aperture photometry ({filter_} image)",
            indent=indent,
        )

    ###
    #   Define apertures
    #
    aperture, annulus_aperture = define_apertures(
        image,
        radius_aperture,
        inner_annulus_radius,
        outer_annulus_radius,
        radii_unit,
    )

    ###
    #   Extract photometry
    #
    #   Extract aperture
    photometry_tbl = aperture_photometry(
        data,
        aperture,
        mask=ccd.mask,
        error=uncertainty,
    )

    #   Extract background and calculate median - extract background aperture
    if background_estimate_simple:
        bkg_median, bkg_err = background_simple(image, annulus_aperture)

        #   Add median background to the output table
        photometry_tbl['annulus_median'] = bkg_median

        #   Calculate background for the apertures add to the output table
        photometry_tbl['aper_bkg'] = bkg_median * aperture.area
    else:
        bkg_phot = aperture_photometry(
            data,
            annulus_aperture,
            mask=ccd.mask,
            error=uncertainty,
        )

        #   Calculate aperture background and the corresponding error
        photometry_tbl['aper_bkg'] = (bkg_phot['aperture_sum']
                                      * aperture.area / annulus_aperture.area)

        photometry_tbl['aper_bkg_err'] = (bkg_phot['aperture_sum_err']
                                          * aperture.area / annulus_aperture.area)

        bkg_err = photometry_tbl['aper_bkg_err']

    #   Subtract background from aperture flux and add it to the
    #   output table
    photometry_tbl['aper_sum_bkgsub'] = (photometry_tbl['aperture_sum']
                                         - photometry_tbl['aper_bkg'])

    #   Define flux column
    #   (necessary to have the same column names for aperture and PSF
    #   photometry)
    photometry_tbl['flux_fit'] = photometry_tbl['aper_sum_bkgsub']

    # Error estimate
    if uncertainty is not None:
        err_column = photometry_tbl['aperture_sum_err']
    else:
        err_column = photometry_tbl['flux_fit'] ** 0.5

    photometry_tbl['flux_unc'] = compute_photometric_uncertainties(
        err_column,
        aperture.area,
        annulus_aperture.area,
        bkg_err,
    )

    #   Rename position columns
    photometry_tbl.rename_column('xcenter', 'x_fit')
    photometry_tbl.rename_column('ycenter', 'y_fit')

    #   Convert distance/radius to the border to pixel.
    if radii_unit == 'pixel':
        required_distance_to_edge = int(outer_annulus_radius)
    elif radii_unit == 'arcsec':
        pixel_scale = image.pixscale
        required_distance_to_edge = int(
            round(outer_annulus_radius / pixel_scale)
        )
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nException in aperture_extract(): '"
            f"\n'r_unit ({radii_unit}) not known -> Exit {style.Bcolors.ENDC}"
        )

    #   Remove objects that are too close to the image edges
    photometry_tbl = utilities.rm_edge_objects(
        photometry_tbl,
        data,
        required_distance_to_edge,
        terminal_logger=terminal_logger,
    )

    #   Remove negative flux values as they are not physical
    flux = np.array(photometry_tbl['flux_fit'])
    mask = np.where(flux > 0.)
    photometry_tbl = photometry_tbl[mask]

    #   Add photometry to image class
    image.photometry = photometry_tbl

    ###
    #   Plot star map with aperture overlay
    #
    if plot_aperture_positions:
        plot.plot_apertures(
            image.outpath.name,
            data,
            aperture,
            annulus_aperture,
            f'{filter_}_{image.pd}',
        )

    #   Number of stars
    n_objects = len(flux)

    #   Useful info
    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            f"{n_objects} good objects extracted from the image",
            indent=indent + 1,
            style_name='OK',
        )
    else:
        terminal_output.print_to_terminal(
            f"{n_objects} good objects extracted from the image",
            indent=indent + 1,
            style_name='OK',
        )


#   TODO: Check were this is used and if it is still functional, rename
def correlate_preserve_calibration_objects(
        image_series: 'analyze.ImageSeries', filter_list,
        calib_method='APASS',
        magnitude_range=(0., 18.5),
        vizier_dict: dict[str, str] | None = None, calib_file=None,
        max_pixel_between_objects=3,
        own_correlation_option=1,
        verbose=False,
        cross_identification_limit=1,
        reference_image_id=0,
        n_allowed_non_detections_object=1,
        expected_bad_image_fraction=1.0,
        protect_reference_obj=True,
        plot_only_reference_starmap=True,
        correlation_method='astropy',
        separation_limit=2. * u.arcsec):
    """
        Correlate results from all images, while preserving the calibration
        stars

        Parameters
        ----------
        image_series
            Image series object with all image data taken in a specific
            filter

        filter_list                     : `list` with `strings`
            Filter list

        calib_method                    : `string`, optional
            Calibration method
            Default is ``APASS``.

        magnitude_range                 : `tuple` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        vizier_dict
            Identifiers of catalogs, containing calibration data
            Default is ``None``.

        calib_file                      : `string`, optional
            Path to the calibration file
            Default is ``None``.

        max_pixel_between_objects       : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option          : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        verbose                         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        cross_identification_limit      : `integer`, optional
            Cross-identification limit between multiple objects in the current
            image and one object in the reference image. The current image is
            rejected when this limit is reached.
            Default is ``1``.

        reference_image_id              : `integer`, optional
            ID of the reference image
            Default is ``0``.

        n_allowed_non_detections_object : `integer`, optional
            Maximum number of times an object may not be detected in an image.
            When this limit is reached, the object will be removed.
            Default is ``i`.

        expected_bad_image_fraction     : `float`, optional
            Fraction of low quality images, i.e. those images for which a
            reduced number of objects with valid source positions are expected.
            Default is ``1.0``.

        protect_reference_obj           : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        plot_only_reference_starmap     : `boolean`, optional
            If True only the starmap for the reference image will be created.
            Default is ``True``.

        correlation_method              : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        separation_limit                : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    """
    ###
    #   Load calibration data
    #
    calib_tbl, column_names, ra_unit = calib.load_calibration_data_table(
        image_series.image_list[reference_image_id],
        filter_list,
        calibration_method=calib_method,
        magnitude_range=magnitude_range,
        vizier_dict=vizier_dict,
        path_calibration_file=calib_file,
    )

    #   Number of calibration stars
    n_calib_stars = len(calib_tbl)

    if n_calib_stars == 0:
        raise Exception(
            f"{style.Bcolors.FAIL} \nNo match between calibrations stars and "
            f"the\n extracted stars detected. -> EXIT {style.Bcolors.ENDC}"
        )

    ###
    #   Find IDs of calibration stars to ensure they are not deleted in
    #   the correlation process
    #
    #   Lists for IDs, and xy coordinates
    calib_stars_ids = []
    calib_x_pixel_positions = []
    calib_y_pixel_positions = []

    #   Loop over all calibration stars
    for k in range(0, n_calib_stars):
        #   Find the calibration star
        #   TODO: Fix this
        id_calib_star, ref_count, x_calib_star, y_calib_star = correlate.posi_obj_srcor_img(
            image_series.image_list[reference_image_id],
            calib_tbl[column_names['ra']].data[k],
            calib_tbl[column_names['dec']].data[k],
            image_series.wcs,
            dcr=max_pixel_between_objects,
            option=own_correlation_option,
            ra_unit=ra_unit,
            verbose=verbose,
        )
        if verbose:
            terminal_output.print_to_terminal('')

        #   Add ID and coordinates of the calibration star to the lists
        if ref_count != 0:
            calib_stars_ids.append(id_calib_star[1][0])
            calib_x_pixel_positions.append(x_calib_star)
            calib_y_pixel_positions.append(y_calib_star)
    terminal_output.print_to_terminal(
        f"{len(calib_stars_ids):d} matches",
        indent=3,
        style_name='OKBLUE',
    )
    terminal_output.print_to_terminal()

    ###
    #   Correlate the results from all images
    #
    correlate.correlate_image_series_images(
        image_series,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        cross_identification_limit=cross_identification_limit,
        reference_obj_ids=calib_stars_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        expected_bad_image_fraction=expected_bad_image_fraction,
        protect_reference_obj=protect_reference_obj,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
    )

    ###
    #   Plot image with the final positions overlaid (final version)
    #
    utilities.prepare_and_plot_starmap_from_image_series(
        image_series,
        calib_x_pixel_positions,
        calib_y_pixel_positions,
        plot_reference_only=plot_only_reference_starmap,
    )


def extract_multiprocessing(
        image_series: 'analyze.ImageSeries', n_cores_multiprocessing,
        sigma_object_psf,
        sigma_value_background_clipping=5.,
        multiplier_background_rms=5., size_epsf_region=25,
        fraction_epsf_stars=0.2,
        oversampling_factor_epsf=2, max_n_iterations_epsf=7,
        use_initial_positions_epsf=True,
        object_finder_method='IRAF',
        multiplier_background_rms_epsf=5.0,
        multiplier_dao_grouper_epsf=2.0,
        strict_cleaning_epsf_results=True,
        minimum_n_eps_stars=25,
        photometry_extraction_method='PSF',
        radius_aperture=5., inner_annulus_radius=7.,
        outer_annulus_radius=10., radii_unit='arcsec',
        strict_epsf_checks=True,
        identify_objects_on_image=True,
        plots_for_all_images=False,
        plot_for_reference_image_only=True):
    """
        Extract flux and object positions using multiprocessing

        Parameters
        ----------
        image_series
            Image series object with all image data taken in a specific
            filter

        n_cores_multiprocessing         : `integer`
            Number of cores to use during multiprocessing.

        sigma_object_psf                : `dictionary`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_value_background_clipping : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multiplier_background_rms       : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf_region                : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        fraction_epsf_stars             : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling_factor_epsf        : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        max_n_iterations_epsf           : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.
            Default is ``7``.

        use_initial_positions_epsf      : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        object_finder_method            : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multiplier_background_rms_epsf  : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multiplier_dao_grouper_epsf     : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning_epsf_results    : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        minimum_n_eps_stars             : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        photometry_extraction_method    : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        radius_aperture                 : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        inner_annulus_radius            : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        outer_annulus_radius            : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        radii_unit                      : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_epsf_checks              : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        identify_objects_on_image       : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        plots_for_all_images            : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_for_reference_image_only   : `boolean`, optional
            If True a star map plots only for the reference image is created
            Default is ``True``.
    """
    #   Get filter
    filter_ = image_series.filt

    ###
    #   Find the stars (via DAO or IRAF StarFinder)
    #
    #   Commented out as it seems unnecessary...
    #
    # if not identify_objects_on_image:
    #     determine_background(
    #         image_series.ref_img,
    #         sigma_background=sigma_value_background_clipping,
    #     )
    #
    #     find_stars(
    #         image_series.ref_img,
    #         sigma_object_psf[filter_],
    #         multiplier_background_rms=multiplier_background_rms,
    #         method=object_finder_method,
    #     )

    ###
    #   Main loop: Extract stars and info from all images, using
    #              multiprocessing
    #
    #   Initialize multiprocessing object
    executor = utilities.Executor(n_cores_multiprocessing)

    #   Main loop
    for image in image_series.image_list:
        #   Set positions of the reference image if required
        if not identify_objects_on_image:
            image.positions = image_series.ref_img.positions

        #   Extract photometry
        executor.schedule(
            main_extract,
            args=(
                image,
                sigma_object_psf[filter_],
            ),
            kwargs={
                'multiprocessing': True,
                'sigma_value_background_clipping': sigma_value_background_clipping,
                'multiplier_background_rms': multiplier_background_rms,
                'size_epsf_region': size_epsf_region,
                'fraction_epsf_stars': fraction_epsf_stars,
                'oversampling_factor_epsf': oversampling_factor_epsf,
                'max_n_iterations_epsf': max_n_iterations_epsf,
                'use_initial_positions_epsf': use_initial_positions_epsf,
                'object_finder_method': object_finder_method,
                'multiplier_background_rms_epsf': multiplier_background_rms_epsf,
                'multiplier_dao_grouper_epsf': multiplier_dao_grouper_epsf,
                'strict_cleaning_epsf_results': strict_cleaning_epsf_results,
                'minimum_n_eps_stars': minimum_n_eps_stars,
                'strict_epsf_checks': strict_epsf_checks,
                'id_reference_image': image_series.reference_image_id,
                'photometry_extraction_method': photometry_extraction_method,
                'radius_aperture': radius_aperture,
                'inner_annulus_radius': inner_annulus_radius,
                'outer_annulus_radius': outer_annulus_radius,
                'radii_unit': radii_unit,
                'identify_objects_on_image': identify_objects_on_image,
                'plots_for_all_images': plots_for_all_images,
                'plot_for_reference_image_only': plot_for_reference_image_only,
            }
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Extraction using multiprocessing failed '
            f'for {filter_} :({style.Bcolors.ENDC}'
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
    for img in image_series.image_list:
        for pd, tbl in res:
            if pd == img.pd:
                img.photometry = tbl
                tmp_list.append(img)

    image_series.image_list = tmp_list


def main_extract(image, sigma_object_psf, multiprocessing=False,
                 sigma_value_background_clipping=5.,
                 multiplier_background_rms=5., size_epsf_region=25,
                 fraction_epsf_stars=0.2, oversampling_factor_epsf=2,
                 max_n_iterations_epsf=7, use_initial_positions_epsf=True,
                 object_finder_method='IRAF',
                 multiplier_background_rms_epsf=5.0,
                 multiplier_dao_grouper_epsf=2.0,
                 strict_cleaning_epsf_results=True, minimum_n_eps_stars=25,
                 id_reference_image=0, photometry_extraction_method='PSF',
                 radius_aperture=4., inner_annulus_radius=7.,
                 outer_annulus_radius=10., radii_unit='arcsec',
                 strict_epsf_checks=True, identify_objects_on_image=True,
                 cosmic_ray_removal=False, limiting_contrast_rm_cosmics=5.,
                 read_noise=8., sigma_clipping_value=4.5,
                 saturation_level=65535., plots_for_all_images=False,
                 plot_for_reference_image_only=True):
    """
        Main function to extract the information from the individual images

        Parameters
        ----------
        image                           : `image.class`
            Image class with all image specific properties

        sigma_object_psf                : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        multiprocessing                 : `boolean`, optional
            If True, the routine is set up to meet the requirements of
            multiprocessing, such as returning results and delayed
            output to the terminal.

        sigma_value_background_clipping : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5``.

        multiplier_background_rms       : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5``.

        size_epsf_region                : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        fraction_epsf_stars             : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling_factor_epsf        : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        max_n_iterations_epsf           : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        use_initial_positions_epsf      : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        object_finder_method            : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multiplier_background_rms_epsf  : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multiplier_dao_grouper_epsf     : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning_epsf_results    : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        minimum_n_eps_stars             : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        id_reference_image              : `integer`, optional
            ID of the reference image
            Default is ``0``.

        photometry_extraction_method    : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        radius_aperture                 : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        inner_annulus_radius            : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        outer_annulus_radius            : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        radii_unit                      : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_epsf_checks              : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        identify_objects_on_image       : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        cosmic_ray_removal              : `bool`
            If True cosmic rays will be removed from the image.
            Default is ``False``.

        limiting_contrast_rm_cosmics    : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        read_noise                      : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigma_clipping_value            : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        saturation_level                : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        plots_for_all_images            : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_for_reference_image_only   : `boolean`, optional
            If True a star map plot only for the reference image is created
            Default is ``True``.
    """
    ###
    #   Initialize output class in case of multiprocessing
    #
    if multiprocessing:
        terminal_logger = terminal_output.TerminalLog()
        terminal_logger.add_to_cache(
            f"Image: {image.pd}",
            style_name='UNDERLINE',
        )
    else:
        terminal_output.print_to_terminal(
            f"Image: {image.pd}",
            indent=2,
            style_name='UNDERLINE',
        )
        terminal_logger = None

    ###
    #   Remove cosmics (optional)
    #
    if cosmic_ray_removal:
        rm_cosmic_rays(
            image,
            limiting_contrast=limiting_contrast_rm_cosmics,
            read_noise=read_noise,
            sigma_clipping_value=sigma_clipping_value,
            saturation_level=saturation_level,
        )

    ###
    #   Estimate and remove background
    #
    determine_background(
        image,
        sigma_background=sigma_value_background_clipping,
    )

    ###
    #   Find the stars (via DAO or IRAF StarFinder)
    #
    if identify_objects_on_image:
        find_stars(
            image,
            sigma_object_psf,
            multiplier_background_rms=multiplier_background_rms,
            method=object_finder_method,
            terminal_logger=terminal_logger,
        )

    if photometry_extraction_method == 'PSF':
        #   Check size of ePSF extraction region
        if size_epsf_region % 2 == 0:
            size_epsf_region = size_epsf_region + 1

        ###
        #   Check if enough stars have been detected to allow ePSF
        #   calculations
        #
        check_epsf_stars(
            image,
            size_epsf_region=size_epsf_region,
            minimum_n_stars=minimum_n_eps_stars,
            fraction_epsf_stars=fraction_epsf_stars,
            terminal_logger=terminal_logger,
            strict_epsf_checks=strict_epsf_checks,
        )

        ###
        #   Plot images with the identified stars overlaid
        #
        if plots_for_all_images or (plot_for_reference_image_only
                                    and image.pd == id_reference_image):
            plot.starmap(
                image.outpath.name,
                image.get_data(),
                image.filt,
                image.positions,
                tbl_2=image.positions_epsf,
                label='identified stars',
                label_2='stars used to determine the ePSF',
                rts=f'Initial object identification [Image: {image.pd}]',
                name_object=image.object_name,
                wcs=image.wcs,
                terminal_logger=terminal_logger,
            )

        ###
        #   Calculate the ePSF
        #
        determine_epsf(
            image,
            size_epsf_region=size_epsf_region,
            oversampling_factor=oversampling_factor_epsf,
            max_n_iterations=max_n_iterations_epsf,
            minimum_n_stars=minimum_n_eps_stars,
            multiprocess_plots=False,
            terminal_logger=terminal_logger,
        )

        ###
        #   Plot the ePSFs
        #
        plot.plot_epsf(
            image.outpath.name,
            {f'img-{image.pd}-{image.filt}': image.epsf},
            terminal_logger=terminal_logger,
            name_object=image.object_name,
            indent=2,
        )

        ###
        #   Performing the PSF photometry
        #
        extraction_epsf(
            image,
            sigma_object_psf,
            sigma_background=sigma_value_background_clipping,
            use_initial_positions=use_initial_positions_epsf,
            finder_method=object_finder_method,
            size_epsf_region=size_epsf_region,
            multiplier_background_rms=multiplier_background_rms_epsf,
            multiplier_dao_grouper=multiplier_dao_grouper_epsf,
            strict_cleaning_results=strict_cleaning_epsf_results,
            terminal_logger=terminal_logger,
        )

        ###
        #   Plot original and residual image
        #
        plot.plot_residual(
            image.object_name,
            {f'{image.filt}, Image ID: {image.pd}': image.get_data()},
            {f'{image.filt}, Image ID: {image.pd}': image.residual_image},
            image.outpath.name,
            terminal_logger=terminal_logger,
            name_object=image.object_name,
            indent=2,
        )

    elif photometry_extraction_method == 'APER':
        ###
        #   Perform aperture photometry
        #
        if image.pd == id_reference_image:
            plot_aperture_positions = True
        else:
            plot_aperture_positions = False

        extraction_aperture(
            image,
            radius_aperture,
            inner_annulus_radius,
            outer_annulus_radius,
            radii_unit=radii_unit,
            plot_aperture_positions=plot_aperture_positions,
            terminal_logger=terminal_logger,
            indent=3,
        )

    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nExtraction method "
            f"({photometry_extraction_method}) not "
            f"valid: use either APER or PSF {style.Bcolors.ENDC}"
        )

    #   Conversion of flux to magnitudes
    #   TODO: Move this to the calibration stage, where it makes more sense
    magnitudes, magnitudes_error = utilities.flux_to_magnitudes(
        image.photometry['flux_fit'],
        image.photometry['flux_unc'],
    )

    image.photometry['mags_fit'] = magnitudes
    image.photometry['mags_unc'] = magnitudes_error

    ###
    #   Plot images with extracted stars overlaid
    #
    if plots_for_all_images or (plot_for_reference_image_only
                                and image.pd == id_reference_image):
        utilities.prepare_and_plot_starmap(
            image,
            terminal_logger=terminal_logger,
        )

    if multiprocessing:
        terminal_logger.print_to_terminal('')
    else:
        terminal_output.print_to_terminal('')

    if multiprocessing:
        return copy.deepcopy(image.pd), copy.deepcopy(image.photometry)


def extract_flux(
        observation: Observation, filter_list, name_objects, image_paths,
        output_dir, sigma_object_psf, wcs_method='astrometry',
        force_wcs_determ=False, sigma_value_background_clipping=5.,
        multiplier_background_rms=5., size_epsf_region=25,
        fraction_epsf_stars=0.2, oversampling_factor_epsf=2,
        max_n_iterations_epsf=7, use_initial_positions_epsf=True,
        object_finder_method='IRAF',
        multiplier_background_rms_epsf=5.0,
        multiplier_dao_grouper_epsf=2.0,
        strict_cleaning_epsf_results=True, minimum_n_eps_stars=25,
        reference_image_id=0, strict_epsf_checks=True,
        photometry_extraction_method='PSF', radius_aperture=5.,
        inner_annulus_radius=7., outer_annulus_radius=10.,
        radii_unit='arcsec', cosmic_ray_removal=False,
        limiting_contrast_rm_cosmics=5., read_noise=8.,
        sigma_clipping_value=4.5, saturation_level=65535.,
        plots_for_all_images=False,
        plot_for_reference_image_only=True):
    """
        Extract flux and fill the observation container

        Parameters
        ----------
        observation
            Container object with image series objects for each filter

        filter_list                     : `list` of `string`
            Filter list

        name_objects                    : `list` of `string`
            Name of the object

        image_paths                     : `dictionary`
            Paths to images: key - filter name; value - path

        output_dir                      : `string`
            Path, where the output should be stored.

        sigma_object_psf                : `dictionary`
            Sigma of the objects PSF, assuming it is a Gaussian

        wcs_method                      : `string`, optional
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        force_wcs_determ                : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        sigma_value_background_clipping : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multiplier_background_rms       : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        size_epsf_region                : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        fraction_epsf_stars             : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling_factor_epsf        : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        max_n_iterations_epsf           : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        use_initial_positions_epsf      : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        object_finder_method            : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multiplier_background_rms_epsf  : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multiplier_dao_grouper_epsf     : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning_epsf_results    : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        minimum_n_eps_stars             : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        reference_image_id              : `integer`, optional
            ID of the reference image
            Default is ``0``.

        photometry_extraction_method    : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        radius_aperture                 : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        inner_annulus_radius            : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        outer_annulus_radius            : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        radii_unit                      : `string`, optional
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``arcsec``.

        strict_epsf_checks              : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        cosmic_ray_removal              : `bool`
            If True cosmic rays will be removed from the image.
            Default is ``False``.

        limiting_contrast_rm_cosmics    : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        read_noise                      : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigma_clipping_value            : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        saturation_level                : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        plots_for_all_images            : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_for_reference_image_only   : `boolean`, optional
            If True a star map plots only for the reference image [reference_image_id] is
            created
            Default is ``True``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'tables'),
    )

    ###
    #   Loop over all filter
    #
    for filter_ in filter_list:
        terminal_output.print_to_terminal(
            f"Analyzing {filter_} images",
            style_name='HEADER',
        )

        #   Check input paths
        checks.check_file(image_paths[filter_])

        #   Initialize image series object
        observation.image_series_dict[filter_] = current_image_series = ImageSeries(
            filter_,
            name_objects,
            image_paths[filter_],
            output_dir,
        )

        ###
        #   Find the WCS solution for the image
        #
        try:
            utilities.find_wcs(
                current_image_series,
                reference_image_id=0,
                method=wcs_method,
                force_wcs_determ=force_wcs_determ,
                indent=3,
            )
        except Exception as e:
            #   Get the WCS from one of the other filters, if they have one
            for wcs_filter in filter_list:
                wcs = getattr(
                    observation.image_series_dict[wcs_filter],
                    'wcs',
                    None,
                )
                if wcs is not None:
                    current_image_series.set_wcs(wcs)
                    terminal_output.print_to_terminal(
                        f"WCS could not be determined for filter {filter_}"
                        f"The WCS of filter {wcs_filter} will be used instead."
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
            current_image_series.image_list[reference_image_id],
            sigma_object_psf[filter_],
            sigma_value_background_clipping=sigma_value_background_clipping,
            multiplier_background_rms=multiplier_background_rms,
            size_epsf_region=size_epsf_region,
            fraction_epsf_stars=fraction_epsf_stars,
            oversampling_factor_epsf=oversampling_factor_epsf,
            max_n_iterations_epsf=max_n_iterations_epsf,
            use_initial_positions_epsf=use_initial_positions_epsf,
            object_finder_method=object_finder_method,
            multiplier_background_rms_epsf=multiplier_background_rms_epsf,
            multiplier_dao_grouper_epsf=multiplier_dao_grouper_epsf,
            strict_cleaning_epsf_results=strict_cleaning_epsf_results,
            minimum_n_eps_stars=minimum_n_eps_stars,
            strict_epsf_checks=strict_epsf_checks,
            photometry_extraction_method=photometry_extraction_method,
            radius_aperture=radius_aperture,
            inner_annulus_radius=inner_annulus_radius,
            outer_annulus_radius=outer_annulus_radius,
            radii_unit=radii_unit,
            cosmic_ray_removal=cosmic_ray_removal,
            limiting_contrast_rm_cosmics=limiting_contrast_rm_cosmics,
            read_noise=read_noise,
            sigma_clipping_value=sigma_clipping_value,
            saturation_level=saturation_level,
            plots_for_all_images=plots_for_all_images,
            plot_for_reference_image_only=plot_for_reference_image_only,
        )

    if photometry_extraction_method == 'PSF':
        ###
        #   Plot the ePSFs
        #
        p = mp.Process(
            target=plot.plot_epsf,
            args=(output_dir, observation.get_ref_epsf(),),
        )
        p.start()

        ###
        #   Plot original and residual image
        #
        p = mp.Process(
            target=plot.plot_residual,
            args=(
                name_objects,
                observation.get_ref_img(),
                observation.get_ref_residual_img(),
                output_dir,
            ),
            kwargs={
                'name_object': 'reference image'
            }
        )
        p.start()


def extract_flux_multi(
        observation: Observation, filter_list: list[str],
        image_paths: dict[str, str], output_dir: str,
        sigma_object_psf: dict[str, float],
        n_cores_multiprocessing: int = 6, wcs_method: str = 'astrometry',
        force_wcs_determ: bool = False,
        sigma_value_background_clipping: float = 5.,
        multiplier_background_rms: float = 5., size_epsf_region: int = 25,
        fraction_epsf_stars: float = 0.2, oversampling_factor_epsf: int = 2,
        max_n_iterations_epsf: int = 7, object_finder_method: str = 'IRAF',
        multiplier_background_rms_epsf: float = 5.0,
        multiplier_dao_grouper_epsf: float = 2.0,
        strict_cleaning_epsf_results: bool = True,
        minimum_n_eps_stars: int = 25, strict_epsf_checks: bool = True,
        photometry_extraction_method: str = 'PSF', radius_aperture: float = 5.,
        inner_annulus_radius: float = 7., outer_annulus_radius: float = 10.,
        radii_unit: str = 'arcsec', max_pixel_between_objects: float = 3.,
        own_correlation_option: int = 1, cross_identification_limit: int = 1,
        reference_image_id: int = 0, n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u = 2. * u.arcsec, verbose: bool = False,
        identify_objects_on_image: bool = True,
        plots_for_all_images: bool = False,
        plot_for_reference_image_only: bool = True) -> None:
    """
        Extract flux from multiple images per filter and add results to
        the observation container

        Parameters
        ----------
        observation
            Container object with image series objects for each filter

        filter_list
            Filter list

        image_paths
            Paths to images: key - filter name; value - path

        output_dir
            Path, where the output should be stored.

        sigma_object_psf
            Sigma of the objects PSF, assuming it is a Gaussian

        n_cores_multiprocessing
            Number of cores to use for multicore processing
            Default is ``6``.

        wcs_method
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        force_wcs_determ
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        sigma_value_background_clipping
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multiplier_background_rms
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf_region
            Size of the extraction region in pixel
            Default is `25``.

        fraction_epsf_stars
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling_factor_epsf
            ePSF oversampling factor
            Default is ``2``.

        max_n_iterations_epsf
            Number of ePSF iterations
            Default is ``7``.

        object_finder_method
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multiplier_background_rms_epsf
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multiplier_dao_grouper_epsf
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning_epsf_results
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        minimum_n_eps_stars
            Minimal number of required ePSF stars
            Default is ``25``.

        photometry_extraction_method
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        radius_aperture
            Radius of the stellar aperture
            Default is ``5``.

        inner_annulus_radius
            Inner radius of the background annulus
            Default is ``7``.

        outer_annulus_radius
            Outer radius of the background annulus
            Default is ``10``.

        radii_unit
            Unit of the radii above. Permitted values are
            ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_epsf_checks
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        max_pixel_between_objects
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option
            Option for the srcor correlation function
            Default is ``1``.

        cross_identification_limit
            Cross-identification limit between multiple objects in the current
            image and one object in the reference image. The current image is
            rejected when this limit is reached.
            Default is ``1``.

        reference_image_id
            ID of the reference image
            Default is ``0``.

        n_allowed_non_detections_object
            Maximum number of times an object may not be detected in an image.
            When this limit is reached, the object will be removed.
            Default is ``i`.

        expected_bad_image_fraction
            Fraction of low quality images, i.e. those images for which a
            reduced number of objects with valid source positions are expected.
            Default is ``1.0``.

        protect_reference_obj
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correlation_method
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        separation_limit
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose
            If True additional output will be printed to the command line.
            Default is ``False``.

        identify_objects_on_image
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        plots_for_all_images
            If True star map plots for all stars are created
            Default is ``False``.

        plot_for_reference_image_only
            If True a star map plot only for the reference image is created
            Default is ``True``.
    """
    ###
    #   Check output directories
    #
    checks.check_output_directories(output_dir, os.path.join(output_dir, 'tables'))

    ###
    #   Check image directories
    #
    checks.check_dir(image_paths)

    #   Outer loop over all filter
    for filter_ in filter_list:
        terminal_output.print_to_terminal(
            f"Analyzing {filter_} images",
            style_name='HEADER',
        )

        #   Initialize image series object
        observation.image_series_dict[filter_] = ImageSeries(
            filter_,
            observation.get_object_of_interest_names(),
            image_paths[filter_],
            output_dir,
            reference_image_id=reference_image_id,
        )

        ###
        #   Find the WCS solution for the image
        #
        utilities.find_wcs(
            observation.image_series_dict[filter_],
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
            observation.image_series_dict[filter_],
            n_cores_multiprocessing,
            sigma_object_psf,
            sigma_value_background_clipping=sigma_value_background_clipping,
            multiplier_background_rms=multiplier_background_rms,
            size_epsf_region=size_epsf_region,
            fraction_epsf_stars=fraction_epsf_stars,
            oversampling_factor_epsf=oversampling_factor_epsf,
            max_n_iterations_epsf=max_n_iterations_epsf,
            object_finder_method=object_finder_method,
            multiplier_background_rms_epsf=multiplier_background_rms_epsf,
            multiplier_dao_grouper_epsf=multiplier_dao_grouper_epsf,
            strict_cleaning_epsf_results=strict_cleaning_epsf_results,
            minimum_n_eps_stars=minimum_n_eps_stars,
            strict_epsf_checks=strict_epsf_checks,
            photometry_extraction_method=photometry_extraction_method,
            radius_aperture=radius_aperture,
            inner_annulus_radius=inner_annulus_radius,
            outer_annulus_radius=outer_annulus_radius,
            radii_unit=radii_unit,
            identify_objects_on_image=identify_objects_on_image,
            plots_for_all_images=plots_for_all_images,
            plot_for_reference_image_only=plot_for_reference_image_only,
        )

        ###
        #   Correlate results from all images, while preserving
        #   the variable star
        #
        correlate.correlate_preserve_variable(
            observation,
            filter_,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            cross_identification_limit=cross_identification_limit,
            reference_image_id=reference_image_id,
            n_allowed_non_detections_object=n_allowed_non_detections_object,
            expected_bad_image_fraction=expected_bad_image_fraction,
            protect_reference_obj=protect_reference_obj,
            verbose=verbose,
            plot_reference_only=plot_for_reference_image_only,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
        )


def correlate_calibrate(
        observation: Observation, filter_list: list[str],
        max_pixel_between_objects: int = 3, own_correlation_option: int = 1,
        reference_image_id: int = 0, calibration_method: str = 'APASS',
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None, object_id: int = None,
        magnitude_range: tuple[float, float] = (0., 18.5),
        transformation_coefficients_dict: dict[str, (float | str)] | None = None,
        derive_transformation_coefficients: bool = False,
        plot_sigma: bool = False, photometry_extraction_method: str = '',
        extract_only_circular_region: bool = False, region_radius: float = 600.,
        identify_data_cluster: bool = False, clean_objs_using_pm: bool = False,
        max_distance_cluster: float = 6., find_cluster_para_set: int = 1,
        correlation_method: str = 'astropy', separation_limit: u = 2. * u.arcsec,
        aperture_radius: float = 4., radii_unit: str = 'arcsec',
        convert_magnitudes: bool = False, target_filter_system: str = 'SDSS',
        region_to_select_calibration_stars: regions.RectanglePixelRegion | None = None,
        calculate_zero_point_statistic: bool = True,
        distribution_samples: int = 1000) -> None:
    """
        Correlate photometric extraction results from 2 images and calibrate
        the magnitudes.

        Parameters
        ----------
        observation
            Container object with image series objects for each filter

        filter_list
            List with filter names

        max_pixel_between_objects
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option
            Option for the srcor correlation function
            Default is ``1``.

        reference_image_id
            Reference image ID
            Default is ``0``.

        calibration_method
            Calibration method
            Default is ``APASS``.

        vizier_dict
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``None``.

        path_calibration_file
            Path to the calibration file
            Default is ``None``.

        object_id
            ID of the object
            Default is ``None``.

        magnitude_range
            Magnitude range
            Default is ``(0.,18.5)``.

        transformation_coefficients_dict
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_transformation_coefficients
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``

        plot_sigma
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        photometry_extraction_method
            Applied extraction method. Possibilities: ePSF or APER`
            Default is ``''``.

        extract_only_circular_region
            If True the extracted objects will be filtered such that only
            objects with ``radius`` will be returned.
            Default is ``False``.

        region_radius
            Radius around the object in arcsec.
            Default is ``600``.

        identify_data_cluster
            If True cluster in the Gaia distance and proper motion data
            will be identified.
            Default is ``False``.

        clean_objs_using_pm
            If True only the object list will be clean based on their
            proper motion.
            Default is ``False``.

        max_distance_cluster
            Expected maximal distance of the cluster in kpc. Used to
            restrict the parameter space to facilitate an easy
            identification of the star cluster.
            Default is ``6``.

        find_cluster_para_set
            Parameter set used to identify the star cluster in proper
            motion and distance data.

        correlation_method
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        separation_limit
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        aperture_radius
            Radius of the aperture used to derive the limiting magnitude
            Default is ``4``.

        radii_unit
            Unit of the radii above. Permitted values are
            ``pixel`` and ``arcsec``.
            Default is ``arcsec``.

        convert_magnitudes
            If True the magnitudes will be converted to another
            filter systems specified in `target_filter_system`.
            Default is ``False``.

        target_filter_system
            Photometric system the magnitudes should be converted to
            Default is ``SDSS``.

        region_to_select_calibration_stars
            Region in which to select calibration stars. This is a useful
            feature in instances where not the entire field of view can be
            utilized for calibration purposes.
            Default is ``None``.

        calculate_zero_point_statistic
            If `True` a statistic on the zero points will be calculated.
            Default is ``True``.

        distribution_samples
            Number of samples used for distributions
            Default is `1000`.
    """
    ###
    #   Correlate the stellar positions from the different filter
    #
    correlate.correlate_image_series(
        observation,
        filter_list,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
    )

    ###
    #   Plot image with the final positions overlaid (final version)
    #
    if len(filter_list) > 1:
        utilities.prepare_and_plot_starmap_from_observation(
            observation,
            filter_list,
        )

    ###
    #   Calibrate the magnitudes
    #
    #   Load calibration information
    calib.derive_calibration(
        observation,
        filter_list,
        calibration_method=calibration_method,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        vizier_dict=vizier_dict,
        path_calibration_file=path_calibration_file,
        magnitude_range=magnitude_range,
        region_to_select_calibration_stars=region_to_select_calibration_stars,
    )
    calibration_filters = observation.CalibParameters.column_names

    #   Find filter combinations for which magnitude transformation is possible
    _, usable_filter_combinations = utilities.find_filter_for_magnitude_transformation(
        filter_list,
        calibration_filters,
    )

    for filter_combination in usable_filter_combinations:
        #   Apply calibration and perform magnitude transformation
        trans.apply_calibration(
            observation,
            filter_combination,
            transformation_coefficients_dict=transformation_coefficients_dict,
            derive_transformation_coefficients=derive_transformation_coefficients,
            plot_sigma=plot_sigma,
            photometry_extraction_method=photometry_extraction_method,
            calculate_zero_point_statistic=calculate_zero_point_statistic,
            distribution_samples=distribution_samples,
        )

        ###
        #   Restrict results to specific areas of the image and filter by means
        #   of proper motion and distance using Gaia
        #
        utilities.post_process_results(
            observation,
            filter_combination,
            id_object=object_id,
            extraction_method=photometry_extraction_method,
            extract_only_circular_region=extract_only_circular_region,
            region_radius=region_radius,
            data_cluster=identify_data_cluster,
            clean_objects_using_proper_motion=clean_objs_using_pm,
            max_distance_cluster=max_distance_cluster,
            find_cluster_para_set=find_cluster_para_set,
            convert_magnitudes=convert_magnitudes,
            target_filter_system=target_filter_system,
            distribution_samples=distribution_samples,
        )

        ###
        #   Determine limiting magnitudes
        #
        utilities.derive_limiting_magnitude(
            observation,
            filter_combination,
            reference_image_id,
            aperture_radius=aperture_radius,
            radii_unit=radii_unit,
        )


def calibrate_data_mk_light_curve(
        observation: Observation, filter_list: list[str], output_dir: str,
        valid_filter_combinations: list[list[str]] | None = None,
        binning_factor: float | None = None,
        transformation_coefficients_dict: dict[str, (float | str)] | None = None,
        derive_transformation_coefficients: bool = False,
        reference_image_id: int = 0, calibration_method: str = 'APASS',
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None,
        magnitude_range: tuple[float, float] = (0., 18.5),
        max_pixel_between_objects: float = 3., own_correlation_option: int = 1,
        cross_identification_limit: int = 1,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        photometry_extraction_method: str = '',
        correlation_method: str = 'astropy',
        separation_limit: u =2. * u.arcsec, verbose: bool = False,
        plot_sigma: bool = False,
        region_to_select_calibration_stars: regions.RectanglePixelRegion | None = None,
        calculate_zero_point_statistic: bool = True,
        distribution_samples: int = 1000) -> None:
    """
    Calculate magnitudes, calibrate, and plot light curves

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        List with filter names

    output_dir
        Path, where the output should be stored.

    valid_filter_combinations
        Valid filter combinations to calculate magnitude transformation
        Default is ``None``.

    binning_factor
        Binning factor for the light curve.
        Default is ``None```.

    transformation_coefficients_dict
        Calibration coefficients for the magnitude transformation
        Default is ``None``.

    derive_transformation_coefficients
        If True the magnitude transformation coefficients will be
        calculated from the current data even if calibration coefficients
        are available in the database.
        Default is ``False``

    reference_image_id
        ID of the reference image
        Default is ``0``.

    calibration_method
        Calibration method
        Default is ``APASS``.

    vizier_dict
        Dictionary with identifiers of the Vizier catalogs with valid
        calibration data
        Default is ``None``.

    path_calibration_file
        Path to the calibration file
        Default is ``None``.

    magnitude_range
        Magnitude range
        Default is ``(0.,18.5)``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``1`.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    photometry_extraction_method
        Applied extraction method. Possibilities: ePSF or APER`
        Default is ``''``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    plot_sigma
        If True sigma clipped magnitudes will be plotted.
        Default is ``False``.

    region_to_select_calibration_stars
        Region in which to select calibration stars. This is a useful
        feature in instances where not the entire field of view can be
        utilized for calibration purposes.
        Default is ``None``.

    calculate_zero_point_statistic
        If `True` a statistic on the zero points will be calculated.
        Default is ``True``.

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.
    """
    #   Check if correlation with observed objects can be applied directly
    #   after loading the calibration data. If only one filter and thus one
    #   image series is available, correlation will be applied immediately.
    if len(filter_list) == 1:
        correlate_with_observed_objects = True
    else:
        correlate_with_observed_objects = False

    #   Make coordinates object for variable star
    coordinates_objects_of_interest = observation.objects_of_interest_coordinates
    if coordinates_objects_of_interest is None:
        raise RuntimeError(
            f"SkyCoord object for objects of interest does not exit."
        )
    # coordinates_objects_of_interest = SkyCoord(
    #     ra_object,
    #     dec_object,
    #     unit=(ra_unit, dec_unit),
    #     frame="icrs",
    # )

    #   Load calibration information
    calib.derive_calibration(
        observation,
        filter_list,
        calibration_method=calibration_method,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        vizier_dict=vizier_dict,
        path_calibration_file=path_calibration_file,
        magnitude_range=magnitude_range,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
        region_to_select_calibration_stars=region_to_select_calibration_stars,
        correlate_with_observed_objects=correlate_with_observed_objects,
        reference_image_id=reference_image_id,
        coordinates_obj_to_rm=coordinates_objects_of_interest,
    )
    calibration_filters = observation.CalibParameters.column_names
    terminal_output.print_to_terminal('')

    #   Determine usable filter combinations -> Filters must be in a valid
    #   filter combination for the magnitude transformation and calibration
    #   data must be available for the filter.
    valid_filter, usable_filter_combinations = utilities.find_filter_for_magnitude_transformation(
        filter_list,
        calibration_filters,
        valid_filter_combinations=valid_filter_combinations,
    )

    #   Correlate star positions from the different filter
    if valid_filter:
        correlate.correlate_image_series(
            observation,
            valid_filter,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            cross_identification_limit=cross_identification_limit,
            n_allowed_non_detections_object=n_allowed_non_detections_object,
            expected_bad_image_fraction=expected_bad_image_fraction,
            protect_reference_obj=protect_reference_obj,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            verbose=verbose,
            reference_image_id=reference_image_id,
        )

    ###
    #   Calibrate magnitudes
    #
    #   Perform magnitude transformation
    #   TODO: Convert this to matrix calculation over all filter simultaneously
    processed_filter = []
    for filter_set in usable_filter_combinations:
        #   Apply calibration and perform magnitude transformation
        trans.apply_calibration(
            observation,
            filter_set,
            transformation_coefficients_dict=transformation_coefficients_dict,
            derive_transformation_coefficients=derive_transformation_coefficients,
            photometry_extraction_method=photometry_extraction_method,
            plot_sigma=plot_sigma,
            calculate_zero_point_statistic=calculate_zero_point_statistic,
            distribution_samples=distribution_samples,
        )
        #   TODO: Replace with table_mags_transformed
        calibrated_magnitudes = getattr(
            observation,
            # 'array_mags_not_transformed',
            'array_mags_transformed',
            None,
        )

        for filter_ in filter_set:
            terminal_output.print_to_terminal(
                f"Working on filter: {filter_}",
                style_name='OKBLUE',
            )
            
            ###
            #   Plot light curve
            #
            #   Create a Time object for the observation times
            observation_times = Time(
                observation.image_series_dict[filter_].get_obs_time(),
                format='jd',
            )

            for object_ in observation.objects_of_interest:
                object_name = object_.name
                id_object = object_.id_in_image_series[filter_]
                transit_time = object_.transit_time
                period = object_.period

                #   Create a time series object
                time_series = utilities.mk_time_series(
                    observation_times,
                    calibrated_magnitudes,
                    filter_,
                    id_object,
                )

                #   Write time series
                time_series.write(
                    f'{output_dir}/tables/light_curve_{object_name}_{filter_}'
                    f'_{filter_set[0]}-{filter_set[1]}.dat',
                    format='ascii',
                    overwrite=True,
                )
                time_series.write(
                    f'{output_dir}/tables/light_curve_{object_name}_{filter_}'
                    f'_{filter_set[0]}-{filter_set[1]}.csv',
                    format='ascii.csv',
                    overwrite=True,
                )

                #   Plot light curve over JD
                plot.light_curve_jd(
                    time_series,
                    filter_,
                    f'{filter_}_err',
                    output_dir,
                    name_object=object_name,
                    file_name_suffix=f'_{filter_set[0]}-{filter_set[1]}',
                )

                #   Plot the light curve folded on the period
                if transit_time != '?' and period > 0.:
                    plot.light_curve_fold(
                        time_series,
                        filter_,
                        f'{filter_}_err',
                        output_dir,
                        transit_time,
                        period,
                        binning_factor=binning_factor,
                        name_object=object_name,
                        file_name_suffix=f'_{filter_set[0]}-{filter_set[1]}',
                    )

                processed_filter.append(filter_)

    #   Process those filters for which magnitude transformation is not possible
    for filter_ in filter_list:
        #   Check if filter is not yet processed
        if filter_ not in processed_filter:
            terminal_output.print_to_terminal(
                f"Working on filter: {filter_}",
                style_name='OKBLUE',
            )

            #   Check if calibration data is available
            if f'mag{filter_}' not in calibration_filters:
                terminal_output.print_to_terminal(
                    "Magnitude calibration not possible because no "
                    f"calibration data is available for filter {filter_}. "
                    "Use normalized flux for light curve.",
                    indent=2,
                    style_name='WARNING',
                )

                #   Get image_series
                image_series = observation.image_series_dict[filter_]

                #   Quasi calibration of the flux data
                trans.flux_calibration_image_series(
                    image_series,
                    distribution_samples=distribution_samples,
                )

                #   Normalize data if no calibration magnitudes are available
                trans.flux_normalization_image_series(
                    image_series,
                    distribution_samples=distribution_samples
                )

                #   TODO: Is this necessary? Use return value?
                plot_quantity = image_series.quasi_calibrated_flux_normalized
                # plot_quantity = image_series.quasi_calibrated_flux
            else:
                #   Apply calibration
                trans.apply_calibration(
                    observation,
                    [filter_],
                    photometry_extraction_method=photometry_extraction_method,
                    calculate_zero_point_statistic=calculate_zero_point_statistic,
                    distribution_samples=distribution_samples,
                )
                #   TODO: Replace with table_mags_not_transformed and table_mags_transformed
                plot_quantity = getattr(
                    observation,
                    'array_mags_not_transformed',
                    None,
                )

            #   TODO: Make lightcurve plots for all object + highlight calibration stars
            ###
            #   Plot light curve
            #
            #   Create a Time object for the observation times
            observation_times = Time(
                observation.image_series_dict[filter_].get_obs_time(),
                format='jd',
            )

            for object_ in observation.objects_of_interest:
                object_name = object_.name
                id_object = object_.id_in_image_series[filter_]
                transit_time = object_.transit_time
                period = object_.period

                #   Create a time series object
                time_series = utilities.mk_time_series(
                    observation_times,
                    plot_quantity,
                    filter_,
                    id_object,
                )

                #   Write time series
                time_series.write(
                    f'{output_dir}/tables/light_curve_{object_name}_{filter_}.dat',
                    format='ascii',
                    overwrite=True,
                )
                time_series.write(
                    f'{output_dir}/tables/light_curve_{object_name}_{filter_}.csv',
                    format='ascii.csv',
                    overwrite=True,
                )

                #   Plot light curve over JD
                plot.light_curve_jd(
                    time_series,
                    filter_,
                    f'{filter_}_err',
                    output_dir,
                    name_object=object_name,
                )

                #   Plot the light curve folded on the period
                if transit_time != '?' and period > 0.:
                    plot.light_curve_fold(
                        time_series,
                        filter_,
                        f'{filter_}_err',
                        output_dir,
                        transit_time,
                        period,
                        binning_factor=binning_factor,
                        name_object=object_name,
                    )


def subtract_archive_img_from_img(
        filter_: str, name_objects: list[str], image_paths: dict[str, str],
        output_dir: str, wcs_method: str = 'astrometry',
        plot_comp: bool = True, hips_source: str = 'CDS/P/DSS2/blue') -> None:
    """
        Subtraction of a reference/archival image from the input image.
        The installation of Hotpants is required.

        Parameters
        ----------
        filter_
            Filter identifier

        name_objects
            Name of the object

        image_paths
            Paths to images: key - filter name; value - path

        output_dir
            Path, where the output should be stored.

        wcs_method
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        plot_comp
            If `True` a plot with the original and reference image will
            be created.
            Default is ``True``.

        hips_source
            ID string of the image catalog that will be queried using the
            hips service.
            Default is ``CDS/P/DSS2/blue``.
    """
    ###
    #   Check output directories
    #
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'subtract'),
    )
    output_dir = os.path.join(output_dir, 'subtract')

    ###
    #   Check input path
    #
    for path in image_paths.keys():
        checks.check_file(path)

    ###
    #   Trim image as needed (currently images with < 4*10^6 are required)
    #
    #   Load image
    ccd_image = CCDData.read(image_paths)

    #   Trim
    pixel_max_x = 2501
    # pixel_max_x = 2502
    pixel_max_y = 1599
    ccd_image = ccdp.trim_image(ccd_image[0:pixel_max_y, 0:pixel_max_x])
    ccd_image.meta['NAXIS1'] = pixel_max_x
    ccd_image.meta['NAXIS2'] = pixel_max_y

    #   Save trimmed file
    basename = base_utilities.get_basename(image_paths)
    file_name = f'{basename}_trimmed.fit'
    file_path = os.path.join(output_dir, file_name)
    ccd_image.write(file_path, overwrite=True)

    ###
    #   Initialize image series object
    #
    image_series = ImageSeries(
        filter_,
        name_objects,
        image_paths,
        output_dir,
    )

    ###
    #   Find the WCS solution for the image
    #
    utilities.find_wcs(
        image_series,
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
        wcs=image_series.wcs,
        get_query_payload=False,
        format='fits',
        verbose=True,
    )
    #   Save downloaded file
    hips_hdus.writeto(os.path.join(output_dir, 'hips.fits'), overwrite=True)

    ###
    #   Plot original and reference image
    #
    if plot_comp:
        plot.compare_images(
            output_dir,
            image_series.image_list[0].get_data(),
            hips_hdus[0].data,
        )

    ###
    #   Perform image subtraction
    #
    #   Get image and image data
    ccd_image = image_series.image_list[0].read_image()
    hips_data = hips_hdus[0].data.astype('float64').byteswap().newbyteorder()

    #   Run Hotpants
    subtraction.run_hotpants(
        ccd_image.data,
        hips_data,
        ccd_image.mask,
        np.zeros(hips_data.shape, dtype=bool),
        image_gain=1.,
        # template_gain=1,
        template_gain=None,
        err=ccd_image.uncertainty.array,
        # err=True,
        template_err=True,
        # verbose=True,
        _workdir=output_dir,
        # _exe=exe_path,
    )
