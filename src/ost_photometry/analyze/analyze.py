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
    psf,
)
from photutils.psf import (
    extract_stars,
    SourceGrouper,
    IterativePSFPhotometry,
)
from photutils.detection import IRAFStarFinder
from photutils.background import (
    MMMBackground,
    MADStdBackgroundRMS,
    Background2D,
    MedianBackground,
    LocalBackground,
)
from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus,
)

import ccdproc as ccdp

from astropy.stats import SigmaClip

from astropy.table import Table, Column
from astropy.time import Time
from astropy.nddata import NDData
from astropy.stats import (gaussian_sigma_to_fwhm, sigma_clipped_stats)
from astropy.modeling.fitting import LevMarLSQFitter, LMLSQFitter, TRFLSQFitter
from astropy.coordinates import SkyCoord, name_resolve
import astropy.units as u
from astropy.nddata import CCDData
from astropy import wcs

from regions import RectanglePixelRegion

#   hips2fits module is not in the Ubuntu 22.04 package version
#   of astroquery (0.4.1)
# from astroquery.hips2fits import hips2fits
from astroquery.hips2fits import hips2fitsClass

import regions

import multiprocessing as mp

from . import utilities, calibration_data, calibration, plots, correlate
# from . import subtraction

from .. import style, checks, terminal_output

from .. import utilities as base_utilities
from ..utilities import Image

warnings.filterwarnings('ignore', category=UserWarning, append=True)


############################################################################
#                           Routines & definitions                         #
############################################################################


class ObjectOfInterest:
    def __init__(
            self, ra: str | float | None, dec: str | float | None,
            ra_unit: str | u.quantity.Quantity | None,
            dec_unit: str | u.quantity.Quantity | None, name: str):
        #   Set sky coordinates object
        self.coordinates_object = SkyCoord(
            ra=ra,
            dec=dec,
            unit=(ra_unit, dec_unit),
            frame="icrs"
        )

        #   Set right ascension
        self.ra = self.coordinates_object.ra.degree

        #   Set declination
        self.dec = self.coordinates_object.dec.degree

        #   Set object_name
        self.name = name

        #   ID of object in the image series
        #   Syntax: {'filter': 'id'}
        self.id_in_image_series: dict[str, int] = {}

        #   Set defaults for period and transit time
        self.transit_time: str | None = None
        self.period: float | None = None


class ImageSeries:
    """
        Image series class: Used to handle a series of images,
                            e.g. taken with a specific filter.
    """

    def __init__(self, filter_: str, path: str, output_dir: str,
                 reference_image_id: int = 0):
        #   Setup file list
        if os.path.isdir(path):
            formats: list[str] = [".FIT", ".fit", ".FITS", ".fits"]
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

        #   Sort file list
        file_list.sort()

        #   Check if the id of the reference image is valid
        if reference_image_id > len(file_list):
            raise ValueError(
                f'{style.Bcolors.FAIL} ERROR: Reference image ID '
                '[reference_image_id] is larger than the total number of '
                f'images! -> EXIT {style.Bcolors.ENDC}'
            )

        #   Set filter
        self.filter_: str = filter_

        #   Set ID of the reference image
        self.reference_image_id: int = reference_image_id

        #   Prepare image list
        self.image_list: list[Image] = []

        #   Set path to output directory
        self.out_path: Path = Path(output_dir)

        #   Fill image list
        terminal_output.print_to_terminal(
            "Read images and calculate field of view, pixel scale, etc. ... ",
            indent=2,
        )
        #   TODO: Convert image_list to dictionary?
        for image_id, file_name in enumerate(file_list):
            self.image_list.append(
                #   Prepare image class instance
                Image(image_id, filter_, f'{path}/{file_name}', output_dir)
            )


        #   Set start time for image series
        self.start_jd = self.image_list[0].jd

        #   Set reference image
        self.reference_image = self.image_list[reference_image_id]

        #   Set field of view
        self.field_of_view_x: float | None = getattr(
            self.reference_image,
            'field_of_view_x',
            None,
        )

        #   Set PixelRegion for the field of view
        self.fov_pixel_region: RectanglePixelRegion | None = getattr(
            self.reference_image,
            'fov_pixel_region',
            None,
        )

        #   Set pixel scale
        self.pixel_scale: float | None = getattr(
            self.reference_image,
            'pixel_scale',
            None,
        )

        #   Set coordinates of image center
        self.coordinates_image_center: SkyCoord | None = getattr(
            self.reference_image,
            'coordinates_image_center',
            None,
        )

        #   Set instrument
        self.instrument: str | None = getattr(
            self.reference_image,
            'instrument',
            None,
        )

        #   Get image shape
        self.image_shape = self.reference_image.get_data().shape

        #   Set wcs default
        self.wcs: wcs.WCS | None = None

    #   Set wcs
    def set_wcs(self, w: wcs.WCS) -> None:
        self.wcs = w
        for img in self.image_list:
            img.wcs = w

    #   Get extracted photometry of all images
    def get_photometry(self) -> dict[str, Table]:
        photo_dict: dict[str, Table] = {}
        for img in self.image_list:
            # photo_dict[str(img.pd)] = img.photometry
            photo_dict[str(img.pd)] = getattr(img, 'photometry', None)

        return photo_dict

    #   Get image IDs of all images
    def get_image_ids(self) -> list[int]:
        img_ids: list[int] = []
        for img in self.image_list:
            img_ids.append(img.pd)

        return img_ids

    #   Get sigma clipped mean of the air mass
    def mean_sigma_clip_air_mass(self) -> float:
        am_list: list[float] = []
        for img in self.image_list:
            # am_list.append(img.air_mass)
            am_list.append(getattr(img, 'air_mass', 0.))

        return sigma_clipped_stats(am_list, sigma=1.5)[0]

    #   Get median of the air mass
    def median_air_mass(self) -> float:
        am_list: list[float] = []
        for img in self.image_list:
            # am_list.append(img.air_mass)
            am_list.append(getattr(img, 'air_mass', 0.))

        return np.median(am_list)

    #   Get air mass
    def get_air_mass(self) -> list[float]:
        am_list: list[float] = []
        for img in self.image_list:
            # am_list.append(img.air_mass)
            am_list.append(getattr(img, 'air_mass', 0.))

        return am_list

    #   Get observation times
    def get_observation_time(self) -> np.ndarray:
        obs_time_list: list[float] = []
        for img in self.image_list:
            # obs_time_list.append(img.jd)
            obs_time_list.append(getattr(img, 'jd', 0.))

        return np.array(obs_time_list)

    #   Get median of the observation time
    def median_observation_time(self) -> float:
        obs_time_list: list[float] = []
        for img in self.image_list:
            # obs_time_list.append(img.jd)
            obs_time_list.append(getattr(img, 'jd', 0.))

        return np.median(obs_time_list)

    #   Get list with dictionary and image class objects
    def get_list_dict(self) -> list[dict[str, Image]]:
        dict_list: list[dict[str, Image]] = []
        for img in self.image_list:
            dict_list.append({img.filter_: img})

        return dict_list

    #   Get object positions in pixel coordinates
    #   TODO: improve?
    def get_object_positions_pixel(self) \
            -> tuple[list[Column], list[Column], int]:
        tbl_s = self.get_photometry()
        n_max_list: list[int] = []
        x: list[Column] = []
        y: list[Column] = []
        for i, tbl in enumerate(tbl_s.values()):
            x.append(tbl['x_fit'])
            y.append(tbl['y_fit'])
            n_max_list.append(len(x[i]))

        return x, y, np.max(n_max_list)

    def get_flux_distribution(
            self, distribution_samples: int = 1000
            ) -> list[unc.core.NdarrayDistribution]:
        #   Get data
        tbl_s = list(self.get_photometry().values())

        #   Create list of distributions
        flux_list: list[unc.core.NdarrayDistribution] = []
        for tbl in tbl_s:
            flux_list.append(
                unc.normal(
                    tbl['flux_fit'] * u.mag,
                    std=tbl['flux_err'] * u.mag,
                    n_samples=distribution_samples,
                )
            )

        return flux_list

    def get_flux_array(self) -> tuple[np.ndarray, np.ndarray]:
        #   Get data
        tbl_s = list(self.get_photometry().values())

        #   Expects the number of objects in each table to be the same.
        n_images = len(tbl_s)
        n_objects = len(tbl_s[0])

        flux = np.zeros((n_images, n_objects))
        flux_err = np.zeros((n_images, n_objects))

        for i, tbl in enumerate(tbl_s):
            flux[i] = tbl['flux_fit']
            flux_err[i] = tbl['flux_err']

        return flux, flux_err


class Observation:
    """
        Container class for all data taken during an observation session
    """

    def __init__(self, **kwargs):
        #   Prepare dictionary for image series
        self.image_series_dict: dict[str, ImageSeries] = {}

        #   Add additional keywords
        self.__dict__.update(kwargs)

        #   Check for object of interest
        #   Parameters: right ascension, declination, units, object names,
        #   periods, and transit times
        ra_objects: (list[str] | None) = kwargs.get('ra_objects', None)
        ra_unit: (str | None) = kwargs.get('ra_unit', None)
        dec_objects: (list[str] | None) = kwargs.get('dec_objects', None)
        dec_unit: (str | None) = kwargs.get('dec_unit', None)
        object_names: (list[str] | None) = kwargs.get('object_names', None)
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
        ra_list = []
        dec_list = []
        for object_ in self.objects_of_interest:
            ra_list.append(object_.ra)
            dec_list.append(object_.dec)
            self.objects_of_interest_coordinates = SkyCoord(
                ra_list,
                dec_list,
                unit=(u.degree, u.degree),
                frame="icrs",
            )

        #   Prepare attribute for calibration data
        self.calib_parameters: calibration_data.CalibParameters | None = None

        #   Prepare attribute for calibrated data
        self.table_magnitudes: Table | None = None
        # self.table_mags_transformed: Table | None = None
        # self.table_mags_not_transformed: Table | None = None

    #   Get ePSF objects of all images
    # def get_epsf(self):
    #     epsf_dict = {}
    def get_epsf(self) -> dict[str, list[psf.EPSFModel]]:
        epsf_dict: dict[str, list[psf.EPSFModel | None]] = {}
        for key, image_series in self.image_series_dict.items():
            epsf_list: list[psf.EPSFModel | None] = []
            for img in image_series.image_list:
                epsf_list.append(img.epsf)
            epsf_dict[key] = epsf_list

        return epsf_dict

    #   Get ePSF object of the reference image
    # def get_reference_epsf(self):
    #     epsf_dict = {}
    def get_reference_epsf(self) -> dict[str, psf.EPSFModel]:
        epsf_dict: dict[str, psf.EPSFModel | None] = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_id = image_series.reference_image_id

            img = image_series.image_list[reference_image_id]

            epsf_dict[key] = img.epsf

        return epsf_dict

    #   Get reference image
    def get_reference_image(self) -> dict[str, np.ndarray]:
        img_dict: dict[str, np.ndarray] = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_id = image_series.reference_image_id

            img = image_series.image_list[reference_image_id]

            img_dict[key] = img.get_data()

        return img_dict

    #   Get residual image belonging to the reference image
    def get_reference_image_residual(self) -> dict[str, np.ndarray]:
        img_dict: dict[str, np.ndarray] = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_id = image_series.reference_image_id

            img = image_series.image_list[reference_image_id]

            img_dict[key] = img.residual_image

        return img_dict

    #   Get image series for a specific set of filters
    def get_image_series(
            self, filter_list: list[str] | set[str]) -> dict[str, ImageSeries]:
        image_series_dict: dict[str, ImageSeries] = {}
        for filter_ in filter_list:
            image_series_dict[filter_] = self.image_series_dict[filter_]

        return image_series_dict

    #   Get the IDs of the objects of interest within the detected objects on
    #   the images
    def get_ids_object_of_interest(
            self,
            filter_: str | None = None,
            reference_image_series_id: int | None = None
            ) -> list[int]:
        if filter_ is None and reference_image_series_id is None:
            terminal_output.print_to_terminal(
                "Neither a filter nor an image series ID was provided to "
                "compile the IDs for the objects of interest.The image series ID "
                "is assumed to be 0.",
                style_name='WARNING',
            )
            reference_image_series_id: int = 0

        object_of_interest_ids: list[int] = []
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
    def get_object_of_interest_names(self) -> list[str]:
        name_list: list[str] = []

        for object_ in self.objects_of_interest:
            name_list.append(object_.name)

        return name_list

    #   Get object right ascensions
    def get_object_ras(self) -> list[float]:
        ra_list: list[float] = []

        for object_ in self.objects_of_interest:
            ra_list.append(object_.ra)

        return ra_list

    #   Get object declinations
    def get_object_decs(self) -> list[float]:
        dec_list: list[float] = []

        for object_ in self.objects_of_interest:
            dec_list.append(object_.dec)

        return dec_list

    def extract_flux(
            self, filter_list: list[str], image_paths: dict[str, str],
            output_dir: str, sigma_object_psf: dict[str, float],
            wcs_method: str = 'astrometry', force_wcs_determ: bool = False,
            sigma_value_background_clipping: float = 5.,
            multiplier_background_rms: float = 5., size_epsf_region: int = 25,
            size_extraction_region_epsf: int = 11,
            epsf_fitter: str = 'LMLSQFitter',
            n_iterations_eps_extraction: int = 1,
            fraction_epsf_stars: float = 0.2,
            oversampling_factor_epsf: int = 1,
            max_n_iterations_epsf_determination: int = 7,
            use_initial_positions_epsf: bool = True,
            object_finder_method: str = 'IRAF',
            multiplier_background_rms_epsf: float = 5.0,
            multiplier_grouper_epsf: float = 2.0,
            strict_cleaning_epsf_results: bool = True,
            minimum_n_eps_stars: int = 15,
            reference_image_id: int = 0, strict_epsf_checks: bool = True,
            photometry_extraction_method: str = 'PSF',
            radius_aperture: float = 5., inner_annulus_radius: float = 7.,
            outer_annulus_radius: float = 10., radii_unit: str = 'arcsec',
            cosmic_ray_removal: bool = False,
            limiting_contrast_rm_cosmics: float = 5., read_noise: float = 8.,
            sigma_clipping_value: float = 4.5,
            saturation_level: float = 65535.,
            plots_for_all_images: bool = False,
            plot_for_reference_image_only: bool = True,
            use_wcs_projection_for_star_maps: bool = True,
            file_type_plots: str = 'pdf') -> None:
        """
        Extract flux and fill the observation container

        Parameters
        ----------
        filter_list
            Filter list

        image_paths
            Paths to images: key - filter name; value - path

        output_dir
            Path, where the output should be stored.

        sigma_object_psf
            Sigma of the objects PSF, assuming it is a Gaussian

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
            Default is ``5.0``.

        size_epsf_region
            Size of the extraction region in pixel
            Default is `25``.

        size_extraction_region_epsf
            Size of the extraction region in pixel
            Default is ``11``.

        epsf_fitter
            Fitter function used during ePSF fitting to the data.
            Options are: ``LevMarLSQFitter``, ``LMLSQFitter`` and ``TRFLSQFitter``
            Default is ``LMLSQFitter``.

        n_iterations_eps_extraction
            Number of extraction iterations in the ePSF fit to the data. In certain
            cases, such as very crowded fields, numbers greater than 1 can lead to
            very large CPU loads and recursions within astropy that may exceed the
            defined limits.
            Default is ``1``.

        fraction_epsf_stars
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling_factor_epsf
            ePSF oversampling factor
            Default is ``1``.

        max_n_iterations_epsf_determination
            Number of ePSF iterations
            Default is ``7``.

        use_initial_positions_epsf
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        object_finder_method
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multiplier_background_rms_epsf
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multiplier_grouper_epsf
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning_epsf_results
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        minimum_n_eps_stars
            Minimal number of required ePSF stars
            Default is ``15``.

        reference_image_id
            ID of the reference image
            Default is ``0``.

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
            Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
            Default is ``arcsec``.

        strict_epsf_checks
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        cosmic_ray_removal
            If True cosmic rays will be removed from the image.
            Default is ``False``.

        limiting_contrast_rm_cosmics
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        read_noise
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigma_clipping_value
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        saturation_level
            Saturation limit of the camera chip.
            Default is ``65535``.

        plots_for_all_images
            If True star map plots for all stars are created
            Default is ``False``.

        plot_for_reference_image_only
            If True a star map plots only for the reference image [reference_image_id] is
            created
            Default is ``True``.

        use_wcs_projection_for_star_maps
            If ``True`` the starmap will be plotted with sky coordinates instead
            of pixel coordinates
            Default is ``True``.

        file_type_plots
            Type of plot file to be created
            Default is ``pdf``.
        """
        #   Check output directories
        checks.check_output_directories(
            output_dir,
            os.path.join(output_dir, 'tables'),
        )

        #   Loop over all filter
        for filter_ in filter_list:
            terminal_output.print_to_terminal(
                f"Analyzing {filter_} images",
                style_name='HEADER',
            )

            #   Check input paths
            checks.check_file(image_paths[filter_])

            #   Initialize image series object
            self.image_series_dict[filter_] = current_image_series = ImageSeries(
                filter_,
                image_paths[filter_],
                output_dir,
            )

            #   Find the WCS solution for the image
            try:
                utilities.find_wcs(
                    current_image_series,
                    reference_image_id=0,
                    method=wcs_method,
                    force_wcs_determination=force_wcs_determ,
                    indent=3,
                )
            except RuntimeError as e:
                #   Get the WCS from one of the other filters, if they have one
                for wcs_filter in filter_list:
                    reference_wcs = getattr(
                        self.image_series_dict[wcs_filter],
                        'wcs',
                        None,
                    )
                    if reference_wcs is not None:
                        current_image_series.set_wcs(reference_wcs)
                        terminal_output.print_to_terminal(
                            f"WCS could not be determined for filter {filter_}"
                            f"The WCS of filter {wcs_filter} will be used instead."
                            f"This could lead to problems...",
                            indent=1,
                            style_name='WARNING',
                        )
                        break
                else:
                    raise RuntimeError(e)

            #   Main extraction
            main_extract(
                current_image_series.image_list[reference_image_id],
                sigma_object_psf[filter_],
                sigma_value_background_clipping=sigma_value_background_clipping,
                multiplier_background_rms=multiplier_background_rms,
                size_epsf_region=size_epsf_region,
                size_extraction_region_epsf=size_extraction_region_epsf,
                epsf_fitter=epsf_fitter,
                n_iterations_eps_extraction=n_iterations_eps_extraction,
                fraction_epsf_stars=fraction_epsf_stars,
                oversampling_factor_epsf=oversampling_factor_epsf,
                max_n_iterations_epsf_determination=max_n_iterations_epsf_determination,
                use_initial_positions_epsf=use_initial_positions_epsf,
                object_finder_method=object_finder_method,
                multiplier_background_rms_epsf=multiplier_background_rms_epsf,
                multiplier_grouper_epsf=multiplier_grouper_epsf,
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
                file_type_plots=file_type_plots,
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            )

        if photometry_extraction_method == 'PSF':
            #   Plot the ePSFs
            p = mp.Process(
                target=plots.plot_epsf,
                args=(output_dir, self.get_reference_epsf(),),
                kwargs={'file_type': file_type_plots},
            )
            p.start()

            #   Plot original and residual image
            p = mp.Process(
                target=plots.plot_residual,
                args=(
                    self.get_reference_image(),
                    self.get_reference_image_residual(),
                    output_dir,
                ),
                kwargs={
                    'file_type': file_type_plots,
                }
            )
            p.start()

    def extract_flux_multi(
            self, filter_list: list[str], image_paths: dict[str, str],
            output_dir: str, sigma_object_psf: dict[str, float],
            n_cores_multiprocessing: int = 6, wcs_method: str = 'astrometry',
            force_wcs_determ: bool = False,
            sigma_value_background_clipping: float = 5.,
            multiplier_background_rms: float = 5., size_epsf_region: int = 25,
            size_extraction_region_epsf: int = 11,
            epsf_fitter: str = 'LMLSQFitter',
            n_iterations_eps_extraction: int = 1,
            fraction_epsf_stars: float = 0.2,
            oversampling_factor_epsf: int = 1,
            max_n_iterations_epsf_determination: int = 7,
            use_initial_positions_epsf: bool = True,
            object_finder_method: str = 'IRAF',
            multiplier_background_rms_epsf: float = 5.0,
            multiplier_grouper_epsf: float = 2.0,
            strict_cleaning_epsf_results: bool = True,
            minimum_n_eps_stars: int = 15, strict_epsf_checks: bool = True,
            photometry_extraction_method: str = 'PSF',
            radius_aperture: float = 5., inner_annulus_radius: float = 7.,
            outer_annulus_radius: float = 10., radii_unit: str = 'arcsec',
            max_pixel_between_objects: int = 3,
            own_correlation_option: int = 1,
            cross_identification_limit: int = 1, reference_image_id: int = 0,
            n_allowed_non_detections_object: int = 1,
            expected_bad_image_fraction: float = 1.0,
            protect_reference_obj: bool = True,
            correlation_method: str = 'astropy',
            separation_limit: u.quantity.Quantity = 2. * u.arcsec,
            verbose: bool = False, identify_objects_on_image: bool = True,
            duplicate_handling_object_identification: dict[str, str] | None = None,
            plots_for_all_images: bool = False,
            plot_for_reference_image_only: bool = True,
            use_wcs_projection_for_star_maps: bool = True,
            file_type_plots: str = 'pdf') -> None:
        """
        Extract flux from multiple images per filter and add results to
        the observation container

        Parameters
        ----------
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
            Default is ``5``.

        size_epsf_region
            Size of the extraction region in pixel
            Default is `25``.

        size_extraction_region_epsf
            Size of the extraction region in pixel
            Default is ``11``.

        epsf_fitter
            Fitter function used during ePSF fitting to the data.
            Options are: ``LevMarLSQFitter``, ``LMLSQFitter`` and ``TRFLSQFitter``
            Default is ``LMLSQFitter``.

        n_iterations_eps_extraction
            Number of extraction iterations in the ePSF fit to the data. In certain
            cases, such as very crowded fields, numbers greater than 1 can lead to
            very large CPU loads and recursions within astropy that may exceed the
            defined limits.
            Default is ``1``.

        fraction_epsf_stars
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling_factor_epsf
            ePSF oversampling factor
            Default is ``1``.

        max_n_iterations_epsf_determination
            Number of ePSF iterations
            Default is ``7``.

        use_initial_positions_epsf
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.


        object_finder_method
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multiplier_background_rms_epsf
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multiplier_grouper_epsf
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning_epsf_results
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        minimum_n_eps_stars
            Minimal number of required ePSF stars
            Default is ``15``.

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

        duplicate_handling_object_identification
            Specifies how to handle multiple object identification filtering.
            There are two options for each 'correlation_method':
                'own':     'first_in_list' and 'flux'.  The 'first_in_list'
                            filtering just takes the first obtained result.
                'astropy': 'distance' and 'flux'. The 'distance' filtering is
                            based on the distance between the correlated objects.
                            In this case, the one with the smallest distance is
                            used.
            The second option for both correlation method is based on the measure
            flux values. In this case the largest one is used.
            Default is ``None``.

        plots_for_all_images
            If True star map plots for all stars are created
            Default is ``False``.

        plot_for_reference_image_only
            If True a star map plot only for the reference image is created
            Default is ``True``.

        use_wcs_projection_for_star_maps
            If ``True`` the starmap will be plotted with sky coordinates instead
            of pixel coordinates
            Default is ``True``.

        file_type_plots
            Type of plot file to be created
            Default is ``pdf``.
        """
        #   Check output directories
        checks.check_output_directories(output_dir, os.path.join(output_dir, 'tables'))

        #   Check image directories
        checks.check_dir(image_paths)

        #   Outer loop over all filter
        for filter_ in filter_list:
            terminal_output.print_to_terminal(
                f"Analyzing {filter_} images",
                style_name='HEADER',
            )

            #   Initialize image series object
            self.image_series_dict[filter_] = ImageSeries(
                filter_,
                image_paths[filter_],
                output_dir,
                reference_image_id=reference_image_id,
            )

            #   Find the WCS solution for the image
            utilities.find_wcs(
                self.image_series_dict[filter_],
                reference_image_id=reference_image_id,
                method=wcs_method,
                force_wcs_determination=force_wcs_determ,
                indent=3,
            )

            #   Main extraction of object positions and object fluxes
            #   using multiprocessing
            extract_multiprocessing(
                self.image_series_dict[filter_],
                n_cores_multiprocessing,
                sigma_object_psf,
                sigma_value_background_clipping=sigma_value_background_clipping,
                multiplier_background_rms=multiplier_background_rms,
                size_epsf_region=size_epsf_region,
                size_extraction_region_epsf=size_extraction_region_epsf,
                epsf_fitter=epsf_fitter,
                n_iterations_eps_extraction=n_iterations_eps_extraction,
                fraction_epsf_stars=fraction_epsf_stars,
                oversampling_factor_epsf=oversampling_factor_epsf,
                max_n_iterations_epsf_determination=max_n_iterations_epsf_determination,
                object_finder_method=object_finder_method,
                multiplier_background_rms_epsf=multiplier_background_rms_epsf,
                multiplier_grouper_epsf=multiplier_grouper_epsf,
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
                file_type_plots=file_type_plots,
                use_initial_positions_epsf=use_initial_positions_epsf,
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            )

            #   Correlate results from all images within the current image
            #   series, while preserving the variable objects
            correlate.correlate_preserve_variable(
                self,
                filter_,
                max_pixel_between_objects=max_pixel_between_objects,
                own_correlation_option=own_correlation_option,
                cross_identification_limit=cross_identification_limit,
                reference_image_id=reference_image_id,
                n_allowed_non_detections_object=n_allowed_non_detections_object,
                expected_bad_image_fraction=expected_bad_image_fraction,
                protect_reference_obj=protect_reference_obj,
                verbose=verbose,
                duplicate_handling_object_identification=duplicate_handling_object_identification,
                plot_reference_only=plot_for_reference_image_only,
                correlation_method=correlation_method,
                separation_limit=separation_limit,
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
                file_type_plots=file_type_plots,
            )

    def correlate_calibrate(
            self, filter_list: list[str], max_pixel_between_objects: int = 3,
            own_correlation_option: int = 1, reference_image_id: int = 0,
            calibration_method: str = 'APASS',
            vizier_dict: dict[str, str] | None = None,
            path_calibration_file: str | None = None, object_id: int = None,
            magnitude_range: tuple[float, float] = (0., 18.5),
            transformation_coefficients_dict: dict[str, (float | str)] | None = None,
            derive_transformation_coefficients: bool = False,
            plot_sigma: bool = False, photometry_extraction_method: str = '',
            extract_only_circular_region: bool = False,
            region_radius: float = 600.,
            identify_cluster_gaia_data: bool = False,
            clean_objs_using_pm: bool = False,
            max_distance_cluster: float = 6., find_cluster_para_set: int = 1,
            correlation_method: str = 'astropy',
            separation_limit: u.quantity.Quantity = 2. * u.arcsec,
            aperture_radius: float = 4., radii_unit: str = 'arcsec',
            convert_magnitudes: bool = False,
            target_filter_system: str = 'SDSS',
            region_to_select_calibration_stars: regions.RectanglePixelRegion | None = None,
            calculate_zero_point_statistic: bool = True,
            distribution_samples: int = 1000,
            duplicate_handling_object_identification: dict[str, str] | None = None,
            file_type_plots: str = 'pdf',
            use_wcs_projection_for_star_maps: bool = True) -> None:
        """
        Correlate photometric extraction results from 2 images and calibrate
        the magnitudes.

        Parameters
        ----------
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

        identify_cluster_gaia_data
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
            Default is ``1``.

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

        duplicate_handling_object_identification
            Specifies how to handle multiple object identification filtering during
            object identification.
            There are two options for each 'correlation_method':
                'own':     'first_in_list' and 'flux'.  The 'first_in_list'
                            filtering just takes the first obtained result.
                'astropy': 'distance' and 'flux'. The 'distance' filtering is
                            based on the distance between the correlated objects.
                            In this case, the one with the smallest distance is
                            used.
            The second option for both correlation method is based on the measure
            flux values. In this case the largest one is used.
            Default is ``None``.

        file_type_plots
            Type of plot file to be created
            Default is ``pdf``.

        use_wcs_projection_for_star_maps
            If ``True`` the starmap will be plotted with sky coordinates instead
            of pixel coordinates
            Default is ``True``.
        """
        #   Correlate the stellar positions from the different filter
        correlate.correlate_image_series(
            self,
            filter_list,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            file_type_plots=file_type_plots,
            duplicate_handling_object_identification=duplicate_handling_object_identification,
        )

        #   Plot image with the final positions overlaid (final version)
        if len(filter_list) > 1:
            utilities.prepare_and_plot_starmap_from_observation(
                self,
                filter_list,
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
                file_type_plots=file_type_plots,
            )

        #   Calibrate the magnitudes
        #   Load calibration information
        calibration_data.derive_calibration(
            self,
            filter_list,
            calibration_method=calibration_method,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            vizier_dict=vizier_dict,
            path_calibration_file=path_calibration_file,
            magnitude_range=magnitude_range,
            region_to_select_calibration_stars=region_to_select_calibration_stars,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )
        calibration_filters = self.calib_parameters.column_names

        #   Find filter combinations for which magnitude transformation is possible
        _, usable_filter_combinations = utilities.find_filter_for_magnitude_transformation(
            filter_list,
            calibration_filters,
        )

        for filter_combination in usable_filter_combinations:
            #   Apply calibration and perform magnitude transformation
            calibration.apply_calibration(
                self,
                filter_combination,
                transformation_coefficients_dict=transformation_coefficients_dict,
                derive_transformation_coefficients=derive_transformation_coefficients,
                plot_sigma=plot_sigma,
                photometry_extraction_method=photometry_extraction_method,
                calculate_zero_point_statistic=calculate_zero_point_statistic,
                distribution_samples=distribution_samples,
                file_type_plots=file_type_plots,
                add_progress_bar=False,
            )

            #   Restrict results to specific areas of the image and filter by means
            #   of proper motion and distance using Gaia
            utilities.post_process_results(
                self,
                filter_combination,
                id_object=object_id,
                extraction_method=photometry_extraction_method,
                extract_only_circular_region=extract_only_circular_region,
                region_radius=region_radius,
                identify_cluster_gaia_data=identify_cluster_gaia_data,
                clean_objects_using_proper_motion=clean_objs_using_pm,
                max_distance_cluster=max_distance_cluster,
                find_cluster_para_set=find_cluster_para_set,
                convert_magnitudes=convert_magnitudes,
                target_filter_system=target_filter_system,
                distribution_samples=distribution_samples,
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
                file_type_plots=file_type_plots,
            )

            #   Determine limiting magnitudes
            utilities.derive_limiting_magnitude(
                self,
                filter_combination,
                reference_image_id,
                aperture_radius=aperture_radius,
                radii_unit=radii_unit,
                file_type_plots=file_type_plots,
                use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            )

    def calibrate_data_mk_light_curve(
            self, filter_list: list[str], output_dir: str,
            valid_filter_combinations: list[list[str]] | None = None,
            binning_factor: float | None = None,
            transformation_coefficients_dict: dict[str, (float | str)] | None = None,
            derive_transformation_coefficients: bool = False,
            calibration_method: str = 'APASS',
            vizier_dict: dict[str, str] | None = None,
            path_calibration_file: str | None = None,
            magnitude_range: tuple[float, float] = (0., 18.5),
            max_pixel_between_objects: int = 3, own_correlation_option: int = 1,
            cross_identification_limit: int = 1,
            n_allowed_non_detections_object: int = 1,
            expected_bad_image_fraction: float = 1.0,
            protect_reference_objects: bool = True,
            protect_calibration_objects: bool = True,
            photometry_extraction_method: str = '',
            correlation_method: str = 'astropy',
            separation_limit: u.quantity.Quantity = 2. * u.arcsec,
            verbose: bool = False, plot_sigma: bool = False,
            region_to_select_calibration_stars: regions.RectanglePixelRegion | None = None,
            calculate_zero_point_statistic: bool = True,
            n_cores_multiprocessing_calibration: int | None = None,
            distribution_samples: int = 1000, plot_light_curve_all: bool = False,
            plot_light_curve_calibration_objects: bool = True,
            file_type_plots: str = 'pdf',
            duplicate_handling_object_identification: dict[str, str] = None,
            use_wcs_projection_for_star_maps: bool = True) -> None:
        """
        Calculate magnitudes, calibrate, and plot light curves

        Parameters
        ----------
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

        protect_reference_objects
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        protect_calibration_objects
            If ``False`` calibration objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``False``.

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

        n_cores_multiprocessing_calibration
            Number of core used for multicore processing
            Default is ``None``.

        distribution_samples
            Number of samples used for distributions
            Default is ``1000``.

        plot_light_curve_calibration_objects
            It ``True`` the light curves of all calibration objects
            will be plotted.
            Default is ``True``.

        plot_light_curve_all
            It ``True`` the light curves of all objects will be plotted.
            Default is ``False``.

        file_type_plots
            Type of plot file to be created
            Default is ``pdf``.

        duplicate_handling_object_identification
            Specifies how to handle multiple object identification filtering during
            object identification.
            There are two options for each 'correlation_method':
                'own':     'first_in_list' and 'flux'.  The 'first_in_list'
                            filtering just takes the first obtained result.
                'astropy': 'distance' and 'flux'. The 'distance' filtering is
                            based on the distance between the correlated objects.
                            In this case, the one with the smallest distance is
                            used.
            The second option for both correlation method is based on the measure
            flux values. In this case the largest one is used.
            Default is ``None``.

        use_wcs_projection_for_star_maps
            If ``True`` the starmap will be plotted with sky coordinates instead
            of pixel coordinates
            Default is ``True``.
        """
        #   Clear lightcurve directories
        checks.check_output_directories(f'{output_dir}/lightcurve')
        if plot_light_curve_all:
            checks.clear_directory(Path(f'{output_dir}/lightcurve/by_id'))
        if plot_light_curve_calibration_objects:
            checks.clear_directory(Path(f'{output_dir}/lightcurve/calibration'))

        #   Check if correlation with observed objects can be applied directly
        #   after loading the calibration data. If only one filter and thus one
        #   image series is available, correlation will be applied immediately.
        if len(filter_list) == 1:
            correlate_with_observed_objects = True
        else:
            correlate_with_observed_objects = False

        #   Get coordinates for objects of interest
        coordinates_objects_of_interest = self.objects_of_interest_coordinates
        if coordinates_objects_of_interest is None:
            raise RuntimeError(
                f"SkyCoord object for objects of interest does not exit."
            )

        #   Load calibration information
        calibration_data.derive_calibration(
            self,
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
            coordinates_obj_to_rm=coordinates_objects_of_interest,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )
        calibration_filters = self.calib_parameters.column_names
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
                self,
                valid_filter,
                max_pixel_between_objects=max_pixel_between_objects,
                own_correlation_option=own_correlation_option,
                cross_identification_limit=cross_identification_limit,
                n_allowed_non_detections_object=n_allowed_non_detections_object,
                expected_bad_image_fraction=expected_bad_image_fraction,
                protect_reference_obj=protect_reference_objects,
                protect_calibration_objects=protect_calibration_objects,
                correlation_method=correlation_method,
                separation_limit=separation_limit,
                verbose=verbose,
                file_type_plots=file_type_plots,
                duplicate_handling_object_identification=duplicate_handling_object_identification,
            )

        #   Calibrate magnitudes
        #
        #   Get IDs of calibration stars
        ids_calibration_objects = self.calib_parameters.ids_calibration_objects
        if ids_calibration_objects is None:
            ids_calibration_objects = [ids_calibration_objects]

        #   Perform magnitude transformation
        #   TODO: Convert this to matrix calculation over all filter simultaneously
        processed_filter = []
        for filter_set in usable_filter_combinations:
            #   Apply calibration and perform magnitude transformation
            calibration.apply_calibration(
                self,
                filter_set,
                transformation_coefficients_dict=transformation_coefficients_dict,
                derive_transformation_coefficients=derive_transformation_coefficients,
                photometry_extraction_method=photometry_extraction_method,
                plot_sigma=plot_sigma,
                calculate_zero_point_statistic=calculate_zero_point_statistic,
                n_cores_multiprocessing=n_cores_multiprocessing_calibration,
                distribution_samples=distribution_samples,
                file_type_plots=file_type_plots,
            )

            for filter_ in filter_set:
                terminal_output.print_to_terminal(
                    f"Create light curves in filter: {filter_}",
                    style_name='OKBLUE',
                )

                #   Get IDs of the object of interests
                ids_object_of_interest = self.get_ids_object_of_interest(
                    filter_=filter_
                )

                #   Plot light curve
                #
                #   Create a Time object for the observation times
                observation_times = Time(
                    self.image_series_dict[filter_].get_observation_time(),
                    format='jd',
                )

                for object_ in self.objects_of_interest:
                    utilities.prepare_plot_time_series(
                        self.table_magnitudes,
                        observation_times,
                        filter_,
                        object_.name,
                        object_.id_in_image_series[filter_],
                        output_dir,
                        binning_factor,
                        transit_time=object_.transit_time,
                        period=object_.period,
                        file_name_suffix=f'_{filter_set[0]}-{filter_set[1]}',
                        file_type_plots=file_type_plots,
                    )

                if plot_light_curve_all:
                    for index in np.arange(len(self.table_magnitudes)):
                        if (index not in ids_object_of_interest
                                and index not in ids_calibration_objects):
                            p = mp.Process(
                                target=utilities.prepare_plot_time_series,
                                args=(
                                    self.table_magnitudes,
                                    observation_times,
                                    filter_,
                                    str(index),
                                    index,
                                    output_dir,
                                    binning_factor,
                                ),
                                kwargs={
                                    'file_name_suffix': f'_{filter_set[0]}-{filter_set[1]}',
                                    'subdirectory': '/by_id',
                                    'file_type_plots': file_type_plots,
                                }
                            )
                            p.start()

                if (plot_light_curve_calibration_objects
                        and ids_calibration_objects.any()):
                    for index in ids_calibration_objects:
                        p = mp.Process(
                            target=utilities.prepare_plot_time_series,
                            args=(
                                self.table_magnitudes,
                                observation_times,
                                filter_,
                                str(index),
                                index,
                                output_dir,
                                binning_factor,
                            ),
                            kwargs={
                                'file_name_suffix': f'_{filter_set[0]}-{filter_set[1]}',
                                'subdirectory': '/calibration',
                                'file_type_plots': file_type_plots,
                            }
                        )
                        p.start()

                processed_filter.append(filter_)

        #   Process those filters for which magnitude transformation is not possible
        for filter_ in filter_list:
            #   Check if filter is not yet processed
            if filter_ not in processed_filter:
                terminal_output.print_to_terminal(
                    f"Working on filter: {filter_}",
                    style_name='OKBLUE',
                )

                #   Get IDs of the object of interests
                ids_object_of_interest = self.get_ids_object_of_interest(
                    filter_=filter_
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
                    image_series = self.image_series_dict[filter_]

                    #   Quasi calibration of the flux data
                    quasi_calibrated_flux = calibration.quasi_flux_calibration_image_series(
                        image_series,
                        distribution_samples=distribution_samples,
                    )

                    #   Normalize data if no calibration magnitudes are available
                    quasi_calibrated_normalized_flux = calibration.flux_normalization_image_series(
                        image_series,
                        quasi_calibrated_flux=quasi_calibrated_flux,
                        distribution_samples=distribution_samples
                    )

                    plot_quantity = quasi_calibrated_normalized_flux
                else:
                    #   Apply calibration
                    calibration.apply_calibration(
                        self,
                        [filter_],
                        photometry_extraction_method=photometry_extraction_method,
                        calculate_zero_point_statistic=calculate_zero_point_statistic,
                        n_cores_multiprocessing=n_cores_multiprocessing_calibration,
                        distribution_samples=distribution_samples,
                        file_type_plots=file_type_plots,
                    )
                    plot_quantity = self.table_magnitudes

                #   Plot light curve
                #
                #   Create a Time object for the observation times
                observation_times = Time(
                    self.image_series_dict[filter_].get_observation_time(),
                    format='jd',
                )

                for object_ in self.objects_of_interest:
                    utilities.prepare_plot_time_series(
                        plot_quantity,
                        observation_times,
                        filter_,
                        object_.name,
                        object_.id_in_image_series[filter_],
                        output_dir,
                        binning_factor,
                        transit_time=object_.transit_time,
                        period=object_.period,
                        file_type_plots=file_type_plots,
                        calibration_type='simple',
                    )

                if plot_light_curve_all:
                    if isinstance(plot_quantity, unc.core.NdarrayDistribution):
                        shape_array = plot_quantity.shape
                        index_array = np.arange(shape_array[1])
                    else:
                        index_array = np.arange(len(plot_quantity))
                    for index in index_array:
                        if (index not in ids_object_of_interest
                                and index not in ids_calibration_objects):
                            p = mp.Process(
                                target=utilities.prepare_plot_time_series,
                                args=(
                                    plot_quantity,
                                    observation_times,
                                    filter_,
                                    str(index),
                                    index,
                                    output_dir,
                                    binning_factor,
                                ),
                                kwargs={
                                    'calibration_type': 'simple',
                                    'subdirectory': '/by_id',
                                    'file_type_plots': file_type_plots,
                                }
                            )
                            p.start()

                if (plot_light_curve_calibration_objects
                        and ids_calibration_objects != [None]):
                    for index in ids_calibration_objects:
                        p = mp.Process(
                            target=utilities.prepare_plot_time_series,
                            args=(
                                plot_quantity,
                                observation_times,
                                filter_,
                                str(index),
                                index,
                                output_dir,
                                binning_factor,
                            ),
                            kwargs={
                                'calibration_type': 'simple',
                                'subdirectory': '/calibration',
                                'file_type_plots': file_type_plots,
                            }
                        )
                        p.start()


def rm_cosmic_rays(
        image: Image, limiting_contrast: float = 5., read_noise: float = 8.,
        sigma_clipping_value: float = 4.5, saturation_level: float = 65535.,
        verbose: bool = False, add_mask: bool = True,
        terminal_logger: terminal_output.TerminalLog | None = None) -> None:
    """
        Remove cosmic rays

        Parameters
        ----------
        image
            Object with all image specific properties

        limiting_contrast
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        read_noise
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigma_clipping_value
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        saturation_level
            Saturation limit of the camera chip.
            Default is ``65535``.

        verbose
            If True additional output will be printed to the command line.
            Default is ``False``.

        add_mask
            If True add hot and bad pixel mask to the reduced science images.
            Default is ``True``.

        terminal_logger
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
            str(image.out_path),
            'cosmics_rm',
            file_name,
        )

        #   Check if the 'cosmics_rm' directory already exits.
        #   If not, create it.
        checks.check_output_directories(os.path.join(str(image.out_path), 'cosmics_rm'))

        #   Save image
        reduced.write(image.path, overwrite=True)


def determine_background(
        image: Image, sigma_background: float = 5.,
        two_d_background: bool = True, apply_background: bool = True,
        verbose: bool = False) -> tuple[float, float]:
    """
    Determine background, using photutils

    Parameters
    ----------
    image
        Object with all image specific properties

    sigma_background
        Sigma used for the sigma clipping of the background
        Default is ``5.``.

    two_d_background
        If True a 2D background will be estimated and subtracted.
        Default is ``True``.

    apply_background
        If True path and file name will be set to the background
        subtracted images, so that those will automatically be used in
        further processing steps.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    Returns
    -------
    background_value
        Image background

    rms_background
        Root mean square of the image background
    """
    if verbose:
        terminal_output.print_to_terminal(
            f"Determine background: {image.filter_} filter",
            indent=2,
        )

    #   Load image data
    ccd = image.read_image()

    #   Set up sigma clipping
    sigma_clip = SigmaClip(sigma=sigma_background)

    #   Calculate background RMS
    background_rms = MADStdBackgroundRMS(sigma_clip=sigma_clip)
    rms_background = background_rms(ccd.data)

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
        background_value = bkg.background_median
    else:
        #   Estimate 1D background
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
        background_value = mmm_bkg.calc_background(ccd.data)

        #   Remove background
        image_no_bg = ccd.subtract(background_value)

        #   Put metadata back on the image, because it is lost while
        #   subtracting the background
        image_no_bg.meta = ccd.meta
        image_no_bg.meta['HIERARCH'] = '1D background removed'

        #   Add Header keyword to mark the file as background subtracted
        image_no_bg.meta['NO_BG'] = True

    #   Define name and save image
    file_name = f'{base_utilities.get_basename(image.filename)}_no_bkg.fit'
    output_path = image.out_path / 'no_bkg'
    checks.check_output_directories(output_path)
    image_no_bg.write(output_path / file_name, overwrite=True)

    #   Set new path and file
    #   -> Background subtracted image will be used in further processing steps
    if apply_background:
        image.path = output_path / file_name
        image.filename = file_name

    return background_value, rms_background


def find_stars(
        image: Image, sigma_object_psf: float, rms_background: float,
        multiplier_background_rms: float = 5., method: str = 'IRAF',
        terminal_logger: terminal_output.TerminalLog | None = None,
        indent: int = 2) -> None:
    """
        Find the stars on the images, using photutils and search and select
        stars for the ePSF stars

        Parameters
        ----------
        image
            Object with all image specific properties

        sigma_object_psf
            Sigma of the objects PSF, assuming it is a Gaussian

        rms_background
            Root mean square of the image background

        multiplier_background_rms
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5``.

        method
            Finder method DAO or IRAF
            Default is ``IRAF``.

        terminal_logger
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent
            Indentation for the console output lines
            Default is ``2``.
    """
    if terminal_logger is not None:
        terminal_logger.add_to_cache("Identify stars", indent=indent)
    else:
        terminal_output.print_to_terminal("Identify stars", indent=indent)

    #   Load image data
    ccd = image.read_image()

    #   Use background RMS as sigma
    sigma = rms_background

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


def check_epsf_stars(
        image: Image, size_epsf_region: int = 25, minimum_n_stars: int = 25,
        fraction_epsf_stars: float = 0.2,
        terminal_logger: terminal_output.TerminalLog | None = None,
        strict_epsf_checks: bool = True, indent: int = 2) -> Table:
    """
    Select ePSF stars and check if there are enough

    Parameters
    ----------
    image
        Object with all image specific properties

    size_epsf_region
        Size of the extraction region in pixel
        Default is ``25``.

    minimum_n_stars
        Minimal number of stars required for the ePSF calculations
        Default is ``25``.

    fraction_epsf_stars
        Fraction of all stars that should be used to calculate the ePSF
        Default is ``0.2``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    strict_epsf_checks
        If True a stringent test of the ePSF conditions is applied.
        Default is ``True``.

    indent
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
    identification_string = f'{image.pd}. {image.filter_}'

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

    #   Return ePSF stars
    return tbl_epsf_stars


def determine_epsf(
        image: Image, epsf_star_positions: Table, size_epsf_region: int = 25,
        oversampling_factor: int = 2, max_n_iterations: int = 7,
        minimum_n_stars: int = 25, multiprocess_plots: bool = True,
        terminal_logger: terminal_output.TerminalLog | None = None,
        file_type_plots: str = 'pdf', indent: int = 2) -> None:
    """
    Main function to determine the ePSF, using photutils

    Parameters
    ----------
    image
        Object with all image specific properties

    epsf_star_positions
        Table with position of the ePSF stars

    size_epsf_region
        Size of the extraction region in pixel
        Default is ``25``.

    oversampling_factor
        ePSF oversampling factor
        Default is ``2``.

    max_n_iterations
        Number of ePSF iterations
        Default is ``7``.

    minimum_n_stars
        Minimal number of stars required for the ePSF calculations
        Default is ``25``.

    multiprocess_plots
        If True multiprocessing is used for plotting.
        Default is ``True``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    #   Get image data
    data = image.get_data()

    #   Number of ePSF stars
    n_epsf = len(epsf_star_positions)

    if n_epsf < minimum_n_stars:
        terminal_logger.add_to_cache(
            f"The number of ePSF stars is less than required."
            f"{n_epsf} ePSF stars available. {minimum_n_stars} were "
            "requested.",
            indent=indent,
            style_name='WARNING',
        )

    #   Get object name
    # name_object = image.object_name

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
    stars_tbl['x'] = epsf_star_positions['xcentroid']
    stars_tbl['y'] = epsf_star_positions['ycentroid']

    #   Put image into NDData container (required by "extract_stars")
    nd_data = NDData(data=data)

    #   Extract cutouts of the selected stars
    stars = extract_stars(nd_data, stars_tbl, size=size_epsf_region)

    #   Combine plot identification string
    string = f'img-{image.pd}-{image.filter_}'

    #   Get output directory
    output_dir = image.out_path.name

    #   Plot the brightest ePSF stars
    if multiprocess_plots:
        p = mp.Process(
            target=plots.plot_cutouts,
            args=(output_dir, stars, string),
            kwargs={'file_type': file_type_plots, }
        )
        p.start()
    else:
        plots.plot_cutouts(
            output_dir,
            stars,
            string,
            terminal_logger=terminal_logger,
            file_type=file_type_plots,
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


def extraction_epsf(
        image: Image, sigma_object_psf: float, background_rms: float,
        sigma_background: float = 5., use_initial_positions: bool = True,
        finder_method: str = 'IRAF', size_extraction_region: int = 11,
        epsf_fitter: str = 'LMLSQFitter', n_iterations_eps_extraction: int = 1,
        multiplier_background_rms: float = 5.0,
        multiplier_grouper: float = 2.0,
        strict_cleaning_results: bool = True,
        terminal_logger: terminal_output.TerminalLog | None = None,
        rm_background: bool = False, indent: int = 2) -> None:
    """
    Main function to perform the eEPSF photometry, using photutils

    Parameters
    ----------
    image
        Object with all image specific properties

    sigma_object_psf
        Sigma of the objects PSF, assuming it is a Gaussian

    background_rms
        Root mean square of the image background

    sigma_background
        Sigma used for the sigma clipping of the background
        Default is ``5.``.

    use_initial_positions
        If True the initial positions from a previous object
        identification procedure will be used. If False the objects
        will be identified by means of the ``method_finder`` method.
        Default is ``True``.

    finder_method
        Finder method DAO or IRAF
        Default is ``IRAF``.

    size_extraction_region
        Size of the extraction region in pixel
        Default is ``11``.

    epsf_fitter
        Fitter function used during ePSF fitting to the data.
        Options are: ``LevMarLSQFitter``, ``LMLSQFitter`` and ``TRFLSQFitter``
        Default is ``LMLSQFitter``.

    n_iterations_eps_extraction
        Number of extraction iterations in the ePSF fit to the data. In certain
        cases, such as very crowded fields, numbers greater than 1 can lead to
        very large CPU loads and recursions within astropy that may exceed the
        defined limits.
        Default is ``1``.

    multiplier_background_rms
        Multiplier for the background RMS, used to calculate the
        threshold to identify stars
        Default is ``5.0``.

    multiplier_grouper
        Multiplier for the DAO grouper
        Default is ``2.0``.

    strict_cleaning_results
        If True objects with negative flux uncertainties will be removed
        Default is ``True``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    rm_background
        If True the background will be estimated and considered.
        Default is ``False``. -> It is expected that the background
        was removed before.

    indent
        Indentation for the console output lines
        Default is ``2`.
    """
    #   Get output path
    output_path = image.out_path

    #   Check output directories
    checks.check_output_directories(
        output_path,
        output_path / 'tables',
    )

    #   Get image data
    data = image.get_data()

    #   Get image uncertainty information
    error = image.get_error()

    #   Get image mask
    image_mask = image.get_mask()

    #   Get filter
    filter_ = image.filter_

    #   Get already identified objects (position and flux)
    initial_positions = None
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
    if epsf_fitter == 'LevMarLSQFitter':
        fitter = LevMarLSQFitter()
    elif epsf_fitter == 'LMLSQFitter':
        fitter = LMLSQFitter()
    elif epsf_fitter == 'TRFLSQFitter':
        fitter = TRFLSQFitter()
    else:
        terminal_output.print_to_terminal(
            f"WARNING: Fitter method ({epsf_fitter}) for ePSF "
            f"extraction not known: Switching to LMLSQFitter.",
            style_name='WARNING',
            indent=indent,
        )
        fitter = LMLSQFitter()

    #   Make sure the size of the extraction region is uneven
    if size_extraction_region % 2 == 0:
        size_extraction_region = size_extraction_region + 1

    #   Set up sigma clipping
    if rm_background:
        sigma_clip = SigmaClip(sigma=sigma_background)
        mmm_bkg = MMMBackground(sigma_clip=sigma_clip)
        local_bkg_estimator = LocalBackground(7, 10, mmm_bkg)
    else:
        local_bkg_estimator = None

    #   Group sources into clusters based on a minimum separation distance
    source_grouper = SourceGrouper(
        min_separation=multiplier_grouper * sigma_object_psf * gaussian_sigma_to_fwhm
    )

    #  Set up the overall class to extract the data
    photometry = IterativePSFPhotometry(
        psf_model=epsf,
        fit_shape=(size_extraction_region, size_extraction_region),
        # fit_shape=(11, 11),
        finder=finder,
        grouper=source_grouper,
        fitter=fitter,
        maxiters=n_iterations_eps_extraction,
        localbkg_estimator=local_bkg_estimator,
        mode='all',
        aperture_radius=(size_extraction_region - 1) / 2
        # aperture_radius=(11 - 1) / 2
    )

    #   Extract the photometry and make a t\\able
    if use_initial_positions:
        result_tbl = photometry(
            data=data,
            error=error,
            mask=image_mask,
            init_params=initial_positions,
        )
    else:
        result_tbl = photometry(data=data, error=error, mask=image_mask)

    #   Check if result table contains a 'flux_err' column
    #   For some reason, it's missing for some extractions....
    #   The following has be deactivated for test purposes (20.08.2024)
    if 'flux_err' not in result_tbl.colnames:
        #   Calculate a very, very rough approximation of the uncertainty
        #   by means of the actual extraction result 'flux_fit' and the
        #   early estimate 'flux_0'
        estimated_uncertainty = np.absolute(
            result_tbl['flux_fit'] - result_tbl['flux_init']
        )
        result_tbl.add_column(estimated_uncertainty, name='flux_err')

    #   Clean output for objects with NANs in uncertainties
    try:
        uncertainty_mask = np.invert(np.isnan(result_tbl['flux_err'].value))
        result_tbl = result_tbl[uncertainty_mask]
    except KeyError:
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
            bad_results = np.where(result_tbl['flux_err'].data < 0.)
            n_bad_objects += len(bad_results)
            result_tbl.remove_rows(bad_results)
    except KeyError:
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
    except KeyError:
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
    except KeyError:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nTable produced by "
            "IterativePSFPhotometry is empty after cleaning up "
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
    # residual_image = photometry.get_residual_image()
    residual_image = photometry.make_residual_image(
        data,
        (size_extraction_region, size_extraction_region)
    )

    #   Add photometry and residual image to image class
    image.photometry = result_tbl
    image.residual_image = residual_image


def compute_aperture_photometry_uncertainties(
        flux_variance: np.ndarray, aperture_area: float, annulus_area: float,
        uncertainty_background: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    This function is largely borrowed from the Space Telescope Science
    Institute's wfc3_photometry package:

    https://github.com/spacetelescope/wfc3_photometry

    It computes the flux errors for aperture photometry using the DAOPHOT
    style computation:

    err = sqrt (Poisson_noise / gain
        + ap_area * stdev**2
        + ap_area**2 * stdev**2 / nsky)

    Parameters
    ----------
    flux_variance
        Extracted aperture flux data or the error^2 of the extraction
        if available -> proxy for the Poisson noise

    aperture_area
        Photometric aperture area

    annulus_area
        Sky annulus area

    uncertainty_background
        Uncertainty in the sky measurement

    gain
        Electrons per ADU
        Default is ``1.0``. Usually we already work with gain corrected
        data.

    Returns
    -------
    flux_error
        Uncertainty of flux measurements
    """

    #   Calculate flux error as above
    bg_variance_terms = ((aperture_area * uncertainty_background ** 2.) *
                         (1. + aperture_area / annulus_area))
    variance = flux_variance / gain + bg_variance_terms
    flux_error = variance ** .5

    return flux_error


def define_apertures(
        image: Image, aperture_radius: float, inner_annulus_radius: float,
        outer_annulus_radius: float, unit_radii: str
        ) -> tuple[CircularAperture, CircularAnnulus]:
    """
    Define stellar and background apertures

    Parameters
    ----------
    image
        Object with all image specific properties

    aperture_radius
        Radius of the stellar aperture

    inner_annulus_radius
        Inner radius of the background annulus

    outer_annulus_radius
        Outer radius of the background annulus

    unit_radii
        Unit of the radii above. Permitted values are ``pixel``
        and ``arcsec``.

    Returns
    -------
    aperture
        Stellar aperture

    annulus_aperture
        Background annulus
    """
    #   Get position information
    tbl = image.positions

    #   Extract positions and prepare a position list
    try:
        x_positions = tbl['x_fit']
        y_positions = tbl['y_fit']
    except KeyError:
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
    pixel_scale = image.pixel_scale
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


def background_simple(
        image: Image, annulus_aperture: CircularAnnulus
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate background from annulus

    Parameters
    ----------
    image
        Object with all image specific properties

    annulus_aperture
        Background annulus

    Returns
    -------
    bkg_median
        Median of the background

    bkg_standard_deviation
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


def extraction_aperture(
        image: Image, radius_aperture: float, inner_annulus_radius: float,
        outer_annulus_radius: float, radii_unit: str = 'pixel',
        background_estimate_simple: bool = False,
        plot_aperture_positions: bool = False,
        terminal_logger: terminal_output.TerminalLog | None = None,
        file_type_plots: str = 'pdf', indent: int = 2) -> None:
    """
    Perform aperture photometry using the photutils aperture package

    Parameters
    ----------
    image
        Object with all image specific properties

    radius_aperture
        Radius of the stellar aperture

    inner_annulus_radius
        Inner radius of the background annulus

    outer_annulus_radius
        Outer radius of the background annulus

    radii_unit
        Unit of the radii above. Permitted values are ``pixel`` and
        ``arcsec``.
        Default is ``pixel``.

    background_estimate_simple
        If True the background will be extract by a simple algorithm that
        calculates the median within the background annulus. If False the
        background will be extracted using
        photutils.aperture.aperture_photometry.
        Default is ``False``.

    plot_aperture_positions
        IF true a plot showing the apertures in relation to image is
        created.
        Default is ``False``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    #   Load image data and uncertainty
    ccd = image.read_image()
    data = ccd.data
    uncertainty = ccd.uncertainty.array

    #   Get filter
    filter_ = image.filter_

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

    #   Define apertures
    aperture, annulus_aperture = define_apertures(
        image,
        radius_aperture,
        inner_annulus_radius,
        outer_annulus_radius,
        radii_unit,
    )

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

    photometry_tbl['flux_err'] = compute_aperture_photometry_uncertainties(
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
        pixel_scale = image.pixel_scale
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

    #   Plot star map with aperture overlay
    if plot_aperture_positions:
        plots.plot_apertures(
            image.out_path.name,
            data,
            aperture,
            annulus_aperture,
            f'{filter_}_{image.pd}',
            file_type=file_type_plots,
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


def extract_multiprocessing(
        image_series: ImageSeries, n_cores_multiprocessing: int,
        sigma_object_psf: dict[str, float],
        sigma_value_background_clipping: float = 5.,
        multiplier_background_rms: float = 5., size_epsf_region: int = 25,
        size_extraction_region_epsf: int = 11,
        epsf_fitter: str = 'LMLSQFitter',
        n_iterations_eps_extraction: int = 1,
        fraction_epsf_stars: float = 0.2,
        oversampling_factor_epsf: int = 1,
        max_n_iterations_epsf_determination: int = 7,
        use_initial_positions_epsf: bool = True,
        object_finder_method: str = 'IRAF',
        multiplier_background_rms_epsf: float = 5.0,
        multiplier_grouper_epsf: float = 2.0,
        strict_cleaning_epsf_results: bool = True,
        minimum_n_eps_stars: int = 15,
        photometry_extraction_method: str = 'PSF',
        radius_aperture: float = 5., inner_annulus_radius: float = 7.,
        outer_annulus_radius: float = 10., radii_unit: str = 'arcsec',
        strict_epsf_checks: bool = True,
        identify_objects_on_image: bool = True,
        plots_for_all_images: bool = False,
        plot_for_reference_image_only: bool = True,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf'):
    """
    Extract flux and object positions using multiprocessing

    Parameters
    ----------
    image_series
        Image series object with all image data taken in a specific
        filter

    n_cores_multiprocessing
        Number of cores to use during multiprocessing.

    sigma_object_psf
        Sigma of the objects PSF, assuming it is a Gaussian

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

    size_extraction_region_epsf
        Size of the extraction region in pixel
        Default is ``11``.

    epsf_fitter
        Fitter function used during ePSF fitting to the data.
        Options are: ``LevMarLSQFitter``, ``LMLSQFitter`` and ``TRFLSQFitter``
        Default is ``LMLSQFitter``.

    n_iterations_eps_extraction
        Number of extraction iterations in the ePSF fit to the data. In certain
        cases, such as very crowded fields, numbers greater than 1 can lead to
        very large CPU loads and recursions within astropy that may exceed the
        defined limits.
        Default is ``1``.

    fraction_epsf_stars
        Fraction of all stars that should be used to calculate the ePSF
        Default is ``0.2``.

    oversampling_factor_epsf
        ePSF oversampling factor
        Default is ``1``.

    max_n_iterations_epsf_determination
        Number of ePSF iterations
        Default is ``7``.
        Default is ``7``.

    use_initial_positions_epsf
        If True the initial positions from a previous object
        identification procedure will be used. If False the objects
        will be identified by means of the ``method_finder`` method.
        Default is ``True``.

    object_finder_method
        Finder method DAO or IRAF
        Default is ``IRAF``.

    multiplier_background_rms_epsf
        Multiplier for the background RMS, used to calculate the
        threshold to identify stars
        Default is ``5.0``.

    multiplier_grouper_epsf
        Multiplier for the DAO grouper
        Default is ``5.0``.

    strict_cleaning_epsf_results
        If True objects with negative flux uncertainties will be removed
        Default is ``True``.

    minimum_n_eps_stars
        Minimal number of required ePSF stars
        Default is ``15``.

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
        Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
        Default is ``pixel``.

    strict_epsf_checks
        If True a stringent test of the ePSF conditions is applied.
        Default is ``True``.

    identify_objects_on_image
        If `True` the objects on the image will be identified. If `False`
        it is assumed that object identification was performed in advance.
        Default is ``True``.

    plots_for_all_images
        If True star map plots for all stars are created
        Default is ``False``.

    plot_for_reference_image_only
        If True a star map plots only for the reference image is created
        Default is ``True``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Get filter
    filter_ = image_series.filter_

    #   Find the stars (using DAO or IRAF StarFinder) on the reference image,
    #   if these positions are to be used for all images in the series.
    if not identify_objects_on_image:
        _, background_rms = determine_background(
            image_series.reference_image,
            sigma_background=sigma_value_background_clipping,
        )

        find_stars(
            image_series.reference_image,
            sigma_object_psf[filter_],
            background_rms,
            multiplier_background_rms=multiplier_background_rms,
            method=object_finder_method,
        )

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
            image.positions = image_series.reference_image.positions

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
                'size_extraction_region_epsf': size_extraction_region_epsf,
                'epsf_fitter': epsf_fitter,
                'n_iterations_eps_extraction': n_iterations_eps_extraction,
                'fraction_epsf_stars': fraction_epsf_stars,
                'oversampling_factor_epsf': oversampling_factor_epsf,
                'max_n_iterations_epsf_determination': max_n_iterations_epsf_determination,
                'use_initial_positions_epsf': use_initial_positions_epsf,
                'object_finder_method': object_finder_method,
                'multiplier_background_rms_epsf': multiplier_background_rms_epsf,
                'multiplier_grouper_epsf': multiplier_grouper_epsf,
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
                'file_type_plots': file_type_plots,
                'use_wcs_projection_for_star_maps': use_wcs_projection_for_star_maps,
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


def main_extract(
        image: Image, sigma_object_psf: float,
        multiprocessing: bool = False,
        sigma_value_background_clipping: float = 5.,
        multiplier_background_rms: float = 5., size_epsf_region: int = 25,
        size_extraction_region_epsf: int = 11, epsf_fitter: str = 'LMLSQFitter',
        n_iterations_eps_extraction: int = 1,
        fraction_epsf_stars: float = 0.2, oversampling_factor_epsf: int = 1,
        max_n_iterations_epsf_determination: int = 7,
        use_initial_positions_epsf: bool = True,
        object_finder_method: str = 'IRAF',
        multiplier_background_rms_epsf: float = 5.0,
        multiplier_grouper_epsf: float = 2.0,
        strict_cleaning_epsf_results: bool = True,
        minimum_n_eps_stars: int = 15,
        id_reference_image: int = 0, photometry_extraction_method: str = 'PSF',
        radius_aperture: float = 4., inner_annulus_radius: float = 7.,
        outer_annulus_radius: float = 10., radii_unit: str = 'arcsec',
        strict_epsf_checks: bool = True, identify_objects_on_image: bool = True,
        cosmic_ray_removal: bool = False,
        limiting_contrast_rm_cosmics: float = 5.,
        read_noise: float = 8., sigma_clipping_value: float = 4.5,
        saturation_level: float = 65535., plots_for_all_images: bool = False,
        plot_for_reference_image_only: bool = True,
        file_type_plots: str = 'pdf',
        use_wcs_projection_for_star_maps: bool = True) -> None | tuple[int, Table]:
    """
    Main function to extract the information from the individual images

    Parameters
    ----------
    image
        Object all image specific properties

    sigma_object_psf
        Sigma of the objects PSF, assuming it is a Gaussian

    multiprocessing
        If True, the routine is set up to meet the requirements of
        multiprocessing, such as returning results and delayed
        output to the terminal.

    sigma_value_background_clipping
        Sigma used for the sigma clipping of the background
        Default is ``5``.

    multiplier_background_rms
        Multiplier for the background RMS, used to calculate the
        threshold to identify stars
        Default is ``5``.

    size_epsf_region
        Size of the extraction region in pixel
        Default is ``25``.

    size_extraction_region_epsf
        Size of the extraction region in pixel
        Default is ``11``.

    epsf_fitter
        Fitter function used during ePSF fitting to the data.
        Options are: ``LevMarLSQFitter``, ``LMLSQFitter`` and ``TRFLSQFitter``
        Default is ``LMLSQFitter``.

    n_iterations_eps_extraction
        Number of extraction iterations in the ePSF fit to the data. In certain
        cases, such as very crowded fields, numbers greater than 1 can lead to
        very large CPU loads and recursions within astropy that may exceed the
        defined limits.
        Default is ``1``.

    fraction_epsf_stars
        Fraction of all stars that should be used to calculate the ePSF
        Default is ``0.2``.

    oversampling_factor_epsf
        ePSF oversampling factor
        Default is ``1``.

    max_n_iterations_epsf_determination
        Number of ePSF iterations
        Default is ``7``.

    use_initial_positions_epsf
        If True the initial positions from a previous object
        identification procedure will be used. If False the objects
        will be identified by means of the ``method_finder`` method.
        Default is ``True``.

    object_finder_method
        Finder method DAO or IRAF
        Default is ``IRAF``.

    multiplier_background_rms_epsf
        Multiplier for the background RMS, used to calculate the
        threshold to identify stars
        Default is ``5.0``.

    multiplier_grouper_epsf
        Multiplier for the DAO grouper
        Default is ``5.0``.

    strict_cleaning_epsf_results
        If True objects with negative flux uncertainties will be removed
        Default is ``True``.

    minimum_n_eps_stars
        Minimal number of required ePSF stars
        Default is ``15``.

    id_reference_image
        ID of the reference image
        Default is ``0``.

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
        Unit of the radii above. Permitted values are ``pixel`` and ``arcsec``.
        Default is ``pixel``.

    strict_epsf_checks
        If True a stringent test of the ePSF conditions is applied.
        Default is ``True``.

    identify_objects_on_image
        If `True` the objects on the image will be identified. If `False`
        it is assumed that object identification was performed in advance.
        Default is ``True``.

    cosmic_ray_removal
        If True cosmic rays will be removed from the image.
        Default is ``False``.

    limiting_contrast_rm_cosmics
        Parameter for the cosmic ray removal: Minimum contrast between
        Laplacian image and the fine structure image.
        Default is ``5``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    sigma_clipping_value
        Parameter for the cosmic ray removal: Fractional detection limit
        for neighboring pixels.
        Default is ``4.5``.

    saturation_level
        Saturation limit of the camera chip.
        Default is ``65535``.

    plots_for_all_images
        If True star map plots for all stars are created
        Default is ``False``.

    plot_for_reference_image_only
        If True a star map plot only for the reference image is created
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.
    """
    #   Initialize output class in case of multiprocessing
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

    #   Remove cosmics (optional)
    if cosmic_ray_removal:
        rm_cosmic_rays(
            image,
            limiting_contrast=limiting_contrast_rm_cosmics,
            read_noise=read_noise,
            sigma_clipping_value=sigma_clipping_value,
            saturation_level=saturation_level,
        )

    #   Estimate and remove background
    _, rms_background = determine_background(
        image,
        sigma_background=sigma_value_background_clipping,
    )

    #   Find the stars (via DAO or IRAF StarFinder)
    if identify_objects_on_image:
        find_stars(
            image,
            sigma_object_psf,
            rms_background,
            multiplier_background_rms=multiplier_background_rms,
            method=object_finder_method,
            terminal_logger=terminal_logger,
        )

    if photometry_extraction_method == 'PSF':
        #   Check size of ePSF extraction region
        if size_epsf_region % 2 == 0:
            size_epsf_region = size_epsf_region + 1

        #   Check if enough stars have been detected to allow ePSF
        #   calculations
        epsf_stars = check_epsf_stars(
            image,
            size_epsf_region=size_epsf_region,
            minimum_n_stars=minimum_n_eps_stars,
            fraction_epsf_stars=fraction_epsf_stars,
            terminal_logger=terminal_logger,
            strict_epsf_checks=strict_epsf_checks,
        )

        #   Plot images with the identified stars overlaid
        if plots_for_all_images or (plot_for_reference_image_only
                                    and image.pd == id_reference_image):
            plots.starmap(
                image.out_path.name,
                image.get_data(),
                image.filter_,
                image.positions,
                tbl_2=epsf_stars,
                label='identified stars',
                label_2='stars used to determine the ePSF',
                rts=f'Initial object identification [Image: {image.pd}]',
                wcs_image=image.wcs,
                use_wcs_projection=use_wcs_projection_for_star_maps,
                terminal_logger=terminal_logger,
                file_type=file_type_plots,
            )

        #   Calculate the ePSF
        determine_epsf(
            image,
            epsf_stars,
            size_epsf_region=size_epsf_region,
            oversampling_factor=oversampling_factor_epsf,
            max_n_iterations=max_n_iterations_epsf_determination,
            minimum_n_stars=minimum_n_eps_stars,
            multiprocess_plots=False,
            terminal_logger=terminal_logger,
            file_type_plots=file_type_plots,
        )

        #   Plot the ePSFs
        plots.plot_epsf(
            image.out_path.name,
            {f'img-{image.pd}-{image.filter_}': image.epsf},
            terminal_logger=terminal_logger,
            file_type=file_type_plots,
            id_image=f'_{image.pd}',
            indent=2,
        )

        #   Performing the PSF photometry
        extraction_epsf(
            image,
            sigma_object_psf,
            rms_background,
            sigma_background=sigma_value_background_clipping,
            use_initial_positions=use_initial_positions_epsf,
            finder_method=object_finder_method,
            size_extraction_region=size_extraction_region_epsf,
            epsf_fitter=epsf_fitter,
            n_iterations_eps_extraction=n_iterations_eps_extraction,
            multiplier_background_rms=multiplier_background_rms_epsf,
            multiplier_grouper=multiplier_grouper_epsf,
            strict_cleaning_results=strict_cleaning_epsf_results,
            terminal_logger=terminal_logger,
        )

        #   Plot original and residual image
        plots.plot_residual(
            {f'{image.filter_}, Image ID: {image.pd}': image.get_data()},
            {f'{image.filter_}, Image ID: {image.pd}': image.residual_image},
            image.out_path.name,
            terminal_logger=terminal_logger,
            file_type=file_type_plots,
            # name_object=image.object_name,
            indent=2,
        )

    elif photometry_extraction_method == 'APER':
        #   Perform aperture photometry
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
            file_type_plots=file_type_plots,
            indent=3,
        )

    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nExtraction method "
            f"({photometry_extraction_method}) not "
            f"valid: use either APER or PSF {style.Bcolors.ENDC}"
        )

    #   Conversion of flux to magnitudes
    #   TODO: Move this to the calibration stage, where it makes more sense?
    magnitudes, magnitudes_error = utilities.flux_to_magnitudes(
        image.photometry['flux_fit'],
        image.photometry['flux_err'],
    )

    image.photometry['mags_fit'] = magnitudes
    image.photometry['mags_unc'] = magnitudes_error

    #   Plot images with extracted stars overlaid
    if plots_for_all_images or (plot_for_reference_image_only
                                and image.pd == id_reference_image):
        utilities.prepare_and_plot_starmap(
            image,
            terminal_logger=terminal_logger,
            file_type_plots=file_type_plots,
            label=f'Stars with photometric extractions ({photometry_extraction_method})',
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )

    if multiprocessing:
        terminal_logger.print_to_terminal('')
    else:
        terminal_output.print_to_terminal('')

    if multiprocessing:
        # return copy.deepcopy(image.pd), copy.deepcopy(image.photometry)
        return image.pd, image.photometry


def subtract_archive_img_from_img(
        filter_: str, image_path: str, output_dir: str,
        wcs_method: str = 'astrometry', plot_comp: bool = True,
        hips_source: str = 'CDS/P/DSS2/blue',
        file_type_plots: str = 'pdf') -> None:
    """
    Subtraction of a reference/archival image from the input image.
    The installation of Hotpants is required.

    Parameters
    ----------
    filter_
        Filter identifier

    image_path
        Path to images

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

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'subtract'),
    )
    output_dir = os.path.join(output_dir, 'subtract')

    #   Check input path
    checks.check_file(image_path)

    #   Trim image as needed (currently images with < 4*10^6 are required)
    #
    #   Load image
    ccd_image = CCDData.read(image_path)

    #   Trim
    pixel_max_x = 2501
    # pixel_max_x = 2502
    pixel_max_y = 1599
    ccd_image = ccdp.trim_image(ccd_image[0:pixel_max_y, 0:pixel_max_x])
    ccd_image.meta['NAXIS1'] = pixel_max_x
    ccd_image.meta['NAXIS2'] = pixel_max_y

    #   Save trimmed file
    basename = base_utilities.get_basename(image_path)
    file_name = f'{basename}_trimmed.fit'
    file_path = os.path.join(output_dir, file_name)
    ccd_image.write(file_path, overwrite=True)

    #   Initialize image series object
    image_series = ImageSeries(
        filter_,
        image_path,
        output_dir,
    )

    #   Find the WCS solution for the image
    utilities.find_wcs(
        image_series,
        reference_image_id=0,
        method=wcs_method,
        indent=3,
    )

    #   Get image via hips2fits
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

    #   Plot original and reference image
    if plot_comp:
        plots.compare_images(
            output_dir,
            image_series.image_list[0].get_data(),
            hips_hdus[0].data,
            file_type=file_type_plots,
        )

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
