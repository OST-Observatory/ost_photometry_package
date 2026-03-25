############################################################################
#                               Libraries                                  #
############################################################################

import multiprocessing as mp
import os
import warnings
from collections import Counter
from pathlib import Path

import astropy.units as u
import ccdproc as ccdp
import numpy as np
import regions
import yaml
from astropy import uncertainty as unc
from astropy.coordinates import SkyCoord, name_resolve
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.time import Time

#   hips2fits module is not in the Ubuntu 22.04 package version
#   of astroquery (0.4.1)
# from astroquery.hips2fits import hips2fits
from astroquery.hips2fits import hips2fitsClass
from photutils.psf import (
    ImagePSF,
)
from skimage.transform import SimilarityTransform

# from . import subtraction
from .. import checks, style, terminal_output
from .. import utilities as base_utilities
from ..utilities import Image
from . import calibration, calibration_data, correlate, plots, utilities
from .extraction import extract_multiprocessing, main_extract
from .models import ImageSeries, ObjectOfInterest
from .pipeline import AnalysisContext, AnalysisPipeline, PipelineConfig

# Suppress noisy dependency UserWarnings. Package diagnostics use
# OstPhotometryAnalyzeWarning (see warnings_types) so they are not hidden here.
warnings.filterwarnings("ignore", category=UserWarning, append=True)


############################################################################
#                           Routines & definitions                         #
############################################################################


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
        ra_objects: list[str] | None = kwargs.get("ra_objects", None)
        ra_unit: str | None = kwargs.get("ra_unit", None)
        dec_objects: list[str] | None = kwargs.get("dec_objects", None)
        dec_unit: str | None = kwargs.get("dec_unit", None)
        object_names: list[str] | None = kwargs.get("object_names", None)
        periods: list[float] | None = kwargs.get("periods", None)
        transit_times: list[str] | None = kwargs.get("transit_times", None)

        add_periods = False
        if all([periods, transit_times]):
            add_periods = True

        #   Setup object of interests
        self.objects_of_interest = []

        #   Case 1: All base parameters are provided
        if all([ra_objects, dec_objects, ra_unit, dec_unit, object_names]):
            len_names = len(object_names)
            if len_names == len(ra_objects) and len_names == len(ra_objects):
                for i, (name, ra, dec) in enumerate(
                    zip(object_names, ra_objects, dec_objects)
                ):
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
    def get_epsf(self) -> dict[str, list[ImagePSF]]:
        epsf_dict: dict[str, list[ImagePSF]] = {}
        for key, image_series in self.image_series_dict.items():
            epsf_list: list[ImagePSF] = []
            for img in image_series.image_list:
                epsf_list.append(img.epsf)
            epsf_dict[key] = epsf_list

        return epsf_dict

    #   Get ePSF object of the reference image
    # def get_reference_epsf(self):
    #     epsf_dict = {}
    def get_reference_epsf(self) -> dict[str, list[ImagePSF]]:
        epsf_dict: dict[str, list[ImagePSF]] = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_id = image_series.reference_image_id

            img = image_series.image_list[reference_image_id]

            epsf_dict[key] = [img.epsf]

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

            if img.residual_image is not None:
                img_dict[key] = img.residual_image

        return img_dict

    #   Get image series for a specific set of filters
    def get_image_series(
        self, filter_list: list[str] | set[str]
    ) -> dict[str, ImageSeries]:
        image_series_dict: dict[str, ImageSeries] = {}
        for filter_ in filter_list:
            image_series_dict[filter_] = self.image_series_dict[filter_]

        return image_series_dict

    #   Get the IDs of the objects of interest within the detected objects on
    #   the images
    def get_ids_object_of_interest(
        self, filter_: str | None = None, reference_image_series_id: int | None = None
    ) -> list[int]:
        if filter_ is None and reference_image_series_id is None:
            terminal_output.print_to_terminal(
                "Neither a filter nor an image series ID was provided to "
                "compile the IDs for the objects of interest.The image series ID "
                "is assumed to be 0.",
                style_name="WARNING",
            )
            reference_image_series_id: int = 0

        object_of_interest_ids: list[int] = []
        for object_ in self.objects_of_interest:
            ids_object_of_interest = object_.id_in_image_series
            if ids_object_of_interest:
                if filter_ is not None:
                    object_of_interest_ids.append(ids_object_of_interest[filter_])
                else:
                    #   TODO: This is dirty... :( Can you fix it?
                    object_of_interest_ids.append(
                        ids_object_of_interest[
                            list(ids_object_of_interest.keys())[
                                reference_image_series_id
                            ]
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

    def run_pipeline(
        self,
        filter_list: list[str],
        image_paths: dict[str, str] | None = None,
        output_dir: str | None = None,
        config: PipelineConfig | None = None,
        extraction_mode: str = "auto",
        **kwargs,
    ) -> None:
        """
        Run the full analysis pipeline (WCS, extraction, correlation, calibration).

        Parameters
        ----------
        filter_list
            List of filter names.
        image_paths
            Paths to images or directories: key=filter name, value=path.
            If None, uses existing image_series_dict (e.g. after extract_flux).
        output_dir
            Output directory. If None and image_paths is None, taken from
            existing image series.
        config
            Pipeline configuration. If None, uses defaults.
        extraction_mode
            "single" (1 image per filter), "multi" (N images), or "auto".
        **kwargs
            Override config attributes.
        """
        cfg = config or PipelineConfig()
        cfg.extraction_mode = extraction_mode
        for key, val in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

        if image_paths is not None and output_dir is not None:
            checks.check_output_directories(
                output_dir,
                os.path.join(output_dir, "tables"),
            )
            if cfg.extraction_mode == "multi" or (
                cfg.extraction_mode == "auto"
                and all(os.path.isdir(image_paths.get(f, "")) for f in filter_list)
            ):
                checks.check_dir(image_paths)
            else:
                for f in filter_list:
                    checks.check_file(image_paths[f])
            context = self._to_context(filter_list, image_paths, output_dir, cfg)
        else:
            # Use existing image_series_dict (e.g. correlate_calibrate)
            if output_dir is None and filter_list:
                output_dir = str(self.image_series_dict[filter_list[0]].out_path)
            context = self._to_context_from_observation(filter_list, output_dir, cfg)

        context = AnalysisPipeline(config=cfg).run(context)
        self._from_context(context)

    def _to_context(
        self,
        filter_list: list[str],
        image_paths: dict[str, str],
        output_dir: str,
        config: PipelineConfig,
    ) -> AnalysisContext:
        """Build AnalysisContext from paths."""
        for filter_ in filter_list:
            self.image_series_dict[filter_] = ImageSeries(
                filter_,
                image_paths[filter_],
                output_dir,
                reference_image_id=config.reference_image_id,
            )

        context = AnalysisContext(
            image_series_dict=self.image_series_dict,
            filter_list=filter_list,
            output_dir=output_dir,
            objects_of_interest=self.objects_of_interest,
            calib_parameters=self.calib_parameters,
            table_magnitudes=self.table_magnitudes,
        )
        context._observation = self
        return context

    def _to_context_from_observation(
        self,
        filter_list: list[str],
        output_dir: str,
        config: PipelineConfig,
    ) -> AnalysisContext:
        """Build AnalysisContext from existing Observation (no image_paths)."""
        image_series_dict = {f: self.image_series_dict[f] for f in filter_list}
        context = AnalysisContext(
            image_series_dict=image_series_dict,
            filter_list=filter_list,
            output_dir=output_dir,
            objects_of_interest=self.objects_of_interest,
            calib_parameters=self.calib_parameters,
            table_magnitudes=self.table_magnitudes,
        )
        context._observation = self
        return context

    def _from_context(self, context: AnalysisContext) -> None:
        """Copy pipeline results back to Observation."""
        self.calib_parameters = context.calib_parameters
        self.table_magnitudes = context.table_magnitudes

    def extract_flux(
        self,
        filter_list: list[str],
        image_paths: dict[str, str],
        output_dir: str,
        fwhm_object_psf: dict[str, float] | None = None,
        wcs_method: str = "astrometry",
        force_wcs_determination: bool = False,
        sigma_value_background_clipping: float = 5.0,
        multiplier_background_rms: float = 5.0,
        size_epsf_region: int = 25,
        size_extraction_region_epsf: int = 11,
        epsf_fitter: str = "TRFLSQFitter",
        n_iterations_eps_extraction: int = 1,
        fraction_epsf_stars: float = 0.2,
        oversampling_factor_epsf: int = 4,
        max_n_iterations_epsf_determination: int = 7,
        use_initial_positions_epsf: bool = True,
        object_finder_method: str = "IRAF",
        multiplier_background_rms_epsf: float = 5.0,
        multiplier_grouper_epsf: float = 2.0,
        strict_cleaning_epsf_results: bool = True,
        minimum_n_eps_stars: int = 15,
        reference_image_id: int = 0,
        strict_epsf_checks: bool = True,
        photometry_extraction_method: str = "PSF",
        radius_aperture: float = 5.0,
        inner_annulus_radius: float = 7.0,
        outer_annulus_radius: float = 10.0,
        radii_unit: str = "arcsec",
        cosmic_ray_removal: bool = False,
        limiting_contrast_rm_cosmics: float = 5.0,
        read_noise: float = 8.0,
        sigma_clipping_value: float = 4.5,
        saturation_level: float = 65535.0,
        plots_for_all_images: bool = False,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = "pdf",
        annotate_image: bool = False,
        magnitude_limit_image_annotation: float | None = None,
        filter_magnitude_limit_image_annotation: str | None = None,
        transform_object_positions_to_reference: bool = False,
    ) -> None:
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

        fwhm_object_psf
            FWHM of the objects PSF, assuming it is a Gaussian
            Default is ``None``.

        wcs_method
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        force_wcs_determination
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
            Default is ``4``.

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

        use_wcs_projection_for_star_maps
            If ``True`` the starmap will be plotted with sky coordinates instead
            of pixel coordinates
            Default is ``True``.

        file_type_plots
            Type of plot file to be created
            Default is ``pdf``.

        annotate_image
            If ``True``, a starmap will be created with known Simbad objects marked.
            Default is ``False``.

        magnitude_limit_image_annotation
            Limiting magnitude, only objects brighter as this limit will be shown
            Default is ``None``.

        filter_magnitude_limit_image_annotation
            Name of the filter (e.g. 'V')
            Default is ``None``.

        transform_object_positions_to_reference
                    If ``True``, the object pixel coordinates extracted from the images
                    are be transformed to the reference frame defined by the reference
                    image. It assumes that similarity transformations are available
                    from an image correlation run.
                    Default is ``False``.
        """
        self.run_pipeline(
            filter_list,
            image_paths,
            output_dir,
            extraction_mode="single",
            fwhm_object_psf=fwhm_object_psf,
            wcs_method=wcs_method,
            force_wcs_determination=force_wcs_determination,
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
            reference_image_id=reference_image_id,
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
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            file_type_plots=file_type_plots,
            annotate_image=annotate_image,
            magnitude_limit_image_annotation=magnitude_limit_image_annotation,
            filter_magnitude_limit_image_annotation=filter_magnitude_limit_image_annotation,
            skip_correlation_intra=True,
            skip_correlation_inter=True,
            skip_calibration=True,
            skip_post_process=True,
        )

    def extract_flux_multi(
        self,
        filter_list: list[str],
        image_paths: dict[str, str],
        output_dir: str,
        fwhm_object_psf: dict[str, float] | None = None,
        n_cores_multiprocessing: int = 6,
        wcs_method: str = "astrometry",
        force_wcs_determination: bool = False,
        sigma_value_background_clipping: float = 5.0,
        multiplier_background_rms: float = 5.0,
        size_epsf_region: int = 25,
        size_extraction_region_epsf: int = 11,
        epsf_fitter: str = "TRFLSQFitter",
        n_iterations_eps_extraction: int = 1,
        fraction_epsf_stars: float = 0.2,
        oversampling_factor_epsf: int = 4,
        max_n_iterations_epsf_determination: int = 7,
        use_initial_positions_epsf: bool = True,
        object_finder_method: str = "IRAF",
        multiplier_background_rms_epsf: float = 5.0,
        multiplier_grouper_epsf: float = 2.0,
        strict_cleaning_epsf_results: bool = True,
        minimum_n_eps_stars: int = 15,
        strict_epsf_checks: bool = True,
        photometry_extraction_method: str = "PSF",
        radius_aperture: float = 5.0,
        inner_annulus_radius: float = 7.0,
        outer_annulus_radius: float = 10.0,
        radii_unit: str = "arcsec",
        max_pixel_between_objects: int = 3,
        own_correlation_option: int = 1,
        cross_identification_limit: int = 1,
        reference_image_id: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        correlation_method: str = "astropy",
        separation_limit: u.quantity.Quantity = 2.0 * u.arcsec,
        verbose: bool = False,
        duplicate_handling_object_identification: dict[str, str] | None = None,
        plots_for_all_images: bool = False,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = "pdf",
        annotate_reference_image: bool = False,
        magnitude_limit_image_annotation: float | None = None,
        filter_magnitude_limit_image_annotation: str | None = None,
        transform_object_positions_to_reference: bool = False,
    ) -> None:
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

        fwhm_object_psf
            FWHM of the objects PSF, assuming it is a Gaussian
            Default is ``None``.

        n_cores_multiprocessing
            Number of cores to use for multicore processing
            Default is ``6``.

        wcs_method
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        force_wcs_determination
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
            Default is ``4``.

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

        use_wcs_projection_for_star_maps
            If ``True`` the starmap will be plotted with sky coordinates
            instead.
            of pixel coordinates
            Default is ``True``.

        file_type_plots
            Type of plot file to be created
            Default is ``pdf``.

        annotate_reference_image
            If ``True``, a starmap will be created with known Simbad objects
            marked.
            Default is ``False``.

        magnitude_limit_image_annotation
            Limiting magnitude, only objects brighter as this limit will be
            shown.
            Default is ``None``.

        filter_magnitude_limit_image_annotation
            Name of the filter (e.g. 'V')
            Default is ``None``.

        transform_object_positions_to_reference
            If ``True``, the object pixel coordinates extracted from the images
            are be transformed to the reference frame defined by the reference
            image. It assumes that similarity transformations are available
            from an image correlation run.
            Default is ``False``.
        """
        self.run_pipeline(
            filter_list,
            image_paths,
            output_dir,
            extraction_mode="multi",
            fwhm_object_psf=fwhm_object_psf,
            n_cores_multiprocessing=n_cores_multiprocessing,
            wcs_method=wcs_method,
            force_wcs_determination=force_wcs_determination,
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
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            cross_identification_limit=cross_identification_limit,
            reference_image_id=reference_image_id,
            n_allowed_non_detections_object=n_allowed_non_detections_object,
            expected_bad_image_fraction=expected_bad_image_fraction,
            protect_reference_obj=protect_reference_obj,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            verbose=verbose,
            duplicate_handling_object_identification=duplicate_handling_object_identification,
            plots_for_all_images=plots_for_all_images,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            file_type_plots=file_type_plots,
            annotate_reference_image=annotate_reference_image,
            magnitude_limit_image_annotation=magnitude_limit_image_annotation,
            filter_magnitude_limit_image_annotation=filter_magnitude_limit_image_annotation,
            transform_object_positions_to_reference=transform_object_positions_to_reference,
            skip_correlation_inter=True,
            skip_calibration=True,
            skip_post_process=True,
        )

    #   TODO: Rename to reflect that it is only used for stacked data
    def correlate_calibrate(
        self,
        filter_list: list[str],
        max_pixel_between_objects: int = 3,
        own_correlation_option: int = 1,
        reference_image_id: int = 0,
        calibration_method: str = "APASS",
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None,
        object_id: int = None,
        magnitude_range: tuple[float, float] = (0.0, 18.5),
        apply_transformation: bool = True,
        transformation_coefficients_dict: dict[str, (float | str)] | None = None,
        derive_transformation_coefficients: bool = False,
        photometry_extraction_method: str = "",
        extract_only_circular_region: bool = False,
        region_radius: float = 600.0,
        identify_cluster_gaia_data: bool = False,
        clean_objs_using_pm: bool = False,
        max_distance_cluster: float = 6.0,
        find_cluster_para_set: int = 1,
        correlation_method: str = "astropy",
        separation_limit: u.quantity.Quantity = 2.0 * u.arcsec,
        aperture_radius: float = 4.0,
        radii_unit: str = "arcsec",
        convert_magnitudes: bool = False,
        target_filter_system: str = "SDSS",
        region_to_select_calibration_stars: regions.RectanglePixelRegion | None = None,
        calculate_zero_point_statistic: bool = True,
        distribution_samples: int = 1000,
        duplicate_handling_object_identification: dict[str, str] | None = None,
        file_type_plots: str = "pdf",
        use_wcs_projection_for_star_maps: bool = True,
    ) -> None:
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

        apply_transformation
            If ``True``, magnitude transformation is applied if possible.
            Default is ``True``.

        transformation_coefficients_dict
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_transformation_coefficients
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the database.
            Default is ``False``

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
        self.run_pipeline(
            filter_list,
            image_paths=None,
            output_dir=None,
            skip_wcs=True,
            skip_extraction=True,
            skip_correlation_intra=True,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            reference_image_id=reference_image_id,
            calibration_method=calibration_method,
            vizier_dict=vizier_dict,
            path_calibration_file=path_calibration_file,
            object_id=object_id,
            magnitude_range=magnitude_range,
            apply_transformation=apply_transformation,
            transformation_coefficients_dict=transformation_coefficients_dict,
            derive_transformation_coefficients=derive_transformation_coefficients,
            photometry_extraction_method=photometry_extraction_method,
            extract_only_circular_region=extract_only_circular_region,
            region_radius=region_radius,
            identify_cluster_gaia_data=identify_cluster_gaia_data,
            clean_objs_using_pm=clean_objs_using_pm,
            max_distance_cluster=max_distance_cluster,
            find_cluster_para_set=find_cluster_para_set,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            aperture_radius=aperture_radius,
            radii_unit=radii_unit,
            convert_magnitudes=convert_magnitudes,
            target_filter_system=target_filter_system,
            region_to_select_calibration_stars=region_to_select_calibration_stars,
            calculate_zero_point_statistic=calculate_zero_point_statistic,
            distribution_samples=distribution_samples,
            duplicate_handling_object_identification=duplicate_handling_object_identification,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )

    def calibrate_data_mk_light_curve(
        self,
        filter_list: list[str],
        output_dir: str,
        valid_filter_combinations: list[list[str]] | None = None,
        binning_factor: float | None = None,
        apply_transformation: bool = True,
        transformation_coefficients_dict: dict[str, (float | str)] | None = None,
        derive_transformation_coefficients: bool = False,
        calibration_method: str = "APASS",
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None,
        magnitude_range: tuple[float, float] = (0.0, 18.5),
        max_pixel_between_objects: int = 3,
        own_correlation_option: int = 1,
        cross_identification_limit: int = 1,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_objects: bool = True,
        protect_calibration_objects: bool = True,
        photometry_extraction_method: str = "",
        correlation_method: str = "astropy",
        separation_limit: u.quantity.Quantity = 2.0 * u.arcsec,
        verbose: bool = False,
        region_to_select_calibration_stars: regions.RectanglePixelRegion | None = None,
        calculate_zero_point_statistic: bool = True,
        n_cores_multiprocessing_calibration: int | None = None,
        distribution_samples: int = 1000,
        plot_light_curve_all: bool = False,
        plot_light_curve_calibration_objects: bool = True,
        file_type_plots: str = "pdf",
        duplicate_handling_object_identification: dict[str, str] = None,
        use_wcs_projection_for_star_maps: bool = True,
    ) -> None:
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

        apply_transformation
            If ``True``, magnitude transformation is applied if possible.
            Default is ``True``.

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
        checks.check_output_directories(f"{output_dir}/lightcurve")
        if plot_light_curve_all:
            checks.clear_directory(Path(f"{output_dir}/lightcurve/by_id"))
        if plot_light_curve_calibration_objects:
            checks.clear_directory(Path(f"{output_dir}/lightcurve/calibration"))

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
            coordinates_obj_to_rm=coordinates_objects_of_interest,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            correlate_with_observed_objects=False,
        )
        calibration_filters = self.calib_parameters.column_names
        terminal_output.print_to_terminal("")

        #   Determine usable filter combinations -> The filters must be in a valid
        #   filter combination for magnitude transformation and calibration
        #   data must be available for the filters.
        valid_filter, usable_filter_combinations = (
            utilities.find_filter_for_magnitude_transformation(
                filter_list,
                calibration_filters,
                valid_filter_combinations=valid_filter_combinations,
            )
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

        #   Perform magnitude transformation
        #   TODO: Convert this to matrix calculation over all filter simultaneously
        processed_filter = []
        if apply_transformation:
            for filter_set in usable_filter_combinations:
                #   Apply calibration and perform magnitude transformation
                calibration.apply_calibration(
                    self,
                    filter_set,
                    apply_transformation=apply_transformation,
                    transformation_coefficients_dict=transformation_coefficients_dict,
                    derive_transformation_coefficients=derive_transformation_coefficients,
                    photometry_extraction_method=photometry_extraction_method,
                    calculate_zero_point_statistic=calculate_zero_point_statistic,
                    n_cores_multiprocessing=n_cores_multiprocessing_calibration,
                    distribution_samples=distribution_samples,
                    file_type_plots=file_type_plots,
                )

                for filter_ in filter_set:
                    terminal_output.print_to_terminal(
                        f"Create light curves in filter: {filter_}",
                        style_name="OKBLUE",
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
                        format="jd",
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
                            file_name_suffix=f"_{filter_set[0]}-{filter_set[1]}",
                            file_type_plots=file_type_plots,
                        )

                    if plot_light_curve_all:
                        for index in np.arange(len(self.table_magnitudes)):
                            if (
                                index not in ids_object_of_interest
                                and index not in ids_calibration_objects
                            ):
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
                                        "file_name_suffix": f"_{filter_set[0]}-{filter_set[1]}",
                                        "subdirectory": "/by_id",
                                        "file_type_plots": file_type_plots,
                                    },
                                )
                                p.start()

                    if (
                        plot_light_curve_calibration_objects
                        and ids_calibration_objects.any()
                    ):
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
                                    "file_name_suffix": f"_{filter_set[0]}-{filter_set[1]}",
                                    "subdirectory": "/calibration",
                                    "file_type_plots": file_type_plots,
                                },
                            )
                            p.start()

                    processed_filter.append(filter_)

        #   Process those filters for which magnitude transformation is not possible
        for filter_ in filter_list:
            #   Check if filter is not yet processed
            if filter_ not in processed_filter:
                terminal_output.print_to_terminal(
                    f"Working on filter: {filter_}",
                    style_name="OKBLUE",
                )

                #   Get IDs of the object of interests
                ids_object_of_interest = self.get_ids_object_of_interest(
                    filter_=filter_
                )

                #   Check if calibration data is available
                if f"mag{filter_}" not in calibration_filters:
                    terminal_output.print_to_terminal(
                        "Magnitude calibration not possible because no "
                        f"calibration data is available for filter {filter_}. "
                        "Use normalized flux for light curve.",
                        indent=2,
                        style_name="WARNING",
                    )

                    #   Get image_series
                    image_series = self.image_series_dict[filter_]

                    #   Quasi calibration of the flux data
                    quasi_calibrated_flux = (
                        calibration.quasi_flux_calibration_image_series(
                            image_series,
                            distribution_samples=distribution_samples,
                        )
                    )

                    #   Normalize data if no calibration magnitudes are available
                    quasi_calibrated_normalized_flux = (
                        calibration.flux_normalization_image_series(
                            image_series,
                            quasi_calibrated_flux=quasi_calibrated_flux,
                            distribution_samples=distribution_samples,
                        )
                    )

                    plot_quantity = quasi_calibrated_normalized_flux
                else:
                    #   Correlation of observation objects with calibration
                    #   objects
                    if self.calib_parameters.ids_calibration_objects is None:
                        correlate.select_calibration_objects(
                            self,
                            [filter_],
                            correlation_method=correlation_method,
                            separation_limit=separation_limit,
                            max_pixel_between_objects=max_pixel_between_objects,
                            own_correlation_option=own_correlation_option,
                            file_type_plots=file_type_plots,
                            indent=2,
                        )

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
                    format="jd",
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
                        calibration_type="simple",
                    )

                if plot_light_curve_all:
                    if isinstance(plot_quantity, unc.core.NdarrayDistribution):
                        shape_array = plot_quantity.shape
                        index_array = np.arange(shape_array[1])
                    else:
                        index_array = np.arange(len(plot_quantity))
                    for index in index_array:
                        if (
                            index not in ids_object_of_interest
                            and index not in ids_calibration_objects
                        ):
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
                                    "calibration_type": "simple",
                                    "subdirectory": "/by_id",
                                    "file_type_plots": file_type_plots,
                                },
                            )
                            p.start()

                if (
                    plot_light_curve_calibration_objects
                    and ids_calibration_objects is not None
                    and ids_calibration_objects.any()
                    and f"mag{filter_}" in calibration_filters
                ):
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
                                "calibration_type": "simple",
                                "subdirectory": "/calibration",
                                "file_type_plots": file_type_plots,
                            },
                        )
                        p.start()


def transform_object_positions(
    image_series: ImageSeries | list[Image], output_dir: str | None = None
) -> None | list[Image]:
    """
    Use the provided similarity transformations to transform the object
    positions in each image to the reference frame.

    Parameters
    ----------
    image_series
        List or image series object with the images that should be transformed

    output_dir
        Path to the shared output directory
        Default is ``None``.
    """
    #   Get list with images and output directory if possible
    if isinstance(image_series, list):
        image_list = image_series
        if output_dir is None:
            terminal_output.print_to_terminal(
                "No output directory specified. Use: 'output/' ",
                indent=2,
                style_name="WARNING",
            )
            output_path = Path("./output")
        else:
            checks.check_path(output_dir)
            output_path = Path(output_dir)

    elif isinstance(image_series, ImageSeries):
        image_list = image_series.image_list
        if output_dir is None:
            output_path = image_series.out_path
        else:
            terminal_output.print_to_terminal(
                "Additional output path passed to "
                f"'transform_object_positions': {output_dir}. Use this "
                "instead of the one specified in the image series passed.",
                indent=2,
                style_name="WARNING",
            )
            checks.check_path(output_dir)
            output_path = Path(output_dir)

    else:
        raise ValueError(
            f"{style.Bcolors.FAIL} ERROR: Neither an ImageSeries object nor a "
            f"list of Image objects was provided. The type provided was "
            f"{type(image_series)}. -> EXIT {style.Bcolors.ENDC}"
        )

    #   Get reference image and image name
    if isinstance(image_series, ImageSeries):
        reference_image = image_series.reference_image
    else:
        reference_image = image_list[0]
    reference_file_name = reference_image.filename
    reference_base_name = base_utilities.get_basename(reference_file_name)

    #   Set default path
    path_transformation = output_path / "image_transformations/"

    #   Load reference transformation matrix
    reference_transformation_file = f"{path_transformation}/{reference_base_name}.yaml"
    try:
        with open(reference_transformation_file) as f:
            loaded = yaml.safe_load(f)
            reference_matrix = np.array(loaded)
    except FileNotFoundError as e:
        terminal_output.print_to_terminal(
            f"The image transformation matrix file does not exist for the "
            f"reference image. Without this information, transformation "
            f"to the reference frame is not possible. -> Exit {e}.",
            style_name="ERROR",
        )
        raise FileNotFoundError(e)

    #   Prepare reference similarity transform object
    reference_trans = SimilarityTransform(reference_matrix)

    #   Transform object positions for all images
    image_ids_to_rm = []
    for i, image in enumerate(image_list):
        #   Get coordinates
        x_pixel_coordinates = image.photometry["x_fit"].value
        y_pixel_coordinates = image.photometry["y_fit"].value

        #   Load transformation matrix
        file_name = image.filename
        base_name = base_utilities.get_basename(file_name)
        path_transformation_file = f"{path_transformation}/{base_name}.yaml"
        try:
            with open(path_transformation_file) as f:
                loaded = yaml.safe_load(f)
                matrix = np.array(loaded)
        except FileNotFoundError:
            terminal_output.print_to_terminal(
                f"The image transformation matrix file does not exist for the "
                f"current image. Without this information, transformation "
                f"to the reference frame is not possible. -> Skip this image.",
                style_name="WARNING",
            )
            image_ids_to_rm.append(i)
            continue

        #   Prepare similarity transform object
        current_trans = SimilarityTransform(matrix)

        #   Transform coordinates
        transformed_coordinates = reference_trans(
            current_trans.inverse(list(zip(x_pixel_coordinates, y_pixel_coordinates)))
        )

        #   Write object positions back to image object
        image.photometry["x_fit"] = transformed_coordinates[:, 0]
        image.photometry["y_fit"] = transformed_coordinates[:, 1]

    #   Remove images without transformation from the image list and return
    for i in reversed(image_ids_to_rm):
        image_list.pop(i)
    if isinstance(image_series, ImageSeries):
        image_series.image_list = image_list
    else:
        return image_list


def subtract_archive_img_from_img(
    filter_: str,
    image_path: str,
    output_dir: str,
    wcs_method: str = "astrometry",
    plot_comp: bool = True,
    hips_source: str = "CDS/P/DSS2/blue",
    file_type_plots: str = "pdf",
) -> None:
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
        os.path.join(output_dir, "subtract"),
    )
    output_dir = os.path.join(output_dir, "subtract")

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
    ccd_image.meta["NAXIS1"] = pixel_max_x
    ccd_image.meta["NAXIS2"] = pixel_max_y

    #   Save trimmed file
    basename = base_utilities.get_basename(image_path)
    file_name = f"{basename}_trimmed.fit"
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
    hips_instance.server = (
        "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
    )
    print(hips_instance.timeout)
    print(hips_instance.server)
    # hips_hdus = hips2fits.query_with_wcs(
    hips_hdus = hips_instance.query_with_wcs(
        hips=hips_source,
        wcs=image_series.wcs,
        get_query_payload=False,
        format="fits",
        verbose=True,
    )
    #   Save downloaded file
    hips_hdus.writeto(os.path.join(output_dir, "hips.fits"), overwrite=True)

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
    hips_data = hips_hdus[0].data.astype("float64").byteswap().newbyteorder()

    #   Run Hotpants
    subtraction.run_hotpants(
        ccd_image.data,
        hips_data,
        ccd_image.mask,
        np.zeros(hips_data.shape, dtype=bool),
        image_gain=1.0,
        # template_gain=1,
        template_gain=None,
        err=ccd_image.uncertainty.array,
        # err=True,
        template_err=True,
        # verbose=True,
        _workdir=output_dir,
        # _exe=exe_path,
    )
