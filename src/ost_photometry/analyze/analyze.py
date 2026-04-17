############################################################################
#                               Libraries                                  #
############################################################################

import os
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import SkyCoord, name_resolve
from astropy.table import Table
from photutils.psf import (
    ImagePSF,
)
from skimage.transform import SimilarityTransform

from .. import checks, style, terminal_output
from .. import utilities as base_utilities
from ..utilities import Image
from . import calibration_data
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
            reference_image_index = image_series.reference_image_index

            img = image_series.image_list[reference_image_index]

            epsf_dict[key] = [img.epsf]

        return epsf_dict

    #   Get reference image
    def get_reference_image(self) -> dict[str, np.ndarray]:
        img_dict: dict[str, np.ndarray] = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_index = image_series.reference_image_index

            img = image_series.image_list[reference_image_index]

            img_dict[key] = img.get_data()

        return img_dict

    #   Get residual image belonging to the reference image
    def get_reference_image_residual(self) -> dict[str, np.ndarray]:
        img_dict: dict[str, np.ndarray] = {}
        for key, image_series in self.image_series_dict.items():
            reference_image_index = image_series.reference_image_index

            img = image_series.image_list[reference_image_index]

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
            If None, uses existing image_series_dict (e.g. after a prior pipeline run).
        output_dir
            Output directory. If None and image_paths is None, taken from
            existing image series.
        config
            Pipeline configuration. If None, uses defaults.
        extraction_mode
            "single" (1 image per filter), "multi" (N images), or "auto".
        **kwargs
            Override config attributes. Nested diagnostic flags can be set as
            ``diagnostic_plots__<field>=True`` (see ``DiagnosticPlots`` in
            ``pipeline.config``), e.g. ``diagnostic_plots__photometry_mag_vs_error_scatter=True``.
        """
        cfg = config or PipelineConfig()
        cfg.extraction_mode = extraction_mode
        diag_prefix = "diagnostic_plots__"
        for key, val in kwargs.items():
            if key.startswith(diag_prefix):
                sub = key[len(diag_prefix) :]
                if hasattr(cfg.diagnostic_plots, sub):
                    setattr(cfg.diagnostic_plots, sub, val)
            elif hasattr(cfg, key):
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
            # Use existing image_series_dict (no new ImageSeries from paths)
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
                reference_image_index=config.reference_image_index,
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
