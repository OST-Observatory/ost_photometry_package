############################################################################
#                               Libraries                                  #
############################################################################

import os
from pathlib import Path

import numpy as np
import yaml
from skimage.transform import SimilarityTransform

from .. import checks, terminal_output
from .. import utilities as base_utilities
from ..output_layout import tables_dir
from .image import AnalysisImage
from .models import ImageSeries
from .observation import Observation
from .pipeline import AnalysisContext, AnalysisPipeline, PipelineConfig

# Suppress noisy dependency UserWarnings. Package diagnostics use
# OstPhotometryAnalyzeWarning (see warnings_types) so they are not hidden here.
# warnings.filterwarnings("ignore", category=UserWarning, append=True)


############################################################################
#                           Routines & definitions                         #
############################################################################


def transform_object_positions(
    image_series: ImageSeries | list[AnalysisImage], output_dir: str | None = None
) -> None | list[AnalysisImage]:
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
            "Neither an ImageSeries object nor a list of Image objects was provided. "
            f"The type provided was {type(image_series)}."
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
        raise FileNotFoundError(e) from e

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
                "The image transformation matrix file does not exist for the "
                "current image. Without this information, transformation "
                "to the reference frame is not possible. -> Skip this image.",
                style_name="WARNING",
            )
            image_ids_to_rm.append(i)
            continue

        #   Prepare similarity transform object
        current_trans = SimilarityTransform(matrix)

        #   Transform coordinates
        transformed_coordinates = reference_trans(
            current_trans.inverse(
                list(zip(x_pixel_coordinates, y_pixel_coordinates, strict=True))
            )
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


def _run_pipeline_impl(
    self: Observation,
    filter_list: list[str],
    image_paths: dict[str, str] | None = None,
    output_dir: str | None = None,
    config: PipelineConfig | None = None,
    extraction_mode: str = "auto",
    **kwargs,
) -> None:
    """Run the full analysis pipeline."""
    cfg = config or PipelineConfig()
    cfg.extraction_mode = extraction_mode
    cfg.apply_overrides(**kwargs)

    if image_paths is not None and output_dir is not None:
        checks.check_output_directories(
            output_dir,
            tables_dir(output_dir),
        )
        if cfg.extraction_mode == "multi" or (
            cfg.extraction_mode == "auto"
            and all(os.path.isdir(image_paths.get(f, "")) for f in filter_list)
        ):
            checks.check_dir(image_paths)
        else:
            for f in filter_list:
                checks.check_file(image_paths[f])
        context = _to_context(self, filter_list, image_paths, output_dir, cfg)
    else:
        if output_dir is None and filter_list:
            output_dir = str(self.image_series_dict[filter_list[0]].out_path)
        context = _to_context_from_observation(self, filter_list, output_dir, cfg)

    context = AnalysisPipeline(config=cfg).run(context)
    _from_context(self, context)


def _to_context(
    observation: Observation,
    filter_list: list[str],
    image_paths: dict[str, str],
    output_dir: str,
    config: PipelineConfig,
) -> AnalysisContext:
    for filter_ in filter_list:
        observation.image_series_dict[filter_] = ImageSeries(
            filter_,
            image_paths[filter_],
            output_dir,
            reference_image_index=config.reference_image_index,
        )

    context = AnalysisContext(
        image_series_dict=observation.image_series_dict,
        filter_list=filter_list,
        output_dir=output_dir,
        objects_of_interest=observation.objects_of_interest,
        table_magnitudes=observation.table_magnitudes,
        observation=observation,
    )
    return context


def _to_context_from_observation(
    observation: Observation,
    filter_list: list[str],
    output_dir: str,
    config: PipelineConfig,
) -> AnalysisContext:
    image_series_dict = {f: observation.image_series_dict[f] for f in filter_list}
    return AnalysisContext(
        image_series_dict=image_series_dict,
        filter_list=filter_list,
        output_dir=output_dir,
        objects_of_interest=observation.objects_of_interest,
        table_magnitudes=observation.table_magnitudes,
        observation=observation,
    )


def _from_context(observation: Observation, context: AnalysisContext) -> None:
    observation.table_magnitudes = context.table_magnitudes
    if context.observation is not None:
        observation.image_series_dict = context.observation.image_series_dict


Observation.run_pipeline = _run_pipeline_impl  # type: ignore[method-assign, attr-defined]
