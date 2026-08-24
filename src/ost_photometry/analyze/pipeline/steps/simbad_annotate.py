"""Optional Simbad overlay on each filter's reference image."""

from __future__ import annotations

from .... import terminal_output
from ...post_processing.simbad_annotate import annotate_reference_image_with_simbad
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


class SimbadAnnotateStep(base.PipelineStep):
    """
    Post-processing: mark Simbad objects on the reference image of each filter.

    Independent of calibration. Controlled by ``annotate_image`` (single-image
    extraction) and ``annotate_reference_image`` (multi-image extraction).
    """

    name = "simbad_annotate"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        ext = config.extraction
        mode = context.get_extraction_mode(config)
        if mode == "single":
            return not ext.annotate_image
        return not ext.annotate_reference_image

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        ext = config.extraction
        terminal_output.print_to_terminal(
            "Annotating reference images with Simbad objects",
            style_name="HEADER",
        )
        for filter_ in context.filter_list:
            series = context.image_series_dict.get(filter_)
            if series is None or not getattr(series, "image_list", None):
                terminal_output.print_to_terminal(
                    f"No image series for filter {filter_!r}; skipping Simbad overlay.",
                    indent=2,
                    style_name="WARNING",
                )
                continue
            idx = getattr(series, "reference_image_index", 0)
            images = series.image_list
            if idx < 0 or idx >= len(images):
                terminal_output.print_to_terminal(
                    f"Invalid reference_image_index={idx} for filter {filter_!r}; "
                    "skipping Simbad overlay.",
                    indent=2,
                    style_name="WARNING",
                )
                continue
            annotate_reference_image_with_simbad(
                images[idx],
                file_type=ext.file_type_plots,
                filter_mag=ext.filter_magnitude_limit_image_annotation,
                mag_limit=ext.magnitude_limit_image_annotation,
            )
        return context
