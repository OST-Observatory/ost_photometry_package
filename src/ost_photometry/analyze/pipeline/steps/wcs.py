"""WCS determination step."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig


class WcsStep(base.PipelineStep):
    """Determine WCS for each filter's image series."""

    name = "wcs"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        return config.skip_wcs or context.wcs_determined

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        from .... import terminal_output
        from ...utilities import find_wcs

        for filter_ in context.filter_list:
            terminal_output.print_to_terminal(
                f"Determining WCS for {filter_} images",
                style_name="HEADER",
            )

            image_series = context.image_series_dict[filter_]

            try:
                find_wcs(
                    image_series,
                    reference_image_index=config.reference_image_index,
                    method=config.wcs_method,
                    force_wcs_determination=config.force_wcs_determination,
                    indent=3,
                )
            except RuntimeError as e:
                # Fallback: use WCS from another filter if available
                for wcs_filter in context.filter_list:
                    ref_series = context.image_series_dict.get(wcs_filter)
                    if ref_series is None:
                        continue
                    reference_wcs = getattr(ref_series, "wcs", None)
                    if reference_wcs is not None:
                        image_series.set_wcs(reference_wcs)
                        terminal_output.print_to_terminal(
                            f"WCS could not be determined for filter {filter_}. "
                            f"Using WCS of filter {wcs_filter} instead. "
                            "This could lead to problems...",
                            indent=1,
                            style_name="WARNING",
                        )
                        break
                else:
                    raise RuntimeError(e) from e

        context.wcs_determined = True
        return context
