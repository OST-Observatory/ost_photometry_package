"""Analysis context - shared data container for the pipeline."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from astropy.table import Table

if TYPE_CHECKING:
    from ..calibration_data import CalibParameters
    from ..models import ImageSeries, ObjectOfInterest


@dataclass
class AnalysisContext:
    """
    Data container shared by all pipeline steps.

    Holds image series, filter list, output directory, and intermediate
    results. Used for both legacy (Observation/ImageSeries) and
    differential (table-based) pipelines.
    """

    # Required (Legacy)
    image_series_dict: dict
    filter_list: list[str]
    output_dir: str

    # Optional (Legacy)
    objects_of_interest: list = field(default_factory=list)
    calib_parameters: object = None
    table_magnitudes: Table | None = None

    # Optional (new pipeline / bridging)
    # epoch_id (e.g. epoch_000) -> multi-band photometry Table for differential calibration
    calibration_epochs: dict = field(default_factory=dict)
    calibration_epoch_meta: dict = field(default_factory=dict)  # epoch_id -> metadata dict
    calibration_epochs_skipped: list = field(
        default_factory=list
    )  # skipped pairing attempts (for terminal logging)
    extinction_coefficients: dict | None = None  # from ExtinctionFitStep

    # Reference to Observation when pipeline is run from Observation.run_pipeline
    # (needed for correlate steps which use get_ids_object_of_interest etc.)
    _observation: object = None

    # Metadata for skip logic
    wcs_determined: bool = False
    extraction_done: bool = False
    correlation_intra_done: bool = False
    correlation_inter_done: bool = False

    def has_multiple_images_per_filter(self) -> bool:
        """Return True if any filter has more than one image."""
        for image_series in self.image_series_dict.values():
            if len(image_series.image_list) > 1:
                return True
        return False

    def get_extraction_mode(self, config: "PipelineConfig") -> str:
        """
        Determine extraction mode from config or context.

        Returns "single" or "multi".
        """
        if config.extraction_mode == "single":
            return "single"
        if config.extraction_mode == "multi":
            return "multi"
        # auto: infer from data
        return "multi" if self.has_multiple_images_per_filter() else "single"
