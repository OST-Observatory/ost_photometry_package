"""Pipeline orchestrator."""

from .base import PipelineStep
from .config import PipelineConfig
from .context import AnalysisContext
from .steps import (
    CalibrationStep,
    CorrelationInterStep,
    CorrelationIntraStep,
    DeriveLimitingMagnitudeStep,
    ExtinctionFitStep,
    ExtractionStep,
    HipsReferenceSubtractStep,
    LightCurveStep,
    PostProcessClusterGaiaStep,
    PostProcessMagnitudeConvertStep,
    PostProcessProperMotionStep,
    PostProcessRegionStep,
    PostProcessSaveMagnitudesStep,
    SimbadAnnotateStep,
    WcsStep,
)


class AnalysisPipeline:
    """Orchestrator that runs pipeline steps in sequence."""

    def __init__(
        self,
        steps: list[PipelineStep] | None = None,
        config: PipelineConfig | None = None,
    ):
        self.steps = steps or self._default_steps()
        self.config = config or PipelineConfig()

    def _default_steps(self) -> list[PipelineStep]:
        return [
            WcsStep(),
            ExtractionStep(),
            SimbadAnnotateStep(),
            CorrelationIntraStep(),
            CorrelationInterStep(),
            ExtinctionFitStep(),
            CalibrationStep(),
            PostProcessRegionStep(),
            PostProcessClusterGaiaStep(),
            PostProcessProperMotionStep(),
            PostProcessMagnitudeConvertStep(),
            PostProcessSaveMagnitudesStep(),
            DeriveLimitingMagnitudeStep(),
            HipsReferenceSubtractStep(),
            LightCurveStep(),
        ]

    def _validate_magnitude_config(self) -> None:
        from ..post_processing.magnitude_systems import (
            apply_target_filter_system_alias,
            validate_magnitude_output_request,
        )

        cfg = self.config
        if cfg.target_filter_system:
            ofs, oms = apply_target_filter_system_alias(cfg.target_filter_system)
            if ofs is not None and cfg.output_filter_set == "auto":
                cfg.output_filter_set = ofs
            if oms is not None and cfg.output_magnitude_system == "auto":
                cfg.output_magnitude_system = oms
        validate_magnitude_output_request(
            output_filter_set=cfg.output_filter_set,
            output_magnitude_system=cfg.output_magnitude_system,
        )

    def run(self, context: AnalysisContext) -> AnalysisContext:
        self._validate_magnitude_config()
        _move_objects_to_observation_epoch(context)
        for step in self.steps:
            if step.skip(context, self.config):
                continue
            context = step.run(context, self.config)
        return context


def _move_objects_to_observation_epoch(context: AnalysisContext) -> None:
    """Objects of interest with a proper motion to the median observation time."""
    import numpy as np
    from astropy.coordinates import SkyCoord

    from ... import terminal_output

    jds = [
        float(jd)
        for series in (getattr(context, "image_series_dict", None) or {}).values()
        for jd in getattr(series, "get_observation_time", lambda: [])()
        if jd
    ]
    objects = getattr(context, "objects_of_interest", None) or []
    if not jds or not objects:
        return
    jd_obs = float(np.median(jds))
    moved = False
    for obj in objects:
        shift = obj.move_to_epoch(jd_obs) if hasattr(obj, "move_to_epoch") else 0.0
        if shift > 0.05:
            moved = True
            terminal_output.print_to_terminal(
                f"{obj.name}: position moved by its proper motion to the observation epoch "
                f"({shift:.2f}\")",
                indent=1,
            )
    observation = getattr(context, "observation", None)
    if moved and observation is not None and getattr(observation, "objects_of_interest_coordinates", None) is not None:
        observation.objects_of_interest_coordinates = SkyCoord(
            [o.ra for o in objects], [o.dec for o in objects], unit="deg", frame="icrs"
        )
