"""HiPS archival template subtraction for the reference science image."""

from __future__ import annotations

import shutil
from pathlib import Path

from .... import checks, terminal_output
from ....output_layout import work_dir
from ...post_processing.hips_reference_subtract import run_hips_reference_subtraction
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


def _hotpants_executable(config: PipelineConfig) -> str | None:
    exe = config.hips_reference_subtraction_hotpants_executable
    if exe:
        p = Path(exe)
        if p.is_file():
            return str(p.resolve())
        return shutil.which(exe)
    return shutil.which("hotpants")


class HipsReferenceSubtractStep(base.PipelineStep):
    """
    Post-processing: subtract a HiPS archive cutout from one science image.

    Subtracts with HOTPANTS when that binary is on ``PATH`` (or
    `hips_reference_subtraction_hotpants_executable`), otherwise a Python
    Alard–Lupton kernel (`hips_reference_subtraction_backend="auto"`).
    Disabled by default (`skip_hips_reference_subtraction`). HiPS/network
    failures warn and leave `context.hips_subtract_result` unset; analysis continues.
    """

    name = "hips_reference_subtract"

    def skip(self, context: AnalysisContext, config: PipelineConfig) -> bool:
        if config.skip_hips_reference_subtraction:
            return True
        obs = context.require_observation()
        if obs is None:
            return True
        flist = getattr(context, "filter_list", None) or []
        filter_ = config.hips_reference_subtraction_filter or (
            flist[0] if flist else None
        )
        if filter_ is None:
            return True
        series = obs.image_series_dict.get(filter_)
        if series is None or not getattr(series, "image_list", None):
            return True
        idx = config.hips_reference_subtraction_image_index
        if idx < 0 or idx >= len(series.image_list):
            return True
        backend = (config.hips_reference_subtraction_backend or "auto").strip().lower()
        if backend == "hotpants" and _hotpants_executable(config) is None:
            terminal_output.print_to_terminal(
                "HipsReferenceSubtractStep: backend is hotpants but the executable "
                "was not found; skipping (install HOTPANTS, set "
                "hips_reference_subtraction_hotpants_executable, or use "
                "hips_reference_subtraction_backend='auto' / 'alard_lupton').",
                style_name="WARNING",
            )
            return True
        return False

    def run(self, context: AnalysisContext, config: PipelineConfig) -> AnalysisContext:
        obs = context.require_observation()
        assert obs is not None
        output_dir = context.output_dir
        flist = context.filter_list or []
        filter_ = config.hips_reference_subtraction_filter or flist[0]
        series = obs.image_series_dict[filter_]
        idx = config.hips_reference_subtraction_image_index
        im = series.image_list[idx]
        science_path = str(im.path)
        checks.check_file(science_path)

        workdir = work_dir(output_dir, "subtract")
        checks.check_output_directories(str(Path(output_dir)), str(workdir))

        wcs_method = (
            config.hips_reference_subtraction_wcs_method or config.wcs_method
        )
        reuse = (
            series if config.hips_reference_subtraction_reuse_pipeline_wcs else None
        )
        extra = config.hips_reference_subtraction_hotpants_extra_args

        terminal_output.print_to_terminal(
            "HiPS reference image subtraction",
            style_name="HEADER",
        )
        try:
            result = run_hips_reference_subtraction(
                filter_,
                science_path,
                workdir,
                wcs_method=wcs_method,
                plot_comp=config.hips_reference_subtraction_plot_comp,
                hips_source=config.hips_reference_subtraction_hips_source,
                file_type_plots=config.file_type_plots,
                trim_slice_yx=config.hips_reference_subtraction_trim,
                reuse_wcs_image_series=reuse,
                hips_timeout_ms=config.hips_reference_subtraction_timeout_ms,
                hips_server=config.hips_reference_subtraction_server,
                hips_fallback_servers=config.hips_reference_subtraction_fallback_servers,
                hips_retries=config.hips_reference_subtraction_retries,
                hips_retry_backoff_s=config.hips_reference_subtraction_retry_backoff_s,
                hips_use_cache=config.hips_reference_subtraction_use_cache,
                hips_verbose=config.hips_reference_subtraction_verbose,
                subtract_backend=config.hips_reference_subtraction_backend,
                hotpants_executable=_hotpants_executable(config),
                hotpants_extra_args=extra if extra else None,
                hotpants_output_filename=config.hips_reference_subtraction_output_filename,
            )
        except Exception as exc:
            terminal_output.print_to_terminal(
                f"HipsReferenceSubtractStep failed; continuing analysis ({exc})",
                indent=2,
                style_name="WARNING",
            )
            context.hips_subtract_result = None
            return context
        context.hips_subtract_result = result
        cache_note = " (cached HiPS cutout)" if result.hips_from_cache else ""
        terminal_output.print_to_terminal(
            f"Difference image: {result.difference_fits} "
            f"[{result.hips_source}, {result.subtract_backend}]{cache_note}",
            indent=2,
            style_name="OKGREEN",
        )
        return context
