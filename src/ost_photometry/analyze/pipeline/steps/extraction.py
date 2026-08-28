"""Photometry extraction step (single or multi image per filter)."""

import multiprocessing as mp

from .... import terminal_output
from ....output_layout import diagnostics_dir
from ... import plots
from ...diagnostic_plot_hooks import run_diagnostic_plots_phase
from ...extraction import extract_multiprocessing, main_extract
from .. import base
from ..config import PipelineConfig
from ..context import AnalysisContext


class ExtractionStep(base.PipelineStep):
    """Extract flux and positions: single reference image or all images per filter."""

    name = "extraction"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        return config.skip_extraction or context.extraction_done

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        ext = config.extraction
        mode = context.get_extraction_mode(config)
        ref_id = ext.reference_image_index
        fwhm_dict = ext.fwhm_object_psf

        for filter_ in context.filter_list:
            terminal_output.print_to_terminal(
                f"Analyzing {filter_} images",
                style_name="HEADER",
            )

            image_series = context.image_series_dict[filter_]
            fwhm = fwhm_dict.get(filter_) if fwhm_dict else None

            if mode == "single":
                ref_image = image_series.image_list[ref_id]
                main_extract(ref_image, **ext.main_extract_kwargs(fwhm=fwhm))
            else:
                extract_multiprocessing(
                    image_series,
                    ext.n_cores_multiprocessing,
                    fwhm_object_psf=fwhm_dict,
                    **ext.extract_multiprocessing_kwargs(),
                )

            if mode == "multi" and ext.transform_object_positions_to_reference:
                from ...analyze import transform_object_positions

                transform_object_positions(
                    image_series,
                    output_dir=context.output_dir,
                )

        if mode == "single" and ext.photometry_extraction_method == "PSF":
            epsf_dict = {}
            img_dict = {}
            residual_dict = {}
            for key, image_series in context.image_series_dict.items():
                ref_idx = image_series.reference_image_index
                img = image_series.image_list[ref_idx]
                epsf_dict[key] = [img.epsf]
                img_dict[key] = img.get_data()
                if img.residual_image is not None:
                    residual_dict[key] = img.residual_image

            p = mp.Process(
                target=plots.plot_epsf,
                args=(str(diagnostics_dir(context.output_dir, "extraction")), epsf_dict),
                kwargs={"file_type": ext.file_type_plots},
            )
            p.start()
            p = mp.Process(
                target=plots.plot_residual,
                args=(img_dict, residual_dict, str(diagnostics_dir(context.output_dir, "extraction"))),
                kwargs={"file_type": ext.file_type_plots},
            )
            p.start()

        run_diagnostic_plots_phase(context, config, "extraction")

        context.extraction_done = True
        return context
