"""Photometry extraction step (single or multi image per filter)."""

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ...extraction import main_extract, extract_multiprocessing


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
        from .... import terminal_output
        from ... import plots
        import multiprocessing as mp

        mode = context.get_extraction_mode(config)
        ref_id = config.reference_image_id
        fwhm_dict = config.fwhm_object_psf

        for filter_ in context.filter_list:
            terminal_output.print_to_terminal(
                f"Analyzing {filter_} images",
                style_name="HEADER",
            )

            image_series = context.image_series_dict[filter_]
            fwhm = fwhm_dict.get(filter_) if fwhm_dict else None

            if mode == "single":
                ref_image = image_series.image_list[ref_id]
                main_extract(
                    ref_image,
                    fwhm_object_psf=fwhm,
                    sigma_value_background_clipping=config.sigma_value_background_clipping,
                    multiplier_background_rms=config.multiplier_background_rms,
                    size_epsf_region=config.size_epsf_region,
                    size_extraction_region_epsf=config.size_extraction_region_epsf,
                    epsf_fitter=config.epsf_fitter,
                    n_iterations_eps_extraction=config.n_iterations_eps_extraction,
                    fraction_epsf_stars=config.fraction_epsf_stars,
                    oversampling_factor_epsf=config.oversampling_factor_epsf,
                    max_n_iterations_epsf_determination=config.max_n_iterations_epsf_determination,
                    use_initial_positions_epsf=config.use_initial_positions_epsf,
                    object_finder_method=config.object_finder_method,
                    multiplier_background_rms_epsf=config.multiplier_background_rms_epsf,
                    multiplier_grouper_epsf=config.multiplier_grouper_epsf,
                    strict_cleaning_epsf_results=config.strict_cleaning_epsf_results,
                    minimum_n_eps_stars=config.minimum_n_eps_stars,
                    strict_epsf_checks=config.strict_epsf_checks,
                    photometry_extraction_method=config.photometry_extraction_method,
                    radius_aperture=config.radius_aperture,
                    inner_annulus_radius=config.inner_annulus_radius,
                    outer_annulus_radius=config.outer_annulus_radius,
                    radii_unit=config.radii_unit,
                    cosmic_ray_removal=config.cosmic_ray_removal,
                    limiting_contrast_rm_cosmics=config.limiting_contrast_rm_cosmics,
                    read_noise=config.read_noise,
                    sigma_clipping_value=config.sigma_clipping_value,
                    saturation_level=config.saturation_level,
                    plots_for_all_images=config.plots_for_all_images,
                    file_type_plots=config.file_type_plots,
                    use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
                    annotate_image=config.annotate_image,
                    magnitude_limit_image_annotation=config.magnitude_limit_image_annotation,
                    filter_magnitude_limit_image_annotation=config.filter_magnitude_limit_image_annotation,
                )
            else:
                extract_multiprocessing(
                    image_series,
                    config.n_cores_multiprocessing,
                    fwhm_object_psf=fwhm_dict,
                    sigma_value_background_clipping=config.sigma_value_background_clipping,
                    multiplier_background_rms=config.multiplier_background_rms,
                    size_epsf_region=config.size_epsf_region,
                    size_extraction_region_epsf=config.size_extraction_region_epsf,
                    epsf_fitter=config.epsf_fitter,
                    n_iterations_eps_extraction=config.n_iterations_eps_extraction,
                    fraction_epsf_stars=config.fraction_epsf_stars,
                    oversampling_factor_epsf=config.oversampling_factor_epsf,
                    max_n_iterations_epsf_determination=config.max_n_iterations_epsf_determination,
                    object_finder_method=config.object_finder_method,
                    multiplier_background_rms_epsf=config.multiplier_background_rms_epsf,
                    multiplier_grouper_epsf=config.multiplier_grouper_epsf,
                    strict_cleaning_epsf_results=config.strict_cleaning_epsf_results,
                    minimum_n_eps_stars=config.minimum_n_eps_stars,
                    strict_epsf_checks=config.strict_epsf_checks,
                    photometry_extraction_method=config.photometry_extraction_method,
                    radius_aperture=config.radius_aperture,
                    inner_annulus_radius=config.inner_annulus_radius,
                    outer_annulus_radius=config.outer_annulus_radius,
                    radii_unit=config.radii_unit,
                    plots_for_all_images=config.plots_for_all_images,
                    use_wcs_projection_for_star_maps=config.use_wcs_projection_for_star_maps,
                    file_type_plots=config.file_type_plots,
                    use_initial_positions_epsf=config.use_initial_positions_epsf,
                    annotate_reference_image=config.annotate_reference_image,
                    magnitude_limit_image_annotation=config.magnitude_limit_image_annotation,
                    filter_magnitude_limit_image_annotation=config.filter_magnitude_limit_image_annotation,
                )

            # Transform object positions to reference frame (multi mode only)
            if mode == "multi" and config.transform_object_positions_to_reference:
                from ...analyze import transform_object_positions

                transform_object_positions(
                    image_series,
                    output_dir=context.output_dir,
                )

        # Plot ePSFs and residuals for PSF mode (single-image case from extract_flux)
        if mode == "single" and config.photometry_extraction_method == "PSF":
            epsf_dict = {}
            img_dict = {}
            residual_dict = {}
            for key, image_series in context.image_series_dict.items():
                ref_id = image_series.reference_image_id
                img = image_series.image_list[ref_id]
                epsf_dict[key] = [img.epsf]
                img_dict[key] = img.get_data()
                if img.residual_image is not None:
                    residual_dict[key] = img.residual_image

            p = mp.Process(
                target=plots.plot_epsf,
                args=(context.output_dir, epsf_dict),
                kwargs={"file_type": config.file_type_plots},
            )
            p.start()
            p = mp.Process(
                target=plots.plot_residual,
                args=(img_dict, residual_dict, context.output_dir),
                kwargs={"file_type": config.file_type_plots},
            )
            p.start()

        context.extraction_done = True
        return context
