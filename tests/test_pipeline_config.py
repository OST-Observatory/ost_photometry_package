"""Tests for PipelineConfig sub-structure and flat compatibility."""

from helpers import load_module_from_path, pkg_src


def test_pipeline_config_flat_and_nested_access():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig
    ExtractionConfig = cfg_mod.ExtractionConfig

    cfg = PipelineConfig(skip_extraction=True, calibration_module="differential")
    assert cfg.skip_extraction is True
    assert cfg.extraction.skip_extraction is True
    assert cfg.calibration_module == "differential"

    ext = ExtractionConfig()
    kw = ext.main_extract_kwargs(fwhm=3.5)
    assert kw["fwhm_object_psf"] == 3.5
    assert "sigma_value_background_clipping" in kw

    cfg.apply_overrides(diagnostic_plots__photometry_mag_vs_error_scatter=True)
    assert cfg.diagnostic_plots.photometry_mag_vs_error_scatter is True
