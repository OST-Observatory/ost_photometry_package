"""Tests for PipelineConfig sub-structure, presets, and flat compatibility."""

import warnings

from helpers import load_module_from_path, pkg_src


def test_pipeline_config_flat_and_nested_access():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig
    ExtractionConfig = cfg_mod.ExtractionConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = PipelineConfig(skip_extraction=True, calibration_module="differential")
    assert cfg.skip_extraction is True
    assert cfg.extraction.skip_extraction is True
    assert cfg.resolved_calibration_strategy() == "linear_fit"

    ext = ExtractionConfig()
    kw = ext.main_extract_kwargs(fwhm=3.5)
    assert kw["fwhm_object_psf"] == 3.5
    assert "sigma_value_background_clipping" in kw

    cfg.apply_overrides(diagnostic_plots__photometry_mag_vs_error_scatter=True)
    assert cfg.diagnostic_plots.photometry_mag_vs_error_scatter is True


def test_pipeline_config_from_preset_overrides():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig.from_preset("c7_variable", overrides={"fit_sigma_clip": 3.5})
    assert cfg.fit_sigma_clip == 3.5
    assert cfg.differential_fit_sigma_clip == 3.5
    assert cfg.resolved_extinction_mode() == "none"


def test_calibration_field_aliases():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(exposure_pairing="index", exposure_jd_tolerance=0.05)
    assert cfg.differential_exposure_pairing == "index"
    assert cfg.differential_exposure_jd_tolerance == 0.05
