"""Tests for PipelineConfig sub-structure, presets, and flat compatibility."""

from helpers import load_module_from_path, pkg_src


def test_pipeline_config_flat_and_nested_access():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig
    ExtractionConfig = cfg_mod.ExtractionConfig

    cfg = PipelineConfig(skip_extraction=True, calibration_strategy="linear_fit")
    assert cfg.skip_extraction is True
    assert cfg.extraction.skip_extraction is True
    assert cfg.calibration_strategy == "linear_fit"

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
    assert cfg.calibration_strategy == "linear_fit"
    assert cfg.extinction_mode == "none"


def test_c7_variable_extinction_preset():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig.from_preset("c7_variable_extinction")
    assert cfg.calibration_strategy == "linear_fit"
    assert cfg.extinction_mode == "from_comparison_stars"
    assert cfg.color_term_fit == "auto"
    assert not hasattr(cfg, "fit_extinction_from_data")

    cfg_va = PipelineConfig(extinction_mode="from_value_airmass")
    assert cfg_va.extinction_mode == "from_value_airmass"


def test_calibration_field_names():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(exposure_pairing="index", exposure_jd_tolerance=0.05)
    assert cfg.exposure_pairing == "index"
    assert cfg.exposure_jd_tolerance == 0.05


def test_path_extinction_coefficients_flat_access():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(path_extinction_coefficients="/tmp/site_extinction.json")
    assert cfg.path_extinction_coefficients == "/tmp/site_extinction.json"
    assert cfg.extinction.path_extinction_coefficients == "/tmp/site_extinction.json"
