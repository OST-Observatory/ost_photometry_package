"""Tests for PipelineConfig sub-structure, presets, and flat compatibility."""

import pytest

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

    cfg = PipelineConfig.from_preset("linear_fit_per_night", overrides={"fit_sigma_clip": 3.5})
    assert cfg.fit_sigma_clip == 3.5
    assert cfg.calibration_strategy == "linear_fit"
    assert cfg.extinction_mode == "none"


def test_linear_fit_per_night_extinction_preset():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig.from_preset("linear_fit_per_night_extinction")
    assert cfg.calibration_strategy == "linear_fit"
    assert cfg.extinction_mode == "from_comparison_stars"
    assert cfg.color_term_fit == "auto"
    assert not hasattr(cfg, "fit_extinction_from_data")

    cfg_va = PipelineConfig(extinction_mode="from_value_airmass")
    assert cfg_va.extinction_mode == "from_value_airmass"


def test_deprecated_preset_aliases_still_resolve():
    import warnings

    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = PipelineConfig.from_preset("c7_variable")
    assert cfg.calibration_strategy == "linear_fit"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", DeprecationWarning)
        assert (
            PipelineConfig.from_preset("n2_stack").calibration_strategy == "median_zp"
        )
        assert (
            PipelineConfig.from_preset(
                "c7_variable_extinction"
            ).extinction_mode
            == "from_comparison_stars"
        )
        assert PipelineConfig.from_preset(
            "mk_calib_trans"
        ).protect_calibration_objects is True
        assert (
            PipelineConfig.from_preset("mk_calib_calibrate").calibration_grouping
            == "ensemble"
        )
        assert (
            PipelineConfig.from_preset("ost_site").extinction_mode == "tabulated"
        )


def test_calibration_field_names():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(exposure_pairing="index", exposure_jd_tolerance=0.05)
    assert cfg.exposure_pairing == "index"
    assert cfg.exposure_jd_tolerance == 0.05


def test_protect_calibration_objects_and_extract_protect_calibrators_preset():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(protect_calibration_objects=True)
    assert cfg.protect_calibration_objects is True
    assert cfg.correlation.protect_calibration_objects is True

    mk = PipelineConfig.from_preset("extract_protect_calibrators")
    assert mk.protect_calibration_objects is True
    assert mk.skip_calibration is True
    assert mk.skip_correlation_inter is True
    assert mk.skip_light_curve is True
    assert mk.extinction_mode == "none"


def test_extinction_order_and_k_second_config():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(
        extinction_mode="tabulated",
        extinction_order="second",
        k_second={"B": 0.025, "V": 0.012},
    )
    assert cfg.extinction_order == "second"
    assert cfg.calibration.extinction_order == "second"
    assert cfg.k_second["B"] == pytest.approx(0.025)
    assert cfg.as_flat_dict()["extinction_order"] == "second"
    assert cfg.as_flat_dict()["k_second"]["V"] == pytest.approx(0.012)


def test_linear_fit_ensemble_preset():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig.from_preset("linear_fit_ensemble")
    assert cfg.calibration_strategy == "linear_fit"
    assert cfg.derive_transform_from_data is True
    assert cfg.calibration_grouping == "ensemble"
    assert cfg.extinction_mode == "none"
    assert cfg.color_term_fit == "never"
    assert cfg.skip_calibration is False


def test_tabulated_extinction_preset():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig.from_preset("tabulated_extinction")
    assert cfg.extinction_mode == "tabulated"
    assert cfg.path_extinction_coefficients is None


def test_path_extinction_coefficients_flat_access():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(path_extinction_coefficients="/tmp/site_extinction.json")
    assert cfg.path_extinction_coefficients == "/tmp/site_extinction.json"
    assert cfg.extinction.path_extinction_coefficients == "/tmp/site_extinction.json"
