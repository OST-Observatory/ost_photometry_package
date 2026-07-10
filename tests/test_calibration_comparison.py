"""Legacy vs differential calibration comparison harness (Phase 0+)."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest
from astropy.table import Table

from helpers import load_module_from_path, pkg_src

_PKG_SRC = pkg_src()

pytestmark = pytest.mark.comparison


def _import_submodule(dotted: str):
    return importlib.import_module(dotted)


def _deps_available() -> bool:
    try:
        import photutils  # noqa: F401
        import regions  # noqa: F401
        return True
    except ImportError:
        return False


def _legacy_linear_fit_zp_t(
    inst: np.ndarray, std: np.ndarray, color: np.ndarray
) -> tuple[float, float]:
    """Simple legacy-style fit: delta = T * color + ZP."""
    delta = std - inst
    a = np.column_stack([color, np.ones_like(color)])
    coeffs, _, _, _ = np.linalg.lstsq(a, delta, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _differential_fit_zp_t(
    tbl: Table,
    filter_: str,
    color_filters: tuple[str, str],
    *,
    zp_method: str = "auto",
    extinction_order: str = "none",
) -> tuple[float, float]:
    diff_mod = _import_submodule("ost_photometry.analyze.differential_photometry")
    DifferentialPhotometer = diff_mod.DifferentialPhotometer
    ExtinctionCorrector = _import_submodule(
        "ost_photometry.analyze.extinction"
    ).ExtinctionCorrector
    ExtinctionOrder = _import_submodule("ost_photometry.analyze.extinction").ExtinctionOrder

    mask = np.asarray(tbl["is_comparison"], dtype=bool)
    ext_order = ExtinctionOrder.NONE if extinction_order == "none" else ExtinctionOrder.FIRST
    phot = DifferentialPhotometer(
        color_indices={filter_: color_filters},
        extinction_corrector=ExtinctionCorrector(order=ext_order),
    )
    result = phot.fit_transformation_epoch(
        tbl,
        epoch_id="epoch_000",
        filters=[filter_],
        comparison_mask=mask,
        mag_col_prefix="mag_",
        std_col_prefix="mag_std_",
        determine_color_terms=True,
        zp_method=zp_method,
        output_dir=None,
    )
    tc = result.transformation[filter_]
    return tc.color_term, tc.zero_point


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_legacy_vs_differential_zp_color_term(
    synthetic_calibration_epoch_table, comparison_tolerance
):
    """Both calibration paths should yield similar T/ZP on synthetic data."""
    tbl = synthetic_calibration_epoch_table
    mask = np.asarray(tbl["is_comparison"], dtype=bool)
    color = np.asarray(tbl["mag_B"][mask] - tbl["mag_V"][mask])
    legacy_t, legacy_zp = _legacy_linear_fit_zp_t(
        np.asarray(tbl["mag_V"][mask]),
        np.asarray(tbl["mag_std_V"][mask]),
        color,
    )
    diff_t, diff_zp = _differential_fit_zp_t(tbl, "V", ("B", "V"))

    tol = comparison_tolerance
    assert abs(legacy_t - diff_t) < tol["color_term_abs"], (
        f"color term mismatch: legacy={legacy_t}, differential={diff_t}"
    )
    assert abs(legacy_zp - diff_zp) < tol["zp_abs"], (
        f"zero point mismatch: legacy={legacy_zp}, differential={diff_zp}"
    )


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_median_vs_linear_zp_on_synthetic(synthetic_calibration_epoch_table):
    """Median and linear ZP methods should agree within tolerance on synthetic data."""
    tbl = synthetic_calibration_epoch_table
    _, zp_median = _differential_fit_zp_t(tbl, "V", ("B", "V"), zp_method="median")
    _, zp_linear = _differential_fit_zp_t(tbl, "V", ("B", "V"), zp_method="linear")
    assert abs(zp_median - zp_linear) < 0.2


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_extinction_none_vs_tabulated_constant_airmass(synthetic_calibration_epoch_table):
    """With constant airmass, none vs tabulated extinction should give similar ZP."""
    tbl = synthetic_calibration_epoch_table
    _, zp_none = _differential_fit_zp_t(
        tbl, "V", ("B", "V"), extinction_order="none"
    )
    _, zp_tab = _differential_fit_zp_t(
        tbl, "V", ("B", "V"), extinction_order="first"
    )
    assert abs(zp_none - zp_tab) < 0.05


@pytest.mark.comparison
def test_pipeline_config_presets_and_module_alias():
    """Presets and deprecated calibration_module alias must resolve correctly."""
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _PKG_SRC / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    n2 = PipelineConfig.from_preset("n2_stack")
    assert n2.resolved_calibration_strategy() == "median_zp"
    assert n2.resolved_calibration_grouping() == "per_image"
    assert n2.resolved_extinction_mode() == "none"

    c7 = PipelineConfig.from_preset("c7_variable")
    assert c7.resolved_calibration_strategy() == "linear_fit"
    assert c7.resolved_calibration_grouping() == "per_night"
    assert c7.resolved_extinction_mode() == "none"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", DeprecationWarning)
        legacy = PipelineConfig(calibration_module="legacy")
        diff = PipelineConfig(calibration_module="differential")
    assert legacy.resolved_calibration_strategy() == "median_zp"
    assert diff.resolved_calibration_strategy() == "linear_fit"
    assert diff.resolved_calibration_grouping() == "per_night"


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_calibration_epochs_schema_roundtrip(synthetic_calibration_epoch_table):
    """Bridge-style epoch tables keep required columns for both paths."""
    schema_mod = _import_submodule("ost_photometry.analyze.post_processing.schema")
    ensure_epoch_native = schema_mod.ensure_epoch_native_photometry_table

    tbl = synthetic_calibration_epoch_table
    required = {"id", "ra", "dec", "mag_B", "mag_V"}
    assert required.issubset(set(tbl.colnames))
    native = ensure_epoch_native(tbl)
    assert len(native) == len(tbl)


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_calibration_engine_median_zp(synthetic_calibration_epoch_table):
    """CalibrationEngine median_zp backend fits ZP on synthetic epoch table."""
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _PKG_SRC / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    engine_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.engine",
        _PKG_SRC / "ost_photometry" / "analyze" / "calibration" / "engine.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig
    CalibrationEngine = engine_mod.CalibrationEngine

    cfg = PipelineConfig.from_preset("n2_stack")
    tbl = synthetic_calibration_epoch_table
    epochs = {"epoch_000": tbl}
    results = CalibrationEngine.fit(epochs, cfg, ["B", "V"])
    assert "epoch_000" in results
    assert "V" in results["epoch_000"].transformation
    assert np.isfinite(results["epoch_000"].transformation["V"].zero_point)


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_prepare_calibration_check_plots_array_shapes(
    synthetic_calibration_epoch_table, tmp_path
):
    """Plot helper must pass color, delta, mask arrays of equal length."""
    from unittest.mock import patch

    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _PKG_SRC / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    engine_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.engine",
        _PKG_SRC / "ost_photometry" / "analyze" / "calibration" / "engine.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig
    CalibrationEngine = engine_mod.CalibrationEngine

    cfg = PipelineConfig.from_preset("n2_stack")
    tbl = synthetic_calibration_epoch_table
    epochs = {"epoch_000": tbl}
    results = CalibrationEngine.fit(epochs, cfg, ["B", "V"])

    captured = {}

    def _capture_plot(_out, _eid, plot_data, _coeffs, **_kw):
        captured["plot_data"] = plot_data

    with patch(
        "ost_photometry.analyze.plots.plot_calibration_transformation",
        side_effect=_capture_plot,
    ):
        engine_mod.prepare_calibration_check_plots(
            str(tmp_path), epochs, results, ["B", "V"]
        )

    n = len(tbl)
    for color, delta, mask in captured["plot_data"].values():
        assert len(color) == n
        assert len(delta) == n
        assert len(mask) == n
