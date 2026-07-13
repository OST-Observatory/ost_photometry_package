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
    color_term_fit: str = "auto",
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
        color_term_fit=color_term_fit,
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
def test_never_vs_always_color_term_fit_on_synthetic(synthetic_calibration_epoch_table):
    """Never (median ZP) and always (linear T/ZP) should agree within tolerance on synthetic data."""
    tbl = synthetic_calibration_epoch_table
    _, zp_never = _differential_fit_zp_t(tbl, "V", ("B", "V"), color_term_fit="never")
    _, zp_always = _differential_fit_zp_t(tbl, "V", ("B", "V"), color_term_fit="always")
    assert abs(zp_never - zp_always) < 0.2


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
def test_pipeline_config_presets():
    """Named presets must set expected calibration strategy and grouping."""
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _PKG_SRC / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    n2 = PipelineConfig.from_preset("n2_stack")
    assert n2.calibration_strategy == "median_zp"
    assert n2.calibration_grouping == "per_image"
    assert n2.extinction_mode == "none"

    c7 = PipelineConfig.from_preset("c7_variable")
    assert c7.calibration_strategy == "linear_fit"
    assert c7.calibration_grouping == "per_night"
    assert c7.extinction_mode == "none"


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


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_derive_transform_fit_on_synthetic(synthetic_calibration_epoch_table):
    """Derive-transform backend produces finite c/ZP for two-filter epoch tables."""
    derive_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.derive_transform",
        _PKG_SRC / "ost_photometry" / "analyze" / "calibration" / "derive_transform.py",
    )
    tbl = synthetic_calibration_epoch_table
    fitted = derive_mod.fit_epoch_derive_transform(tbl, "epoch_000", ["B", "V"])
    assert fitted is not None
    result, derive_fit = fitted
    assert "B" in result.transformation
    assert "V" in result.transformation
    assert np.isfinite(result.transformation["B"].color_term)
    assert np.isfinite(result.transformation["B"].zero_point)
    assert derive_fit.n_stars_used >= 5


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_engine_derive_transform_from_data(synthetic_calibration_epoch_table):
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

    cfg = PipelineConfig(
        calibration_strategy="linear_fit",
        derive_transform_from_data=True,
    )
    tbl = synthetic_calibration_epoch_table
    epochs = {"epoch_000": tbl}
    results = CalibrationEngine.fit(epochs, cfg, ["B", "V"])
    assert "epoch_000" in results
    assert "derive_transform" in results["epoch_000"].notes


def test_plot_calibration_transformation_fit_line_ignores_nan_colors(tmp_path):
    """Fit line x-range must use finite masked colors only (cluster tables have NaN elsewhere)."""
    pytest.importorskip("photutils")
    from unittest.mock import patch

    plots_mod = load_module_from_path(
        "ost_photometry.analyze.plots._legacy",
        _PKG_SRC / "ost_photometry" / "analyze" / "plots" / "_legacy.py",
    )
    result_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.result",
        _PKG_SRC / "ost_photometry" / "analyze" / "calibration" / "result.py",
    )
    TransformationCoefficients = result_mod.TransformationCoefficients

    n = 20
    color = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    mask = np.zeros(n, dtype=bool)
    color[0:5] = np.linspace(0.4, 1.0, 5)
    delta[0:5] = 0.5 * color[0:5] + 0.2
    mask[0:5] = True

    coeffs = {
        "B": TransformationCoefficients(
            filter_name="B",
            color_term=0.5,
            zero_point=0.2,
            color_index_filters=("B", "V"),
        )
    }
    plot_calls: list = []

    def _fake_plot(x, y, *args, **kwargs):
        plot_calls.append((np.asarray(x), np.asarray(y)))

    with patch("matplotlib.pyplot.subplots"), patch(
        "matplotlib.pyplot.savefig"
    ), patch("matplotlib.pyplot.close"), patch(
        "ost_photometry.checks.check_output_directories"
    ), patch("matplotlib.pyplot.scatter"), patch(
        "matplotlib.pyplot.axhline"
    ), patch("matplotlib.pyplot.plot", side_effect=_fake_plot):
        plots_mod.plot_calibration_transformation(
            tmp_path,
            "epoch_000",
            {"B": (color, delta, mask)},
            coeffs,
            file_type="pdf",
        )

    assert len(plot_calls) == 1
    x_line, y_line = plot_calls[0]
    assert np.all(np.isfinite(x_line))
    assert np.isclose(x_line[0], 0.4, atol=1e-6)
    assert np.isclose(x_line[-1], 1.0, atol=1e-6)
    assert np.allclose(y_line, 0.5 * x_line + 0.2)
