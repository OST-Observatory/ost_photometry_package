"""Legacy vs differential calibration comparison harness (Phase 0+)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from astropy.table import Table

from helpers import (
    ensure_stub_package,
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
    stub_analyze_package,
)

_PKG_SRC = pkg_src()

pytestmark = pytest.mark.comparison


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


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
    """At constant airmass, first-order extinction is absorbed into ZP (shift k'·X)."""
    tbl = synthetic_calibration_epoch_table
    _, zp_none = _differential_fit_zp_t(
        tbl, "V", ("B", "V"), extinction_order="none"
    )
    _, zp_tab = _differential_fit_zp_t(
        tbl, "V", ("B", "V"), extinction_order="first"
    )
    ext_mod = _import_submodule("ost_photometry.analyze.extinction")
    k_v = ext_mod.DEFAULT_EXTINCTION["V"].k_prime
    x = float(np.median(tbl["airmass"]))
    assert abs((zp_tab - zp_none) - k_v * x) < 0.05


@pytest.mark.comparison
def test_pipeline_config_presets():
    """Named presets must set expected calibration strategy and grouping."""
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _PKG_SRC / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    n2 = PipelineConfig.from_preset("median_zp_per_image")
    assert n2.calibration_strategy == "median_zp"
    assert n2.calibration_grouping == "per_image"
    assert n2.extinction_mode == "none"

    c7 = PipelineConfig.from_preset("linear_fit_per_night")
    assert c7.calibration_strategy == "linear_fit"
    assert c7.calibration_grouping == "per_night"
    assert c7.extinction_mode == "none"


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_calibration_epochs_schema_roundtrip(synthetic_calibration_epoch_table):
    """Bridge-style epoch tables keep required columns for both paths."""
    adapters_mod = _import_submodule(
        "ost_photometry.analyze.post_processing.adapters"
    )
    ensure_epoch_native = adapters_mod.ensure_epoch_native_photometry_table

    tbl = synthetic_calibration_epoch_table
    required = {"id", "epoch_id", "ra", "dec", "mag_B", "mag_V"}
    assert required.issubset(set(tbl.colnames))
    native = ensure_epoch_native(tbl)
    assert len(native) == len(tbl)
    assert native.meta.get("photometry_schema")

    single = tbl.copy()
    single.remove_column("epoch_id")
    native_single = ensure_epoch_native(single)
    assert len(native_single) == len(single)
    assert native_single.meta.get("photometry_schema")


def _load_engine():
    analyze = stub_analyze_package("calibration", "pipeline")
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        analyze / "pipeline" / "config.py",
    )
    engine_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.engine",
        analyze / "calibration" / "engine.py",
    )
    return cfg_mod, engine_mod


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_calibration_engine_median_zp(synthetic_calibration_epoch_table):
    """CalibrationEngine median_zp backend fits ZP on synthetic epoch table."""
    cfg_mod, engine_mod = _load_engine()
    PipelineConfig = cfg_mod.PipelineConfig
    CalibrationEngine = engine_mod.CalibrationEngine

    cfg = PipelineConfig.from_preset("median_zp_per_image")
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

    cfg_mod, engine_mod = _load_engine()
    PipelineConfig = cfg_mod.PipelineConfig
    CalibrationEngine = engine_mod.CalibrationEngine

    cfg = PipelineConfig.from_preset("median_zp_per_image")
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
    analyze = stub_analyze_package("calibration")
    derive_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.derive_transform",
        analyze / "calibration" / "derive_transform.py",
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
def test_derive_transform_sigma_clip_rejects_outlier(synthetic_calibration_epoch_table):
    """Tight fit_sigma_clip should drop a large residual outlier from the used mask."""
    analyze = stub_analyze_package("calibration")
    derive_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.derive_transform",
        analyze / "calibration" / "derive_transform.py",
    )
    tbl = synthetic_calibration_epoch_table.copy()
    # Corrupt one comparison star far from the color-fit locus.
    tbl["mag_B"][0] = float(tbl["mag_B"][0]) + 2.5
    fitted_loose = derive_mod.fit_epoch_derive_transform(
        tbl, "epoch_000", ["B", "V"], sigma_clip=10.0
    )
    fitted_tight = derive_mod.fit_epoch_derive_transform(
        tbl, "epoch_000", ["B", "V"], sigma_clip=2.0
    )
    assert fitted_loose is not None and fitted_tight is not None
    _, fit_loose = fitted_loose
    _, fit_tight = fitted_tight
    assert fit_tight.n_stars_used < fit_loose.n_stars_used
    assert not fit_tight.comparison_mask[0]


@pytest.mark.comparison
@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_engine_derive_transform_from_data(synthetic_calibration_epoch_table):
    cfg_mod, engine_mod = _load_engine()
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
    pytest.importorskip("matplotlib")
    from unittest.mock import MagicMock, patch

    root = pkg_src() / "ost_photometry"
    analyze = root / "analyze"
    ensure_stub_package("ost_photometry", path=root)
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package(
        "ost_photometry.analyze.calibration",
        path=analyze / "calibration",
    )
    ensure_stub_package("ost_photometry.analyze.plots", path=analyze / "plots")
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        analyze / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.extinction",
        analyze / "extinction.py",
    )
    result_mod = load_module_from_path(
        "ost_photometry.analyze.calibration.result",
        analyze / "calibration" / "result.py",
    )
    qc_mod = load_module_from_path(
        "ost_photometry.analyze.plots.calibration_qc",
        analyze / "plots" / "calibration_qc.py",
    )
    TransformationCoefficients = result_mod.TransformationCoefficients
    plot_calibration_transformation = qc_mod.plot_calibration_transformation

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
        return MagicMock()

    fig = MagicMock()
    ax1 = MagicMock()
    ax2 = MagicMock()
    ax1.plot.side_effect = _fake_plot
    qc = "ost_photometry.analyze.plots.calibration_qc"

    with patch(f"{qc}.plt.subplots", return_value=(fig, (ax1, ax2))), patch(
        f"{qc}.plt.savefig"
    ), patch(f"{qc}.plt.close"), patch(f"{qc}.plt.tight_layout"), patch(
        f"{qc}.checks.check_output_directories"
    ):
        plot_calibration_transformation(
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
