"""Tests for filter-set / Vega–AB magnitude-system model and conversion."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from astropy.table import Table

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _deps_available() -> bool:
    try:
        import photutils  # noqa: F401
        import regions  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_namespace() -> None:
    """Register package namespaces without executing analyze.__init__."""
    root = pkg_src() / "ost_photometry"
    ensure_stub_package("ost_photometry", path=root)
    ensure_stub_package("ost_photometry.analyze", path=root / "analyze")
    ensure_stub_package(
        "ost_photometry.analyze.post_processing",
        path=root / "analyze" / "post_processing",
    )
    ensure_stub_package(
        "ost_photometry.analyze.calibration_sources",
        path=root / "analyze" / "calibration_sources",
    )


def _ms_mod():
    _ensure_namespace()
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        pkg_src() / "ost_photometry" / "analyze" / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.calibration_parameters",
        pkg_src() / "ost_photometry" / "calibration_parameters.py",
    )
    try:
        load_module_from_path(
            "ost_photometry.terminal_output",
            pkg_src() / "ost_photometry" / "terminal_output.py",
        )
    except Exception:
        to = ModuleType("ost_photometry.terminal_output")

        def _print(*_a, **_k):
            return None

        to.print_to_terminal = _print  # type: ignore[attr-defined]
        sys.modules["ost_photometry.terminal_output"] = to
        sys.modules["ost_photometry"].terminal_output = to  # type: ignore[attr-defined]

    import ost_photometry as ost

    ost.calibration_parameters = sys.modules["ost_photometry.calibration_parameters"]  # type: ignore[attr-defined]
    ost.terminal_output = sys.modules["ost_photometry.terminal_output"]  # type: ignore[attr-defined]

    return load_module_from_path(
        "ost_photometry.analyze.post_processing.magnitude_systems",
        pkg_src()
        / "ost_photometry"
        / "analyze"
        / "post_processing"
        / "magnitude_systems.py",
    )


def _load_convert_stack():
    _ms_mod()
    load_module_from_path(
        "ost_photometry.analyze.calibration_sources.transforms",
        pkg_src()
        / "ost_photometry"
        / "analyze"
        / "calibration_sources"
        / "transforms.py",
    )
    return load_module_from_path(
        "ost_photometry.analyze.post_processing.convert",
        pkg_src() / "ost_photometry" / "analyze" / "post_processing" / "convert.py",
    )


def test_illegal_sdss_vega_aborts():
    ms = _ms_mod()
    with pytest.raises(ValueError, match="SDSS"):
        ms.validate_magnitude_output_request(
            output_filter_set="sdss",
            output_magnitude_system="vega",
        )


def test_auto_defaults_follow_catalog():
    ms = _ms_mod()
    eff = ms.resolve_effective_output(
        output_filter_set="auto",
        output_magnitude_system="auto",
        calibrated_filter_set="bessell",
        catalog_magnitude_system="vega",
        convert_magnitudes=False,
    )
    assert eff.filter_set == "bessell"
    assert eff.magnitude_system == "vega"
    assert not eff.needs_convert

    eff_sdss = ms.resolve_effective_output(
        output_filter_set="auto",
        output_magnitude_system="auto",
        calibrated_filter_set="sdss",
        catalog_magnitude_system="ab",
        convert_magnitudes=False,
    )
    assert eff_sdss.filter_set == "sdss"
    assert eff_sdss.magnitude_system == "ab"


def test_resolve_catalog_apass_and_sdss():
    ms = _ms_mod()
    assert ms.resolve_catalog_magnitude_system("APASS") == "vega"
    assert ms.resolve_catalog_magnitude_system("SDSS_Release_16") == "ab"
    assert ms.resolve_catalog_magnitude_system("no_such_catalog") == "unknown"


def test_axis_suffix():
    ms = _ms_mod()
    assert "Vega" in ms.magnitude_system_axis_suffix("vega")
    assert "AB" in ms.magnitude_system_axis_suffix("ab")


def test_vega_to_ab_offset_on_v():
    ms = _ms_mod()
    assert ms.vega_to_ab_offset("V") == pytest.approx(0.02)


def test_infer_filter_set():
    ms = _ms_mod()
    assert ms.infer_filter_set(["B", "V"]) == "bessell"
    assert ms.infer_filter_set(["g", "r", "i"]) == "sdss"
    assert ms.infer_filter_set(["B", "g"]) == "mixed"


def test_zp_flip_changes_mag_cal_v():
    convert = _load_convert_stack()
    tbl = Table()
    tbl["mag_cal_V"] = np.array([12.0, 13.0])
    tbl["err_cal_V"] = np.array([0.01, 0.02])
    out = convert.convert_magnitudes_to_other_system(
        tbl,
        output_filter_set="bessell",
        output_magnitude_system="ab",
        calibration_source="APASS",
        source_magnitude_system="vega",
        source_filter_set="bessell",
    )
    assert np.allclose(out["mag_cal_V"], np.array([12.0, 13.0]) + 0.02)
    assert np.allclose(out["err_cal_V"], [0.01, 0.02])
    assert out.meta.get("ost_photometry.magnitude_system") == "ab"


def test_bessell_to_sdss_jordi_smoke():
    convert = _load_convert_stack()
    tbl = Table()
    n = 5
    tbl["mag_cal_B"] = np.full(n, 14.0)
    tbl["err_cal_B"] = np.full(n, 0.02)
    tbl["mag_cal_V"] = np.full(n, 13.5)
    tbl["err_cal_V"] = np.full(n, 0.02)
    out = convert.convert_magnitudes_to_other_system(
        tbl,
        output_filter_set="sdss",
        output_magnitude_system="ab",
        calibration_source="APASS",
        source_magnitude_system="vega",
        source_filter_set="bessell",
        distribution_samples=200,
    )
    assert "mag_cal_g" in out.colnames or "mag_sdss_g" in out.colnames
    assert out.meta.get("ost_photometry.filter_set") == "sdss"
    assert out.meta.get("ost_photometry.magnitude_system") == "ab"


def test_sdss_to_bessell_lupton_smoke():
    convert = _load_convert_stack()
    tbl = Table()
    n = 4
    tbl["mag_cal_g"] = np.full(n, 15.0)
    tbl["err_cal_g"] = np.full(n, 0.02)
    tbl["mag_cal_r"] = np.full(n, 14.5)
    tbl["err_cal_r"] = np.full(n, 0.02)
    tbl["mag_cal_i"] = np.full(n, 14.2)
    tbl["err_cal_i"] = np.full(n, 0.02)
    out = convert.convert_magnitudes_to_other_system(
        tbl,
        output_filter_set="bessell",
        output_magnitude_system="vega",
        calibration_source="SDSS_Release_16",
        source_magnitude_system="ab",
        source_filter_set="sdss",
    )
    assert "mag_cal_B" in out.colnames
    assert "mag_cal_V" in out.colnames
    assert "mag_cal_R" in out.colnames
    assert "mag_cal_I" in out.colnames


@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_magnitude_convert_step_not_skipped_for_linear_fit():
    from ost_photometry.analyze.pipeline.config import PipelineConfig
    from ost_photometry.analyze.pipeline.context import AnalysisContext
    from ost_photometry.analyze.pipeline.steps.magnitude_convert import (
        PostProcessMagnitudeConvertStep,
    )

    cfg = PipelineConfig(
        calibration_strategy="linear_fit",
        convert_magnitudes=True,
        output_filter_set="bessell",
        output_magnitude_system="ab",
    )
    ctx = AnalysisContext(
        image_series_dict={},
        filter_list=["V"],
        output_dir="/tmp",
    )
    tbl = Table()
    tbl["mag_cal_V"] = [12.0]
    tbl["err_cal_V"] = [0.01]
    ctx.table_magnitudes = tbl
    step = PostProcessMagnitudeConvertStep()
    assert step.skip(ctx, cfg) is False


def test_target_filter_system_alias():
    ms = _ms_mod()
    assert ms.apply_target_filter_system_alias("SDSS") == ("sdss", "ab")
    assert ms.apply_target_filter_system_alias("AB") == (None, "ab")
    assert ms.apply_target_filter_system_alias("BESSELL") == ("bessell", "vega")


def test_clear_filter_does_not_expect_catalog_standards():
    ms = _ms_mod()
    assert ms.filter_expects_catalog_standards("V")
    assert ms.filter_expects_catalog_standards("g")
    assert not ms.filter_expects_catalog_standards("Clear")
    assert not ms.filter_expects_catalog_standards("C")
    assert not ms.filter_expects_catalog_standards("luminance")


def test_partition_catalog_fit_skips_clear_without_aborting():
    ms = _ms_mod()
    cat = Table({"mag_std_V": np.array([12.0, 13.0]), "mag_std_B": np.array([12.5, 13.5])})
    covered, missing = ms.partition_catalog_fit_filters(["V", "Clear"], cat)
    assert covered == ["V"]
    assert missing == ["Clear"]
    covered_none, missing_none = ms.partition_catalog_fit_filters(["Clear"], None)
    assert covered_none == []
    assert missing_none == ["Clear"]
    with pytest.raises(ValueError, match="missing columns"):
        ms.require_catalog_bands_for_filters(cat, ["V", "Clear"])
    ms.require_catalog_bands_for_filters(cat, ["V"])


def test_jordi_r_branch_per_star_and_both_branches_add_R():
    """V-R on both sides of 0.93 in one table: no crash, each star its branch."""
    convert = _load_convert_stack()
    tbl = Table()
    tbl["mag_cal_V"] = np.array([13.0, 14.0])
    tbl["err_cal_V"] = np.array([0.01, 0.01])
    tbl["mag_cal_R"] = np.array([12.5, 12.8])  # V-R = 0.5 and 1.2
    tbl["err_cal_R"] = np.array([0.01, 0.01])
    tbl["mag_cal_B"] = np.array([13.6, 15.4])
    tbl["err_cal_B"] = np.array([0.01, 0.01])
    out = convert.convert_magnitudes_to_other_system(
        tbl,
        output_filter_set="sdss",
        output_magnitude_system="ab",
        calibration_source="APASS",
        source_magnitude_system="vega",
        source_filter_set="bessell",
        distribution_samples=2000,
    )
    r_sdss = np.asarray(out["mag_sdss_r"], dtype=float)
    # Jordi et al. (2006): r - R = 0.267 (V-R) + 0.088 below, 0.77 (V-R) - 0.37 above.
    assert r_sdss[0] == pytest.approx(12.5 + 0.267 * 0.5 + 0.088, abs=0.01)
    assert r_sdss[1] == pytest.approx(12.8 + 0.77 * 1.2 - 0.37, abs=0.06)


def test_lupton_u_error_counts_g_once():
    mod = _load_convert_stack()
    transforms = __import__(
        "ost_photometry.analyze.calibration_sources.transforms", fromlist=["x"]
    )
    del mod
    u_, g_, r_ = np.array([16.0]), np.array([15.0]), np.array([14.6])
    e = np.array([0.02])
    U, eU = transforms.johnson_u_from_sloan_ugr(u_, g_, r_, e, e, e)
    b = g_ + 0.3130 * (g_ - r_) + 0.2271
    assert U[0] == pytest.approx(b[0] + 0.78 * (u_[0] - g_[0]) - 0.88)
    sig = transforms.RELATION_SIGMA
    expected = np.sqrt(
        (0.78 * 0.02) ** 2 + (0.533 * 0.02) ** 2 + (0.313 * 0.02) ** 2
        + sig["B"] ** 2 + sig["U-B"] ** 2
    )
    assert eU[0] == pytest.approx(expected)


def test_missing_input_errors_give_nan_not_zero():
    convert = _load_convert_stack()
    tbl = Table()
    tbl["mag_cal_g"] = np.array([15.0])
    tbl["mag_cal_r"] = np.array([14.5])
    out = convert.convert_magnitudes_to_other_system(
        tbl,
        output_filter_set="bessell",
        output_magnitude_system="vega",
        calibration_source="SDSS_Release_16",
        source_magnitude_system="ab",
        source_filter_set="sdss",
    )
    assert np.isfinite(out["mag_cal_V"][0])
    assert np.isnan(out["err_cal_V"][0])
