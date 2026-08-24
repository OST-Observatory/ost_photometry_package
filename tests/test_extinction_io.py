"""Tests for extinction coefficient IO and aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from helpers import (
    ensure_stub_package,
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
)


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _load_extinction_io():
    pkg = pkg_src()
    analyze_dir = pkg / "ost_photometry" / "analyze"
    analyze_mod = ensure_stub_package("ost_photometry.analyze", path=analyze_dir)
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        analyze_dir / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.extinction",
        analyze_dir / "extinction.py",
    )
    if getattr(analyze_mod, "__file__", None) is None:
        analyze_mod.extinction = sys.modules["ost_photometry.analyze.extinction"]
        analyze_mod.warnings_types = sys.modules["ost_photometry.analyze.warnings_types"]
    return load_module_from_path(
        "ost_photometry.analyze.extinction_io",
        analyze_dir / "extinction_io.py",
    )


def _load_linear_backend():
    analyze_dir = pkg_src() / "ost_photometry" / "analyze"
    ensure_stub_package("ost_photometry.analyze", path=analyze_dir)
    ensure_stub_package(
        "ost_photometry.analyze.calibration",
        path=analyze_dir / "calibration",
    )
    ensure_stub_package(
        "ost_photometry.analyze.calibration.backends",
        path=analyze_dir / "calibration" / "backends",
    )
    return load_module_from_path(
        "ost_photometry.analyze.calibration.backends.linear",
        analyze_dir / "calibration" / "backends" / "linear.py",
    )


@pytest.fixture
def extinction_io():
    return _load_extinction_io()


@pytest.fixture
def sample_coeff(extinction_io):
    ExtinctionCoefficients = sys.modules["ost_photometry.analyze.extinction"].ExtinctionCoefficients
    return {
        "B": ExtinctionCoefficients("B", k_prime=0.31, k_prime_err=0.03),
        "V": ExtinctionCoefficients("V", k_prime=0.19, k_prime_err=0.02),
    }


def test_round_trip_wrapped(extinction_io, sample_coeff, tmp_path):
    path = tmp_path / "ext.json"
    extinction_io.save_extinction_coefficients(
        path, sample_coeff, meta={"site": "test"}, wrapped=True
    )
    loaded = extinction_io.load_extinction_coefficients(path)
    assert loaded["B"].k_prime == pytest.approx(0.31)
    assert loaded["V"].k_prime == pytest.approx(0.19)
    with path.open() as f:
        raw = json.load(f)
    assert raw["meta"]["site"] == "test"


def test_round_trip_flat_legacy(extinction_io, sample_coeff, tmp_path):
    path = tmp_path / "flat.json"
    extinction_io.save_extinction_coefficients(path, sample_coeff, wrapped=False)
    loaded = extinction_io.load_extinction_coefficients(path)
    assert "B" in loaded


def test_resolve_custom_path(extinction_io, sample_coeff, tmp_path):
    path = tmp_path / "site.json"
    extinction_io.save_extinction_coefficients(path, sample_coeff)
    resolved = extinction_io.resolve_tabulated_extinction_coefficients(
        path, warn_on_fallback=False
    )
    assert resolved["B"].k_prime == pytest.approx(0.31)
    assert resolved["U"].k_prime == pytest.approx(0.60)


def test_resolve_missing_file_falls_back(extinction_io, tmp_path):
    resolved = extinction_io.resolve_tabulated_extinction_coefficients(
        tmp_path / "missing.json",
        warn_on_fallback=False,
    )
    assert resolved["V"].k_prime == pytest.approx(0.20)


def test_aggregate_median(extinction_io, tmp_path):
    ExtinctionCoefficients = sys.modules["ost_photometry.analyze.extinction"].ExtinctionCoefficients

    nights = []
    for i, k in enumerate([0.30, 0.32, 0.50]):
        p = tmp_path / f"night_{i}.json"
        extinction_io.save_extinction_coefficients(
            p,
            {"B": ExtinctionCoefficients("B", k_prime=k, k_prime_err=0.02)},
            meta={"night": i},
        )
        nights.append(p)

    coeffs, meta = extinction_io.aggregate_extinction_coefficients(
        nights, sigma_clip=2.5
    )
    assert coeffs["B"].k_prime == pytest.approx(0.31)
    assert meta["n_input_nights"] == 3
    assert meta["per_filter"]["B"]["n_nights"] == 2


@pytest.mark.skipif(
    importlib.util.find_spec("photutils") is None,
    reason="photutils required for build_calibrator import chain",
)
def test_build_calibrator_tabulated(tmp_path):
    extinction_io = _load_extinction_io()
    ExtinctionCoefficients = sys.modules["ost_photometry.analyze.extinction"].ExtinctionCoefficients
    linear = _load_linear_backend()
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    sample_coeff = {
        "B": ExtinctionCoefficients("B", k_prime=0.31, k_prime_err=0.03),
    }
    path = tmp_path / "site.json"
    extinction_io.save_extinction_coefficients(path, sample_coeff)
    config = cfg_mod.PipelineConfig(
        calibration_strategy="linear_fit",
        extinction_mode="tabulated",
        path_extinction_coefficients=str(path),
    )
    cal = linear.build_calibrator(config)
    assert cal.extinction.coefficients["B"].k_prime == pytest.approx(0.31)


def test_resolve_pipeline_extinction_order(extinction_io):
    ExtinctionOrder = sys.modules["ost_photometry.analyze.extinction"].ExtinctionOrder

    class _Cfg:
        extinction_mode = "none"
        extinction_order = "second"

    assert (
        extinction_io.resolve_pipeline_extinction_order(_Cfg())
        == ExtinctionOrder.NONE
    )

    class _Cfg2:
        extinction_mode = "tabulated"
        extinction_order = "second"

    assert (
        extinction_io.resolve_pipeline_extinction_order(_Cfg2())
        == ExtinctionOrder.SECOND
    )

    class _Cfg3:
        extinction_mode = "tabulated"
        extinction_order = "first"

    assert (
        extinction_io.resolve_pipeline_extinction_order(_Cfg3())
        == ExtinctionOrder.FIRST
    )


def test_apply_k_second_overrides_and_enrich(extinction_io, sample_coeff):
    ExtinctionCoefficients = sys.modules[
        "ost_photometry.analyze.extinction"
    ].ExtinctionCoefficients
    ExtinctionOrder = sys.modules["ost_photometry.analyze.extinction"].ExtinctionOrder

    # Fitted k' only
    fitted = {
        "B": ExtinctionCoefficients("B", k_prime=0.35, k_prime_err=0.02, k_second=0.0),
        "V": ExtinctionCoefficients("V", k_prime=0.18, k_prime_err=0.01, k_second=0.0),
    }
    enriched = extinction_io.enrich_second_order_from_tabulated(fitted)
    assert enriched["B"].k_prime == pytest.approx(0.35)
    assert enriched["B"].k_second != 0.0  # from site/default table

    overridden = extinction_io.apply_k_second_overrides(
        enriched, {"B": 0.055}, color_indices={"B": ("B", "V")}
    )
    assert overridden["B"].k_second == pytest.approx(0.055)
    assert overridden["B"].color_filter_1 == "B"
    assert overridden["B"].color_filter_2 == "V"

    class _Cfg:
        extinction_mode = "from_comparison_stars"
        extinction_order = "second"
        k_second = {"V": 0.009}
        path_extinction_coefficients = None
        color_indices = None

    finalized = extinction_io.finalize_pipeline_extinction_coefficients(_Cfg(), fitted)
    assert finalized["V"].k_second == pytest.approx(0.009)
    assert finalized["B"].k_second != 0.0
    assert extinction_io.resolve_pipeline_extinction_order(_Cfg()) == ExtinctionOrder.SECOND


@pytest.mark.skipif(
    importlib.util.find_spec("photutils") is None,
    reason="photutils required for build_calibrator import chain",
)
def test_build_calibrator_second_order_and_k_second_override(tmp_path):
    extinction_io = _load_extinction_io()
    ExtinctionCoefficients = sys.modules[
        "ost_photometry.analyze.extinction"
    ].ExtinctionCoefficients
    ExtinctionOrder = sys.modules["ost_photometry.analyze.extinction"].ExtinctionOrder
    linear = _load_linear_backend()
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    sample_coeff = {
        "B": ExtinctionCoefficients(
            "B",
            k_prime=0.31,
            k_prime_err=0.03,
            k_second=0.02,
            color_filter_1="B",
            color_filter_2="V",
        ),
    }
    path = tmp_path / "site.json"
    extinction_io.save_extinction_coefficients(path, sample_coeff)
    config = cfg_mod.PipelineConfig(
        calibration_strategy="linear_fit",
        extinction_mode="tabulated",
        extinction_order="second",
        path_extinction_coefficients=str(path),
        k_second={"B": 0.04},
    )
    cal = linear.build_calibrator(config)
    assert cal.extinction.order == ExtinctionOrder.SECOND
    assert cal.extinction.coefficients["B"].k_prime == pytest.approx(0.31)
    assert cal.extinction.coefficients["B"].k_second == pytest.approx(0.04)
