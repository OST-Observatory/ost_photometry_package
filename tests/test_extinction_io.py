"""Tests for extinction coefficient IO and aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from helpers import load_module_from_path, pkg_src


def _load_extinction_io():
    pkg = pkg_src()
    load_module_from_path(
        "ost_photometry.analyze.warnings_types",
        pkg / "ost_photometry" / "analyze" / "warnings_types.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.extinction",
        pkg / "ost_photometry" / "analyze" / "extinction.py",
    )
    analyze_stub = types.ModuleType("ost_photometry.analyze")
    analyze_stub.extinction = sys.modules["ost_photometry.analyze.extinction"]
    analyze_stub.warnings_types = sys.modules["ost_photometry.analyze.warnings_types"]
    sys.modules["ost_photometry.analyze"] = analyze_stub
    return load_module_from_path(
        "ost_photometry.analyze.extinction_io",
        pkg / "ost_photometry" / "analyze" / "extinction_io.py",
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
    linear = load_module_from_path(
        "ost_photometry.analyze.calibration.backends.linear",
        pkg_src() / "ost_photometry" / "analyze" / "calibration" / "backends" / "linear.py",
    )
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
