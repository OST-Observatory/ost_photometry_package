"""Tests for legacy magnitude table helpers."""

from __future__ import annotations

import sys

import pytest
from astropy.table import Table

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _legacy_magnitudes():
    root = pkg_src() / "ost_photometry"
    analyze = root / "analyze"
    ensure_stub_package("ost_photometry", path=root)
    ensure_stub_package("ost_photometry.calibration_parameters")
    ensure_stub_package("ost_photometry.terminal_output")
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package(
        "ost_photometry.analyze.post_processing",
        path=analyze / "post_processing",
    )

    term = sys.modules["ost_photometry.terminal_output"]
    if getattr(term, "__file__", None) is None:
        term.print_to_terminal = lambda *a, **k: None
    calib = sys.modules["ost_photometry.calibration_parameters"]
    if getattr(calib, "__file__", None) is None:
        calib.valid_filter_combinations_for_transformation = [["B", "V"], ["V", "R"]]

    return load_module_from_path(
        "ost_photometry.analyze.utils.legacy_magnitudes",
        pkg_src() / "ost_photometry" / "analyze" / "utils" / "legacy_magnitudes.py",
    )


def test_transformation_keys_for_table_magnitudes():
    mod = _legacy_magnitudes()
    tbl = Table(
        {
            "id": [1],
            "B (transformed, image=0)": [12.0],
            "V (transformed, image=0)": [11.5],
        }
    )
    keys = mod.transformation_keys_for_table_magnitudes(tbl, ["B", "V"])
    assert keys["magB"].startswith("B")
    assert keys["magV"].startswith("V")


def test_find_filter_for_magnitude_transformation():
    mod = _legacy_magnitudes()
    valid, combos = mod.find_filter_for_magnitude_transformation(
        ["B", "V"],
        {"magB": "ok", "magV": "ok"},
    )
    assert valid == {"B", "V"}
    assert ["B", "V"] in combos
