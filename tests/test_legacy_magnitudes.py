"""Tests for legacy magnitude table helpers."""

from __future__ import annotations

import sys
import types

import pytest
from astropy.table import Table

from helpers import load_module_from_path, pkg_src


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    before = sys.modules.copy()
    yield
    sys.modules.clear()
    sys.modules.update(before)


def _legacy_magnitudes():
    for name in (
        "ost_photometry",
        "ost_photometry.calibration_parameters",
        "ost_photometry.terminal_output",
        "ost_photometry.analyze",
        "ost_photometry.analyze.post_processing",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    sys.modules["ost_photometry.terminal_output"].print_to_terminal = (
        lambda *a, **k: None
    )
    sys.modules[
        "ost_photometry.calibration_parameters"
    ].valid_filter_combinations_for_transformation = [["B", "V"], ["V", "R"]]

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
