"""Tests for legacy magnitude table helpers."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from helpers import load_module_from_path, pkg_src


def _legacy_magnitudes():
    import sys
    import types

    for name in (
        "ost_photometry",
        "ost_photometry.calibration_parameters",
        "ost_photometry.checks",
        "ost_photometry.terminal_output",
        "ost_photometry.analyze",
        "ost_photometry.analyze.post_processing",
        "ost_photometry.analyze.post_processing.adapters",
        "ost_photometry.analyze.post_processing.io",
        "ost_photometry.analyze.post_processing.light_curve",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    sys.modules["ost_photometry.terminal_output"].print_to_terminal = (
        lambda *a, **k: None
    )
    sys.modules["ost_photometry.checks"].check_output_directories = lambda *a, **k: None
    sys.modules[
        "ost_photometry.calibration_parameters"
    ].valid_filter_combinations_for_transformation = [["B", "V"], ["V", "R"]]
    adapters = sys.modules["ost_photometry.analyze.post_processing.adapters"]
    adapters.ensure_epoch_native_photometry_table = lambda t: t
    io = sys.modules["ost_photometry.analyze.post_processing.io"]
    io.write_epoch_native_magnitudes = lambda *a, **k: None
    lc = sys.modules["ost_photometry.analyze.post_processing.light_curve"]
    lc.attach_observation_jd_column = lambda t, *a, **k: t

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


def test_calibrated_epochs_to_legacy_wide_table():
    mod = _legacy_magnitudes()
    rows = []
    for epoch, mag_b, mag_v in (("epoch_0", 12.0, 11.0), ("epoch_1", 12.1, 11.1)):
        rows.append(
            {
                "id": 0,
                "x": 10.0,
                "y": 20.0,
                "ra": 1.0,
                "dec": 2.0,
                "epoch_id": epoch,
                "mag_cal_B": mag_b,
                "err_cal_B": 0.01,
                "mag_cal_V": mag_v,
                "err_cal_V": 0.02,
            }
        )
    calibrated = Table(rows=rows)
    wide = mod.calibrated_epochs_to_legacy_wide_table(calibrated, ["B", "V"])
    assert len(wide) == 1
    assert "B (transformed, image=0)" in wide.colnames
    assert "V (transformed, image=1)" in wide.colnames
    assert wide["B (transformed, image=0)"][0] == 12.0
    assert wide["V (transformed, image=1)"][0] == 11.1


def test_find_filter_for_magnitude_transformation():
    mod = _legacy_magnitudes()
    valid, combos = mod.find_filter_for_magnitude_transformation(
        ["B", "V"],
        {"magB": "ok", "magV": "ok"},
    )
    assert valid == {"B", "V"}
    assert ["B", "V"] in combos
