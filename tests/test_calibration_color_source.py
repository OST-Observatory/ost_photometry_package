"""Colour used by the colour term when the second filter is not calibrated."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

pytest.importorskip("photutils")
pytest.importorskip("regions")

from ost_photometry.analyze.calibration import DifferentialPhotometer  # noqa: E402
from ost_photometry.analyze.calibration.result import (  # noqa: E402
    CalibrationResult,
    TransformationCoefficients,
)


def _tc(filter_name: str, t: float, zp: float) -> TransformationCoefficients:
    return TransformationCoefficients(
        filter_name=filter_name,
        color_term=t,
        color_term_err=0.01,
        zero_point=zp,
        zero_point_err=0.01,
        color_index_filters=("B", "V"),
    )


def _table() -> Table:
    # True B-V = 0.5, 0.8, 1.1; instrumental zero points differ by 2 mag.
    v_std = np.array([12.0, 12.5, 13.0])
    bv = np.array([0.5, 0.8, 1.1])
    tbl = Table()
    tbl["mag_V"] = v_std - 1.0 - 0.05 * bv
    tbl["err_V"] = np.full(3, 0.01)
    tbl["mag_B"] = v_std + bv - 3.0
    tbl["err_B"] = np.full(3, 0.01)
    tbl["mag_std_V"] = v_std
    tbl["mag_std_B"] = v_std + bv
    return tbl


def test_uncalibrated_second_filter_uses_catalog_colour_not_instrumental():
    tbl = _table()
    cal = CalibrationResult(identifier="e0", transformation={"V": _tc("V", 0.05, 1.0)})
    out = DifferentialPhotometer().apply_transform_to_table(tbl, cal, filters=["B", "V"])
    # Instrumental B-V is off by 2 mag; using it would bias V by 0.05·2 = 0.1.
    assert np.allclose(out["mag_cal_V"], tbl["mag_std_V"], atol=1e-9)
    assert set(np.asarray(out["cal_color_source_V"])) == {"catalog"}


def test_both_calibrated_uses_calibrated_colour():
    tbl = _table()
    tbl.remove_columns(["mag_std_B", "mag_std_V"])
    cal = CalibrationResult(
        identifier="e0",
        transformation={"V": _tc("V", 0.05, 1.0), "B": _tc("B", 0.0, 3.0)},
    )
    out = DifferentialPhotometer().apply_transform_to_table(tbl, cal, filters=["B", "V"])
    v_true = np.array([12.0, 12.5, 13.0])
    assert np.allclose(out["mag_cal_V"], v_true, atol=1e-6)
    assert set(np.asarray(out["cal_color_source_V"])) == {"calibrated"}


def test_star_missing_in_other_filter():
    tbl = _table()
    tbl["mag_B"][1] = np.nan
    tbl["mag_std_B"][2] = np.nan
    tbl["mag_B"][2] = np.nan
    cal = CalibrationResult(
        identifier="e0",
        transformation={"V": _tc("V", 0.05, 1.0), "B": _tc("B", 0.0, 3.0)},
    )
    out = DifferentialPhotometer().apply_transform_to_table(tbl, cal, filters=["B", "V"])
    src = np.asarray(out["cal_color_source_V"])
    # Row 1: B missing, catalog colour exists -> catalog; row 2: nothing -> NaN.
    assert list(src) == ["calibrated", "catalog", "none"]
    assert np.isfinite(out["mag_cal_V"][1])
    assert np.isnan(out["mag_cal_V"][2])


def test_zero_colour_term_needs_no_colour():
    tbl = _table()
    tbl["mag_B"][:] = np.nan
    tbl.remove_columns(["mag_std_B"])
    tc = TransformationCoefficients(
        filter_name="V",
        color_term=0.0,
        color_term_err=0.0,
        zero_point=1.0,
        zero_point_err=0.01,
        color_index_filters=("B", "V"),
    )
    cal = CalibrationResult(identifier="e0", transformation={"V": tc})
    out = DifferentialPhotometer().apply_transform_to_table(tbl, cal, filters=["V"])
    assert np.all(np.isfinite(out["mag_cal_V"]))
    assert set(np.asarray(out["cal_color_source_V"])) == {"not_needed"}
