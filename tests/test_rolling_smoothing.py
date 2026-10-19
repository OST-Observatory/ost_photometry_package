"""Rolling smoothing of per-epoch transformations keeps stars of typical colour."""

from __future__ import annotations

import pytest

from ost_photometry.analyze.calibration.photometer import _apply_rolling_smooth_to_per_image_results
from ost_photometry.analyze.calibration.result import CalibrationResult, TransformationCoefficients


def _results(c_ref=0.7, zp_at_c=25.0):
    out = {}
    for eid, t in (("e1", 0.08), ("e2", 0.12), ("e3", 0.10)):
        r = CalibrationResult(identifier=eid)
        r.transformation["V"] = TransformationCoefficients(
            filter_name="V", color_term=t, color_term_err=0.01, zero_point=zp_at_c - t * c_ref,
            zero_point_err=0.005, color_ref=c_ref,
        )
        out[eid] = r
    return out


@pytest.mark.parametrize(("t_mode", "zp_mode"), [("median", "none"), ("median", "median"), ("mean", "mean")])
def test_star_at_reference_colour_is_unchanged(t_mode, zp_mode):
    res = _results()
    _apply_rolling_smooth_to_per_image_results(
        ["e1", "e2", "e3"], res, ["V"], window=3, color_term_mode=t_mode, zero_point_mode=zp_mode
    )
    for r in res.values():
        tc = r.transformation["V"]
        assert tc.zero_point + tc.color_term * 0.7 == pytest.approx(25.0, abs=1e-9)
    assert res["e2"].transformation["V"].color_term == pytest.approx(0.10 if t_mode == "median" else 0.10)
