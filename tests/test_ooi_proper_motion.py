"""Objects of interest follow their proper motion to the observation epoch."""

from __future__ import annotations

import astropy.units as u
import pytest


def test_move_to_epoch():
    from ost_photometry.analyze.models import ObjectOfInterest

    ooi = ObjectOfInterest(269.452, 4.6933, u.deg, u.deg, "Barnard's Star")
    ooi.pm_ra_cosdec, ooi.pm_dec = -801.551, 10362.394  # mas/yr
    shift = ooi.move_to_epoch(2451545.0 + 25 * 365.25)
    assert shift == pytest.approx(25 * 10.393, rel=1e-3)  # arcsec
    assert ooi.dec == pytest.approx(4.6933 + 25 * 10.362394 / 3600, abs=1e-6)
    assert ooi.coordinates_object.dec.deg == pytest.approx(ooi.dec)
    still = ObjectOfInterest(10.0, 20.0, u.deg, u.deg, "no pm")
    assert still.move_to_epoch(2460000.0) == 0.0 and still.ra == 10.0
