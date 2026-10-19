"""Header airmass, pointing, and airmass computed for frames without AIRMASS."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from ost_photometry.analyze.extinction import airmass_for_image, fill_missing_airmass
from ost_photometry.image import header_airmass, header_jd_start_mid, header_pointing

OST = EarthLocation(lat=52.409184 * u.deg, lon=12.973185 * u.deg, height=39 * u.m)


def test_header_airmass_missing_or_invalid_is_none():
    assert header_airmass(fits.Header({"AIRMASS": 1.23})) == pytest.approx(1.23)
    assert header_airmass(fits.Header()) is None
    assert header_airmass(fits.Header({"AIRMASS": "n/a"})) is None
    assert header_airmass(fits.Header({"AIRMASS": 0.5})) is None


def test_header_pointing_qhy_and_nina_styles():
    qhy = header_pointing(fits.Header({"OBJCTRA": "07 27 39.95", "OBJCTDEC": "+24 20 11.5"}))
    nina = header_pointing(fits.Header({"RA": 111.9165, "DEC": 24.3365}))
    assert qhy.ra.deg == pytest.approx(111.9165, abs=1e-3)
    assert nina.dec.deg == pytest.approx(24.3365)
    assert header_pointing(fits.Header()) is None


class _Image:
    def __init__(self, header, jd, air_mass=None):
        self._header = header
        self.jd = jd
        self.air_mass = air_mass
        self.wcs = None

    def get_header(self):
        return self._header


def _expected(coord, jd_mid):
    alt = coord.transform_to(
        AltAz(obstime=Time(jd_mid, format="jd"), location=OST)
    ).alt.deg
    return 1.0 / np.sin(np.radians(alt + 244 / (165 + 47 * alt**1.1)))


def test_airmass_computed_at_mid_exposure_from_pointing():
    coord = SkyCoord(111.9165 * u.deg, 24.3365 * u.deg)
    jd = Time("2022-01-06T22:59:13", scale="utc").jd
    header = fits.Header({"RA": 111.9165, "DEC": 24.3365, "EXPTIME": 600.0, "JD": jd})
    # Image.jd is the mid-exposure time; the airmass uses it as it is
    _start, mid = header_jd_start_mid(header)
    image = _Image(header, mid)
    value = airmass_for_image(image, OST)
    assert value == pytest.approx(_expected(coord, jd + 300 / 86400), rel=1e-6)
    # Without a site or a pointing nothing can be computed.
    assert airmass_for_image(image, None) is None
    assert airmass_for_image(_Image(fits.Header({"EXPTIME": 20}), jd), OST) is None


def test_fill_missing_airmass_keeps_header_values():
    jd = Time("2022-01-06T22:59:13", scale="utc").jd
    header = fits.Header({"RA": 111.9165, "DEC": 24.3365, "EXPTIME": 20.0})
    have = _Image(header, jd, air_mass=1.5)
    need = _Image(header, jd)
    lost = _Image(fits.Header({"EXPTIME": 20.0}), jd)
    computed, missing = fill_missing_airmass([have, need, lost], OST)
    assert (computed, missing) == (1, 1)
    assert have.air_mass == 1.5
    assert 1.0 < need.air_mass < 3.0
    assert lost.air_mass is None


def test_header_jd_start_mid():
    jd = 2460000.25
    start, mid = header_jd_start_mid(fits.Header({"JD": jd, "EXPTIME": 120.0}))
    assert start == jd and mid == pytest.approx(jd + 60 / 86400, abs=1e-9)
    # stacks: the weighted mean mid-time wins over JD + EXPTIME/2
    stack = fits.Header({"JD": jd, "EXPTIME": 1200.0, "MJD_MID": 60000.0})
    assert header_jd_start_mid(stack)[1] == pytest.approx(2460000.5)
    start, mid = header_jd_start_mid(fits.Header({"DATE-OBS": "2023-02-24T18:00:00", "EXPTIME": 10.0}))
    assert mid - start == pytest.approx(5 / 86400, abs=1e-9)
    with pytest.raises(ValueError):
        header_jd_start_mid(fits.Header({"EXPTIME": 10.0}))


def test_second_order_uses_standard_colour_of_uncorrected_mags():
    """b, v with different zero points: m0 - m_true is the same for every
    colour and airmass (colour on the standard scale, taken before the
    loop overwrites mag_B)."""
    import numpy as np
    from astropy.table import Table

    from ost_photometry.analyze.extinction import (
        ExtinctionCoefficients,
        ExtinctionCorrector,
        ExtinctionOrder,
    )

    rng = np.random.default_rng(1)
    n = 30
    bv = rng.uniform(0.0, 1.6, n)
    v_true = rng.uniform(11, 14, n)
    b_true = v_true + bv
    rows = []
    k = {"B": (0.30, -0.03), "V": (0.18, 0.0)}
    for x in (1.1, 1.6, 2.2):
        b = b_true + 24.1 + k["B"][0] * x + k["B"][1] * x * bv
        v = v_true + 23.4 + k["V"][0] * x
        rows.append(Table({"mag_B": b, "mag_V": v, "mag_std_B": b_true, "mag_std_V": v_true,
                           "airmass": np.full(n, x)}))
    corrector = ExtinctionCorrector(order=ExtinctionOrder.SECOND)
    corrector.coefficients = {
        f: ExtinctionCoefficients(f, k_prime=kp, k_second=ks, color_filter_1="B", color_filter_2="V")
        for f, (kp, ks) in k.items()
    }
    for tbl in rows:
        out = corrector.correct(tbl, mag_col_prefix="mag_", output_prefix="mag_", filters=["B", "V"],
                                catalog_color_prefix="mag_std_")
        np.testing.assert_allclose(out["mag_B"] - b_true, 24.1, atol=1e-6)
        np.testing.assert_allclose(out["mag_V"] - v_true, 23.4, atol=1e-9)
        assert np.all(out["ext_second_order_B"])
