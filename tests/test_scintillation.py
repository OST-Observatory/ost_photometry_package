"""Telescope catalog and the scintillation term of photometric errors."""

from __future__ import annotations

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable, Table

from ost_photometry.analyze.scintillation import (
    add_scintillation_to_photometry,
    apply_scintillation_to_image_series,
    scintillation_fraction,
)
from ost_photometry.telescope_specs import (
    aperture_diameter_m,
    normalize_telescope_id,
    resolve_telescope,
    telescope_from_focal_length,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Planewave CDK20", "CDK20"),
        ("OST CDK20", "CDK20"),
        ("PlaneWave CDK 20", "CDK20"),
        ("Celestron C14", "C14"),
        ("Celestron RASA 11", "RASA11"),
        ("C8", "C8"),
        ("Coronado SolarMax II 60", "SolarMaxII60"),
        ("Meade LX200", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_telescope_id(name, expected):
    assert normalize_telescope_id(name) == expected


def test_focal_length_fallback_and_aperture():
    assert telescope_from_focal_length(3454.0) == "CDK20"
    assert telescope_from_focal_length(3470.0) == "CDK20"  # within 2 %
    assert telescope_from_focal_length(620.0) == "RASA11"
    assert telescope_from_focal_length(1500.0) is None
    assert telescope_from_focal_length(None) is None
    assert resolve_telescope("unknown scope", 3454) == "CDK20"
    assert aperture_diameter_m("CDK20") == pytest.approx(0.508)
    assert aperture_diameter_m("C11") == pytest.approx(0.2794)
    assert aperture_diameter_m(None) is None


def test_scintillation_matches_osborn_eq7_and_scales():
    # CDK20 at the OST, 20 s, X = 1.15: about 0.002 (HAT-P-20 test night).
    frac = scintillation_fraction(0.508, 20.0, 1.15, 39.0)
    expected = math.sqrt(
        1e-5 * 1.5**2 * 0.508 ** (-4 / 3) / 20.0 * 1.15**3 * math.exp(-2 * 39 / 8000)
    )
    assert frac == pytest.approx(expected)
    assert 0.0019 < frac < 0.0022
    # sigma ~ D^-2/3, t^-1/2, X^3/2, linear in C_Y.
    assert scintillation_fraction(0.254, 20, 1.15, 39) / frac == pytest.approx(2 ** (2 / 3))
    assert scintillation_fraction(0.508, 80, 1.15, 39) / frac == pytest.approx(0.5)
    assert scintillation_fraction(0.508, 20, 2.3, 39) / frac == pytest.approx(2**1.5)
    assert scintillation_fraction(0.508, 20, 1.15, 39, coefficient=1.0) / frac == pytest.approx(
        1 / 1.5
    )
    # Missing airmass counts as zenith; no aperture or exposure: NaN.
    assert scintillation_fraction(0.508, 20, float("nan")) == scintillation_fraction(0.508, 20, 1.0)
    assert math.isnan(scintillation_fraction(0.0, 20, 1.2))
    assert math.isnan(scintillation_fraction(0.508, float("nan"), 1.2))


def _phot() -> Table:
    flux = np.array([1.0e6, 1.0e4])
    err = np.array([1.0e3, 1.0e2])  # 0.1 % and 1 %
    return Table(
        {
            "flux_fit": flux,
            "flux_err": err,
            "mags_unc": 2.5 / np.log(10) * err / flux,
        }
    )


def test_scintillation_dominates_bright_stars_only():
    tbl = add_scintillation_to_photometry(_phot(), 0.002)
    rel = np.asarray(tbl["flux_err"]) / np.asarray(tbl["flux_fit"])
    assert rel[0] == pytest.approx(math.hypot(0.001, 0.002))
    assert rel[1] == pytest.approx(math.hypot(0.01, 0.002))
    assert np.allclose(tbl["flux_err_phot"], [1.0e3, 1.0e2])
    assert np.allclose(tbl["mags_unc"], 2.5 / np.log(10) * rel)
    assert tbl.meta["scintillation_fraction"] == 0.002


def test_scintillation_is_not_added_twice():
    once = add_scintillation_to_photometry(_phot(), 0.002)
    twice = add_scintillation_to_photometry(once.copy(), 0.002)
    assert np.allclose(once["flux_err"], twice["flux_err"])
    assert np.allclose(once["mags_unc"], twice["mags_unc"])


def test_scintillation_keeps_quantity_units():
    tbl = QTable(
        {
            "flux_fit": [1.0e5, 2.0e5] * u.electron / u.s,
            "flux_err": [100.0, 150.0] * u.electron / u.s,
            "mags_unc": [0.001, 0.0008] * u.mag,
        }
    )
    out = add_scintillation_to_photometry(tbl, 0.003)
    assert out["flux_err"].unit == u.electron / u.s
    assert out["mags_unc"].unit == u.mag
    assert out["flux_err"][0].value == pytest.approx(math.hypot(100.0, 300.0))


class _FakeImage:
    def __init__(self, telescope, focal, exptime, airmass):
        self._header = {"TELESCOP": telescope, "FOCALLEN": focal, "EXPTIME": exptime}
        self.air_mass = airmass
        self.photometry = _phot()

    def get_header(self):
        return self._header


class _FakeSeries:
    def __init__(self, images):
        self.image_list = images
        self.filter_ = "V"


def test_series_uses_catalog_aperture_exposure_and_airmass():
    images = [_FakeImage("Planewave CDK20", 3454, 20.0, x) for x in (1.1, 1.5)]
    summary = apply_scintillation_to_image_series(_FakeSeries(images), altitude_m=39.0)
    assert summary["telescope"] == "CDK20"
    assert summary["n_images"] == 2
    for im, x in zip(images, (1.1, 1.5), strict=True):
        frac = scintillation_fraction(0.508, 20.0, x, 39.0)
        assert im.photometry.meta["scintillation_fraction"] == pytest.approx(frac)


def test_series_unknown_telescope_is_left_alone():
    images = [_FakeImage("Home-built Newton", 1000, 20.0, 1.2)]
    assert apply_scintillation_to_image_series(_FakeSeries(images)) is None
    assert "flux_err_phot" not in images[0].photometry.colnames
    # An explicit aperture still works.
    summary = apply_scintillation_to_image_series(_FakeSeries(images), aperture_m=0.2)
    assert summary["aperture_m"] == 0.2
    assert "flux_err_phot" in images[0].photometry.colnames
