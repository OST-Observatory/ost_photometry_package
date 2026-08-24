"""Characterization tests for calibration and extinction math cores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from helpers import isolated_sys_modules, load_module_from_path, pkg_src  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _deps_available() -> bool:
    try:
        import photutils  # noqa: F401
        import regions  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_extinction_calculate_airmass():
    import astropy.units as u

    extinction_mod = load_module_from_path(
        "ost_photometry.analyze.extinction",
        pkg_src() / "ost_photometry" / "analyze" / "extinction.py",
    )
    loc = EarthLocation(lat=52.4 * u.deg, lon=13.0 * u.deg)
    coord = SkyCoord(ra=180 * u.deg, dec=45 * u.deg)
    t = Time("2024-01-01T00:00:00")
    am = extinction_mod.calculate_airmass(coord, t, loc)
    assert 1.0 < float(am) < 3.0


@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_extinction_corrector_first_order():
    extinction_mod = load_module_from_path(
        "ost_photometry.analyze.extinction",
        pkg_src() / "ost_photometry" / "analyze" / "extinction.py",
    )
    from astropy.table import Table

    ExtinctionCorrector = extinction_mod.ExtinctionCorrector
    ExtinctionCoefficients = extinction_mod.ExtinctionCoefficients
    ExtinctionOrder = extinction_mod.ExtinctionOrder

    tbl = Table()
    tbl["mag_V"] = np.array([12.0, 13.0])
    tbl["airmass"] = np.array([1.0, 2.0])
    corr = ExtinctionCorrector(
        { "V": ExtinctionCoefficients("V", k_prime=0.15) },
        order=ExtinctionOrder.FIRST,
    )
    out = corr.correct(tbl, mag_col_prefix="mag_", filters=["V"], inplace=False)
    assert float(out["mag_V"][0]) == pytest.approx(12.0)
    assert float(out["mag_V"][1]) == pytest.approx(13.0)
    assert float(out["mag_ext_V"][0]) == pytest.approx(12.0 - 0.15 * 1.0)
    assert float(out["mag_ext_V"][1]) == pytest.approx(13.0 - 0.15 * 2.0)


@pytest.mark.skipif(not _deps_available(), reason="requires photutils and regions")
def test_differential_photometer_zp_only(synthetic_calibration_epoch_table):
    diff_mod = __import__(
        "ost_photometry.analyze.differential_photometry",
        fromlist=["DifferentialPhotometer"],
    )
    DifferentialPhotometer = diff_mod.DifferentialPhotometer

    tbl = synthetic_calibration_epoch_table
    mask = np.asarray(tbl["is_comparison"], dtype=bool)
    phot = DifferentialPhotometer()
    result = phot.fit_transformation_epoch(
        tbl,
        epoch_id="epoch_000",
        filters=["V"],
        comparison_mask=mask,
        determine_color_terms=False,
    )
    tc = result.transformation["V"]
    assert tc.n_stars_used >= 3
    assert np.isfinite(tc.zero_point)
