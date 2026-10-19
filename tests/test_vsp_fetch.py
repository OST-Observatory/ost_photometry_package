"""AAVSO VSP chart parsing: sexagesimal positions, missing bands, cuts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

PAYLOAD = {
    "photometry": [
        {"auid": "000-AAA-001", "ra": "19:44:42.85", "dec": "+54:49:42.9",
         "bands": [{"band": "V", "mag": 12.10, "error": 0.02}, {"band": "B", "mag": 12.80, "error": None}]},
        {"auid": "000-AAA-002", "ra": "19:45:00.00", "dec": "-01:30:00.0",
         "bands": [{"band": "V", "mag": 13.00, "error": 0.03}, {"band": "Ic", "mag": 12.2, "error": 0.04}]},
        {"auid": "000-AAA-003", "ra": "19:46:00.00", "dec": "+10:00:00.0",
         "bands": [{"band": "V", "mag": 8.0, "error": 0.01}]},
    ]
}


def test_vsp_chart_is_parsed(monkeypatch):
    from ost_photometry.analyze.calibration_sources import fetch

    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return SimpleNamespace(status_code=200, json=lambda: PAYLOAD)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    out = fetch._fetch_vsp_standard(SkyCoord(296.0, 54.8, unit="deg"), 20.0, ["B", "V", "I"], (10.0, 15.0), 1)
    assert '"' not in calls[0]
    assert list(out["id"]) == ["000-AAA-001", "000-AAA-002"]  # V = 8 is outside 10-15
    assert np.isclose(out["ra"][0], 296.17854, atol=1e-4) and np.isclose(out["dec"][0], 54.82858, atol=1e-4)
    assert np.isclose(out["dec"][1], -1.5)
    assert np.isnan(out["mag_std_B"][1]) and np.isnan(out["err_std_B"][0])
    assert out["mag_std_B"][0] == 12.80 and out["mag_std_I"][1] == 12.2


def test_apass_primed_ri_and_single_night_stars():
    from astropy.table import Table

    from ost_photometry.analyze.calibration_sources import fetch
    from ost_photometry.analyze.calibration_sources.transforms import (
        add_johnson_ri_to_standard_table,
        sdss_ri_from_primed_ri,
    )

    r, i = sdss_ri_from_primed_ri(np.array([14.0]), np.array([13.0]))
    assert r[0] == pytest.approx(14.0 + 0.035 * 0.79) and i[0] == pytest.approx(13.0 + 0.041 * 0.79)
    tbl = Table({"mag_std_r": [14.0], "mag_std_i": [13.0], "err_std_r": [0.02], "err_std_i": [0.02]})
    plain = add_johnson_ri_to_standard_table(tbl)
    primed = add_johnson_ri_to_standard_table(tbl, primed=True)
    assert abs(primed["mag_std_I"][0] - plain["mag_std_I"][0]) > 0.01
    assert primed["mag_std_r"][0] == 14.0
    out = Table({"ra": [1.0, 2.0, 3.0], "dec": [0.0, 0.0, 0.0], "mag_std_V": [12.0, 12.5, 13.0],
                 "err_std_V": [0.02, 0.0, 0.03]})
    q = fetch._apass_quality(out, np.array([3.0, 4.0, 1.0]), 1)
    assert len(q) == 2 and np.isnan(q["mag_std_V"][1]) and q["mag_std_V"][0] == 12.0
