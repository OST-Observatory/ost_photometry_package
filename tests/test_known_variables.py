"""VSX / known-variable exclusion from the calibration catalog."""

from __future__ import annotations

import sys
import types

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from helpers import (
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
    stub_analyze_package,
)


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _load_known_variables(monkeypatch):
    src = pkg_src()
    stub_analyze_package("calibration_sources")
    load_module_from_path("ost_photometry.style", src / "ost_photometry" / "style.py")
    load_module_from_path(
        "ost_photometry.terminal_output",
        src / "ost_photometry" / "terminal_output.py",
    )

    vizier_mod = types.ModuleType(
        "ost_photometry.analyze.calibration_sources.vizier_query"
    )
    vizier_mod.get_vizier_catalog = lambda *a, **k: (Table(), {}, "")
    sys.modules["ost_photometry.analyze.calibration_sources.vizier_query"] = vizier_mod

    return load_module_from_path(
        "ost_photometry.analyze.calibration_sources.known_variables",
        src
        / "ost_photometry"
        / "analyze"
        / "calibration_sources"
        / "known_variables.py",
    )


def _catalog() -> Table:
    return Table(
        {
            "ra": np.array([10.0, 10.01, 11.0]),
            "dec": np.array([20.0, 20.0, 21.0]),
            "mag_std_V": np.array([12.0, 13.0, 14.0]),
        }
    )


def _center() -> SkyCoord:
    return SkyCoord(10.0 * u.deg, 20.0 * u.deg)


def test_xmatch_drops_catalog_row(monkeypatch):
    mod = _load_known_variables(monkeypatch)

    def _fake_xm(positions, radius, catalog_identifier="B/vsx/vsx"):
        return Table({"ost_cat_row": np.array([0], dtype=np.int64)})

    monkeypatch.setattr(mod, "query_vsx_xmatch", _fake_xm)
    out = mod.drop_catalog_rows_near_known_variables(
        _catalog(), _center(), 15.0, radius=1.0 * u.arcsec
    )
    assert len(out) == 2
    assert 10.0 not in np.asarray(out["ra"], dtype=float)
    assert out.meta[mod.KNOWN_VARIABLES_EXCLUDED_META] is True


def test_xmatch_empty_keeps_all(monkeypatch):
    mod = _load_known_variables(monkeypatch)
    monkeypatch.setattr(mod, "query_vsx_xmatch", lambda *a, **k: Table())
    out = mod.drop_catalog_rows_near_known_variables(_catalog(), _center(), 15.0)
    assert len(out) == 3
    assert out.meta[mod.KNOWN_VARIABLES_EXCLUDED_META] is True


def test_cone_fallback_when_xmatch_fails(monkeypatch):
    mod = _load_known_variables(monkeypatch)
    vsx = Table({"RAJ2000": [10.0], "DEJ2000": [20.0]})

    def _boom(*_a, **_k):
        raise ConnectionError("xmatch down")

    def _fake_get(*_a, **_k):
        return vsx, {"ra": "RAJ2000", "dec": "DEJ2000"}, u.deg

    monkeypatch.setattr(mod, "query_vsx_xmatch", _boom)
    monkeypatch.setattr(mod, "get_vizier_catalog", _fake_get)
    out = mod.drop_catalog_rows_near_known_variables(
        _catalog(), _center(), 15.0, radius=1.0 * u.arcsec
    )
    assert len(out) == 2
    assert 10.0 not in np.asarray(out["ra"], dtype=float)


def test_query_failure_keeps_catalog(monkeypatch):
    mod = _load_known_variables(monkeypatch)

    def _boom(*_a, **_k):
        raise ConnectionError("vizier down")

    monkeypatch.setattr(mod, "query_vsx_xmatch", _boom)
    monkeypatch.setattr(mod, "get_vizier_catalog", _boom)
    cat = _catalog()
    out = mod.drop_catalog_rows_near_known_variables(cat, _center(), 15.0)
    assert len(out) == 3
    assert not out.meta.get(mod.KNOWN_VARIABLES_EXCLUDED_META)
