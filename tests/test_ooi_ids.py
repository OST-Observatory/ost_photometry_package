"""OOI photometry ID resolution (correlated id vs pre-alignment rows)."""

from __future__ import annotations

from types import SimpleNamespace

from helpers import load_module_from_path, pkg_src


def _ooi_ids():
    return load_module_from_path(
        "ost_photometry.analyze.ooi_ids",
        pkg_src() / "ost_photometry" / "analyze" / "ooi_ids.py",
    )


def test_ooi_photometry_id_prefers_correlated_id():
    mod = _ooi_ids()
    obj = SimpleNamespace(
        correlated_id=4,
        id_in_image_series={"B": 11, "V": 12},
    )
    assert mod.ooi_photometry_id(obj, filter_="B") == 4
    assert mod.ooi_photometry_id(obj) == 4


def test_ooi_photometry_id_falls_back_to_filter_row():
    mod = _ooi_ids()
    obj = SimpleNamespace(
        correlated_id=None,
        id_in_image_series={"B": 11, "V": 12},
    )
    assert mod.ooi_photometry_id(obj, filter_="V") == 12
    assert mod.ooi_photometry_id(obj, reference_image_series_id=0) == 11
    assert mod.ooi_photometry_ids([obj], filter_="B") == [11]


def test_ooi_photometry_id_skips_missing_and_none():
    mod = _ooi_ids()
    missing = SimpleNamespace(correlated_id=None, id_in_image_series={})
    none_id = SimpleNamespace(correlated_id=None, id_in_image_series={"B": None})
    ok = SimpleNamespace(correlated_id=3, id_in_image_series={})
    assert mod.ooi_photometry_id(missing) is None
    assert mod.ooi_photometry_id(none_id, filter_="B") is None
    assert mod.ooi_photometry_ids([missing, none_id, ok]) == [3]


def test_set_ooi_correlated_ids_from_filter():
    mod = _ooi_ids()
    obj = SimpleNamespace(correlated_id=None, id_in_image_series={"V": 7, "B": 2})
    mod.set_ooi_correlated_ids_from_filter([obj], "V")
    assert obj.correlated_id == 7
    lost = SimpleNamespace(correlated_id=5, id_in_image_series={"V": None})
    mod.set_ooi_correlated_ids_from_filter([lost], "V")
    assert lost.correlated_id is None
