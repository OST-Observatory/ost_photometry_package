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


def test_bind_ooi_ids_from_photometry_uses_track_id_not_row():
    from astropy.table import Table

    mod = _ooi_ids()
    phot = Table({"id": [7, 3, 11], "x_fit": [0.0, 1.0, 2.0]})
    obj = SimpleNamespace(correlated_id=None, id_in_image_series={"V": 1})
    mod.bind_ooi_ids_from_photometry([obj], "V", phot)
    assert obj.id_in_image_series["V"] == 3
    assert obj.correlated_id == 3
    assert mod.ooi_photometry_id(obj, filter_="V") == 3


def test_bind_ooi_ids_from_photometry_dense_id_equals_row():
    from astropy.table import Table

    mod = _ooi_ids()
    phot = Table({"id": [0, 1, 2]})
    obj = SimpleNamespace(correlated_id=None, id_in_image_series={"B": 2})
    mod.bind_ooi_ids_from_photometry([obj], "B", phot)
    assert obj.correlated_id == 2


def test_bind_ooi_ids_can_leave_correlated_id_unset():
    from astropy.table import Table

    mod = _ooi_ids()
    phot = Table({"id": [9, 8]})
    obj = SimpleNamespace(correlated_id=None, id_in_image_series={"B": 0, "V": 1})
    mod.bind_ooi_ids_from_photometry([obj], "B", phot, set_correlated_id=False)
    assert obj.id_in_image_series["B"] == 9
    assert obj.correlated_id is None
    assert mod.ooi_photometry_id(obj, filter_="B") == 9
