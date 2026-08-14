"""Tests for unified protected-object correlation."""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
from astropy.table import Table

from helpers import load_module_from_path, pkg_src


def _protection_module():
    return load_module_from_path(
        "ost_photometry.analyze.correlate.protection",
        pkg_src() / "ost_photometry" / "analyze" / "correlate" / "protection.py",
    )


def test_merge_protected_object_ids_explicit_always_included():
    prot = _protection_module()
    merged = prot.merge_protected_object_ids(
        protected_object_ids=[7, 3],
        reference_object_ids=[1, 7],
        calibration_object_ids=[3, 9],
        protect_ooi=True,
        protect_calibration_objects=True,
    )
    assert merged == [7, 3, 1, 9]


def test_merge_protected_object_ids_respects_protect_flags():
    prot = _protection_module()
    merged = prot.merge_protected_object_ids(
        protected_object_ids=[5],
        reference_object_ids=[1, 2],
        calibration_object_ids=[3, 4],
        protect_ooi=False,
        protect_calibration_objects=False,
    )
    assert merged == [5]


def test_merge_protected_object_ids_combines_ooi_and_calibration():
    prot = _protection_module()
    merged = prot.merge_protected_object_ids(
        reference_object_ids=[10],
        calibration_object_ids=[20, 10],
        protect_ooi=True,
        protect_calibration_objects=True,
    )
    assert merged == [10, 20]


def test_protected_object_ids_in_pipeline_config():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(
        protected_object_ids=[4, 8],
        protect_ooi=True,
        protect_calibration_objects=True,
    )
    assert cfg.protected_object_ids == [4, 8]
    assert cfg.correlation.protected_object_ids == [4, 8]


def test_protect_reference_obj_alias():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    PipelineConfig = cfg_mod.PipelineConfig

    cfg = PipelineConfig(protect_reference_obj=False)
    assert cfg.protect_ooi is False


def test_drop_rows_without_standard_mags():
    prot = _protection_module()
    tbl = Table(
        {
            "ra": [1.0, 2.0, 3.0],
            "dec": [0.0, 0.0, 0.0],
            "mag_std_V": [12.0, np.nan, 13.0],
            "err_std_V": [0.01, 0.02, np.nan],
        }
    )
    out = prot._drop_rows_without_standard_mags(tbl, ["V"])
    assert len(out) == 1
    assert float(out["ra"][0]) == 1.0


def test_resolve_calibration_object_ids_uses_standard_catalog():
    """Protect path fetches standard-schema catalogs (not legacy calibration_data)."""
    prot = _protection_module()
    std = Table(
        {
            "ra": [150.0, 150.1],
            "dec": [10.0, 10.1],
            "mag_std_V": [12.0, 13.0],
            "err_std_V": [0.01, 0.02],
        }
    )
    fetched: list[bool] = []

    def fake_fetch(*_a, **_kw):
        fetched.append(True)
        return std

    def fake_determine(_img, ra, _dec, _wcs, **_kw):
        if abs(float(ra) - 150.0) < 1e-6:
            return {1: [5]}, 1, 11.0, 22.0
        return {1: []}, 0, 0.0, 0.0

    fake_cs = types.ModuleType("ost_photometry.analyze.calibration_sources")
    fake_cs.fetch_standard_calibration_catalog = fake_fetch
    fake_inter = types.ModuleType("ost_photometry.analyze.correlate.inter")
    fake_inter.determine_object_position = fake_determine
    sys.modules["ost_photometry.analyze.calibration_sources"] = fake_cs
    sys.modules["ost_photometry.analyze.correlate.inter"] = fake_inter

    image = MagicMock()
    image.coordinates_image_center = MagicMock()
    image.field_of_view_x = 20.0
    series = MagicMock()
    series.reference_image_index = 0
    series.image_list = [image]
    series.wcs = MagicMock()

    ids, xs, ys = prot.resolve_calibration_object_ids(series, ["V"])
    assert fetched == [True]
    assert ids == [5]
    assert xs == [11.0]
    assert ys == [22.0]
