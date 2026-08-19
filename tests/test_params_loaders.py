"""Tests for JSON/YAML parameter loaders."""

from __future__ import annotations

import sys
import types

from helpers import load_module_from_path, pkg_src


def _utilities():
    pkg = pkg_src()
    # utilities.py imports image/wcs at module level; stub them for this unit test.
    for name in (
        "ost_photometry.checks",
        "ost_photometry.terminal_output",
        "ost_photometry.image",
        "ost_photometry.wcs",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    wcs_mod = sys.modules["ost_photometry.wcs"]
    for attr in (
        "check_wcs_exists",
        "find_wcs_astap",
        "find_wcs_astrometry",
        "find_wcs_twirl",
        "persist_wcs_to_fits",
        "sync_image_coordinates_from_wcs",
    ):
        setattr(wcs_mod, attr, lambda *a, **k: None)
    sys.modules["ost_photometry.image"].Image = object
    return load_module_from_path(
        "ost_photometry.utilities_params",
        pkg / "ost_photometry" / "utilities.py",
    )


def test_read_params_from_json_mapping(tmp_path):
    utils = _utilities()
    path = tmp_path / "ok.json"
    path.write_text('{"a": 1, "b": "x"}\n', encoding="utf-8")
    assert utils.read_params_from_json(str(path)) == {"a": 1, "b": "x"}


def test_read_params_from_json_rejects_non_mapping(tmp_path):
    utils = _utilities()
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    assert utils.read_params_from_json(str(path)) == {}


def test_read_params_from_json_missing_or_invalid(tmp_path):
    utils = _utilities()
    assert utils.read_params_from_json(str(tmp_path / "missing.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert utils.read_params_from_json(str(bad)) == {}


def test_read_params_from_yaml_mapping(tmp_path):
    utils = _utilities()
    path = tmp_path / "ok.yaml"
    path.write_text("a: 1\nb: x\n", encoding="utf-8")
    assert utils.read_params_from_yaml(str(path)) == {"a": 1, "b": "x"}


def test_read_params_from_yaml_rejects_non_mapping(tmp_path):
    utils = _utilities()
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    assert utils.read_params_from_yaml(str(path)) == {}


def test_read_params_from_yaml_empty_or_missing(tmp_path):
    utils = _utilities()
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert utils.read_params_from_yaml(str(empty)) == {}
    assert utils.read_params_from_yaml(str(tmp_path / "missing.yaml")) == {}
