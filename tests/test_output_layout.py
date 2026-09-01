"""Canonical analysis output layout directories."""

from __future__ import annotations

import pytest

from helpers import load_module_from_path, pkg_src


def _layout():
    src = pkg_src()
    return load_module_from_path(
        "ost_photometry.output_layout",
        src / "ost_photometry" / "output_layout.py",
    )


def test_diagnostics_and_results_and_work_dirs(tmp_path):
    layout = _layout()
    cal = layout.diagnostics_dir(tmp_path, "calibration")
    assert cal == tmp_path / "diagnostics" / "calibration"
    assert cal.is_dir()

    cmds = layout.diagnostics_dir(tmp_path, "cmds")
    assert cmds == tmp_path / "diagnostics" / "cmds"
    assert cmds.is_dir()

    lc = layout.results_dir(tmp_path, "lightcurves", "by_id")
    assert lc == tmp_path / "results" / "lightcurves" / "by_id"
    assert lc.is_dir()

    wcs = layout.work_dir(tmp_path, "wcs_images")
    assert wcs == tmp_path / "work" / "wcs_images"
    assert wcs.is_dir()

    tables = layout.tables_dir(tmp_path)
    assert tables == tmp_path / "tables"


def test_extraction_plot_dir_splits_gallery(tmp_path):
    layout = _layout()
    ref = layout.extraction_plot_dir(tmp_path, gallery=False)
    extra = layout.extraction_plot_dir(tmp_path, gallery=True)
    assert ref == tmp_path / "diagnostics" / "extraction"
    assert extra == tmp_path / "work" / "extraction"


def test_unknown_step_raises():
    layout = _layout()
    with pytest.raises(ValueError, match="Unknown diagnostic step"):
        layout.diagnostics_dir(".", "not_a_step")
