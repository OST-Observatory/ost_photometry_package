"""Sparse correlation apply helpers, flux stacking, and auto-reference."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.table import Table

from helpers import isolated_sys_modules, load_module_from_path, pkg_src


def _tracks_module():
    src = pkg_src()
    load_module_from_path(
        "ost_photometry.style",
        src / "ost_photometry" / "style.py",
    )
    load_module_from_path(
        "ost_photometry.terminal_output",
        src / "ost_photometry" / "terminal_output.py",
    )
    return load_module_from_path(
        "ost_photometry.analyze.correlate.tracks",
        src / "ost_photometry" / "analyze" / "correlate" / "tracks.py",
    )


def test_effective_miss_limit_grows_with_series_length():
    with isolated_sys_modules():
        tracks = _tracks_module()
        assert tracks.effective_miss_limit(10, 5, None) == 5
        assert tracks.effective_miss_limit(80, 5, 0.3) == max(5, int(0.7 * 80))


def test_apply_sparse_track_ids_sets_id_and_drops_unmatched():
    with isolated_sys_modules():
        tracks = _tracks_module()
        phot = Table(
            {
                "x_fit": [1.0, 2.0, 3.0],
                "flux_fit": [10.0, 20.0, 30.0],
            }
        )
        out = tracks.apply_sparse_track_ids_to_table(phot, np.array([1, -1, 0]))
        # Track 0 → row 1, track 2 → row 0; row 2 unmatched.
        assert list(out["id"]) == [2, 0]
        assert list(np.asarray(out["x_fit"])) == [1.0, 2.0]


def test_sparse_apply_then_bind_ooi_uses_track_id_not_table_row():
    """Identify writes the remaining-table row; LC/calibration join on track id."""
    with isolated_sys_modules():
        tracks = _tracks_module()
        ooi = load_module_from_path(
            "ost_photometry.analyze.ooi_ids",
            pkg_src() / "ost_photometry" / "analyze" / "ooi_ids.py",
        )
        phot = Table(
            {
                "x_fit": [1.0, 2.0, 3.0],
                "flux_fit": [10.0, 20.0, 30.0],
            }
        )
        # Track 0 → native row 1, track 2 → native row 0; native row 2 dropped.
        out = tracks.apply_sparse_track_ids_to_table(phot, np.array([1, -1, 0]))
        obj = SimpleNamespace(correlated_id=None, id_in_image_series={"V": 0})
        ooi.bind_ooi_ids_from_photometry([obj], "V", out)
        assert list(out["id"]) == [2, 0]
        assert obj.correlated_id == 2
        assert ooi.ooi_photometry_id(obj, filter_="V") == 2


def test_flux_arrays_from_photometry_tables_pad_on_id():
    with isolated_sys_modules():
        tracks = _tracks_module()
        t0 = Table(
            {
                "id": [0, 1],
                "flux_fit": [10.0, 20.0],
                "flux_err": [0.1, 0.2],
            }
        )
        t1 = Table(
            {
                "id": [1],
                "flux_fit": [21.0],
                "flux_err": [0.3],
            }
        )
        flux, err = tracks.flux_arrays_from_photometry_tables([t0, t1])
        assert flux.shape == (2, 2)
        assert np.isnan(flux[1, 0])
        assert flux[1, 1] == 21.0
        assert err[0, 0] == 0.1


def test_pick_auto_reference_image_prefers_most_detections():
    with isolated_sys_modules():
        tracks = _tracks_module()
        img0 = SimpleNamespace(
            photometry=Table({"flux_fit": [1.0, np.nan]}),
            fwhm=3.0,
        )
        img1 = SimpleNamespace(
            photometry=Table({"flux_fit": [1.0, 2.0, 3.0]}),
            fwhm=4.0,
        )
        series = SimpleNamespace(image_list=[img0, img1])
        assert tracks.pick_auto_reference_image(series) == 1


def test_resolved_series_reference_index_uses_series_not_auto_config():
    with isolated_sys_modules():
        tracks = _tracks_module()
        series = SimpleNamespace(image_list=[0, 1, 2], reference_image_index=2)
        config = SimpleNamespace(reference_image_index="auto")
        assert tracks.resolved_series_reference_index(series, config) == 2
        series_auto = SimpleNamespace(image_list=[0, 1], reference_image_index="auto")
        assert tracks.resolved_series_reference_index(series_auto, config) == 0
        assert tracks.coerce_reference_image_index("auto", 4) == 0
