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
