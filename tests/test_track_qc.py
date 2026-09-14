"""Track QC statistics and figure after intra-filter correlation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from helpers import (
    isolated_sys_modules,
    load_module_from_path,
    pkg_src,
    stub_analyze_package,
)


def _qc_module():
    return load_module_from_path(
        "ost_photometry.analyze.correlate.qc",
        pkg_src() / "ost_photometry" / "analyze" / "correlate" / "qc.py",
    )


def _plot_module():
    stub_analyze_package("plots")
    root = pkg_src() / "ost_photometry"
    load_module_from_path("ost_photometry.output_layout", root / "output_layout.py")
    return load_module_from_path(
        "ost_photometry.analyze.plots.correlation_qc",
        root / "analyze" / "plots" / "correlation_qc.py",
    )


def _wcs(dx: float = 0.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [50.0 + dx, 50.0]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    return w


def _series(*, drift_px: float = 0.0, mix_last: bool = False, per_frame_wcs: bool = True):
    """Five frames, four tracks; optional per-frame drift and a mixed track."""
    rng = np.random.default_rng(1)
    x0 = np.array([20.0, 40.0, 60.0, 80.0])
    y0 = np.array([25.0, 45.0, 65.0, 85.0])
    mags = np.array([12.0, 13.0, 14.0, 15.0])
    images = []
    for i in range(5):
        shift = drift_px * i
        x = x0 + shift + rng.normal(0, 0.05, 4)
        y = y0 + rng.normal(0, 0.05, 4)
        ids = np.arange(4)
        if i == 4 and mix_last:
            x[3] += 12.0  # track 3 jumps to a different star on the last frame
        if i == 2:
            # track 1 missing on frame 2
            keep = ids != 1
            x, y, ids, m = x[keep], y[keep], ids[keep], mags[keep]
        else:
            m = mags
        phot = Table(
            {
                "id": ids,
                "x_fit": x,
                "y_fit": y,
                "mags_fit": m + rng.normal(0, 0.01, ids.size),
            }
        )
        # per-frame WCS compensates the drift so sky positions stay fixed
        w = _wcs(dx=shift) if per_frame_wcs else _wcs()
        images.append(SimpleNamespace(photometry=phot, wcs=w, jd=2460000.0 + i * 0.01))
    return SimpleNamespace(image_list=images, reference_image_index=0)


def test_collect_track_qc_registered_series():
    with isolated_sys_modules():
        qc_mod = _qc_module()
        qc = qc_mod.collect_track_qc(
            _series(),
            filter_="V",
            coordinate_frame="pixel",
            pixel_radius=3.0,
            separation_limit_arcsec=2.0,
            min_detection_fraction=0.3,
            ooi_ids=[2],
        )
        assert qc["n_frames"] == 5 and qc["n_tracks"] == 4
        assert list(qc["frame_n_matched"]) == [4, 4, 3, 4, 4]
        assert list(qc["track_n_det"]) == [5, 4, 5, 5]
        assert np.all(np.abs(qc["frame_dx"]) < 0.5)
        assert np.all(qc["track_max_px"] < 1.0)
        assert not np.any(qc["track_suspect"])
        tbl = qc_mod.track_qc_table(qc)
        assert list(tbl["is_ooi"]) == [False, False, True, False]
        assert "0 suspect" in qc_mod.track_qc_summary(qc)


def test_collect_track_qc_flags_mixed_track_and_drift():
    with isolated_sys_modules():
        qc_mod = _qc_module()
        qc = qc_mod.collect_track_qc(
            _series(drift_px=2.0, mix_last=True),
            filter_="V",
            coordinate_frame="sky",
            pixel_radius=3.0,
            separation_limit_arcsec=2.0,
        )
        # frame drift of 2 px per frame shows up in the per-frame shift
        assert np.allclose(qc["frame_dx"][1:], [2.0, 4.0, 6.0, 8.0], atol=0.3)
        assert qc["worst_frame_index"] == 4
        # sky positions are stable except for the mixed track (12 px = 12")
        assert qc["track_max_arcsec"][3] > 10.0
        assert np.all(qc["track_max_arcsec"][:3] < 1.0)
        assert list(qc["track_suspect"]) == [False, False, False, True]
        assert "1 suspect" in qc_mod.track_qc_summary(qc)


def test_plot_track_qc_writes_figure(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    with isolated_sys_modules():
        qc_mod = _qc_module()
        plot_mod = _plot_module()
        qc = qc_mod.collect_track_qc(
            _series(drift_px=1.0, mix_last=True),
            filter_="V",
            coordinate_frame="sky",
            pixel_radius=3.0,
            separation_limit_arcsec=2.0,
            min_detection_fraction=0.3,
            ooi_ids=[2],
        )
        path = plot_mod.plot_track_qc(qc, tmp_path, "png")
        assert path is not None and path.exists()
        assert path.name == "track_qc_V.png"
        assert path.parent == tmp_path / "diagnostics" / "correlation"
