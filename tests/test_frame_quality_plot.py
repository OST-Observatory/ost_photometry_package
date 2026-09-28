"""Smoke test for the frame-quality overview plot."""

from __future__ import annotations

import numpy as np
import pytest


def test_frame_quality_overview_writes_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("ccdproc")
    import matplotlib

    matplotlib.use("Agg")
    from ost_photometry.reduce.plots import frame_quality_overview

    n = 6
    rows = {
        "file": np.array([f"img_{i:03d}.fit" for i in range(n)]),
        "fwhm_px": np.array([3.0, 3.2, np.nan, 2.8, 5.5, 3.1]),
        "fwhm_arcsec": np.full(n, np.nan),
        "roundness": np.array([0.1, 0.12, np.nan, 0.08, 0.4, 0.1]),
        "n_stars": np.array([50, 48, 0, 55, 20, 49]),
        "background": np.array([10.0, 10.5, 11.0, 9.8, 15.0, 10.2]),
        "rejected": np.array([False, False, True, False, True, False]),
        "is_reference": np.array([False, False, False, True, False, False]),
        "aligned": np.array([True, True, False, True, False, False]),
    }
    path = frame_quality_overview(
        rows, tmp_path, "V", fwhm_max=4.0, fwhm_unit="px", alignment_known=True
    )
    assert path == tmp_path / "diagnostics" / "frame_quality" / "frame_quality_V.pdf"
    assert path.is_file()
    assert path.stat().st_size > 1000

    #   arcsec requested but unavailable -> falls back to pixels, still writes.
    path2 = frame_quality_overview(rows, tmp_path, "r''", fwhm_max=2.0, fwhm_unit="arcsec")
    assert path2.name == "frame_quality_rp.pdf"
    assert path2.is_file()


def test_plot_frame_quality_wrapper(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("ccdproc")
    import matplotlib

    matplotlib.use("Agg")
    from ost_photometry.reduce.frame_selection import quality_table_from_rows
    from ost_photometry.reduce.quality import plot_frame_quality

    rows = []
    for filt in ("B", "V"):
        for i in range(3):
            rows.append(
                {"file": f"{filt}_{i}.fit", "filter": filt, "jd": 2460000.0 + i,
                 "fwhm_px": 3.0 + i, "n_stars": 40, "roundness": 0.1,
                 "background": 10.0, "aligned": True}
            )
    table = quality_table_from_rows(rows)
    paths = plot_frame_quality(
        table, tmp_path, selection={"fwhm_max": 4.5}, blocking=True
    )
    assert len(paths) == 2
    assert all(p.is_file() for p in paths)
    assert plot_frame_quality(table[:0], tmp_path) == []
