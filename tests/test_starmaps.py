"""Starmap overlays, pixel WCS scatter, and correlation table helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table
from astropy.wcs import WCS

from helpers import pkg_src

_PKG_SRC = pkg_src()
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


def _plotting_stack_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        import photutils  # noqa: F401
        import regions  # noqa: F401
    except ImportError:
        return False
    return True


def _tan_wcs(nx: int = 80, ny: int = 60, scale_deg: float = 1.0 / 3600.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-scale_deg, scale_deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _run_plot_inline(target, args=(), kwargs=None):
    target(*args, **(kwargs or {}))


def test_xy_column_names_prefers_x_fit_and_accepts_xfit():
    from ost_photometry.analyze.utils.photometry import xy_column_names

    both = Table({"x": [1.0], "y": [2.0], "x_fit": [3.0], "y_fit": [4.0]})
    assert xy_column_names(both) == ("x_fit", "y_fit")
    alt = Table({"xfit": [1.0], "yfit": [2.0]})
    assert xy_column_names(alt) == ("xfit", "yfit")
    plain = Table({"x": [1.0], "y": [2.0]})
    assert xy_column_names(plain) == ("x", "y")


def test_prepare_and_plot_starmap_keeps_photometry_ids(tmp_path: Path):
    from ost_photometry.analyze.utils.starmaps import prepare_and_plot_starmap

    tbl = Table({"id": [10, 20], "x_fit": [1.0, 2.0], "y_fit": [3.0, 4.0]})
    image = SimpleNamespace(
        photometry=tbl,
        get_data=lambda: np.zeros((12, 12)),
        filter_="V",
        wcs=None,
        out_path=tmp_path,
        image_id=0,
        filename="ref.fits",
        path="ref.fits",
    )
    captured: dict = {}

    def fake_starmap(_out, _data, _filt, tbl_xy, **_kwargs):
        captured["tbl"] = tbl_xy

    with (
        patch(
            "ost_photometry.analyze.utils.starmaps.start_plot_process",
            _run_plot_inline,
        ),
        patch(
            "ost_photometry.analyze.utils.starmaps.plots.starmap",
            fake_starmap,
        ),
    ):
        prepare_and_plot_starmap(
            image,
            add_image_id=False,
            use_wcs_projection_for_star_maps=False,
        )
    assert list(captured["tbl"]["id"]) == [10, 20]


def test_calibrator_overlay_table_has_one_row_per_star(tmp_path: Path):
    from ost_photometry.analyze.utils.starmaps import (
        prepare_and_plot_starmap_from_image_series,
    )

    phot = Table({"id": [1], "x_fit": [1.0], "y_fit": [2.0]})
    img = SimpleNamespace(
        get_data=lambda: np.zeros((8, 8)),
        photometry=phot,
        filename="a.fits",
    )
    series = SimpleNamespace(
        image_list=[img],
        reference_image_index=0,
        filter_="V",
        wcs=None,
        out_path=tmp_path,
        get_image_ids=lambda: [0],
    )
    captured: dict = {}

    def fake_starmap(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    xs = [10.0, 20.0, 30.0]
    ys = [11.0, 21.0, 31.0]
    with (
        patch(
            "ost_photometry.analyze.utils.starmaps.start_plot_process",
            _run_plot_inline,
        ),
        patch(
            "ost_photometry.analyze.utils.starmaps.plots.starmap",
            fake_starmap,
        ),
    ):
        prepare_and_plot_starmap_from_image_series(series, xs, ys)
    tbl_2 = captured["kwargs"]["tbl_2"]
    assert len(tbl_2) == 3
    np.testing.assert_array_equal(tbl_2["x_centroid"], xs)


def test_prepare_and_plot_starmap_from_observation_single_filter(tmp_path: Path):
    from ost_photometry.analyze.utils.starmaps import (
        prepare_and_plot_starmap_from_observation,
    )

    image = SimpleNamespace(
        get_data=lambda: np.zeros((8, 8)),
        photometry=Table({"x_fit": [1.0], "y_fit": [2.0]}),
        wcs=None,
        out_path=tmp_path,
    )
    observation = SimpleNamespace(
        image_series_dict={"V": SimpleNamespace(reference_image=image)},
    )
    captured: dict = {}

    def fake_starmap(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    with (
        patch(
            "ost_photometry.analyze.utils.starmaps.start_plot_process",
            _run_plot_inline,
        ),
        patch(
            "ost_photometry.analyze.utils.starmaps.plots.starmap",
            fake_starmap,
        ),
    ):
        prepare_and_plot_starmap_from_observation(observation, ["V"])
    assert captured["kwargs"]["label"] == "Stars identified in V filter"


@pytest.mark.skipif(not _plotting_stack_available(), reason="matplotlib/photutils")
def test_starmap_wcs_marker_lands_on_known_pixel(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PathCollection

    from ost_photometry.analyze.plots import starmaps as sm

    nx, ny = 80, 60
    px, py = 40.0, 25.0
    image = np.zeros((ny, nx))
    image[int(py), int(px)] = 100.0
    tbl = Table({"id": [7], "x_centroid": [px], "y_centroid": [py]})
    captured: dict = {}
    real_close = sm.plt.close

    def close_and_capture(*args, **kwargs):
        ax = sm.plt.gca()
        captured["xlim"] = ax.get_xlim()
        cols = [c for c in ax.collections if isinstance(c, PathCollection)]
        captured["offsets"] = np.asarray(cols[0].get_offsets())
        captured["transform"] = cols[0].get_transform()
        captured["pixel"] = ax.get_transform("pixel")
        real_close(*args, **kwargs)

    with patch.object(sm.plt, "close", close_and_capture):
        sm.starmap(
            str(tmp_path),
            image,
            "V",
            tbl,
            wcs_image=_tan_wcs(nx, ny),
            use_wcs_projection=True,
            file_type="png",
            rts=None,
        )
    offsets = captured["offsets"]
    assert offsets.shape == (1, 2)
    assert abs(offsets[0, 0] - px) < 1.0
    assert abs(offsets[0, 1] - py) < 1.0
    assert captured["xlim"][0] == pytest.approx(-0.5)
    assert captured["xlim"][1] == pytest.approx(nx - 0.5)


@pytest.mark.skipif(not _plotting_stack_available(), reason="matplotlib/photutils")
def test_compare_images_titles_science_hips(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import ost_photometry.analyze.plots.starmaps as sm
    from ost_photometry.analyze.plots.starmaps import compare_images

    sci = np.ones((16, 16))
    ref = np.ones((16, 16)) * 2.0
    titles: list[str] = []
    real_savefig = sm.plt.savefig

    def savefig_and_capture(*args, **kwargs):
        titles.extend(ax.get_title() for ax in sm.plt.gcf().axes)
        return real_savefig(*args, **kwargs)

    with patch.object(sm.plt, "savefig", savefig_and_capture):
        compare_images(str(tmp_path), sci, ref, file_type="png")
    assert titles == ["Science", "HiPS"]


@pytest.mark.skipif(not _plotting_stack_available(), reason="matplotlib/photutils")
def test_simbad_extent_ellipse_cluster_circle_and_fov_cap():
    from astropy.coordinates import SkyCoord

    from ost_photometry.analyze.plots import starmaps as sm

    ny, nx = 100, 100
    wcs_image = _tan_wcs(nx, ny)
    center = SkyCoord(150.0, 2.0, unit="deg")

    cluster = Table(
        {
            "galdim_majaxis": [1.0],
            "galdim_minaxis": [0.4],
            "galdim_angle": [30.0],
        }
    )[0]
    ellipse = sm._simbad_extent_ellipse(cluster, center, wcs_image, (ny, nx))
    assert ellipse is not None
    assert ellipse.width != pytest.approx(ellipse.height)

    circle_row = Table({"galdim_majaxis": [0.8]})[0]
    circle = sm._simbad_extent_ellipse(circle_row, center, wcs_image, (ny, nx))
    assert circle is not None
    assert circle.width == pytest.approx(circle.height, rel=0.05)

    giant = Table({"galdim_majaxis": [30.0]})[0]
    assert sm._simbad_extent_ellipse(giant, center, wcs_image, (ny, nx)) is None


@pytest.mark.skipif(not _plotting_stack_available(), reason="matplotlib/photutils")
def test_covariance_ellipse_pixels_requires_points_and_fov():
    from ost_photometry.analyze.plots.starmaps import covariance_ellipse_pixels

    rng = np.random.default_rng(0)
    x = 40.0 + rng.normal(0.0, 2.0, size=12)
    y = 30.0 + rng.normal(0.0, 1.0, size=12)
    ell = covariance_ellipse_pixels(x, y, image_shape=(80, 80))
    assert ell is not None
    assert covariance_ellipse_pixels(x[:3], y[:3]) is None
    huge = covariance_ellipse_pixels(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 2000.0, 0.0, 2000.0, 0.0]),
        image_shape=(20, 20),
    )
    assert huge is None


@pytest.mark.skipif(not _plotting_stack_available(), reason="matplotlib/photutils")
def test_starmap_draws_covariance_ellipse_without_xy_attr(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.patches import Ellipse

    from ost_photometry.analyze.plots.starmaps import (
        covariance_ellipse_pixels,
        starmap,
    )

    rng = np.random.default_rng(1)
    x = 20.0 + rng.normal(0.0, 1.5, size=10)
    y = 15.0 + rng.normal(0.0, 1.0, size=10)
    ellipse = covariance_ellipse_pixels(x, y, image_shape=(40, 40))
    assert ellipse is not None

    tbl = Table({"id": [1], "x_centroid": [20.0], "y_centroid": [15.0]})
    tbl_2 = Table({"id": np.arange(len(x)), "x_centroid": x, "y_centroid": y})
    starmap(
        str(tmp_path),
        np.zeros((40, 40)),
        "V",
        tbl,
        tbl_2=tbl_2,
        extra_patches=[ellipse],
        use_wcs_projection=False,
        file_type="png",
        rts=None,
    )
    assert any((tmp_path / "starmaps").glob("starmap_V*.png"))
    # Ellipse constructor takes xy=center; the instance stores it as .center
    assert isinstance(ellipse, Ellipse)
