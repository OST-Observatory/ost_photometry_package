"""Tests for shared FWHM estimation helpers."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from ost_photometry.fwhm import (
    estimate_fwhm_from_finder_table,
    estimate_fwhm_from_positions,
    estimate_image_fwhm,
    filter_finder_table_by_fwhm_scale,
    filter_table_finite_cutouts,
    finite_cutout_star_mask,
    roundness_range_for_finder,
    select_sources_for_fwhm_fit,
    source_positions_from_table,
)


def _source_table(n_sources: int) -> Table:
    flux = np.linspace(100.0, 100.0 + n_sources - 1, n_sources)
    return Table(
        {
            "x_centroid": np.arange(n_sources, dtype=float) + 30.0,
            "y_centroid": np.arange(n_sources, dtype=float) + 30.0,
            "flux": flux,
        }
    )


def test_select_sources_for_fwhm_fit_uses_flux_window():
    table = _source_table(50)
    selected = select_sources_for_fwhm_fit(table, data_shape=(200, 200), n_select=20)
    assert 5 <= len(selected) <= 20
    # Should prefer brighter end of the 70–97% window, not the very brightest tip
    assert float(np.max(selected["flux"])) <= float(np.percentile(table["flux"], 99))


def test_select_sources_for_fwhm_fit_keeps_small_tables():
    table = _source_table(12)
    selected = select_sources_for_fwhm_fit(table, data_shape=(200, 200), min_objects=40)
    assert len(selected) <= 12
    assert len(selected) >= 1


def test_select_sources_excludes_edge_stars():
    table = Table(
        {
            "x_centroid": [5.0, 100.0, 100.0],
            "y_centroid": [5.0, 100.0, 100.0],
            "flux": [200.0, 150.0, 140.0],
        }
    )
    selected = select_sources_for_fwhm_fit(
        table, data_shape=(200, 200), fit_shape=25, min_objects=1, n_select=3
    )
    assert len(selected) >= 1
    assert float(np.min(selected["x_centroid"])) > 10.0


def test_source_positions_from_table_supports_xy_columns():
    table = Table({"x": [1.5, 2.5], "y": [3.5, 4.5], "flux": [1.0, 2.0]})
    assert source_positions_from_table(table) == [(1.5, 3.5), (2.5, 4.5)]


def test_estimate_fwhm_from_positions_returns_default_for_empty_sources():
    data = np.zeros((50, 50))
    fwhm, error = estimate_fwhm_from_positions(data, [], default_fwhm=4.0)
    assert fwhm == 4.0
    assert error == "no sources available for FWHM fit"


def test_finder_fwhm_column_used_when_in_range():
    table = Table(
        {
            "x_centroid": np.linspace(20, 80, 20),
            "y_centroid": np.linspace(20, 80, 20),
            "flux": np.linspace(100, 200, 20),
            "fwhm": np.full(20, 4.5),
            "sharpness": np.full(20, 0.5),
            "roundness1": np.full(20, 0.0),
        }
    )
    fwhm, err, meta = estimate_fwhm_from_finder_table(
        table, default_fwhm=4.0, data_shape=(120, 120)
    )
    assert err is None
    assert fwhm == pytest.approx(4.5, abs=0.1)
    assert meta["source"] == "finder_column"


def test_finder_fwhm_ignores_junk_before_quality_filter():
    """Noise-like FWHM values on bad stars must not dominate the aggregate."""
    n = 30
    table = Table(
        {
            "x_centroid": np.linspace(20, 100, n),
            "y_centroid": np.linspace(20, 100, n),
            "flux": np.linspace(50, 200, n),
            # Most "detections" look like cosmics (tiny FWHM + bad sharpness)
            "fwhm": np.concatenate([np.full(20, 1.2), np.full(10, 6.0)]),
            "sharpness": np.concatenate([np.full(20, 0.05), np.full(10, 0.5)]),
            "roundness1": np.zeros(n),
        }
    )
    fwhm, err, meta = estimate_fwhm_from_finder_table(
        table, default_fwhm=4.0, data_shape=(140, 140), min_valid=5
    )
    assert err is None
    assert meta["source"] == "finder_column"
    assert fwhm == pytest.approx(6.0, abs=0.3)


def test_finder_fwhm_uses_iraf_roundness_column():
    """IRAF catalogs have ``roundness``, not DAO ``roundness1``."""
    n = 40
    table = Table(
        {
            "x_centroid": np.linspace(20, 100, n),
            "y_centroid": np.linspace(20, 100, n),
            "flux": np.linspace(50, 200, n),
            "fwhm": np.concatenate([np.full(25, 1.3), np.full(15, 6.2)]),
            "sharpness": np.concatenate([np.full(25, 0.05), np.full(15, 0.5)]),
            "roundness": np.concatenate([np.full(25, 0.9), np.full(15, 0.1)]),
        }
    )
    fwhm, err, meta = estimate_fwhm_from_finder_table(
        table, default_fwhm=4.0, data_shape=(140, 140), min_valid=5
    )
    assert err is None
    assert meta["source"] == "finder_column"
    assert fwhm == pytest.approx(6.2, abs=0.3)


def test_finder_fwhm_near_minimum_is_rejected():
    """A median sitting on fwhm_estimate_min is compact noise, not seeing."""
    n = 30
    table = Table(
        {
            "x_centroid": np.linspace(20, 100, n),
            "y_centroid": np.linspace(20, 100, n),
            "flux": np.linspace(100, 200, n),
            "fwhm": np.full(n, 2.51),
            "sharpness": np.full(n, 0.5),
            "roundness": np.full(n, 0.1),
        }
    )
    fwhm, err, meta = estimate_fwhm_from_finder_table(
        table, default_fwhm=4.0, data_shape=(140, 140), min_fwhm=2.0
    )
    assert fwhm == 4.0
    assert err is not None
    assert "lower limit" in err


def test_per_star_outliers_discarded_before_aggregate():
    """A few huge fits must not force fallback if enough stars are in range."""
    pytest.importorskip("photutils")
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 0.05, (120, 120))
    true_fwhm = 3.5
    sigma = true_fwhm / 2.355
    y_grid, x_grid = np.mgrid[0:120, 0:120]
    positions = [(25.0, 25.0), (60.0, 30.0), (40.0, 80.0), (90.0, 70.0), (75.0, 95.0)]
    for x0, y0 in positions:
        data += 120.0 * np.exp(
            -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2.0 * sigma**2)
        )
    # Add a bright blended junk peak that can inflate a bad fit
    data += 500.0 * np.exp(-((x_grid - 50.0) ** 2 + (y_grid - 50.0) ** 2) / (2.0 * 8.0**2))

    fwhm, error = estimate_fwhm_from_positions(
        data,
        positions,
        default_fwhm=4.0,
        fit_shape=25,
        min_fwhm=2.0,
        max_fwhm=9.0,
        min_valid=3,
    )
    assert error is None
    assert fwhm == pytest.approx(true_fwhm, rel=0.35)


@pytest.mark.parametrize(
    "test_name",
    ["synthetic_gaussian", "out_of_range"],
)
def test_estimate_fwhm_from_positions_with_photutils(test_name: str):
    pytest.importorskip("photutils")

    if test_name == "synthetic_gaussian":
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 0.05, (80, 80))
        true_fwhm = 3.2
        sigma = true_fwhm / 2.355
        y_grid, x_grid = np.mgrid[0:80, 0:80]
        positions = [(20.0, 20.0), (55.0, 25.0), (30.0, 60.0)]

        for x0, y0 in positions:
            data += 100.0 * np.exp(
                -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2.0 * sigma**2)
            )

        fwhm, error = estimate_fwhm_from_positions(
            data,
            positions,
            default_fwhm=4.0,
            fit_shape=25,
            min_valid=2,
        )

        assert error is None
        assert fwhm == pytest.approx(true_fwhm, rel=0.25)
        return

    data = np.zeros((40, 40))
    data[20, 20] = 1000.0
    fwhm, error = estimate_fwhm_from_positions(
        data,
        [(20.0, 20.0)],
        default_fwhm=4.0,
        min_fwhm=2.0,
        max_fwhm=9.0,
        min_valid=1,
    )
    assert fwhm == 4.0
    assert error is not None


def test_estimate_image_fwhm_prefers_finder_column():
    data = np.zeros((100, 100))
    table = Table(
        {
            "x_centroid": np.linspace(20, 80, 15),
            "y_centroid": np.linspace(20, 80, 15),
            "flux": np.linspace(50, 150, 15),
            "fwhm": np.concatenate([np.full(12, 5.0), np.full(3, 25.0)]),
            "sharpness": np.full(15, 0.5),
            "roundness1": np.zeros(15),
        }
    )
    fwhm, err, meta = estimate_image_fwhm(data, table, default_fwhm=4.0, min_valid=5)
    assert err is None
    assert meta["source"] == "finder_column"
    assert fwhm == pytest.approx(5.0, abs=0.2)


def test_finite_cutout_star_mask_rejects_nan_and_masked_stamps():
    data = np.ones((40, 40), dtype=float)
    data[18:23, 18:23] = np.nan
    extra = np.zeros((40, 40), dtype=bool)
    extra[8:12, 8:12] = True
    x = np.array([10.0, 20.0, 30.0])
    y = np.array([10.0, 20.0, 30.0])
    ok = finite_cutout_star_mask(data, x, y, size=5, extra_mask=extra)
    np.testing.assert_array_equal(ok, [False, False, True])


def test_filter_table_finite_cutouts_drops_bad_stars():
    data = np.ones((40, 40), dtype=float)
    data[18, 20] = np.inf
    tbl = Table(
        {
            "x": np.array([10.0, 20.0, np.nan]),
            "y": np.array([10.0, 20.0, 30.0]),
        }
    )
    filtered, n_rejected = filter_table_finite_cutouts(tbl, data, size=5)
    assert n_rejected == 2
    assert len(filtered) == 1
    assert filtered["x"][0] == 10.0


def test_filter_table_finite_cutouts_passthrough_when_clean():
    data = np.ones((30, 30), dtype=float)
    tbl = Table({"x_centroid": [15.0], "y_centroid": [15.0]})
    filtered, n_rejected = filter_table_finite_cutouts(tbl, data, size=7)
    assert n_rejected == 0
    assert len(filtered) == 1
    np.testing.assert_array_equal(filtered["x_centroid"], [15.0])


def test_roundness_range_for_finder_maps_dao_window_on_iraf():
    assert roundness_range_for_finder("IRAF", (-1.0, 1.0)) == (0.0, 0.5)
    assert roundness_range_for_finder("IRAF", (0.0, 0.2)) == (0.0, 0.2)
    assert roundness_range_for_finder("DAO", (-1.0, 1.0)) == (-1.0, 1.0)


def test_filter_finder_table_by_fwhm_scale_drops_noise_peaks():
    table = Table(
        {
            "x_centroid": [10.0, 20.0, 30.0],
            "y_centroid": [10.0, 20.0, 30.0],
            "fwhm": [1.2, 6.1, 18.0],
        }
    )
    kept, n_removed = filter_finder_table_by_fwhm_scale(
        table, 6.0, scale_range=(0.5, 2.0)
    )
    assert n_removed == 2
    assert len(kept) == 1
    assert float(kept["fwhm"][0]) == pytest.approx(6.1)


def test_filter_finder_table_by_fwhm_scale_noop_without_column():
    table = Table({"x_centroid": [10.0], "y_centroid": [10.0]})
    kept, n_removed = filter_finder_table_by_fwhm_scale(table, 6.0)
    assert n_removed == 0
    assert len(kept) == 1
