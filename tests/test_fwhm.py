"""Tests for shared FWHM estimation helpers."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from ost_photometry.fwhm import (
    estimate_fwhm_from_positions,
    select_sources_for_fwhm_fit,
    source_positions_from_table,
)


def _source_table(n_sources: int) -> Table:
    flux = np.linspace(100.0, 100.0 + n_sources - 1, n_sources)
    return Table(
        {
            "x_centroid": np.arange(n_sources, dtype=float) + 10.0,
            "y_centroid": np.arange(n_sources, dtype=float) + 10.0,
            "flux": flux,
        }
    )


def test_select_sources_for_fwhm_fit_uses_middle_slice_for_large_tables():
    table = _source_table(50)
    selected = select_sources_for_fwhm_fit(table)
    assert len(selected) == 20
    assert selected["flux"][0] == pytest.approx(120.0)
    assert selected["flux"][-1] == pytest.approx(139.0)


def test_select_sources_for_fwhm_fit_keeps_small_tables():
    table = _source_table(12)
    selected = select_sources_for_fwhm_fit(table)
    assert len(selected) == 12
    assert selected["flux"][0] == pytest.approx(100.0)


def test_source_positions_from_table_supports_xy_columns():
    table = Table({"x": [1.5, 2.5], "y": [3.5, 4.5], "flux": [1.0, 2.0]})
    assert source_positions_from_table(table) == [(1.5, 3.5), (2.5, 4.5)]


def test_estimate_fwhm_from_positions_returns_default_for_empty_sources():
    data = np.zeros((50, 50))
    fwhm, error = estimate_fwhm_from_positions(data, [], default_fwhm=4.0)
    assert fwhm == 4.0
    assert error == "no sources available for FWHM fit"


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
    )
    assert fwhm == 4.0
    assert error is not None
    assert "outside" in error
