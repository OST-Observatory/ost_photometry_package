"""PSF photometry must not flatten the pixel-error map before the fit."""

from __future__ import annotations

import numpy as np
import pytest

from ost_photometry.analyze.extraction import _sanitize_pixel_error


def test_sanitize_keeps_finite_positive_pixels():
    err = np.array(
        [[1.0, np.nan], [-2.0, 3.0], [np.inf, 4.0]],
        dtype=float,
    )
    out = _sanitize_pixel_error(err)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(3.0)
    assert out[2, 1] == pytest.approx(4.0)
    fill = 4.0
    assert out[0, 1] == pytest.approx(fill)
    assert out[1, 0] == pytest.approx(fill)
    assert out[2, 0] == pytest.approx(fill)


def test_sanitize_does_not_set_all_pixels_to_the_maximum():
    """Regression: inverted NaN mask used to assign max(σ) to every finite pixel."""
    rng = np.random.default_rng(0)
    err = rng.uniform(0.5, 2.0, size=(32, 32))
    err[0, 0] = 80.0
    err[1, 1] = np.nan
    out = _sanitize_pixel_error(err)
    typical = np.isfinite(err) & (err > 0.0) & (err < 10.0)
    assert np.allclose(out[typical], err[typical])
    assert out[0, 0] == pytest.approx(80.0)
    assert out[1, 1] == pytest.approx(80.0)
    assert np.median(out) < 10.0


def test_sanitize_copies_input():
    err = np.array([[1.0, np.nan], [2.0, 3.0]])
    out = _sanitize_pixel_error(err)
    assert not np.shares_memory(out, err)
    assert np.isnan(err[0, 1])
