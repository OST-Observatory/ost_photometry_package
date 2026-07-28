"""Tests for duplicate removal and error propagation helpers."""

from __future__ import annotations

import numpy as np

from helpers import load_module_from_path, pkg_src


def _duplicates():
    return load_module_from_path(
        "ost_photometry.analyze.utils.duplicates",
        pkg_src() / "ost_photometry" / "analyze" / "utils" / "duplicates.py",
    )


def _errors():
    return load_module_from_path(
        "ost_photometry.analyze.utils.errors",
        pkg_src() / "ost_photometry" / "analyze" / "utils" / "errors.py",
    )


def test_clear_duplicates_keeps_best_match_per_key():
    clear_duplicates = _duplicates().clear_duplicates
    data = np.array([1, 1, 1, 2, 2], dtype=int)
    distance = np.array([0.8, 0.2, 0.5, 1.0, 0.3], dtype=float)
    partner = np.array([10, 11, 12, 20, 21], dtype=int)

    data_out, distance_out, partner_out = clear_duplicates(data, distance, partner)

    assert list(data_out) == [1, 2]
    assert list(distance_out) == [0.2, 0.3]
    assert list(partner_out) == [11, 21]


def test_clear_duplicates_handles_large_duplicate_groups():
    clear_duplicates = _duplicates().clear_duplicates
    n_sources = 5000
    data = np.zeros(n_sources, dtype=int)
    distance = np.linspace(0.0, 1.0, n_sources)
    partner = np.arange(n_sources)

    data_out, distance_out, partner_out = clear_duplicates(data, distance, partner)

    assert data_out.size == 1
    assert distance_out[0] == 0.0
    assert partner_out[0] == 0


def test_err_prop_combines_in_quadrature():
    err_prop = _errors().err_prop
    assert err_prop(3.0, 4.0) == 5.0
    out = err_prop(np.array([3.0, 5.0]), np.array([4.0, 12.0]))
    np.testing.assert_allclose(out, [5.0, 13.0])
