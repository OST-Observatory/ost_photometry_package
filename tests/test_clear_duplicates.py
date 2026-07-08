"""Tests for duplicate removal in cross-matching."""

import numpy as np

from ost_photometry.analyze.utilities import clear_duplicates


def test_clear_duplicates_keeps_best_match_per_key():
    data = np.array([1, 1, 1, 2, 2], dtype=int)
    distance = np.array([0.8, 0.2, 0.5, 1.0, 0.3], dtype=float)
    partner = np.array([10, 11, 12, 20, 21], dtype=int)

    data_out, distance_out, partner_out = clear_duplicates(data, distance, partner)

    assert list(data_out) == [1, 2]
    assert list(distance_out) == [0.2, 0.3]
    assert list(partner_out) == [11, 21]


def test_clear_duplicates_handles_large_duplicate_groups():
    n_sources = 5000
    data = np.zeros(n_sources, dtype=int)
    distance = np.linspace(0.0, 1.0, n_sources)
    partner = np.arange(n_sources)

    data_out, distance_out, partner_out = clear_duplicates(data, distance, partner)

    assert data_out.size == 1
    assert distance_out[0] == 0.0
    assert partner_out[0] == 0
