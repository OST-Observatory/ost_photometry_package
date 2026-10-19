"""Broadband extinction ratios for absolute CMDs."""

from __future__ import annotations

import pytest


def test_broadband_ratios_are_consistent():
    from ost_photometry.analyze.plots.cmd_reddening import broadband_k, reddening_for_absolute_cmd

    for rv in (3.1, 2.8, 4.0):
        assert broadband_k("V", rv) == pytest.approx(rv, abs=1e-3)
        assert broadband_k("B", rv) - broadband_k("V", rv) == pytest.approx(1.0, abs=1e-3)
    # A_I/A_V and A_U/A_V for R_V = 3.1 (Cardelli et al. 1989: ~0.56-0.60, ~1.53-1.57)
    assert broadband_k("I", 3.1) / 3.1 == pytest.approx(0.57, abs=0.03)
    assert broadband_k("U", 3.1) / 3.1 == pytest.approx(1.56, abs=0.04)
    a_r, e_vr, _, _ = reddening_for_absolute_cmd("V", "R", 3.1, 0.3)
    assert a_r == pytest.approx(0.3 * broadband_k("R", 3.1))
    assert e_vr == pytest.approx(0.3 * (3.1 - broadband_k("R", 3.1)), rel=1e-3)
