"""Reddening and optional R_V / E(B-V) error terms for absolute CMDs."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_module_from_path, pkg_src


def _cmd_reddening():
    return load_module_from_path(
        "ost_photometry.analyze.plots.cmd_reddening",
        pkg_src() / "ost_photometry" / "analyze" / "plots" / "cmd_reddening.py",
    )


def test_bv_reddening_without_parameter_errors():
    mod = _cmd_reddening()
    a_v, excess, a_v_err, excess_err = mod.reddening_for_absolute_cmd(
        "B", "V", 3.1, 0.2
    )
    assert a_v == pytest.approx(0.62)
    assert excess == pytest.approx(0.2)
    assert a_v_err == 0.0
    assert excess_err == 0.0


def test_bv_reddening_propagates_independent_parameter_errors():
    mod = _cmd_reddening()
    a_v, excess, a_v_err, excess_err = mod.reddening_for_absolute_cmd(
        "B", "V", 3.1, 0.2, e_b_v_err=0.01, rv_err=0.1
    )
    assert a_v == pytest.approx(0.62)
    assert excess == pytest.approx(0.2)
    # σ(A_V)^2 = (R_V σ_E)^2 + (E σ_R)^2 ; σ(E(B-V)) = σ_E
    assert a_v_err == pytest.approx(np.hypot(3.1 * 0.01, 0.2 * 0.1))
    assert excess_err == pytest.approx(0.01)


def test_combine_cmd_error_bars_photometry_and_reddening():
    mod = _cmd_reddening()
    phot = np.array([0.03, 0.04])
    combined = mod.combine_cmd_error_bars(phot, 0.02)
    np.testing.assert_allclose(combined, np.hypot(phot, 0.02))
    np.testing.assert_allclose(mod.combine_cmd_error_bars(phot, None), phot)
    assert mod.combine_cmd_error_bars(None, None) is None
    assert float(mod.combine_cmd_error_bars(None, 0.05)) == pytest.approx(0.05)


def test_vr_reddening_uses_fitzpatrick_curve():
    mod = _cmd_reddening()
    a_r, excess, a_r_err, excess_err = mod.reddening_for_absolute_cmd(
        "V", "R", 3.1, 0.2, e_b_v_err=0.01, rv_err=0.1
    )
    assert a_r > 0.0
    assert excess > 0.0
    assert a_r_err > 0.0
    assert excess_err > 0.0
    # A_R = k_R E(B-V) is smaller than A_V = R_V E(B-V)
    assert a_r < 3.1 * 0.2


def test_cmd_correction_offsets_observation_vs_isochrone():
    mod = _cmd_reddening()
    a_v, excess = 0.62, 0.2
    m_m = 10.0
    dmag_obs, dcol_obs, dmag_iso, dcol_iso, *_ = mod.cmd_correction_offsets(
        a_v, excess, m_m, apply_to="observation"
    )
    assert dmag_obs == pytest.approx(-(a_v + m_m))
    assert dcol_obs == pytest.approx(-excess)
    assert dmag_iso == 0.0
    assert dcol_iso == 0.0

    dmag_obs, dcol_obs, dmag_iso, dcol_iso, *_ = mod.cmd_correction_offsets(
        a_v, excess, m_m, apply_to="isochrone"
    )
    assert dmag_obs == 0.0
    assert dcol_obs == 0.0
    assert dmag_iso == pytest.approx(a_v + m_m)
    assert dcol_iso == pytest.approx(excess)

    with pytest.raises(ValueError, match="apply_to"):
        mod.cmd_correction_offsets(a_v, excess, m_m, apply_to="neither")


def test_cmd_correction_offsets_propagates_m_m_err():
    mod = _cmd_reddening()
    a_v, excess = 0.62, 0.2
    m_m = 10.0
    a_v_err = 0.05
    excess_err = 0.01
    m_m_err = 0.15
    _, _, _, _, dmag_err, dcol_err = mod.cmd_correction_offsets(
        a_v, excess, m_m,
        apply_to="observation",
        a_filter_2_err=a_v_err,
        relative_extinction_err=excess_err,
        m_m_err=m_m_err,
    )
    assert dmag_err == pytest.approx(np.hypot(a_v_err, m_m_err))
    assert dcol_err == pytest.approx(excess_err)
    # without m_m_err: only a_v_err
    _, _, _, _, dmag_err_no_mm, _ = mod.cmd_correction_offsets(
        a_v, excess, m_m,
        apply_to="observation",
        a_filter_2_err=a_v_err,
        relative_extinction_err=excess_err,
    )
    assert dmag_err_no_mm == pytest.approx(a_v_err)
