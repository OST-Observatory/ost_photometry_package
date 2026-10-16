"""Cluster membership with per-star astrometric errors (extreme deconvolution)."""

from __future__ import annotations

import numpy as np
import pytest

from ost_photometry.analyze.utils.cluster_membership import (  # noqa: E402
    membership_from_astrometry,
)


def _field(rng):
    n_cl, n_fd = 300, 900
    # cluster: tight intrinsic distribution; half of the members are faint
    cl_err = np.where(rng.random(n_cl) < 0.5, 0.05, 0.6)
    cl = np.column_stack([
        rng.normal(-1.0, 0.05, n_cl), rng.normal(-2.0, 0.05, n_cl), rng.normal(0.55, 0.02, n_cl),
    ])
    fd = np.column_stack([
        rng.normal(-2.0, 3.0, n_fd), rng.normal(-3.0, 3.0, n_fd), rng.normal(0.8, 0.5, n_fd),
    ])
    fd_err = rng.uniform(0.05, 0.6, n_fd)
    err = np.concatenate([cl_err, fd_err])
    true = np.concatenate([cl, fd])
    cov = np.zeros((true.shape[0], 3, 3))
    for i in range(3):
        cov[:, i, i] = err**2 * (0.5 if i == 2 else 1.0)
    obs = true + rng.normal(0, 1, true.shape) * np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    is_member = np.r_[np.ones(n_cl, bool), np.zeros(n_fd, bool)]
    return obs, cov, is_member, np.r_[cl_err, fd_err]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_xd_membership_is_complete_clean_and_calibrated(seed):
    """Faint members found, little contamination, Σ P_mem ≈ true member count.

    A plain GMM on the same data either absorbs half of the field (seed 0) or
    loses almost all faint members (seeds 1, 2): its cluster component has
    to describe intrinsic width and errors at once.
    """
    rng = np.random.default_rng(seed)
    obs, cov, member, err = _field(rng)
    kw = dict(simbad_pm_ra=-1.0, simbad_pm_de=-2.0, simbad_plx=0.55)
    xd = membership_from_astrometry(obs[:, 0], obs[:, 1], obs[:, 2], cov=cov, **kw)
    plain = membership_from_astrometry(obs[:, 0], obs[:, 1], obs[:, 2], **kw)
    assert xd.method == "xd" and plain.method == "gmm"
    faint = member & (err > 0.5)
    sel = xd.p_mem >= 0.5
    assert np.mean(sel[faint]) > 0.85
    assert np.mean(sel[~member]) < 0.06
    assert abs(xd.p_mem.sum() - member.sum()) < 0.1 * member.sum()
    plain_sel = plain.p_mem >= 0.5
    plain_ok = np.mean(plain_sel[faint]) > 0.85 and np.mean(plain_sel[~member]) < 0.06
    assert not plain_ok


def test_xd_requires_covariance():
    rng = np.random.default_rng(1)
    obs, _cov, _m, _e = _field(rng)
    with pytest.raises(ValueError, match="cov"):
        membership_from_astrometry(obs[:, 0], obs[:, 1], obs[:, 2], method="xd")
