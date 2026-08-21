"""Shared pytest fixtures for ost_photometry."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from astropy.table import Table

from helpers import pkg_src

_PKG_SRC = pkg_src()
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "comparison: legacy vs differential calibration comparison tests",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that need heavy dependencies or long runtime",
    )


@pytest.fixture
def synthetic_calibration_epoch_table() -> Table:
    """Multi-band epoch table with instrumental and standard magnitudes."""
    rng = np.random.default_rng(42)
    n = 12
    mag_b_inst = 12.0 + rng.normal(0, 0.05, n)
    mag_v_inst = 11.5 + rng.normal(0, 0.05, n)
    color = mag_b_inst - mag_v_inst
    mag_b_std = mag_b_inst + 0.15 + 0.08 * color
    mag_v_std = mag_v_inst + 0.10
    tbl = Table()
    tbl["id"] = np.arange(n)
    tbl["ra"] = np.linspace(120.0, 120.01, n)
    tbl["dec"] = np.linspace(45.0, 45.01, n)
    tbl["mag_B"] = mag_b_inst
    tbl["mag_V"] = mag_v_inst
    tbl["mag_std_B"] = mag_b_std
    tbl["mag_std_V"] = mag_v_std
    tbl["err_B"] = np.full(n, 0.02)
    tbl["err_V"] = np.full(n, 0.02)
    tbl["airmass"] = np.full(n, 1.15)
    tbl["is_comparison"] = np.array([True] * 8 + [False] * 4)
    return tbl


@pytest.fixture
def comparison_tolerance():
  return {"zp_abs": 0.15, "color_term_abs": 0.10}
