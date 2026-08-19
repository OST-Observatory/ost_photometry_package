"""SIMBAD galaxy axes / position-angle parsing for annotated starmaps."""

from __future__ import annotations

import numpy as np
from astropy.table import MaskedColumn, Table

from helpers import load_module_from_path, pkg_src


def _simbad_galaxy_mod():
    return load_module_from_path(
        "ost_photometry.analyze.plots.simbad_galaxy",
        pkg_src() / "ost_photometry" / "analyze" / "plots" / "simbad_galaxy.py",
    )


def test_simbad_galaxy_axes_from_galdim_columns():
    mod = _simbad_galaxy_mod()
    tbl = Table(
        {
            "galdim_majaxis": [12.0],
            "galdim_minaxis": [6.0],
            "galdim_angle": [45.0],
        }
    )
    assert mod.simbad_galaxy_axes_arcmin(tbl[0]) == (12.0, 6.0, 45.0)


def test_simbad_galaxy_axes_from_dimensions_string_defaults_pa():
    mod = _simbad_galaxy_mod()
    tbl = Table({"DIMENSIONS": ["1.5 x 0.75"]})
    assert mod.simbad_galaxy_axes_arcmin(tbl[0]) == (1.5, 0.75, 0.0)


def test_simbad_galaxy_axes_masked_angle_defaults_to_zero():
    mod = _simbad_galaxy_mod()
    tbl = Table(
        {
            "galdim_majaxis": [8.0],
            "galdim_minaxis": [3.0],
            "galdim_angle": MaskedColumn([np.nan], mask=[True]),
        }
    )
    assert mod.simbad_galaxy_axes_arcmin(tbl[0]) == (8.0, 3.0, 0.0)


def test_simbad_galaxy_axes_missing_returns_none():
    mod = _simbad_galaxy_mod()
    tbl = Table({"OTYPE": ["Galaxy"]})
    assert mod.simbad_galaxy_axes_arcmin(tbl[0]) is None
