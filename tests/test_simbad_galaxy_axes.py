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
    assert mod.simbad_angular_axes_arcmin(tbl[0]) == (12.0, 6.0, 45.0)


def test_simbad_angular_axes_major_only_is_circle():
    mod = _simbad_galaxy_mod()
    tbl = Table({"galdim_majaxis": [8.0], "galdim_angle": [20.0]})
    assert mod.simbad_angular_axes_arcmin(tbl[0]) == (8.0, 8.0, 20.0)


def test_simbad_angular_axes_from_single_diameter_string():
    mod = _simbad_galaxy_mod()
    tbl = Table({"DIMENSIONS": ["4.0"]})
    assert mod.simbad_angular_axes_arcmin(tbl[0]) == (4.0, 4.0, 0.0)


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


def test_skycoord_from_simbad_tap_degrees_not_hourangle():
    """TAP basic.ra is degrees; hourangle would shift NGC 7789 by ~9 deg."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    mod = _simbad_galaxy_mod()
    coord = mod.skycoord_from_simbad(359.35, 56.73)
    assert abs(coord.ra.degree - 359.35) < 1e-6
    assert abs(coord.dec.degree - 56.73) < 1e-6
    misparsed = SkyCoord(ra=359.35, dec=56.73, unit=("hourangle", "deg"))
    assert abs(coord.ra.degree - misparsed.ra.degree) > 8.0
    assert coord.separation(misparsed).to_value(u.deg) > 4.0


def test_skycoord_from_simbad_sexagesimal_hourangle():
    mod = _simbad_galaxy_mod()
    coord = mod.skycoord_from_simbad("23 57 24", "+56 42 30")
    assert abs(coord.ra.degree - 359.35) < 0.01
    assert abs(coord.dec.degree - 56.7083) < 0.01


def test_simbad_overlay_kind_tap_codes():
    mod = _simbad_galaxy_mod()
    assert mod.simbad_overlay_kind("*") == "star"
    assert mod.simbad_overlay_kind("Star") == "star"
    assert mod.simbad_overlay_kind("OpC") == "cluster"
    assert mod.simbad_overlay_kind("Cl*") == "cluster"
    assert mod.simbad_overlay_kind("G") == "galaxy"
    assert mod.simbad_overlay_kind("Galaxy") == "galaxy"
    assert mod.simbad_overlay_kind("HII") == "nebula"
    assert mod.simbad_overlay_kind("EmO") == "other"
