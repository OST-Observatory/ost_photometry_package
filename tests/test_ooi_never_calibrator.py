"""Objects of interest never calibrate: by photometry id and by position."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ost_photometry.analyze.calibration_sources.flags import (
    ids_near_positions,
    strip_catalog_standards_for_object_ids,
)
from ost_photometry.analyze.ooi_ids import ooi_coordinates

OOI = SkyCoord(111.9165, 24.3365, unit="deg")


def _epoch() -> Table:
    dec = OOI.dec.deg
    cosd = np.cos(np.radians(dec))
    # rows 0.0", 3.0" and 8.0" east of the OOI, and one far away
    offsets = np.array([0.0, 3.0, 8.0, 300.0]) / 3600.0 / cosd
    return Table({
        "id": [10, 11, 12, 13],
        "ra": OOI.ra.deg + offsets,
        "dec": np.full(4, dec),
        "mag_std_V": [11.3, 13.8, 14.0, 12.5],
        "err_std_V": [0.03] * 4,
    })


def test_detections_at_an_ooi_position_are_found():
    tbl = _epoch()
    assert ids_near_positions(tbl, OOI, 5.0) == {10, 11}
    assert ids_near_positions(tbl, SkyCoord([OOI.ra.deg], [OOI.dec.deg], unit="deg"), 1.0) == {10}
    assert ids_near_positions(tbl, None, 5.0) == set()
    assert ids_near_positions(tbl, OOI, 0.0) == set()


def test_unidentified_ooi_is_cleared_from_the_calibrator_pool():
    """The OOI has no photometry id (correlation missed it): the position
    still keeps its catalog match out of the fit."""
    tbl = _epoch()
    obj = SimpleNamespace(coordinates_object=OOI, correlated_id=None, id_in_image_series={})
    exclude = set() | ids_near_positions(tbl, ooi_coordinates([obj]), 5.0)
    assert strip_catalog_standards_for_object_ids(tbl, exclude) == 2
    assert np.isnan(tbl["mag_std_V"][0]) and np.isnan(tbl["mag_std_V"][1])
    assert tbl["mag_std_V"][2] == 14.0 and tbl["mag_std_V"][3] == 12.5


def test_ooi_coordinates_skip_objects_without_position():
    good = SimpleNamespace(coordinates_object=OOI)
    assert ooi_coordinates([good, SimpleNamespace(coordinates_object=None), SimpleNamespace()]).shape == (1,)
    assert ooi_coordinates([]) is None


def test_config_default_radius():
    from ost_photometry.analyze.pipeline import PipelineConfig

    assert PipelineConfig().ooi_calibrator_exclusion_arcsec == 5.0
