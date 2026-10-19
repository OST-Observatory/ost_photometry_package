"""Vega / AB / SDSS magnitude-system offsets."""

from __future__ import annotations


def test_sdss_catalog_u_z_get_ab_offsets():
    import numpy as np
    from astropy.table import Table

    from ost_photometry.analyze.post_processing.convert import convert_magnitudes_to_other_system

    tbl = Table({"id": [1], "mag_cal_u": [15.0], "err_cal_u": [0.02], "mag_cal_g": [14.0],
                 "err_cal_g": [0.01], "mag_cal_z": [13.0], "err_cal_z": [0.02]})
    out = convert_magnitudes_to_other_system(tbl, output_magnitude_system="ab", calibration_source="V/147/sdss12")
    assert np.isclose(out["mag_cal_u"][0], 14.96) and np.isclose(out["mag_cal_z"][0], 13.02)
    assert out["mag_cal_g"][0] == 14.0
