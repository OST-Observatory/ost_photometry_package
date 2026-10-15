"""QHY frames without the driver readout mode (MaxIm DL "Fast" / "Normal")."""

from __future__ import annotations

import pytest

from ost_photometry import camera_specs as cs
from ost_photometry.reduce.grouping.darks import _split_placeholder_labels


def test_placeholders_are_unknown_real_modes_are_not():
    for label in ("Fast", "Normal", "slow", None, "", "unknown"):
        assert cs.is_unknown_readout_mode(label)
    assert cs.normalize_readout_mode("Fast") == cs.UNKNOWN_READOUT_MODE
    assert not cs.is_unknown_readout_mode("Extend Fullwell 2CMS")
    assert cs.normalize_readout_mode("Extend Fullwell 2CMS") == "extend_fullwell_2cms"


def test_site_measurement_per_binning():
    bin1 = cs.site_measurement("QHY600M", 0, offset=2)
    assert bin1["system_gain"] == pytest.approx(1.314)
    assert bin1["read_noise"] == pytest.approx(4.29 * 1.314)
    bin2 = cs.site_measurement("QHY600M", 0, offset=2, xbin=2, ybin=2)
    assert bin2["read_noise"] == pytest.approx(8.74 * 1.314) and not bin2["derived_read_noise"]
    # bin 3 measured on its own (not 3x the native value)
    bin3 = cs.site_measurement("QHY600M", 0, offset=0, xbin=3, ybin=3)
    assert bin3["read_noise"] == pytest.approx(8.92 * 1.296)
    assert cs.site_measurement("QHY600M", 56) is None


def test_mode_candidates_from_measured_gain_and_read_noise():
    # OST QHY600M at GAIN 0: 1.31 e-/ADU, 5.7 e- per native pixel -> a 2CMS mode
    modes = cs.readout_mode_candidates("QHY600M", 0, system_gain=1.31, read_noise=5.7)
    assert set(modes) <= {"extend_fullwell_2cms", "photography_2cms"} and modes
    assert "high_gain" not in modes and "extend_fullwell" not in modes
    full_well = cs.full_well_adu("QHY600M", 0, modes)
    assert 60000 < full_well < 65535


def _row(label, level, kind="bias"):
    return {"frame_type": kind, "readoutm": label, "stat_median": level, "stat_noise": 4.5}


def test_placeholder_labels_merge_unless_bias_levels_differ():
    key = "qhy600m|1x1|unknown|g0|o2"
    same = [_row("Fast", 34.0), _row("Normal", 34.5), _row("Fast", 33.8), _row("Normal", 0.0, "light")]
    assert set(_split_placeholder_labels(same, [key] * 4)) == {key}
    differ = [_row("Fast", 34.0), _row("Normal", 120.0), _row("Normal", 0.0, "dark")]
    out = _split_placeholder_labels(differ, [key] * 3)
    assert out[0] == "qhy600m|1x1|unknown-fast|g0|o2"
    assert out[1] == out[2] == "qhy600m|1x1|unknown-normal|g0|o2"


def test_camera_info_uses_the_ost_gain_for_maxim_labels():
    from ost_photometry.calibration_parameters import camera_info

    read_noise, gain, *_ = camera_info("QHY600M", "Fast", -10.0, gain_setting=0, offset=2)
    assert gain == pytest.approx(1.314) and read_noise == pytest.approx(4.29 * 1.314)
    # offset 0 is measured only at bin 3: the gain still comes from it
    _rn, gain0, *_ = camera_info("QHY600M", "Normal", -10.0, gain_setting=0, offset=0)
    assert gain0 == pytest.approx(1.296)
