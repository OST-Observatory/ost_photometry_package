"""Camera specification CSVs and interpolation."""

from __future__ import annotations

from ost_photometry.calibration_parameters import camera_info, get_chip_dimensions
from ost_photometry.camera_specs import (
    camera_catalog,
    camera_spec_files,
    interpolate_camera_curve,
    parse_spec_filename,
)


def test_parse_qhy600_photography():
    meta = parse_spec_filename("qhy600M_system_gain_photography.csv")
    assert meta is not None
    assert meta["camera"] == "qhy600m"
    assert meta["quantity"] == "system_gain"
    assert meta["mode"] == "photography"


def test_parse_combined_high_gain_modes():
    meta = parse_spec_filename(
        "qhy600M_system_gain_high_gain_and_high_gain_2cms.csv"
    )
    assert meta is not None
    assert meta["mode"] == "high_gain"
    assert "high_gain_2cms" in (meta["extra_modes"] or "")


def test_parse_stf_dark_and_485c_qe():
    dark = parse_spec_filename("stf8300_dark_current.csv")
    assert dark is not None
    assert dark["camera"] == "stf8300"
    assert dark["quantity"] == "dark_current"
    qe = parse_spec_filename("qhy5III485C_QE_green.csv")
    assert qe is not None
    assert qe["quantity"] == "qe"
    assert qe["channel"] == "green"
    fuel = parse_spec_filename("qhy5III462_Fuelwell_vs_gain.csv")
    assert fuel is not None
    assert fuel["quantity"] == "fullwell"


def test_catalog_ships_qhy600_and_stf():
    names = camera_spec_files()
    assert "qhy600M_system_gain_photography.csv" in names
    assert "stf8300_dark_current.csv" in names
    cameras = camera_catalog()["cameras"]
    assert "qhy600m" in cameras
    assert "stf8300" in cameras
    assert cameras["qhy600m"]["chip_mm"] == [32.0, 24.0]


def test_interpolate_qhy600_gain_and_read_noise():
    gain = interpolate_camera_curve(
        "QHY600M",
        "system_gain",
        0.0,
        "PhotoGraphic DSO",
        for_camera_info=True,
    )
    rn = interpolate_camera_curve(
        "QHY600M",
        "readout_noise",
        0.0,
        "PhotoGraphic DSO",
        for_camera_info=True,
    )
    dark = interpolate_camera_curve(
        "QHY600M",
        "dark_current",
        -20.0,
        None,
        for_camera_info=True,
    )
    assert gain is not None and 1.0 < gain < 1.6
    assert rn is not None and 6.0 < rn < 9.0
    assert dark is not None and 0.0005 < dark < 0.01


def test_qhy268m_uses_268mc_curves():
    # The drop has 268MC read-noise and dark traces, but no system-gain CSVs.
    rn = interpolate_camera_curve(
        "QHY268M",
        "readout_noise",
        0.0,
        "Extend Fullwell 2CMS",
        for_camera_info=True,
    )
    assert rn is not None and 4.0 < rn < 8.0


def test_stf8300_dark_from_csv():
    dark = interpolate_camera_curve(
        "SBIG STF-8300 CCD Camera",
        "dark_current",
        -15.0,
        None,
        for_camera_info=True,
    )
    assert dark is not None and dark > 0.0


def test_camera_info_qhy600_finite():
    rn, gain, dark, width, height = camera_info(
        "QHY600M",
        "Extend Fullwell 2CMS",
        -10.0,
        gain_setting=30,
    )
    assert rn > 0
    assert gain > 0
    assert dark > 0
    assert width == 32.0
    assert height == 24.0


def test_485c_gain_and_qe():
    gain = interpolate_camera_curve(
        "QHY485C",
        "system_gain",
        50.0,
        "readout_mode_0",
        for_camera_info=True,
    )
    assert gain is not None and 0.05 < gain < 2.0
    qe = interpolate_camera_curve(
        "QHY485C",
        "qe",
        550.0,
        None,
        for_camera_info=False,
    )
    assert qe is not None and 0.0 < qe < 1.0


def test_parse_asi2600_stats_channels():
    red = parse_spec_filename("asi2600_stats_red.csv")
    pink = parse_spec_filename("asi2600_stats_pink.csv")
    blue = parse_spec_filename("asi2600_stats_blue.csv")
    assert red is not None and red["quantity"] == "readout_noise"
    assert pink is not None and pink["quantity"] == "system_gain"
    assert blue is not None and blue["quantity"] == "dynamic_range"


def test_462_single_mode_gain_and_read_noise():
    gain = interpolate_camera_curve(
        "QHY5III462",
        "system_gain",
        100.0,
        "default",
        for_camera_info=True,
    )
    rn = interpolate_camera_curve(
        "QHY5III462",
        "readout_noise",
        100.0,
        "default",
        for_camera_info=True,
    )
    assert gain is not None and 0.05 < gain < 3.0
    assert rn is not None and 0.2 < rn < 5.0


def test_asi2600_gain_and_read_noise_from_stats():
    gain = interpolate_camera_curve(
        "ZWO ASI2600MC",
        "system_gain",
        0.0,
        None,
        for_camera_info=True,
    )
    rn = interpolate_camera_curve(
        "ZWO ASI2600MC",
        "readout_noise",
        0.0,
        None,
        for_camera_info=True,
    )
    assert gain is not None and 0.5 < gain < 1.2
    assert rn is not None and 2.0 < rn < 5.0


def test_get_chip_dimensions_returns_width_height():
    width, height = get_chip_dimensions("QHY600M")
    assert (width, height) == (32.0, 24.0)


def test_camera_info_stf8300_uses_catalog_defaults():
    rn, gain, dark, width, height = camera_info(
        "SBIG STF-8300 CCD Camera",
        "default",
        -15.0,
        gain_setting=None,
    )
    assert rn == 9.3
    assert gain is None
    assert dark > 0
    assert width == 17.96
    assert height == 13.52
