"""Tests for gain / EGAIN and readout-mode header handling."""

from unittest.mock import MagicMock

from astropy.table import MaskedColumn, Table

from ost_photometry.reduce.instrument import resolve_readout_mode, resolve_system_gain


def _mock_collection(summary: Table, files: list[str] | None = None):
    collection = MagicMock()
    collection.summary = summary
    collection.files = files or [f"image_{index}.fits" for index in range(len(summary))]
    return collection


def test_resolve_readout_mode_defaults_for_legacy_camera_without_keyword():
    summary = Table()
    readout_mode = resolve_readout_mode(
        _mock_collection(summary),
        "SBIG STF-8300 CCD Camera",
    )
    assert readout_mode == "default"


def test_resolve_readout_mode_defaults_for_qhy_without_keyword():
    summary = Table()
    readout_mode = resolve_readout_mode(
        _mock_collection(summary),
        "QHY268M",
    )
    assert readout_mode == "Extend Fullwell 2CMS"


def test_resolve_readout_mode_reads_readoutm_keyword():
    summary = Table(
        [MaskedColumn(data=["Slow"], mask=[False])],
        names=["readoutm"],
    )
    readout_mode = resolve_readout_mode(
        _mock_collection(summary),
        "SBIG STF-8300 CCD Camera",
    )
    assert readout_mode == "Slow"


def test_resolve_readout_mode_maps_qhy_numeric_mode():
    summary = Table(
        [MaskedColumn(data=[1], mask=[False])],
        names=["readmode"],
    )
    readout_mode = resolve_readout_mode(
        _mock_collection(summary),
        "QHY600M",
    )
    assert readout_mode == "High Gain Mode"


def test_resolve_system_gain_prefers_user_override():
    gain = resolve_system_gain(
        "QHY600M",
        gain_setting=26,
        egain=1.5,
        calibration_gain=1.292,
        user_gain=2.0,
    )
    assert gain == 2.0


def test_resolve_system_gain_uses_egain_for_legacy_camera():
    gain = resolve_system_gain(
        "SBIG STF-8300 CCD Camera",
        gain_setting=None,
        egain=1.45,
        calibration_gain=None,
    )
    assert gain == 1.45


def test_resolve_system_gain_uses_calibration_for_qhy_with_default_egain():
    gain = resolve_system_gain(
        "QHY268M",
        gain_setting=26,
        egain=1.0,
        calibration_gain=1.292,
    )
    assert gain == 1.292


def test_resolve_system_gain_prefers_egain_for_qhy_when_not_default():
    gain = resolve_system_gain(
        "QHY268M",
        gain_setting=26,
        egain=1.45,
        calibration_gain=1.292,
    )
    assert gain == 1.45


def test_resolve_system_gain_returns_none_without_any_source():
    gain = resolve_system_gain(
        "unknown camera",
        gain_setting=None,
        egain=None,
        calibration_gain=None,
    )
    assert gain is None
