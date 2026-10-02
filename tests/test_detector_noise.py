"""Gain / read noise: binning scaling, measurement from frame pairs, uncertainty."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table

from ost_photometry.camera_specs import binning_mode
from ost_photometry.reduce.detector_noise import (
    NoiseMeasurement,
    add_signal_uncertainty,
    binned_read_noise,
    check_noise_source,
    measure_detector_noise,
)

SHAPE = (300, 400)
GAIN = 1.4  # e-/ADU
READ_NOISE_ADU = 11.0
BIAS_LEVEL = 730.0


def _write(path, data, header):
    hdr = fits.Header()
    for key, value in header.items():
        hdr[key] = value
    fits.writeto(path, np.clip(data, 0, 65535).astype(np.uint16), hdr, overwrite=True)
    return str(path)


def _frames(tmp_path, *, n_bias=4, n_flats=4, n_darks=0, extra=None, seed=3):
    """Bias / dark / flat frames with known gain and read noise."""
    rng = np.random.default_rng(seed)
    extra = dict(extra or {})
    pattern = BIAS_LEVEL + rng.normal(0, 3, SHAPE)  # fixed pattern, cancels in pairs
    yy, xx = np.indices(SHAPE)
    illumination = (1.0 - 0.1 * (((xx - 200) / 200) ** 2 + ((yy - 150) / 150) ** 2)) \
        * (1 + rng.normal(0, 0.02, SHAPE))  # vignetting and pixel response
    files = {"bias": [], "dark": [], "flat": []}
    for i in range(n_bias):
        data = pattern + rng.normal(0, READ_NOISE_ADU, SHAPE)
        files["bias"].append(_write(tmp_path / f"bias_{i}.fit", data,
                                    {"IMAGETYP": "Bias Frame", "EXPTIME": 0.0, **extra}))
    for i in range(n_darks):
        data = pattern + 2.0 + rng.normal(0, READ_NOISE_ADU, SHAPE)
        files["dark"].append(_write(tmp_path / f"dark_{i}.fit", data,
                                    {"IMAGETYP": "Dark Frame", "EXPTIME": 2.0, **extra}))
    for i in range(n_flats):
        electrons = rng.poisson(30000.0 * illumination)
        data = pattern + electrons / GAIN + rng.normal(0, READ_NOISE_ADU, SHAPE)
        files["flat"].append(_write(tmp_path / f"flat_{i}.fit", data,
                                    {"IMAGETYP": "Flat Field", "EXPTIME": 2.0, "FILTER": "V",
                                     "DATE-OBS": f"2022-03-08T18:0{i}:00", **extra}))
    return files


def test_binning_mode_from_catalog_and_name():
    assert binning_mode("QHY600M") == "digital"
    assert binning_mode("QHYCCD-Cameras-Capture") == "digital"
    assert binning_mode("ZWO ASI2600") == "digital"
    assert binning_mode("SBIG STF-8300 CCD Camera") == "charge"
    assert binning_mode("SBIG ST-8 3 CCD Camera") == "charge"
    assert binning_mode("Some Camera") is None


def test_binned_read_noise():
    assert binned_read_noise(5.83, "QHY600M", 3, 3) == (pytest.approx(17.49), "digital")
    assert binned_read_noise(5.83, "QHY600M", 1, 1)[0] == pytest.approx(5.83)
    assert binned_read_noise(9.3, "SBIG STF-8300 CCD Camera", 2, 2) == (9.3, "charge")
    assert binned_read_noise(9.3, "Some Camera", 2, 2) == (9.3, None)
    assert binned_read_noise(4.0, "Some Camera", 2, 2, binning_mode="digital")[0] == 8.0
    assert binned_read_noise(4.0, "QHY600M", 2, 2, binning_mode="charge")[0] == 4.0
    with pytest.raises(ValueError, match="binning_mode"):
        binned_read_noise(4.0, "QHY600M", 2, 2, binning_mode="sum")
    with pytest.raises(ValueError, match="camera_noise_source"):
        check_noise_source("header")


def test_measure_detector_noise_from_bias_and_flat_pairs(tmp_path):
    files = _frames(tmp_path)
    m = measure_detector_noise(files["bias"], files["flat"], saturation_level=65535)
    assert m.zero_source == "bias" and m.n_zero_pairs == 2 and m.n_flat_pairs == 2
    assert m.read_noise_adu == pytest.approx(READ_NOISE_ADU, rel=0.03)
    assert m.gain == pytest.approx(GAIN, rel=0.05)
    assert m.read_noise() == pytest.approx(m.read_noise_adu * m.gain)
    assert m.read_noise(2.0) == pytest.approx(2.0 * m.read_noise_adu)


def test_measure_detector_noise_dark_fallback_and_failures(tmp_path):
    files = _frames(tmp_path, n_bias=0, n_darks=2)
    m = measure_detector_noise([], files["flat"], files["dark"], saturation_level=65535)
    assert m.zero_source == "dark"
    assert m.read_noise_adu == pytest.approx(READ_NOISE_ADU, rel=0.05)
    assert m.gain == pytest.approx(GAIN, rel=0.05)
    # One bias frame and no darks: no pair, no measurement.
    assert measure_detector_noise(files["flat"][:1], files["flat"]) is None
    # Bias pairs but no flats: read noise only.
    (tmp_path / "b").mkdir()
    files = _frames(tmp_path / "b", n_flats=0)
    m = measure_detector_noise(files["bias"], [])
    assert m.gain is None and m.read_noise() is None


def test_flats_near_saturation_are_not_used(tmp_path):
    files = _frames(tmp_path)
    m = measure_detector_noise(files["bias"], files["flat"], saturation_level=20000)
    assert m.gain is None


def test_add_signal_uncertainty_uses_bias_free_signal():
    data = np.array([[0.0, 100.0], [1000.0, -50.0]])
    ccd = CCDData(data, unit="adu")
    out = add_signal_uncertainty(ccd, gain=2.0, read_noise=10.0)
    expected = np.sqrt(np.clip(data, 0, None) * 2.0 + 100.0) / 2.0
    np.testing.assert_allclose(out.uncertainty.array, expected)
    # An attached uncertainty (subtracted masters) adds in quadrature.
    ccd.uncertainty = StdDevUncertainty(np.full(data.shape, 3.0))
    out = add_signal_uncertainty(ccd, gain=2.0, read_noise=10.0)
    np.testing.assert_allclose(out.uncertainty.array, np.hypot(expected, 3.0))


def test_resolve_cosmic_ray_removal():
    from ost_photometry.reduce.workflow.science import resolve_cosmic_ray_removal

    assert resolve_cosmic_ray_removal("auto", 10, min_frames=7) is False
    assert resolve_cosmic_ray_removal("auto", 6, min_frames=7) is True
    assert resolve_cosmic_ray_removal("auto", 0) is True  # not stacked
    assert resolve_cosmic_ray_removal(True, 50) is True
    assert resolve_cosmic_ray_removal(False, 0) is False
    with pytest.raises(ValueError, match="rm_cosmic_rays"):
        resolve_cosmic_ray_removal("always", 3)


QHY_HEADER = {"INSTRUME": "QHY600M", "XBINNING": 3, "YBINNING": 3, "READOUTM": "Photographic DSO",
              "GAIN": 0, "OFFSET": 5, "CCD-TEMP": -10.0, "SET-TEMP": -10.0}


def test_resolve_camera_parameters_catalog_and_measured(tmp_path):
    pytest.importorskip("ccdproc")
    from ost_photometry import calibration_parameters
    from ost_photometry.reduce.image_collection import image_file_collection
    from ost_photometry.reduce.workflow.config import ReduceConfig
    from ost_photometry.reduce.workflow.main import resolve_camera_parameters

    _frames(tmp_path, extra=QHY_HEADER)
    collection = image_file_collection(tmp_path)
    types = calibration_parameters.get_image_types()

    def resolve(**kwargs):
        cfg = ReduceConfig(image_path=tmp_path, output_dir=tmp_path, image_type_dir=types,
                           ignore_readout_mode_mismatch=True, **kwargs)
        return resolve_camera_parameters(collection, cfg)

    native = resolve(binning_mode="charge")
    catalog = resolve()
    assert catalog.binning == (3, 3) and catalog.binning_mode == "digital"
    assert catalog.noise_source == "catalog"
    assert catalog.read_noise == pytest.approx(3.0 * native.read_noise)

    measured = resolve(camera_noise_source="measured")
    assert measured.noise_source == "measured"
    assert measured.gain == pytest.approx(GAIN, rel=0.05)
    assert measured.read_noise == pytest.approx(READ_NOISE_ADU * measured.gain, rel=0.03)

    # Explicit values always win.
    user = resolve(camera_noise_source="measured", gain=2.0, read_noise=7.0)
    assert (user.gain, user.read_noise, user.noise_source) == (2.0, 7.0, "user")
    # Measured read noise with a user gain: converted with that gain.
    mixed = resolve(camera_noise_source="measured", gain=2.0)
    assert mixed.read_noise == pytest.approx(READ_NOISE_ADU * 2.0, rel=0.03)


def test_measure_plan_noise_shares_gain_across_offsets(tmp_path):
    from ost_photometry.reduce.workflow.groups import measure_plan_noise

    rows = []
    for offset in (5, 30):
        (tmp_path / f"o{offset}").mkdir()
        files = _frames(tmp_path / f"o{offset}", seed=offset)
        eid = f"qhy600m|3x3|photography|g0|o{offset}|t-10"
        for kind, frame_type in (("bias", "bias"), ("flat", "flat")):
            n = len(files[kind]) if offset == 5 or kind == "bias" else 0
            for path in files[kind][:n]:
                rows.append((path, frame_type, eid))
    frames = Table(rows=rows, names=("local_path", "frame_type", "electronic_id"))
    noise = measure_plan_noise(frames, saturation_level=65535, log=lambda *_: None)
    assert set(noise) == {"qhy600m|3x3|photography|g0|o5|t-10",
                          "qhy600m|3x3|photography|g0|o30|t-10"}
    with_flats, without_flats = noise.values()
    assert isinstance(without_flats, NoiseMeasurement)
    # Offset 30 has no flats: it gets the gain measured with offset 5.
    assert without_flats.gain == with_flats.gain == pytest.approx(GAIN, rel=0.05)
    assert without_flats.read_noise_adu == pytest.approx(READ_NOISE_ADU, rel=0.05)


def test_stacked_frame_counts():
    from ost_photometry.reduce.workflow.groups import stacked_frame_counts

    frames = Table(rows=[
        ("1", "T1", "qhy600m", "V"), ("2", "T1", "qhy600m", "V"), ("3", "T1", "qhy600m", "B"),
        ("4", "T2", "qhy600m", "V"), ("5", "T1", "qhy600m", "V"),
    ], names=("frame_id", "target_id", "camera", "filter"))
    plan = {"units": {"u1": {"lights": ["1", "2", "3", "4"]}, "u2": {"lights": ["5"]}},
            "targets": {"T1": {"stack": True}, "T2": {"stack": False}}}
    assert stacked_frame_counts(plan, frames) == {"1": 3, "2": 3, "5": 3, "3": 1, "4": 0}


def test_saturation_in_electrons():
    from ost_photometry.reduce.detector_noise import saturation_in_electrons

    assert saturation_in_electrons(65535, 735, 1.3, 1.0) == pytest.approx(64800 * 1.3)
    assert saturation_in_electrons(65535, 735, 1.3, 1.2) == pytest.approx(64800 * 1.3 / 1.2)
    assert saturation_in_electrons(100, 735, 1.3) == 0.0


def test_stack_noise_values():
    from ost_photometry.reduce.detector_noise import stack_noise_values

    headers = [{"EXPTIME": 60.0, "RDNOISE": 17.0, "SATLEVEL": 80000.0} for _ in range(4)]
    rn, sat = stack_noise_values(headers, None, rate_images=True, total_exptime=240.0)
    assert rn == pytest.approx(17.0 * 2)  # electrons summed over 4 frames
    assert sat == pytest.approx(4 * 80000.0)
    # Mixed exposures: the short frame dominates the rate noise; the long
    # frames saturate at the lowest rate.
    headers[0] = {"EXPTIME": 10.0, "RDNOISE": 17.0, "SATLEVEL": 80000.0}
    rn, sat = stack_noise_values(headers, [1, 1, 1, 1], rate_images=True, total_exptime=190.0)
    rate_noise = np.sqrt((17 / 10) ** 2 + 3 * (17 / 60) ** 2) / 4
    assert rn == pytest.approx(190.0 * rate_noise)
    assert sat == pytest.approx(190.0 * 80000.0 / 60.0)
    # Images in electrons: mean of the frames.
    rn, sat = stack_noise_values(headers, None, rate_images=False, total_exptime=None)
    assert rn == pytest.approx(17.0 / 2) and sat == pytest.approx(80000.0)
    # A frame without the keywords: unknown.
    assert stack_noise_values([*headers, {"EXPTIME": 60.0}], None, rate_images=True,
                              total_exptime=250.0) == (None, None)


def test_cosmic_header_flags():
    from ost_photometry.fits_headers import (
        clear_cosmics_identified,
        cosmics_identified,
        mark_cosmics_identified,
    )

    header = fits.Header()
    mark_cosmics_identified(header, handling="clipped")
    assert cosmics_identified(header) and header["CRCLIP"]
    clear_cosmics_identified(header)
    assert not cosmics_identified(header) and "CRCLIP" not in header


def test_analysis_cosmic_parameters_from_header():
    from ost_photometry.analyze.extraction import cosmic_ray_noise_parameters

    header = fits.Header({"RDNOISE": 45.0, "SATLEVEL": 5e5})
    assert cosmic_ray_noise_parameters(header) == (45.0, 5e5, [])
    assert cosmic_ray_noise_parameters(header, 10.0, 6e4)[:2] == (10.0, 6e4)
    rn, sat, notes = cosmic_ray_noise_parameters(fits.Header())
    assert (rn, sat) == (8.0, 65535.0) and len(notes) == 2
