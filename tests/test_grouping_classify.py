"""Setup keys and frame-type classification on synthetic frames."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from ost_photometry.archive.manifest import manifest_from_rows
from ost_photometry.reduce.grouping import setup_keys
from ost_photometry.reduce.grouping.classify import (
    BIAS,
    DARK,
    FLAT,
    LIGHT,
    SPECTROSCOPY,
    UNKNOWN,
    classify_frame,
    classify_frames,
    count_stars,
    frame_statistics,
    header_frame_type,
    type_disagreements,
)

SHAPE = (300, 400)
RNG = np.random.default_rng(5)


def _bias():
    return 730.0 + RNG.normal(0, 4, SHAPE)


def _dark():
    data = _bias() + 5.0
    hot = RNG.integers(0, SHAPE[0] * SHAPE[1], 400)
    data.flat[hot] += 3000.0  # isolated hot pixels must not count as stars
    return data


def _flat(level=30000.0):
    yy, xx = np.indices(SHAPE)
    vignetting = 1.0 - 0.15 * (((xx - 180) / 200) ** 2 + ((yy - 140) / 150) ** 2)
    return 730.0 + level * vignetting + RNG.normal(0, np.sqrt(level), SHAPE)


def _light(n_stars=40):
    data = _bias() + 200.0 + RNG.normal(0, 8, SHAPE)
    yy, xx = np.indices(SHAPE)
    for _ in range(n_stars):
        x, y = RNG.uniform(20, SHAPE[1] - 20), RNG.uniform(20, SHAPE[0] - 20)
        data += RNG.uniform(500, 5000) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2))
    return data


def _spectrum():
    data = _bias() + 50.0
    for y0 in (80, 150, 220):  # three horizontal traces (orders)
        data[y0 - 3 : y0 + 3, :] += 4000.0
    return data


def _write(path, data, **header):
    hdr = fits.Header()
    for key, value in header.items():
        hdr[key] = value
    fits.writeto(path, np.clip(data, 0, 65535).astype(np.uint16), hdr, overwrite=True)
    return str(path)


def test_header_frame_type():
    assert header_frame_type("Bias Frame") == BIAS
    assert header_frame_type("DARK") == DARK
    assert header_frame_type("Flat Field") == FLAT
    assert header_frame_type("Light Frame") == LIGHT
    assert header_frame_type("") == UNKNOWN
    assert header_frame_type("Tricolor Image") == UNKNOWN


def test_count_stars_ignores_hot_pixels():
    assert count_stars(_dark()) <= 2
    assert count_stars(_light(40)) >= 25
    assert count_stars(_flat()) <= 2


def test_frame_statistics_detects_spectral_traces(tmp_path):
    spec = frame_statistics(_write(tmp_path / "s.fit", _spectrum()))
    light = frame_statistics(_write(tmp_path / "l.fit", _light()))
    assert spec["anisotropy"] > 15 and spec["amplitude"] > 0.4
    assert light["anisotropy"] < 5
    assert light["fill"] == pytest.approx(light["median"] / 65535.0)


def test_classify_frames_end_to_end(tmp_path):
    frames = [
        ("bias.fit", _bias(), "Bias Frame", 0.0),
        ("dark.fit", _dark(), "Dark Frame", 60.0),
        ("flat.fit", _flat(), "Flat Field", 5.0),
        ("light.fit", _light(), "Light Frame", 60.0),
        ("mislabelled.fit", _light(), "Flat Field", 60.0),
        ("spectrum.fit", _spectrum(), "Light Frame", 300.0),
    ]
    rows = []
    for i, (name, data, imagetyp, exptime) in enumerate(frames):
        path = _write(tmp_path / name, data, IMAGETYP=imagetyp, EXPTIME=exptime)
        rows.append({"frame_id": str(i), "file_name": name, "local_path": path,
                     "imagetyp": imagetyp, "exptime": exptime, "jd": float(i),
                     "instrume": "QHY600M", "naxis1": SHAPE[1], "naxis2": SHAPE[0]})
    table = classify_frames(manifest_from_rows(rows))
    kinds = dict(zip(table["file_name"], table["frame_type"], strict=True))
    assert kinds == {
        "bias.fit": BIAS,
        "dark.fit": DARK,
        "flat.fit": FLAT,
        "light.fit": LIGHT,
        "mislabelled.fit": LIGHT,
        "spectrum.fit": SPECTROSCOPY,
    }
    note = table["type_note"][list(table["file_name"]).index("mislabelled.fit")]
    assert "header says flat" in note
    assert list(type_disagreements(table)["file_name"]) == ["mislabelled.fit", "spectrum.fit"]


def test_classify_frame_priorities_without_pixels():
    base = {"imagetyp": "Light Frame", "exptime": 60.0}
    assert classify_frame({**base, "exposure_type_user": "FL"}, None)[0] == FLAT
    assert classify_frame({**base, "spectrograph": "D"}, None)[0] == SPECTROSCOPY
    kind, confidence, note = classify_frame(base, None)
    assert kind == LIGHT and confidence < 0.9 and "no pixel data" in note
    assert classify_frame({"exposure_type_ml": "DA"}, None)[0] == DARK
    assert classify_frame({}, None)[0] == UNKNOWN


def test_classify_dark_vs_flat_without_bias_uses_header():
    stats = {"median": 2000.0, "noise": 10.0, "fill": 0.03, "n_stars": 0,
             "anisotropy": 1.2, "amplitude": 0.05}
    assert classify_frame({"imagetyp": "Flat Field", "exptime": 2.0}, stats)[0] == FLAT
    assert classify_frame({"imagetyp": "Dark Frame", "exptime": 2.0}, stats)[0] == DARK
    assert classify_frame({"exptime": 2.0}, stats)[0] == UNKNOWN


def test_camera_ids_and_keys():
    qhy = {"instrume": "QHYCCD-Cameras-Capture", "naxis1": 3192, "naxis2": 2129,
           "xbinning": 3, "ybinning": 3, "readoutm": "Normal", "gain": 0, "offset": 5,
           "filter": "V", "telescop": "Planewave CDK20"}
    assert setup_keys.camera_id(qhy) == "qhy600m"
    assert setup_keys.camera_id({"instrument_archive": "SBIG ST-8 3 CCD Camera"}) == "sbig-st-8"
    assert setup_keys.camera_id({"instrume": "SBIG ST-8"}) == "sbig-st-8"
    assert setup_keys.camera_id({}) == "unknown"
    assert setup_keys.telescope_id(qhy) == "CDK20"
    assert setup_keys.telescope_id({"focallen": 3454.0}) == "f3454"
    assert setup_keys.electronic_key(qhy) == "qhy600m|3x3|normal|g0|o5"
    other_offset = {**qhy, "offset": 2}
    assert setup_keys.electronic_key(other_offset) != setup_keys.electronic_key(qhy)
    assert setup_keys.optical_key(qhy) == "qhy600m|3x3|V|CDK20"
    assert setup_keys.filter_name({"filter": "**LEER**"}) == "none"


def test_temperature_clusters():
    labels = setup_keys.temperature_clusters([-20.0, -19.2, -10.0, float("nan"), -21.5, -9.0])
    assert labels[0] == labels[1] == labels[4]
    assert labels[2] == labels[5] != labels[0]
    assert labels[3] == -1
    assert setup_keys.sensor_temperature({"set_temp": -20.0, "ccd_temp": -19.0}) == -20.0
    assert setup_keys.sensor_temperature({"ccd_temp": -19.0}) == -19.0


def test_saturated_and_name_hints():
    sat = {"median": 65000.0, "noise": 50.0, "fill": 0.99, "n_stars": 0,
           "anisotropy": 1.0, "amplitude": 0.01}
    assert classify_frame({"imagetyp": "Light Frame", "exptime": 40.0}, sat)[0] == "saturated"
    stars = {"median": 1000.0, "noise": 10.0, "fill": 0.02, "n_stars": 50,
             "anisotropy": 1.2, "amplitude": 0.05}
    for name in ("spectrum_1800s.fit", "NeAr_20s.fit", "thar.fit", "autoguiderFOV.fit"):
        assert classify_frame({"file_name": name, "exptime": 20.0}, stars)[0] == SPECTROSCOPY
    assert classify_frame({"file_name": "x.fit", "source_path": "2022/CalibDADOS/x.fit",
                           "exptime": 20.0}, stars)[0] == SPECTROSCOPY
    assert classify_frame({"file_name": "M57-0001_V_60s.fit", "exptime": 60.0},
                          stars)[0] == LIGHT
