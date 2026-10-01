"""Camera orientation from WCS, read-only ASTAP solving, instrument names."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy import wcs as astropy_wcs
from astropy.io import fits

from ost_photometry import wcs as wcs_mod
from ost_photometry.camera_specs import normalize_instrument_name


def _cd(theta_deg: float, scale_arcsec: float = 1.0, flipped: bool = False) -> np.ndarray:
    """CD matrix with +y axis at position angle ``theta`` (north through east)."""
    s = scale_arcsec / 3600.0
    c, n = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    # Columns are the (east, north) directions of the +x and +y pixel axes.
    # Normal parity: +x points west when north is up (det < 0).
    if flipped:
        return s * np.array([[c, n], [-n, c]])
    return s * np.array([[-c, n], [n, c]])


def _tan_wcs(cd: np.ndarray, crval=(150.0, 30.0), crpix=(50.0, 40.0)) -> astropy_wcs.WCS:
    w = astropy_wcs.WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cd = cd
    return w


@pytest.mark.parametrize("theta", [0.0, 12.5, 151.6, 270.0])
def test_position_angle_recovers_rotation(theta):
    pa, parity, scale = wcs_mod.position_angle_from_wcs(_tan_wcs(_cd(theta, 0.677)))
    assert pa == pytest.approx(theta, abs=1e-6)
    assert parity == "normal"
    assert scale == pytest.approx(0.677, rel=1e-6)


def test_pier_flip_is_180_degrees_and_mod180_equal():
    pa_west, _, _ = wcs_mod.position_angle_from_wcs(_tan_wcs(_cd(331.6)))
    pa_east, _, _ = wcs_mod.position_angle_from_wcs(_tan_wcs(_cd(151.6)))
    assert wcs_mod.orientation_difference_mod180(pa_west, pa_east) == pytest.approx(0.0, abs=1e-6)
    assert wcs_mod.orientation_difference_mod180(151.6, 159.9) == pytest.approx(8.3, abs=1e-6)
    assert wcs_mod.orientation_difference_mod180(1.0, 179.0) == pytest.approx(2.0, abs=1e-6)


def test_parity_detects_mirrored_image():
    _, parity, _ = wcs_mod.position_angle_from_wcs(_tan_wcs(_cd(30.0, flipped=True)))
    assert parity == "flipped"


def test_position_angle_from_pc_cdelt():
    w = astropy_wcs.WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [10.0, 10.0]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    pa, parity, scale = wcs_mod.position_angle_from_wcs(w)
    assert pa == pytest.approx(0.0, abs=1e-9)
    assert parity == "normal"
    assert scale == pytest.approx(1.0)


def test_astap_command_hint_and_blind_search():
    cmd = wcs_mod.astap_command("f.fit", fov_deg=0.6, ra_deg=180.0, dec_deg=-10.0)
    assert cmd[:3] == ["astap_cli", "-f", "f.fit"]
    assert cmd[cmd.index("-r") + 1] == "3"
    assert cmd[cmd.index("-ra") + 1] == "12"
    assert cmd[cmd.index("-spd") + 1] == "80"
    assert cmd[-1] == "-update"
    blind = wcs_mod.astap_command("f.fit", fov_deg=None, update=False)
    assert "-ra" not in blind and "-spd" not in blind
    assert blind[blind.index("-r") + 1] == "180"
    assert blind[blind.index("-fov") + 1] == "0"
    assert "-update" not in blind
    assert wcs_mod.astap_pointing_arguments(float("nan"), 3.0) == []


def _write_uint16_frame(path: Path) -> bytes:
    data = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) % 500) + 1000
    header = fits.Header()
    header["OBJECT"] = "test"
    fits.writeto(path, data, header, overwrite=True)
    return path.read_bytes()


def _fake_astap_success(solved_cd: np.ndarray):
    """Fake ASTAP: writes a TAN WCS into the copied file and reports success."""

    def run(cmd, shell=False, text=True, capture_output=True, timeout=None):
        target = Path(cmd[cmd.index("-f") + 1])
        with fits.open(target, mode="update") as hdul:
            hdul[0].header.update(_tan_wcs(solved_cd, crpix=(32.0, 24.0)).to_header())
            hdul.flush()
        target.with_suffix(".ini").write_text("PLTSOLVD=T\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="Solution found: 10 20", stderr="")

    return run


def test_solve_astap_copy_leaves_source_untouched(tmp_path, monkeypatch):
    source = tmp_path / "raw.fit"
    original = _write_uint16_frame(source)
    work = tmp_path / "work"
    monkeypatch.setattr(wcs_mod.subprocess, "run", _fake_astap_success(_cd(42.0, 2.0)))

    solved = wcs_mod.solve_astap_copy(source, work_dir=work, fov_deg=0.1)

    assert solved is not None
    pa, _, scale = wcs_mod.position_angle_from_wcs(solved)
    assert pa == pytest.approx(42.0, abs=1e-6)
    assert scale == pytest.approx(2.0)
    assert source.read_bytes() == original
    assert list(work.iterdir()) == []  # temporary copy and side files removed


def test_solve_astap_copy_returns_none_on_failure_and_timeout(tmp_path, monkeypatch):
    source = tmp_path / "raw.fit"
    _write_uint16_frame(source)

    def no_solution(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="No solution found", stderr="")

    monkeypatch.setattr(wcs_mod.subprocess, "run", no_solution)
    assert wcs_mod.solve_astap_copy(source, work_dir=tmp_path / "w") is None

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(wcs_mod.subprocess, "run", timeout)
    assert wcs_mod.solve_astap_copy(source, work_dir=tmp_path / "w", timeout=1) is None


@pytest.mark.parametrize(
    ("instrume", "naxis", "binning", "expected"),
    [
        ("QHYCCD-Cameras-Capture", (4788, 3194), (2, 2), "QHY600M"),
        ("QHYCCD-Cameras-Capture", (3192, 2129), (3, 3), "QHY600M"),
        ("QHYCCD-Cameras2-Capture", (3126, 2088), (2, 2), "QHY268M"),
        ("QHYCCD-Cameras-Capture", (6252, 4176), (1, 1), "QHY268M"),
        ("QHYCCD-Cameras-Capture", (3864, 2180), (1, 1), "QHY485C"),
        ("ASI Camera (1)", (3124, 2088), (2, 2), "ZWO ASI2600"),
        ("QHYCCD-Cameras-Capture", (100, 100), (1, 1), ""),
        ("QHY600M-abc", (1, 1), (1, 1), "QHY600M"),
        ("  SBIG ST-8 3 CCD Camera ", (1530, 1020), (1, 1), "SBIG ST-8 3 CCD Camera"),
    ],
)
def test_normalize_instrument_name(instrume, naxis, binning, expected):
    assert normalize_instrument_name(instrume, *naxis, *binning) == expected


def test_normalize_instrument_name_tolerates_missing_geometry():
    assert normalize_instrument_name("QHYCCD-Cameras-Capture", None, "x") == ""
    assert normalize_instrument_name(None) == ""
