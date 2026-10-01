"""Manifest schema, header parsing and local-directory manifests."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from ost_photometry.archive.local import manifest_from_directory
from ost_photometry.archive.manifest import (
    apply_header,
    header_fields,
    manifest_from_rows,
    parse_ra_dec,
    read_manifest,
    write_manifest,
)


def test_parse_ra_dec_formats_and_placeholders():
    ra, dec = parse_ra_dec("18 53 35", "+33 01 45")
    assert ra == pytest.approx(283.396, abs=1e-3)
    assert dec == pytest.approx(33.029, abs=1e-3)
    assert parse_ra_dec("12:00:00", "-10:30:00")[0] == pytest.approx(180.0)
    assert parse_ra_dec(283.4, 33.0) == (283.4, 33.0)
    assert np.isnan(parse_ra_dec("", "")[0])
    assert np.isnan(parse_ra_dec("garbage", "x")[0])
    assert np.isnan(parse_ra_dec(400.0, 10.0)[0])


def test_header_fields_variants():
    header = fits.Header()
    header["IMAGETYP"] = "Flat Field"
    header["BINNING"] = "2x2"
    header["READMODE"] = "Fast"
    header["CCD-TEMP"] = -9.8
    header["PIERSIDE"] = "west"
    header["DATE-OBS"] = "2022-12-27T01:27:00"
    header["EXPOSURE"] = 30
    fields = header_fields(header)
    assert fields["xbinning"] == 2 and fields["ybinning"] == 2
    assert fields["readoutm"] == "Fast"
    assert fields["ccd_temp"] == pytest.approx(-9.8)
    assert fields["pierside"] == "WEST"
    assert fields["exptime"] == 30.0
    assert fields["jd"] == pytest.approx(2459940.5604, abs=1e-3)
    assert np.isnan(fields["header_ra"])


def test_apply_header_keeps_archive_time_and_fills_pointing():
    row = {"jd": 2459000.0, "exptime": 60.0, "ra": float("nan"), "dec": float("nan")}
    header = fits.Header()
    header["DATE-OBS"] = "2022-03-09T01:00:00"
    header["EXPTIME"] = 1.0
    header["OBJCTRA"] = "12 00 00"
    header["OBJCTDEC"] = "+10 00 00"
    apply_header(row, header)
    assert row["jd"] == 2459000.0 and row["exptime"] == 60.0
    assert row["ra"] == pytest.approx(180.0)
    assert "header_ra" not in row


def test_manifest_roundtrip(tmp_path):
    table = manifest_from_rows(
        [
            {"frame_id": "2", "jd": 2.0, "filter": "V", "plate_solved": True},
            {"frame_id": "1", "jd": 1.0, "unknown_column": "dropped"},
            {"frame_id": "3"},
        ]
    )
    assert list(table["frame_id"]) == ["1", "2", "3"]  # sorted by jd, nan last
    assert "unknown_column" not in table.colnames
    path = write_manifest(table, tmp_path / "m" / "manifest.ecsv")
    back = read_manifest(path)
    assert list(back["frame_id"]) == ["1", "2", "3"]
    assert list(back["filter"]) == ["", "V", ""]
    assert back["plate_solved"].dtype == np.bool_
    assert back["xbinning"][0] == 1


def _write(path, **cards):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    for key, value in cards.items():
        header[key.replace("_", "-") if key in ("DATE_OBS", "CCD_TEMP") else key] = value
    fits.writeto(path, np.zeros((4, 6), dtype=np.uint16), header, overwrite=True)


def test_manifest_from_directory(tmp_path):
    _write(tmp_path / "2022-03-08" / "m57" / "l1.fit", IMAGETYP="Light Frame",
           DATE_OBS="2022-03-09T01:00:00", OBJCTRA="18 53 35", OBJCTDEC="+33 01 45")
    _write(tmp_path / "2022-03-08" / "flats" / "f1.fits", IMAGETYP="Flat Field",
           DATE_OBS="2022-03-09T05:00:00", FILTER="V")
    _write(tmp_path / "2022-06-23" / "l2.fit", IMAGETYP="LIGHT", DATE_OBS="2022-06-23T21:00:00")
    (tmp_path / "2022-03-08" / "notes.txt").write_text("ignore")
    (tmp_path / "2022-03-08" / "broken.fit").write_text("not fits")
    table = manifest_from_directory(tmp_path)
    assert list(table["file_name"]) == ["l1.fit", "f1.fits", "l2.fit"]
    assert list(table["run"]) == ["2022-03-08", "2022-03-08", "2022-06-23"]
    assert list(table["role"]) == ["target", "calibration", "target"]
    assert table["ra"][0] == pytest.approx(283.396, abs=1e-3)
    assert all(table["downloaded"])
    assert len(set(table["frame_id"])) == 3
