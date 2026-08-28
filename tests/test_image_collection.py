"""ImageFileCollection helper skips directories and non-FITS files."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning


def _write_fits(path) -> None:
    fits.writeto(path, np.zeros((4, 4), dtype=np.float32), overwrite=True)


def test_fits_filenames_skips_directories_and_non_fits(tmp_path):
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce.image_collection import fits_filenames

    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "notes.txt").write_text("not fits")
    _write_fits(tmp_path / "star.fits")
    _write_fits(tmp_path / "bias.FIT")

    assert fits_filenames(tmp_path) == ["bias.FIT", "star.fits"]


def test_image_file_collection_does_not_warn_on_analysis_dirs(tmp_path, caplog):
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce.image_collection import image_file_collection

    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "legacy_fit").mkdir()
    _write_fits(tmp_path / "star.fits")

    with caplog.at_level(logging.WARNING, logger="ccdproc.image_collection"):
        collection = image_file_collection(tmp_path)

    assert collection.files == ["star.fits"]
    assert "Is a directory" not in caplog.text
    assert "unable to get FITS header" not in caplog.text


def test_image_file_collection_empty_when_only_directories(tmp_path, caplog):
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce.image_collection import image_file_collection

    (tmp_path / "diagnostics").mkdir()

    with (
        caplog.at_level(logging.WARNING, logger="ccdproc.image_collection"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", AstropyUserWarning)
        collection = image_file_collection(tmp_path)

    assert collection.files == []
    assert "Is a directory" not in caplog.text
