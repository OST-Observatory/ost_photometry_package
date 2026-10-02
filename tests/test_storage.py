"""Storage type of written images and removal of redundant reduced frames."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty

from ost_photometry.reduce.storage import (
    cast_for_storage,
    cast_like,
    check_storage_dtype,
    file_float_dtype,
)


def _ccd(dtype=np.float64):
    return CCDData(np.arange(12, dtype=dtype).reshape(3, 4), unit="electron",
                   uncertainty=StdDevUncertainty(np.ones((3, 4), dtype=dtype)))


def test_check_storage_dtype():
    assert check_storage_dtype(None) == np.float32
    assert check_storage_dtype("float64") == np.float64
    with pytest.raises(ValueError, match="storage_dtype"):
        check_storage_dtype("float16")


def test_cast_for_storage_casts_data_and_uncertainty():
    ccd = cast_for_storage(_ccd(), "float32")
    assert ccd.data.dtype == np.float32 and ccd.uncertainty.array.dtype == np.float32
    raw = CCDData(np.zeros((2, 2), dtype=np.uint16), unit="adu")
    assert cast_for_storage(raw, "float32").data.dtype == np.uint16  # raw stays


def test_cast_like_follows_the_input(tmp_path):
    out = cast_like(_ccd(), _ccd(np.float32))
    assert out.data.dtype == np.float32
    assert cast_like(_ccd(np.float32), _ccd(np.float64)).data.dtype == np.float64
    assert cast_like(_ccd(), None).data.dtype == np.float64
    assert cast_like(_ccd(), np.dtype("uint16")).data.dtype == np.float64

    path = tmp_path / "f.fit"
    cast_for_storage(_ccd(), "float32").write(path)
    assert file_float_dtype(path) == np.float32
    with fits.open(path) as hdul:
        assert hdul[0].header["BITPIX"] == -32 and hdul["UNCERT"].header["BITPIX"] == -32
    fits.writeto(tmp_path / "raw.fit", np.zeros((2, 2), dtype=np.int16))
    assert file_float_dtype(tmp_path / "raw.fit") is None


def test_remove_aligned_originals(tmp_path):
    from ost_photometry.reduce.workflow.combine import remove_aligned_originals

    reduced, aligned = tmp_path / "reduced", tmp_path / "aligned"
    reduced.mkdir()
    aligned.mkdir()
    for name in ("a.fit", "b.fit"):
        (reduced / name).write_text("x")
    (aligned / "a.fit").write_text("x")  # b could not be aligned
    originals = {"a.fit": reduced / "a.fit", "b.fit": reduced / "b.fit"}
    assert remove_aligned_originals(originals, aligned) == 1
    assert not (reduced / "a.fit").exists() and (reduced / "b.fit").exists()
