"""Floating-point type of the images written by the reduction.

Calculations run in float64; the reduced frames, masters and everything
derived from them are written as ``storage_dtype`` (default float32). The
rounding of float32 (relative 6e-8) is five orders of magnitude below the
noise of a pixel, and the files take half the space. Steps that only
transform images (alignment, stacking) keep the floating type of their
input, so the choice made in the reduction carries through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty

#: Values of ``storage_dtype``.
STORAGE_DTYPES = ("float32", "float64")

DEFAULT_STORAGE_DTYPE = "float32"

_BITPIX_FLOAT = {-32: np.dtype("float32"), -64: np.dtype("float64")}


def check_storage_dtype(storage_dtype: str | np.dtype | None) -> np.dtype:
    """Validated storage type; ``None`` means the default."""
    if storage_dtype is None:
        storage_dtype = DEFAULT_STORAGE_DTYPE
    dtype = np.dtype(storage_dtype)
    if dtype.name not in STORAGE_DTYPES:
        raise ValueError(f"storage_dtype must be one of {STORAGE_DTYPES}, got {storage_dtype!r}")
    return dtype


def cast_for_storage(ccd: CCDData, storage_dtype: str | np.dtype | None) -> CCDData:
    """Cast floating data and the uncertainty to ``storage_dtype`` (in place).

    Integer data (raw frames) stays as it is.
    """
    dtype = check_storage_dtype(storage_dtype)
    if np.issubdtype(ccd.data.dtype, np.floating) and ccd.data.dtype != dtype:
        ccd.data = ccd.data.astype(dtype)
    if ccd.uncertainty is not None:
        array = np.asarray(ccd.uncertainty.array)
        if np.issubdtype(array.dtype, np.floating) and array.dtype != dtype:
            ccd.uncertainty = StdDevUncertainty(array.astype(dtype))
    return ccd


def float_dtype(data: np.ndarray | CCDData) -> np.dtype | None:
    """Floating type of an array or CCD, ``None`` for integer data."""
    array = data.data if isinstance(data, CCDData) else np.asarray(data)
    return array.dtype if np.issubdtype(array.dtype, np.floating) else None


def file_float_dtype(path: str | Path) -> np.dtype | None:
    """Floating type of the primary image of a FITS file (from ``BITPIX``)."""
    return _BITPIX_FLOAT.get(int(fits.getheader(path).get("BITPIX", 0)))


def cast_like(ccd: CCDData, reference: np.ndarray | CCDData | np.dtype | None) -> CCDData:
    """Cast ``ccd`` to the floating type of ``reference`` (in place).

    Used where an image is transformed (aligned, enlarged, stacked) so that
    the output keeps the storage type of its input. Integer or missing
    references leave ``ccd`` unchanged.
    """
    if isinstance(reference, np.dtype):
        dtype = reference if np.issubdtype(reference, np.floating) else None
    elif reference is None:
        dtype = None
    else:
        dtype = float_dtype(reference)
    if dtype is None or dtype.name not in STORAGE_DTYPES:
        return ccd
    return cast_for_storage(ccd, dtype)


__all__ = [
    "DEFAULT_STORAGE_DTYPE",
    "STORAGE_DTYPES",
    "cast_for_storage",
    "cast_like",
    "check_storage_dtype",
    "file_float_dtype",
    "float_dtype",
]
