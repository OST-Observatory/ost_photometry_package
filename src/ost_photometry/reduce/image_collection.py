"""Build ccdproc ImageFileCollections that ignore directories and non-FITS files."""

from __future__ import annotations

from pathlib import Path

import ccdproc as ccdp

_FITS_SUFFIXES = (
    ".fit",
    ".fits",
    ".fts",
    ".fit.gz",
    ".fits.gz",
    ".fts.gz",
    ".fit.fz",
    ".fits.fz",
    ".fts.fz",
)


def fits_filenames(location: str | Path) -> list[str]:
    """Return FITS basenames in ``location`` (non-recursive, files only)."""
    path = Path(location)
    if not path.is_dir():
        return []
    names = [
        entry.name
        for entry in path.iterdir()
        if entry.is_file() and entry.name.lower().endswith(_FITS_SUFFIXES)
    ]
    names.sort()
    return names


def image_file_collection(location: str | Path, **kwargs) -> ccdp.ImageFileCollection:
    """``ImageFileCollection`` that does not try to read directories as FITS.

    ccdproc globs ``*fit``, which matches analysis folders such as
    ``output/extinction_fit``, then logs ``Is a directory`` while reading
    headers. An empty ``filenames`` list is treated as unset, so that case
    uses ``glob_exclude='*'`` instead of falling back to the directory glob.
    """
    kwargs = dict(kwargs)
    if kwargs.get("filenames") is None and kwargs.get("glob_include") is None:
        names = fits_filenames(location)
        if names:
            kwargs["filenames"] = names
        elif kwargs.get("glob_exclude") is None:
            kwargs["glob_exclude"] = "*"
    return ccdp.ImageFileCollection(location=str(location), **kwargs)


__all__ = ["fits_filenames", "image_file_collection"]
