"""Build a manifest from a local directory tree (no archive access needed)."""

from __future__ import annotations

import os
from pathlib import Path

from astropy.io import fits
from astropy.table import Table

from .manifest import (
    ROLE_CALIBRATION,
    ROLE_TARGET,
    apply_header,
    manifest_from_rows,
)

FITS_SUFFIXES = (".fit", ".fits", ".fts", ".fit.gz", ".fits.gz", ".fts.gz")

_LIGHT_TYPES = {"light frame", "light", "object", "science"}


def iter_fits_files(root: str | Path) -> list[Path]:
    """All FITS files below ``root`` (recursive, sorted, no hidden directories)."""
    root = Path(root)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.lower().endswith(FITS_SUFFIXES):
                found.append(Path(dirpath) / name)
    return found


def manifest_from_directory(
    root: str | Path,
    *,
    run_from_top_folder: bool = True,
) -> Table:
    """Manifest of every FITS file below ``root``.

    The first path component below ``root`` is used as the run name (the
    archive's own layout); frames directly in ``root`` get the folder name.
    Header light frames get the role ``target``, everything else
    ``calibration``; the real frame type is decided by the classification
    step, not here. Unreadable files are skipped.
    """
    root = Path(root)
    rows: list[dict[str, object]] = []
    for index, path in enumerate(iter_fits_files(root)):
        try:
            header = fits.getheader(path)
        except (OSError, ValueError):
            continue
        relative = path.relative_to(root)
        run = relative.parts[0] if run_from_top_folder and len(relative.parts) > 1 else root.name
        row: dict[str, object] = {
            "frame_id": f"L{index:06d}",
            "run": run,
            "file_name": path.name,
            "size": path.stat().st_size,
            "local_path": str(path),
            "source_path": str(relative),
            "downloaded": True,
        }
        apply_header(row, header)
        imagetyp = str(row.get("imagetyp", "")).strip().lower()
        row["role"] = ROLE_TARGET if imagetyp in _LIGHT_TYPES else ROLE_CALIBRATION
        rows.append(row)
    return manifest_from_rows(rows)


__all__ = ["FITS_SUFFIXES", "iter_fits_files", "manifest_from_directory"]
