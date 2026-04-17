"""
Image subtraction via the external `hotpants` program (HOTPANTS / Aaron Roodman style).

Requires a `hotpants` binary on ``PATH`` or passed explicitly. FITS inputs must carry
compatible WCS headers where hotpants expects them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData

from .. import terminal_output


def run_hotpants(
    science_ccd: CCDData,
    template_hdu: fits.ImageHDU | fits.PrimaryHDU,
    *,
    workdir: str | Path,
    output_filename: str = "hotpants_diff.fits",
    template_mask: np.ndarray | None = None,
    image_gain: float = 1.0,
    template_gain: float | None = None,
    hotpants_executable: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> Path:
    """
    Run HOTPANTS image subtraction: science minus convolved-matched template.

    Writes temporary FITS files under ``workdir`` and invokes::

        hotpants -inim <sci> -tmplim <tmpl> -outim <out> [-imi ...] [-tmpl ...] ...

    Parameters
    ----------
    science_ccd
        Science exposure including WCS in header; mask used for ``-imi`` if present.
    template_hdu
        Template (e.g. HiPS cutout) HDU with data and WCS header.
    workdir
        Working directory for intermediate FITS and output.
    output_filename
        Name of the output difference image (written under ``workdir``).
    template_mask
        Boolean array, same shape as template; ``True`` = bad pixel for ``-tmpl``.
    image_gain, template_gain
        Passed as ``-ig`` / ``-tg`` when template_gain is not ``None``.
    hotpants_executable
        Path to ``hotpants``; default ``shutil.which("hotpants")``.
    extra_args
        Additional CLI tokens inserted after mask options and before ``-ig`` /
        ``-outim`` (e.g. ``["-n", "i", "5", "5"]``).

    Returns
    -------
    Path
        Path to the subtracted FITS written by hotpants.
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    exe = hotpants_executable or shutil.which("hotpants")
    if not exe:
        raise RuntimeError(
            "hotpants executable not found (not on PATH). Install HOTPANTS or pass "
            "hotpants_executable=... to run_hotpants()."
        )

    sci_path = work / "_hotpants_inim.fits"
    tmpl_path = work / "_hotpants_tmplim.fits"
    out_path = work / output_filename

    science_ccd.write(sci_path, overwrite=True)

    tdata = np.asarray(template_hdu.data, dtype=np.float64)
    if hasattr(tdata, "byteswap"):
        tdata = tdata.byteswap().newbyteorder("=")
    thdr = template_hdu.header.copy()
    fits.PrimaryHDU(data=tdata, header=thdr).writeto(tmpl_path, overwrite=True)

    cmd: list[str] = [str(exe), "-inim", str(sci_path)]

    sci_mask = getattr(science_ccd, "mask", None)
    if sci_mask is not None and np.any(sci_mask):
        mpath = work / "_hotpants_imask.fits"
        bad = np.asarray(sci_mask, dtype=np.uint8)
        fits.PrimaryHDU(bad).writeto(mpath, overwrite=True)
        cmd.extend(["-imi", str(mpath)])

    cmd.extend(["-tmplim", str(tmpl_path)])

    if template_mask is not None and np.any(template_mask):
        mpath_t = work / "_hotpants_tmask.fits"
        bad_t = np.asarray(template_mask, dtype=np.uint8)
        fits.PrimaryHDU(bad_t).writeto(mpath_t, overwrite=True)
        cmd.extend(["-tmpl", str(mpath_t)])

    if extra_args:
        cmd.extend(list(extra_args))

    cmd.extend(["-ig", str(float(image_gain))])
    if template_gain is not None:
        cmd.extend(["-tg", str(float(template_gain))])
    cmd.extend(["-outim", str(out_path)])

    terminal_output.print_to_terminal(
        f"Running hotpants → {out_path.name}",
        style_name="INFO",
    )
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or "(no output)"
        raise RuntimeError(
            f"hotpants failed (exit {proc.returncode}): {msg[:2000]}"
        )

    if not out_path.is_file():
        raise RuntimeError(f"hotpants reported success but output missing: {out_path}")

    return out_path
