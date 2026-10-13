"""
Image subtraction: HOTPANTS (optional binary) or a Python Alard–Lupton kernel.

``subtract_science_template`` is the public entry. ``backend="auto"`` uses
HOTPANTS when the executable is on ``PATH``, otherwise Alard–Lupton
(numpy/scipy/photutils — no extra stack).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty

from .. import terminal_output

SubtractBackend = Literal["auto", "hotpants", "alard_lupton"]


def image_gain_from_ccd(ccd: CCDData) -> float | None:
    """Electrons per image unit: ``EXPTIME`` for e⁻/s, 1 for e⁻, ``EGAIN`` for ADU.

    ``None`` when the unit or the needed header keyword is unknown.
    """
    unit = getattr(ccd, "unit", None)
    header = getattr(ccd, "header", None) or {}
    if unit is None:
        return None
    try:
        if unit.is_equivalent(u.electron / u.s):
            exptime = header.get("EXPTIME", header.get("EXPOSURE"))
            return float(exptime) if exptime else None
        if unit.is_equivalent(u.electron):
            return 1.0
        if unit.is_equivalent(u.adu):
            gain = header.get("EGAIN", header.get("GAIN"))
            return float(gain) if gain else None
    except (TypeError, ValueError):
        return None
    return None


def _science_sigma(ccd: CCDData) -> np.ndarray | None:
    unc = getattr(ccd, "uncertainty", None)
    if unc is None:
        return None
    return np.asarray(unc.represent_as(StdDevUncertainty).array, dtype=float)


def _attach_uncertainty(diff_path: Path, sigma: np.ndarray) -> None:
    """Append σ as the ``UNCERT`` extension (CCDData convention)."""
    with fits.open(diff_path, mode="update") as hdul:
        if "UNCERT" in hdul:
            del hdul["UNCERT"]
        hdr = fits.Header()
        hdr["UTYPE"] = "StdDevUncertainty"
        hdul.append(fits.ImageHDU(np.asarray(sigma, dtype=np.float32), header=hdr, name="UNCERT"))


def resolve_subtract_backend(
    backend: SubtractBackend | str | None = "auto",
    hotpants_executable: str | None = None,
) -> str:
    """``hotpants`` or ``alard_lupton``. ``auto`` prefers HOTPANTS when present."""
    name = (backend or "auto").strip().lower()
    if name in ("alard_lupton", "alard-lupton", "python"):
        return "alard_lupton"
    exe = hotpants_executable or shutil.which("hotpants")
    if name == "hotpants":
        if not exe:
            raise RuntimeError(
                "subtract backend is hotpants but the executable was not found"
            )
        return "hotpants"
    if exe:
        return "hotpants"
    return "alard_lupton"


def subtract_science_template(
    science_ccd: CCDData,
    template_hdu: fits.ImageHDU | fits.PrimaryHDU,
    *,
    workdir: str | Path,
    output_filename: str = "diff.fits",
    backend: str = "auto",
    template_mask: np.ndarray | None = None,
    image_gain: float | None = None,
    template_gain: float | None = None,
    hotpants_executable: str | None = None,
    extra_args: Sequence[str] | None = None,
    star_xy: np.ndarray | None = None,
) -> Path:
    """
    Science minus PSF-matched template. Dispatches to HOTPANTS or Alard–Lupton.

    ``image_gain`` (e⁻ per image unit) defaults to :func:`image_gain_from_ccd`.
    With a science uncertainty the difference FITS carries its σ as
    ``UNCERT`` (template noise not included).
    """
    resolved = resolve_subtract_backend(backend, hotpants_executable)
    if resolved == "hotpants":
        return run_hotpants(
            science_ccd,
            template_hdu,
            workdir=workdir,
            output_filename=output_filename,
            template_mask=template_mask,
            image_gain=image_gain,
            template_gain=template_gain,
            hotpants_executable=hotpants_executable,
            extra_args=extra_args,
        )
    from .subtraction_alard_lupton import run_alard_lupton

    terminal_output.print_to_terminal(
        "Image subtraction backend: Alard–Lupton (Python)",
        indent=2,
        style_name="NORMAL",
    )
    return run_alard_lupton(
        science_ccd,
        template_hdu,
        workdir=workdir,
        output_filename=output_filename,
        star_xy=star_xy,
    )


def run_hotpants(
    science_ccd: CCDData,
    template_hdu: fits.ImageHDU | fits.PrimaryHDU,
    *,
    workdir: str | Path,
    output_filename: str = "diff.fits",
    template_mask: np.ndarray | None = None,
    image_gain: float | None = None,
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
        Electrons per image unit, passed as ``-ig`` / ``-tg``. ``image_gain``
        defaults to :func:`image_gain_from_ccd` (``EXPTIME`` for e⁻/s
        images); ``-tg`` only when given. The science uncertainty (σ) is
        passed as ``-ini``; the difference σ (``-oni``) is stored as the
        ``UNCERT`` extension of the output.
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

    # float64 in native byte order (FITS data are big-endian)
    tdata = np.array(template_hdu.data, dtype=np.float64)
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

    sigma = _science_sigma(science_ccd)
    noise_out = None
    if sigma is not None:
        npath = work / "_hotpants_inoise.fits"
        fits.PrimaryHDU(sigma.astype(np.float32)).writeto(npath, overwrite=True)
        noise_out = work / "_hotpants_onoise.fits"
        cmd.extend(["-ini", str(npath), "-oni", str(noise_out)])

    gain = image_gain if image_gain is not None else image_gain_from_ccd(science_ccd)
    if gain is None:
        terminal_output.print_to_terminal(
            "HOTPANTS: image gain unknown (unit/EXPTIME/EGAIN); using 1 e⁻ per unit"
            + ("" if sigma is not None else " for its noise model"),
            style_name="WARNING",
        )
        gain = 1.0
    cmd.extend(["-ig", str(float(gain))])
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
    if noise_out is not None and noise_out.is_file():
        _attach_uncertainty(out_path, fits.getdata(noise_out))

    return out_path
