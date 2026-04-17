"""Fetch a HiPS archival cutout and subtract it from science with HOTPANTS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData
from astroquery.hips2fits import hips2fitsClass

from ... import checks, terminal_output
from ...utilities import get_basename
from .. import plots, subtraction
from ..models import ImageSeries
from ..utilities import find_wcs

DEFAULT_HIPS_SERVER = (
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
)

# Optional fixed trim matching an older one-off workflow (cropped 1599×2501).
LEGACY_SUBTRACT_TRIM: tuple[int, int, int, int] = (0, 1599, 0, 2501)


@dataclass(frozen=True)
class HipsReferenceSubtractResult:
    work_dir: Path
    difference_fits: Path
    hips_fits: Path
    science_fits_path: Path


def run_hips_reference_subtraction(
    filter_: str,
    science_image_path: str,
    work_output_dir: str | Path,
    *,
    wcs_method: str = "astrometry",
    plot_comp: bool = True,
    hips_source: str = "CDS/P/DSS2/blue",
    file_type_plots: str = "pdf",
    trim_slice_yx: tuple[int, int, int, int] | None = None,
    reuse_wcs_image_series: ImageSeries | None = None,
    hips_timeout_ms: int = 120_000,
    hips_server: str = DEFAULT_HIPS_SERVER,
    hips_verbose: bool = False,
    hotpants_executable: str | None = None,
    hotpants_extra_args: Sequence[str] | None = None,
    hotpants_output_filename: str = "hotpants_diff.fits",
    image_gain: float = 1.0,
    template_gain: float | None = None,
) -> HipsReferenceSubtractResult:
    """
    Trim (optional), solve or reuse WCS, query HiPS2FITS, optionally plot, run HOTPANTS.

    Parameters
    ----------
    trim_slice_yx
        If set, ``(y0, y1, x0, x1)`` slice applied as ``ccd[y0:y1, x0:x1]`` before
        WCS and HiPS. Pipeline default is ``None`` (full frame). Use
        `LEGACY_SUBTRACT_TRIM` for the old fixed crop.
    reuse_wcs_image_series
        When ``trim_slice_yx`` is ``None`` and this series has ``.wcs`` set (e.g. after
        the pipeline WCS step), that WCS is reused and ``find_wcs`` is skipped.
    """
    workdir = Path(work_output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    checks.check_file(science_image_path)

    ccd_image = CCDData.read(science_image_path)

    if trim_slice_yx is not None:
        y0, y1, x0, x1 = trim_slice_yx
        ccd_image = ccdp.trim_image(ccd_image[y0:y1, x0:x1])
        trimmed_name = f"{get_basename(science_image_path)}_trimmed.fit"
        path_for_series = workdir / trimmed_name
        ccd_image.write(path_for_series, overwrite=True)
        path_for_series = str(path_for_series.resolve())
    else:
        path_for_series = str(Path(science_image_path).resolve())

    can_reuse_wcs = (
        trim_slice_yx is None
        and reuse_wcs_image_series is not None
        and getattr(reuse_wcs_image_series, "wcs", None) is not None
    )

    if can_reuse_wcs:
        wcs_obj = reuse_wcs_image_series.wcs
    else:
        #   TODO: Extend find_wcs function to work with Image objects as well as ImageSeries objects
        tmp_series = ImageSeries(filter_, path_for_series, str(workdir))
        find_wcs(
            tmp_series,
            reference_image_index=0,
            method=wcs_method,
            indent=2,
        )
        wcs_obj = tmp_series.wcs

    hips_instance = hips2fitsClass()
    hips_instance.timeout = hips_timeout_ms
    hips_instance.server = hips_server

    terminal_output.print_to_terminal(
        f"HiPS2FITS query ({hips_source}) …",
        indent=2,
        style_name="INFO",
    )
    hips_hdus = hips_instance.query_with_wcs(
        hips=hips_source,
        wcs=wcs_obj,
        get_query_payload=False,
        format="fits",
        verbose=hips_verbose,
    )

    hips_path = workdir / "hips.fits"
    hips_hdus.writeto(str(hips_path), overwrite=True)

    if plot_comp:
        sci_plot = np.asarray(ccd_image.data, dtype=np.float64)
        ref_plot = np.asarray(hips_hdus[0].data, dtype=np.float64)
        plots.compare_images(
            str(workdir),
            SimpleNamespace(data=sci_plot),
            SimpleNamespace(data=ref_plot),
            file_type=file_type_plots,
        )

    diff_path = subtraction.run_hotpants(
        ccd_image,
        hips_hdus[0],
        workdir=str(workdir),
        output_filename=hotpants_output_filename,
        template_mask=None,
        image_gain=image_gain,
        template_gain=template_gain,
        hotpants_executable=hotpants_executable,
        extra_args=hotpants_extra_args,
    )

    return HipsReferenceSubtractResult(
        work_dir=workdir,
        difference_fits=diff_path,
        hips_fits=hips_path,
        science_fits_path=Path(path_for_series),
    )
