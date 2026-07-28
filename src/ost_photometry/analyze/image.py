"""Analysis image: base :class:`~ost_photometry.image.Image` plus photometry state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..image import Image

if TYPE_CHECKING:
    from astropy.table import Table
    from photutils.psf import ImagePSF


class AnalysisImage(Image):
    """
    Image used in the analysis pipeline.

    Adds extraction / calibration state (photometry tables, ePSF, residuals,
    zero-point samples) on top of the shared FITS/metadata container.
    """

    def __init__(
        self,
        image_id: int,
        filter_: str,
        path: str | Path,
        output_dir: str | Path,
    ) -> None:
        super().__init__(image_id, filter_, path, output_dir)
        self.epsf: ImagePSF | None = None
        self.residual_image: np.ndarray | None = None
        self.photometry: Table | None = None
        self.positions: Table | None = None
        self.zp: np.ndarray | None = None
