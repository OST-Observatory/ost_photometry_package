"""Base image container: metadata + on-demand FITS access."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time
from regions import PixCoord, RectanglePixelRegion

from . import calibration_parameters, style, terminal_output


class Image:
    """
    Image class: metadata + on-demand FITS access (no persistent pixel cache).

    Use ``with image.open() as ccd:`` inside a pipeline step to avoid repeated
    disk reads without holding all images in RAM across the full run.

    Analysis-only fields (photometry tables, ePSF, …) live on
    :class:`~ost_photometry.analyze.image.AnalysisImage`.
    """

    def __init__(
        self, image_id: int, filter_: str, path: str | Path, output_dir: str | Path
    ) -> None:
        self.image_id: int = image_id
        self.filter_: str = filter_
        if isinstance(path, Path):
            self.filename: str = path.name
            self.path: Path = path
        else:
            self.filename = path.split("/")[-1]
            self.path = Path(path)
        if isinstance(output_dir, Path):
            self.out_path: Path = output_dir
        else:
            self.out_path = Path(output_dir)

        self.wcs: wcs.WCS | None = None
        self.instrument: str | None = None
        self.field_of_view_y: float | None = None
        self.field_of_view_x: float | None = None
        self.coordinates_image_center: SkyCoord | None = None
        self.pixel_scale: float | None = None
        self.fov_pixel_region: RectanglePixelRegion | None = None
        self.air_mass: float | None = None
        self.jd: float | None = None
        self.fwhm: float = 4.0

        self._header_cache: fits.Header | None = None
        self._active_ccd: CCDData | None = None
        self.ensure_metadata()

    def ensure_metadata(self) -> None:
        """Load header-derived metadata once (small memory footprint)."""
        if self.field_of_view_x is None:
            self.calculate_field_of_view_etc()

    @contextmanager
    def open(self, memmap: bool = True):
        """Short-lived FITS session for a processing step (released on exit)."""
        ccd = CCDData.read(self.path, memmap=memmap)
        self._active_ccd = ccd
        try:
            yield ccd
        finally:
            self._active_ccd = None

    def read_image(self, memmap: bool = True) -> CCDData:
        return self._read_ccd(memmap=memmap)

    def get_header(self) -> fits.Header:
        if self._header_cache is None:
            self._header_cache = fits.getheader(self.path)
        return self._header_cache

    def _read_ccd(self, memmap: bool = True) -> CCDData:
        if self._active_ccd is not None:
            return self._active_ccd
        return CCDData.read(self.path, memmap=memmap)

    def get_data(self) -> np.ndarray:
        return self._read_ccd().data

    def get_error(self) -> np.ndarray:
        ccd = self._read_ccd()
        if ccd.uncertainty is None:
            raise ValueError(f"No uncertainty extension in {self.path}")
        return ccd.uncertainty.array

    def get_mask(self) -> np.ndarray | None:
        return self._read_ccd().mask

    def get_shape(self) -> tuple[int, int]:
        if self._active_ccd is not None:
            return self._active_ccd.data.shape
        header = self.get_header()
        return int(header.get("NAXIS1", 0)), int(header.get("NAXIS2", 0))

    def calculate_field_of_view_etc(self) -> None:
        #   Get header
        header = self.get_header()

        #   Read focal length - set default to 3454. mm
        focal_length = header.get("FOCALLEN", 3454.0)

        #   Read ra and dec of image center
        ra = header.get("OBJCTRA", "00 00 00")
        dec = header.get("OBJCTDEC", "+00 00 00")

        #   Convert ra & dec to degrees
        coordinates_sky = SkyCoord(
            ra,
            dec,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )

        #   Number of pixels
        n_pixel_x = header.get("NAXIS1", 0)
        n_pixel_y = header.get("NAXIS2", 0)

        if n_pixel_x == 0:
            raise ValueError(
                f"{style.Bcolors.FAIL}\nException in calculate_field_of_view(): X "
                f"dimension of the image is 0 {style.Bcolors.ENDC}"
            )
        if n_pixel_y == 0:
            raise ValueError(
                f"{style.Bcolors.FAIL}\nException in calculate_field_of_view(): Y "
                f"dimension of the image is 0 {style.Bcolors.ENDC}"
            )

        #   Get binning
        x_binning = header.get("XBINNING", 1)
        y_binning = header.get("YBINNING", 1)

        #   Set instrument
        instrument = header.get("INSTRUME", "")

        if instrument in ["QHYCCD-Cameras-Capture", "QHYCCD-Cameras2-Capture"]:
            #   Physical chip dimensions in pixel
            physical_dimension_x = n_pixel_x * x_binning
            physical_dimension_y = n_pixel_y * y_binning

            #   Set instrument
            if physical_dimension_x == 9576 and physical_dimension_y in [6387, 6388]:
                instrument = "QHY600M"
            elif physical_dimension_x in [6280, 6279] and physical_dimension_y in [
                4210,
                4209,
            ]:
                instrument = "QHY268M"
            elif physical_dimension_x == 3864 and physical_dimension_y in [2180, 2178]:
                instrument = "QHY485C"
            else:
                instrument = ""

        #   Calculate chip size in mm
        if "XPIXSZ" in header:
            pixel_width = header["XPIXSZ"]
            chip_length = n_pixel_x * float(pixel_width) / 1000
            chip_height = n_pixel_y * float(pixel_width) / 1000
        elif "PIXSIZE1" in header:
            pixel_width = header["PIXSIZE1"]
            chip_length = n_pixel_x * float(pixel_width) / 1000
            chip_height = n_pixel_y * float(pixel_width) / 1000
        else:
            terminal_output.print_to_terminal(
                "Warning chip dimension could not be determined from Header. "
                "Use default values, assuming the image has not been cropped. "
                "This may be completely wrong. ",
                indent=1,
                style_name="WARNING",
            )
            chip_length, chip_height = calibration_parameters.get_chip_dimensions(
                instrument
            )

        #   Calculate field of view
        field_of_view_x = 2 * np.arctan(chip_length / 2 / focal_length)
        field_of_view_y = 2 * np.arctan(chip_height / 2 / focal_length)

        #   Convert to arc min
        field_of_view_x = field_of_view_x * 360.0 / 2.0 / np.pi * 60.0
        field_of_view_y = field_of_view_y * 360.0 / 2.0 / np.pi * 60.0

        #   Calculate pixel scale [arcsec/pixel]
        pixel_scale = field_of_view_x * 60 / n_pixel_x

        #   Create RectanglePixelRegion that covers the field of view
        pixel_region = RectanglePixelRegion(
            center=PixCoord(x=int(n_pixel_x / 2), y=int(n_pixel_y / 2)),
            width=n_pixel_x,
            height=n_pixel_y,
        )

        #   Add to image class
        self.coordinates_image_center = coordinates_sky
        self.field_of_view_x = field_of_view_x
        self.field_of_view_y = field_of_view_y
        self.instrument = instrument
        self.pixel_scale = pixel_scale
        self.fov_pixel_region = pixel_region

        #   Add JD (observation time) and air mass from Header to image class
        jd = header.get("JD", None)
        if jd is None:
            obs_time = header.get("DATE-OBS", None)
            if not obs_time:
                raise ValueError(
                    f"{style.Bcolors.FAIL} \tERROR: No information about the "
                    "observation time was found in the header"
                    f"{style.Bcolors.ENDC}"
                )
            jd = Time(obs_time, format="fits").jd

        self.jd = jd
        self.air_mass = header.get("AIRMASS", 1.0)

        #  Add instrument to image class
        self.instrument = instrument
