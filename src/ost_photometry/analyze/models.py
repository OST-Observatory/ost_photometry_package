"""Data models for photometry analysis: ObjectOfInterest, ImageSeries."""

import os
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from regions import RectanglePixelRegion

from .. import style, terminal_output
from .image import AnalysisImage


def _air_mass_values(image_list: list[AnalysisImage]) -> list[float]:
    """Collect finite air-mass values; raise if none are available."""
    values: list[float] = []
    for img in image_list:
        am = getattr(img, "air_mass", None)
        if am is None:
            continue
        values.append(float(am))
    if not values:
        raise ValueError("No air mass values available in image series")
    return values


class ObjectOfInterest:
    """Represents an astronomical object of interest with sky coordinates."""

    def __init__(
        self,
        ra: str | float | None,
        dec: str | float | None,
        ra_unit: str | u.quantity.Quantity | None,
        dec_unit: str | u.quantity.Quantity | None,
        name: str,
    ):
        #   Set sky coordinates object
        self.coordinates_object = SkyCoord(
            ra=ra, dec=dec, unit=(ra_unit, dec_unit), frame="icrs"
        )

        #   Set right ascension
        if self.coordinates_object.ra is not None:
            self.ra = self.coordinates_object.ra.degree

        #   Set declination
        if self.coordinates_object.dec is not None:
            self.dec = self.coordinates_object.dec.degree

        #   Set object_name
        self.name = name

        #   ID of object in the image series
        #   Syntax: {'filter': 'id'}
        self.id_in_image_series: dict[str, int] = {}

        #   Set defaults for period and transit time
        self.transit_time: str | None = None
        self.period: float | None = None


class ImageSeries:
    """
    Image series class: Used to handle a series of images,
                        e.g. taken with a specific filter.
    """

    def __init__(
        self, filter_: str, path: str, output_dir: str, reference_image_index: int = 0
    ):
        #   Setup file list
        if os.path.isdir(path):
            formats: list[str] = [".FIT", ".fit"]
            file_list = os.listdir(path)

            #   Remove not FITS entries
            temp_list: list[str] = []
            for file_i in file_list:
                for j, form in enumerate(formats):
                    if file_i.find(form) != -1:
                        temp_list.append(file_i)
            file_list = temp_list

            #   Sort file list
            file_list.sort(key=lambda x: int(x.split("_")[0]))
        elif os.path.isfile(path):
            file_list = [str(path).split("/")[-1]]
            path = os.path.dirname(path)
        else:
            raise RuntimeError(
            "ERROR: Provided path is neither a file nor a directory"
        )

        #   Add file list
        self.file_list: list[str] = file_list

        #   Check if any image was detected
        if len(self.file_list) <= 0:
            raise ValueError(
                f"{style.Bcolors.FAIL} ERROR: No FITS image detected in "
                f"{path}! -> EXIT {style.Bcolors.ENDC}"
            )

        #   Check if the id of the reference image is valid
        if reference_image_index > len(self.file_list):
            raise ValueError(
                f"{style.Bcolors.FAIL} ERROR: Reference image index "
                "[reference_image_index] is larger than the total number of "
                f"images! -> EXIT {style.Bcolors.ENDC}"
            )

        #   Set filter
        self.filter_: str = filter_

        #   Set list index of the reference image in image_list
        self.reference_image_index: int = reference_image_index

        #   Prepare image list
        self.image_list: list[AnalysisImage] = []

        #   Set path to output directory
        self.out_path: Path = Path(output_dir)

        #   Fill image list
        terminal_output.print_to_terminal(
            "Read images and calculate field of view, pixel scale, etc. ... ",
            indent=2,
        )
        for image_id, file_name in enumerate(file_list):
            self.image_list.append(
                AnalysisImage(image_id, filter_, f"{path}/{file_name}", output_dir)
            )

        #   Set start time for image series
        if len(self.image_list) > 0:
            self.start_jd: float | None = self.image_list[0].jd
        else:
            self.start_jd: float | None = None

        #   Set reference image
        self.reference_image = self.image_list[reference_image_index]

        #   Set field of view
        self.field_of_view_x: float | None = getattr(
            self.reference_image,
            "field_of_view_x",
            None,
        )

        #   Set PixelRegion for the field of view
        self.fov_pixel_region: RectanglePixelRegion | None = getattr(
            self.reference_image,
            "fov_pixel_region",
            None,
        )

        #   Set pixel scale
        self.pixel_scale: float | None = getattr(
            self.reference_image,
            "pixel_scale",
            None,
        )

        #   Set coordinates of image center
        self.coordinates_image_center = getattr(
            self.reference_image,
            "coordinates_image_center",
            None,
        )

        #   Set instrument
        self.instrument: str | None = getattr(
            self.reference_image,
            "instrument",
            None,
        )

        #   Get image shape
        self.image_shape = self.reference_image.get_data().shape

        #   Set wcs default
        self.wcs: wcs.WCS | None = None

    def set_wcs(self, w: wcs.WCS) -> None:
        from ost_photometry.wcs import sync_image_coordinates_from_wcs

        self.wcs = w
        for img in self.image_list:
            img.wcs = w
            sync_image_coordinates_from_wcs(img, w)

    def get_photometry(self) -> dict[str, object]:
        from astropy.table import Table

        photo_dict: dict[str, object] = {}
        for img in self.image_list:
            photo_dict[str(img.image_id)] = getattr(img, "photometry", None)
        return photo_dict

    def get_image_ids(self) -> list[int]:
        img_ids: list[int] = []
        for img in self.image_list:
            img_ids.append(img.image_id)
        return img_ids

    def mean_sigma_clip_air_mass(self) -> float:
        return sigma_clipped_stats(_air_mass_values(self.image_list), sigma=1.5)[0]

    def median_air_mass(self) -> np.floating:
        return np.median(_air_mass_values(self.image_list))

    def get_air_mass(self) -> list[float | None]:
        return [getattr(img, "air_mass", None) for img in self.image_list]

    def get_observation_time(self) -> np.ndarray:
        obs_time_list: list[float] = []
        for img in self.image_list:
            obs_time_list.append(getattr(img, "jd", 0.0))
        return np.array(obs_time_list)

    def median_observation_time(self) -> np.floating:
        obs_time_list: list[float] = []
        for img in self.image_list:
            obs_time_list.append(getattr(img, "jd", 0.0))
        return np.median(obs_time_list)

    def get_list_dict(self) -> list[dict[str, AnalysisImage]]:
        dict_list: list[dict[str, AnalysisImage]] = []
        for img in self.image_list:
            dict_list.append({img.filter_: img})
        return dict_list

    def get_object_positions_pixel(self) -> tuple[list, list, int]:
        tbl_s = self.get_photometry()
        n_max_list: list[int] = []
        x: list = []
        y: list = []
        for i, tbl in enumerate(tbl_s.values()):
            if tbl is not None:
                x.append(tbl["x_fit"])
                y.append(tbl["y_fit"])
                n_max_list.append(len(x[i]))
        return x, y, np.max(n_max_list)

    def get_flux_distribution(
        self, distribution_samples: int = 1000
    ) -> list:
        from astropy import uncertainty as unc

        tbl_s = list(self.get_photometry().values())
        flux_list: list = []
        for tbl in tbl_s:
            if tbl is not None:
                flux_list.append(
                    unc.normal(
                        tbl["flux_fit"] * u.mag,
                        std=tbl["flux_err"] * u.mag,
                        n_samples=distribution_samples,
                    )
                )
        return flux_list

    def get_flux_array(self) -> tuple[np.ndarray, np.ndarray]:
        tbl_s = list(self.get_photometry().values())
        n_images = len(tbl_s)
        n_objects = len(tbl_s[0])
        flux = np.zeros((n_images, n_objects))
        flux_err = np.zeros((n_images, n_objects))
        for i, tbl in enumerate(tbl_s):
            if tbl is not None:
                flux[i] = tbl["flux_fit"]
                flux_err[i] = tbl["flux_err"]
        return flux, flux_err
