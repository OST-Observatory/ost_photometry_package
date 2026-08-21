"""Observation session container and factory."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, name_resolve
from astropy.table import Table
from photutils.psf import ImagePSF

from .. import terminal_output
from .models import ImageSeries, ObjectOfInterest
from .ooi_ids import ooi_photometry_ids


class Observation:
    """Container for data taken during an observation session."""

    objects_of_interest: list[ObjectOfInterest]
    image_series_dict: dict[str, ImageSeries]
    objects_of_interest_coordinates: SkyCoord | None
    table_magnitudes: Table | None

    def __init__(self, **kwargs):
        """ Backward-compatible constructor delegating to :meth:`from_config`."""
        if kwargs:
            built = self.from_config(**kwargs)
            self.objects_of_interest = built.objects_of_interest
            self.image_series_dict = built.image_series_dict
            self.objects_of_interest_coordinates = built.objects_of_interest_coordinates
            self.table_magnitudes = built.table_magnitudes
        else:
            self.objects_of_interest = []
            self.image_series_dict = {}
            self.objects_of_interest_coordinates = None
            self.table_magnitudes = None

    @classmethod
    def from_config(cls, **kwargs) -> Observation:
        """Build an observation from explicit keyword parameters (no ``__dict__.update``)."""
        ra_objects: list[str] | None = kwargs.get("ra_objects")
        ra_unit: str | None = kwargs.get("ra_unit")
        dec_objects: list[str] | None = kwargs.get("dec_objects")
        dec_unit: str | None = kwargs.get("dec_unit")
        object_names: list[str] | None = kwargs.get("object_names")
        periods: list[float] | None = kwargs.get("periods")
        transit_times: list[str] | None = kwargs.get("transit_times")
        resolve_names: bool = kwargs.get("resolve_names", True)

        obs = object.__new__(cls)
        objects: list[ObjectOfInterest] = []
        add_periods = bool(periods and transit_times)

        if all([ra_objects, dec_objects, ra_unit, dec_unit, object_names]):
            len_names = len(object_names)
            if len_names == len(ra_objects) and len_names == len(dec_objects):
                for i, (name, ra, dec) in enumerate(
                    zip(object_names, ra_objects, dec_objects, strict=True)
                ):
                    ooi = ObjectOfInterest(ra, dec, ra_unit, dec_unit, name)
                    if add_periods:
                        ooi.period = periods[i]
                        ooi.transit_time = transit_times[i]
                    objects.append(ooi)
        elif object_names is not None:
            for i, name in enumerate(object_names):
                if resolve_names:
                    try:
                        sky_coordinates = SkyCoord.from_name(name)
                        ooi = ObjectOfInterest(
                            sky_coordinates.ra.degree,
                            sky_coordinates.dec.degree,
                            u.degree,
                            u.degree,
                            name,
                        )
                    except name_resolve.NameResolveError:
                        ooi = ObjectOfInterest(None, None, None, None, name)
                else:
                    ooi = ObjectOfInterest(None, None, None, None, name)
                if add_periods:
                    ooi.period = periods[i]
                    ooi.transit_time = transit_times[i]
                objects.append(ooi)

        obs.objects_of_interest = objects
        obs.image_series_dict = dict(kwargs.get("image_series_dict") or {})
        obs.table_magnitudes = kwargs.get("table_magnitudes")
        if objects:
            ra_list = [o.ra for o in objects]
            dec_list = [o.dec for o in objects]
            obs.objects_of_interest_coordinates = SkyCoord(
                ra_list, dec_list, unit=(u.degree, u.degree), frame="icrs"
            )
        else:
            obs.objects_of_interest_coordinates = None
        return obs

    def get_epsf(self) -> dict[str, list[ImagePSF]]:
        epsf_dict: dict[str, list[ImagePSF]] = {}
        for key, image_series in self.image_series_dict.items():
            epsf_dict[key] = [img.epsf for img in image_series.image_list]
        return epsf_dict

    def get_reference_epsf(self) -> dict[str, list[ImagePSF]]:
        epsf_dict: dict[str, list[ImagePSF]] = {}
        for key, image_series in self.image_series_dict.items():
            ref_idx = image_series.reference_image_index
            img = image_series.image_list[ref_idx]
            epsf_dict[key] = [img.epsf]
        return epsf_dict

    def get_reference_image(self) -> dict[str, np.ndarray]:
        img_dict: dict[str, np.ndarray] = {}
        for key, image_series in self.image_series_dict.items():
            ref_idx = image_series.reference_image_index
            img = image_series.image_list[ref_idx]
            img_dict[key] = img.get_data()
        return img_dict

    def get_reference_image_residual(self) -> dict[str, np.ndarray]:
        img_dict: dict[str, np.ndarray] = {}
        for key, image_series in self.image_series_dict.items():
            ref_idx = image_series.reference_image_index
            img = image_series.image_list[ref_idx]
            if img.residual_image is not None:
                img_dict[key] = img.residual_image
        return img_dict

    def get_image_series(self, filter_list: list[str] | set[str]) -> dict[str, ImageSeries]:
        return {f: self.image_series_dict[f] for f in filter_list}

    def get_ids_object_of_interest(
        self,
        filter_: str | None = None,
        reference_image_series_id: int | None = None,
    ) -> list[int]:
        if filter_ is None and reference_image_series_id is None:
            terminal_output.print_to_terminal(
                "Neither a filter nor an image series ID was provided to "
                "compile the IDs for the objects of interest. The image series ID "
                "is assumed to be 0.",
                style_name="WARNING",
            )
            reference_image_series_id = 0

        return ooi_photometry_ids(
            self.objects_of_interest,
            filter_=filter_,
            reference_image_series_id=reference_image_series_id,
        )

    def get_object_of_interest_names(self) -> list[str]:
        return [object_.name for object_ in self.objects_of_interest]

    def get_object_ras(self) -> list[float]:
        return [object_.ra for object_ in self.objects_of_interest]

    def get_object_decs(self) -> list[float]:
        return [object_.dec for object_ in self.objects_of_interest]
