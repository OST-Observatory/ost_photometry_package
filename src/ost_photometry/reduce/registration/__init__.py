"""Image registration: alignment, shifts, and trim helpers."""

from .align import align_image_main, align_images, make_big_images
from .shift_methods import SHIFT_METHODS, SUPPORTED_SHIFT_METHODS
from .shifts import (
    apply_astro_align,
    apply_optical_flow,
    apply_xy_image_shift,
    astro_align,
    calculate_xy_image_shifts,
    calculate_xy_image_shifts_core,
    optical_flow_align,
    own_image_cross_correlation,
)
from .trim import (
    calculate_index_from_shifts,
    calculate_min_max_image_shifts,
    trim_ccd,
    trim_image,
    trim_image_simple,
)
from .wcs_align import (
    apply_wcs_align,
    celestial_wcs_from_ccd,
    ensure_celestial_wcs_on_fits,
    fits_has_celestial_wcs,
    pixel_offset_on_reference,
    reproject_ccd_onto_wcs,
)

__all__ = [
    "SHIFT_METHODS",
    "SUPPORTED_SHIFT_METHODS",
    "align_image_main",
    "align_images",
    "apply_astro_align",
    "apply_optical_flow",
    "apply_wcs_align",
    "apply_xy_image_shift",
    "astro_align",
    "calculate_index_from_shifts",
    "calculate_min_max_image_shifts",
    "calculate_xy_image_shifts",
    "calculate_xy_image_shifts_core",
    "celestial_wcs_from_ccd",
    "ensure_celestial_wcs_on_fits",
    "fits_has_celestial_wcs",
    "make_big_images",
    "optical_flow_align",
    "own_image_cross_correlation",
    "pixel_offset_on_reference",
    "reproject_ccd_onto_wcs",
    "trim_ccd",
    "trim_image",
    "trim_image_simple",
]
