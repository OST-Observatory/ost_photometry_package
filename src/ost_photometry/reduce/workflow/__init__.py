"""Reduction workflow implementation."""

from .bias import master_bias
from .config import ReduceConfig
from .constants import (
    REDUCE_STATUS_REDUCED,
    REDUCE_STATUS_SKIP_NO_FILTER,
    REDUCE_STATUS_SKIP_NO_MASTER_FLAT,
)
from .dark import master_dark, master_dark_stacking, reduce_dark, reduce_dark_image
from .dispatch import master_image_list, reduce_master
from .flat import master_flat, reduce_flat, reduce_flat_image, stack_flat_images
from .main import reduce_main
from .science import reduce_light, reduce_light_image
from .stack import stack_filter_images, stack_image

__all__ = [
    "REDUCE_STATUS_REDUCED",
    "REDUCE_STATUS_SKIP_NO_FILTER",
    "REDUCE_STATUS_SKIP_NO_MASTER_FLAT",
    "ReduceConfig",
    "master_bias",
    "master_dark",
    "master_dark_stacking",
    "master_flat",
    "master_image_list",
    "reduce_dark",
    "reduce_dark_image",
    "reduce_flat",
    "reduce_flat_image",
    "reduce_light",
    "reduce_light_image",
    "reduce_main",
    "reduce_master",
    "stack_filter_images",
    "stack_flat_images",
    "stack_image",
]
