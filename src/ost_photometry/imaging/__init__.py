"""Colour images from the per-filter stacks (RGB, LRGB, SHO, HOO, custom mixes).

The stacks of one target share one pixel grid (both pipelines register all
filters onto one reference). :func:`compose` mixes them into R / G / B,
neutralises the sky, optionally removes gradients and balances the colours
on the stars, stretches and writes TIFF / PNG / JPEG (:mod:`.export`).
"""

from .composite import ColorResult, ColorSettings, compose, load_channels
from .discovery import stacks_from_directory, stacks_from_summary
from .export import write_images
from .filters import canonical_filter
from .palettes import PALETTES, Palette, resolve_palette

__all__ = [
    "PALETTES",
    "ColorResult",
    "ColorSettings",
    "Palette",
    "canonical_filter",
    "compose",
    "load_channels",
    "resolve_palette",
    "stacks_from_directory",
    "stacks_from_summary",
    "write_images",
]
