"""Canonical ``shift_method`` names for image registration."""

from __future__ import annotations

SHIFT_METHODS: dict[str, str] = {
    "aa_true": "astroalign similarity (default; dense fields / sub-pixel stacks)",
    "wcs": "reproject onto the reference celestial WCS (solve if missing)",
    "aa": "astroalign translation only",
    "skimage": "skimage phase correlation (translation)",
    "own": "in-house FFT phase correlation (slow; prefer skimage)",
    "flow": "optical-flow TV-L1 (not recommended for stacking)",
}

SUPPORTED_SHIFT_METHODS = tuple(SHIFT_METHODS)

__all__ = ["SHIFT_METHODS", "SUPPORTED_SHIFT_METHODS"]
