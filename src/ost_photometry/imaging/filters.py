"""Canonical filter names for colour mixing (header spellings vary)."""

from __future__ import annotations

import re

#: Compact spelling (lower case, no spaces / punctuation) -> canonical name.
_ALIASES = {
    "u": "U", "b": "B", "v": "V", "g": "G", "r": "R", "i": "I",
    "blue": "B", "green": "G", "red": "R",
    "clear": "L", "l": "L", "lum": "L", "luminance": "L", "none": "L", "c": "L",
    "ha": "Ha", "halpha": "Ha", "hydrogenalpha": "Ha",
    "oiii": "OIII", "o3": "OIII",
    "sii": "SII", "s2": "SII",
}

#: Names :func:`canonical_filter` can return.
CANONICAL_FILTERS = ("U", "B", "V", "G", "R", "I", "L", "Ha", "OIII", "SII")


def canonical_filter(name: object) -> str:
    """``Halpha`` / ``H alpha`` -> ``Ha``, ``O III`` -> ``OIII``, ``Clear`` -> ``L``.

    Johnson / Sloan-like single letters keep their case (``V``, ``R``).
    Unknown names are returned stripped and unchanged.
    """
    text = str(name or "").strip()
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    return _ALIASES.get(compact, text)


__all__ = ["CANONICAL_FILTERS", "canonical_filter"]
