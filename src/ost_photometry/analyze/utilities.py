"""Analysis utilities facade.

Prefer importing from here for a stable public surface. Implementations live
under :mod:`ost_photometry.analyze.utils` (``duplicates``, ``starmaps``,
``cluster_selection``, …).
"""

from .utils import *  # noqa: F403
from .utils import _SIMBAD_EXPORTS
from .utils import __all__ as _utils_all

__all__ = [*_utils_all, *_SIMBAD_EXPORTS]


def __getattr__(name: str):
    if name in _SIMBAD_EXPORTS:
        from .utils import simbad_annotate as _simbad

        return getattr(_simbad, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
