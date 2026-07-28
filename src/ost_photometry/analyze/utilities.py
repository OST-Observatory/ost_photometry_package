"""Analysis utilities facade.

Prefer importing from here for a stable public surface. Implementations live
under :mod:`ost_photometry.analyze.utils` (e.g. ``duplicates``, ``errors``) with
remaining helpers still in ``utils._legacy``.
"""

from .utils import *  # noqa: F403
