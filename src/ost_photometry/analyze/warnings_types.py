"""Warning categories for ost_photometry.analyze.

``analyze.py`` applies ``filterwarnings(..., category=UserWarning)`` to reduce
noise from dependencies. Diagnostics from this package use
``OstPhotometryAnalyzeWarning`` (a plain ``Warning``, not ``UserWarning``) so
they remain visible unless explicitly filtered.
"""


class OstPhotometryAnalyzeWarning(Warning):
    """Non-UserWarning category for intentional analyze-module messages."""
