"""ePSF star-count selection helpers (no heavy photometry deps)."""

from __future__ import annotations


def n_epsf_stars_to_select(
    n_stars: int,
    *,
    fraction_epsf_stars: float = 0.2,
    minimum_n_stars: int = 15,
    maximum_n_stars: int | None = 100,
) -> int:
    """
    How many bright stars (below the 99th-percentile flux cut) to try for ePSF.

    Takes ``int(n_stars * fraction_epsf_stars)``, then clamps to
    ``[minimum_n_stars, maximum_n_stars]`` (``maximum_n_stars=None`` → no upper
    cap). Prevents dense fields from feeding hundreds of faint stars into the
    ePSF when only a fraction of detections was requested.
    """
    if n_stars < 0:
        raise ValueError(f"n_stars must be >= 0, got {n_stars}")
    if fraction_epsf_stars < 0:
        raise ValueError(
            f"fraction_epsf_stars must be >= 0, got {fraction_epsf_stars}"
        )
    if minimum_n_stars < 1:
        raise ValueError(f"minimum_n_stars must be >= 1, got {minimum_n_stars}")
    if maximum_n_stars is not None and maximum_n_stars < minimum_n_stars:
        raise ValueError(
            f"maximum_n_stars ({maximum_n_stars}) must be >= minimum_n_stars "
            f"({minimum_n_stars}), or None for no upper cap."
        )

    n = int(n_stars * fraction_epsf_stars)
    if n < minimum_n_stars:
        n = minimum_n_stars
    if maximum_n_stars is not None and n > maximum_n_stars:
        n = maximum_n_stars
    return n


__all__ = ["n_epsf_stars_to_select"]
