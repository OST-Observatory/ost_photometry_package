"""Green noise removal (SCNR, Siril "Remove Green Noise").

Subtractive Chromatic Noise Reduction as in PixInsight / Siril: green is
limited by a reference built from red and blue, so green casts of the sky
and green noise disappear. Works on the stretched image (0 ... 1).

Protection methods (``m`` the reference, ``a`` the amount):

* ``average_neutral``: ``G' = min(G, (R + B) / 2)``
* ``maximum_neutral``: ``G' = min(G, max(R, B))``
* ``maximum_mask``:    ``G' = G (1 - a)(1 - m) + G m``, ``m = max(R, B)``
* ``additive_mask``:   ``G' = G (1 - a)(1 - m) + G m``, ``m = min(1, R + B)``

For the two neutral methods the amount blends the result with the original
(``a = 1``: as in Siril). With ``preserve_lightness`` the CIELAB lightness
of the input is restored afterwards, so the image does not get darker.

In narrowband SHO / HOO images green is mostly Halpha: SCNR then turns it
gold / orange (the "Hubble palette" look) rather than removing noise.
On a noisy grey sky only the positive green noise is clipped: green drops by
a fraction of the noise and the sky turns slightly magenta (as in Siril).
"""

from __future__ import annotations

import warnings

import numpy as np

#: Values of ``method`` in :func:`remove_green_noise`.
SCNR_METHODS = ("average_neutral", "maximum_neutral", "maximum_mask", "additive_mask")


def remove_green_noise(
    image: np.ndarray,
    *,
    method: str = "average_neutral",
    amount: float = 1.0,
    preserve_lightness: bool = True,
) -> np.ndarray:
    """SCNR on an RGB image (``..., 3``, values 0 ... 1); returns a new image."""
    if method not in SCNR_METHODS:
        raise ValueError(f"method must be one of {SCNR_METHODS}, got {method!r}")
    a = float(amount)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"amount must be between 0 and 1, got {amount!r}")
    rgb = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if method == "average_neutral":
        green = (1.0 - a) * g + a * np.minimum(g, 0.5 * (r + b))
    elif method == "maximum_neutral":
        green = (1.0 - a) * g + a * np.minimum(g, np.maximum(r, b))
    else:
        m = np.maximum(r, b) if method == "maximum_mask" else np.minimum(1.0, r + b)
        green = g * (1.0 - a) * (1.0 - m) + g * m
    out = rgb.copy()
    out[..., 1] = green
    if preserve_lightness:
        from skimage.color import lab2rgb, rgb2lab

        lab = rgb2lab(out)
        lab[..., 0] = rgb2lab(rgb)[..., 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # out of gamut: clipped
            out = lab2rgb(lab).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


__all__ = ["SCNR_METHODS", "remove_green_noise"]
