"""Mix, balance and stretch the stacks of one target into a colour image.

Order of the steps (each optional except mixing and stretching):

1. crop to the field all filters cover, replacement of isolated cold
   pixels (``cosmetic_correction``), gradient removal per filter
   (``remove_gradient``),
2. mixing of the filters into R / G / B (and the luminance),
3. background neutralisation (default): the sky is zero in every channel,
4. white balance: star-based (``star_color_calibration``, broadband only)
   or manual ``channel_factors``,
5. stretch, linked (one for all channels, keeps broadband colours) or per
   channel (narrowband),
6. LRGB: the stretched luminance replaces the lightness in CIELAB,
7. saturation (CIELAB chroma) and clipping to 0 ... 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..core.pixel_masks import fill_masked_pixels
from .background import (
    cosmetic_correction,
    neutralize_background,
    remove_gradient,
    sky_level,
)
from .color_calibration import star_white_balance
from .filters import canonical_filter
from .palettes import CHANNELS, Palette, resolve_palette
from .stretch import apply_stretch, stretch_parameters


@dataclass
class ColorSettings:
    """Parameters of :func:`compose` (defaults: neutral sky, auto stretch)."""

    palette: str = "auto"
    custom_mix: dict[str, dict[str, float]] | None = None
    luminance: str | None = None
    neutralize_background: bool = True
    crop_to_common_field: bool = True
    cosmetic_correction: bool = True
    remove_gradient: bool = False
    gradient_method: str = "polynomial"
    gradient_degree: int = 2
    gradient_box: int | None = None
    star_color_calibration: bool = False
    channel_factors: dict[str, float] | None = None
    stretch: str = "auto"
    linked_stretch: bool | None = None
    target_background: float = 0.25
    shadows_sigma: float = 2.8
    highlight_percentile: float = 99.995
    saturation: float = 1.0


@dataclass
class ColorResult:
    """RGB image (rows as in the FITS files, 0 ... 1) and a processing log."""

    rgb: np.ndarray
    palette: Palette
    header: fits.Header
    log: dict[str, object] = field(default_factory=dict)


@dataclass
class _Channel:
    data: np.ndarray
    header: fits.Header
    path: str
    valid: np.ndarray


def load_channels(stacks: Mapping[str, str | Path]) -> dict[str, _Channel]:
    """Read the stacks (``{filter: path}``) keyed by canonical filter name.

    Masked pixels (defects, rejected in every frame) get the median of their
    neighbours; what remains (outside the field of the frames) gets the sky
    level. ``valid`` keeps the original coverage for the crop to the common
    field. All stacks must share one grid.
    """
    out: dict[str, _Channel] = {}
    shape = None
    for name, path in stacks.items():
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            header = hdul[0].header.copy()
            mask = np.asarray(hdul["MASK"].data, dtype=bool) if "MASK" in hdul else None
        bad = ~np.isfinite(data)
        if mask is not None and mask.shape == data.shape:
            bad |= mask
        valid = ~bad
        if bad.any():
            data = np.asarray(fill_masked_pixels(data, bad), dtype=np.float32)
            rest = ~np.isfinite(data)
            if rest.any():
                level, _std = sky_level(data, rest)
                data = np.where(rest, np.float32(level), data)
        if shape is not None and data.shape != shape:
            raise ValueError(
                f"{path}: shape {data.shape} differs from {shape}; the stacks of one "
                "colour image must share one pixel grid (register all filters together)"
            )
        shape = data.shape
        key = canonical_filter(header.get("FILTER", name) or name)
        out[key] = _Channel(data, header, str(path), valid)
    return out


def common_field(valid: np.ndarray, min_fraction: float = 0.9) -> tuple[slice, slice]:
    """Rows / columns to keep: edges are trimmed while less than
    ``min_fraction`` of a border row / column is covered by every filter."""
    r0, r1, c0, c1 = 0, valid.shape[0], 0, valid.shape[1]
    changed = True
    while changed and r1 - r0 > 2 and c1 - c0 > 2:
        changed = False
        sub = valid[r0:r1, c0:c1]
        if sub[0].mean() < min_fraction:
            r0, changed = r0 + 1, True
        if sub[-1].mean() < min_fraction:
            r1, changed = r1 - 1, True
        if sub[:, 0].mean() < min_fraction:
            c0, changed = c0 + 1, True
        if sub[:, -1].mean() < min_fraction:
            c1, changed = c1 - 1, True
    return slice(r0, r1), slice(c0, c1)


def sky_colour_spread(image: np.ndarray, tiles: int = 4) -> float:
    """Largest channel difference of the median sky over ``tiles x tiles``
    tiles of a stretched image (0 = grey everywhere)."""
    h, w, _ = image.shape
    spread = 0.0
    for i in range(tiles):
        for j in range(tiles):
            tile = image[i * h // tiles:(i + 1) * h // tiles, j * w // tiles:(j + 1) * w // tiles]
            median = np.median(tile.reshape(-1, 3), axis=0)
            spread = max(spread, float(median.max() - median.min()))
    return spread


def _mix(weights: Mapping[str, float], filters: Mapping[str, np.ndarray]) -> np.ndarray:
    total = sum(float(w) for w in weights.values())
    mixed = sum(np.float32(w) * filters[f] for f, w in weights.items())
    return (mixed / np.float32(total if total else 1.0)).astype(np.float32)


def _fwhm(channels: Mapping[str, _Channel]) -> float:
    values = [float(c.header.get("FWHMMED", np.nan)) for c in channels.values()]
    values = [v for v in values if np.isfinite(v) and v > 0]
    return float(np.median(values)) if values else 4.0


def _saturation(palette: Palette, channels: Mapping[str, _Channel]) -> dict[str, float] | None:
    """Saturation of each output channel in the data units (e-/s): the lowest
    ``SATLEVEL / EXPTIME`` of its filters (header of the stacks)."""
    out = {}
    for ch, weights in palette.mix.items():
        limits = []
        for f in weights:
            header = channels[f].header
            sat, exptime = header.get("SATLEVEL"), header.get("EXPTIME")
            if sat is None or not exptime:
                return None
            limits.append(float(sat) / float(exptime))
        out[ch] = min(limits)
    return out


def compose(stacks: Mapping[str, str | Path], settings: ColorSettings | None = None
            ) -> ColorResult:
    """Colour image from the stacks of one target (``{filter: path}``)."""
    settings = settings or ColorSettings()
    channels = load_channels(stacks)
    palette = resolve_palette(settings.palette, channels, custom_mix=settings.custom_mix,
                              luminance=settings.luminance)
    log: dict[str, object] = {"settings": asdict(settings), "palette": palette.name,
                              "mix": {k: dict(v) for k, v in palette.mix.items()},
                              "luminance": palette.luminance,
                              "files": {k: c.path for k, c in channels.items()
                                        if k in palette.filters()}}
    data = {k: channels[k].data for k in palette.filters()}
    if settings.crop_to_common_field:
        valid = np.logical_and.reduce([channels[k].valid for k in palette.filters()])
        rows, cols = common_field(valid)
        if (rows.stop - rows.start, cols.stop - cols.start) != valid.shape:
            data = {k: v[rows, cols] for k, v in data.items()}
            log["crop"] = {"rows": [rows.start, rows.stop], "columns": [cols.start, cols.stop]}
    if settings.cosmetic_correction:
        fixed = {k: cosmetic_correction(v) for k, v in data.items()}
        data = {k: v for k, (v, _n) in fixed.items()}
        log["cold_pixels_replaced"] = {k: n for k, (_v, n) in fixed.items()}
    if settings.remove_gradient:
        data = {k: remove_gradient(v, method=settings.gradient_method,
                                   degree=settings.gradient_degree,
                                   box=settings.gradient_box)[0] for k, v in data.items()}
    rgb = {ch: _mix(palette.mix[ch], data) for ch in CHANNELS}
    lum = data[palette.luminance] if palette.luminance else None
    if settings.neutralize_background:
        rgb, levels = neutralize_background(rgb)
        log["sky_levels"] = levels
        if lum is not None:
            lum = lum - np.float32(sky_level(lum)[0])

    factors = {ch: 1.0 for ch in CHANNELS}
    if settings.channel_factors:
        factors.update({str(k): float(v) for k, v in settings.channel_factors.items()})
    elif settings.star_color_calibration:
        if palette.broadband:
            factors, n_stars = star_white_balance(
                rgb, fwhm=_fwhm(channels), saturation=_saturation(palette, channels))
            log["calibration_stars"] = n_stars
        else:
            log["note"] = "star colour calibration skipped for a narrowband palette"
    rgb = {ch: rgb[ch] * np.float32(factors[ch]) for ch in CHANNELS}
    log["channel_factors"] = factors

    linked = palette.linked_stretch if settings.linked_stretch is None else settings.linked_stretch
    kwargs = {"method": settings.stretch, "target_background": settings.target_background,
              "shadows_sigma": settings.shadows_sigma,
              "highlight_percentile": settings.highlight_percentile}
    if linked:
        params = stretch_parameters(sum(rgb.values()) / np.float32(3.0), **kwargs)
        stretched = {ch: apply_stretch(rgb[ch], params) for ch in CHANNELS}
        log["stretch"] = {"linked": True, **params}
    else:
        per = {ch: stretch_parameters(rgb[ch], **kwargs) for ch in CHANNELS}
        stretched = {ch: apply_stretch(rgb[ch], per[ch]) for ch in CHANNELS}
        log["stretch"] = {"linked": False, **{ch: p for ch, p in per.items()}}
    image = np.stack([stretched[ch] for ch in CHANNELS], axis=-1)

    if lum is not None or settings.saturation != 1.0:
        from skimage.color import lab2rgb, rgb2lab

        lab = rgb2lab(image)
        if lum is not None:
            lum_params = stretch_parameters(lum, **kwargs)
            lab[..., 0] = 100.0 * apply_stretch(lum, lum_params)
            log["luminance_stretch"] = lum_params
        if settings.saturation != 1.0:
            lab[..., 1:] *= float(settings.saturation)
        import warnings

        with warnings.catch_warnings():
            # out-of-gamut colours are clipped below anyway
            warnings.simplefilter("ignore", UserWarning)
            image = lab2rgb(lab).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    spread = sky_colour_spread(image)
    log["sky_colour_spread"] = round(spread, 3)
    if spread > 0.08 and not settings.remove_gradient:
        log["hint"] = (f"the sky colour varies by {spread:.2f} across the field; "
                       "remove_gradient=True usually fixes such colour gradients")
    first = channels[next(iter(palette.filters()))].header
    return ColorResult(image, palette, first, log)


__all__ = ["ColorResult", "ColorSettings", "compose", "load_channels"]
