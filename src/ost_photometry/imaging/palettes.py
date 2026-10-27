"""Channel mixes: which filters go into R, G and B (and the luminance)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .filters import canonical_filter

CHANNELS = ("R", "G", "B")

#: Luminance component that stands for the colour channels themselves
#: (synthetic luminance, mean of the mixed R, G and B).
RGB_LUMINANCE = "RGB"


@dataclass(frozen=True)
class Palette:
    """Weighted filters per output channel, an optional luminance mix.

    ``luminance``: weighted filters (and ``"RGB"``, the colour channels)
    whose noise-normalised sum replaces the lightness (see ``compose``).

    ``linked_stretch``: one stretch for all channels (keeps the colours of
    broadband data); narrowband channels differ by orders of magnitude and
    are stretched one by one.
    """

    name: str
    mix: Mapping[str, Mapping[str, float]]
    luminance: Mapping[str, float] | None = None
    linked_stretch: bool = True
    broadband: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def filters(self) -> set[str]:
        used = {f for weights in self.mix.values() for f in weights}
        if self.luminance:
            used |= {f for f in self.luminance if f != RGB_LUMINANCE}
        return used


def _rgb(available: set[str]) -> dict[str, dict[str, float]]:
    green = "G" if "G" in available else "V"
    return {"R": {"R": 1.0}, "G": {green: 1.0}, "B": {"B": 1.0}}


#: Palette names; ``auto`` picks one from the filters (see :func:`resolve_palette`).
PALETTES = ("auto", "rgb", "lrgb", "sho", "hoo", "custom")


def _check(name: str, palette: Palette, available: set[str]) -> Palette:
    missing = sorted(palette.filters() - available)
    if missing:
        raise ValueError(
            f"Palette {name!r} needs filter(s) {missing}; available: {sorted(available)}"
        )
    return palette


def luminance_mix(luminance: str | Mapping[str, float] | None) -> dict[str, float] | None:
    """``"Clear"`` -> ``{"L": 1}``, ``{"Clear": 0.7, "Ha": 0.3}`` -> canonical
    names; ``"RGB"`` (the colour channels) is kept. Weights must be >= 0
    with a positive sum."""
    if luminance is None or luminance == "" or luminance == {}:
        return None
    items = {luminance: 1.0} if isinstance(luminance, str) else dict(luminance)
    mix = {}
    for name, weight in items.items():
        key = RGB_LUMINANCE if str(name).strip().upper() == RGB_LUMINANCE else canonical_filter(name)
        mix[key] = mix.get(key, 0.0) + float(weight)
    if any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError(f"luminance weights must be >= 0 with a positive sum, got {luminance!r}")
    return {k: w for k, w in mix.items() if w > 0}


def resolve_palette(
    name: str,
    available: Iterable[str],
    *,
    custom_mix: Mapping[str, Mapping[str, float]] | None = None,
    luminance: str | Mapping[str, float] | None = None,
) -> Palette:
    """Palette for the canonical filters in ``available``.

    ``auto``: LRGB if R, G or V, B and L exist; RGB with R, G or V, B; SHO
    with SII, Ha, OIII; HOO with Ha and OIII. ``custom`` uses ``custom_mix``
    (``{"R": {"Ha": 0.8, "R": 0.2}, "G": {"V": 1}, "B": {"B": 1}}``).
    ``luminance`` (any palette): a filter or a weighted mix such as
    ``{"Clear": 0.7, "Ha": 0.3}`` or ``{"Clear": 0.5, "RGB": 0.5}``; it
    replaces the palette's own (LRGB: Clear).
    """
    palette = _resolve(name, set(available), custom_mix)
    mix = luminance_mix(luminance)
    if mix is not None:
        from dataclasses import replace

        palette = _check(palette.name, replace(palette, luminance=mix), set(available))
    return palette


def _resolve(
    name: str,
    available: set[str],
    custom_mix: Mapping[str, Mapping[str, float]] | None,
) -> Palette:
    if name not in PALETTES:
        raise ValueError(f"palette must be one of {PALETTES}, got {name!r}")
    has_rgb = {"R", "B"} <= available and ({"G", "V"} & available)
    if name == "auto":
        if has_rgb and "L" in available:
            name = "lrgb"
        elif has_rgb:
            name = "rgb"
        elif {"SII", "Ha", "OIII"} <= available:
            name = "sho"
        elif {"Ha", "OIII"} <= available:
            name = "hoo"
        else:
            raise ValueError(
                "No palette fits the filters "
                f"{sorted(available)}; use palette='custom' with custom_mix"
            )
    if name == "rgb":
        return _check(name, Palette("rgb", _rgb(available)), available)
    if name == "lrgb":
        return _check(name, Palette("lrgb", _rgb(available), luminance={"L": 1.0}), available)
    if name == "sho":
        return _check(name, Palette(
            "sho", {"R": {"SII": 1.0}, "G": {"Ha": 1.0}, "B": {"OIII": 1.0}},
            linked_stretch=False, broadband=False), available)
    if name == "hoo":
        return _check(name, Palette(
            "hoo", {"R": {"Ha": 1.0}, "G": {"OIII": 1.0}, "B": {"OIII": 1.0}},
            linked_stretch=False, broadband=False), available)
    if not custom_mix:
        raise ValueError("palette='custom' needs custom_mix")
    mix = {ch: {canonical_filter(f): float(w) for f, w in dict(custom_mix.get(ch) or {}).items()}
           for ch in CHANNELS}
    empty = [ch for ch in CHANNELS if not mix[ch]]
    if empty:
        raise ValueError(f"custom_mix has no filter for channel(s) {empty}")
    # Linked stretch also for mixes like Ha into R (HaRGB); pure narrowband
    # mixes are better stretched per channel (settings.linked_stretch).
    narrow = {"Ha", "OIII", "SII"}
    broadband = not any(set(w) & narrow for w in mix.values())
    only_narrow = all(set(w) <= narrow for w in mix.values())
    return _check("custom", Palette("custom", mix, linked_stretch=not only_narrow,
                                    broadband=broadband), available)


__all__ = ["CHANNELS", "PALETTES", "RGB_LUMINANCE", "Palette", "luminance_mix", "resolve_palette"]
