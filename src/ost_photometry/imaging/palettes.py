"""Channel mixes: which filters go into R, G and B (and the luminance)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .filters import canonical_filter

CHANNELS = ("R", "G", "B")


@dataclass(frozen=True)
class Palette:
    """Weighted filters per output channel, an optional luminance filter.

    ``linked_stretch``: one stretch for all channels (keeps the colours of
    broadband data); narrowband channels differ by orders of magnitude and
    are stretched one by one.
    """

    name: str
    mix: Mapping[str, Mapping[str, float]]
    luminance: str | None = None
    linked_stretch: bool = True
    broadband: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def filters(self) -> set[str]:
        used = {f for weights in self.mix.values() for f in weights}
        if self.luminance:
            used.add(self.luminance)
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


def resolve_palette(
    name: str,
    available: Iterable[str],
    *,
    custom_mix: Mapping[str, Mapping[str, float]] | None = None,
    luminance: str | None = None,
) -> Palette:
    """Palette for the canonical filters in ``available``.

    ``auto``: LRGB if R, G or V, B and L exist; RGB with R, G or V, B; SHO
    with SII, Ha, OIII; HOO with Ha and OIII. ``custom`` uses ``custom_mix``
    (``{"R": {"Ha": 0.8, "R": 0.2}, "G": {"V": 1}, "B": {"B": 1}}``) and an
    optional ``luminance`` filter.
    """
    available = set(available)
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
        return _check(name, Palette("lrgb", _rgb(available), luminance="L"), available)
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
    luminance = canonical_filter(luminance) if luminance else None
    empty = [ch for ch in CHANNELS if not mix[ch]]
    if empty:
        raise ValueError(f"custom_mix has no filter for channel(s) {empty}")
    # Linked stretch also for mixes like Ha into R (HaRGB); pure narrowband
    # mixes are better stretched per channel (settings.linked_stretch).
    narrow = {"Ha", "OIII", "SII"}
    broadband = not any(set(w) & narrow for w in mix.values())
    only_narrow = all(set(w) <= narrow for w in mix.values())
    return _check("custom", Palette("custom", mix, luminance=luminance,
                                    linked_stretch=not only_narrow, broadband=broadband),
                  available)


__all__ = ["CHANNELS", "PALETTES", "Palette", "resolve_palette"]
