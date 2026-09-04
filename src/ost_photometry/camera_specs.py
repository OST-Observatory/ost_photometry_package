"""Camera curves: digitized CSVs as provenance, JSON catalog at runtime.

Manufacturer plots were traced to headerless two-column CSVs under
``ost_photometry/data/camera_specs/``. Those files have no units or mode
metadata; filenames are the only encoding. The reduction code does **not**
read them. ``scripts/build_camera_catalog.py`` compiles them into
``ost_photometry/data/cameras.json``, which carries units, readout-mode
names, aliases, chip size, and a ``usable`` flag.

See ``data/camera_specs/README.md``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

_CATALOG_NAME = "cameras.json"
_SPEC_DIR_NAME = "camera_specs"

CAMERA_ALIASES = {
    "qhy600m": "qhy600m",
    "qhy600": "qhy600m",
    "qhy268m": "qhy268",
    "qhy268c": "qhy268",
    "qhy268mc": "qhy268",
    "qhy268": "qhy268",
    "qhy485c": "qhy5iii485c",
    "qhy5iii485c": "qhy5iii485c",
    "qhy462": "qhy5iii462",
    "qhy462c": "qhy5iii462",
    "qhy5iii462": "qhy5iii462",
    "qhy5iii462c": "qhy5iii462",
    "stf8300": "stf8300",
    "sbig stf-8300 ccd camera": "stf8300",
    "sbig stf-8300": "stf8300",
    "asi2600": "asi2600",
    "zwo asi2600": "asi2600",
    "zwo asi2600mc": "asi2600",
    "zwo asi2600mm": "asi2600",
}

READOUT_ALIASES = {
    "photographic dso": "photography",
    "photography mode": "photography",
    "photography": "photography",
    "photography mode 2cms": "photography_2cms",
    "photography 2cms": "photography_2cms",
    "high gain mode": "high_gain",
    "high gain": "high_gain",
    "high gain mode 2cms": "high_gain_2cms",
    "high gain 2cms": "high_gain_2cms",
    "extend fullwell": "extend_fullwell",
    "extend fullwell 2cms": "extend_fullwell_2cms",
    "default": "default",
    "readout_mode_0": "readout_mode_0",
    "readout_mode_1": "readout_mode_1",
    "0": "readout_mode_0",
    "1": "readout_mode_1",
    "2": "extend_fullwell",
    "3": "extend_fullwell_2cms",
}

# Digitized traces that must not enter the catalog as usable curves.
SKIP_FILES = frozenset(
    {
        "qhy268MC_fullwell_photography.csv",
        "qhy268MC_fullwell_photography_2cms.csv",
    }
)

# ASI2600 ``stats_{colour}`` files encode different quantities, not Bayer channels.
_ASI2600_STATS = {
    "blue": "dynamic_range",
    "red": "readout_noise",
    "pink": "system_gain",
}

_CAMERA_PREFIXES = (
    ("qhy5III485C", "qhy5iii485c"),
    ("qhy5III462", "qhy5iii462"),
    ("QHY5III462", "qhy5iii462"),
    ("qhy268MC", "qhy268"),
    ("qhy268C", "qhy268"),
    ("qhy268", "qhy268"),
    ("qhy600M", "qhy600m"),
    ("stf8300", "stf8300"),
    ("asi2600", "asi2600"),
)

_QUANTITY_TOKENS = (
    "system_gain",
    "readout_noise",
    "dark_current",
    "dynamic_range",
    "fullwell",
    "fuelwell",
    "response_curves",
    "response_curve",
    "linearity",
    "stats",
    "QE",
)

_MODE_TOKENS = (
    "high_gain_and_high_gain_2cms",
    "photography_2cms",
    "photography",
    "high_gain_2cms",
    "high_gain",
    "extend_fullwell_2cms",
    "extend_fullwell",
    "readout_mode_1",
    "readout_mode_0",
)


def spec_dir() -> Path:
    """Directory of digitized source CSVs and plots (may be absent in a wheel)."""
    return Path(str(files("ost_photometry.data") / _SPEC_DIR_NAME))


def normalize_camera_id(camera: str) -> str | None:
    key = str(camera).strip().lower().replace("_", " ")
    key = " ".join(key.split())
    compact = key.replace(" ", "").replace("-", "")
    if compact in CAMERA_ALIASES:
        return CAMERA_ALIASES[compact]
    if key in CAMERA_ALIASES:
        return CAMERA_ALIASES[key]
    for alias, cam_id in CAMERA_ALIASES.items():
        if alias.replace(" ", "").replace("-", "") == compact:
            return cam_id
    return None


def normalize_readout_mode(readout_mode: str | None) -> str | None:
    if readout_mode is None:
        return None
    key = str(readout_mode).strip().lower().replace("-", " ")
    key = " ".join(key.replace("_", " ").split())
    compact = key.replace(" ", "")
    if key in READOUT_ALIASES:
        return READOUT_ALIASES[key]
    if compact in READOUT_ALIASES:
        return READOUT_ALIASES[compact]
    underscored = key.replace(" ", "_")
    if underscored in READOUT_ALIASES:
        return READOUT_ALIASES[underscored]
    return underscored or None


def parse_spec_filename(filename: str) -> dict[str, str | None] | None:
    """Return camera / quantity / mode for a ``camera_specs`` filename."""
    name = Path(filename).name
    if name in SKIP_FILES:
        return None
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    if ext not in {".csv", ".png", ".jpg", ".jpeg"}:
        return None
    camera = None
    rest = stem
    for prefix, cam_id in _CAMERA_PREFIXES:
        if stem == prefix or stem.startswith(prefix + "_"):
            camera = cam_id
            rest = stem[len(prefix) :].lstrip("_")
            break
    if camera is None:
        return None
    rest = rest.replace("_vs_gain", "").replace("_vs_temp", "")
    rest = rest.replace("_vs_temp.", "").strip("_")
    quantity = None
    rest_lower = rest.lower()
    for token in _QUANTITY_TOKENS:
        token_lower = token.lower()
        if (
            rest_lower == token_lower
            or rest_lower.startswith(token_lower + "_")
            or rest_lower.endswith("_" + token_lower)
        ):
            quantity = "fullwell" if token_lower == "fuelwell" else token_lower
            if token_lower == "qe" or token_lower.startswith("response"):
                quantity = "qe"
            # Strip the matched token without depending on original case.
            idx = rest_lower.find(token_lower)
            rest = (rest[:idx] + rest[idx + len(token_lower) :]).strip("_")
            rest_lower = rest.lower()
            break
    if quantity is None:
        return None
    channel = None
    for colour in ("red", "green", "blue", "pink"):
        if rest == colour or rest.endswith("_" + colour):
            channel = colour
            rest = rest[: -len(colour)].strip("_") if rest.endswith(colour) else rest
            break
    modes: list[str] = []
    mode_rest = rest
    for token in _MODE_TOKENS:
        if token in mode_rest:
            if token == "high_gain_and_high_gain_2cms":
                modes.extend(["high_gain", "high_gain_2cms"])
            else:
                modes.append(token)
            mode_rest = mode_rest.replace(token, "", 1).strip("_")
            break
    if camera == "asi2600" and quantity == "stats" and channel in _ASI2600_STATS:
        quantity = _ASI2600_STATS[channel]
        channel = None
    mode = modes[0] if modes else None
    extra_modes = modes[1:] if len(modes) > 1 else []
    return {
        "camera": camera,
        "quantity": quantity,
        "mode": mode,
        "channel": channel,
        "filename": name,
        "extra_modes": ",".join(extra_modes) if extra_modes else None,
    }


def load_xy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a headerless two-column digitized curve and sort unique *x*."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    _, uniq = np.unique(x, return_index=True)
    return x[uniq], y[uniq]


@lru_cache(maxsize=1)
def camera_catalog() -> dict:
    """Load the runtime camera catalog shipped with the package."""
    resource = files("ost_photometry.data").joinpath(_CATALOG_NAME)
    return json.loads(resource.read_text(encoding="utf-8"))


def _camera_record(camera: str) -> dict | None:
    cam = normalize_camera_id(camera)
    if cam is None:
        return None
    return camera_catalog().get("cameras", {}).get(cam)


def camera_defaults(camera: str) -> dict:
    """Scalar defaults from the catalog (may be empty)."""
    rec = _camera_record(camera)
    if rec is None:
        return {}
    defaults = rec.get("defaults") or {}
    return dict(defaults)


def chip_size(camera: str) -> tuple[float, float] | None:
    """Chip width and height in millimetres, or ``None`` if unknown."""
    rec = _camera_record(camera)
    if rec is None:
        return None
    mm = rec.get("chip_mm")
    if not mm or len(mm) != 2:
        return None
    return float(mm[0]), float(mm[1])


def _curve_series(
    rec: dict,
    quantity: str,
    mode: str | None,
    *,
    for_camera_info: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    info_quantities = rec.get("camera_info_quantities") or []
    if for_camera_info and quantity not in info_quantities:
        return None
    curves = rec.get("curves") or []
    fallback = None
    for curve in curves:
        if curve.get("quantity") != quantity:
            continue
        if not curve.get("usable", True):
            continue
        x = np.asarray(curve["x"], dtype=float)
        y = np.asarray(curve["y"], dtype=float)
        if x.size < 2:
            continue
        curve_mode = curve.get("readout_mode")
        if curve_mode == mode:
            return x, y
        if curve_mode is None and fallback is None:
            fallback = (x, y)
    if fallback is not None:
        return fallback
    return None


def interpolate_camera_curve(
    camera: str,
    quantity: str,
    x: float | None,
    readout_mode: str | None = None,
    *,
    for_camera_info: bool = False,
) -> float | None:
    """Linear interpolation on a catalog curve, or ``None``."""
    if x is None or not np.isfinite(float(x)):
        return None
    rec = _camera_record(camera)
    if rec is None:
        return None
    mode = normalize_readout_mode(readout_mode)
    series = _curve_series(rec, quantity, mode, for_camera_info=for_camera_info)
    if series is None:
        return None
    xg, yg = series
    y = float(np.interp(float(x), xg, yg))
    if not np.isfinite(y):
        return None
    return y


def camera_spec_files() -> list[str]:
    """Source CSV names recorded in the catalog (and on disk, when present)."""
    names: set[str] = set()
    for rec in camera_catalog().get("cameras", {}).values():
        for curve in rec.get("curves") or []:
            source = curve.get("source")
            if source:
                names.add(str(source))
    spec = spec_dir()
    if spec.is_dir():
        names.update(p.name for p in spec.glob("*.csv"))
    return sorted(names)
