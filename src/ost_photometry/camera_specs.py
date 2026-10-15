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
import re
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

_CATALOG_NAME = "cameras.json"
_MEASUREMENTS_NAME = "camera_measurements.json"

#: ``READOUTM`` values that do not name the camera's readout mode. MaxIm DL
#: writes its own exposure-tab setting ("Fast" / "Normal" / "Slow") for
#: QHY cameras driven through ASCOM; the driver mode is not in the header.
PLACEHOLDER_READOUT_MODES = frozenset({"fast", "normal", "slow"})

#: Normalised name of a readout mode that is not known from the header.
UNKNOWN_READOUT_MODE = "unknown"
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
    "photographic dso 2cms": "photography_2cms",
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


#: Driver names that do not identify the camera model; the model is derived
#: from the physical chip size (NAXIS x binning) instead.
GENERIC_INSTRUMENT_NAMES = frozenset(
    {
        "QHYCCD-Cameras-Capture",
        "QHYCCD-Cameras2-Capture",
        "ASI Camera (1)",
        "ASI Camera (2)",
    }
)

#: ``(model, allowed physical widths, allowed physical heights)``; several
#: values cover overscan / driver cropping variants seen in the archive.
PHYSICAL_CHIP_SIZES: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
    ("QHY600M", (9576, 9600), (6388, 6387, 6422)),
    ("QHY268M", (6280, 6279, 6252), (4210, 4209, 4176)),
    ("QHY485C", (3864,), (2180, 2178)),
    ("QHY5III462C", (1920,), (1080, 1078)),
    ("ZWO ASI2600", (6248,), (4176,)),
)


def _as_int(value: object, default: int) -> int:
    try:
        number = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def normalize_instrument_name(
    instrume: str | None,
    naxis1: object = None,
    naxis2: object = None,
    xbinning: object = 1,
    ybinning: object = 1,
) -> str:
    """Camera model from the ``INSTRUME`` keyword and the chip geometry.

    Names that already contain ``QHY600M`` / ``QHY268M`` are shortened to
    the model. Generic driver names (:data:`GENERIC_INSTRUMENT_NAMES`) are
    resolved through the physical chip size; an unknown size returns ``""``.
    Any other name is returned stripped and unchanged.
    """
    name = str(instrume or "").strip()
    if "QHY268M" in name:
        return "QHY268M"
    if "QHY600M" in name:
        return "QHY600M"
    if name not in GENERIC_INSTRUMENT_NAMES:
        return name
    nx = _as_int(naxis1, 0) * _as_int(xbinning, 1)
    ny = _as_int(naxis2, 0) * _as_int(ybinning, 1)
    for model, widths, heights in PHYSICAL_CHIP_SIZES:
        if nx in widths and ny in heights:
            return model
    return ""


#: Words in driver mode names that do not change the mode: newer QHY SDKs
#: call the standard modes "High Gain Mode 16BIT", "Photographic DSO 16BIT",
#: "Extend Fullwell Mode" (versus the "... 2CMS" variants).
_READOUT_FILLER_WORDS = frozenset({"16bit", "mode"})


def _lookup_readout_alias(key: str) -> str | None:
    for candidate in (key, key.replace(" ", ""), key.replace(" ", "_")):
        if candidate in READOUT_ALIASES:
            return READOUT_ALIASES[candidate]
    return None


def normalize_readout_mode(readout_mode: str | None) -> str | None:
    if readout_mode is None:
        return None
    key = str(readout_mode).strip().lower().replace("-", " ").replace("(", " ").replace(")", " ")
    key = " ".join(key.replace("_", " ").split())
    if key in PLACEHOLDER_READOUT_MODES or key == UNKNOWN_READOUT_MODE:
        return UNKNOWN_READOUT_MODE
    found = _lookup_readout_alias(key)
    if found is not None:
        return found
    core = " ".join(w for w in key.split() if w not in _READOUT_FILLER_WORDS)
    if core and core != key:
        found = _lookup_readout_alias(core)
        if found is not None:
            return found
    return key.replace(" ", "_") or None


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


#: ``digital``: pixels are summed after the readout (CMOS), so the read noise
#: of a binned pixel grows with ``sqrt(xbin * ybin)``. ``charge``: the charge
#: is binned on the chip (CCD) and each binned pixel is read once.
BINNING_MODES = ("digital", "charge")

_DIGITAL_BINNING_NAME = re.compile(r"qhy|zwo|\basi(\b|\d)")
_CHARGE_BINNING_NAME = re.compile(r"sbig|\bccd\b")


def binning_mode(camera: str) -> str | None:
    """How ``camera`` bins (:data:`BINNING_MODES`), or ``None`` if unknown.

    The catalog field ``binning_mode`` wins; cameras outside the catalog
    are recognised by name (QHY / ZWO: digital, SBIG / CCD: charge).
    """
    rec = _camera_record(camera)
    if rec is not None and rec.get("binning_mode") in BINNING_MODES:
        return str(rec["binning_mode"])
    name = str(camera or "").strip().lower()
    if _DIGITAL_BINNING_NAME.search(name):
        return "digital"
    if _CHARGE_BINNING_NAME.search(name):
        return "charge"
    return None


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
    """Linear interpolation on a catalog curve, or ``None``.

    Outside the tabulated range the curve's end value is used (with a
    warning): extrapolating a gain or read-noise curve is less safe than
    holding it, but the value is then an approximation.
    """
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
    if float(x) < float(np.min(xg)) or float(x) > float(np.max(xg)):
        import warnings

        warnings.warn(
            f"{camera} {quantity} curve covers {float(np.min(xg)):g}-{float(np.max(xg)):g}; "
            f"{float(x):g} is outside, the end value is used.",
            stacklevel=2,
        )
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


def is_unknown_readout_mode(readout_mode: object) -> bool:
    """True for a missing mode and for placeholders such as MaxIm's "Fast" / "Normal"."""
    if readout_mode is None:
        return True
    text = str(readout_mode).strip()
    if not text or text.lower() in {"none", "nan", "--"}:
        return True
    return normalize_readout_mode(text) == UNKNOWN_READOUT_MODE


@lru_cache(maxsize=1)
def camera_measurements() -> dict:
    """OST measurements of gain and read noise (``data/camera_measurements.json``)."""
    resource = files("ost_photometry.data").joinpath(_MEASUREMENTS_NAME)
    return json.loads(resource.read_text(encoding="utf-8"))


def site_measurement(
    camera: str,
    gain_setting: float | None,
    *,
    offset: float | None = None,
    xbin: int = 1,
    ybin: int = 1,
) -> dict | None:
    """Measured ``system_gain`` (e-/ADU) and ``read_noise`` (e- per binned pixel).

    Entries must match the gain setting; an entry with the same offset is
    preferred. The read noise is the value measured at this binning; for a
    binning without a measurement it is derived from the native one
    (``√(xbin·ybin)``, digital summing), and ``derived_read_noise`` says so.
    Returns ``None`` without a matching entry.
    """
    cam = normalize_camera_id(camera)
    if cam is None or gain_setting is None:
        return None
    entries = camera_measurements().get("cameras", {}).get(cam) or []
    try:
        gs = float(gain_setting)
    except (TypeError, ValueError):
        return None
    matches = [e for e in entries if float(e.get("gain_setting", float("nan"))) == gs]
    if not matches:
        return None
    if offset is not None:
        same = [e for e in matches if e.get("offset") is not None and float(e["offset"]) == float(offset)]
        matches = same or matches
    n = max(int(xbin), 1) * max(int(ybin), 1)
    key = str(max(int(xbin), int(ybin))) if int(xbin) == int(ybin) else f"{int(xbin)}x{int(ybin)}"
    for entry in matches:
        rn = entry.get("read_noise_adu") or {}
        if key in rn:
            gain = float(entry["system_gain"])
            return {"system_gain": gain, "read_noise": float(rn[key]) * gain,
                    "derived_read_noise": False, "entry": entry}
    for entry in matches:
        rn = entry.get("read_noise_adu") or {}
        if "1" in rn:
            gain = float(entry["system_gain"])
            return {"system_gain": gain, "read_noise": float(rn["1"]) * gain * float(np.sqrt(n)),
                    "derived_read_noise": n > 1, "entry": entry}
    entry = matches[0]
    return {"system_gain": float(entry["system_gain"]), "read_noise": None,
            "derived_read_noise": False, "entry": entry}


def _quiet_curves(camera: str, quantities, x, mode) -> list[float | None]:
    """Curve values for a scan over modes (end-value warnings not repeated)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [interpolate_camera_curve(camera, q, x, mode) for q in quantities]


def catalog_readout_modes(camera: str, quantity: str = "readout_noise") -> list[str]:
    """Readout modes with a usable ``quantity`` curve in the catalog."""
    rec = _camera_record(camera)
    if rec is None:
        return []
    modes = [c.get("readout_mode") for c in rec.get("curves") or []
             if c.get("quantity") == quantity and c.get("usable", True)]
    return list(dict.fromkeys(m for m in modes if m))


def readout_mode_candidates(
    camera: str,
    gain_setting: float | None,
    *,
    system_gain: float | None,
    read_noise: float | None,
    read_noise_tolerance: float = 0.2,
    gain_tolerance: float = 0.3,
) -> list[str]:
    """Catalog modes whose curves agree with a measured gain and native read noise.

    Relative tolerances: the manufacturer gain curves of the QHY600M are
    15 % above the OST measurements, the read noise separates the modes
    (2CMS 5.7-5.8 e-, others 7.4-7.7 e-, high gain 3.3-3.6 e- at GAIN 0).
    """
    if gain_setting is None:
        return []
    out = []
    for mode in catalog_readout_modes(camera):
        rn, g = _quiet_curves(camera, ("readout_noise", "system_gain"), gain_setting, mode)
        if read_noise is not None and rn is not None and abs(read_noise - rn) > read_noise_tolerance * rn:
            continue
        if system_gain is not None and g is not None and abs(system_gain - g) > gain_tolerance * g:
            continue
        out.append(mode)
    return out


def full_well_adu(camera: str, gain_setting: float | None, modes: list[str]) -> float | None:
    """Smallest full well among ``modes``, in ADU (catalog full well / catalog gain).

    The full well and the gain come from the same manufacturer curves, so
    their ratio stays right when the absolute gain is off.
    """
    if gain_setting is None:
        return None
    values = []
    for mode in modes:
        fw, g = _quiet_curves(camera, ("fullwell", "system_gain"), gain_setting, mode)
        if fw is not None and g is not None and g > 0:
            values.append(fw / g)
    return min(values) if values else None
