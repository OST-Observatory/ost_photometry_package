#!/usr/bin/env python3
"""Compile digitized camera CSVs into the runtime JSON catalog.

Reads ``src/ost_photometry/data/camera_specs/*.csv`` and writes
``src/ost_photometry/data/cameras.json``. The reduction pipeline loads only
the JSON file.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ost_photometry.camera_specs import (  # noqa: E402
    SKIP_FILES,
    load_xy_csv,
    parse_spec_filename,
)

_DATA = _SRC / "ost_photometry" / "data"
_SPEC_DIR = _DATA / "camera_specs"
_OUT = _DATA / "cameras.json"

# Scalar chip sizes (mm) previously hardcoded in calibration_parameters.
_CHIP_MM = {
    "qhy600m": [32.00, 24.00],
    "qhy268": [23.45, 15.7],
    "stf8300": [17.96, 13.52],
    # IMX571, same optical format as QHY268.
    "asi2600": [23.45, 15.7],
    # IMX485, 3864×2180 at 2.9 μm.
    "qhy5iii485c": [11.21, 6.32],
    # IMX462, 1920×1080 at 2.9 μm.
    "qhy5iii462": [5.57, 3.13],
}

_CAMERA_INFO_QUANTITIES = {
    "qhy600m": ["system_gain", "readout_noise", "dark_current"],
    "qhy268": ["system_gain", "readout_noise", "dark_current"],
    "stf8300": ["dark_current"],
    "qhy5iii485c": ["system_gain", "readout_noise"],
    "qhy5iii462": ["system_gain", "readout_noise"],
    "asi2600": ["system_gain", "readout_noise"],
}

_DEFAULTS = {
    "stf8300": {"readout_noise": 9.3, "system_gain": None},
}

_DISPLAY = {
    "qhy600m": "QHY600M",
    "qhy268": "QHY268M/C",
    "qhy5iii485c": "QHY5III485C",
    "qhy5iii462": "QHY5III462",
    "stf8300": "SBIG STF-8300",
    "asi2600": "ZWO ASI2600",
}

_UNITS = {
    "system_gain": {
        "x_quantity": "gain_setting",
        "x_unit": "1",
        "y_quantity": "system_gain",
        "y_unit": "electron / adu",
    },
    "readout_noise": {
        "x_quantity": "gain_setting",
        "x_unit": "1",
        "y_quantity": "readout_noise",
        "y_unit": "electron",
    },
    "dark_current": {
        "x_quantity": "temperature",
        "x_unit": "deg_C",
        "y_quantity": "dark_current",
        "y_unit": "electron / s",
    },
    "fullwell": {
        "x_quantity": "gain_setting",
        "x_unit": "1",
        "y_quantity": "full_well",
        "y_unit": "electron",
    },
    "dynamic_range": {
        "x_quantity": "gain_setting",
        "x_unit": "1",
        "y_quantity": "dynamic_range",
        "y_unit": "unknown",
    },
    "qe": {
        "x_quantity": "wavelength",
        "x_unit": "nm",
        "y_quantity": "quantum_efficiency",
        "y_unit": "unknown",
    },
    "linearity": {
        "x_quantity": "signal",
        "x_unit": "unknown",
        "y_quantity": "measured",
        "y_unit": "unknown",
    },
    "stats": {
        "x_quantity": "unknown",
        "x_unit": "unknown",
        "y_quantity": "unknown",
        "y_unit": "unknown",
    },
}

_SKIP_NOTES = {
    "qhy268MC_fullwell_photography.csv": "digitization is scattered with large negative y",
    "qhy268MC_fullwell_photography_2cms.csv": "digitization is scattered with large negative y",
}

_UNUSABLE = {
    ("qhy5iii485c", "fullwell"): (
        "x runs to ~1e4 and y to ~95; likely GAIN in column 2 (not applied)"
    ),
    ("qhy5iii485c", "dynamic_range"): (
        "y runs to ~1e4 at GAIN 0–100; may be full well rather than DR in stops"
    ),
}

_CAMERA_NOTES = {
    "qhy268": "QHY268M uses the QHY268MC (IMX571) traces. No system-gain CSV in the drop.",
    "qhy5iii462": "Single readout mode. IMX462 1920×1080 at 2.9 μm.",
    "qhy5iii485c": "IMX485 3864×2180 at 2.9 μm. QE wavelength stored as nm (~400–1000).",
    "asi2600": "stats_red = RN, stats_pink = e-/ADU, stats_blue = DR. Dark CSV was empty and removed.",
}


def _round_points(x, y) -> tuple[list[float], list[float]]:
    return [round(float(v), 8) for v in x], [round(float(v), 8) for v in y]


def _clean(quantity: str, x, y):
    if quantity in {"system_gain", "readout_noise", "dark_current", "fullwell", "qe"}:
        keep = y >= 0
        x, y = x[keep], y[keep]
    if quantity in {"system_gain", "readout_noise", "fullwell"} and x.size:
        keep = x >= -1.0
        x, y = x[keep], y[keep]
    return x, y


def _usability(camera: str, quantity: str, source: str, n_points: int) -> tuple[bool, str]:
    if n_points < 2:
        return False, "fewer than two finite points after cleaning"
    note = _UNUSABLE.get((camera, quantity))
    if note:
        return False, note
    if source in _SKIP_NOTES:
        return False, _SKIP_NOTES[source]
    return True, ""


def build() -> dict:
    cameras: dict[str, dict] = {}
    skipped: list[dict] = []

    for path in sorted(_SPEC_DIR.glob("*.csv")):
        if path.name in SKIP_FILES:
            skipped.append({"source": path.name, "reason": _SKIP_NOTES.get(path.name, "skipped")})
            continue
        meta = parse_spec_filename(path.name)
        if meta is None:
            skipped.append({"source": path.name, "reason": "unparsed filename"})
            continue
        try:
            x, y = load_xy_csv(path)
        except (OSError, ValueError) as exc:
            skipped.append({"source": path.name, "reason": str(exc)})
            continue
        quantity = str(meta["quantity"])
        camera = str(meta["camera"])
        x, y = _clean(quantity, x, y)
        modes = [meta["mode"]]
        if meta["extra_modes"]:
            modes.extend(meta["extra_modes"].split(","))
        usable, note = _usability(camera, quantity, path.name, int(x.size))
        units = dict(_UNITS.get(quantity, {
            "x_quantity": "unknown",
            "x_unit": "unknown",
            "y_quantity": quantity,
            "y_unit": "unknown",
        }))
        xs, ys = _round_points(x, y)
        rec = cameras.setdefault(
            camera,
            {
                "display_name": _DISPLAY.get(camera, camera),
                "chip_mm": _CHIP_MM.get(camera),
                "defaults": _DEFAULTS.get(camera, {}),
                "camera_info_quantities": list(_CAMERA_INFO_QUANTITIES.get(camera, [])),
                "notes": _CAMERA_NOTES.get(camera, ""),
                "curves": [],
            },
        )
        for mode in modes:
            rec["curves"].append(
                {
                    "quantity": quantity,
                    "readout_mode": mode,
                    "channel": meta["channel"],
                    **units,
                    "source": path.name,
                    "usable": usable,
                    "notes": note,
                    "x": xs,
                    "y": ys,
                }
            )

    catalog = {
        "meta": {
            "updated": date.today().isoformat(),
            "source_dir": "ost_photometry/data/camera_specs",
            "note": (
                "Runtime camera parameters. Built from digitized manufacturer "
                "CSVs by scripts/build_camera_catalog.py. Do not edit curve "
                "arrays by hand; change the CSVs and rebuild."
            ),
            "skipped": skipped,
        },
        "cameras": cameras,
    }
    return catalog


def main() -> None:
    catalog = build()
    _OUT.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    n_cam = len(catalog["cameras"])
    n_curves = sum(len(c["curves"]) for c in catalog["cameras"].values())
    n_usable = sum(
        1
        for c in catalog["cameras"].values()
        for curve in c["curves"]
        if curve["usable"]
    )
    print(f"Wrote {_OUT.relative_to(_REPO_ROOT)}")
    print(f"  cameras={n_cam} curves={n_curves} usable={n_usable} skipped={len(catalog['meta']['skipped'])}")


if __name__ == "__main__":
    main()
