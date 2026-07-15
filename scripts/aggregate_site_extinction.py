#!/usr/bin/env python3
"""Aggregate per-night extinction JSON files into a site coefficient table."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_extinction_io():
    """Load extinction_io without importing heavy analyze.__init__ dependencies."""
    pkg = _SRC / "ost_photometry" / "analyze"
    spec_w = importlib.util.spec_from_file_location(
        "ost_photometry.analyze.warnings_types",
        pkg / "warnings_types.py",
    )
    mod_w = importlib.util.module_from_spec(spec_w)
    sys.modules["ost_photometry.analyze.warnings_types"] = mod_w
    spec_w.loader.exec_module(mod_w)

    spec_e = importlib.util.spec_from_file_location(
        "ost_photometry.analyze.extinction",
        pkg / "extinction.py",
    )
    mod_e = importlib.util.module_from_spec(spec_e)
    sys.modules["ost_photometry.analyze.extinction"] = mod_e
    spec_e.loader.exec_module(mod_e)

    analyze_stub = types.ModuleType("ost_photometry.analyze")
    analyze_stub.extinction = mod_e
    analyze_stub.warnings_types = mod_w
    sys.modules["ost_photometry.analyze"] = analyze_stub

    spec_io = importlib.util.spec_from_file_location(
        "ost_photometry.analyze.extinction_io",
        pkg / "extinction_io.py",
    )
    mod_io = importlib.util.module_from_spec(spec_io)
    sys.modules["ost_photometry.analyze.extinction_io"] = mod_io
    spec_io.loader.exec_module(mod_io)
    return mod_io


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate night extinction JSON files into a site table.",
    )
    parser.add_argument(
        "--nights",
        nargs="+",
        required=True,
        help="Paths to per-night extinction_coefficients.json files",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output site table path (e.g. src/ost_photometry/data/ost_potsdam_extinction.json)",
    )
    parser.add_argument("--site", default="OST_Potsdam", help="Site name for meta block")
    parser.add_argument(
        "--statistic",
        choices=["median", "weighted_median"],
        default="median",
    )
    parser.add_argument("--sigma-clip", type=float, default=2.5)
    parser.add_argument(
        "--method",
        default="value_airmass",
        help="Label stored in meta.method",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write QC PDF plots alongside the aggregated site table",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for QC plots (default: parent directory of --out)",
    )
    args = parser.parse_args(argv)

    io = _load_extinction_io()
    coeffs, meta = io.aggregate_extinction_coefficients(
        args.nights,
        statistic=args.statistic,
        sigma_clip=args.sigma_clip,
        site=args.site,
        method=args.method,
    )
    if not coeffs:
        print("No coefficients aggregated; check input files.", file=sys.stderr)
        return 1

    io.save_extinction_coefficients(args.out, coeffs, meta=meta)
    print(f"Wrote {args.out} ({len(coeffs)} filters, {meta['n_input_nights']} nights)")
    for filt, ec in sorted(coeffs.items()):
        pf = meta.get("per_filter", {}).get(filt, {})
        n = pf.get("n_nights", "?")
        print(f"  {filt}: k' = {ec.k_prime:.4f} ± {ec.k_prime_err:.4f}  (n_nights={n})")

    if args.plot:
        plot_dir = args.plot_dir or args.out.parent
        plots = io.write_extinction_aggregation_qc_plots(
            args.nights,
            coeffs,
            meta,
            plot_dir,
            site=args.site,
        )
        for path in plots:
            print(f"QC plot: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
