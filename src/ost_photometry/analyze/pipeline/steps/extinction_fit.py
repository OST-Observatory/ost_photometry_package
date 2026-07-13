"""Extinction fit step: determine ExtinctionCoefficients via fit_extinction_from_value_airmass."""

from pathlib import Path

from .. import base
from ..context import AnalysisContext
from ..config import PipelineConfig
from ...extinction import (
    ExtinctionCoefficients,
    fit_extinction_from_value_airmass,
    observation_to_extinction_fit_table,
)
from ...extinction_io import save_extinction_coefficients


class ExtinctionFitStep(base.PipelineStep):
    """
    Determine extinction coefficients from reduced observation (cat-star.org method).

    Input: reduced observation series (WCS, extraction, correlation done).
    Output: ExtinctionCoefficients per filter, saved to JSON and stored in context.

    Requires stars observed at different airmasses (e.g. over several hours).
    Best done on clear nights with G2V or similar stars.
    """

    name = "extinction_fit"

    def skip(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> bool:
        return config.extinction_mode != "from_value_airmass"

    def run(
        self,
        context: AnalysisContext,
        config: PipelineConfig,
    ) -> AnalysisContext:
        from .... import terminal_output

        obs = context.require_observation()
        if obs is None:
            raise RuntimeError(
                "ExtinctionFitStep requires context._observation (run via Observation.run_pipeline)"
            )

        terminal_output.print_to_terminal(
            "Extinction fit (cat-star.org method: flux/mag vs airmass)",
            style_name="HEADER",
        )

        filter_list = context.filter_list
        if not filter_list:
            raise RuntimeError("No filter_list in context.")

        # Build table from observation
        use_flux = getattr(config, "extinction_fit_use_flux", False)
        mag_col = getattr(config, "extinction_fit_mag_col", "mags_fit")
        data = observation_to_extinction_fit_table(
            obs,
            filter_list,
            mag_col=mag_col,
            use_flux=use_flux,
        )

        if len(data) < 3:
            raise RuntimeError(
                "Insufficient data for extinction fit. Need multiple images "
                "of same stars at different airmasses (observe over several hours)."
            )

        # Build value_cols from available filter columns
        value_cols = {}
        for f in filter_list:
            col = f"flux_{f}" if use_flux else f"mag_{f}"
            if col in data.colnames:
                value_cols[f] = col

        if not value_cols:
            raise RuntimeError(
                "No magnitude/flux columns found. Check filter_list and mag_col."
            )

        # Fit
        output_dir = Path(context.output_dir)
        file_type = getattr(config, "file_type_plots", "pdf")
        coefficients = fit_extinction_from_value_airmass(
            data,
            value_cols=value_cols,
            fallback_airmass_col="airmass",
            id_col="id",
            use_magnitude=not use_flux,
            output_dir=str(output_dir),
            file_type=file_type,
        )

        if not coefficients:
            raise RuntimeError("Extinction fit produced no coefficients.")

        # Store in context
        context.extinction_coefficients = coefficients

        # Save to file
        out_name = getattr(config, "extinction_coefficients_filename", "extinction_coefficients.json")
        out_path = output_dir / out_name
        meta = {"method": "value_airmass", "site": "OST_Potsdam"}
        night_id = getattr(config, "extinction_night_id", None)
        if night_id:
            meta["night_id"] = night_id
        save_extinction_coefficients(out_path, coefficients, meta=meta)

        terminal_output.print_to_terminal(
            f"Extinction coefficients saved to {out_path}",
            style_name="INFO",
        )
        for k, ec in coefficients.items():
            terminal_output.print_to_terminal(f"  {ec}", style_name="INFO")

        return context
