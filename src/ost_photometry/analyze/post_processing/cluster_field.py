"""Cluster / field post-processing (region, Gaia, proper motion)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.table import Table

from ... import style
from .. import utilities
from . import io
from .imaging import ImagingPlotContext

if TYPE_CHECKING:
    from .. import analyze

PostProcessClusterPhase = Literal["region", "gaia", "pm"]


def _vstack_multiepoch_source(tbl: Table | None) -> Table | None:
    """Return a copy of the full vstack if multiple ``epoch_id`` values exist."""
    if tbl is None or len(tbl) == 0 or "epoch_id" not in tbl.colnames:
        return None
    if len(np.unique(np.asarray(tbl["epoch_id"]))) <= 1:
        return None
    return tbl


def _cluster_work_slice(tbl: Table) -> Table:
    """Use first ``epoch_id`` for region/Gaia/PM when the table is a multi-epoch vstack."""
    src = _vstack_multiepoch_source(tbl)
    if src is None:
        return tbl
    u = np.unique(np.asarray(tbl["epoch_id"]))
    ref = u[0]
    return tbl[np.asarray(tbl["epoch_id"]) == ref]


def _expand_star_ids_to_all_epochs(full: Table, ids: np.ndarray) -> Table:
    """Keep every epoch row for sources whose ``id`` is in ``ids``."""
    id_set = np.unique(np.asarray(ids, dtype=int))
    mask = np.isin(np.asarray(full["id"], dtype=int), id_set)
    return full[mask]


def write_post_processed_cluster_field_table(
    observation: analyze.Observation,
    filter_list: list[str],
    *,
    object_id: int | None = None,
    extraction_method: str = "",
) -> None:
    """Write ``observation.table_magnitudes`` as post-processed ECSV (one file per ``filter_list`` combo)."""
    tbl = observation.table_magnitudes
    if tbl is None or len(tbl) == 0:
        return

    if len(filter_list) == 2:
        rts = f"_{filter_list[0]}-{filter_list[1]}_post_processed"
    elif len(filter_list) == 1:
        rts = "_post_processed"
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nThis should never happen: Number of "
            f"{len(filter_list)} were provided, but only 1 or 2 are supported."
            f"{style.Bcolors.ENDC}"
        )

    io.write_epoch_native_magnitudes(
        observation,
        tbl,
        object_id=object_id,
        rts=rts,
        photometry_extraction_method=extraction_method,
    )


def apply_cluster_field_phase(
    observation: analyze.Observation,
    phase: PostProcessClusterPhase,
    *,
    plot_context: ImagingPlotContext,
    region_radius: float = 600.0,
    max_distance_cluster: float = 6.0,
    find_cluster_para_set: int = 1,
    cluster_selection_id: int | None = None,
    use_wcs_projection_for_star_maps: bool = True,
    file_type_plots: str = "pdf",
    input_table: Table | None = None,
) -> None:
    """
    Apply a single cluster-field phase in place on ``observation.table_magnitudes``.

    ``plot_context`` carries WCS, filter label, output paths, and Vizier cone fields for
    Gaia steps (typically built from the reference band, e.g. ``filter_list[0]``).

    For multi-epoch tables, work is done on the first ``epoch_id`` slice, then surviving
    ``id`` values are expanded to all epochs.
    """
    tbl = input_table if input_table is not None else observation.table_magnitudes
    if tbl is None or len(tbl) == 0:
        return

    multiepoch_full = _vstack_multiepoch_source(tbl)
    work = _cluster_work_slice(tbl)

    if phase == "region":
        work, _ = utilities.region_selection(
            observation.objects_of_interest_coordinates,
            work,
            plot_context=plot_context,
            radius=region_radius,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )
    elif phase == "gaia":
        work, _, _, _ = utilities.find_cluster(
            work,
            observation.get_object_of_interest_names(),
            plot_context=plot_context,
            max_distance=max_distance_cluster,
            parameter_set=find_cluster_para_set,
            cluster_selection_id=cluster_selection_id,
            file_type_plots=file_type_plots,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        )
    elif phase == "pm":
        work, _, _ = utilities.proper_motion_selection(
            work,
            plot_context=plot_context,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            file_type_plots=file_type_plots,
        )
    else:
        raise ValueError(f"Unknown cluster-field phase: {phase!r}")

    if multiepoch_full is not None and len(work) > 0:
        work = _expand_star_ids_to_all_epochs(
            multiepoch_full,
            np.asarray(work["id"]),
        )

    observation.table_magnitudes = work


def post_process_cluster_field(
    observation: analyze.Observation,
    filter_list: list[str],
    object_id: int | None = None,
    extraction_method: str = "",
    extract_only_circular_region: bool = False,
    region_radius: float = 600,
    identify_cluster_gaia_data: bool = False,
    clean_objects_using_proper_motion: bool = False,
    max_distance_cluster: float = 6.0,
    find_cluster_para_set: int = 1,
    cluster_selection_id: int | None = None,
    convert_magnitudes: bool = False,
    target_filter_system: str | None = None,
    output_filter_set: str = "auto",
    output_magnitude_system: str = "auto",
    input_table: Table | None = None,
    distribution_samples: int = 1000,
    use_wcs_projection_for_star_maps: bool = True,
    file_type_plots: str = "pdf",
    *,
    plot_context: ImagingPlotContext,
    skip_cluster_region_step: bool = False,
    skip_cluster_gaia_step: bool = False,
    skip_cluster_pm_step: bool = False,
    skip_magnitude_convert_step: bool = False,
    skip_save_post_processed_magnitudes: bool = False,
    calibration_source: str | None = None,
) -> None:
    """
    Restrict results to specific areas of the image and filter by means
    of proper motion and distance using Gaia

    Parameters
    ----------
    observation
        Container object with image series objects for each
        filter

    filter_list
        Filter names

    object_id
        ID of the object
        Default is ``None``.

    extraction_method
        Applied extraction method. Possibilities: ePSF or APER`
        Default is ``''``.

    extract_only_circular_region
        If True the extracted objects will be filtered such that only
        objects with ``radius`` will be returned.
        Default is ``False``.

    region_radius
        Radius around the object in arcsec.
        Default is ``600``.

    identify_cluster_gaia_data
        If True cluster in the Gaia distance and proper motion data
        will be identified.
        Default is ``False``.

    clean_objects_using_proper_motion
        If True only the object list will be clean based on their
        proper motion.
        Default is ``False``.

    max_distance_cluster
        Expected maximal distance of the cluster in kpc. Used to
        restrict the parameter space to facilitate an easy
        identification of the star cluster.
        Default is ``6``.

    find_cluster_para_set
        Parameter set used to identify the star cluster in proper
        motion and distance data.
        Default is ``1``.

    convert_magnitudes
        If True, run ``apply_magnitude_system_convert_on_observation`` (see
        ``post_processing.magnitude_convert``) after the cluster steps on the **full**
        magnitude table (all epochs). Default is ``False``.

    target_filter_system
        Deprecated alias (``SDSS`` / ``AB`` / ``BESSELL``). Prefer
        ``output_filter_set`` / ``output_magnitude_system``.

    output_filter_set, output_magnitude_system
        Preferred output filter family and Vega/AB system (``auto`` follows catalog).

    input_table
        Table containing magnitudes etc. If None are provided,
        the table will be read from the observation container.
        Default is ``None``.

    distribution_samples
        Number of samples used for distributions
        Default is `1000`.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    plot_context
        :class:`ImagingPlotContext` for cluster-field steps (e.g.
        ``imaging_context_from_image_series(observation.image_series_dict[filter_list[0]])``).
    """
    region_on = extract_only_circular_region and not skip_cluster_region_step
    gaia_on = identify_cluster_gaia_data and not skip_cluster_gaia_step
    pm_on = clean_objects_using_proper_motion and not skip_cluster_pm_step
    convert_on = convert_magnitudes and not skip_magnitude_convert_step
    if not region_on and not gaia_on and not pm_on and not convert_on:
        return

    if not filter_list:
        raise RuntimeError("post_process_cluster_field: filter_list is empty.")

    common = dict(
        plot_context=plot_context,
        region_radius=region_radius,
        max_distance_cluster=max_distance_cluster,
        find_cluster_para_set=find_cluster_para_set,
        cluster_selection_id=cluster_selection_id,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )

    pending_input: Table | None = input_table
    if region_on:
        apply_cluster_field_phase(
            observation, "region", input_table=pending_input, **common
        )
        pending_input = None
    if gaia_on:
        apply_cluster_field_phase(
            observation, "gaia", input_table=pending_input, **common
        )
        pending_input = None
    if pm_on:
        apply_cluster_field_phase(
            observation, "pm", input_table=pending_input, **common
        )
        pending_input = None
    if convert_on:
        from .magnitude_convert import apply_magnitude_system_convert_on_observation

        apply_magnitude_system_convert_on_observation(
            observation,
            target_filter_system=target_filter_system,
            output_filter_set=output_filter_set,
            output_magnitude_system=output_magnitude_system,
            convert_magnitudes=True,
            distribution_samples=distribution_samples,
            calibration_source=calibration_source,
            input_table=pending_input,
        )

    if not skip_save_post_processed_magnitudes:
        write_post_processed_cluster_field_table(
            observation,
            filter_list,
            object_id=object_id,
            extraction_method=extraction_method,
        )
