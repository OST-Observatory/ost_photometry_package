"""Starmap, annotation, and image-comparison plots."""
from __future__ import annotations

import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.visualization import ImageNormalize, ZScaleInterval, simple_norm
from matplotlib.patches import Ellipse
from photutils.utils import ImageDepth
from regions import EllipseSkyRegion

from ... import checks, style, terminal_output
from ...output_layout import diagnostics_dir, results_dir
from ..utils.photometry import xy_column_names
from .simbad_galaxy import (
    simbad_angular_axes_arcmin,
    simbad_overlay_kind,
    skycoord_from_simbad,
)

plt.switch_backend("Agg")


def _plot_image_array(image) -> np.ndarray:
    """2-D float image from an ndarray, CCDData, HDU, Quantity, or ``.data`` container."""
    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = getattr(image, "data", image)
        if hasattr(arr, "value") and not isinstance(arr, np.ndarray):
            arr = arr.value
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 3 and out.shape[-1] in (3, 4):
        out = out.mean(axis=-1)
    while out.ndim > 2:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {out.shape}")
    return np.ascontiguousarray(out)


def compare_images(
        output_dir: str, original_image: np.ndarray,
        comparison_image: np.ndarray, file_type: str = 'pdf') -> None:
    """
    Plot two images for comparison

    Parameters
    ----------
    output_dir
        Output directory

    original_image
        Original image (ndarray, CCDData, FITS HDU, or an object with ``.data``)

    comparison_image
        Comparison image (same types as ``original_image``)

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    sci = _plot_image_array(original_image)
    ref = _plot_image_array(comparison_image)

    #   Prepare plot
    plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    #   Original image: normalize and plot
    norm = simple_norm(sci, 'log', percent=99.)
    ax1.imshow(sci, norm=norm, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Science')

    #   Comparison image: normalize and plot
    norm = simple_norm(ref, 'log', percent=99.)
    ax2.imshow(ref, norm=norm, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('HiPS')

    #   Save the plot
    plt.savefig(
        f'{output_dir}/img_comparison.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()


def _sanitize_starmap_filename_part(text: str) -> str:
    replace_dict = {
        ',': '', '.': '', '\\': '', '[': '', '&': '', ' ': '_',
        ':': '', ']': '', '{': '', '}': '', '(': '', ')': '',
        '=': '_',
    }
    for key, value in replace_dict.items():
        text = text.replace(key, value)
    # Collapse runs of underscores from removed punctuation
    while '__' in text:
        text = text.replace('__', '_')
    return text.strip('_').lower()


_REF_FIG_AREA = 20.0 * 9.0
_REF_SCATTER_S = 40.0
_REF_ANNOT_MARKERSIZE = 11.0
_REF_EXTENDED_MARKERSIZE = 14.0
_REF_CENTER_MARKERSIZE = 8.0
_ID_LABEL_MAX = 40
_FOV_ELLIPSE_MAX_FACTOR = 3.0


def _starmap_figsize(ny: int, nx: int) -> tuple[float, float]:
    """Keep the historical ~180 in² figure area; aspect follows the image."""
    aspect = float(ny) / float(max(int(nx), 1))
    aspect = float(np.clip(aspect, 0.25, 4.0))
    width = float(np.sqrt(_REF_FIG_AREA / aspect))
    return width, width * aspect


def _marker_scale_from_figsize(figsize: tuple[float, float], marker_scale: float) -> float:
    area = float(figsize[0]) * float(figsize[1])
    return float(marker_scale) * (area / _REF_FIG_AREA)


def _pixel_transform(ax):
    getter = getattr(ax, "get_transform", None)
    if getter is None:
        return None
    try:
        return getter("pixel")
    except (TypeError, ValueError, AttributeError):
        return None


def _overlay_kwargs(ax) -> dict:
    pix = _pixel_transform(ax)
    return {"transform": pix} if pix is not None else {}


def _set_pixel_limits(ax, ny: int, nx: int) -> None:
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)


def _xy_values(tbl: Table, x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    x = tbl[x_column]
    y = tbl[y_column]
    if hasattr(x, "value"):
        x = x.value
    if hasattr(y, "value"):
        y = y.value
    return np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()


def _xy_columns(tbl: Table, which: str) -> tuple[str, str]:
    cols = xy_column_names(tbl)
    if cols is None:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo valid X and Y column found for "
            f"{which}. {style.Bcolors.ENDC}"
        )
    return cols


def _ellipse_center(patch) -> tuple[float, float]:
    """Pixel center of an Ellipse (``center``, not Rectangle-style ``xy``)."""
    center = getattr(patch, "center", None)
    if center is None:
        getter = getattr(patch, "get_center", None)
        if getter is not None:
            center = getter()
        else:
            center = getattr(patch, "xy", None)
    arr = np.asarray(center, dtype=float).ravel()
    if arr.size < 2:
        raise AttributeError(
            f"{type(patch).__name__} has no 2-D center (got {center!r})"
        )
    return float(arr[0]), float(arr[1])


def _ellipse_fits_image(ellipse: Ellipse, image_shape: tuple[int, ...]) -> bool:
    ny, nx = int(image_shape[0]), int(image_shape[1])
    diag = float(np.hypot(nx, ny))
    return max(float(ellipse.width), float(ellipse.height)) <= _FOV_ELLIPSE_MAX_FACTOR * diag


def covariance_ellipse_pixels(
    x,
    y,
    *,
    n_sigma: float = 2.0,
    min_points: int = 5,
    image_shape: tuple[int, ...] | None = None,
) -> Ellipse | None:
    """2σ covariance ellipse in pixel coordinates, or ``None`` if unusable."""
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[ok], yy[ok]
    if xx.size < int(min_points):
        return None
    cov = np.cov(xx, yy)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width = 2.0 * float(n_sigma) * float(np.sqrt(eigvals[0]))
    height = 2.0 * float(n_sigma) * float(np.sqrt(max(eigvals[1], 0.0)))
    if width <= 0 or height <= 0:
        return None
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    ellipse = Ellipse(
        (float(np.mean(xx)), float(np.mean(yy))),
        width=width,
        height=height,
        angle=angle,
    )
    if image_shape is not None and not _ellipse_fits_image(ellipse, image_shape):
        return None
    return ellipse


def starmap(
        output_dir: str, image: np.ndarray, filter_: str, tbl: Table,
        tbl_2: Table = None, label: str = 'Identified stars',
        label_2: str = 'Identified stars (set 2)', rts: str | None = None,
        mode: str | None = None, magnitude_column: str | None = None,
        name_object: str | None = None,
        wcs_image: wcs.WCS = None, use_wcs_projection: bool = True,
        terminal_logger: terminal_output.TerminalLog | None = None,
        file_type: str = 'pdf', indent: int = 2,
        filename_suffix: str | None = None,
        marker_scale: float = 1.0,
        extra_patches: list | None = None) -> None:
    """
    Plot star maps  -> overlays of the determined star positions on FITS
                    -> supports different versions

    Parameters
    ----------
    output_dir
        Output directory

    image
        The image data

    filter_
        Filter identifier

    tbl
        Astropy table with data of the objects

    tbl_2
        Second astropy table with data of special objects
        Default is ``None``

    label
        Identifier for the objects in `tbl`
        Default is ``Identified stars``

    label_2
        Identifier for the objects in `tbl_2`
        Default is ``Identified stars (set 2)``

    rts
        Expression characterizing the plot (used in the title / log message).
        Default is ``None``

    filename_suffix
        Optional stem used in the output filename instead of ``rts``.
        Prefer this when the title should include an image basename that must
        not appear in the path. Default is ``None`` (fall back to ``rts``).

    marker_scale
        Multiplier on the historical scatter size (``s=40`` at 20×9 in).
        Default is ``1``.

    extra_patches
        Optional matplotlib patches (pixel coordinates) drawn on the image.

    mode
        String used to switch between different plot modes
        Default is ``None``

    magnitude_column
        When ``mode='mags'``, use this column for labels (e.g. ``mag_cal_V``).
        If ``None``, uses ``mag_cali_trans`` or ``mag_cali_no-trans``.
        Default is ``None``.

    name_object
        Name of the object
        Default is ``None``

    wcs_image
        WCS information
        Default is ``None``

    use_wcs_projection
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'starmaps'),
    )

    if rts is not None:
        if terminal_logger is not None:
            terminal_logger.add_to_cache(
                f"Plot {filter_} band image with stars overlaid ({rts})",
                style_name='NORMAL',
                indent=indent,
            )
        else:
            terminal_output.print_to_terminal(
                f"Plot {filter_} band image with stars overlaid ({rts})",
                style_name='NORMAL',
                indent=indent,
            )

    #   Check if column with X and Y coordinates are available for table 1
    x_column, y_column = _xy_columns(tbl, "table 1")
    x_column_2 = y_column_2 = None
    if tbl_2 is not None:
        x_column_2, y_column_2 = _xy_columns(tbl_2, "table 2")

    ny, nx = int(np.asarray(image).shape[0]), int(np.asarray(image).shape[1])
    figsize = _starmap_figsize(ny, nx)
    scale = _marker_scale_from_figsize(figsize, marker_scale)
    scatter_s = _REF_SCATTER_S * scale

    #   Set layout
    fig = plt.figure(figsize=figsize)

    if not use_wcs_projection:
        ax = fig.add_subplot()
    else:
        if wcs_image is not None:
            ax = plt.subplot(projection=wcs_image)
        else:
            terminal_output.print_to_terminal(
                "Sky projection for master plot not possible, since no WCS "
                "was provided. Use Pixel coordinates instead.",
                style_name='WARNING',
                indent=indent,
            )
            ax = fig.add_subplot()

    overlay = _overlay_kwargs(ax)

    #   Limit the space for the object names in case several are given
    if isinstance(name_object, list):
        name_object = ', '.join(name_object)
        if len(name_object) > 20:
            name_object = name_object[0:16] + ' ...'

    #   Set title of the complete plot
    if rts is None and name_object is None:
        sub_title = f'Star map ({filter_} filter)'
    elif rts is None:
        sub_title = f'{name_object} - {filter_} filter'
    elif name_object is None:
        sub_title = f'{filter_} filter, {rts}'
    else:
        sub_title = f'{name_object} - {filter_} filter, {rts}'

    fig.suptitle(sub_title, fontsize=17)

    #   Set up normalization for the image
    norm = ImageNormalize(image, interval=ZScaleInterval(contrast=0.15, ))

    #   Display the actual image
    ax.imshow(
        image,
        cmap='PuBu',
        origin='lower',
        norm=norm,
        interpolation='nearest',
    )

    x, y = _xy_values(tbl, x_column, y_column)
    ax.scatter(
        x,
        y,
        s=scatter_s,
        facecolors=(0.5, 0., 0.5, 0.2),
        edgecolors=(0.5, 0., 0.5, 0.7),
        lw=0.9,
        label=label,
        **overlay,
    )
    x2 = y2 = None
    if tbl_2 is not None:
        x2, y2 = _xy_values(tbl_2, x_column_2, y_column_2)
        ax.scatter(
            x2,
            y2,
            s=scatter_s,
            facecolors=(0., 0.7, 0.35, 0.2),
            edgecolors=(0., 0.7, 0.35, 0.7),
            lw=0.9,
            label=label_2,
            **overlay,
        )

    for patch in extra_patches or ():
        if patch is None:
            continue
        drawn = Ellipse(
            _ellipse_center(patch),
            width=float(patch.width),
            height=float(patch.height),
            angle=float(getattr(patch, "angle", 0.0)),
        )
        drawn.set_facecolor("none")
        drawn.set_edgecolor((0.0, 0.55, 0.25, 0.9))
        drawn.set_linewidth(1.4)
        if overlay:
            drawn.set_transform(overlay["transform"])
        ax.add_patch(drawn)

    _set_pixel_limits(ax, ny, nx)

    def _annotate_xy(xs, ys, texts, color="purple"):
        for i, (xi, yi) in enumerate(zip(xs, ys, strict=True)):
            ax.text(
                xi + 11,
                yi + 8,
                f" {texts[i]}",
                fontdict=style.font,
                color=color,
                **overlay,
            )

    if mode == 'mags':
        if magnitude_column is not None:
            magnitudes = tbl[magnitude_column]
        else:
            try:
                magnitudes = tbl['mag_cali_trans']
            except KeyError:
                magnitudes = tbl['mag_cali_no-trans']
        mag_text = []
        for mag_i in magnitudes:
            if hasattr(mag_i, 'value'):
                mag_i = mag_i.value
            mag_text.append(f"{float(mag_i):.1f}")
        _annotate_xy(x, y, mag_text)
    elif mode == 'list':
        _annotate_xy(x, y, [str(i) for i in range(len(x))])
    else:
        if tbl_2 is not None and x2 is not None:
            if 'id' in tbl_2.colnames:
                texts = [str(v) for v in tbl_2['id']]
            else:
                texts = [str(i) for i in range(len(x2))]
            _annotate_xy(x2, y2, texts, color="green")
        elif 'id' in tbl.colnames and len(x) <= _ID_LABEL_MAX:
            _annotate_xy(x, y, [str(v) for v in tbl['id']])

    #   Define the ticks
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
    )
    ax.minorticks_on()

    #   Set labels
    if wcs_image is not None:
        ax.set_xlabel("Right ascension", fontsize=16)
        ax.set_ylabel("Declination", fontsize=16)
    else:
        ax.set_xlabel("[pixel]", fontsize=16)
        ax.set_ylabel("[pixel]", fontsize=16)

    #   Enable grid for WCS
    # if wcs is not None:
    ax.grid(True, color='white', linestyle='--')

    #   Plot legend
    ax.legend(
        bbox_to_anchor=(0., 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.,
    )

    #   Write the plot to disk (title may include image basename; filename must not)
    stem_source = filename_suffix if filename_suffix is not None else rts
    if stem_source is None:
        plt.savefig(
            f'{output_dir}/starmaps/starmap_{filter_}.{file_type}',
            bbox_inches='tight',
            format=file_type,
        )
    else:
        stem = _sanitize_starmap_filename_part(stem_source)
        plt.savefig(
            f"{output_dir}/starmaps/starmap_{filter_}_{stem}.{file_type}",
            bbox_inches='tight',
            format=file_type,
        )
    # plt.show()
    plt.close()



def plot_limiting_mag_sky_apertures(
        output_dir: str, img_data: np.ndarray, mask: np.ndarray,
        image_depth: ImageDepth, file_type: str = 'pdf') -> None:
    """
    Plot the sky apertures that are used to estimate the limiting magnitude

    Parameters
    ----------
    output_dir
        Output directory

    img_data
        Image data

    mask
        Indicating the position of detected objects

    image_depth
        Object used to derive the limiting magnitude

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    from scipy.ndimage import binary_dilation

    out = diagnostics_dir(output_dir, "extraction")

    # ImageDepth places apertures on the *dilated* mask; show that for QC.
    mask_bool = np.asarray(mask, dtype=bool)
    if np.any(mask_bool) and getattr(image_depth, "dilate_footprint", None) is not None:
        mask_show = binary_dilation(mask_bool, structure=image_depth.dilate_footprint)
    else:
        mask_show = mask_bool

    #   Plot magnitudes
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

    #   Set title
    ax[0].set_title('Data with blank apertures')
    ax[1].set_title('Dilated source mask + blank apertures')

    #   Normalize the image data and plot
    norm = ImageNormalize(img_data, interval=ZScaleInterval(contrast=0.15, ))
    ax[0].imshow(
        img_data,
        norm=norm,
        cmap='PuBu',
        interpolation='nearest',
        origin='lower',
    )

    #   Plot mask with object positions (as used for aperture placement)
    ax[1].imshow(
        mask_show,
        interpolation='none',
        origin='lower',
    )

    #   Plot apertures used to derive limiting magnitude
    if getattr(image_depth, "apertures", None):
        image_depth.apertures[0].plot(ax[0], color='purple', lw=0.2)
        image_depth.apertures[0].plot(ax[1], color='orange', lw=0.2)

    plt.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.05,
        top=0.95,
        wspace=0.2,
    )

    #   Set labels
    label_font_size = 10
    ax[0].set_xlabel("[pixel]", fontsize=label_font_size)
    ax[0].set_ylabel("[pixel]", fontsize=label_font_size)
    ax[1].set_xlabel("[pixel]", fontsize=label_font_size)
    ax[1].set_ylabel("[pixel]", fontsize=label_font_size)

    #   Save plot
    plt.savefig(
        f'{out}/limiting_mag_sky_regions.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()


def _sky_ellipse_from_simbad(
        center: SkyCoord, wcs_image: wcs.WCS,
        major_arcmin: float, minor_arcmin: float, pa_deg: float,
        ) -> Ellipse | None:
    """Sky ellipse in SIMBAD PA (East of North) converted to pixel coordinates."""
    if major_arcmin <= 0 or minor_arcmin <= 0:
        return None
    pixel_ellipse = EllipseSkyRegion(
        center=center,
        width=major_arcmin * u.arcmin,
        height=minor_arcmin * u.arcmin,
        angle=pa_deg * u.deg,
    ).to_pixel(wcs_image)
    angle = pixel_ellipse.angle
    angle_deg = (
        float(angle.to(u.deg).value) if hasattr(angle, "to") else float(angle)
    )
    return Ellipse(
        (pixel_ellipse.center.x, pixel_ellipse.center.y),
        width=float(pixel_ellipse.width),
        height=float(pixel_ellipse.height),
        angle=angle_deg,
    )


def _simbad_extent_ellipse(
    obj,
    coord: SkyCoord,
    wcs_image: wcs.WCS,
    image_shape: tuple[int, ...],
) -> Ellipse | None:
    axes = simbad_angular_axes_arcmin(obj)
    if axes is None:
        return None
    try:
        ellipse = _sky_ellipse_from_simbad(coord, wcs_image, *axes)
    except (AttributeError, TypeError, ValueError):
        return None
    if ellipse is None or not _ellipse_fits_image(ellipse, image_shape):
        return None
    return ellipse


def _legend_marker(legend_elements, label, color, marker, markersize):
    if any(e.get_label() == label for e in legend_elements):
        return
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markerfacecolor='none',
            markersize=markersize,
            linestyle='None',
            label=label,
        )
    )


def plot_annotated_image(
        image_data: np.ndarray, wcs_image: wcs.WCS, simbad_objects: Table,
        output_dir: Path, filter_: str, file_type: str = 'pdf',
        filter_mag: str | None = None, mag_limit: float | None = None,
        marker_scale: float = 1.0,
    ) -> None :
    """
    Visualises the image and marks objects from the Simbad database.

    Magnitude filtering is applied at query time; ``filter_mag`` / ``mag_limit``
    are kept for callers and ignored here.
    """
    del filter_mag, mag_limit
    out = results_dir(output_dir, "starmaps")
    ny, nx = int(image_data.shape[0]), int(image_data.shape[1])
    figsize = _starmap_figsize(ny, nx)
    scale = _marker_scale_from_figsize(figsize, marker_scale)
    star_ms = _REF_ANNOT_MARKERSIZE * scale
    extended_ms = _REF_EXTENDED_MARKERSIZE * scale
    center_ms = _REF_CENTER_MARKERSIZE * scale

    _fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': wcs_image})
    overlay = _overlay_kwargs(ax)

    norm = ImageNormalize(image_data, interval=ZScaleInterval(contrast=0.1, ))
    ax.imshow(
        image_data,
        cmap='gray',
        origin='lower',
        norm=norm,
        interpolation='nearest',
    )
    ax.tick_params(axis='both', which='both', direction='in')
    ax.minorticks_on()
    ax.set_xlabel("Right ascension", fontsize=16)
    ax.set_ylabel("Declination", fontsize=16)
    ax.grid(True, color='white', linestyle='--')
    _set_pixel_limits(ax, ny, nx)

    legend_elements = []

    table = simbad_objects.copy()
    for old, new in (
        ("ra", "RA"),
        ("dec", "DEC"),
        ("otype", "OTYPE"),
        ("main_id", "MAIN_ID"),
    ):
        if old in table.colnames and new not in table.colnames:
            table.rename_column(old, new)

    name_index = 0
    for obj in table:
        ra, dec = obj['RA'], obj['DEC']
        obj_type = obj['OTYPE']
        name = obj['MAIN_ID']
        if isinstance(name, bytes):
            name = name.decode()

        coord = skycoord_from_simbad(ra, dec)
        x, y = wcs_image.world_to_pixel(coord)
        x = float(np.asarray(x).ravel()[0])
        y = float(np.asarray(y).ravel()[0])

        if not (0 <= x < nx and 0 <= y < ny):
            continue

        kind = simbad_overlay_kind(obj_type)
        plot_marker = True
        marker_size = star_ms
        ellipse = None
        if kind == "star":
            color, marker = 'lightblue', '*'
            _legend_marker(legend_elements, 'Star', color, marker, 8)
        elif kind == "cluster":
            color, marker = 'gold', 'h'
            ellipse = _simbad_extent_ellipse(obj, coord, wcs_image, image_data.shape)
            marker_size = center_ms if ellipse is not None else extended_ms
            _legend_marker(legend_elements, 'Cluster', color, marker, 8)
        elif kind == "galaxy":
            color, marker = 'lightsalmon', 's'
            ellipse = _simbad_extent_ellipse(obj, coord, wcs_image, image_data.shape)
            marker_size = center_ms if ellipse is not None else extended_ms
            _legend_marker(legend_elements, 'Galaxy', color, marker, 8)
        elif kind == "nebula":
            color, marker = 'lightpink', 'o'
            ellipse = _simbad_extent_ellipse(obj, coord, wcs_image, image_data.shape)
            marker_size = center_ms if ellipse is not None else extended_ms
            _legend_marker(legend_elements, 'Nebula', color, marker, 8)
        else:
            color, marker = 'lightgreen', 'H'
            name = f'{name} ({obj_type})'
            _legend_marker(legend_elements, 'Other', color, marker, 8)

        if ellipse is not None:
            ellipse.set_edgecolor(color)
            ellipse.set_facecolor('none')
            ellipse.set_linewidth(1.5)
            ellipse.set_alpha(0.7)
            if overlay:
                ellipse.set_transform(overlay["transform"])
            ax.add_patch(ellipse)

        if plot_marker:
            ax.plot(
                x,
                y,
                marker=marker,
                markerfacecolor='none',
                markeredgecolor=color,
                markeredgewidth=1.2,
                markersize=marker_size,
                alpha=0.8,
                **overlay,
            )
        dx = 18.0 if name_index % 2 == 0 else -18.0
        dy = 0.0 if name_index % 4 < 2 else 12.0
        ax.text(
            x + dx,
            y + dy,
            name,
            color=color,
            fontsize=8,
            alpha=0.9,
            verticalalignment='center',
            horizontalalignment='left' if dx > 0 else 'right',
            weight="bold",
            **overlay,
        )
        name_index += 1

    ax.legend(
        bbox_to_anchor=(0., 1.02, 1.0, 0.102),
        loc=3,
        handles=legend_elements,
        ncol=5,
        fontsize=8,
        frameon=True,
        mode='expand',
        borderaxespad=0.,
    )
    plt.savefig(
        out / f'annotated_starmap_{filter_}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()


