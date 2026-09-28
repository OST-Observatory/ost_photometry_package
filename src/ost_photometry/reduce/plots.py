############################################################################
#                               Libraries                                  #
############################################################################

from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.visualization import hist, simple_norm
from matplotlib import pyplot as plt
from photutils.psf import EPSFStars
from scipy import stats

from .. import checks, terminal_output
from ..output_layout import diagnostics_dir

############################################################################
#                           Routines & definitions                         #
############################################################################

#   Frame-quality plot palette: one series hue for the metric, status colors
#   for rejected frames, neutral ink for text / thresholds.
_FQ_SERIES = "#2a78d6"
_FQ_REJECTED = "#d03b3b"
_FQ_REFERENCE = "#eda100"
_FQ_INK = "#0b0b0b"
_FQ_INK_SECONDARY = "#52514e"
_FQ_GRID = "#e6e5e1"


def _safe_filter_name(filter_: str) -> str:
    return str(filter_).replace("''", "p").replace("/", "_").replace(" ", "_")


def frame_quality_overview(
    rows: dict[str, np.ndarray],
    output_dir: str | Path,
    filter_: str,
    *,
    fwhm_max: float | None = None,
    fwhm_unit: str = "px",
    alignment_known: bool = False,
) -> Path:
    """
    Per-filter frame-quality overview: FWHM, roundness, star count and sky
    background against the frame index, with rejected and reference frames
    marked.

    Parameters
    ----------
    rows
        Plain arrays (no astropy Table, so the plot can run in a child
        process): ``file`` (basenames), ``fwhm_px``, ``fwhm_arcsec``,
        ``roundness``, ``n_stars``, ``background``, ``rejected`` (bool),
        ``is_reference`` (bool), ``aligned`` (bool). Frames are plotted in
        the given order (sorted by observation time upstream).

    output_dir
        Reduction output directory; the PDF goes to
        ``<output_dir>/diagnostics/frame_quality/``.

    filter_
        Filter name (used in the title and file name).

    fwhm_max
        Optional rejection threshold drawn on the FWHM panel, in
        ``fwhm_unit``.

    fwhm_unit
        ``px`` or ``arcsec``; selects which FWHM column is shown.

    alignment_known
        If ``True`` kept frames that could not be aligned are drawn as
        hollow markers.

    Returns
    -------
    path
        Path of the written PDF.
    """
    files = [str(f) for f in rows["file"]]
    n = len(files)
    index = np.arange(n)
    rejected = np.asarray(rows["rejected"], dtype=bool)
    reference = np.asarray(rows["is_reference"], dtype=bool)
    aligned = np.asarray(rows.get("aligned", np.ones(n, dtype=bool)), dtype=bool)
    kept = ~rejected
    not_aligned = kept & ~aligned if alignment_known else np.zeros(n, dtype=bool)

    fwhm_column = "fwhm_arcsec" if fwhm_unit == "arcsec" else "fwhm_px"
    fwhm_values = np.asarray(rows[fwhm_column], dtype=float)
    if fwhm_unit == "arcsec" and not np.any(np.isfinite(fwhm_values)):
        fwhm_values = np.asarray(rows["fwhm_px"], dtype=float)
        fwhm_unit = "px"
        fwhm_max = None
    panels = [
        ("FWHM [arcsec]" if fwhm_unit == "arcsec" else "FWHM [pixel]", fwhm_values),
        ("roundness", np.asarray(rows["roundness"], dtype=float)),
        ("stars detected", np.asarray(rows["n_stars"], dtype=float)),
        ("sky background", np.asarray(rows["background"], dtype=float)),
    ]

    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=1,
        sharex=True,
        figsize=(max(7.0, min(0.28 * n + 3.0, 22.0)), 9.5),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    for ax, (label, values) in zip(axes, panels, strict=True):
        finite = np.isfinite(values)
        ax.plot(
            index[finite],
            values[finite],
            color=_FQ_SERIES,
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
        ax.plot(
            index[kept & finite & ~not_aligned],
            values[kept & finite & ~not_aligned],
            linestyle="none",
            marker="o",
            markersize=6,
            color=_FQ_SERIES,
            zorder=3,
            label="kept",
        )
        if np.any(not_aligned & finite):
            ax.plot(
                index[not_aligned & finite],
                values[not_aligned & finite],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=_FQ_SERIES,
                markeredgewidth=1.5,
                zorder=3,
                label="kept, not aligned",
            )
        if np.any(rejected & finite):
            ax.plot(
                index[rejected & finite],
                values[rejected & finite],
                linestyle="none",
                marker="x",
                markersize=8,
                markeredgewidth=1.8,
                color=_FQ_REJECTED,
                zorder=4,
                label="rejected",
            )
        if np.any(reference & finite):
            ax.plot(
                index[reference & finite],
                values[reference & finite],
                linestyle="none",
                marker="*",
                markersize=13,
                markerfacecolor=_FQ_REFERENCE,
                markeredgecolor=_FQ_INK,
                markeredgewidth=0.8,
                zorder=5,
                label="reference",
            )
        ax.set_ylabel(label, color=_FQ_INK)
        ax.grid(True, color=_FQ_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_FQ_INK_SECONDARY)
        ax.tick_params(colors=_FQ_INK_SECONDARY)

    if fwhm_max is not None and np.isfinite(fwhm_max):
        axes[0].axhline(
            float(fwhm_max),
            color=_FQ_INK_SECONDARY,
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
        axes[0].annotate(
            f"fwhm_max = {float(fwhm_max):g} {fwhm_unit}",
            xy=(0.995, float(fwhm_max)),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=_FQ_INK_SECONDARY,
        )

    #   Legend once, on the top panel; entries de-duplicated.
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        seen.setdefault(label, handle)
    axes[0].legend(
        list(seen.values()),
        list(seen.keys()),
        loc="upper left",
        fontsize=8,
        frameon=False,
        ncol=len(seen),
        bbox_to_anchor=(0.0, 1.22),
    )

    #   Frame names as tick labels; thin them out for long series.
    step = max(1, int(np.ceil(n / 40.0)))
    axes[-1].set_xticks(index[::step])
    axes[-1].set_xticklabels(files[::step], rotation=90, fontsize=7, color=_FQ_INK_SECONDARY)
    axes[-1].set_xlabel("frame (observation order)", color=_FQ_INK)
    axes[-1].set_xlim(-0.6, max(n - 0.4, 0.6))

    n_rejected = int(np.count_nonzero(rejected))
    fig.suptitle(
        f"Frame quality, filter {filter_}: {n} frames, {n - n_rejected} kept, "
        f"{n_rejected} rejected",
        color=_FQ_INK,
        fontsize=12,
        y=1.03,
    )

    out_dir = diagnostics_dir(output_dir, "frame_quality")
    path = out_dir / f"frame_quality_{_safe_filter_name(filter_)}.pdf"
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return path


def cross_correlation_matrix(
        image_data: np.ndarray, cross_correlation_data: np.ndarray) -> None:
    """
    Debug plot showing the cc matrix, created during image correlation

    Parameters
    ----------
    image_data
        Image data array

    cross_correlation_data
        Array with the data of the cc matrix
    """
    #   Norm of image
    norm = simple_norm(image_data, 'log', percent=99.)

    #   Initialize sub plots
    plt.subplot(121)

    #   Plot image
    plt.imshow(image_data, norm=norm, cmap='gray')

    #   Set title & ticks
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])

    #   Norm of cc matrix
    norm = simple_norm(
        np.absolute(cross_correlation_data),
        'log',
        percent=99.,
    )

    #   Plot cc matrix
    plt.subplot(122)
    plt.imshow(
        np.absolute(cross_correlation_data),
        norm=norm,
        cmap='gray',
    )

    #   Set title & ticks
    plt.title('cc')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_dark_with_distributions(
        image_data: np.ndarray, read_noise: float, dark_current: float,
        output_dir: Path, n_images: int = 1, exposure_time: float = 1.,
        gain: float = 1., show_poisson_distribution: bool = True,
        show_gaussian_distribution: bool = True) -> None:
    """
    Plot the distribution of dark pixel values, optionally over-plotting
    the expected Poisson and normal distributions corresponding to dark
    current only or read noise only.

    Parameters
    ----------
    image_data
        Image data

    read_noise
        The read noise, in electrons

    dark_current
        The dark current in electrons/sec/pixel

    output_dir
        Path pointing to the main storage location

    n_images
        If the image is formed from the average of some number of dark
        frames then the resulting Poisson distribution depends on the
        number of images, as does the expected standard deviation of the
        Gaussian.
        Default is ``1``.

    exposure_time
        Exposure time, in seconds
        Default is ``1.``.

    gain
        The gain of the camera, in electron/ADU
        Default is ``1.``.

    show_poisson_distribution
        If ``True``, over plot a Poisson distribution with mean equal to
        the expected dark counts for the number of images
        Default is ``True``.

    show_gaussian_distribution
        If ``True``, over plot a normal distribution with mean equal to the
        expected dark counts and standard deviation equal to the read
        noise, scaled as appropriate for the number of images
        Default is ``True``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'reduce_plots',
    )

    #   Scale image
    image_data = image_data * gain / exposure_time

    #   Use bmh style
    # plt.style.use('bmh')

    #   Set layout of image
    plt.figure(figsize=(20, 9))

    #   Get
    plt.hist(
        image_data.flatten(),
        bins=20,
        align='mid',
        density=True,
        label="Dark frame",
    )

    #   Expected mean of the dark
    expected_mean_dark = dark_current * exposure_time / gain

    #   Plot Poisson
    if show_poisson_distribution:
        #   Account for number of exposures
        poisson_distribution = stats.poisson(expected_mean_dark * n_images)

        #   X range
        x_axis_poisson = np.arange(0, 300, 1)

        #   Prepare normalization
        new_area = np.sum(
            1 / n_images * poisson_distribution.pmf(x_axis_poisson)
        )

        plt.plot(
            x_axis_poisson / n_images,
            poisson_distribution.pmf(x_axis_poisson) / new_area,
            label=f"Poisson distribution, mean of {expected_mean_dark:5.2f} "
                  f"counts",
        )

    #   Plot Gaussian
    if show_gaussian_distribution:
        #   The expected width of the Gaussian depends on the number of images
        expected_scale = read_noise / gain * np.sqrt(n_images)

        #   Mean value is same as for the Poisson distribution (account for
        #   number of images)
        expected_mean = expected_mean_dark * n_images

        #
        gauss = stats.norm(loc=expected_mean, scale=expected_scale)

        #   X range
        x_axis_gauss = np.linspace(
            expected_mean - 5 * expected_scale,
            expected_mean + 5 * expected_scale,
            num=100,
        )

        plt.plot(
            x_axis_gauss / n_images,
            gauss.pdf(x_axis_gauss) * n_images,
            label='Gaussian, standard dev is read noise in counts',
        )

    #   Labels
    plt.xlabel(f"Dark counts in {exposure_time} sec exposure")
    plt.ylabel("Fraction of pixels (area normalized to 1)")
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'dark_with_distributions_{}.pdf'.format(
        str(exposure_time).replace("''", "p")
    )
    plt.savefig(
        output_dir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def plot_histogram(
        image_data: np.ndarray, output_dir: Path, gain: int,
        exposure_time: float) -> None:
    """
    Plot image histogram for dark images

    Parameters
    ----------
    image_data
        Dark frame to histogram

    output_dir
        Path pointing to the main storage location

    gain
        The gain of the camera, in electron/ADU

    exposure_time
        Exposure time, in seconds
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'reduce_plots',
    )

    #   Scale image
    image_data = image_data * gain / exposure_time

    #   Use bmh style
    # plt.style.use('bmh')

    #   Set layout of image
    plt.figure(figsize=(20, 9))

    #   Create histogram
    hist(
        image_data.flatten(),
        bins=5000,
        density=False,
        label=f'{exposure_time} sec dark',
        alpha=0.4,
    )

    #   Labels
    plt.xlabel('Dark current, $e^-$/sec')
    plt.ylabel('Number of pixels')
    plt.loglog()
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'dark_hist_{}.pdf'.format(
        str(exposure_time).replace("''", "p")
    )
    plt.savefig(
        output_dir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def plot_median_of_flat_fields(
        image_file_collection: ccdp.ImageFileCollection,
        image_type: str | list[str] | None, output_dir: Path, filter_: str) -> None:
    """
    Plot median and mean of each flat field in a file collection

    Parameters
    ----------
    image_file_collection
        File collection with the flat fields to analyze

    image_type
        Header keyword characterizing the flats

    output_dir
        Path pointing to the main storage location

    filter_
        Filter

    Idea/Reference
    --------------
        # https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'reduce_plots',
    )

    #   Calculate median and mean for each image
    median_count = []
    mean_count = []
    if isinstance(image_type, str):
        for data in image_file_collection.data(imagetyp=image_type, filter=filter_):
            median_count.append(np.median(data))
            mean_count.append(np.mean(data))
    elif isinstance(image_type, list):
        for type in image_type:
            for data in image_file_collection.data(imagetyp=type, filter=filter_):
                    median_count.append(np.median(data))
                    mean_count.append(np.mean(data))
    elif image_type is None:
        terminal_output.print_to_terminal(
            "PLot of the median flat field not possible, because image_type "
            "is None.",
            style_name='WARNING',
        )
        return
    else:
        terminal_output.print_to_terminal(
            f"PLot of the median flat field not possible, because the data "
            f"type of the variable image_type is not known: Current type "
            f"is {type(image_type)}",
            style_name='WARNING',
        )
        return

    #   Use bmh style
    # plt.style.use('bmh')

    #   Set layout of image
    plt.figure(figsize=(20, 9))

    #   Plot mean & median
    plt.plot(median_count, label='median')
    plt.plot(mean_count, label='mean')

    #   Plot labels
    plt.xlabel('Image number')
    plt.ylabel('Count (ADU)')
    plt.title('Pixel value in calibrated flat frames')
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'flat_median_{}.pdf'.format(filter_.replace("''", "p"))
    plt.savefig(
        output_dir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def cutouts_fwhm_stars(
        output_dir: Path, n_stars: int, sub_images_fwhm_stars: EPSFStars,
        filter_: str, basename: str) -> None:
    """
    Plots cutouts around the stars used to estimate the FWHM

    Parameters
    ----------
    output_dir
        Path to the directory where the master files should be saved to

    n_stars
        Number of stars

    sub_images_fwhm_stars
        Sub images (squares) extracted around the FWHM stars

    filter_
        Filter name

    basename
        Name of the image file
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'cutouts',
    )

    #   Set number of rows and columns for the plot
    n_rows = 5
    n_columns = 5

    #   Prepare plot
    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(20, 20),
        squeeze=True,
    )
    ax = ax.ravel()

    #   Set title of the complete plot
    fig.suptitle(
        f'Cutouts of the FWHM stars ({filter_}), {basename})',
        fontsize=20,
    )

    #   Loop over the cutouts (default: 25)
    for i in range(n_stars):
        # Set up normalization for the image
        norm = simple_norm(sub_images_fwhm_stars[i], 'log', percent=99.)

        # Plot individual cutouts
        ax[i].set_xlabel("[pixel]")
        ax[i].set_ylabel("[pixel]")
        ax[i].imshow(
            sub_images_fwhm_stars[i],
            norm=norm,
            origin='lower',
            cmap='viridis',
        )

    #   Write the plot to disk
    plt.savefig(
        f'{output_dir}/cutouts/cutouts_FWHM-stars_{filter_}_{basename}.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def aberration_inspector(
        image_data: np.ndarray, output_dir: Path, filter_: str,
        cutout_size_percent: float | int = 15,
        border_cutouts_percent: float | int = 3) -> None:
    """
    Crop and display the edges and center of an image

    Parameters
    ----------
    image_data
        2D image data array

    output_dir
        Path to the directory where the master files should be saved to

    filter_
        Filter name

    cutout_size_percent
        Cutout size as a percentage of the Y dimension of the image

    border_cutouts_percent
        Size of the borders around the cutouts as a percentage of the
        Y dimension of the image
    """
    #   Image dimensions and center
    data_shape = image_data.shape
    y_dimension = data_shape[0]
    x_dimension = data_shape[1]

    y_center = int(y_dimension / 2)
    x_center = int(x_dimension / 2)

    #   Cutout dimension
    cutout_fraction = cutout_size_percent / 100
    cutout_dimension = int(data_shape[0] * cutout_fraction)
    half_cutout_dimension = int(cutout_dimension / 2)

    #   Cutouts
    upper_left_edge = image_data[0:cutout_dimension, 0:cutout_dimension]
    upper_right_edge = image_data[0:cutout_dimension, -cutout_dimension:]
    lower_right_edge = image_data[-cutout_dimension:, -cutout_dimension:]
    lower_left_edge = image_data[-cutout_dimension:, 0:cutout_dimension]
    center = image_data[
             y_center - half_cutout_dimension:y_center + half_cutout_dimension,
             x_center - half_cutout_dimension:x_center + half_cutout_dimension
             ]

    #   Size of the borders between cutouts
    border_cutouts_scale_factor = border_cutouts_percent / 100
    half_border_size = cutout_dimension * border_cutouts_scale_factor

    #   New array to plot
    dimension_cutout_array = int(
        2 * cutout_dimension + cutout_dimension * border_cutouts_scale_factor
    )
    cutout_array = np.ones((dimension_cutout_array, dimension_cutout_array))

    #   Fill new array with cutouts
    cutout_array[0:cutout_dimension, 0:cutout_dimension] = upper_left_edge
    cutout_array[0:cutout_dimension, -cutout_dimension:] = upper_right_edge
    cutout_array[-cutout_dimension:, -cutout_dimension:] = lower_right_edge
    cutout_array[-cutout_dimension:, 0:cutout_dimension] = lower_left_edge

    if dimension_cutout_array % 2 == 0:
        xy_center_cutout_array = int(dimension_cutout_array / 2 + 1)
    else:
        xy_center_cutout_array = int((dimension_cutout_array - 1) / 2 + 1)

    center_start = xy_center_cutout_array - half_cutout_dimension
    center_end = xy_center_cutout_array + half_cutout_dimension
    cutout_array[center_start:center_end, center_start:center_end] = center

    #   Add borders to central cutout

    cutout_array[
        int(center_start - half_border_size / 2):int(center_start + half_border_size / 2),
        int(center_start - half_border_size / 2):int(center_end + half_border_size / 2)
    ] = 1.
    cutout_array[
        int(center_end - half_border_size / 2):int(center_end + half_border_size / 2),
        int(center_start - half_border_size / 2):int(center_end + half_border_size / 2)
    ] = 1.
    cutout_array[
        center_start:center_end,
        int(center_start - half_border_size / 2):int(center_start + half_border_size / 2)
    ] = 1.
    cutout_array[
        center_start:center_end,
        int(center_end - half_border_size / 2):int(center_end + half_border_size / 2)
    ] = 1.

    #   Define figure
    plt.figure(figsize=(12, 12))

    #   Image normalization
    image_normalization = simple_norm(
        cutout_array,
        stretch='log',
        min_percent=1,
        percent=99.9,
    )

    #   Plot data
    plt.imshow(
        cutout_array,
        norm=image_normalization,
        cmap='Greys',
        origin='lower',
    )

    plt.axis('off')

    #   Write the plot to disk
    plt.savefig(
        f'{output_dir}/aberration/aberration_control_cutouts_{filter_}.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()
