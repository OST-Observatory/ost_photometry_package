"""Colour images from per-filter stacks (ost_photometry.imaging)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table

from ost_photometry.imaging import (
    ColorSettings,
    canonical_filter,
    compose,
    resolve_palette,
    stacks_from_directory,
    stacks_from_summary,
    write_images,
)
from ost_photometry.imaging.background import (
    cosmetic_correction,
    neutralize_background,
    remove_gradient,
    sky_level,
    source_mask,
)
from ost_photometry.imaging.color_calibration import star_white_balance
from ost_photometry.imaging.composite import common_field
from ost_photometry.imaging.stretch import apply_stretch, stretch_parameters

SHAPE = (300, 400)


def _stars(rng, n=30, shape=SHAPE, sigma=1.6, separation=22.0):
    yy, xx = np.indices(shape)
    image = np.zeros(shape)
    positions = []
    for _ in range(20 * n):  # isolated stars, as many as fit (at most n)
        if len(positions) == n:
            break
        x, y = rng.uniform(15, shape[1] - 15), rng.uniform(15, shape[0] - 15)
        if all((x - a) ** 2 + (y - b) ** 2 > separation**2 for a, b in positions):
            positions.append((x, y))
    for x, y in positions:
        image += rng.uniform(200, 2000) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return image


def _frame(stars, factor, sky, rng, noise=1.0):
    return (sky + factor * stars + rng.normal(0, noise, stars.shape)).astype(np.float32)


def _write_stacks(tmp_path, filters, *, factors=None, skies=None, seed=1):
    rng = np.random.default_rng(seed)
    stars = _stars(rng)
    paths = {}
    for i, filt in enumerate(filters):
        factor = (factors or {}).get(filt, 1.0)
        sky = (skies or {}).get(filt, 10.0 + i)
        ccd = CCDData(_frame(stars, factor, sky, rng), unit="electron/s")
        ccd.meta.update({"FILTER": filt, "EXPTIME": 600.0, "FWHMMED": 3.8,
                         "SATLEVEL": 1e9, "OBJECT": "Test"})
        path = tmp_path / f"combined_filter_{filt.replace(' ', '_')}.fit"
        ccd.write(path)
        paths[filt] = str(path)
    return paths


def test_canonical_filters():
    assert canonical_filter("H alpha") == canonical_filter("Halpha") == "Ha"
    assert canonical_filter("O III") == "OIII" and canonical_filter("S II") == "SII"
    assert canonical_filter("Clear") == canonical_filter("C") == "L"
    assert canonical_filter("V") == "V" and canonical_filter("r") == "R"
    assert canonical_filter("Weird") == "Weird"


def test_resolve_palette():
    assert resolve_palette("auto", {"R", "V", "B"}).name == "rgb"
    assert resolve_palette("auto", {"R", "V", "B"}).mix["G"] == {"V": 1.0}
    assert resolve_palette("auto", {"R", "G", "B", "L"}).name == "lrgb"
    sho = resolve_palette("auto", {"SII", "Ha", "OIII"})
    assert sho.name == "sho" and not sho.linked_stretch and not sho.broadband
    assert resolve_palette("auto", {"Ha", "OIII"}).mix["B"] == {"OIII": 1.0}
    custom = resolve_palette("custom", {"Ha", "R", "V", "B"},
                             custom_mix={"R": {"H alpha": 0.8, "R": 0.2}, "G": {"V": 1},
                                         "B": {"B": 1}})
    assert custom.mix["R"] == {"Ha": 0.8, "R": 0.2} and custom.linked_stretch
    with pytest.raises(ValueError, match="No palette"):
        resolve_palette("auto", {"V"})
    with pytest.raises(ValueError, match="needs filter"):
        resolve_palette("sho", {"Ha", "OIII"})


def test_autostretch_maps_the_sky_to_the_target():
    rng = np.random.default_rng(0)
    data = 50.0 + rng.normal(0, 2.0, (300, 300))
    data[100:110, 100:110] += 5000.0
    params = stretch_parameters(data, method="auto", target_background=0.25)
    out = apply_stretch(data, params)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.median(out) == pytest.approx(0.25, abs=0.02)
    ramp = np.linspace(params["black"], params["white"], 50)
    assert np.all(np.diff(apply_stretch(ramp, params)) >= 0)
    for method in ("asinh", "linear"):
        stretched = apply_stretch(data, stretch_parameters(data, method=method))
        assert 0.0 <= stretched.min() and stretched.max() <= 1.0


def test_neutralize_background_and_gradient():
    rng = np.random.default_rng(2)
    stars = _stars(rng)
    yy, xx = np.indices(SHAPE)
    gradient = 0.05 * xx + 0.02 * yy
    red = _frame(stars, 1.0, 30.0, rng) + gradient.astype(np.float32)
    blue = _frame(stars, 1.0, 5.0, rng)
    out, levels = neutralize_background({"R": red, "B": blue})
    assert abs(sky_level(out["R"])[0]) < 0.5 and abs(sky_level(out["B"])[0]) < 0.5
    fixed, model = remove_gradient(red)
    left, right = fixed[:, 10:40], fixed[:, -40:-10]
    assert abs(sky_level(left)[0] - sky_level(right)[0]) < 1.0  # was ~13
    assert model.shape == SHAPE
    with pytest.raises(ValueError, match="method"):
        remove_gradient(red, method="spline")


def _flat_residual_scene(rng):
    """Sky with a bright band along the top border (flat residual), an
    extended object in the middle, stars and noise."""
    yy, xx = np.indices(SHAPE)
    band = 4.0 * np.exp(-((yy - SHAPE[0]) ** 2) / (2 * 60.0**2))
    nebula = 6.0 * np.exp(-((xx - 200) ** 2 + (yy - 150) ** 2) / (2 * 25.0**2))
    data = _frame(_stars(rng), 1.0, 100.0, rng) + (band + nebula).astype(np.float32)
    return data, band, nebula


def test_gradient_touching_the_border_is_sky_not_an_object():
    data, band, nebula = _flat_residual_scene(np.random.default_rng(5))
    top = (slice(SHAPE[0] - 30, SHAPE[0] - 5), slice(20, -20))
    centre = (slice(140, 160), slice(190, 210))
    # without the option the bright band is masked like an object
    assert source_mask(data)[top].mean() > 0.9
    mask = source_mask(data, border_background_fraction=0.01)
    assert mask[top].mean() < 0.2
    assert mask[centre].all()
    fixed, _model = remove_gradient(data, method="background2d", box=40)
    # before: the band was masked as an object and stayed (about 3.4)
    old, _ = remove_gradient(data, method="background2d", box=40, mask=source_mask(data))

    def step(img):
        return abs(sky_level(img[top])[0] - sky_level(img[100:200, 20:80])[0])

    assert step(old) > 2.5
    assert step(fixed) < 1.5
    sky_mid = sky_level(fixed[100:200, 20:80])[0]
    # the object keeps (most of) its light: it is masked, not modelled
    contrast = float(np.median(fixed[centre]) - sky_mid)
    assert contrast == pytest.approx(float(np.median(nebula[centre])), rel=0.25)


def test_cosmetic_correction_replaces_cold_pixels_only():
    rng = np.random.default_rng(3)
    data = _frame(_stars(rng), 1.0, 20.0, rng)
    data[50, 60] = -100.0
    fixed, n = cosmetic_correction(data)
    assert n >= 1 and fixed[50, 60] > 10.0
    assert np.count_nonzero(fixed != data) <= 5  # star flanks untouched


def test_star_white_balance_recovers_channel_ratios():
    rng = np.random.default_rng(4)
    stars = _stars(rng)
    channels = {"R": _frame(stars, 0.5, 0.0, rng), "G": _frame(stars, 1.0, 0.0, rng),
                "B": _frame(stars, 2.0, 0.0, rng)}
    factors, n = star_white_balance(channels, fwhm=3.8)
    assert n >= 15
    assert factors["G"] == 1.0
    assert factors["R"] == pytest.approx(2.0, rel=0.02)
    assert factors["B"] == pytest.approx(0.5, rel=0.02)


def test_common_field_trims_uncovered_edges():
    valid = np.ones((50, 80), dtype=bool)
    valid[:, -7:] = False
    valid[:3, :] = False
    rows, cols = common_field(valid)
    assert (rows.start, rows.stop, cols.start, cols.stop) == (3, 50, 0, 73)


@pytest.mark.parametrize(
    ("filters", "palette", "extra"),
    [
        (["R", "V", "B"], "auto", {"star_color_calibration": True}),
        (["R", "V", "B", "Clear"], "lrgb", {"remove_gradient": True}),
        (["H alpha", "O III"], "hoo", {}),
        (["S II", "H alpha", "O III"], "sho", {"green_noise_removal": "average_neutral"}),
        (["H alpha", "R", "V", "B"], "custom",
         {"custom_mix": {"R": {"Ha": 0.7, "R": 0.3}, "G": {"V": 1}, "B": {"B": 1}},
          "saturation": 1.3}),
    ],
)
def test_compose_end_to_end(tmp_path, filters, palette, extra):
    stacks = _write_stacks(tmp_path, filters, factors={"R": 0.6, "B": 1.6})
    result = compose(stacks, ColorSettings(palette=palette, **extra))
    assert result.rgb.shape == (*SHAPE, 3) and result.rgb.dtype == np.float32
    assert 0.0 <= result.rgb.min() and result.rgb.max() <= 1.0
    sky = np.median(result.rgb.reshape(-1, 3), axis=0)
    assert np.ptp(sky) < 0.03  # neutral sky (also after SCNR: re-neutralised)
    if extra.get("star_color_calibration"):
        assert result.log["calibration_stars"] >= 15
        stars = result.rgb[result.rgb.mean(axis=2) > 0.8]
        assert np.ptp(np.median(stars, axis=0)) < 0.05  # star colours balanced


def test_compose_rejects_different_grids_and_masks_are_filled(tmp_path):
    stacks = _write_stacks(tmp_path, ["R", "V", "B"])
    with fits.open(stacks["B"], mode="update") as hdul:
        mask = np.zeros(SHAPE, dtype=np.uint8)
        mask[40, 40] = 1
        hdul.append(fits.ImageHDU(mask, name="MASK"))
        hdul[0].data[40, 40] = -1e4
        hdul.flush()
    result = compose(stacks, ColorSettings(palette="rgb"))
    assert result.rgb[40, 40].min() > 0.05  # filled, not a black hole
    other = CCDData(np.zeros((10, 10), dtype=np.float32), unit="electron/s")
    other.meta["FILTER"] = "R"
    other.write(tmp_path / "small.fit")
    with pytest.raises(ValueError, match="one pixel grid"):
        compose({**stacks, "R": str(tmp_path / "small.fit")})


def test_write_images_and_orientation(tmp_path):
    import tifffile
    from PIL import Image

    stacks = _write_stacks(tmp_path, ["R", "V", "B"])
    result = compose(stacks)
    result.rgb[0, :, :] = 1.0  # FITS bottom row: white
    files = write_images(result, tmp_path / "out" / "test_rgb")
    assert sorted(p.suffix for p in files) == [".jpg", ".png", ".tif"]
    tiff = tifffile.imread(tmp_path / "out" / "test_rgb.tif")
    assert tiff.dtype == np.uint16 and tiff.shape == (*SHAPE, 3)
    assert tiff[-1].min() == 65535  # bottom of the file = FITS row 0
    png = np.asarray(Image.open(tmp_path / "out" / "test_rgb.png"))
    assert png.dtype == np.uint8 and png[-1].min() == 255
    with pytest.raises(ValueError, match="format"):
        write_images(result, tmp_path / "x", formats=["gif"])


def test_stack_discovery(tmp_path):
    stacks = _write_stacks(tmp_path, ["R", "V", "B"])
    found = stacks_from_directory(tmp_path)
    assert found == stacks
    summary = Table(rows=[
        ("M57", "qhy600m", "R", "a"), ("M57", "qhy600m", "V", "b"),
        ("M57", "qhy268", "R", "c"), ("M57", "combined", "R", "d"),
        ("M104", "qhy600m", "B", "e"), ("M104", "qhy268", "B", "f"),
    ], names=("target_name", "camera", "filter", "path"))
    sets = {(s.target, s.camera): s.stacks for s in stacks_from_summary(summary)}
    assert sets == {("M57", "combined"): {"R": "d"}, ("M104", "qhy600m"): {"B": "e"},
                    ("M104", "qhy268"): {"B": "f"}}
    only = stacks_from_summary(summary, camera="qhy600m")
    assert {s.target for s in only} == {"M57", "M104"}


def test_scnr_formulas_and_neutral_colours():
    from ost_photometry.imaging import SCNR_METHODS, remove_green_noise

    image = np.array([[[0.2, 0.8, 0.4],   # green excess
                       [0.5, 0.5, 0.5],   # grey
                       [0.6, 0.3, 0.2]]], dtype=np.float32)  # no excess
    out = remove_green_noise(image, preserve_lightness=False)
    assert out[0, 0, 1] == pytest.approx(0.3)  # min(G, (R+B)/2)
    np.testing.assert_allclose(out[0, 1:], image[0, 1:], atol=1e-6)
    np.testing.assert_allclose(out[..., [0, 2]], image[..., [0, 2]])
    out = remove_green_noise(image, method="maximum_neutral", preserve_lightness=False)
    assert out[0, 0, 1] == pytest.approx(0.4)  # min(G, max(R, B))
    out = remove_green_noise(image, method="maximum_mask", amount=0.5, preserve_lightness=False)
    m = 0.4
    assert out[0, 0, 1] == pytest.approx(0.8 * 0.5 * (1 - m) + 0.8 * m)
    out = remove_green_noise(image, method="additive_mask", amount=1.0, preserve_lightness=False)
    assert out[0, 0, 1] == pytest.approx(0.8 * 0.6)  # m = min(1, R + B) = 0.6
    # amount 0 leaves the image unchanged for every method
    for method in SCNR_METHODS:
        np.testing.assert_allclose(
            remove_green_noise(image, method=method, amount=0.0, preserve_lightness=False), image,
            atol=1e-6)
    with pytest.raises(ValueError, match="method"):
        remove_green_noise(image, method="hue")
    with pytest.raises(ValueError, match="amount"):
        remove_green_noise(image, amount=1.5)


def test_scnr_preserves_lightness():
    from skimage.color import rgb2lab

    from ost_photometry.imaging import remove_green_noise

    rng = np.random.default_rng(1)
    image = np.clip(rng.uniform(0.1, 0.6, (20, 20, 3)) + [0.0, 0.15, 0.0], 0, 1).astype(np.float32)
    plain = remove_green_noise(image, preserve_lightness=False)
    kept = remove_green_noise(image, preserve_lightness=True)
    l_in = rgb2lab(image)[..., 0]
    assert np.mean(rgb2lab(plain)[..., 0]) < np.mean(l_in) - 2  # darker without
    np.testing.assert_allclose(rgb2lab(kept)[..., 0], l_in, atol=1.5)
    # green still reduced relative to red and blue
    assert np.mean(kept[..., 1] - 0.5 * (kept[..., 0] + kept[..., 2])) < np.mean(
        image[..., 1] - 0.5 * (image[..., 0] + image[..., 2]))
