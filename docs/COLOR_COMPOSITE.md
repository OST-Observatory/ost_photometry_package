# Colour images from the stacks

`ost_photometry.imaging` turns the per-filter stacks of one target (all
filters on one pixel grid, e-/s) into a colour image. Scripts:
`auxiliary_scripts/astro_imaging/3_color_composite.py` (stacks of
`1_reduce_and_stack.py` / `2_restack.py`) and
`auxiliary_scripts/archive_pipeline/5_color_composite.py` (every target of
`stacks/summary.ecsv` or `restack/summary.ecsv`).

```python
from ost_photometry.imaging import ColorSettings, compose, stacks_from_directory, write_images

result = compose(stacks_from_directory("output/"),
                 ColorSettings(palette="lrgb", remove_gradient=True,
                               star_color_calibration=True))
write_images(result, "output/color/M104_lrgb")   # .tif (16 bit), .png, .jpg
```

## Palettes

Filter names are normalised (`Halpha` / `H alpha` → `Ha`, `O III` → `OIII`,
`S II` → `SII`, `Clear` / `C` → `L`).

| Palette | R | G | B | Luminance | Stretch |
|---------|---|---|---|-----------|---------|
| `rgb` | R | G or V | B | – | linked |
| `lrgb` | R | G or V | B | L (Clear) | linked |
| `sho` | SII | Ha | OIII | – | per channel |
| `hoo` | Ha | OIII | OIII | – | per channel |
| `custom` | `custom_mix` | | | `luminance` | linked unless all narrowband |
| `auto` | LRGB → RGB → SHO → HOO, the first the filters allow | | | | |

`custom_mix` example (Hα into red): `{"R": {"Ha": 0.7, "R": 0.3}, "G": {"V": 1}, "B": {"B": 1}}`.

## Processing order (`ColorSettings`)

1. Masked pixels get the median of their neighbours; the image is cut to the
   field all filters cover (`crop_to_common_field`); cold / dead pixels are
   replaced (`cosmetic_correction`).
2. Optional gradient removal per filter (`remove_gradient`): `polynomial`
   (default, degree `gradient_degree` = 2, fitted to the sigma-clipped sky of
   a 16-tile grid with stars and the halos of extended objects masked) or
   `background2d` (`gradient_box`; follows smaller structures, but large
   objects end up partly in the model).
3. Mixing into R / G / B.
4. Sky neutralisation (default): the sky of every channel is set to zero.
5. White balance: `star_color_calibration` (median star neutral, isolated
   unsaturated stars, saturation from `SATLEVEL / EXPTIME`; broadband only) or
   fixed `channel_factors`.
6. Stretch: `auto` (midtone transfer: black point at sky − `shadows_sigma` σ,
   sky at `target_background`), `asinh` or `linear`; linked or per channel.
7. LRGB: the stretched luminance replaces the CIELAB lightness;
   `saturation` scales the chroma.

The result carries a log (palette, mix, sky levels, factors, stars used,
stretch parameters, crop); `write_images` stores it as JSON in the TIFF
description. `sky_colour_spread` measures how much the sky colour varies over
the field; above 0.08 the log suggests `remove_gradient=True`.

## Limits

No deconvolution, noise or star reduction, no photometric colour calibration
against catalogues, no rotation to north up. Colour gradients from flats that
do not match between filters (dust donuts, vignetting) are only partly
removed: the polynomial follows smooth shapes, not rings.
