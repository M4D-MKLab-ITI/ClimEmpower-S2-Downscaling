# ClimEmpower — Sentinel-2 Pansharpening / Downscaling

This repository implements the spatial downscaling methodology developed for the
[ClimEmpower](https://cordis.europa.eu/project/id/101112728) project
(Horizon Europe, Grant No. 101112728), Work Package 2, Deliverable D2.3 —
*"Methodology of spatial data downscaling, social sensing and fusion of different types"*,
Chapter 4: *Development of spatial downscaling Copernicus Satellite images*.

It provides a small, installable Python package, `climempower_downscaling`, plus an
example notebook, for enhancing the spatial resolution of Sentinel-2's 20m spectral bands
to 10m using pansharpening.

## Background

Sentinel-2's Multispectral Instrument (MSI) delivers 13 bands at three resolutions:

| Resolution | Bands | Description |
|---|---|---|
| 10m | B02, B03, B04, B08 | Blue, Green, Red, NIR |
| 20m | B05, B06, B07, B8A, B11, B12 | Red Edge 1–3, Narrow NIR, SWIR 1–2 |
| 60m | B01, B09, B10 | Coastal aerosol, Water vapour, Cirrus |

Unlike commercial sensors such as QuickBird, WorldView, or GeoEye-1, Sentinel-2 has no
dedicated high-resolution panchromatic (PAN) band, which is normally what pansharpening
fuses with the lower-resolution bands. This repo builds a **synthetic panchromatic band**
by averaging the four 10m bands (blue, green, red, NIR), then uses it to sharpen the six
20m bands up to the 10m grid.

## Methods implemented

Four pansharpening methods are implemented in `climempower_downscaling`, matching the
methodology described in D2.3 §4.3.2:

- **Brovey Transform** — injects the panchromatic band's brightness into each multispectral
  band via a ratio-based algebraic combination.
- **IHS Transform** — replaces the intensity component of an Intensity-Hue-Saturation color
  transform with the panchromatic band.
- **Wavelet Transform** — substitutes the panchromatic band's approximation coefficients
  (2-level Haar decomposition) into each multispectral band before reconstruction. Per the
  deliverable's findings (§4.4), this method is more prone to artifacts (banding) and
  produced comparatively less sharp results than the other three methods — worth keeping
  in mind when choosing a method.
- **PNN (Pansharpening Neural Network)** — a compact 3-layer CNN (Masi et al., 2016),
  concatenating the six upsampled 20m bands with the pan band as a 7-channel input
  (64 filters, 9×9 → 32 filters, 5×5 → 6 filters, 5×5, ReLU activations, Adam optimizer,
  learning rate 5e-4, MSE loss, PSNR metric). It's trained **zero-shot per scene**: the
  real pan/multispectral pair is degraded to synthesize a lower-resolution training pair
  (32×32 patches, stride 8), the network trains on that synthetic pair, then runs on the
  real data via overlap-tiled inference (400×400 tiles, 320×320 kept per tile). Per the
  deliverable, Brovey, IHS, and PNN produced the clearest, most reliable results.

All three classical methods (Brovey, IHS, Wavelet) require a common sharpening ratio
between the pan and multispectral inputs; PNN additionally requires `tensorflow`.

## Package structure

```
climempower_downscaling/
    preprocessing.py   # SAFE extraction, band discovery, AOI subsetting, panchromatic
                        # band construction, normalization
    pansharpening.py   # brovey, ihs, wavelet
    pnn.py              # build_pnn_model, train_pnn, reconstruct_pnn, run_pnn
    io_utils.py          # GeoTIFF read/write (with band descriptions), georeference helpers
    utils.py              # upsample_interp23, BaseCoeff
    visualization.py       # plot_band_comparison, make_rgb_composite, plot_band_histogram
examples/
    demo_pansharpening.ipynb   # end-to-end walkthrough
tests/                          # pytest smoke tests
Weights/PNN_model.h5             # PNN weights pretrained on a Sentinel-2 scene (6-band setup)
```

## Installation

```bash
pip install -e .
```

or, for the classical methods only (no `tensorflow` dependency):

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from climempower_downscaling import (
    discover_band_paths, open_bands, make_window, read_band_subset,
    generate_panchromatic_band, normalize_to_unit_range, stack_bands,
    brovey, ihs, wavelet,
)
from climempower_downscaling.pnn import run_pnn

safe_dir = "path/to/S2..._MSIL2A_....SAFE"
paths_10m = discover_band_paths(safe_dir, "10m")
paths_20m = discover_band_paths(safe_dir, "20m")

datasets_10m = open_bands({b: paths_10m[b] for b in ("B02", "B03", "B04", "B08")})
datasets_20m = open_bands({b: paths_20m[b] for b in ("B05", "B06", "B07", "B8A", "B11", "B12")})

window_10m, window_20m = make_window((0, 0, 2000, 2000)), make_window((0, 0, 1000, 1000))
subset_10m = read_band_subset(datasets_10m, window_10m)
subset_20m = read_band_subset(datasets_20m, window_20m)

pan, _, _ = normalize_to_unit_range(
    generate_panchromatic_band(subset_10m["B04"], subset_10m["B03"], subset_10m["B02"], subset_10m["B08"])[..., None]
)
hs, _, _ = normalize_to_unit_range(stack_bands(subset_20m))

sharpened = brovey(pan, hs)  # or ihs(pan, hs), wavelet(pan, hs), run_pnn(pan, hs, ...)
```

See `examples/demo_pansharpening.ipynb` for a complete walkthrough — loading data, running
all four methods, visual comparison, and saving georeferenced GeoTIFF output.

## Citation

If you use this code, please cite the deliverable it implements:

> Chatzichristaki Ch., Karystinakis K., Moumtzidou A., Murano I. (2024). *Methodology of
> spatial data downscaling, social sensing and fusion of different data types.*
> Deliverable D2.3 of the Horizon Europe project ClimEmpower.

## License

MIT — see [LICENSE](LICENSE).
