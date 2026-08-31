# ClimEmpower — Sentinel-2 Pansharpening / Downscaling

A small, installable Python package, `climempower_downscaling`, plus an example notebook,
for enhancing the spatial resolution of Sentinel-2's 20m spectral bands to 10m using
pansharpening. Developed for the [ClimEmpower](https://cordis.europa.eu/project/id/101112728)
Horizon Europe project — see [Citation](#citation) below.

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

Four pansharpening methods are implemented in `climempower_downscaling`, adapted from the
MIT-licensed [codegaj/py_pansharpening](https://github.com/codegaj/py_pansharpening) toolbox:

- **Brovey Transform** — injects the panchromatic band's brightness into each multispectral
  band via a ratio-based algebraic combination.
- **IHS Transform** — replaces the intensity component of an Intensity-Hue-Saturation color
  transform with the panchromatic band.
- **Wavelet Transform** — substitutes the panchromatic band's approximation coefficients
  (2-level Haar decomposition) into each multispectral band before reconstruction. In
  practice this method is more prone to artifacts (banding) and produces comparatively
  less sharp results than the other three — worth keeping in mind when choosing a method.
- **PNN (Pansharpening Neural Network)** — a compact 3-layer CNN (Masi et al., 2016), whose
  implementation in `py_pansharpening` itself credits
  [sergiovitale/pansharpening-cnn-python-version](https://github.com/sergiovitale/pansharpening-cnn-python-version),
  concatenating the six upsampled 20m bands with the pan band as a 7-channel input
  (64 filters, 9×9 → 32 filters, 5×5 → 6 filters, 5×5, ReLU activations, Adam optimizer,
  learning rate 5e-4, MSE loss, PSNR metric). It's trained **zero-shot per scene**: the
  real pan/multispectral pair is degraded to synthesize a lower-resolution training pair
  (32×32 patches, stride 8), the network trains on that synthetic pair, then runs on the
  real data via overlap-tiled inference (400×400 tiles, 320×320 kept per tile). Brovey,
  IHS, and PNN generally produce the clearest, most reliable results of the four.

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

See `examples/demo_pansharpening.ipynb` for a full walkthrough — loading a Sentinel-2 SAFE
product, running all four methods, visual comparison, and saving georeferenced GeoTIFF
output.

## Citation

If you use this repository, please cite it:

> M4D-MKLab-ITI. *ClimEmpower-S2-Downscaling.*
> https://github.com/M4D-MKLab-ITI/ClimEmpower-S2-Downscaling

## Credits

The pansharpening implementations are adapted from
[codegaj/py_pansharpening](https://github.com/codegaj/py_pansharpening) (MIT license).
PNN is additionally based on the paper:

> Masi G, Cozzolino D, Verdoliva L, Scarpa G. "Pansharpening by convolutional neural
> networks." Remote Sensing, 2016, 8(7): 594.

and on [sergiovitale/pansharpening-cnn-python-version](https://github.com/sergiovitale/pansharpening-cnn-python-version),
which `py_pansharpening`'s own PNN implementation credits as its source.

## License

MIT — see [LICENSE](LICENSE).
