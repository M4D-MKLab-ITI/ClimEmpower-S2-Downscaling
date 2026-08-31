"""Low-level array utilities shared by the classical and PNN pansharpening methods."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# 1D interpolation filter coefficients (CDF 9/7-style biorthogonal wavelet kernel),
# built once at import time and reused by upsample_interp23.
_CDF23_HALF = 2 * np.array(
    [
        0.5,
        0.305334091185,
        0,
        -0.072698593239,
        0,
        0.021809577942,
        0,
        -0.005192756653,
        0,
        0.000807762146,
        0,
        -0.000060081482,
    ]
)


def _base_coeff() -> np.ndarray:
    """Build the symmetric 23-tap interpolation kernel used by ``upsample_interp23``."""
    reversed_half = _CDF23_HALF[::-1]
    return np.insert(_CDF23_HALF, 0, reversed_half[:-1])


BaseCoeff = _base_coeff()


def upsample_interp23(image: np.ndarray, ratio: int) -> np.ndarray:
    """Upsample a multi-band image by a power-of-two ratio via interpolation.

    Performs 2D interpolation on a multi-band image to increase its spatial
    resolution by ``ratio``, using the fixed ``BaseCoeff`` filter kernel. The
    image is progressively doubled in size (zero-inserted then filtered) until
    the target ratio is reached, so ``ratio`` must be a power of two.

    Parameters
    ----------
    image : np.ndarray
        Array of shape (H, W, bands).
    ratio : int
        Upsampling factor. Must be a power of two; 1 returns ``image`` unchanged.

    Returns
    -------
    np.ndarray
        Upsampled array of shape (H * ratio, W * ratio, bands).
    """
    if ratio == 1:
        return image

    image = np.transpose(image, (2, 0, 1))
    bands, rows, cols = image.shape

    upsampled = image
    first = True
    for level in range(1, int(np.log2(ratio)) + 1):
        expanded = np.zeros((bands, 2**level * rows, 2**level * cols))
        if first:
            expanded[:, 1 : expanded.shape[1] : 2, 1 : expanded.shape[2] : 2] = upsampled
            first = False
        else:
            expanded[:, 0 : expanded.shape[1] : 2, 0 : expanded.shape[2] : 2] = upsampled

        for band_idx in range(bands):
            band = expanded[band_idx, :, :]
            for row in range(band.shape[0]):
                band[row, :] = ndimage.correlate(band[row, :], BaseCoeff, mode="wrap")
            for col in range(band.shape[1]):
                band[:, col] = ndimage.correlate(band[:, col], BaseCoeff, mode="wrap")
            expanded[band_idx, :, :] = band
        upsampled = expanded

    return np.transpose(upsampled, (1, 2, 0))
